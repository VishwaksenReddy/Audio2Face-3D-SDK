#include "inference_sessions.h"

#include "audio2x/cuda_utils.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <utility>

namespace a2fserver {
namespace {

std::string RandomHex(std::size_t bytes) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);
    std::string out;
    out.reserve(bytes * 2);
    for (std::size_t i = 0; i < bytes; ++i) {
        const auto v = static_cast<std::uint8_t>(dist(gen));
        out.push_back(kHex[v >> 4]);
        out.push_back(kHex[v & 0x0f]);
    }
    return out;
}

std::string ExecutionOptionToString(nva2f::IGeometryExecutor::ExecutionOption opt) {
    using Opt = nva2f::IGeometryExecutor::ExecutionOption;
    switch (opt) {
        case Opt::None:
            return "None";
        case Opt::Skin:
            return "Skin";
        case Opt::Tongue:
            return "Tongue";
        case Opt::SkinTongue:
            return "SkinTongue";
        default:
            return "Unknown";
    }
}

void AppendU32LE(std::vector<std::uint8_t>& buf, std::uint32_t v) {
    buf.push_back(static_cast<std::uint8_t>((v)&0xff));
    buf.push_back(static_cast<std::uint8_t>((v >> 8) & 0xff));
    buf.push_back(static_cast<std::uint8_t>((v >> 16) & 0xff));
    buf.push_back(static_cast<std::uint8_t>((v >> 24) & 0xff));
}

void AppendU64LE(std::vector<std::uint8_t>& buf, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        buf.push_back(static_cast<std::uint8_t>((v >> (i * 8)) & 0xff));
    }
}

void AppendI64LE(std::vector<std::uint8_t>& buf, std::int64_t v) {
    AppendU64LE(buf, static_cast<std::uint64_t>(v));
}

} // namespace

bool SessionContext::Init(const ServerConfig& config) {
    if (!config.useGpuSolver) {
        std::cerr << "Only GPU blendshape solver is supported by this server build.\n";
        return false;
    }

    _cudaDeviceForThread = config.cudaDevice;
    _modelJsonPath = config.modelJsonPath;
    _executionOptionStr = ExecutionOptionToString(config.executionOption);
    _useGpuSolver = config.useGpuSolver;

    if (auto err = nva2x::SetCudaDeviceIfNeeded(config.cudaDevice)) {
        std::cerr << "Failed to set CUDA device: " << err.message() << "\n";
        return false;
    }

    if (!config.diffusion) {
        _bundle.reset(nva2f::ReadRegressionBlendshapeSolveExecutorBundle(
            1,
            config.modelJsonPath.c_str(),
            config.executionOption,
            config.useGpuSolver,
            config.fpsNumerator,
            config.fpsDenominator,
            nullptr,
            nullptr
        ));
    } else {
        _bundle.reset(nva2f::ReadDiffusionBlendshapeSolveExecutorBundle(
            1,
            config.modelJsonPath.c_str(),
            config.executionOption,
            config.useGpuSolver,
            config.diffusionIdentity,
            config.diffusionConstantNoise,
            nullptr,
            nullptr
        ));
    }

    if (!_bundle) {
        std::cerr << "Failed to create executor bundle from model: " << config.modelJsonPath << "\n";
        return false;
    }

    auto& executor = _bundle->GetExecutor();
    if (executor.GetResultType() != nva2f::IBlendshapeExecutor::ResultsType::DEVICE) {
        std::cerr << "Expected DEVICE results type from GPU solver.\n";
        return false;
    }

    if (auto err = executor.SetResultsCallback(&SessionContext::OnDeviceResults, this)) {
        std::cerr << "Failed to set results callback: " << err.message() << "\n";
        return false;
    }

    _samplingRate = executor.GetSamplingRate();
    executor.GetFrameRate(_fpsNumerator, _fpsDenominator);
    _weightCount = executor.GetWeightCount();

    if (!BuildChannelList()) {
        return false;
    }

    const std::size_t stagingSize = _weightCount * kMaxStagedFrames;
    _weightsStaging.reset(nva2x::CreateHostPinnedTensorFloat(stagingSize));
    if (!_weightsStaging) {
        std::cerr << "Failed to allocate pinned host staging buffer.\n";
        return false;
    }

    return ResetForReuse();
}

bool SessionContext::Start(a2fws::Socket* wsSocket) {
    std::lock_guard<std::mutex> lock(_mutex);
    _wsSocket = wsSocket;
    _sessionId = RandomHex(16);
    _pendingFrames.clear();
    _nextFrameIndex = 0;
    return true;
}

void SessionContext::Stop() {
    std::lock_guard<std::mutex> lock(_mutex);
    _wsSocket = nullptr;
}

bool SessionContext::ResetForReuse() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_bundle) {
        return false;
    }
    auto& executor = _bundle->GetExecutor();
    (void)executor.Wait(0);

    if (auto err = executor.Reset(0)) {
        std::cerr << "Executor reset failed: " << err.message() << "\n";
        return false;
    }
    if (auto err = _bundle->GetAudioAccumulator(0).Reset()) {
        std::cerr << "Audio accumulator reset failed: " << err.message() << "\n";
        return false;
    }
    if (auto err = _bundle->GetEmotionAccumulator(0).Reset()) {
        std::cerr << "Emotion accumulator reset failed: " << err.message() << "\n";
        return false;
    }
    if (!InitNeutralEmotionLocked()) {
        return false;
    }

    _pendingFrames.clear();
    _nextFrameIndex = 0;
    _lastCudaStream = nullptr;
    return true;
}

std::string SessionContext::SessionId() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _sessionId;
}

nlohmann::json SessionContext::DescribeSessionStarted() const {
    std::lock_guard<std::mutex> lock(_mutex);
    nlohmann::json msg;
    msg["type"] = "SessionStarted";
    msg["protocol"] = {{"version", kProtocolVersion}};
    msg["session_id"] = _sessionId;
    msg["model"] = _modelJsonPath;
    msg["options"] = {
        {"use_gpu_solver", _useGpuSolver},
        {"execution_option", _executionOptionStr},
    };
    msg["sampling_rate"] = _samplingRate;
    msg["frame_rate"] = {{"numerator", _fpsNumerator}, {"denominator", _fpsDenominator}};
    msg["weight_count"] = _weightCount;
    msg["channels"] = _channels;
    msg["channel_groups"] = {
        {{"name", "skin"}, {"count", _skinWeightCount}},
        {{"name", "tongue"}, {"count", _tongueWeightCount}},
    };
    return msg;
}

bool SessionContext::PushAudio(std::int64_t startSampleIndex, const std::int16_t* pcm, std::size_t sampleCount) {
    if (startSampleIndex < 0) {
        std::lock_guard<std::mutex> lock(_mutex);
        SendErrorLocked("startSampleIndex must be >= 0");
        return false;
    }

    if (auto err = nva2x::SetCudaDeviceIfNeeded(_cudaDeviceForThread)) {
        std::lock_guard<std::mutex> lock(_mutex);
        SendErrorLocked(std::string("Failed to set CUDA device: ") + err.message());
        return false;
    }

    std::lock_guard<std::mutex> lock(_mutex);
    if (!_wsSocket) {
        return false;
    }
    if (!_bundle) {
        SendErrorLocked("Internal error: missing executor bundle");
        return false;
    }

    auto& audioAccumulator = _bundle->GetAudioAccumulator(0);
    auto& executor = _bundle->GetExecutor();

    const std::size_t accumulated = audioAccumulator.NbAccumulatedSamples();
    const auto startU = static_cast<std::size_t>(startSampleIndex);
    if (startU < accumulated) {
        SendErrorLocked("PushAudio startSampleIndex is behind the accumulator (out-of-order audio)");
        return false;
    }

    const std::size_t gap = startU - accumulated;
    if (gap > 16000 * 10) {
        SendErrorLocked("Audio gap too large");
        return false;
    }

    if (gap > 0) {
        _audioFloatScratch.assign(gap, 0.0f);
        if (auto err = audioAccumulator.Accumulate(
                nva2x::HostTensorFloatConstView{_audioFloatScratch.data(), _audioFloatScratch.size()},
                _bundle->GetCudaStream().Data()
            )) {
            SendErrorLocked(std::string("Failed to fill audio gap: ") + err.message());
            return false;
        }
    }

    _audioFloatScratch.resize(sampleCount);
    for (std::size_t i = 0; i < sampleCount; ++i) {
        _audioFloatScratch[i] = static_cast<float>(pcm[i]) / 32768.0f;
    }

    if (auto err = audioAccumulator.Accumulate(
            nva2x::HostTensorFloatConstView{_audioFloatScratch.data(), _audioFloatScratch.size()},
            _bundle->GetCudaStream().Data()
        )) {
        SendErrorLocked(std::string("Failed to accumulate audio: ") + err.message());
        return false;
    }

    while (nva2x::GetNbReadyTracks(executor) > 0) {
        if (auto err = executor.Execute(nullptr)) {
            SendErrorLocked(std::string("Execute() failed: ") + err.message());
            return false;
        }
        if (_pendingFrames.size() >= kFlushThresholdFrames) {
            if (!FlushPendingFramesLocked()) {
                return false;
            }
        }
    }

    if (!FlushPendingFramesLocked()) {
        return false;
    }

    // Drop processed audio/emotion to bound memory.
    const auto dropBefore = executor.GetNextAudioSampleToRead(0);
    (void)audioAccumulator.DropSamplesBefore(dropBefore);
    const auto dropEmotionBefore = executor.GetNextEmotionTimestampToRead(0);
    (void)_bundle->GetEmotionAccumulator(0).DropEmotionsBefore(dropEmotionBefore);

    return true;
}

bool SessionContext::OnDeviceResults(void* userdata, const nva2f::IBlendshapeExecutor::DeviceResults& results) {
    auto& self = *static_cast<SessionContext*>(userdata);
    if (!self._wsSocket) {
        return false;
    }
    if (results.weights.Size() == 0) {
        return true;
    }
    if (results.weights.Size() != self._weightCount) {
        self.SendErrorLocked("Unexpected weight vector size from executor");
        return false;
    }
    if (self._pendingFrames.size() >= kMaxStagedFrames) {
        self.SendErrorLocked("Too many pending frames (client too slow?)");
        return false;
    }

    const std::size_t slotIndex = self._pendingFrames.size();
    const auto dst = self._weightsStaging->View(slotIndex * self._weightCount, self._weightCount);
    if (auto err = nva2x::CopyDeviceToHost(dst, results.weights, results.cudaStream)) {
        self.SendErrorLocked(std::string("CopyDeviceToHost failed: ") + err.message());
        return false;
    }

    self._lastCudaStream = results.cudaStream;
    PendingFrame frame;
    frame.frameIndex = self._nextFrameIndex++;
    frame.tsCurrent = results.timeStampCurrentFrame;
    frame.tsNext = results.timeStampNextFrame;
    frame.slotIndex = slotIndex;
    self._pendingFrames.push_back(frame);
    return true;
}

bool SessionContext::BuildChannelList() {
    _channels.clear();
    _skinWeightCount = 0;
    _tongueWeightCount = 0;

    nva2f::IBlendshapeSolver* skinSolver = nullptr;
    nva2f::IBlendshapeSolver* tongueSolver = nullptr;
    (void)nva2f::GetExecutorSkinSolver(_bundle->GetExecutor(), 0, &skinSolver);
    (void)nva2f::GetExecutorTongueSolver(_bundle->GetExecutor(), 0, &tongueSolver);

    if (skinSolver) {
        _skinWeightCount = static_cast<std::size_t>(skinSolver->NumBlendshapePoses());
        for (std::size_t i = 0; i < _skinWeightCount; ++i) {
            _channels.emplace_back(skinSolver->GetPoseName(i));
        }
    }
    if (tongueSolver) {
        _tongueWeightCount = static_cast<std::size_t>(tongueSolver->NumBlendshapePoses());
        for (std::size_t i = 0; i < _tongueWeightCount; ++i) {
            _channels.emplace_back(tongueSolver->GetPoseName(i));
        }
    }

    if (_channels.size() != _weightCount) {
        std::cerr << "Channel count mismatch (channels=" << _channels.size() << ", weights=" << _weightCount << ")\n";
        return false;
    }
    return true;
}

bool SessionContext::InitNeutralEmotionLocked() {
    auto& emotionAccumulator = _bundle->GetEmotionAccumulator(0);
    std::vector<float> zeros(emotionAccumulator.GetEmotionSize(), 0.0f);
    if (auto err = emotionAccumulator.Accumulate(
            0,
            nva2x::HostTensorFloatConstView{zeros.data(), zeros.size()},
            _bundle->GetCudaStream().Data()
        )) {
        std::cerr << "Failed to set neutral emotion: " << err.message() << "\n";
        return false;
    }
    if (auto err = emotionAccumulator.Close()) {
        std::cerr << "Failed to close emotion accumulator: " << err.message() << "\n";
        return false;
    }
    return true;
}

void SessionContext::SendErrorLocked(const std::string& message) const {
    if (!_wsSocket) {
        return;
    }
    nlohmann::json msg;
    msg["type"] = "Error";
    msg["message"] = message;
    const auto text = msg.dump();
    (void)a2fws::SendFrame(*_wsSocket, a2fws::Opcode::Text, text.data(), text.size());
}

bool SessionContext::FlushPendingFramesLocked() {
    if (_pendingFrames.empty()) {
        return true;
    }

    if (!_lastCudaStream) {
        SendErrorLocked("Internal error: no CUDA stream associated with pending frames");
        return false;
    }

    if (auto err = _bundle->GetCudaStream().Synchronize()) {
        SendErrorLocked(std::string("CUDA stream synchronization failed: ") + err.message());
        return false;
    }

    for (const auto& frame : _pendingFrames) {
        const auto weights = _weightsStaging->View(frame.slotIndex * _weightCount, _weightCount);
        std::vector<std::uint8_t> payload;
        payload.reserve(40 + (_weightCount * sizeof(float)));
        AppendU32LE(payload, kFrameMagicA2FB);
        AppendU32LE(payload, kProtocolVersion);
        AppendU32LE(payload, static_cast<std::uint32_t>(_weightCount));
        AppendU32LE(payload, 0);
        AppendU64LE(payload, frame.frameIndex);
        AppendI64LE(payload, frame.tsCurrent);
        AppendI64LE(payload, frame.tsNext);
        const auto* floatBytes = reinterpret_cast<const std::uint8_t*>(weights.Data());
        payload.insert(payload.end(), floatBytes, floatBytes + (_weightCount * sizeof(float)));

        if (!a2fws::SendFrame(*_wsSocket, a2fws::Opcode::Binary, payload.data(), payload.size())) {
            return false;
        }
    }

    _pendingFrames.clear();
    return true;
}

bool SessionPool::Init(const ServerConfig& config) {
    _config = config;
    _sessions.clear();
    _freeIndices.clear();

    _sessions.reserve(config.maxSessions);

    for (std::size_t i = 0; i < config.maxSessions; ++i) {
        auto session = std::make_unique<SessionContext>();
        if (!session->Init(config)) {
            std::cerr << "Failed to init session " << i << "\n";
            return false;
        }
        _sessions.push_back(std::move(session));
        _freeIndices.push_back(i);
    }
    return true;
}

std::optional<std::size_t> SessionPool::Acquire(a2fws::Socket* wsSocket) {
    std::size_t idx = 0;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_freeIndices.empty()) {
            return std::nullopt;
        }
        idx = _freeIndices.back();
        _freeIndices.pop_back();
    }

    if (!_sessions[idx]->ResetForReuse()) {
        std::lock_guard<std::mutex> lock(_mutex);
        _freeIndices.push_back(idx);
        return std::nullopt;
    }
    (void)_sessions[idx]->Start(wsSocket);
    return idx;
}

void SessionPool::Release(std::size_t idx) {
    if (idx >= _sessions.size()) {
        return;
    }
    _sessions[idx]->Stop();
    std::lock_guard<std::mutex> lock(_mutex);
    _freeIndices.push_back(idx);
}

SessionContext& SessionPool::Get(std::size_t idx) { return *_sessions[idx]; }

} // namespace a2fserver
