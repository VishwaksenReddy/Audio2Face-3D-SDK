#pragma once

#include "websocket_server.h"

#include "audio2face/audio2face.h"
#include "audio2x/tensor_float.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace a2fserver {

struct ServerConfig {
    std::string host{"0.0.0.0"};
    int port{8765};
    int cudaDevice{0};
    std::size_t maxSessions{4};

    std::string modelJsonPath;
    bool diffusion{false};
    std::size_t diffusionIdentity{0};
    bool diffusionConstantNoise{true};

    bool useGpuSolver{true};
    nva2f::IGeometryExecutor::ExecutionOption executionOption{nva2f::IGeometryExecutor::ExecutionOption::SkinTongue};

    std::size_t fpsNumerator{60};
    std::size_t fpsDenominator{1};
};

class SessionContext {
public:
    SessionContext() = default;
    SessionContext(const SessionContext&) = delete;
    SessionContext& operator=(const SessionContext&) = delete;

    bool Init(const ServerConfig& config);

    bool Start(a2fws::Socket* wsSocket);
    void Stop();

    bool ResetForReuse();

    std::string SessionId() const;
    nlohmann::json DescribeSessionStarted() const;

    bool PushAudio(std::int64_t startSampleIndex, const std::int16_t* pcm, std::size_t sampleCount);

private:
    struct Destroyer {
        template <typename T> void operator()(T* obj) const {
            if (obj) {
                obj->Destroy();
            }
        }
    };

    template <typename T>
    using UniqueSdkPtr = std::unique_ptr<T, Destroyer>;

    struct PendingFrame {
        std::uint64_t frameIndex{0};
        std::int64_t tsCurrent{0};
        std::int64_t tsNext{0};
        std::size_t slotIndex{0};
    };

    static bool OnDeviceResults(void* userdata, const nva2f::IBlendshapeExecutor::DeviceResults& results);

    bool BuildChannelList();
    bool InitNeutralEmotionLocked();
    void SendErrorLocked(const std::string& message) const;
    bool FlushPendingFramesLocked();

private:
    mutable std::mutex _mutex;
    a2fws::Socket* _wsSocket{nullptr};
    int _cudaDeviceForThread{0};

    UniqueSdkPtr<nva2f::IBlendshapeExecutorBundle> _bundle;
    UniqueSdkPtr<nva2x::IHostTensorFloat> _weightsStaging;
    std::vector<float> _audioFloatScratch;

    std::string _sessionId;
    std::string _modelJsonPath;
    std::string _executionOptionStr;
    bool _useGpuSolver{true};
    std::size_t _samplingRate{0};
    std::size_t _fpsNumerator{0};
    std::size_t _fpsDenominator{0};
    std::size_t _weightCount{0};
    std::vector<std::string> _channels;
    std::size_t _skinWeightCount{0};
    std::size_t _tongueWeightCount{0};

    cudaStream_t _lastCudaStream{nullptr};
    std::uint64_t _nextFrameIndex{0};
    std::vector<PendingFrame> _pendingFrames;

    static constexpr std::uint32_t kFrameMagicA2FB = 0x42463241; // "A2FB" little-endian
    static constexpr std::uint32_t kProtocolVersion = 1;
    static constexpr std::size_t kMaxStagedFrames = 256;
    static constexpr std::size_t kFlushThresholdFrames = 32;
};

class SessionPool {
public:
    bool Init(const ServerConfig& config);
    std::optional<std::size_t> Acquire(a2fws::Socket* wsSocket);
    void Release(std::size_t idx);
    SessionContext& Get(std::size_t idx);

private:
    std::mutex _mutex;
    ServerConfig _config;
    std::vector<std::unique_ptr<SessionContext>> _sessions;
    std::vector<std::size_t> _freeIndices;
};

} // namespace a2fserver
