// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
#include "utils.h"

#include <cstdint>
#include <AudioFile.h>

std::vector<float> upsample(const std::vector<float>& input, int targetSampleRate, int originalSampleRate) {
    std::vector<float> output;
    float ratio = static_cast<float>(targetSampleRate) / originalSampleRate;

    for (size_t i = 0; i < input.size(); ++i) {
        output.push_back(input[i]);
        if (i < input.size() - 1) {
            float nextSample = input[i + 1];
            for (float t = 1.0f; t < ratio; t += 1.0f) {
                float interpolatedSample = input[i] + (nextSample - input[i]) * (t / ratio);
                output.push_back(interpolatedSample);
            }
        }
    }
    return output;
}

std::vector<float> downsample(const std::vector<float>& input, int targetSampleRate, int originalSampleRate) { //decimate
    std::vector<float> output;
    float ratio = static_cast<float>(originalSampleRate) / targetSampleRate;

    for (size_t i = 0; i < input.size(); i += static_cast<size_t>(ratio)) {
        output.push_back(input[i]);
    }

    return output;
}

std::vector<float> readAudio(const std::string& filename) {
    AudioFile<float> audio(filename);
    if(audio.getNumChannels() == 0 || audio.getLengthInSeconds() == 0) return {};
    const auto sr = audio.getSampleRate();
    // FIXME: Hard-coded number of samples, we should use audio_params.samplerate from the network info.
    if(sr == 16000) return audio.samples[0];
    const auto original = audio.samples[0];

    if(sr < 16000) {
        std::cerr << "Unsupported sample rate " << sr << std::endl;
        return {};
    }

        //really bad resampling, let's use matx poly_resample, which is the sampe implementation of scipy poly resample
    const int multiple = sr/16000;
    if(multiple * 16000 == sr) // multiple of 16khz khz
    {
        return downsample(original, 16000, sr);
    }
    if(audio.getSampleRate() == 24000) // 44.1 khz
    {
        const int lcm  = 48000;
        return downsample(upsample(original,  lcm, sr), 16000, lcm);
    }
    if(sr == 44100 || sr == 88200) // 44.1 khz 88.2khz
    {
        const int lcm  = 7056000;
        return downsample(upsample(original,  lcm, sr), 16000, lcm);
    }

    std::cerr << "Unsupported sample rate " << sr << std::endl;
    return {}; //not supported
}

std::vector<float> loadAudio() {
    // OPTME: allow for switching audio track
    return readAudio(TEST_DATA_DIR "sample-data/audio_4sec_16k_s16le.wav");
}

void AddDefaultEmotion(benchmark::State& state, nva2f::IGeometryExecutorBundle& bundle) {
    const auto nbTracks = bundle.GetExecutor().GetNbTracks();
    std::vector<float> emptyEmotion;
    for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
        auto& emotionAccumulator = bundle.GetEmotionAccumulator(trackIndex);
        emptyEmotion.resize(emotionAccumulator.GetEmotionSize(), 0.0f);
        CHECK_AND_SKIP(!emotionAccumulator.Accumulate(
            0, nva2x::HostTensorFloatConstView{emptyEmotion.data(), emptyEmotion.size()}, bundle.GetCudaStream().Data()
            ));
        CHECK_AND_SKIP(!emotionAccumulator.Close());
    }
}

void AddDefaultEmotion(benchmark::State& state, nva2f::IBlendshapeExecutorBundle& bundle) {
    const auto nbTracks = bundle.GetExecutor().GetNbTracks();
    std::vector<float> emptyEmotion;
    for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
        auto& emotionAccumulator = bundle.GetEmotionAccumulator(trackIndex);
        emptyEmotion.resize(emotionAccumulator.GetEmotionSize(), 0.0f);
        CHECK_AND_SKIP(!emotionAccumulator.Accumulate(
            0, nva2x::HostTensorFloatConstView{emptyEmotion.data(), emptyEmotion.size()}, bundle.GetCudaStream().Data()
            ));
        CHECK_AND_SKIP(!emotionAccumulator.Close());
    }
}

TimePoint startTimer() {
    return Clock::now();
}

double getElapsedMilliseconds(const TimePoint& startTime) {
    return std::chrono::duration<double, std::milli>(Clock::now() - startTime).count();
}

// GeometryExecutorResultsCollector implementations
void GeometryExecutorResultsCollector::Init(nva2f::IGeometryExecutorBundle* bundle, benchmark::State& state) {
    _bundle = bundle;
    CHECK_AND_SKIP(!_bundle->GetExecutor().SetResultsCallback(callbackForGeometryExecutor, &_callbackData));
    ResetCounters();
}

bool GeometryExecutorResultsCollector::callbackForGeometryExecutor(void* userdata, const nva2f::IGeometryExecutor::Results& results) {
    auto& data = *static_cast<GeometryExecutorCallbackData*>(userdata);
    data.frameIndices[results.trackIndex] += 1;
    return true;
}

void GeometryExecutorResultsCollector::ResetCounters() {
    _callbackData.frameIndices.clear();
    _callbackData.frameIndices.resize(_bundle->GetExecutor().GetNbTracks(), 0);
}

std::size_t GeometryExecutorResultsCollector::GetTotalFrames() const {
    return std::accumulate(_callbackData.frameIndices.begin(), _callbackData.frameIndices.end(), 0);
}

bool GeometryExecutorResultsCollector::HasFrameGenerated(std::size_t trackIndex) const {
    return _callbackData.frameIndices[trackIndex] > 0;
}

bool GeometryExecutorResultsCollector::Wait() {
    if (_bundle->GetCudaStream().Synchronize()) {
        return false;
    }
    return true;
}

// BlendshapeSolveExecutorResultsCollector implementations
void BlendshapeSolveExecutorResultsCollector::Init(nva2f::IBlendshapeExecutorBundle* bundle, benchmark::State& state) {
    _bundle = bundle;
    _callbackData.state = &state;
    _callbackData.weightViews.resize(_bundle->GetExecutor().GetNbTracks(), {});
    auto& executor = _bundle->GetExecutor();
    if (executor.GetResultType() == nva2f::IBlendshapeExecutor::ResultsType::HOST) {
        auto callback = [](void* userdata, const nva2f::IBlendshapeExecutor::HostResults& results, std::error_code errorCode) -> void {
            callbackForHostBlendshapeSolveExecutor(userdata, results, errorCode);
        };
        CHECK_AND_SKIP(!executor.SetResultsCallback(callback, &_callbackData));
    } else if (executor.GetResultType() == nva2f::IBlendshapeExecutor::ResultsType::DEVICE) {
        _weightHostPinnedBatch.clear();
        for (std::size_t trackIndex = 0; trackIndex < executor.GetNbTracks(); ++trackIndex) {
            _weightHostPinnedBatch.emplace_back(nva2x::CreateHostPinnedTensorFloat(executor.GetWeightCount()));
            _callbackData.weightViews[trackIndex] = *(_weightHostPinnedBatch.back());
        }
        CHECK_AND_SKIP(!executor.SetResultsCallback(callbackForDeviceBlendshapeSolveExecutor, &_callbackData));
    } else {
        state.SkipWithError("Unknown results type.");
    }
    ResetCounters();
}

void BlendshapeSolveExecutorResultsCollector::callbackForHostBlendshapeSolveExecutor(void* userdata, const nva2f::IBlendshapeExecutor::HostResults& results, std::error_code errorCode) {
    auto& data = *static_cast<BlendshapeSolveExecutorCallbackData*>(userdata);
    data.frameIndices[results.trackIndex] += 1;
}

bool BlendshapeSolveExecutorResultsCollector::callbackForDeviceBlendshapeSolveExecutor(void* userdata, const nva2f::IBlendshapeExecutor::DeviceResults& results) {
    auto& data = *static_cast<BlendshapeSolveExecutorCallbackData*>(userdata);
    auto& state = *data.state;
    // copy to pinned host buffer for a fair comparison
    if (data.weightViews[results.trackIndex].Size() > 0 && results.weights.Size() > 0) {
        CHECK_AND_SKIP(!nva2x::CopyDeviceToHost(data.weightViews[results.trackIndex], results.weights, results.cudaStream));
        data.frameIndices[results.trackIndex] += 1;
    }
    return true;
}

void BlendshapeSolveExecutorResultsCollector::ResetCounters() {
    _callbackData.frameIndices.clear();
    _callbackData.frameIndices.resize(_bundle->GetExecutor().GetNbTracks(), 0);
}

std::size_t BlendshapeSolveExecutorResultsCollector::GetTotalFrames() const {
    return std::accumulate(_callbackData.frameIndices.begin(), _callbackData.frameIndices.end(), 0);
}

bool BlendshapeSolveExecutorResultsCollector::HasFrameGenerated(std::size_t trackIndex) const {
    return _callbackData.frameIndices[trackIndex] > 0;
}

bool BlendshapeSolveExecutorResultsCollector::Wait() {
    std::size_t nbTracks =  _bundle->GetExecutor().GetNbTracks();
    for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
        if (_bundle->GetExecutor().Wait(trackIndex)) {
            return false;
        }
    }
    return true;
}

// RunExecutorOffline implementation
template<typename A2FExecutorBundleType>
void RunExecutorOffline(
    benchmark::State& state,
    UniquePtr<A2FExecutorBundleType>& a2fExecutorBundle
) {
    const auto nbTracks = a2fExecutorBundle->GetExecutor().GetNbTracks();

    using A2FResultsCollectorType = std::conditional_t<
        std::is_same_v<A2FExecutorBundleType, nva2f::IGeometryExecutorBundle>,
        GeometryExecutorResultsCollector,
        BlendshapeSolveExecutorResultsCollector
    >;
    A2FResultsCollectorType a2fExecutorResultsCollector;
    a2fExecutorResultsCollector.Init(a2fExecutorBundle.get(), state);

    // Then, load all the audio and accumulate it.
    const auto audioBuffer = loadAudio();
    CHECK_AND_SKIP(!audioBuffer.empty());
    for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
        // We put same amount of audio in each track to test the executor scalability
        CHECK_AND_SKIP(
            !a2fExecutorBundle->GetAudioAccumulator(trackIndex).Accumulate(
            nva2x::HostTensorFloatConstView{audioBuffer.data(), audioBuffer.size()}, a2fExecutorBundle->GetCudaStream().Data()
            )
            );
        CHECK_AND_SKIP(!a2fExecutorBundle->GetAudioAccumulator(trackIndex).Close());
    }

    AddDefaultEmotion(state, *a2fExecutorBundle);

    // warm-up
    // Run until at least one frame is available, because execution for diffusion
    // can return 0 frames for the first execution in the padding before the audio.
    while (!a2fExecutorResultsCollector.HasFrameGenerated(0)) {
        CHECK_AND_SKIP(nva2x::GetNbReadyTracks(a2fExecutorBundle->GetExecutor()) > 0);
        CHECK_AND_SKIP(!a2fExecutorBundle->GetExecutor().Execute(nullptr));
        CHECK_AND_SKIP(!a2fExecutorBundle->GetCudaStream().Synchronize());
    }
    a2fExecutorResultsCollector.ResetCounters();

    for (auto _ : state) {
        state.PauseTiming();
        CHECK_AND_SKIP(!a2fExecutorBundle->GetCudaStream().Synchronize());
        for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
            CHECK_AND_SKIP(!a2fExecutorBundle->GetExecutor().Reset(trackIndex));
            CHECK_AND_SKIP(!a2fExecutorBundle->GetEmotionAccumulator(trackIndex).Reset());
        }
        AddDefaultEmotion(state, *a2fExecutorBundle);
        CHECK_AND_SKIP(!a2fExecutorBundle->GetCudaStream().Synchronize());
        state.ResumeTiming();
        // Process all geometry
        auto startTimeA2F = startTimer();
        while (nva2x::GetNbReadyTracks(a2fExecutorBundle->GetExecutor()) > 0) {
            CHECK_AND_SKIP(!a2fExecutorBundle->GetExecutor().Execute(nullptr));
        }
        state.counters["A2FExecuteTime(ms)"] = getElapsedMilliseconds(startTimeA2F);
        CHECK_AND_SKIP(a2fExecutorResultsCollector.Wait());
        state.counters["A2FTotalTime(ms)"] = getElapsedMilliseconds(startTimeA2F);
    }

    std::size_t totalFrames = a2fExecutorResultsCollector.GetTotalFrames();
    state.SetItemsProcessed(totalFrames);
    state.counters["A2FAvgMultiTrackProcessingTime(ms)"] = state.counters["A2FTotalTime(ms)"] / totalFrames * nbTracks;
    state.counters["A2FAvgPerTrackProcessingTime(ms)"] = state.counters["A2FTotalTime(ms)"] / totalFrames;
    state.counters["TotalTime(ms)"] = state.counters["A2FTotalTime(ms)"];
    state.counters["nbTracks"] = static_cast<double>(nbTracks); // state.counters only accepts double
}

// RunExecutorStreaming implementation
template<typename A2FExecutorBundleType>
void RunExecutorStreaming(
    benchmark::State& state,
    std::size_t audioChunkSize,
    UniquePtr<A2FExecutorBundleType>& bundle
) {
    assert(audioChunkSize > 0);
    auto& executor = bundle->GetExecutor();
    const auto nbTracks = executor.GetNbTracks();

    using A2FResultsCollectorType = std::conditional_t<
        std::is_same_v<A2FExecutorBundleType, nva2f::IGeometryExecutorBundle>,
        GeometryExecutorResultsCollector,
        BlendshapeSolveExecutorResultsCollector
    >;
    A2FResultsCollectorType a2fExecutorResultsCollector;
    a2fExecutorResultsCollector.Init(bundle.get(), state);

    // Load all the audio, but don't accumulate it yet.
    const auto audioBuffer = loadAudio();
    CHECK_AND_SKIP(!audioBuffer.empty());

    auto processAvailableData = [&]() {
        while (nva2x::GetNbReadyTracks(bundle->GetExecutor()) > 0) {
            CHECK_AND_SKIP(!bundle->GetExecutor().Execute(nullptr));
        }
    };

    AddDefaultEmotion(state, *bundle);

    // warm-up
    // Run until at least one frame is available, because execution for diffusion
    // can return 0 frames for the first execution in the padding before the audio.
    for (std::size_t i = 0; i < audioBuffer.size() && (!a2fExecutorResultsCollector.HasFrameGenerated(0)); i += audioChunkSize) {
        const auto chunkData = audioBuffer.data() + i;
        const auto chunkSize = std::min(audioChunkSize, audioBuffer.size() - i);
        for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
            CHECK_AND_SKIP(!bundle->GetAudioAccumulator(trackIndex).Accumulate(
                nva2x::HostTensorFloatConstView{chunkData, chunkSize}, bundle->GetCudaStream().Data()
                )
            );
        }
        // Process available data.
        processAvailableData();
    }
    a2fExecutorResultsCollector.ResetCounters();

    for (auto _ : state) {
        state.PauseTiming();
        for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
            CHECK_AND_SKIP(!executor.Reset(trackIndex));
            CHECK_AND_SKIP(!bundle->GetEmotionAccumulator(trackIndex).Reset());
            CHECK_AND_SKIP(!bundle->GetAudioAccumulator(trackIndex).Reset());
        }
        AddDefaultEmotion(state, *bundle);
        if constexpr (std::is_same_v<A2FExecutorBundleType, nva2f::IBlendshapeExecutorBundle>) {
            for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
                CHECK_AND_SKIP(!executor.Wait(trackIndex));
            }
        } else if constexpr (std::is_same_v<A2FExecutorBundleType, nva2f::IGeometryExecutorBundle>) {
            CHECK_AND_SKIP(!bundle->GetCudaStream().Synchronize());
        }
        state.ResumeTiming();
        auto startTime = startTimer();
        for (std::size_t i = 0; i < audioBuffer.size(); i += audioChunkSize) {
            const auto chunkData = audioBuffer.data() + i;
            const auto chunkSize = std::min(audioChunkSize, audioBuffer.size() - i);
            for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
                CHECK_AND_SKIP(!bundle->GetAudioAccumulator(trackIndex).Accumulate(
                    nva2x::HostTensorFloatConstView{chunkData, chunkSize}, bundle->GetCudaStream().Data()
                    )
                );
            }
            // Process available data.
            processAvailableData();
        }
        for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
            CHECK_AND_SKIP(!bundle->GetAudioAccumulator(trackIndex).Close());
        }
        // After closing the audio, we might be able to do more processing.
        processAvailableData();
        if constexpr (std::is_same_v<A2FExecutorBundleType, nva2f::IBlendshapeExecutorBundle>) {
            for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
                CHECK_AND_SKIP(!executor.Wait(trackIndex));
            }
        } else if constexpr (std::is_same_v<A2FExecutorBundleType, nva2f::IGeometryExecutorBundle>) {
            CHECK_AND_SKIP(!bundle->GetCudaStream().Synchronize());
        }
        state.counters["TotalTime(ms)"] = getElapsedMilliseconds(startTime);
    }

    std::size_t totalFrames = a2fExecutorResultsCollector.GetTotalFrames();
    state.SetItemsProcessed(totalFrames);
    state.counters["AvgMultiTrackProcessingTime(ms)"] = state.counters["TotalTime(ms)"] / totalFrames * nbTracks;
    state.counters["AvgPerTrackProcessingTime(ms)"] = state.counters["TotalTime(ms)"] / totalFrames;
    state.counters["nbTracks"] = static_cast<double>(nbTracks); // state.counters only accepts double
}

// Explicit template instantiations
template void RunExecutorOffline<nva2f::IGeometryExecutorBundle>(
    benchmark::State& state,
    UniquePtr<nva2f::IGeometryExecutorBundle>& a2fExecutorBundle
);

template void RunExecutorOffline<nva2f::IBlendshapeExecutorBundle>(
    benchmark::State& state,
    UniquePtr<nva2f::IBlendshapeExecutorBundle>& a2fExecutorBundle
);

template void RunExecutorStreaming<nva2f::IGeometryExecutorBundle>(
    benchmark::State& state,
    std::size_t audioChunkSize,
    UniquePtr<nva2f::IGeometryExecutorBundle>& bundle
);

template void RunExecutorStreaming<nva2f::IBlendshapeExecutorBundle>(
    benchmark::State& state,
    std::size_t audioChunkSize,
    UniquePtr<nva2f::IBlendshapeExecutorBundle>& bundle
);
