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
#pragma once

#include "audio2face/audio2face.h"

#include <benchmark/benchmark.h>

#include <array>
#include <chrono>
#include <memory>
#include <numeric>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

struct Destroyer {
    template <typename T> void operator()(T *obj) const {
        obj->Destroy();
    }
};

template <typename T>
using UniquePtr = std::unique_ptr<T, Destroyer>;
template <typename T>
UniquePtr<T> ToUniquePtr(T* ptr) { return UniquePtr<T>(ptr); }

#define CHECK_AND_SKIP(condition) \
    if (!(condition)) { state.SkipWithError(#condition " failed."); }

constexpr const int kDeviceID = 0;
constexpr const char* DIFFUSION_MODEL = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/multi-diffusion/model.json";
constexpr std::array<const char*, 3> REGRESSION_MODELS = {
    TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/claire/model.json",
    TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/james/model.json",
    TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/model.json",
};

constexpr const char* DIFFUSION_MODEL_FP16 = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/multi-diffusion/model_fp16.json";
constexpr std::array<const char*, 3> REGRESSION_MODELS_FP16 = {
    TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/claire/model_fp16.json",
    TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/james/model_fp16.json",
    TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/model_fp16.json",
};

std::vector<float> readAudio(const std::string& filename);
std::vector<float> loadAudio();
template<typename ExecutionOption>
std::string geometryExecutionOptionToString(ExecutionOption option) {
    static const std::unordered_map<ExecutionOption, std::string> enumMap = {
        { ExecutionOption::Skin, "Skin" },
        { ExecutionOption::Tongue, "Tongue" },
        { ExecutionOption::Jaw, "Jaw" },
        { ExecutionOption::Eyes, "Eyes" }
    };
    std::string result;
    for (const auto& [flag, name] : enumMap) {
        if ((option & flag) != ExecutionOption::None) {
            if (!result.empty()) {
                result += " | ";
            }
            result += name;
        }
    }
    return result.empty() ? "None" : result;
}

template<typename ExecutionOption>
std::string blendshapeExecutionOptionToString(ExecutionOption option) {
    static const std::unordered_map<ExecutionOption, std::string> enumMap = {
        { ExecutionOption::Skin, "Skin" },
        { ExecutionOption::Tongue, "Tongue" }
    };
    std::string result;
    for (const auto& [flag, name] : enumMap) {
        if ((option & flag) != ExecutionOption::None) {
            if (!result.empty()) {
                result += " | ";
            }
            result += name;
        }
    }
    return result.empty() ? "None" : result;
}

void AddDefaultEmotion(benchmark::State& state, nva2f::IGeometryExecutorBundle& bundle);
void AddDefaultEmotion(benchmark::State& state, nva2f::IBlendshapeExecutorBundle& bundle);

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
static_assert(Clock::is_steady, "Clock is not steady");

TimePoint startTimer();
double getElapsedMilliseconds(const TimePoint& startTime);

class GeometryExecutorResultsCollector {
public:
    void Init(nva2f::IGeometryExecutorBundle* bundle, benchmark::State& state);
    static bool callbackForGeometryExecutor(void* userdata, const nva2f::IGeometryExecutor::Results& results);
    void ResetCounters();
    std::size_t GetTotalFrames() const;
    bool HasFrameGenerated(std::size_t trackIndex) const;
    bool Wait();

private:
    struct GeometryExecutorCallbackData {
        std::vector<std::size_t> frameIndices;
    } _callbackData{};

    nva2f::IGeometryExecutorBundle* _bundle;
};

class BlendshapeSolveExecutorResultsCollector {
public:
    void Init(nva2f::IBlendshapeExecutorBundle* bundle, benchmark::State& state);
    static void callbackForHostBlendshapeSolveExecutor(void* userdata, const nva2f::IBlendshapeExecutor::HostResults& results, std::error_code errorCode);
    static bool callbackForDeviceBlendshapeSolveExecutor(void* userdata, const nva2f::IBlendshapeExecutor::DeviceResults& results);
    void ResetCounters();
    std::size_t GetTotalFrames() const;
    bool HasFrameGenerated(std::size_t trackIndex) const;
    bool Wait();

private:
    struct BlendshapeSolveExecutorCallbackData {
        std::vector<std::size_t> frameIndices;
        std::vector<nva2x::HostTensorFloatView> weightViews;   // only used for device_results_callback
        benchmark::State* state;
    } _callbackData{};
    nva2f::IBlendshapeExecutorBundle* _bundle;
    std::vector<UniquePtr<nva2x::IHostTensorFloat>> _weightHostPinnedBatch;
};

template<typename A2FExecutorBundleType>
void RunExecutorOffline(
    benchmark::State& state,
    UniquePtr<A2FExecutorBundleType>& a2fExecutorBundle
);

template<typename A2FExecutorBundleType>
void RunExecutorStreaming(
    benchmark::State& state,
    std::size_t audioChunkSize,
    UniquePtr<A2FExecutorBundleType>& bundle
);
