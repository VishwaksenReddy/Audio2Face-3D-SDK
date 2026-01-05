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

#include "audio2face/audio2face.h"
#include "audio2x/cuda_utils.h"
#include "audio2x/cuda_stream.h"
#include "audio2x/tensor_dict.h"

#include <benchmark/benchmark.h>

#include <sstream>

static void CustomRangesOffline(benchmark::internal::Benchmark* b, std::initializer_list<int64_t> nbTracksArg) {
    using ExecutionOption = nva2f::IGeometryExecutor::ExecutionOption;
    b->UseRealTime();
    b->ArgNames({"FP16", "Identity", "ExecutionOption", "NbTracks"});  // Assign meaningful names
    b->ArgsProduct({
        {0, 1},
        {0, 1, 2},
        {
            static_cast<int>(ExecutionOption::None),
            static_cast<int>(ExecutionOption::Skin),
            static_cast<int>(ExecutionOption::Tongue),
            static_cast<int>(ExecutionOption::Skin | ExecutionOption::Tongue),
            static_cast<int>(ExecutionOption::Skin | ExecutionOption::Tongue | ExecutionOption::Jaw),
            static_cast<int>(ExecutionOption::All)
        },
        nbTracksArg
    }); // Define for all the combinations
}

static void BM_RegressionGeometryExecutorOffline(benchmark::State& state) {
    bool useFP16 = state.range(0);
    auto identity = state.range(1);
    auto executionOption = static_cast<nva2f::IGeometryExecutor::ExecutionOption>(state.range(2));
    auto nbTracks = static_cast<std::size_t>(state.range(3));

    nva2f::IRegressionModel::IGeometryModelInfo* rawModelInfoPtr = nullptr;
    auto bundle = ToUniquePtr(
        nva2f::ReadRegressionGeometryExecutorBundle(
            nbTracks,
            useFP16 ? REGRESSION_MODELS_FP16[identity] : REGRESSION_MODELS[identity],
            executionOption,
            60, 1,
            &rawModelInfoPtr
        )
    );
    CHECK_AND_SKIP(bundle != nullptr);
    CHECK_AND_SKIP(rawModelInfoPtr != nullptr);
    auto modelInfo = ToUniquePtr(rawModelInfoPtr);

    std::ostringstream label;
    label << "FP16: " << useFP16
        << ", identity: " << modelInfo->GetNetworkInfo().GetIdentityName()
        << ", executionOption: " << geometryExecutionOptionToString(executionOption)
        << ", NbTracks: " << nbTracks
        ;
    state.SetLabel(label.str());

    RunExecutorOffline<nva2f::IGeometryExecutorBundle>(state, bundle);
}
BENCHMARK(BM_RegressionGeometryExecutorOffline)->Apply([](benchmark::internal::Benchmark* b) {
    // This can go up to 128 but it would be very slow to benchmark with all the combinations
    return CustomRangesOffline(b, {1, 2, 4, 8, 16});
});

static void BM_DiffusionGeometryExecutorOffline(benchmark::State& state) {
    bool useFP16 = state.range(0);
    auto identity = state.range(1);
    auto executionOption = static_cast<nva2f::IGeometryExecutor::ExecutionOption>(state.range(2));
    auto nbTracks = static_cast<std::size_t>(state.range(3));
    const auto constantNoise = true;

    nva2f::IDiffusionModel::IGeometryModelInfo* rawModelInfoPtr = nullptr;
    auto bundle = ToUniquePtr(
        nva2f::ReadDiffusionGeometryExecutorBundle(
            nbTracks,
            useFP16 ? DIFFUSION_MODEL_FP16 : DIFFUSION_MODEL,
            executionOption,
            identity,
            constantNoise,
            &rawModelInfoPtr
            )
        );
    CHECK_AND_SKIP(bundle != nullptr);
    CHECK_AND_SKIP(rawModelInfoPtr != nullptr);
    auto modelInfo = ToUniquePtr(rawModelInfoPtr);

    std::ostringstream label;
    label << "FP16: " << useFP16
        << ", identity: " << modelInfo->GetNetworkInfo().GetIdentityName(identity)
        << ", executionOption: " << geometryExecutionOptionToString(executionOption)
        << ", NbTracks: " << nbTracks
        ;
    state.SetLabel(label.str());

    RunExecutorOffline<nva2f::IGeometryExecutorBundle>(state, bundle);
}
BENCHMARK(BM_DiffusionGeometryExecutorOffline)->Apply([](benchmark::internal::Benchmark* b) {
    return CustomRangesOffline(b, {1, 2, 4, 8}); // Max batch size for diffusion is 8
});

static void CustomRangesStreaming(benchmark::internal::Benchmark* b, std::initializer_list<int64_t> nbTracksArg) {
    using ExecutionOption = nva2f::IGeometryExecutor::ExecutionOption;
    b->UseRealTime();
    b->ArgNames({"FP16", "Identity", "ExecutionOption", "AudioChunkSize","NbTracks"});  // Assign meaningful names
    b->ArgsProduct({
        {0, 1},
        {0, 1, 2},
        {
            static_cast<int>(ExecutionOption::None),
            static_cast<int>(ExecutionOption::Skin),
            static_cast<int>(ExecutionOption::Tongue),
            static_cast<int>(ExecutionOption::Skin | ExecutionOption::Tongue),
            static_cast<int>(ExecutionOption::Skin | ExecutionOption::Tongue | ExecutionOption::Jaw),
            static_cast<int>(ExecutionOption::All)
        },
        {1, 10, 100, 8000, 16000},
        nbTracksArg
    }); // Define for all the combinations
}

static void BM_RegressionGeometryExecutorStreaming(benchmark::State& state) {
    bool useFP16 = state.range(0);
    auto identity = state.range(1);
    auto executionOption = static_cast<nva2f::IGeometryExecutor::ExecutionOption>(state.range(2));
    auto audioChunkSize = static_cast<std::size_t>(state.range(3));
    auto nbTracks = static_cast<std::size_t>(state.range(4));

    nva2f::IRegressionModel::IGeometryModelInfo* rawModelInfoPtr = nullptr;
    auto bundle = ToUniquePtr(
        nva2f::ReadRegressionGeometryExecutorBundle(
            nbTracks,
            useFP16 ? REGRESSION_MODELS_FP16[identity] : REGRESSION_MODELS[identity],
            executionOption,
            60, 1,
            &rawModelInfoPtr
        )
    );
    CHECK_AND_SKIP(bundle != nullptr);
    CHECK_AND_SKIP(rawModelInfoPtr != nullptr);
    auto modelInfo = ToUniquePtr(rawModelInfoPtr);

    std::ostringstream label;
    label << "FP16: " << useFP16
        << ", identity: " << modelInfo->GetNetworkInfo().GetIdentityName()
        << ", executionOption: " << geometryExecutionOptionToString(executionOption)
        << ", AudioChunkSize: " << audioChunkSize
        << ", NbTracks: " << nbTracks
        ;
    state.SetLabel(label.str());

    RunExecutorStreaming<nva2f::IGeometryExecutorBundle>(state, audioChunkSize, bundle);
}
BENCHMARK(BM_RegressionGeometryExecutorStreaming)->Apply([](benchmark::internal::Benchmark* b) {
    // This can go up to 128 but it would be very slow to benchmark with all the combinations
    return CustomRangesStreaming(b, {1, 2, 4, 8, 16});
});

static void BM_DiffusionGeometryExecutorStreaming(benchmark::State& state) {
    bool useFP16 = state.range(0);
    auto identity = state.range(1);
    auto executionOption = static_cast<nva2f::IGeometryExecutor::ExecutionOption>(state.range(2));
    auto audioChunkSize = static_cast<std::size_t>(state.range(3));
    auto nbTracks = static_cast<std::size_t>(state.range(4));
    const auto constantNoise = true;

    nva2f::IDiffusionModel::IGeometryModelInfo* rawModelInfoPtr = nullptr;
    auto bundle = ToUniquePtr(
        nva2f::ReadDiffusionGeometryExecutorBundle(
            nbTracks,
            useFP16 ? DIFFUSION_MODEL_FP16 : DIFFUSION_MODEL,
            executionOption,
            identity,
            constantNoise,
            &rawModelInfoPtr
            )
        );
    CHECK_AND_SKIP(bundle != nullptr);
    CHECK_AND_SKIP(rawModelInfoPtr != nullptr);
    auto modelInfo = ToUniquePtr(rawModelInfoPtr);

    std::ostringstream label;
    label << "FP16: " << useFP16
        << ", identity: " << modelInfo->GetNetworkInfo().GetIdentityName(identity)
        << ", executionOption: " << geometryExecutionOptionToString(executionOption)
        << ", AudioChunkSize: " << audioChunkSize
        << ", NbTracks: " << nbTracks
        ;
    state.SetLabel(label.str());

    RunExecutorStreaming<nva2f::IGeometryExecutorBundle>(state, audioChunkSize, bundle);
}
BENCHMARK(BM_DiffusionGeometryExecutorStreaming)->Apply([](benchmark::internal::Benchmark* b) {
    return CustomRangesStreaming(b, {1, 2, 4, 8}); // Max batch size for diffusion is 8
});
