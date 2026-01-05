#!/bin/bash
set -e

# At least one arg should be passed
if [ -z "$1" ]; then
    echo "Usage: test_benchmark.sh <executable> [args...]"
    exit 1
fi

# Define the list of arguments
args=(
    "--benchmark_filter=BM_RegressionBlendshapeSolveExecutorOffline/FP16:0/UseGPU:0/Identity:0/ExecutionOption:1/NbTracks:1/real_time"
    "--benchmark_filter=BM_RegressionBlendshapeSolveExecutorStreaming/FP16:0/UseGPU:0/Identity:1/ExecutionOption:1/AudioChunkSize:100/NbTracks:8/real_time"
    "--benchmark_filter=BM_DiffusionBlendshapeSolveExecutorStreaming/FP16:0/UseGPU:1/Identity:2/ExecutionOption:1/AudioChunkSize:100/NbTracks:8/real_time"
    "--benchmark_filter=BM_DiffusionGeometryExecutorOffline/FP16:0/Identity:0/ExecutionOption:3/NbTracks:8/real_time"
    "--benchmark_filter=BM_RegressionGeometryExecutorStreaming/FP16:0/Identity:0/ExecutionOption:7/AudioChunkSize:1/NbTracks:1/real_time"
    "--benchmark_filter=BM_DiffusionGeometryExecutorStreaming/FP16:0/Identity:0/ExecutionOption:2/AudioChunkSize:100/NbTracks:1/real_time"
    "--benchmark_filter=BM_InteractiveExecutorBatch/BatchSize:0/Regression:1/real_time"
    "--benchmark_filter=BM_GeometryInteractiveExecutorLayer/Layer:0/LookBack:0/Regression:1/real_time"
    "--benchmark_filter=BM_BlendshapeInteractiveExecutorLayer/Layer:0/UseGpuSolver:0/LookBack:0/Regression:1/real_time"
)

# Iterate over the arguments and execute the benchmark command
for current_arg in "${args[@]}"; do
    echo "Running: $@ $current_arg"
    bash "$@" "$current_arg"
done
