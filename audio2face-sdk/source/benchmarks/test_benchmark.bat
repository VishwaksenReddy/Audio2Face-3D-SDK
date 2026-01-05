@echo off

set "COMMAND=%*"

@REM At least one arg should be passed
if "%1"=="" (
    echo "Usage: test_benchmark.bat <executable> [args..]"
    exit /b 1
)

REM Define the list of arguments
set args[0]=--benchmark_filter=BM_RegressionBlendshapeSolveExecutorOffline/FP16:0/UseGPU:0/Identity:0/ExecutionOption:1/NbTracks:1/real_time
set args[1]=--benchmark_filter=BM_RegressionBlendshapeSolveExecutorStreaming/FP16:0/UseGPU:0/Identity:1/ExecutionOption:1/AudioChunkSize:100/NbTracks:8/real_time
set args[2]=--benchmark_filter=BM_DiffusionBlendshapeSolveExecutorStreaming/FP16:0/UseGPU:1/Identity:2/ExecutionOption:1/AudioChunkSize:100/NbTracks:8/real_time
set args[3]=--benchmark_filter=BM_DiffusionGeometryExecutorOffline/FP16:0/Identity:0/ExecutionOption:3/NbTracks:8/real_time
set args[4]=--benchmark_filter=BM_RegressionGeometryExecutorStreaming/FP16:0/Identity:0/ExecutionOption:7/AudioChunkSize:1/NbTracks:1/real_time
set args[5]=--benchmark_filter=BM_DiffusionGeometryExecutorStreaming/FP16:0/Identity:0/ExecutionOption:2/AudioChunkSize:100/NbTracks:1/real_time
set args[6]=--benchmark_filter=BM_InteractiveExecutorBatch/BatchSize:0/Regression:1/real_time
set args[7]=--benchmark_filter=BM_GeometryInteractiveExecutorLayer/Layer:0/LookBack:0/Regression:1/real_time
set args[8]=--benchmark_filter=BM_BlendshapeInteractiveExecutorLayer/Layer:0/UseGpuSolver:0/LookBack:0/Regression:1/real_time
set args[9]=--benchmark_filter=BlendshapeSolverBenchmark/BM_Solve/0/0/real_time
set args[10]=--benchmark_filter=BlendshapeSolverBenchmark/BM_CPUSolveAsync/0/3/2/real_time
set args[11]=--benchmark_filter=BlendshapeSolverBatchBenchmark/BM_GPUSolveAsync/10/4/real_time

setlocal enabledelayedexpansion

REM Iterate over the arguments and execute the benchmark command
for /L %%i in (0,1,11) do (
    set "current_arg=!args[%%i]!"
    echo Running: %COMMAND% !current_arg!
    call %COMMAND% !current_arg!

    if !errorlevel! neq 0 (
        exit /b !errorlevel!
    )
)
