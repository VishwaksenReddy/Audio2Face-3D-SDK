@echo off

set "BASE_DIR=%~dp0"

set PYTHONPATH=%BASE_DIR%audio2x-common\scripts;%PYTHONPATH%

if not defined TENSORRT_ROOT_DIR (
    echo TENSORRT_ROOT_DIR is not defined
    exit /b 1
)

set "PATH=%TENSORRT_ROOT_DIR%\bin;%TENSORRT_ROOT_DIR%\lib;%PATH%"

echo Generating test data...
python "%BASE_DIR%audio2x-common/scripts/gen_test_data.py" || exit /b
python "%BASE_DIR%audio2face-sdk/scripts/gen_test_data.py" || exit /b

echo Generating sample data...
python "%BASE_DIR%audio2face-sdk/scripts/gen_sample_data.py" || exit /b
