# Audio2Face 3D SDK

The Audio2Face SDK is a GPU-accelerated toolkit for generating 3D facial animation from speech audio. It uses CUDA and TensorRT for high-performance inference and supports both batch and streaming workflows.

## Build

### Prerequisites

- Windows 10/11 or Linux (Ubuntu 20.04+)
- NVIDIA GPU + CUDA (12.8+ recommended)
- TensorRT (10.13+ recommended)
- Git + Git LFS
- Python 3.8–3.10 (for optional model/data scripts)

### Building (default scripts)

#### Windows

```powershell
git clone https://github.com/NVIDIA/Audio2Face-3D-SDK.git
cd Audio2Face-3D-SDK
git lfs pull
.\fetch_deps.bat release

$env:TENSORRT_ROOT_DIR="C:\path\to\tensorrt"
$env:CUDA_PATH="C:\path\to\cuda"

.\build.bat clean release
.\build.bat all release
```

#### Linux

```bash
git clone https://github.com/NVIDIA/Audio2Face-3D-SDK.git
cd Audio2Face-3D-SDK
git lfs pull
./fetch_deps.sh release

export TENSORRT_ROOT_DIR="path/to/tensorrt"
export CUDA_PATH="path/to/cuda"

./build.sh clean release
./build.sh all release
```

### Build output structure

After a successful build, the main artifacts are located under `_build/<config>/`:

```
_build/
  release/
    audio2face-sdk/
      bin/   # Audio2Face samples, tests, benchmarks
      lib/   # Audio2Face static libraries
    audio2x-common/
      bin/   # Common unit test executables (if enabled)
      lib/   # Common static libraries
    audio2x-sdk/
      bin/      # audio2x shared library (audio2x.dll / libaudio2x.so)
      include/  # Header files (Audio2Face + common)
      lib/      # Import library (Windows) / shared library (Linux)
```

## Models and test/sample data (optional)

The helper scripts download model assets and generate TensorRT engines plus sample/test data. This is required to run the shipped samples/tests/benchmarks, but not required to build the SDK:

### Windows

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r deps\requirements.txt

hf auth login    # if required for access
.\download_models.bat
.\gen_testdata.bat
```

### Linux

```bash
python -m venv venv
source ./venv/bin/activate
pip install -r deps/requirements.txt

hf auth login    # if required for access
./download_models.sh
./gen_testdata.sh
```

## Running samples and tests

Use `run_sample.{bat|sh}` to run executables with the required CUDA/TensorRT libraries on the PATH.

### Windows

```powershell
.\run_sample.bat .\_build\release\audio2face-sdk\bin\audio2face-unit-tests.exe
.\run_sample.bat .\_build\release\audio2face-sdk\bin\sample-a2f-executor.exe
```

### Linux

```bash
./run_sample.sh ./_build/release/audio2face-sdk/bin/audio2face-unit-tests
./run_sample.sh ./_build/release/audio2face-sdk/bin/sample-a2f-executor
```
