# Audio2Face-3D-SDK — Inference Server Build → Bundle Plan

This plan defines how to build the SDK, download models, generate TensorRT engines, validate outputs, and then produce a minimal runnable inference-server “bundle” that keeps only what’s needed to serve WebSocket requests.

The inference server in this repo is a native C++ executable (`audio2face-inference-server`) that links against `audio2x.dll` and loads a `model.json` which references local TensorRT/NPZ assets under `_data/generated/...`.

## Goals

- Provide a **one-command pipeline** for developers:
  1) Build binaries into `_build/<config>/...`
  2) Download base models into `_data/...`
  3) Generate testdata / TensorRT engines into `_data/generated/...`
  4) Validate that the server can run and that the model assets are complete
  5) Produce a minimal “bundle” folder (preferred) or prune in-place (optional)
- Make the pipeline **repeatable** and **safe** (dry-run by default for pruning, clear preflight checks, deterministic file selection).

## Non-Goals

- Replacing the existing build/model scripts (`build.bat`, `download_models.bat`, `gen_testdata.bat`).
- Packaging CUDA/TensorRT runtimes; those remain external dependencies installed on the machine.

## Assumptions / Prerequisites (Windows)

- Windows 10/11.
- A supported compiler toolchain to build the CMake targets (Visual Studio Build Tools / MSVC as required by `build.bat`).
- NVIDIA GPU drivers installed.
- CUDA Toolkit installed and `CUDA_PATH` set (example: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`).
- TensorRT installed and `TENSORRT_ROOT_DIR` set (example: `C:\TensorRT-10.13.0.35`).
- You will run the server with `PATH` containing:
  - `_build\<config>\audio2x-sdk\bin` (for `audio2x.dll`)
  - `%CUDA_PATH%\bin`
  - `%TENSORRT_ROOT_DIR%\lib`

## Keep Set (Minimal Runtime Artifacts)

These are the **only repo artifacts** needed to run the server after the TRT engines are generated.

### Binaries (for one build config)

- `_build/<config>/audio2face-sdk/bin/audio2face-inference-server.exe`
- `_build/<config>/audio2face-sdk/bin/audio2face-inference-server.exe.manifest`
- `_build/<config>/audio2x-sdk/bin/audio2x.dll`
- `_build/<config>/audio2x-sdk/bin/audio2x.dll.manifest`

### Model assets (closure of `model.json`)

The server loads a `model.json` via `--model <path>`. The **entire directory containing that `model.json`** must be preserved, including all files referenced within it.

Default server model path is:
- `_data/generated/audio2face-sdk/samples/data/mark/model.json`

For the default `mark` regression model, keep at least:
- `_data/generated/audio2face-sdk/samples/data/mark/model.json`
- `_data/generated/audio2face-sdk/samples/data/mark/network.trt`
- `_data/generated/audio2face-sdk/samples/data/mark/network_info.json`
- `_data/generated/audio2face-sdk/samples/data/mark/implicit_emo_db.npz`
- `_data/generated/audio2face-sdk/samples/data/mark/model_config.json`
- `_data/generated/audio2face-sdk/samples/data/mark/model_data.npz`
- `_data/generated/audio2face-sdk/samples/data/mark/bs_skin_config.json`
- `_data/generated/audio2face-sdk/samples/data/mark/bs_skin.npz`
- `_data/generated/audio2face-sdk/samples/data/mark/bs_tongue_config.json`
- `_data/generated/audio2face-sdk/samples/data/mark/bs_tongue.npz`

If you choose a different identity/model, keep the corresponding folder under:
- `_data/generated/audio2face-sdk/samples/data/<identity>/...`

### Optional convenience files

- `run_sample.bat` (recommended): sets `PATH` for `audio2x.dll`, CUDA, and TensorRT before launching an exe.
- `LICENSE.txt` and `licenses/` (recommended if the bundle is redistributed internally/externally).

## Developer Deliverables

1) **Orchestrator script** that runs the pipeline end-to-end.
2) **Post-build validation** (file existence + model closure + optional smoke test).
3) **Cleanup/Prune** mechanism (bundle-by-copy preferred; in-place prune optional).
4) **Developer documentation**: how to run the pipeline and how to start the server.

## Orchestrator Script Specification

### Proposed location/name

- `tools/orchestrate_inference_server.ps1`
  - A PowerShell script is preferred because it can parse JSON for validation and has better error handling/logging than `.bat`.

### Interface (recommended)

- `-Config <release|debug>` (default: `release`)
- `-ModelJson <path>` (default: `_data/generated/audio2face-sdk/samples/data/mark/model.json`)
- `-Host <ip>` / `-Port <port>` (optional; for smoke tests only)
- `-OutputBundle <path>` (default: `dist/inference-server-bundle/<config>`)
- `-IncludeLicenses` (default: on)
- `-DryRun` (default: on for prune/bundle steps)
- `-InPlacePrune` (default: off; if on, requires `-Force`)
- `-Force` (required for destructive actions)
- `-SkipBuild` / `-SkipDownloadModels` / `-SkipGenTestdata` (for iterative dev)
- `-NoSmokeTest` (skip runtime validation)

### Behavior (high-level)

1) **Preflight**
   - Verify script is run from repo root (or compute repo root reliably).
   - Verify env vars: `CUDA_PATH`, `TENSORRT_ROOT_DIR`.
   - Verify expected directories exist after each step.
   - Confirm enough disk space (models are large).
   - Log configuration and resolved paths at start.

2) **Build**
   - Invoke: `.\build.bat all <config>`
   - Fail fast if exit code != 0.

3) **Download models**
   - Invoke: `.\download_models.bat`
   - Fail fast if exit code != 0.

4) **Generate TensorRT engines / testdata**
   - Invoke: `.\gen_testdata.bat`
   - Fail fast if exit code != 0.

5) **Post-build validation**
   - Verify binaries exist for the chosen `<config>`:
     - `_build/<config>/audio2face-sdk/bin/audio2face-inference-server.exe`
     - `_build/<config>/audio2x-sdk/bin/audio2x.dll`
   - Validate model JSON closure:
     - Read `-ModelJson`
     - Ensure all referenced files exist relative to the model directory
     - Optional: also check file sizes are non-zero where expected (TRT engine, NPZ, JSON).
   - Optional smoke test:
     - Start server process with `--host 127.0.0.1 --port <free port> --model <ModelJson>`
     - Wait for server to bind (timeout)
     - Optionally connect and issue `StartSession` over WebSocket (minimal client)
     - Terminate the server process cleanly

6) **Bundle (preferred)**
   - Create `-OutputBundle` directory.
   - Copy keep set into bundle:
     - `bin/` containing `audio2face-inference-server.exe` and `audio2x.dll` (and manifests)
     - `models/` containing the full model directory of `-ModelJson`
     - Optional `licenses/` and `LICENSE.txt`
     - Optional helper launch scripts (`start_server.ps1`, `start_server.bat`) that set PATH and run the server.
   - Write a `bundle_manifest.json` containing:
     - Build config
     - ModelJson path inside bundle
     - Timestamp
     - Git commit hash (if available)
     - List of copied files + sizes (and optionally SHA256)

7) **In-place prune (optional, destructive)**
   - Default off.
   - If enabled:
     - Require `-Force`.
     - Move non-kept items to a staging folder (or Windows Recycle Bin if feasible) instead of permanent deletion.
     - Re-run validation after prune.
   - Rationale: bundling-by-copy is safer and avoids breaking future builds.

## Post-Build Validation Details

### File checks (required)

- Ensure `audio2face-inference-server.exe` exists.
- Ensure `audio2x.dll` exists.
- Ensure `-ModelJson` exists.
- Ensure referenced model assets exist (relative paths inside `model.json`):
  - `networkInfoPath`
  - `networkPath` (TRT engine)
  - `emotionDatabasePath`
  - `modelConfigPath`
  - `modelDataPath`
  - `blendshapePaths.skin.config`, `blendshapePaths.skin.data`
  - `blendshapePaths.tongue.config`, `blendshapePaths.tongue.data`

### Smoke test (recommended)

- Launch:
  - `.\_build\<config>\audio2face-sdk\bin\audio2face-inference-server.exe --host 127.0.0.1 --port 8765 --model <ModelJson>`
- Ensure process stays alive for N seconds and binds the port.
- Optional: send a `StartSession` message and verify `SessionStarted` response.

## Cleanup / Prune Strategy

### Preferred output layout (bundle-by-copy)

Example:

- `<OutputBundle>/`
  - `bin/`
    - `audio2face-inference-server.exe`
    - `audio2face-inference-server.exe.manifest`
    - `audio2x.dll`
    - `audio2x.dll.manifest`
  - `models/`
    - `mark/` (or chosen identity)
      - `model.json`
      - `network.trt`
      - `network_info.json`
      - `*.npz`
      - `*.json`
  - `licenses/` (optional)
  - `start_server.ps1` (optional convenience)
  - `bundle_manifest.json`

### In-place prune (only if explicitly desired)

If the project wants to “delete everything and keep only the server” inside the repo:

- Keep:
  - `_build/<config>/audio2face-sdk/bin/audio2face-inference-server.exe*`
  - `_build/<config>/audio2x-sdk/bin/audio2x.dll*`
  - `_data/generated/audio2face-sdk/samples/data/<identity>/...` (or a chosen model directory)
  - `run_sample.bat` (optional)
  - `LICENSE.txt` + `licenses/` (recommended)
- Remove:
  - Source trees: `audio2face-sdk/`, `audio2x-sdk/`, `audio2x-common/`
  - Build/deps scaffolding: `cmake/`, `deps/`, `_deps/`, most of `_build/` except required bins
  - Samples/docs/tools/viewer if not needed: `viewer_Client/`, `docs/`, `tools/`, `sample-data/`

Destructive prune should be implemented as “move aside” first to allow recovery.

## How to Start the Server (after the pipeline)

From repo root (or from a bundle folder), ensure CUDA/TensorRT and `audio2x.dll` are discoverable via `PATH`.

Example (repo root):

```powershell
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
$env:TENSORRT_ROOT_DIR="C:\TensorRT-10.13.0.35"

.\run_sample.bat .\_build\release\audio2face-sdk\bin\audio2face-inference-server.exe --host 127.0.0.1 --port 8765 `
  --model "_data/generated/audio2face-sdk/samples/data/mark/model.json"
```

## Acceptance Criteria

- A developer can run the orchestrator once and get a bundle that starts successfully.
- The bundle contains only the keep set (plus optional convenience/licensing files).
- Validation catches missing/incorrect model asset references before shipping the bundle.
- The plan supports switching models by passing `-ModelJson`.

