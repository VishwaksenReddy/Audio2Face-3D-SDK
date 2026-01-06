# Audio2Face Inference Server (WebSocket)

Build target: `audio2face-inference-server` (output: `_build/<config>/audio2face-sdk/bin/audio2face-inference-server[.exe]`).

## Run (Windows)

Recommended (sets up `PATH` for `audio2x.dll`, CUDA, TensorRT):

```powershell
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
$env:TENSORRT_ROOT_DIR="C:\TensorRT-10.13.0.35"
.\run_sample.bat .\_build\release\audio2face-sdk\bin\audio2face-inference-server.exe --host 127.0.0.1 --port 8765
```

## WebSocket API

This server currently supports **one session per WebSocket connection**.

### `StartSession` (client -> server, text)

```json
{
  "type": "StartSession",
  "model": "_data/generated/audio2face-sdk/samples/data/mark/model.json",
  "fps": 60,
  "options": {
    "use_gpu_solver": true,
    "execution_option": "SkinTongue"
  }
}
```

### `SessionStarted` (server -> client, text)

All `StartSession` fields are optional; if provided, they must match the server configuration (the server is configured at startup and does not dynamically load models per session).

Contains `session_id`, `model`, `options`, `sampling_rate`, `frame_rate`, `weight_count`, and a stable `channels` array describing the weight ordering.

### `PushAudio` (client -> server, binary)

Binary payload:

- Bytes `0..7`: `startSampleIndex` (int64 little-endian, absolute sample index since session start)
- Bytes `8..`: PCM16 mono audio samples (int16 little-endian), sample rate = `16000`

Audio must be **monotonically increasing** in `startSampleIndex`. Gaps are zero-filled; out-of-order chunks are rejected.

### `EndSession` (client -> server, text)

```json
{"type":"EndSession","session_id":"<from SessionStarted>"}
```

### `BlendshapeFrame` (server -> client, binary)

Binary payload:

- `magic` (uint32 LE): `"A2FB"`
- `version` (uint32 LE): `1`
- `weight_count` (uint32 LE)
- `reserved` (uint32 LE): `0`
- `frame_index` (uint64 LE)
- `timestamp_current` (int64 LE, in samples @ 16kHz)
- `timestamp_next` (int64 LE, in samples @ 16kHz)
- `weights` (`weight_count` float32 LE)

## Concurrency

The server pre-allocates a pool of executors at startup; `--max_sessions` controls the maximum number of concurrent sessions (connections that successfully call `StartSession`).
