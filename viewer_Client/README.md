# Minimal three.js GLTF Viewer

## Usage

1. Copy your mesh into this folder as `model.gltf` (and any referenced `.bin` / textures).
2. Start the Audio2Face WebSocket inference server (default: `ws://127.0.0.1:8765`).
3. Start the local viewer server in this folder (serves the static files and exposes the audio processing endpoint used by the UI).

### Windows (PowerShell)

Start the inference server (from the repo root):

```powershell
.\run_sample.bat .\_build\release\audio2face-sdk\bin\audio2face-inference-server.exe --host 127.0.0.1 --port 8765
```

Start the viewer (from the repo root):

```powershell
cd viewer_Client
python serve_viewer.py --host 127.0.0.1 --port 8000
```

Then open `http://localhost:8000/` in your browser.

In the **Audio** panel, select a `.wav` file and click **Process + stream** to send PCM16@16kHz chunks to the C++ server and drive morph targets from the streamed blendshape frames.

Note: `serve_viewer.py` uses `numpy` + `scipy` to resample audio (install via `pip install -r deps/requirements.txt` from the repo root).

## Loading a different file

Use `?model=`:

- `http://localhost:8000/?model=your_mesh.gltf`

## If you see "Failed to resolve module specifier three"

This viewer uses an import map in `index.html`. Make sure you are serving the folder over HTTP (not opening the file directly via `file://`).
