import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const hud = document.getElementById("hud");
const panel = document.getElementById("panel");
const panelHeader = document.getElementById("panelHeader");
const panelHeaderHint = document.getElementById("panelHeaderHint");
const panelBody = document.getElementById("panelBody");

const audioPanel = document.getElementById("audioPanel");
const audioPanelHeader = document.getElementById("audioPanelHeader");
const audioPanelHeaderHint = document.getElementById("audioPanelHeaderHint");
const audioPanelBody = document.getElementById("audioPanelBody");
const audioFileInput = document.getElementById("audioFile");
const audioFileLabel = document.getElementById("audioFileLabel");
const audioSendBtn = document.getElementById("audioSend");
const audioStopBtn = document.getElementById("audioStop");
const audioProgress = document.getElementById("audioProgress");
const audioStatus = document.getElementById("audioStatus");

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f1a);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 2000);
camera.position.set(0.8, 0.7, 1.4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0.1, 0);
controls.update();

scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.1));

const keyLight = new THREE.DirectionalLight(0xffffff, 1.0);
keyLight.position.set(3, 4, 2);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xffffff, 0.35);
fillLight.position.set(-3, 2, -2);
scene.add(fillLight);

const grid = new THREE.GridHelper(2, 20, 0x1b2a41, 0x122033);
grid.position.y = 0;
scene.add(grid);

let tickA2FPlayback = null;

function setPanelCollapsed(collapsed) {
  if (!panel) return;
  panel.classList.toggle("collapsed", collapsed);
  panelHeader?.setAttribute("aria-expanded", collapsed ? "false" : "true");
  if (panelHeaderHint) panelHeaderHint.textContent = collapsed ? "Show" : "Hide";
}

function setAudioPanelCollapsed(collapsed) {
  if (!audioPanel) return;
  audioPanel.classList.toggle("collapsed", collapsed);
  audioPanelHeader?.setAttribute("aria-expanded", collapsed ? "false" : "true");
  if (audioPanelHeaderHint) audioPanelHeaderHint.textContent = collapsed ? "Show" : "Hide";
}

function makeEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text != null) el.textContent = String(text);
  return el;
}

setAudioPanelCollapsed(true);
const toggleAudioPanel = () => setAudioPanelCollapsed(!audioPanel?.classList.contains("collapsed"));
audioPanelHeader?.addEventListener("click", toggleAudioPanel);
audioPanelHeader?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    toggleAudioPanel();
  }
});

function getMorphMeshes(root) {
  const meshes = [];
  root.traverse((o) => {
    const hasInfluences = Array.isArray(o.morphTargetInfluences);
    const hasDict = o.morphTargetDictionary && typeof o.morphTargetDictionary === "object";
    if ((o.isMesh || o.isSkinnedMesh) && hasInfluences && hasDict) meshes.push(o);
  });
  return meshes;
}

function buildBlendshapePanel(root) {
  if (!panel || !panelBody) return;

  const morphMeshes = getMorphMeshes(root);
  if (morphMeshes.length === 0) {
    panel.hidden = true;
    return;
  }

  panel.hidden = false;
  setPanelCollapsed(true);
  panelBody.replaceChildren();

  panelHeaderHint.style.pointerEvents = "none";

  const totalTargets = morphMeshes.reduce((acc, m) => acc + Object.keys(m.morphTargetDictionary || {}).length, 0);
  const headerTitle = document.getElementById("panelHeaderTitle");
  if (headerTitle) headerTitle.textContent = `Blendshapes (${totalTargets})`;

  const headerClick = () => setPanelCollapsed(!panel.classList.contains("collapsed"));
  panelHeader?.addEventListener("click", headerClick);
  panelHeader?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      headerClick();
    }
  });

  const resetAll = makeEl("button", null, "Reset all");
  resetAll.type = "button";
  resetAll.style.width = "100%";
  resetAll.style.padding = "8px 10px";
  resetAll.style.borderRadius = "10px";
  resetAll.style.border = "1px solid rgba(255,255,255,0.12)";
  resetAll.style.background = "rgba(255,255,255,0.06)";
  resetAll.style.color = "white";
  resetAll.style.cursor = "pointer";
  resetAll.addEventListener("click", () => {
    const sliders = panelBody.querySelectorAll("input[type='range']");
    sliders.forEach((s) => {
      s.value = "0";
      s.dispatchEvent(new Event("input", { bubbles: true }));
    });
  });
  panelBody.appendChild(resetAll);
  panelBody.appendChild(makeEl("div", "divider", ""));

  morphMeshes.forEach((mesh, meshIdx) => {
    const title = makeEl("div", "sectionTitle", mesh.name && mesh.name.trim().length > 0 ? mesh.name : `Mesh ${meshIdx + 1}`);
    panelBody.appendChild(title);

    const dictEntries = Object.entries(mesh.morphTargetDictionary || {}).sort((a, b) => a[0].localeCompare(b[0]));
    dictEntries.forEach(([name, index]) => {
      const wrapper = makeEl("div", "sliderRow", "");

      const topRow = makeEl("div", "row", "");
      const label = makeEl("div", "label", name);
      const value = makeEl("div", "value", "");
      topRow.appendChild(label);
      topRow.appendChild(value);

      const slider = document.createElement("input");
      slider.type = "range";
      slider.min = "0";
      slider.max = "1";
      slider.step = "0.001";
      const initial = typeof mesh.morphTargetInfluences[index] === "number" ? mesh.morphTargetInfluences[index] : 0;
      slider.value = String(initial);
      value.textContent = Number(initial).toFixed(3);

      slider.addEventListener("input", () => {
        const v = Number(slider.value);
        mesh.morphTargetInfluences[index] = v;
        value.textContent = v.toFixed(3);
      });

      wrapper.appendChild(topRow);
      wrapper.appendChild(slider);
      panelBody.appendChild(wrapper);
    });

    panelBody.appendChild(makeEl("div", "divider", ""));
  });
}

function inferModelUrl() {
  const params = new URLSearchParams(window.location.search);
  const value = params.get("model");
  if (value && value.trim().length > 0) return value.trim();
  return "./model.gltf";
}

function fitCameraToObject(object3d) {
  const box = new THREE.Box3().setFromObject(object3d);
  if (box.isEmpty()) return;

  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  const maxDim = Math.max(size.x, size.y, size.z);
  const fov = (camera.fov * Math.PI) / 180;
  const distance = (maxDim * 0.5) / Math.tan(fov * 0.5);

  const direction = new THREE.Vector3(1, 0.75, 1).normalize();
  camera.position.copy(center).add(direction.multiplyScalar(distance * 1.6));
  camera.near = Math.max(0.001, distance / 100);
  camera.far = Math.max(10, distance * 50);
  camera.updateProjectionMatrix();

  controls.target.copy(center);
  controls.update();
}

function getStats(object3d) {
  let meshes = 0;
  let tris = 0;

  object3d.traverse((o) => {
    if (!o.isMesh) return;
    meshes += 1;
    const geom = o.geometry;
    if (!geom) return;
    const index = geom.index;
    const pos = geom.attributes?.position;
    if (index) tris += Math.floor(index.count / 3);
    else if (pos) tris += Math.floor(pos.count / 3);
  });

  return { meshes, tris };
}

const loader = new GLTFLoader();
const modelUrl = inferModelUrl();

hud.textContent = `Loading: ${modelUrl}`;

function inferWsUrl() {
  const params = new URLSearchParams(window.location.search);
  const value = params.get("ws");
  if (value && value.trim().length > 0) return value.trim();
  const host = window.location.hostname && window.location.hostname.length > 0 ? window.location.hostname : "localhost";
  return `ws://${host}:8765`;
}

function parseBoolParam(params, name, fallback) {
  const v = params.get(name);
  if (v == null) return fallback;
  const s = String(v).trim().toLowerCase();
  if (s === "1" || s === "true" || s === "yes" || s === "on") return true;
  if (s === "0" || s === "false" || s === "no" || s === "off") return false;
  return fallback;
}

function parseFloatParam(params, name, fallback) {
  const v = params.get(name);
  if (v == null) return fallback;
  const x = Number(v);
  return Number.isFinite(x) ? x : fallback;
}

function clamp01(x) {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

function makeMorphApplier(root) {
  const morphMeshes = getMorphMeshes(root);
  const missingLogged = new Set();

  const entries = morphMeshes.map((mesh) => {
    const dict = mesh.morphTargetDictionary || {};
    const influences = mesh.morphTargetInfluences || [];
    return { mesh, dict, influences };
  });

  return {
    entries,
    missingLogged,
    apply(weightsByName) {
      if (!weightsByName || typeof weightsByName !== "object") return;
      for (const [name, value] of Object.entries(weightsByName)) {
        const v = Number(value);
        if (!Number.isFinite(v)) continue;

        let applied = false;
        for (const { dict, influences } of entries) {
          const idx = dict[name];
          if (typeof idx === "number" && idx >= 0 && idx < influences.length) {
            influences[idx] = clamp01(v);
            applied = true;
          }
        }

        if (!applied && !missingLogged.has(name)) {
          missingLogged.add(name);
          console.warn(`[viewer] Missing blendshape on mesh: ${name}`);
        }
      }
    },
  };
}

function makeEmaSmoother({ enabled, alpha }) {
  const state = new Map();

  return {
    get enabled() {
      return enabled;
    },
    set enabled(v) {
      enabled = Boolean(v);
    },
    get alpha() {
      return alpha;
    },
    set alpha(v) {
      const x = Number(v);
      if (!Number.isFinite(x)) return;
      alpha = Math.max(0, Math.min(1, x));
    },
    smooth(weightsByName) {
      if (!enabled || !weightsByName || typeof weightsByName !== "object") return weightsByName;
      const out = {};
      for (const [name, value] of Object.entries(weightsByName)) {
        const v = Number(value);
        if (!Number.isFinite(v)) continue;
        const prev = state.get(name);
        const next = typeof prev === "number" ? alpha * v + (1 - alpha) * prev : v;
        state.set(name, next);
        out[name] = next;
      }
      return out;
    },
  };
}

function createDeferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function withTimeout(promise, ms, message) {
  let timer = null;
  const timeout = new Promise((_, reject) => {
    timer = window.setTimeout(() => reject(new Error(message)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => {
    if (timer != null) window.clearTimeout(timer);
  });
}

const A2F_FRAME_MAGIC_A2FB = 0x42463241; // "A2FB" little-endian
const A2F_FRAME_HEADER_BYTES = 40;

function getBigInt64LE(dv, offset) {
  if (typeof dv.getBigInt64 === "function") return dv.getBigInt64(offset, true);
  const lo = dv.getUint32(offset, true);
  const hi = dv.getInt32(offset + 4, true);
  return (BigInt(hi) << 32n) | BigInt(lo);
}

function getBigUint64LE(dv, offset) {
  if (typeof dv.getBigUint64 === "function") return dv.getBigUint64(offset, true);
  const lo = dv.getUint32(offset, true);
  const hi = dv.getUint32(offset + 4, true);
  return (BigInt(hi) << 32n) | BigInt(lo);
}

function writeU64LE(dv, offset, value) {
  const v = BigInt(value);
  dv.setUint32(offset, Number(v & 0xffffffffn), true);
  dv.setUint32(offset + 4, Number((v >> 32n) & 0xffffffffn), true);
}

function parseBlendshapeFrame(arrayBuffer, channels) {
  if (!(arrayBuffer instanceof ArrayBuffer)) return null;
  if (arrayBuffer.byteLength < A2F_FRAME_HEADER_BYTES) return null;

  const dv = new DataView(arrayBuffer);
  const magic = dv.getUint32(0, true);
  if (magic !== A2F_FRAME_MAGIC_A2FB) return null;

  const version = dv.getUint32(4, true);
  if (version !== 1) return null;

  const weightCount = dv.getUint32(8, true);
  const requiredBytes = A2F_FRAME_HEADER_BYTES + weightCount * 4;
  if (arrayBuffer.byteLength < requiredBytes) return null;

  const frameIndex = getBigUint64LE(dv, 16);
  const tsCurrent = getBigInt64LE(dv, 24);
  const tsNext = getBigInt64LE(dv, 32);

  const out = {};
  const base = A2F_FRAME_HEADER_BYTES;
  for (let i = 0; i < weightCount; ++i) {
    const name = Array.isArray(channels) && i < channels.length ? channels[i] : `w${i}`;
    out[name] = dv.getFloat32(base + i * 4, true);
  }

  return { frameIndex, tsCurrent, tsNext, weightCount, weightsByName: out };
}

function parseProcessedAudioChunks(arrayBuffer) {
  if (!(arrayBuffer instanceof ArrayBuffer)) throw new Error("Invalid processed audio buffer");
  if (arrayBuffer.byteLength < 16) throw new Error("Processed audio buffer too small");

  const bytes = new Uint8Array(arrayBuffer);
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  if (magic !== "A2PC") throw new Error("Invalid processed audio magic");

  const dv = new DataView(arrayBuffer);
  const version = dv.getUint32(4, true);
  if (version !== 1) throw new Error(`Unsupported processed audio version: ${version}`);

  const sampleRate = dv.getUint32(8, true);
  const chunkCount = dv.getUint32(12, true);
  let offset = 16;

  const chunks = [];
  for (let i = 0; i < chunkCount; ++i) {
    if (offset + 12 > arrayBuffer.byteLength) throw new Error("Truncated processed audio header");
    const startSampleIndex = getBigInt64LE(dv, offset);
    const sampleCount = dv.getUint32(offset + 8, true);
    offset += 12;
    const byteCount = sampleCount * 2;
    if (offset + byteCount > arrayBuffer.byteLength) throw new Error("Truncated processed audio payload");
    const pcm16 = bytes.subarray(offset, offset + byteCount);
    offset += byteCount;
    chunks.push({ startSampleIndex, pcm16 });
  }

  return { sampleRate, chunks };
}

function buildPushAudioMessage(startSampleIndex, pcm16Bytes) {
  const pcm = pcm16Bytes instanceof Uint8Array ? pcm16Bytes : new Uint8Array(pcm16Bytes);
  if ((pcm.byteLength % 2) !== 0) throw new Error("PCM16 payload must be aligned to int16");

  const buf = new ArrayBuffer(8 + pcm.byteLength);
  const dv = new DataView(buf);
  writeU64LE(dv, 0, startSampleIndex);
  new Uint8Array(buf, 8).set(pcm);
  return buf;
}

function attachA2FWebSocket({ applier, smoother, hudState }) {
  const url = inferWsUrl();
  let ws = null;
  let reconnectTimer = null;
  let backoffMs = 250;

  let openDeferred = null;
  let sessionDeferred = null;
  let endDeferred = null;

  let session = null;

  const playback = {
    active: false,
    baseTimeMs: null,
    sampleRate: 16000,
    queue: [],
    head: 0,
  };

  const setStatus = (status) => {
    hudState.wsStatus = status;
  };

  const connect = () => {
    reconnectTimer = null;
    try {
      ws = new WebSocket(url);
    } catch (e) {
      setStatus("error");
      scheduleReconnect();
      return;
    }

    ws.binaryType = "arraybuffer";
    openDeferred = createDeferred();

    setStatus("connecting");

    ws.onopen = () => {
      backoffMs = 250;
      setStatus("connected");
      openDeferred?.resolve();
      openDeferred = null;
      void startSession();
    };

    ws.onclose = () => {
      setStatus("disconnected");
      session = null;
      playback.active = false;
      playback.queue.length = 0;
      playback.baseTimeMs = null;
      playback.head = 0;
      openDeferred?.reject(new Error("WebSocket closed"));
      openDeferred = null;
      sessionDeferred?.reject(new Error("WebSocket closed"));
      sessionDeferred = null;
      endDeferred?.reject(new Error("WebSocket closed"));
      endDeferred = null;
      scheduleReconnect();
    };

    ws.onerror = () => {
      setStatus("error");
    };

    ws.onmessage = async (evt) => {
      const payload = evt.data;
      if (typeof payload === "string") {
        let msg;
        try {
          msg = JSON.parse(payload);
        } catch {
          return;
        }

        const type = msg?.type;
        if (type === "SessionStarted") {
          session = msg;
          hudState.sessionId = typeof msg?.session_id === "string" ? msg.session_id : null;
          hudState.sampleRate = typeof msg?.sampling_rate === "number" ? msg.sampling_rate : null;
          hudState.channelCount = typeof msg?.weight_count === "number" ? msg.weight_count : null;
          sessionDeferred?.resolve(msg);
          sessionDeferred = null;
          return;
        }
        if (type === "SessionEnded") {
          session = null;
          hudState.sessionId = null;
          endDeferred?.resolve(msg);
          endDeferred = null;
          return;
        }
        if (type === "Error") {
          const message = typeof msg?.message === "string" ? msg.message : "Unknown error";
          hudState.lastError = message;
          sessionDeferred?.reject(new Error(message));
          sessionDeferred = null;
          return;
        }

        return;
      }

      let buf = null;
      if (payload instanceof ArrayBuffer) {
        buf = payload;
      } else if (payload instanceof Blob) {
        buf = await payload.arrayBuffer();
      }
      if (!buf) return;

      const channels = session && Array.isArray(session.channels) ? session.channels : null;
      const frame = parseBlendshapeFrame(buf, channels);
      if (!frame) return;

      hudState.lastFrameIndex = frame.frameIndex;
      hudState.lastFrameAtMs = performance.now();

      if (!playback.active) {
        const smoothed = smoother.smooth(frame.weightsByName);
        applier.apply(smoothed);
        return;
      }

      const sr = typeof session?.sampling_rate === "number" ? session.sampling_rate : playback.sampleRate;
      playback.sampleRate = sr;
      const tsMs = Number(frame.tsCurrent) * 1000.0 / sr;
      if (playback.baseTimeMs == null) {
        playback.baseTimeMs = performance.now() - tsMs;
      }
      const playAtMs = playback.baseTimeMs + tsMs;
      playback.queue.push({ playAtMs, weightsByName: frame.weightsByName });
    };
  };

  const scheduleReconnect = () => {
    if (reconnectTimer != null) return;
    const delay = Math.min(5000, backoffMs);
    backoffMs = Math.min(5000, backoffMs * 2);
    reconnectTimer = window.setTimeout(connect, delay);
  };

  connect();

  window.addEventListener("beforeunload", () => {
    try {
      if (ws) ws.close();
    } catch {}
  });

  const waitForOpen = async () => {
    if (ws && ws.readyState === WebSocket.OPEN) return;
    if (!openDeferred) throw new Error("WebSocket not connected");
    await openDeferred.promise;
  };

  const startSession = async () => {
    if (session) return session;
    await waitForOpen();
    if (!ws || ws.readyState !== WebSocket.OPEN) throw new Error("WebSocket is not open");

    sessionDeferred = createDeferred();
    ws.send(JSON.stringify({ type: "StartSession" }));
    return await withTimeout(sessionDeferred.promise, 5000, "StartSession timed out");
  };

  const endSession = async () => {
    if (!session) return;
    await waitForOpen();
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!session.session_id) return;

    endDeferred = createDeferred();
    ws.send(JSON.stringify({ type: "EndSession", session_id: session.session_id }));
    try {
      await withTimeout(endDeferred.promise, 2000, "EndSession timed out");
    } finally {
      endDeferred = null;
    }
  };

  const restartSession = async () => {
    try {
      await endSession();
    } catch {}
    session = null;
    hudState.sessionId = null;
    hudState.lastError = null;
    return await startSession();
  };

  const beginPlayback = () => {
    playback.active = true;
    playback.baseTimeMs = null;
    playback.queue.length = 0;
    playback.head = 0;
  };

  const stopPlayback = () => {
    playback.active = false;
    playback.baseTimeMs = null;
    playback.queue.length = 0;
    playback.head = 0;
  };

  const tick = (nowMs) => {
    if (!playback.active || playback.head >= playback.queue.length) return;
    let latest = null;
    while (playback.head < playback.queue.length && playback.queue[playback.head].playAtMs <= nowMs) {
      latest = playback.queue[playback.head];
      playback.head += 1;
    }
    if (!latest) return;
    const smoothed = smoother.smooth(latest.weightsByName);
    applier.apply(smoothed);
    hudState.queueDepth = playback.queue.length - playback.head;

    if (playback.head > 2048) {
      playback.queue = playback.queue.slice(playback.head);
      playback.head = 0;
    }
  };

  const sendAudioChunk = async (startSampleIndex, pcm16leBytes) => {
    await startSession();
    if (!ws || ws.readyState !== WebSocket.OPEN) throw new Error("WebSocket is not open");
    ws.send(buildPushAudioMessage(startSampleIndex, pcm16leBytes));
  };

  return { url, startSession, endSession, restartSession, sendAudioChunk, beginPlayback, stopPlayback, tick };
}

loader.load(
  modelUrl,
  (gltf) => {
    const root = gltf.scene || gltf.scenes?.[0];
    if (!root) {
      hud.textContent = "Loaded, but no scene found in GLTF.";
      return;
    }

    scene.add(root);
    fitCameraToObject(root);
    buildBlendshapePanel(root);

    const { meshes, tris } = getStats(root);

    const params = new URLSearchParams(window.location.search);
    const smoother = makeEmaSmoother({
      enabled: parseBoolParam(params, "ema", false),
      alpha: parseFloatParam(params, "alpha", 0.6),
    });

    const applier = makeMorphApplier(root);
    const hudState = {
      wsStatus: "init",
      sessionId: null,
      sampleRate: null,
      channelCount: null,
      queueDepth: 0,
      lastFrameIndex: null,
      lastFrameAtMs: null,
      lastError: null,
    };

    const wsClient = attachA2FWebSocket({ applier, smoother, hudState });
    const wsUrl = wsClient.url;
    tickA2FPlayback = wsClient.tick;

    window.addEventListener("keydown", (e) => {
      if (e.key === "s" || e.key === "S") {
        smoother.enabled = !smoother.enabled;
      }
    });

    const setAudioUi = ({ canSend, canStop, statusText, progressText }) => {
      if (audioSendBtn) audioSendBtn.disabled = !canSend;
      if (audioStopBtn) audioStopBtn.disabled = !canStop;
      if (audioStatus && statusText != null) audioStatus.textContent = statusText;
      if (audioProgress && progressText != null) audioProgress.textContent = progressText;
    };

    let activeStreamToken = null;

    const updateSelectedFile = () => {
      const file = audioFileInput?.files?.[0] || null;
      if (audioFileLabel) audioFileLabel.textContent = file ? file.name : "No file selected";
      setAudioUi({ canSend: Boolean(file), canStop: false, statusText: "Idle", progressText: "-" });
    };
    audioFileInput?.addEventListener("change", updateSelectedFile);
    updateSelectedFile();

    const processAudioViaLocalServer = async (file) => {
      const buf = await file.arrayBuffer();
      const resp = await fetch(`/api/process_audio?sr=16000&chunk_s=1.0`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: buf,
      });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || `Audio processing failed (${resp.status})`);
      }
      const outBuf = await resp.arrayBuffer();
      return parseProcessedAudioChunks(outBuf);
    };

    const streamAudioFile = async (file) => {
      const token = { cancelled: false };
      activeStreamToken = token;

      setAudioUi({ canSend: false, canStop: true, statusText: "Preparing session...", progressText: "-" });
      await wsClient.restartSession();
      wsClient.beginPlayback();

      setAudioUi({ canSend: false, canStop: true, statusText: "Processing audio (WAV -> 16k PCM16)...", progressText: "-" });
      const processed = await processAudioViaLocalServer(file);

      const chunks = processed.chunks || [];
      setAudioUi({ canSend: false, canStop: true, statusText: `Streaming ${chunks.length} chunk(s)...`, progressText: `0/${chunks.length}` });

      for (let i = 0; i < chunks.length; ++i) {
        if (token.cancelled) break;
        const c = chunks[i];
        await wsClient.sendAudioChunk(c.startSampleIndex, c.pcm16);
        setAudioUi({ canSend: false, canStop: true, progressText: `${i + 1}/${chunks.length}` });
      }

      if (!token.cancelled) {
        setAudioUi({ canSend: true, canStop: false, statusText: "Done", progressText: `${chunks.length}/${chunks.length}` });
      } else {
        wsClient.stopPlayback();
        setAudioUi({ canSend: true, canStop: false, statusText: "Stopped", progressText: "-" });
      }
      activeStreamToken = null;
    };

    audioSendBtn?.addEventListener("click", async () => {
      const file = audioFileInput?.files?.[0] || null;
      if (!file) return;
      try {
        await streamAudioFile(file);
      } catch (e) {
        wsClient.stopPlayback();
        activeStreamToken = null;
        const msg = e?.message ? String(e.message) : String(e);
        setAudioUi({ canSend: true, canStop: false, statusText: `Error: ${msg}`, progressText: "-" });
      }
    });

    audioStopBtn?.addEventListener("click", () => {
      if (activeStreamToken) activeStreamToken.cancelled = true;
    });

    hud.textContent =
      `Loaded: ${modelUrl}\n` +
      `Meshes: ${meshes}\n` +
      `Triangles: ${tris}\n` +
      `WS: ${wsUrl}\n` +
      `Controls: orbit (LMB), pan (RMB), zoom (wheel)\n` +
      `Toggle EMA: press 'S'  (or use ?ema=1&alpha=0.6)`;

    const updateHud = () => {
      const ws = hudState.wsStatus;
      const sessionId = hudState.sessionId;
      const frameIndex = hudState.lastFrameIndex;
      const err = hudState.lastError;
      const ageMs = hudState.lastFrameAtMs != null ? performance.now() - hudState.lastFrameAtMs : null;

      const lines = [
        `Loaded: ${modelUrl}`,
        `Meshes: ${meshes}`,
        `Triangles: ${tris}`,
        `WS: ${wsUrl} (${ws})`,
        `Session: ${sessionId != null ? sessionId : "none"}${hudState.sampleRate ? ` sr=${hudState.sampleRate}` : ""}${hudState.channelCount ? ` weights=${hudState.channelCount}` : ""}`,
        `EMA: ${smoother.enabled ? `on (alpha=${smoother.alpha.toFixed(2)})` : "off"}`,
        `Last: ${frameIndex != null ? `frame=${String(frameIndex)}` : "none"}${ageMs != null ? ` age=${Math.round(ageMs)}ms` : ""}${hudState.queueDepth ? ` queue=${hudState.queueDepth}` : ""}`,
        err ? `Error: ${err}` : null,
        `Controls: orbit (LMB), pan (RMB), zoom (wheel)`,
        `Toggle EMA: press 'S'  (or use ?ema=1&alpha=0.6)`,
      ].filter(Boolean);

      hud.textContent = lines.join("\n");
      requestAnimationFrame(updateHud);
    };

    updateHud();
  },
  (evt) => {
    if (!evt.total) return;
    const pct = Math.max(0, Math.min(100, Math.round((evt.loaded / evt.total) * 100)));
    hud.textContent = `Loading: ${modelUrl} (${pct}%)`;
  },
  (err) => {
    const message = err?.message ? String(err.message) : String(err);
    hud.textContent =
      `Failed to load: ${modelUrl}\n` +
      `If opening via file://, use a local server.\n` +
      `Error: ${message}`;
  }
);

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

window.addEventListener("resize", onResize, { passive: true });

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  if (tickA2FPlayback) tickA2FPlayback(performance.now());
  renderer.render(scene, camera);
}

animate();
