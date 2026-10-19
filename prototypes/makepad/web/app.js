import { WasmWebGL } from "./makepad_platform/web_gl.js";
import { Microphone, endMessage } from "./microphone.js";
import { AudioSpool } from "./audio-spool.js";

// The UI and Burn each own a Wasm instance; neither shares pointers or GPU handles.
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
const state = window.prototypeState = { events: [], updates: [], errors: [], frames: 0 };
let diagnosticCharacters = 0;
let selectedFile;
let ui;
let audioRequestId = 0;
let pendingAudio = false;
let microphone;
let pumping = false;
const pending = [];
function send(update) {
  state.last = update;
  if (update.kind === "event" || update.kind === "done") {
    state.events.push(update);
    diagnosticCharacters += (update.text?.length || 0) + (update.delta?.length || 0);
    // Each event includes the complete transcript. Bound diagnostics by text
    // size as well as count during long sessions; the UI keeps the full result.
    while (state.events.length > 1 && (state.events.length > 4096 || diagnosticCharacters > 4_000_000)) {
      const old = state.events.shift();
      diagnosticCharacters -= (old.text?.length || 0) + (old.delta?.length || 0);
    }
  }
  state.updates.push(update.kind);
  if (state.updates.length > 1024) state.updates.shift();
  pending.push(update);
  if (!pumping) {
    pumping = true;
    queueMicrotask(() => {
      pumping = false;
      if (!ui?.to_wasm) return;
      for (const update of pending.splice(0)) ui.to_wasm.ToWasmUpdate({ json: JSON.stringify(update) });
      ui.do_wasm_pump();
    });
  }
}
function discardSpool(current) {
  current.spool?.close().catch(error => state.errors.push(`清理临时录音缓存失败：${error}`));
}
function interruptMicrophone(current, reason = "stopped") {
  if (current.cancelled || microphone !== current) return;
  current.cancelled = true;
  current.wantPacket = false;
  // Close the device and discard storage now. GPU interruption completes at its
  // next readback; it never drains the backlog or starts a final decoder pass.
  current.capture.close();
  discardSpool(current);
  const captured = current.capture.captured / (current.capture.context?.sampleRate || 16000);
  state.microphone = { ...state.microphone, phase: "interrupting", capturing: false,
    reason, captured_seconds: captured, backlog_seconds: 0, stored_seconds: 0 };
  send({ kind: "capture", capturing: false, captured_seconds: captured, message: "麦克风已释放 · 正在打断识别…" });
  if (current.started) worker.postMessage({ type: "mic-cancel", id: current.id, reason, captured_seconds: captured });
  else {
    microphone = undefined;
    state.microphone.phase = "closed";
    send({ kind: "done", text: "", delta: "", cancelled: true, message: endMessage(reason), captured_seconds: captured });
  }
}
function storageFailed(current, error) {
  if (microphone !== current || current.cancelled) return;
  state.errors.push(String(error));
  interruptMicrophone(current, "storage-error");
}
async function pullPacket(current) {
  if (microphone !== current || current.cancelled || current.pulling || !current.wantPacket) return;
  current.pulling = true;
  try {
    const pcm = await current.spool.shift();
    if (microphone !== current || current.cancelled) return;
    if (pcm) {
      current.wantPacket = false;
      worker.postMessage({ type: "mic-pcm", id: current.id, pcm }, [pcm.buffer]);
    } else if (current.ending && current.writes === 0) {
      current.wantPacket = false;
      worker.postMessage({ type: "mic-end", id: current.id, reason: current.ending });
    }
  } catch (error) { storageFailed(current, error); }
  finally {
    current.pulling = false;
    // A storage commit may have arrived while shift() was waiting.
    if (microphone === current && !current.cancelled && current.wantPacket &&
        (current.spool.frames > 0 || (current.ending && current.writes === 0))) {
      queueMicrotask(() => void pullPacket(current));
    }
  }
}
worker.onmessage = ({ data }) => {
  if (data.id !== undefined) {
    const current = microphone;
    if (!current || current.id !== data.id) return;
    if (data.kind === "mic-ready") { if (!current.cancelled) current.capture.start(); return; }
    if (data.kind === "mic-pull") { current.wantPacket = true; void pullPacket(current); return; }
    if (data.kind === "mic-credit") { if (!current.cancelled) current.capture.processed(data.frames); return; }
    if (data.kind === "error" || data.kind === "done") {
      if (data.kind === "done" && current.capture.timingWarnings) {
        data.message = `${data.message || "录音已结束"} · 采集曾延迟，可能漏字`;
      }
      current.capture.close();
      discardSpool(current);
      microphone = undefined;
      state.microphone = { ...state.microphone, phase: "closed", capturing: false, backlog_seconds: 0, stored_seconds: 0,
        captured_seconds: data.captured_seconds ?? state.microphone?.captured_seconds ?? 0 };
    }
  }
  if (data.kind === "error") state.errors.push(data.message);
  send(data);
};
worker.onerror = event => {
  if (microphone) { microphone.capture.close(); discardSpool(microphone); }
  microphone = undefined;
  state.microphone = { ...state.microphone, phase: "closed", capturing: false, backlog_seconds: 0, stored_seconds: 0 };
  state.errors.push(event.message);
  send({ kind: "fatal", message: event.message || "WebGPU Worker 异常" });
};

async function startMicrophone(command) {
  const id = ++audioRequestId;
  const current = { id, started: false, writes: 0, wantPacket: false, cancelled: false };
  microphone = current;
  state.events = [];
  diagnosticCharacters = 0;
  state.microphone = { phase: "opening" };
  current.capture = new Microphone({
    onPacket(pcm) {
      if (microphone !== current || current.cancelled) return;
      current.writes++;
      void current.spool.push(pcm).then(stored => {
        if (stored && microphone === current && !current.cancelled) current.capture.stored(pcm.length);
      }).catch(error => storageFailed(current, error)).finally(() => {
        current.writes--;
        void pullPacket(current);
      });
    },
    onMeter(update) {
      if (microphone !== current || current.cancelled) return;
      state.microphone = { ...state.microphone, ...update, phase: update.capturing ? "recording" : "draining",
        rate: current.capture.context.sampleRate, mode: current.capture.mode, mode_reason: current.capture.modeReason,
        stored_seconds: (current.spool?.frames || 0) / current.capture.context.sampleRate };
      send(update);
    },
    onEnd(reason, captured) {
      if (microphone !== current || current.cancelled) return;
      if (reason === "stopped") { interruptMicrophone(current); return; }
      current.ending = reason;
      state.microphone = { ...state.microphone, phase: "draining", capturing: false, reason };
      if (!current.started) {
        microphone = undefined;
        state.microphone.phase = "closed";
        discardSpool(current);
        send({ kind: "done", text: "", delta: "", message: endMessage(reason) });
        return;
      }
      send({ kind: "capture", capturing: false, message: "输入已结束 · 正在处理尾音，仍可点击停止打断",
        captured_seconds: captured / (current.capture.context?.sampleRate || 16000),
        backlog_seconds: current.capture.pending / (current.capture.context?.sampleRate || 16000) });
      void pullPacket(current);
    },
  });
  try {
    const fixture = new URLSearchParams(location.search).get("mic-test");
    if (fixture) send({ kind: "notice", message: "麦克风测试模式 · 使用公开文件音源，不打开输入设备" });
    const opening = current.capture.open(fixture ? new URL(fixture, location.href) : undefined);
    opening.catch(() => {});
    current.spool = await AudioSpool.open();
    if (microphone !== current || current.cancelled) { current.capture.close(); discardSpool(current); return; }
    if (!await opening) { discardSpool(current); return; }
    if (microphone !== current || current.cancelled) { current.capture.close(); discardSpool(current); return; }
    current.started = true;
    state.microphone = { phase: "waiting-for-engine", rate: current.capture.context.sampleRate };
    worker.postMessage({ type: "mic-start", id, rate: current.capture.context.sampleRate,
      context: command.context, config: command.config, language: command.language });
  } catch (error) {
    current.capture.close();
    discardSpool(current);
    if (microphone !== current) return;
    microphone = undefined;
    state.microphone.phase = "closed";
    state.errors.push(String(error));
    send({ kind: "error", message: String(error) });
  }
}

document.querySelector("#audio-file").onchange = ({ target }) => {
  selectedFile = target.files[0];
  // Selecting the same file again must fire change after a path/error edit.
  target.value = "";
  if (selectedFile) send({ kind: "file", text: `已选择：${selectedFile.name}` });
};

class SpeechUI extends WasmWebGL {
  create_js_message_bridge(cx) {
    this.exports.r2t2_register_bridge(cx);
    super.create_js_message_bridge(cx);
  }

  FromWasmCommand({ json }) {
    // Commands are deferred until Makepad has returned from its Wasm dispatch.
    queueMicrotask(async () => {
      let requestId;
      try {
        const command = JSON.parse(json);
        if (command.type === "choose") {
          document.querySelector("#audio-file").click();
        } else if (command.type === "copy") {
          try {
            await navigator.clipboard.writeText(command.text);
            state.clipboardWritten = command.text;
            send({ kind: "notice", message: "已复制全文" });
          } catch (error) {
            send({ kind: "notice", message: `复制失败，请选中文本后使用复制快捷键：${error}` });
          }
        } else if (command.type === "load") {
          worker.postMessage({ type: "load", url: new URL(command.url, location.href).href });
        } else if (command.type === "stop") {
          if (microphone) { interruptMicrophone(microphone); return; }
          ++audioRequestId;
          if (pendingAudio) {
            pendingAudio = false;
            send({ kind: "done", text: "", delta: "", cancelled: true });
          }
          worker.postMessage({ type: "stop" });
        } else if (command.type === "microphone") {
          await startMicrophone(command);
        } else if (command.type === "transcribe") {
          requestId = ++audioRequestId;
          pendingAudio = true;
          state.events = [];
          diagnosticCharacters = 0;
          let bytes;
          if (selectedFile && command.audio === `已选择：${selectedFile.name}`) {
            if (selectedFile.size > 256 * 1024 * 1024) throw new Error("原型支持最大 256 MiB 的音频文件");
            bytes = await selectedFile.arrayBuffer();
          } else {
            const response = await fetch(new URL(command.audio, location.href));
            if (!response.ok) throw new Error(`音频请求失败：HTTP ${response.status}`);
            if (Number(response.headers.get("content-length")) > 256 * 1024 * 1024) throw new Error("音频超过 256 MiB");
            bytes = await response.arrayBuffer();
          }
          if (requestId !== audioRequestId) return;
          const decoder = new OfflineAudioContext(1, 1, 16000);
          const audio = await decoder.decodeAudioData(bytes);
          if (requestId !== audioRequestId) return;
          if (audio.duration > 600) throw new Error("原型支持最长 10 分钟音频");
          const pcm = new Float32Array(audio.length);
          for (let channel = 0; channel < audio.numberOfChannels; channel++) {
            const data = audio.getChannelData(channel);
            for (let i = 0; i < pcm.length; i++) pcm[i] += data[i] / audio.numberOfChannels;
          }
          pendingAudio = false;
          worker.postMessage({ type: "transcribe", pcm, context: command.context, config: command.config, language: command.language }, [pcm.buffer]);
        }
      } catch (error) {
        if (requestId !== undefined && requestId !== audioRequestId) return;
        pendingAudio = false;
        state.errors.push(String(error));
        send({ kind: "error", message: String(error) });
      }
    });
  }
}

try {
  const wasm = await SpeechUI.fetch_and_instantiate_wasm("./r2t2-makepad.wasm");
  ui = new SpeechUI(wasm, {}, document.querySelector("#app"));
  window.makepadPrototype = ui;
  document.querySelector("#boot").remove();
  // Only diagnostic counters: actual responsiveness is checked by clicking the
  // Rust button and observing its rendered counter during model inference.
  function frame() { state.frames++; requestAnimationFrame(frame); }
  requestAnimationFrame(frame);
} catch (error) {
  state.errors.push(String(error));
  document.querySelector("#boot").textContent = `界面启动失败：${error}`;
}
addEventListener("pagehide", () => {
  if (microphone) { microphone.capture.close(); discardSpool(microphone); }
  worker.terminate();
}, { once: true });
