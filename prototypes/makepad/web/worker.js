import { loadModel } from "../../../web/load-model.js";
import { endMessage } from "./microphone.js";

let model, active;
let busy = false;

function sessionState(data) {
  if (!model) throw new Error("请先载入模型");
  const session = model.createTranscription(JSON.stringify({ language: "Chinese", context: data.context || "",
    chunk_ms: 960, lookahead_ms: 0, max_new_tokens: 8, ...data.config }), data.rate ?? 16000);
  return { id: data.id, session, stopHandle: session.stopHandle(),
    microphone: data.type === "mic-start", cancelled: false, processing: false };
}
function report(state, event, kind, extra = {}) {
  postMessage({ ...event, kind, id: state.id, ...extra });
}
async function finish(state, extra = {}) {
  for (const event of await state.session.finishEvents()) {
    report(state, event, event.is_final ? "done" : "event",
      event.is_final ? { ...extra, ...(event.cancelled ? interruptionDetails(state) : {}) } : {});
  }
}
async function feed(state, pcm) {
  for (const event of await state.session.push(pcm)) report(state, event, "event");
}
function release(state) {
  state.stopHandle.free();
  state.session.free();
  if (active === state) { active = undefined; busy = false; }
}
function interruptionDetails(state) {
  return { cancelled: true, message: endMessage(state.reason || "stopped"),
    captured_seconds: state.capturedSeconds ?? (state.rate ? state.receivedFrames / state.rate : 0) };
}
function interrupted(state) {
  report(state, state.session.interrupt(), "done", interruptionDetails(state));
}
function cancel(state, data) {
  state.cancelled = true;
  state.reason = data.reason || "stopped";
  state.capturedSeconds = data.captured_seconds;
  // Independent handle: WebSession's mutable borrow is held by awaited push/finish.
  state.stopHandle.stop();
  if (state.microphone && !state.processing) { interrupted(state); release(state); }
}
async function microphonePacket(state, data) {
  if (state.processing || state.cancelled) return;
  state.processing = true;
  try {
    if (data.type === "mic-end") {
      await finish(state, {
        message: endMessage(data.reason), captured_seconds: state.receivedFrames / state.rate,
      });
      release(state);
    } else {
      if (!(data.pcm instanceof Float32Array)) throw new Error("无效的录音数据包");
      state.receivedFrames += data.pcm.length;
      await feed(state, data.pcm);
      if (state.cancelled) { interrupted(state); release(state); }
      else {
        postMessage({ kind: "mic-credit", id: state.id, frames: data.pcm.length });
        postMessage({ kind: "mic-pull", id: state.id });
      }
    }
  } catch (error) {
    if (state.cancelled && error.name === "Interrupted") interrupted(state);
    else postMessage({ kind: "error", id: state.id, message: String(error) });
    release(state);
  } finally { state.processing = false; }
}
self.onmessage = ({ data }) => {
  if (data.type === "stop" || data.type === "mic-cancel") {
    if (active && (data.id === undefined || data.id === active.id)) cancel(active, data);
    return;
  }
  if (data.type === "mic-pcm" || data.type === "mic-end") {
    if (active?.microphone && data.id === active.id) void microphonePacket(active, data);
    return;
  }
  if (busy) { postMessage({ kind: "error", id: data.id, message: "请等待当前操作完成" }); return; }
  if (data.type === "mic-start") {
    busy = true;
    let state;
    try {
      state = sessionState(data);
      state.rate = data.rate;
      state.receivedFrames = 0;
      active = state;
      postMessage({ kind: "mic-ready", id: data.id });
      postMessage({ kind: "mic-pull", id: data.id });
    } catch (error) {
      if (state) release(state);
      busy = false;
      postMessage({ kind: "error", id: data.id, message: String(error) });
    }
    return;
  }
  void fileCommand(data);
};
async function fileCommand(data) {
  busy = true;
  let state;
  try {
    if (data.type === "load") {
      model?.free(); model = undefined;
      const started = performance.now();
      model = await loadModel(data.url, { precision: "f16", onProgress: progress => {
        postMessage({ kind: "progress", progress, message: `正在载入模型 · ${(progress * 100).toFixed(1)}%` });
      }});
      postMessage({ kind: "ready", message: `模型已就绪 · 载入 ${((performance.now() - started) / 1000).toFixed(1)} s` });
    } else if (data.type === "transcribe") {
      active = state = sessionState(data);
      for (let offset = 0; offset < data.pcm.length && !state.cancelled; offset += 1280) {
        await feed(state, data.pcm.subarray(offset, offset + 1280));
      }
      if (state.cancelled) interrupted(state);
      else await finish(state);
    }
  } catch (error) {
    if (state?.cancelled && error.name === "Interrupted") interrupted(state);
    else postMessage({ kind: "error", message: String(error) });
  } finally { if (state) release(state); busy = false; }
}
