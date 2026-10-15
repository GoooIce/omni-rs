import { loadModel } from "./load-model.js";

let model;
let busy = false;
self.onmessage = async ({ data }) => {
  if (busy) return postMessage({ type: "error", message: "请等待当前操作完成。" });
  busy = true;
  let session;
  try {
    if (data.type === "load") {
      model?.free();
      model = undefined;
      model = await loadModel(data.url, {
        precision: data.precision,
        onProgress: progress => postMessage({ type: "progress", progress }),
      });
      postMessage({ type: "ready" });
    } else if (data.type === "transcribe") {
      if (!model) throw new Error("请先加载模型。");
      session = model.createSession(JSON.stringify({ language: data.language || null }));
      const pcm = data.pcm;
      for (let offset = 0; offset < pcm.length; offset += 320) {
        for (const event of await session.push(pcm.subarray(offset, offset + 320))) {
          postMessage({ type: "event", ...event });
        }
      }
      for (const event of await session.finishEvents()) {
        postMessage({ type: event.is_final ? "done" : "event", ...event });
      }
    }
  } catch (error) {
    postMessage({ type: "error", message: String(error) });
  } finally {
    session?.free();
    busy = false;
  }
};
