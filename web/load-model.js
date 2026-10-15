// Use this module from a module worker. All inference stays on the local device.
import init, { WebModelBuilder } from "../pkg/omni_r2t2.js";

async function checkedFetch(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${url}`);
  return response;
}

export async function loadModel(manifestUrl, { precision = "f16", onProgress = () => {} } = {}) {
  if (!navigator.gpu) throw new Error("WebGPU requires a supported browser and a secure origin.");
  await init();
  const base = new URL(manifestUrl, globalThis.location.href);
  const manifest = await (await checkedFetch(base)).json();
  if (manifest.format !== "omni-r2t2-web-v1") throw new Error("Unsupported model package");
  const config = await (await checkedFetch(new URL(manifest.config, base))).text();
  const tokenizer = new Uint8Array(await (await checkedFetch(new URL(manifest.tokenizer, base))).arrayBuffer());
  const builder = await WebModelBuilder.create(config, tokenizer, precision);
  try {
    for (let i = 0; i < manifest.chunks.length; i++) {
      const chunk = manifest.chunks[i];
      const bytes = new Uint8Array(await (await checkedFetch(new URL(chunk.file, base))).arrayBuffer());
      if (bytes.byteLength !== chunk.bytes) throw new Error(`Truncated parameter: ${chunk.name}`);
      const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
        .map(v => v.toString(16).padStart(2, "0")).join("");
      if (hash !== chunk.sha256) throw new Error(`Checksum mismatch: ${chunk.name}`);
      builder.addTensorChunk(chunk.name, new Uint32Array(chunk.shape), chunk.dtype, chunk.offset, bytes);
      onProgress((i + 1) / manifest.chunks.length);
    }
  } catch (error) {
    builder.free();
    throw error;
  }
  return builder.finish();
}

// const model = await loadModel("/models/web/manifest.json");
// const session = model.createSession(JSON.stringify({ language: "Chinese", chunk_ms: 160 }));
// for await (const pcm16k of source) {
//   for (const event of await session.push(pcm16k)) postMessage(event);
// }
// for (const event of await session.finishEvents()) postMessage(event);
// session.free();
