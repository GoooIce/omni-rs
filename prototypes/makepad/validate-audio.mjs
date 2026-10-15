// Run with Node >= 20 from the repository root; no browser or microphone access.
import fs from 'node:fs/promises';
import vm from 'node:vm';
import assert from 'node:assert/strict';

const root = process.cwd() + '/prototypes/makepad/web/';
const packetCode = (await fs.readFile(root + 'capture-packets.js', 'utf8')).replace(/^export /gm, '');
const workletCode = packetCode + (await fs.readFile(root + 'audio-worklet.js', 'utf8')).replace(/^import .*;\n/gm, '');
const checks = [];
function worklet(rate = 48000) {
  let Processor;
  const messages = [];
  class Base {
    constructor() {
      this.port = { postMessage(data, transfer) { messages.push(structuredClone(data, { transfer })); } };
    }
  }
  vm.runInNewContext(workletCode, { sampleRate: rate, AudioWorkletProcessor: Base, Float32Array,
    registerProcessor(_name, type) { Processor = type; } });
  return { processor: new Processor(), messages };
}
{
  const { processor: p, messages } = worklet();
  p.process([[new Float32Array(4007).fill(0.25), new Float32Array(4007).fill(0.75)]]);
  p.port.onmessage({ data: { type: 'stop' } });
  p.port.onmessage({ data: { type: 'stop' } });
  assert.deepEqual(messages.filter(m => m.pcm).map(m => m.pcm.length), [3840, 167]);
  assert(messages.filter(m => m.pcm).every(m => m.pcm.every(x => x === 0.5)));
  assert.equal(messages.filter(m => m.type === 'end').length, 1);
  assert.equal(messages.at(-1).captured, 4007);
  checks.push('variable render quanta, stereo downmix, exact partial flush, idempotent stop');
}
{
  const { processor: p, messages } = worklet();
  p.process([[new Float32Array(48000 * 8 + 128)]]);
  assert.equal(messages.at(-1).reason, 'storage-stalled');
  assert.equal(messages.at(-1).captured, 48000 * 8);
  assert.equal(messages.filter(m => m.pcm).reduce((n, m) => n + m.pcm.length, 0), 48000 * 8);
  assert.equal(p.process([[new Float32Array(128)]]), false);
  checks.push('failed local storage is bounded; inference does not hold capture credits');
}
{
  const { processor: p, messages } = worklet();
  let cursor = 0;
  for (let n = 0; n < 500; n++) {
    p.process([[new Float32Array(960)]]);
    for (const m of messages.slice(cursor)) if (m.pcm) p.port.onmessage({ data: { type: 'credit', frames: m.pcm.length } });
    cursor = messages.length;
  }
  assert.equal(p.capture.captured, 480000);
  assert.equal(messages.filter(m => m.type === 'end').length, 0);
  p.port.onmessage({ data: { type: 'stop' } });
  checks.push('returned storage credits allow continuous capture beyond the transport capacity');
}
{
  const { processor: p, messages } = worklet();
  p.process([[new Float32Array([0.25, 0.5, NaN, 0.75])]]);
  assert.deepEqual([...messages[0].pcm], [0.25, 0.5]);
  assert.equal(messages.at(-1).reason, 'invalid-audio');
  checks.push('invalid PCM terminates at the valid prefix');
}

const microphoneCode = (await fs.readFile(root + 'microphone.js', 'utf8'))
  .replace(/^import .*;\n/gm, '').replace(/^export /gm, '').replaceAll('import.meta.url', '"http://localhost/microphone.js"');
function microphoneHarness(getUserMedia, brokenWorklet = false) {
  const contexts = [], nodes = [];
  class Context {
    constructor() { this.state = 'running'; this.sampleRate = 48000; this.closes = 0; this.audioWorklet = { addModule: async () => { if (brokenWorklet) throw new Error('worklet unavailable'); } }; contexts.push(this); }
    resume() { return Promise.resolve(); }
    close() { this.state = 'closed'; this.closes++; return Promise.resolve(); }
    createScriptProcessor(size) { const node = new Node(); node.bufferSize = size; return node; }
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
  }
  class Node {
    constructor() { this.port = { postMessage() {}, close() {} }; nodes.push(this); }
    connect() {} disconnect() {}
  }
  const context = { isSecureContext: true, AudioContext: Context, AudioWorkletNode: Node, AudioBufferSourceNode: class {},
    URL, performance, setTimeout, clearTimeout, navigator: { mediaDevices: { getUserMedia } } };
  vm.runInNewContext(packetCode + microphoneCode + '\nglobalThis.MicrophoneForTest = Microphone;', context);
  const ended = [], packets = [];
  const microphone = new context.MicrophoneForTest({ onPacket: p => packets.push(p), onMeter() {}, onEnd: r => ended.push(r) });
  return { microphone, contexts, nodes, ended, packets };
}
const track = () => ({ stops: 0, label: 'TEST INPUT', stop() { this.stops++; } });
const stream = t => ({ getTracks: () => [t], getAudioTracks: () => [t] });
{
  let allow;
  const h = microphoneHarness(() => new Promise(resolve => { allow = resolve; }));
  const opened = h.microphone.open();
  h.microphone.stop();
  const t = track(); allow(stream(t));
  assert.equal(await opened, false);
  assert.equal(t.stops, 1);
  assert.equal(h.contexts[0].closes, 1);
  assert.deepEqual(h.ended, ['stopped']);
  assert.equal(h.nodes.length, 0);
  checks.push('stop during a pending permission request releases a late-granted stream');
}
{
  const h = microphoneHarness(async () => { throw Object.assign(new Error('denied'), { name: 'NotAllowedError' }); });
  await assert.rejects(h.microphone.open(), /权限未获允许/);
  assert.equal(h.contexts[0].closes, 1);
  assert.equal(h.microphone.ended, true);
  checks.push('permission denial closes the audio context and reports an actionable error');
}
{
  const t = track(); const h = microphoneHarness(async () => stream(t));
  assert.equal(await h.microphone.open(), true);
  h.microphone.start();
  h.contexts[0].state = 'suspended'; h.contexts[0].onstatechange();
  assert(t.stops > 0);
  h.nodes[0].port.onmessage({ data: { type: 'pcm', pcm: new Float32Array([0.5]), captured: 1, peak: 0.5 } });
  h.nodes[0].port.onmessage({ data: { type: 'end', captured: 1, reason: 'context-suspended' } });
  h.microphone.stop();
  assert.equal(h.packets.length, 1);
  assert.deepEqual(h.ended, ['context-suspended']);
  assert.equal(h.contexts[0].closes, 1);
  checks.push('suspension releases tracks immediately and still accepts the final worklet packet');
}

{
  const t = track(); const h = microphoneHarness(async () => stream(t), true);
  assert.equal(await h.microphone.open(), true);
  assert.equal(h.microphone.mode, 'compatibility');
  h.microphone.start();
  const packet = time => ({ playbackTime: time, inputBuffer: { length: 2048, numberOfChannels: 2,
    getChannelData: i => new Float32Array(2048).fill(i ? 0.75 : 0.25) } });
  h.nodes[0].onaudioprocess(packet(0));
  h.microphone.stop("fixture-ended");
  assert(t.stops > 0);
  h.nodes[0].onaudioprocess(packet(2048 / 48000));
  await new Promise(r => setTimeout(r, 160));
  assert.deepEqual(h.ended, ['fixture-ended']);
  assert.equal(h.packets.reduce((n, p) => n + p.length, 0), 4096);
  assert(h.packets.every(p => p.every(x => x === 0.5)));
  checks.push('natural end drains buffered input after releasing tracks');
}

{
  const t = track(); const h = microphoneHarness(async () => stream(t), true);
  await h.microphone.open(); h.microphone.start();
  const inputBuffer = { length: 2048, numberOfChannels: 1, getChannelData: () => new Float32Array(2048) };
  h.nodes[0].onaudioprocess({ inputBuffer, playbackTime: 0 });
  h.nodes[0].onaudioprocess({ inputBuffer, playbackTime: 1 });
  assert.deepEqual(h.ended, []);
  assert.equal(h.microphone.timingWarnings, 1);
  assert.equal(h.microphone.packetizer.captured, 4096);
  assert.equal(t.stops, 0);
  h.microphone.stop();
  assert.deepEqual(h.ended, ['stopped']);
  checks.push('compatibility timing gaps are reported while capture continues until manual Stop');
}

{
  const t = track(); const h = microphoneHarness(async () => stream(t), true);
  await h.microphone.open(); h.microphone.start();
  h.nodes[0].onaudioprocess({ playbackTime: 0, inputBuffer: { length: 4096, numberOfChannels: 1,
    getChannelData: () => new Float32Array(4096).fill(0.25) } });
  h.microphone.stop();
  assert.deepEqual(h.ended, ['stopped']);
  assert.equal(h.contexts[0].state, 'closed');
  assert.equal(h.nodes[0].onaudioprocess, null);
  assert.equal(h.packets.reduce((n, p) => n + p.length, 0), 3840);
  assert(t.stops > 0);
  checks.push('manual stop releases capture immediately and discards the partial packet without draining');
}

// Exercise the actual Worker protocol with a borrowed, awaited session.
const workerCode = (await fs.readFile(root + 'worker.js', 'utf8')).replace(/^import .*;\n/gm, '');
const messages = [], sessions = [];
let activePushes = 0, maxPushes = 0;
let nextFailure;
const mockModel = { createTranscription(_config, rate) {
  const s = { rate, failure: nextFailure, samples: 0, finished: false, freed: false, stopRequested: false, finishing: false,
    stopHandle() { return { stop: () => { this.stopRequested = true; }, free() {} }; },
    async push(pcm) {
      assert(!this.freed && !this.finished);
      activePushes++; maxPushes = Math.max(maxPushes, activePushes);
      await new Promise(r => setTimeout(r, 10));
      activePushes--;
      if (this.failure) throw this.failure;
      if (this.stopRequested) { const error = new Error('interrupted'); error.name = 'Interrupted'; throw error; }
      this.samples += pcm.length;
      return [{ text: '', delta: '', audio_seconds: this.samples / this.rate, compute_ms: 2, total_compute_ms: 91, event_count: 7, rtf: 0.4 }];
    },
    async finishEvents() {
      assert.equal(activePushes, 0); this.finishing = true;
      await new Promise(r => setTimeout(r, 10));
      this.finishing = false;
      if (this.stopRequested) return [this.interrupt()];
      this.finished = true;
      return [{ text: '', delta: '', audio_seconds: this.samples / this.rate, compute_ms: 1, is_final: false },
        { text: '', delta: '', audio_seconds: this.samples / this.rate, compute_ms: 0, is_final: true,
          total_compute_ms: 99, event_count: 9, rtf: 0.8 }];
    },
    interrupt() {
      assert.equal(activePushes, 0); assert(!this.finishing);
      this.finished = true;
      return { text: '', delta: '', audio_seconds: this.samples / this.rate, compute_ms: 0, is_final: true, cancelled: true };
    },
    free() { assert.equal(activePushes, 0); assert(!this.finishing && !this.freed); this.freed = true; }
  }; nextFailure = undefined; sessions.push(s); return s;
}, free() {} };
const context = { self: {}, postMessage: m => messages.push(m), performance, Float32Array,
  loadModel: async () => mockModel,
  endMessage: reason => reason };
vm.runInNewContext(workerCode, context);
function send(data) { context.self.onmessage({ data }); }
async function wait(predicate) {
  for (let n = 0; n < 1000; n++) { if (predicate()) return; await new Promise(r => setTimeout(r, 2)); }
  throw new Error('test condition timed out');
}
send({ type: 'load', url: 'test-model' });
await wait(() => messages.some(m => m.kind === 'ready'));
send({ type: 'mic-start', id: 1, rate: 48000 });
for (let n = 0; n < 5; n++) {
  await wait(() => messages.filter(m => m.kind === 'mic-pull' && m.id === 1).length === n + 1);
  send({ type: 'mic-pcm', id: 1, pcm: new Float32Array(n === 4 ? 37 : 320) });
}
await wait(() => messages.filter(m => m.kind === 'mic-pull' && m.id === 1).length === 6);
send({ type: 'mic-end', id: 1, reason: 'fixture-ended' });
await wait(() => messages.some(m => m.kind === 'done' && m.id === 1));
assert.equal(maxPushes, 1);
assert.equal(sessions[0].samples, 1317);
assert(sessions[0].finished && sessions[0].freed);
assert.equal(messages.filter(m => m.kind === 'mic-credit').reduce((n, m) => n + m.frames, 0), 1317);
assert.equal(sessions[0].rate, 48000);
const firstFinal = messages.find(m => m.kind === 'done' && m.id === 1);
assert.equal(firstFinal.total_compute_ms, 99);
assert.equal(firstFinal.event_count, 9);
assert.equal(firstFinal.rtf, 0.8);
assert.equal(messages.filter(m => m.kind === 'event' && m.id === 1).length, 6);
checks.push('Worker pulls one packet at a time; EOF forwards all runner events and statistics before release');
send({ type: 'mic-start', id: 2, rate: 16000 });
send({ type: 'mic-pcm', id: 1, pcm: new Float32Array(99) }); // old recording
send({ type: 'mic-pcm', id: 2, pcm: new Float32Array(13) });
send({ type: 'mic-cancel', id: 2, captured_seconds: 17 });
await wait(() => messages.some(m => m.kind === 'done' && m.id === 2));
assert.equal(sessions[1].samples, 0);
assert(sessions[1].freed);
assert.equal(messages.find(m => m.kind === 'done' && m.id === 2).captured_seconds, 17);
checks.push('Stop interrupts an awaited push via a separate handle; stale PCM cannot enter the next session');
send({ type: 'mic-start', id: 3, rate: 16000 });
send({ type: 'mic-end', id: 3, reason: 'fixture-ended' });
await wait(() => sessions[2].finishing);
send({ type: 'mic-cancel', id: 3 });
await wait(() => messages.some(m => m.kind === 'done' && m.id === 3));
assert(sessions[2].stopRequested && sessions[2].freed);
assert.equal(messages.find(m => m.kind === 'done' && m.id === 3).message, 'stopped');
checks.push('Stop also interrupts final decoding after the audio source ends');
send({ type: 'transcribe', pcm: new Float32Array(3200) });
await wait(() => activePushes > 0);
send({ type: 'stop' });
await wait(() => sessions[3].freed);
assert(sessions[3].stopRequested && sessions[3].samples === 0);
checks.push('file transcription uses the same immediate interruption path');
assert(!messages.some(m => m.kind === 'error'));
nextFailure = new Error('GPU readback failed');
send({ type: 'mic-start', id: 4, rate: 16000 });
send({ type: 'mic-pcm', id: 4, pcm: new Float32Array(320) });
send({ type: 'mic-cancel', id: 4 });
await wait(() => sessions[4].freed);
assert(messages.some(m => m.kind === 'error' && m.id === 4 && m.message.includes('GPU readback failed')));
assert(!messages.some(m => m.kind === 'done' && m.id === 4));
checks.push('a simultaneous stop does not hide an unrelated inference failure');
export const report = { passed: checks.length, checks };
console.log(JSON.stringify(report, null, 2));
