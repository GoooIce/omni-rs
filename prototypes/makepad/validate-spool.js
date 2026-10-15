// Run in the served browser origin; uses real IndexedDB, no device permissions.
import { AudioSpool } from './web/audio-spool.js';
import { CapturePackets } from './web/capture-packets.js';
export async function validateSpool() {
  const assert = (value, message) => { if (!value) throw new Error(message); };
  const checks = [];
  const spool = await AudioSpool.open();
  const name = spool.name;
  let writes = [];
  const packets = new CapturePackets(16000, data => {
    assert(data.type === 'pcm', `unexpected automatic stop: ${data.reason}`);
    writes.push(spool.push(data.pcm).then(() => packets.returnCredit(data.pcm.length)));
  });
  try {
    // One minute of audio stored while the inference consumer is entirely stalled.
    for (let i = 0; i < 750; i++) {
      packets.process([new Float32Array(1280).fill(i)]);
      await Promise.all(writes); writes = [];
    }
    assert(packets.active && spool.frames === 16000 * 60, 'capture stopped or audio lost');
    checks.push('60 seconds of PCM persists with zero inference consumption and capture stays active');
    for (let i = 0; i < 750; i++) {
      const pcm = await spool.shift();
      assert(pcm.length === 1280 && pcm.every(value => value === i), `packet ${i} reordered or corrupted`);
    }
    assert(spool.frames === 0 && await spool.shift() === undefined, 'queue did not drain');
    const count = await new Promise((resolve, reject) => {
      const tx = spool.db.transaction('pcm', 'readonly');
      const request = tx.objectStore('pcm').count();
      request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
    });
    assert(count === 0, 'consumed PCM still retained');
    checks.push('every sample preserved in order and consumed records deleted');
    // Stop races with pending writes; no packet may reopen or survive the spool.
    const pending = Array.from({ length: 25 }, () => spool.push(new Float32Array(1280)));
    await spool.close();
    await Promise.all(pending);
    assert(await spool.push(new Float32Array(1)) === false, 'closed spool accepted audio');
    assert(!(await indexedDB.databases()).some(db => db.name === name), 'temporary database retained after Stop');
    checks.push('Stop aborts pending writes and deletes the complete temporary database');
  } finally { await spool.close(); }
  return { passed: checks.length, checks };
}
