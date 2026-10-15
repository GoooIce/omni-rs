// Temporary, origin-local PCM storage. Inference pulls one packet at a time;
// capture credits are returned after storage commits, independently of the GPU.
export class AudioSpool {
  static async open() {
    const spool = new AudioSpool();
    spool.name = `r2t2-live-audio-${crypto.randomUUID()}`;
    spool.db = await new Promise((resolve, reject) => {
      const request = indexedDB.open(spool.name, 1);
      request.onupgradeneeded = () => request.result.createObjectStore("pcm");
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    return spool;
  }
  constructor() {
    this.write = 0;
    this.committed = 0;
    this.read = 0;
    this.frames = 0;
    this.transactions = new Set();
    this.closed = false;
  }
  transaction(action) {
    return new Promise((resolve, reject) => {
      if (this.closed) { resolve(undefined); return; }
      const tx = this.db.transaction("pcm", "readwrite", { durability: "relaxed" });
      this.transactions.add(tx);
      let result;
      tx.oncomplete = () => { this.transactions.delete(tx); resolve(result); };
      tx.onabort = tx.onerror = () => {
        this.transactions.delete(tx);
        if (this.closed) resolve(undefined);
        else reject(tx.error || new Error("录音临时存储事务失败"));
      };
      action(tx.objectStore("pcm"), value => { result = value; });
    });
  }
  async push(pcm) {
    if (this.closed) return false;
    const key = this.write++;
    await this.transaction(store => store.put(pcm, key));
    if (this.closed) return false;
    this.committed++;
    this.frames += pcm.length;
    return true;
  }
  async shift() {
    if (this.closed || this.read === this.committed) return undefined;
    const key = this.read;
    const pcm = await this.transaction((store, result) => {
      const request = store.get(key);
      request.onsuccess = () => { result(request.result); store.delete(key); };
    });
    if (this.closed) return undefined;
    if (!(pcm instanceof Float32Array)) throw new Error("录音临时缓存不完整");
    this.read++;
    this.frames -= pcm.length;
    return pcm;
  }
  close() {
    if (this.closed) return this.deletion;
    this.closed = true;
    for (const tx of this.transactions) {
      try { tx.abort(); } catch { /* already completed */ }
    }
    this.db.close();
    this.deletion = new Promise((resolve, reject) => {
      const request = indexedDB.deleteDatabase(this.name);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
    return this.deletion;
  }
}
