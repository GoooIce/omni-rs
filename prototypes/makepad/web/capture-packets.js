// Shared packetization for AudioWorklet and the compatibility input adapter.
export class CapturePackets {
  constructor(sampleRate, emit) {
    this.sampleRate = sampleRate;
    this.emit = emit;
    this.active = true;
    // Only bounds packets awaiting local storage, never the inference backlog.
    this.capacity = Math.round(sampleRate * 8);
    this.credit = this.capacity;
    this.packet = new Float32Array(Math.max(1, Math.round(this.sampleRate * 0.08)));
    this.used = 0;
    this.captured = 0;
    this.peak = 0;
  }
  returnCredit(frames) {
    this.credit = Math.min(this.capacity, this.credit + frames);
  }

  flush() {
    if (!this.used) return;
    const pcm = this.used === this.packet.length ? this.packet : this.packet.slice(0, this.used);
    this.emit({ type: "pcm", pcm, captured: this.captured, peak: this.peak });
    this.packet = new Float32Array(Math.max(1, Math.round(this.sampleRate * 0.08)));
    this.used = 0;
    this.peak = 0;
  }
  finish(reason) {
    if (!this.active) return;
    this.active = false;
    this.flush();
    this.emit({ type: "end", reason, captured: this.captured });
  }
  process(channels) {
    if (!this.active) return false;
    if (!channels?.length) return true;
    // Render quantum length can change; don't assume 128 frames.
    for (let i = 0; i < channels[0].length; i++) {
      if (!this.credit) { this.finish("storage-stalled"); return false; }
      let sample = 0;
      for (const channel of channels) sample += channel[i] / channels.length;
      if (!Number.isFinite(sample)) { this.finish("invalid-audio"); return false; }
      this.packet[this.used++] = sample;
      this.peak = Math.max(this.peak, Math.abs(sample));
      this.credit--;
      this.captured++;
      if (this.used === this.packet.length) this.flush();
    }
    // Outputs remain zero: never play the microphone through the speakers.
    return true;
  }
}
