import { CapturePackets } from "./capture-packets.js";

export const endMessage = reason => ({
  stopped: "已打断 · 保留已确认文字，未处理音频已取消",
  "storage-stalled": "临时存储写入未能跟上录音，已停录；请检查设备剩余空间及页面负载",
  "storage-error": "无法写入录音临时缓存，已停录；请检查浏览器存储权限及剩余空间",
  "device-ended": "输入设备中断，已停录并保留已接收音频的转写",
  "context-suspended": "音频设备暂停，已停录；请保持页面前台后重试",
  "worklet-timeout": "音频线程没有响应，已停录；已收到的音频处理完成，最后不足 80 ms 可能未送达",
  "invalid-audio": "输入音频无效，已停录并保留此前的转写",
  "worklet-error": "音频采集线程异常，已停录并保留已收到的音频",
}[reason] || "录音已结束 · 尾音处理完成");

export class Microphone {
  constructor({ onPacket, onEnd, onMeter }) {
    Object.assign(this, { onPacket, onEnd, onMeter });
    this.ending = false;
    this.ended = false;
    this.live = false;
    this.captured = 0;
    this.pending = 0;
    this.lastMeter = 0;
    this.timingWarnings = 0;
    this.name = "默认输入设备";
  }
  async open(testUrl) {
    if (!globalThis.isSecureContext) throw new Error("麦克风需要 HTTPS 或 localhost");
    if (!testUrl && !navigator.mediaDevices?.getUserMedia) throw new Error("浏览器没有麦克风接口");
    try {
      this.context = new AudioContext({ latencyHint: "interactive", sampleRate: 16000 });
    } catch (error) {
      if (error.name !== "NotSupportedError") throw error;
      this.context = new AudioContext({ latencyHint: "interactive" });
    }
    // Called from the button dispatch, before awaiting a permission prompt.
    const resumed = this.context.resume();
    resumed.catch(() => {});
    try {
      if (testUrl) {
        this.name = "测试音源（非麦克风）";
        const response = await fetch(testUrl);
        if (!response.ok) throw new Error(`测试音源：HTTP ${response.status}`);
        const bytes = await response.arrayBuffer();
        if (this.ending) return false;
        const buffer = await this.context.decodeAudioData(bytes);
        if (this.ending) return false;
        this.source = this.context.createBufferSource();
        this.source.buffer = buffer;
        this.source.onended = () => this.stop("fixture-ended");
      } else {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: {
          channelCount: { ideal: 1 }, echoCancellation: false,
          noiseSuppression: false, autoGainControl: false,
        }});
        // Permission can resolve after Stop, navigation, or a newer recording.
        if (this.ending) { stream.getTracks().forEach(track => track.stop()); return false; }
        this.stream = stream;
        this.name = stream.getAudioTracks()[0]?.label || this.name;
        for (const track of stream.getAudioTracks()) {
          track.onended = () => this.stop("device-ended");
          track.onmute = () => this.stop("device-ended");
        }
        this.source = this.context.createMediaStreamSource(stream);
      }
      await this.deadline(resumed, "浏览器音频未能启动，请点击停止后重试");
      if (this.ending) return false;
      try {
        if (!globalThis.AudioWorkletNode) throw new Error("AudioWorklet unavailable");
        await this.deadline(this.context.audioWorklet.addModule(new URL("./audio-worklet.js", import.meta.url)),
          "AudioWorklet 加载超时", 5000);
        if (this.ending) return false;
        this.node = new AudioWorkletNode(this.context, "speech-capture", {
          numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], channelCountMode: "max",
        });
        this.mode = "worklet";
        this.node.port.onmessage = ({ data }) => this.captureMessage(data);
        this.node.onprocessorerror = () => this.finish("worklet-error");
      } catch (error) {
        if (this.ending) return false;
        if (!this.context.createScriptProcessor) throw error;
        // Deprecated, isolated compatibility adapter for browsers with unavailable
        // or stalled Worklets. Inference still runs entirely in the GPU Worker.
        this.mode = "compatibility";
        this.modeReason = String(error);
        this.packetizer = new CapturePackets(this.context.sampleRate, data => this.captureMessage(data));
        this.node = this.context.createScriptProcessor(2048, 2, 1);
        this.node.onaudioprocess = event => {
          if (!this.live || this.ended) return;
          const duration = event.inputBuffer.length / this.context.sampleRate;
          if (!this.ending && this.nextAudioTime !== undefined && Math.abs(event.playbackTime - this.nextAudioTime) > duration / 2) {
            // ScriptProcessor timestamps can jump when the main thread stalls.
            // Keep accepting available samples; missing hardware samples cannot
            // be recovered. Report this explicitly without ending the recording.
            this.timingWarnings++;
            this.meter(0);
          }
          this.nextAudioTime = event.playbackTime + duration;
          const channels = Array.from({ length: event.inputBuffer.numberOfChannels }, (_, i) => event.inputBuffer.getChannelData(i));
          this.packetizer.process(channels);
        };
      }
      return true;
    } catch (error) {
      this.close();
      if (error.name === "NotAllowedError") throw new Error("麦克风权限未获允许，请在浏览器和系统设置中允许录音后重试");
      if (error.name === "NotFoundError") throw new Error("没有可用的麦克风，请连接输入设备后重试");
      if (error.name === "NotReadableError") throw new Error("无法读取麦克风，请检查设备是否被占用及系统录音权限");
      throw error;
    }
  }
  async deadline(promise, message, milliseconds = 10000) {
    let timer;
    try {
      return await Promise.race([promise, new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(message)), milliseconds);
      })]);
    } finally { clearTimeout(timer); }
  }
  captureMessage(data) {
    if (this.ended) return;
    if (data.type === "pcm") {
      this.captured = data.captured;
      this.pending += data.pcm.length;
      this.onPacket(data.pcm);
      if (performance.now() - this.lastMeter >= 100) this.meter(data.peak);
    } else if (data.type === "end") {
      this.captured = data.captured;
      this.finish(data.reason);
    }
  }
  start() {
    if (this.ending || !this.node) return;
    this.live = true;
    this.context.onstatechange = () => {
      if (this.live && !this.ending && this.context.state !== "running") this.stop("context-suspended");
    };
    this.source.connect(this.node);
    this.node.connect(this.context.destination); // output is always zero
    if (this.source instanceof AudioBufferSourceNode) this.source.start();
    this.meter(0);
  }
  processed(frames) {
    this.pending = Math.max(0, this.pending - frames);
    if (performance.now() - this.lastMeter >= 100) this.meter(0);
  }
  stored(frames) {
    if (!this.ending) {
      if (this.packetizer) this.packetizer.returnCredit(frames);
      else this.node?.port.postMessage({ type: "credit", frames });
    }
  }
  meter(peak) {
    this.lastMeter = performance.now();
    this.onMeter({ kind: "capture", capturing: this.live && !this.ending,
      message: this.ending ? "麦克风已释放 · 正在处理尾音…" : `录音中 · ${this.name}${this.mode === "compatibility" ? " · 兼容采集" : ""}${this.timingWarnings ? " · 采集曾延迟，可能漏字" : ""}`,
      captured_seconds: this.captured / this.context.sampleRate,
      backlog_seconds: this.pending / this.context.sampleRate, level: peak, timing_warnings: this.timingWarnings });
  }
  stop(reason = "stopped") {
    // Manual stop is an interruption, including while a natural end is draining.
    if (reason === "stopped") { this.finish(reason); return; }
    if (this.ending) return;
    this.ending = true;
    // Release hardware immediately; don't wait for the GPU or final decoding.
    this.stream?.getTracks().forEach(track => track.stop());
    if (this.live && this.node) {
      this.meter(0);
      if (this.packetizer) {
        // Drain already captured input in the two Web Audio buffers after the
        // hardware stops. This may include a short silent suffix.
        this.stopTimer = setTimeout(() => this.packetizer.finish(reason),
          2 * this.node.bufferSize / this.context.sampleRate * 1000 + 40);
      } else {
        this.node.port.postMessage({ type: "stop", reason });
        this.stopTimer = setTimeout(() => this.finish("worklet-timeout"), 1000);
      }
    } else this.finish(reason);
  }
  finish(reason) {
    if (this.ended) return;
    this.close();
    this.onEnd(reason, this.captured);
  }
  close() {
    this.ending = this.ended = true;
    this.live = false;
    clearTimeout(this.stopTimer);
    this.stream?.getTracks().forEach(track => { track.onended = track.onmute = null; track.stop(); });
    if (this.source) { this.source.onended = null; this.source.disconnect(); }
    if (this.node) { this.node.onaudioprocess = null; this.node.disconnect(); this.node.port?.close(); }
    if (this.context) {
      this.context.onstatechange = null;
      if (this.context.state !== "closed") this.context.close().catch(() => {});
    }
  }
}
