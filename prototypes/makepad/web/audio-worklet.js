import { CapturePackets } from "./capture-packets.js";

class SpeechCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.capture = new CapturePackets(sampleRate, data => {
      this.port.postMessage(data, data.pcm ? [data.pcm.buffer] : []);
    });
    this.port.onmessage = ({ data }) => {
      if (data.type === "credit") this.capture.returnCredit(data.frames);
      if (data.type === "stop") this.capture.finish(data.reason || "stopped");
    };
  }
  process(inputs) {
    // Outputs remain zero; microphone audio is never played to the speakers.
    return this.capture.process(inputs[0]);
  }
}
registerProcessor("speech-capture", SpeechCapture);
