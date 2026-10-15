#!/usr/bin/env python3
"""Compare the Rust frontend with an independent torch.stft / Slaney reference.

Requires only PyTorch, not numpy, transformers, or model weights.
"""
import argparse
from array import array
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile


def reference(samples):
    import torch
    # Match the Rust API's documented final-tail padding to a 10 ms boundary.
    samples = samples + [0.0] * ((-len(samples)) % 160)
    waveform = torch.tensor(samples, dtype=torch.float32)
    power = torch.stft(waveform, 400, 160, window=torch.hann_window(400), return_complex=True)[..., :-1].abs().square()
    fft_hz = torch.linspace(0, 8000, 201, dtype=torch.float64)
    max_mel = 15.0 + math.log(8.0) / (math.log(6.4) / 27.0)
    mel = torch.linspace(0, max_mel, 130, dtype=torch.float64)
    hz = torch.where(mel < 15, mel * (200.0/3.0), 1000 * torch.exp((mel-15) * math.log(6.4)/27.0))
    lower = (fft_hz[:, None] - hz[:-2]) / (hz[1:-1] - hz[:-2])
    upper = (hz[2:] - fft_hz[:, None]) / (hz[2:] - hz[1:-1])
    bank = torch.minimum(lower, upper).clamp(min=0) * (2 / (hz[2:] - hz[:-2]))
    log_mel = (bank.float().T @ power).clamp(min=1e-10).log10()
    return ((torch.maximum(log_mel, log_mel.max()-8)+4)/4).reshape(-1).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    rng = random.Random(20261008)
    cases = {
        "silence": [0.0] * 1280,
        "impulses": [1.0 if i % 743 == 0 else 0.0 for i in range(3200)],
        "tones": [0.3*math.sin(2*math.pi*440*i/16000) + 0.1*math.sin(2*math.pi*2300*i/16000) for i in range(16000)],
        "noise": [rng.uniform(-0.5, 0.5) for _ in range(17760)],
        "sub_hop_tail": [rng.uniform(-0.2, 0.2) for _ in range(16173)],
    }
    with tempfile.TemporaryDirectory(prefix="r2t2-frontend-") as temp:
        for name, samples in cases.items():
            encoded = array("f", samples)
            if sys.byteorder != "little": encoded.byteswap()
            source = Path(temp) / "input.f32le"
            output = Path(temp) / "output.f32le"
            source.write_bytes(encoded.tobytes())
            subprocess.run([str(args.binary.resolve()), str(source), str(output)], check=True, stdout=subprocess.DEVNULL)
            actual = array("f")
            actual.frombytes(output.read_bytes())
            if sys.byteorder != "little": actual.byteswap()
            # Quantize the input to float32 before comparing the two implementations.
            if sys.byteorder != "little": encoded.byteswap()
            expected = reference(encoded.tolist())
            assert len(actual) == len(expected)
            max_error = max(abs(a-b) for a,b in zip(actual, expected))
            rms = math.sqrt(sum((a-b)**2 for a,b in zip(actual, expected))/len(actual))
            print(f"{name}: max_abs={max_error:.8g} rms={rms:.8g}")
            assert max_error < 2e-5, f"frontend mismatch: {name}"


if __name__ == "__main__":
    main()
