#!/usr/bin/env python3
"""Materialize the immutable Mandarin regression corpus (96 AISHELL-1 test clips).

Dependency: pyarrow. Run from the repository root. The fixed source revision and
SHA256 values prevent upstream changes from silently replacing evaluation data.
"""
import hashlib
import io
import http.client
import json
from pathlib import Path
import urllib.request
import wave

from download_model import file_sha256
from evaluate_streaming import normalize

REVISION = "e73dc7f096ecea5c8e5c722446c8d3b4014fdd18"
REPOSITORY = "AudioLLMs/aishell_1_zh_test"
SHARDS = [
    "2466b1e851147943d4a2c46e1dddc8a70a45b733065c7f6a86e44bd88d5579c6",
    "e3d1a5739e652786b732d593a2b25cd41152473988c4e603a22fcd897ed28ff4",
    "ea031c28ea08fec3e765cbea42b1024bd9757fe4aedf7ed2db9f291c05acdd51",
]
SEED = "omni-r2t2-ane-mandarin-v1"
SHARD_BYTES = [364047728, 354834886, 377041689]


def download_verified(url, path, digest, size):
    if path.exists():
        if file_sha256(path) != digest:
            raise ValueError(f"source hash mismatch: {path}")
        print(f"Already verified: {path}", flush=True)
        return
    temporary = path.with_name(path.name + ".part")
    for attempt in range(8):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset > size:
            raise ValueError(f"oversized partial download: {temporary}")
        if offset == size:
            if file_sha256(temporary) != digest:
                raise ValueError(f"download hash mismatch; partial file retained: {temporary}")
            temporary.replace(path)
            return
        print(f"Downloading {path.name}, attempt {attempt + 1}, resume byte {offset}", flush=True)
        headers = {"User-Agent": "omni-r2t2-eval/1", "Range": f"bytes={offset}-"}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                if response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("server returned a different byte range")
                    mode = "ab"
                elif response.status == 200:
                    mode = "wb"  # Server ignored Range: restart without appending duplicate bytes.
                else:
                    raise ValueError(f"unexpected download status {response.status}")
                with temporary.open(mode) as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
        except (OSError, http.client.IncompleteRead) as error:
            print(f"Interrupted: {error}; partial bytes retained", flush=True)
    # The final read may have completed on the last attempt.
    if temporary.exists() and temporary.stat().st_size == size and file_sha256(temporary) == digest:
        temporary.replace(path)
        return
    raise RuntimeError(f"download incomplete after 8 attempts; retained {temporary}")


def main():
    import pyarrow.parquet as pq
    cache = Path("artifacts/eval-corpus/source")
    audio_dir = Path("artifacts/eval-corpus/mandarin-v1")
    cache.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "id": "mandarin-v1", "source": f"https://huggingface.co/datasets/{REPOSITORY}",
        "original_source": "https://www.openslr.org/33/", "original_license": "Apache-2.0",
        "revision": REVISION, "split": "test", "seed": SEED,
        "selection": "32 lowest SHA256(seed:shard_index:row_index) per source shard, before inference",
        "normalization": "NFKC + lowercase - Unicode punctuation - whitespace; v1",
        "scope": "96 full utterances from clean read Mandarin; not a general noise/dialect accuracy claim",
        "shards": [], "cases": [],
    }
    for shard_index, digest in enumerate(SHARDS):
        filename = f"test-{shard_index:05}-of-00003.parquet"
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/data/{filename}"
        path = cache / filename
        download_verified(url, path, digest, SHARD_BYTES[shard_index])
        source = pq.ParquetFile(path)
        rows = source.metadata.num_rows
        selected = sorted(sorted(range(rows), key=lambda row: hashlib.sha256(
            f"{SEED}:{shard_index}:{row}".encode()).digest())[:32])
        manifest["shards"].append({"file": filename, "sha256": digest, "rows": rows, "selected_rows": selected})
        offset = 0
        for batch in source.iter_batches(batch_size=128, columns=["context", "answer"]):
            for row in selected:
                if not offset <= row < offset + batch.num_rows:
                    continue
                entry = batch.slice(row - offset, 1).to_pylist()[0]
                audio = entry["context"]["bytes"]
                case_id = f"aishell-test-{shard_index}-{row:04}"
                destination = audio_dir / f"{case_id}.wav"
                with wave.open(io.BytesIO(audio)) as wav:
                    if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) != (1, 16000, 2, "NONE"):
                        raise ValueError(f"unexpected source audio format: {case_id}")
                    duration = wav.getnframes() / wav.getframerate()
                destination.write_bytes(audio)
                manifest["cases"].append({"id": case_id, "audio": str(destination),
                    "sha256": file_sha256(destination), "duration_seconds": duration,
                    "reference": entry["answer"], "source_audio_path": entry["context"]["path"],
                    "source_shard": shard_index, "source_row": row})
            offset += batch.num_rows
    assert len(manifest["cases"]) == 96
    manifest["reference_characters"] = sum(len(normalize(c["reference"])) for c in manifest["cases"])
    manifest["audio_seconds"] = sum(c["duration_seconds"] for c in manifest["cases"])
    assert manifest["reference_characters"] >= 1000
    path = Path("docs/eval/mandarin-v1.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError("existing fixed manifest differs; do not replace an evaluation set silently")
    path.write_text(text)
    print(f"Saved {path}: 96 clips, {manifest['reference_characters']} characters, {manifest['audio_seconds']:.2f}s")


if __name__ == "__main__":
    main()
