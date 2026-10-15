#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


class WebPackagingTest(unittest.TestCase):
    def test_split_parameters_round_trip_without_value_changes(self):
        with tempfile.TemporaryDirectory(prefix="r2t2-pack-") as temp:
            root = Path(temp)
            source = root / "model"
            source.mkdir()
            # Cross both the transfer piece boundary and an odd tensor boundary.
            body = bytes(range(256)) * 8193
            header = {"weight": {"dtype": "BF16", "shape": [len(body)//2], "data_offsets": [0, len(body)]}}
            encoded = json.dumps(header).encode()
            (source / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)
            (source / "config.json").write_text("{}")
            (source / "tokenizer.json").write_text("{}")
            output = root / "web"
            subprocess.run([sys.executable, str(Path(__file__).with_name("pack_web.py")), str(source), str(output), "--chunk-mib", "1"], check=True)
            manifest = json.loads((output / "manifest.json").read_text())
            reconstructed = bytearray()
            for part in manifest["chunks"]:
                data = (output / part["file"]).read_bytes()
                self.assertEqual(part["offset"], len(reconstructed)//2)
                self.assertEqual(part["sha256"], hashlib.sha256(data).hexdigest())
                reconstructed.extend(data)
            self.assertEqual(reconstructed, body)


if __name__ == "__main__":
    unittest.main()
