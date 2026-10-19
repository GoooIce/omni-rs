"""Device-free checks for the Makepad transcript font and ASR language list."""
import json
from pathlib import Path
import re
import struct
import unittest

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "prototypes/makepad/resources/fonts"
# One code point from each script the transcript face must draw itself.
PROBES = {
    "NotoSansKR-Hangul.ttf": "한",
    "NotoSansArabic-VF.ttf": "ا",
    "NotoSansThai-VF.ttf": "ก",
    "NotoSansDevanagari-VF.ttf": "ह",
}


def cmap_has(path, character):
    data = path.read_bytes()
    if data[:4] not in (b"\x00\x01\x00\x00", b"OTTO", b"true"):
        raise AssertionError(f"{path.name} is not a TrueType or OpenType font")
    tables = {}
    count = struct.unpack_from(">H", data, 4)[0]
    for index in range(count):
        tag, _checksum, offset, length = struct.unpack_from(">4sIII", data, 12 + index * 16)
        tables[tag] = data[offset : offset + length]
    cmap = tables[b"cmap"]
    _version, subtables = struct.unpack_from(">HH", cmap, 0)
    found = False
    for index in range(subtables):
        platform, encoding, offset = struct.unpack_from(">HHI", cmap, 4 + index * 8)
        if (platform, encoding) not in ((0, 3), (0, 4), (3, 1), (3, 10)):
            continue
        if _glyph(cmap, offset, ord(character)):
            found = True
    return found


def _glyph(cmap, offset, code):
    fmt = struct.unpack_from(">H", cmap, offset)[0]
    if fmt == 12:
        groups = struct.unpack_from(">I", cmap, offset + 12)[0]
        for index in range(groups):
            start, end, glyph = struct.unpack_from(">III", cmap, offset + 16 + index * 12)
            if start <= code <= end:
                return glyph + code - start
        return 0
    if fmt != 4:
        return 0
    segments = struct.unpack_from(">H", cmap, offset + 6)[0] // 2
    ends = offset + 14
    starts = ends + 2 * segments + 2
    deltas = starts + 2 * segments
    offsets = deltas + 2 * segments
    for index in range(segments):
        end = struct.unpack_from(">H", cmap, ends + 2 * index)[0]
        start = struct.unpack_from(">H", cmap, starts + 2 * index)[0]
        if not start <= code <= end:
            continue
        delta = struct.unpack_from(">h", cmap, deltas + 2 * index)[0]
        range_offset = struct.unpack_from(">H", cmap, offsets + 2 * index)[0]
        if range_offset == 0:
            return (code + delta) & 0xFFFF
        glyph_offset = offsets + 2 * index + range_offset + 2 * (code - start)
        glyph = struct.unpack_from(">H", cmap, glyph_offset)[0]
        return (glyph + delta) & 0xFFFF if glyph else 0
    return 0


class SpeechFontTests(unittest.TestCase):
    def test_transcript_fonts_cover_required_scripts(self):
        source = (ROOT / "prototypes/makepad/src/main.rs").read_text()
        family = source.split("let speech_font = FontFamily{", 1)[1].split("let speech_heading", 1)[0]
        for name, sample in PROBES.items():
            self.assertIn(f'self:resources/fonts/{name}', family, name)
            self.assertTrue(cmap_has(FONTS / name, sample), f"{name} lacks {sample}")
        self.assertFalse(cmap_has(FONTS / "NotoSansArabic-VF.ttf", "한"))

    def test_parser_languages_match_the_committed_list(self):
        expected = json.loads((ROOT / "prototypes/makepad/resources/asr-languages.json").read_text())
        source = (ROOT / "prototypes/makepad/src/protocol.rs").read_text()
        block = source.split("const ASR_LANGUAGES:", 1)[1].split("];", 1)[0]
        actual = re.findall(r'\("([A-Za-z]+)"', block)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
