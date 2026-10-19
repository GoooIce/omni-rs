# Native UI font

`NotoSansSC-VF.ttf` is the Simplified Chinese subset of the Noto Sans CJK variable font. The UI requests weights 400 and 600 through Makepad's `FontMember.weight`, with IBM Plex Sans for Latin text.

- Source: https://github.com/notofonts/noto-cjk/tree/main/Sans/Variable/TTF/Subset
- Downloaded 2026-10-15 from https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/Variable/TTF/Subset/NotoSansSC-VF.ttf
- License: SIL Open Font License 1.1, included as `OFL.txt`; downloaded from https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/LICENSE
- SHA-256: `d68bafcb48a2707749396aa12bbbd833cb70401f3a9a689fd2902c7e0d295964`

The complete subset is kept so arbitrary Chinese user input does not depend on a system font or a UI-label-only glyph subset.

`speech_font` then falls through to these faces when IBM Plex Sans and Noto Sans SC have no glyph. Japanese kana is already in the SC subset. Latin, Cyrillic, Greek and Vietnamese stay on IBM Plex Sans.

| File | Covers | Source | SHA-256 |
| --- | --- | --- | --- |
| `NotoSansKR-Hangul.ttf` | Hangul jamo and syllables | Subset of [NotoSansKR-VF.ttf](https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/Variable/TTF/Subset/NotoSansKR-VF.ttf) (`9e1d729e7e2b36f9ef439da102f8c134c10aabe46f1c843bf0aca5c043b86f76`), ranges U+1100–11FF, U+3130–318F, U+A960–A97F, U+AC00–D7A3, U+D7B0–D7FF, keeping `fvar` and `GSUB` | `325af5cb2a8372dc2fe188c4dc0ef9b0fb0a6f354b802e9ddd79c0e39b459e33` |
| `NotoSansArabic-VF.ttf` | Arabic and Persian | [Noto Sans Arabic](https://github.com/google/fonts/blob/main/ofl/notosansarabic/NotoSansArabic%5Bwdth%2Cwght%5D.ttf), downloaded 2026-10-19 | `63111b5b2e074dd48cc67692e0a2726d86ee94c1c37fe8598257b7b4e87e869e` |
| `NotoSansThai-VF.ttf` | Thai | [Noto Sans Thai](https://github.com/google/fonts/blob/main/ofl/notosansthai/NotoSansThai%5Bwdth%2Cwght%5D.ttf), downloaded 2026-10-19 | `5a1c559bb539583c8a1fd99d1c5b9491e5e14478c9cd2bd0970d5c3096cc9ef8` |
| `NotoSansDevanagari-VF.ttf` | Hindi | [Noto Sans Devanagari](https://github.com/google/fonts/blob/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf), downloaded 2026-10-19 | `14ec4af41f27482216d1c2229f417ff9b1425e1babb014e57d1d40d03229853e` |

All four are SIL Open Font License 1.1, the same terms as `OFL.txt`.
