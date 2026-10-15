//! Lossless event tokens, timed MIDI, and the two-voice ABC accepted by YuE2.
use crate::{Result, invalid};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
pub(crate) const PREFIX: [u32; 8] = [1, 4, 5, 6, 7, 9, 11, 3];
const DURATIONS: [usize; 24] = [
    1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
    3072, 4096,
];
const PITCHES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];
#[derive(Deserialize)]
struct SchemaFile {
    schema: Schema,
}
#[derive(Deserialize)]
struct Schema {
    structure_labels: Vec<String>,
    full_chord_labels: Vec<String>,
}
fn schema() -> &'static Schema {
    static SCHEMA: std::sync::OnceLock<Schema> = std::sync::OnceLock::new();
    SCHEMA.get_or_init(|| {
        serde_json::from_str::<SchemaFile>(include_str!("score-schema.json"))
            .expect("bundled score schema")
            .schema
    })
}

pub(crate) struct Grammar {
    in_shift: bool,
    shift_run: usize,
    payload: usize,
    last: i8,
    incomplete: u8,
}
impl Default for Grammar {
    fn default() -> Self {
        Self {
            in_shift: true,
            shift_run: 0,
            payload: 0,
            last: -1,
            incomplete: 0,
        }
    }
}
impl Grammar {
    fn allowed(&self, id: u32) -> bool {
        if id == 2 {
            return self.payload > 0;
        }
        if (260..517).contains(&id) {
            return (self.payload > 0 || self.in_shift) && self.shift_run < 4;
        }
        if self.incomplete == 1 {
            return (30709..30965).contains(&id);
        }
        if self.incomplete == 2 {
            return (31398..31678).contains(&id);
        }
        match id {
            517..30517 => self.last < 0,
            30517..30965 => self.last < 1,
            30965..30988 => self.last < 2,
            30988..31012 => self.last < 3,
            31037..31398 => self.last < 4,
            31398..31654 => self.last <= 5,
            _ => false,
        }
    }
    pub fn select(&self, logits: &[f32]) -> Result<u32> {
        if logits.len() != 31678 || logits.iter().any(|v| !v.is_finite()) {
            return Err(invalid("invalid SheetSage2 logits"));
        }
        logits
            .iter()
            .enumerate()
            .filter(|(i, _)| self.allowed(*i as u32))
            .max_by(|(i, a), (j, b)| a.total_cmp(b).then(j.cmp(i)))
            .map(|(id, _)| id as u32)
            .ok_or_else(|| invalid("empty score grammar"))
    }
    pub fn update(&mut self, id: u32) -> bool {
        if id == 2 {
            return true;
        }
        if (260..517).contains(&id) {
            if !self.in_shift && self.payload > 0 {
                self.payload = 0;
                self.last = -1;
                self.incomplete = 0;
            }
            self.in_shift = true;
            self.shift_run += 1;
            return false;
        }
        self.in_shift = false;
        self.shift_run = 0;
        self.payload += 1;
        self.incomplete = 0;
        self.last = match id {
            517..30517 => 0,
            30517..30709 => {
                self.incomplete = 1;
                1
            }
            30709..30965 => 1,
            30965..30988 => 2,
            30988..31012 => 3,
            31037..31398 => 4,
            31398..31654 => {
                self.incomplete = 2;
                5
            }
            31654..31678 => 5,
            _ => -1,
        };
        false
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Note {
    pub pitch: u8,
    pub track: u8,
    pub duration_steps: usize,
    pub start: f64,
    pub end: f64,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Event {
    pub subbeat: usize,
    pub time: f64,
    pub timestamp: Option<f64>,
    pub meter: Option<[usize; 2]>,
    pub eighth_position: Option<usize>,
    pub structure: Option<String>,
    pub key: Option<String>,
    pub chord: Option<String>,
    pub notes: Vec<Note>,
    pub tokens: Vec<u32>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Score {
    pub model: String,
    pub audio_seconds: f64,
    pub tokens: Vec<u32>,
    pub events: Vec<Event>,
    pub abc: Option<String>,
    pub abc_error: Option<String>,
    pub melody_only: bool,
    pub truncated: bool,
    pub warnings: Vec<String>,
}
impl Score {
    pub fn midi(&self) -> Vec<u8> {
        fn vlq(mut n: u32, out: &mut Vec<u8>) {
            let mut bytes = vec![(n & 127) as u8];
            while {
                n >>= 7;
                n > 0
            } {
                bytes.push((n as u8 & 127) | 128);
            }
            out.extend(bytes.into_iter().rev());
        }
        let mut out = b"MThd\0\0\0\x06\0\x01\0\x03\x03\xc0".to_vec();
        let conductor = b"\0\xff\x51\x03\x07\xa1\x20\0\xff\x2f\0";
        out.extend(b"MTrk");
        out.extend((conductor.len() as u32).to_be_bytes());
        out.extend(conductor);
        for track in 0..2 {
            let name = if track == 0 {
                b"Vocal".as_slice()
            } else {
                b"Ins".as_slice()
            };
            let mut data = vec![0, 0xff, 3, name.len() as u8];
            data.extend(name);
            let mut notes = Vec::new();
            for event in &self.events {
                for n in &event.notes {
                    if n.track == track && n.end > n.start {
                        notes.push(((n.start * 1920.0).round() as u32, true, n.pitch));
                        notes.push(((n.end * 1920.0).round() as u32, false, n.pitch));
                    }
                }
            }
            notes.sort_unstable();
            let mut previous = 0;
            for (time, on, pitch) in notes {
                vlq(time - previous, &mut data);
                data.extend([
                    if on { 0x90 | track } else { 0x80 | track },
                    pitch,
                    if on { 100 } else { 0 },
                ]);
                previous = time;
            }
            data.extend([0, 0xff, 0x2f, 0]);
            out.extend(b"MTrk");
            out.extend((data.len() as u32).to_be_bytes());
            out.extend(data);
        }
        out
    }
}
pub(crate) fn decode(
    tokens: Vec<u32>,
    seconds: f64,
    melody_only: bool,
    truncated: bool,
) -> Result<Score> {
    if !tokens.starts_with(&PREFIX) || tokens.last() != Some(&2) {
        return Err(invalid("invalid score token framing"));
    }
    let mut events = Vec::new();
    let mut pos = 8;
    let mut step = 0;
    while pos < tokens.len() - 1 {
        if !(260..517).contains(&tokens[pos]) {
            return Err(invalid("score event has no subbeat shift"));
        }
        while pos < tokens.len() && (260..517).contains(&tokens[pos]) {
            step += (tokens[pos] - 260) as usize;
            pos += 1;
        }
        let mut e = Event {
            subbeat: step,
            ..Default::default()
        };
        while pos < tokens.len() && tokens[pos] != 2 && !(260..517).contains(&tokens[pos]) {
            let id = tokens[pos];
            e.tokens.push(id);
            pos += 1;
            match id {
                517..30517 => e.timestamp = Some(f64::from(id - 517) / 100.0),
                30517..30709 => {
                    let i = (id - 30517) as usize;
                    e.meter = Some([i / 6 + 1, [1, 2, 4, 8, 16, 32][i % 6]]);
                }
                30709..30965 => e.eighth_position = Some((id - 30709) as usize),
                30965..30988 => {
                    e.structure = Some(schema().structure_labels[(id - 30965) as usize].clone())
                }
                30988..31012 => {
                    let i = (id - 30988) as usize;
                    e.key = Some(format!(
                        "{}{}",
                        PITCHES[i % 12],
                        if i >= 12 { "m" } else { "" }
                    ));
                }
                31037..31398 => {
                    e.chord = Some(schema().full_chord_labels[(id - 31037) as usize].clone())
                }
                31398..31654 => {
                    let i = id - 31398;
                    let mut duration_steps = 1;
                    if pos < tokens.len() && (31654..31678).contains(&tokens[pos]) {
                        duration_steps = DURATIONS[(tokens[pos] - 31654) as usize];
                        e.tokens.push(tokens[pos]);
                        pos += 1;
                    }
                    e.notes.push(Note {
                        pitch: (i % 128) as u8,
                        track: (i / 128) as u8,
                        duration_steps,
                        start: 0.0,
                        end: 0.0,
                    });
                }
                _ => return Err(invalid("invalid score event payload")),
            }
        }
        if !e.tokens.is_empty() {
            events.push(e);
        }
    }
    let anchors: BTreeMap<usize, f64> = events
        .iter()
        .filter_map(|e| e.timestamp.map(|t| (e.subbeat, t)))
        .collect();
    let anchors: Vec<_> = anchors.into_iter().collect();
    let mut rates: Vec<f64> = anchors
        .windows(2)
        .map(|v| (v[1].1 - v[0].1) / (v[1].0 - v[0].0).max(1) as f64)
        .collect();
    rates.sort_by(f64::total_cmp);
    let rate = if rates.is_empty() {
        0.125
    } else {
        let n = rates.len();
        let median = (rates[(n - 1) / 2] + rates[n / 2]) / 2.0;
        if median.is_finite() && median > 0.0 {
            median
        } else {
            0.125
        }
    };
    let lookup = |step: usize| -> f64 {
        let t = if anchors.is_empty() {
            step as f64 * 0.125
        } else if step <= anchors[0].0 {
            anchors[0].1 + (step as f64 - anchors[0].0 as f64) * rate
        } else if step >= anchors.last().unwrap().0 {
            let &(s, t) = anchors.last().unwrap();
            t + (step - s) as f64 * rate
        } else {
            let pair = anchors
                .windows(2)
                .find(|v| v[0].0 <= step && step <= v[1].0)
                .unwrap();
            pair[0].1
                + (pair[1].1 - pair[0].1) * (step - pair[0].0) as f64
                    / (pair[1].0 - pair[0].0) as f64
        };
        t.clamp(0.0, seconds)
    };
    for e in &mut events {
        e.time = lookup(e.subbeat);
        for n in &mut e.notes {
            n.start = e.time;
            n.end = lookup(e.subbeat + n.duration_steps);
        }
    }
    let mut score = Score {
        model: "m-a-p/SheetSage2".into(),
        audio_seconds: seconds,
        tokens,
        events,
        abc: None,
        abc_error: None,
        melody_only,
        truncated,
        warnings: Vec::new(),
    };
    if anchors.is_empty() {
        score
            .warnings
            .push("No timestamps: event timing uses the upstream 0.125-second fallback".into());
    }
    if truncated {
        score
            .warnings
            .push("Score decoder reached its token limit".into());
    }
    match render(&score) {
        Ok((abc, warnings)) => {
            score.abc = Some(abc);
            score.warnings.extend(warnings);
        }
        Err(e) => score.abc_error = Some(e.to_string()),
    }
    Ok(score)
}

#[derive(Clone)]
struct Beat {
    time: f64,
    id: usize,
    numerator: usize,
    denominator: usize,
}
struct Measure {
    start: usize,
    end: usize,
    numerator: usize,
    denominator: usize,
    notated: usize,
    pad_before: bool,
}
fn mode(values: impl Iterator<Item = usize>) -> usize {
    let values: Vec<_> = values.collect();
    *values
        .iter()
        .rev()
        .max_by_key(|&&v| values.iter().filter(|&&x| x == v).count())
        .unwrap()
}
fn render(score: &Score) -> Result<(String, Vec<String>)> {
    let mut meter = None;
    let mut beats = Vec::new();
    let mut warnings = Vec::new();
    for e in &score.events {
        meter = e.meter.or(meter);
        if let (Some([num, den]), Some(pos)) = (meter, e.eighth_position) {
            if pos * den % 8 != 0 || pos * den / 8 >= num {
                return Err(invalid("decoded rhythm is off its declared beat grid"));
            }
            beats.push(Beat {
                time: e.time,
                id: pos * den / 8 + 1,
                numerator: num,
                denominator: den,
            });
        }
    }
    if beats.len() < 2 || beats.windows(2).any(|p| p[0].time >= p[1].time) {
        return Err(invalid("ABC requires an increasing beat grid"));
    }
    let downbeats: Vec<_> = beats
        .iter()
        .enumerate()
        .filter_map(|(i, b)| (b.id == 1).then_some(i))
        .collect();
    if downbeats.is_empty() {
        return Err(invalid("ABC requires a decoded downbeat"));
    }
    let mut spans = Vec::new();
    if downbeats[0] > 0 {
        spans.push((0, downbeats[0], true, false));
    }
    for p in downbeats.windows(2) {
        spans.push((p[0], p[1], false, false));
    }
    let last = *downbeats.last().unwrap();
    if last < beats.len() - 1 {
        spans.push((last, beats.len() - 1, false, true));
    }
    let mut measures = Vec::new();
    for (start, end, pickup, partial) in spans {
        let b = &beats[start..end];
        if b.iter().enumerate().any(|(i, v)| v.id != b[0].id + i) {
            return Err(invalid(
                "ABC requires consecutive beat IDs within each measure",
            ));
        }
        let den = mode(b.iter().map(|v| v.denominator));
        let declared = mode(b.iter().map(|v| v.numerator));
        let count = end - start;
        let pad = partial
            && declared >= count
            && b.iter()
                .all(|v| v.denominator == den && v.numerator == declared);
        let notated = if pad { declared } else { count };
        if b.iter()
            .any(|v| v.numerator != notated || v.denominator != den)
        {
            warnings.push(format!(
                "Inferred measure {count}/{den} from downbeat boundaries"
            ));
        }
        measures.push(Measure {
            start,
            end,
            numerator: count,
            denominator: den,
            notated,
            pad_before: pickup,
        });
    }
    if measures.is_empty() {
        return Err(invalid("ABC has no positive-length measure"));
    }
    if measures.len() > 1
        && measures[0].numerator * measures[1].denominator
            < measures[1].notated * measures[0].denominator
    {
        if measures[0].denominator != measures[1].denominator {
            return Err(invalid(
                "pickup and following meter use different denominators",
            ));
        }
        measures[0].notated = measures[1].notated;
        measures[0].pad_before = true;
        warnings.push("Padded the pickup with preceding rests".into());
    }
    let mut times = Vec::new();
    let mut dens = Vec::new();
    let mut quarters = 0.0;
    for m in &measures {
        for i in m.start..m.end {
            for k in 0..4 {
                times.push(beats[i].time + (beats[i + 1].time - beats[i].time) * k as f64 / 4.0);
                dens.push(m.denominator);
                quarters += 1.0 / m.denominator as f64;
            }
        }
    }
    times.push(beats.last().unwrap().time);
    let length = times.len() - 1;
    let quantize = |t: f64| -> usize {
        times
            .windows(2)
            .take_while(|v| t > (v[0] + v[1]) / 2.0)
            .count()
    };
    let first_key = score
        .events
        .iter()
        .find_map(|e| e.key.clone())
        .ok_or_else(|| invalid("ABC requires a decoded musical key"))?;
    let mut keys = vec![first_key; length];
    let mut chords = vec!["N".to_string(); length];
    let mut sections = BTreeMap::<usize, String>::new();
    for e in &score.events {
        let at = quantize(e.time).min(length);
        if let Some(key) = &e.key {
            keys[at..].fill(key.clone());
        }
        if let Some(chord) = &e.chord {
            chords[at..].fill(chord.clone());
        }
        if let Some(label) = &e.structure {
            sections.insert(at, label.clone());
        }
    }
    let mut pitches = [vec![None; length], vec![None; length]];
    let mut onsets = [vec![false; length], vec![false; length]];
    for track in 0..2 {
        let mut notes: Vec<_> = score
            .events
            .iter()
            .flat_map(|e| e.notes.iter())
            .filter(|n| n.track as usize == track && n.end > n.start)
            .cloned()
            .collect();
        notes.sort_by(|a, b| {
            a.start
                .total_cmp(&b.start)
                .then(a.pitch.cmp(&b.pitch))
                .then(a.end.total_cmp(&b.end))
        });
        for i in 0..notes.len() {
            let n = &notes[i];
            let end = if let Some(next) = notes.get(i + 1) {
                if n.end > next.start + 1e-6 {
                    warnings.push(format!(
                        "Notation only: clipped track {track} note at {:.3} to next onset",
                        n.start
                    ));
                    n.end.min(next.start)
                } else {
                    n.end
                }
            } else {
                n.end
            };
            if end <= n.start + 1e-6 {
                continue;
            }
            let a = quantize(n.start).min(length);
            let b = quantize(end).min(length);
            if b <= a {
                return Err(invalid(
                    "a note cannot be represented on the decoded beat grid",
                ));
            }
            if pitches[track][a..b].iter().any(Option::is_some) {
                return Err(invalid("overlapping quantized melody notes"));
            }
            pitches[track][a..b].fill(Some(n.pitch));
            onsets[track][a] = true;
        }
    }
    let unit = measures.iter().map(|m| m.denominator * 4).max().unwrap();
    let seconds = times[length] - times[0];
    if seconds <= 0.0 {
        return Err(invalid("zero-duration ABC beat grid"));
    }
    let tempo = (quarters / seconds * 60.0).round() as usize;
    let first = &measures[0];
    let mut out = format!(
        "X:1\nT:\nM:{}/{}\nL:1/{unit}\nQ:1/4={tempo}\nV: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\nV: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\nK:{}\n",
        first.notated,
        first.denominator,
        portable_key(&keys[0])
    );
    for (index, m) in measures.iter().enumerate() {
        let start = m.start * 4;
        let end = m.end * 4;
        for (_, label) in sections.range(start..end) {
            out.push_str(&format!("% {label}\n"));
        }
        for track in 0..2 {
            out.push_str(if track == 0 { "V: Vocal\n" } else { "V: Ins\n" });
            if index > 0 {
                out.push_str(&format!(
                    "M:{}/{}\nK:{}\n",
                    m.notated,
                    m.denominator,
                    portable_key(&keys[start])
                ));
            }
            let padding = (m.notated - m.numerator) * unit / m.denominator;
            if m.pad_before && padding > 0 {
                duration(&mut out, None, padding, false);
            }
            let mut cursor = start;
            while cursor < end {
                let pitch = pitches[track][cursor];
                let key = &keys[cursor];
                if cursor > start && keys[cursor - 1] != *key {
                    out.push_str(&format!("[K:{}] ", portable_key(key)));
                }
                if track == 0
                    && !score.melody_only
                    && (cursor == start || chords[cursor - 1] != chords[cursor])
                    && let Some(chord) = chord(&chords[cursor])?
                {
                    out.push_str(&format!("\"{chord}\""));
                }
                let mut next = cursor + 1;
                while next < end
                    && pitches[track][next] == pitch
                    && !onsets[track][next]
                    && keys[next] == *key
                    && (score.melody_only || track == 1 || chords[next] == chords[cursor])
                {
                    next += 1;
                }
                let units: usize = dens[cursor..next].iter().map(|d| unit / (4 * d)).sum();
                let tied = pitch.is_some()
                    && next < length
                    && pitches[track][next] == pitch
                    && !onsets[track][next];
                duration(&mut out, pitch, units, tied);
                cursor = next;
            }
            if !m.pad_before && padding > 0 {
                duration(&mut out, None, padding, false);
            }
            out.push_str("|\n");
        }
    }
    Ok((out, warnings))
}
fn portable_key(key: &str) -> String {
    match key {
        "D#" => "Eb",
        "G#" => "Ab",
        "A#" => "Bb",
        "A#m" => "Bbm",
        "D#m" => "Ebm",
        v => v,
    }
    .into()
}
fn note(pitch: u8) -> String {
    let names = [
        "=C", "^C", "=D", "^D", "=E", "=F", "^F", "=G", "^G", "=A", "^A", "=B",
    ];
    let mut s = names[(pitch % 12) as usize].to_string();
    let octave = i32::from(pitch) / 12 - 5;
    if octave > 0 {
        s = s.to_lowercase();
        s.push_str(&"'".repeat((octave - 1) as usize));
    } else if octave < 0 {
        s.push_str(&",".repeat((-octave) as usize));
    }
    s
}
fn duration(out: &mut String, pitch: Option<u8>, mut n: usize, tied: bool) {
    while n > 0 {
        let take = [48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1]
            .into_iter()
            .find(|&x| x <= n)
            .unwrap();
        n -= take;
        out.push_str(&pitch.map(note).unwrap_or_else(|| "z".into()));
        if take != 1 {
            out.push_str(&take.to_string());
        }
        if pitch.is_some() && (n > 0 || tied) {
            out.push('-');
        }
        out.push(' ');
    }
}
fn chord(label: &str) -> Result<Option<String>> {
    if label == "N" {
        return Ok(None);
    }
    let (root, quality) = label
        .split_once(':')
        .ok_or_else(|| invalid("invalid chord"))?;
    let (q, bass) = quality
        .split_once('/')
        .map_or((quality, None), |(q, b)| (q, Some(b)));
    let suffix = match q {
        "maj" => "",
        "min" => "m",
        "dim" => "dim",
        "aug" => "aug",
        "maj7" => "maj7",
        "min7" => "m7",
        "7" => "7",
        "hdim7" => "m7b5",
        "dim7" => "dim7",
        "minmaj7" => "mmaj7",
        "sus2" => "sus2",
        "sus4" => "sus4",
        "sus4(b7)" => "7sus4",
        "maj6" => "6",
        "min6" => "m6",
        _ => return Err(invalid("unsupported chord quality")),
    };
    let mut text = format!("{root}{suffix}");
    if let Some(b) = bass {
        let root = PITCHES
            .iter()
            .position(|&p| p == root)
            .ok_or_else(|| invalid("invalid chord root"))? as i32;
        let degree = b
            .trim_start_matches(['b', '#'])
            .parse::<usize>()
            .map_err(|_| invalid("invalid chord bass"))?;
        if degree == 0 {
            return Err(invalid("invalid bass degree"));
        }
        let offset = [0, 2, 4, 5, 7, 9, 11][(degree - 1) % 7] + b.matches('#').count() as i32
            - b.matches('b').count() as i32;
        text.push('/');
        text.push_str(PITCHES[(root + offset).rem_euclid(12) as usize]);
    }
    Ok(Some(text))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grammar_matches_field_order_and_incomplete_payloads() {
        let mut g = Grammar::default();
        assert!(g.allowed(260));
        assert!(!g.allowed(2));
        g.update(260);
        g.update(30537);
        assert!(g.allowed(30709));
        assert!(!g.allowed(517));
        g.update(30709);
        g.update(31398 + 60);
        assert!(g.allowed(31654));
        assert!(!g.allowed(30988));
        g.update(31658);
        assert!(g.allowed(2));
        g.update(264);
        assert!(g.allowed(517));
    }
    #[test]
    fn midi_preserves_both_tracks_and_abc_uses_actual_meter() {
        let mut tokens = PREFIX.to_vec();
        for beat in 0..4u32 {
            tokens.push(if beat == 0 { 260 } else { 264 });
            tokens.push(517 + beat * 50);
            if beat == 0 {
                tokens.push(30517 + 2 * 6 + 2);
            }
            tokens.push(30709 + (beat % 3) * 2);
            if beat == 0 {
                tokens.push(30988);
            }
            if beat < 3 {
                tokens.push(31398 + 60 + beat);
                tokens.push(31658);
            }
        }
        tokens.push(2);
        let s = decode(tokens, 2.0, true, false).unwrap();
        let abc = s.abc.as_ref().unwrap();
        assert!(abc.contains("M:3/4"));
        assert!(abc.contains("Q:1/4=120"));
        assert!(abc.contains("V: Ins"));
        assert_eq!(&s.midi()[..4], b"MThd");
    }
    #[test]
    fn absent_rhythm_remains_a_reported_score_failure() {
        let mut t = PREFIX.to_vec();
        t.extend([260, 517, 31398 + 60, 31658, 2]);
        let s = decode(t, 1.0, true, false).unwrap();
        assert!(s.abc.is_none());
        assert!(s.abc_error.is_some());
        assert_eq!(s.events[0].notes[0].pitch, 60);
    }
}
