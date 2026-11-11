//! Strict, bounded support for Praat's labelled long TextGrid text format.

const MAX_TIERS: usize = 4096;
const MAX_INTERVALS: usize = 250_000;
const MAX_STRING_BYTES: usize = 64 * 1024;
const MAX_TIME_SECONDS: f64 = 24.0 * 60.0 * 60.0;

pub(crate) struct Interval {
    pub start: f64,
    pub end: f64,
    pub mark: String,
}

pub(crate) struct Tier {
    pub name: String,
    pub intervals: Vec<Interval>,
}

pub(crate) fn decode(bytes: &[u8]) -> Result<String, String> {
    let text = if bytes.starts_with(&[0xff, 0xfe]) || bytes.starts_with(&[0xfe, 0xff]) {
        let little = bytes[0] == 0xff;
        let data = &bytes[2..];
        if !data.len().is_multiple_of(2) {
            return Err("odd-length UTF-16 TextGrid".into());
        }
        let words: Vec<u16> = data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| {
                if little {
                    u16::from_le_bytes([pair[0], pair[1]])
                } else {
                    u16::from_be_bytes([pair[0], pair[1]])
                }
            })
            .collect();
        String::from_utf16(&words).map_err(|_| "invalid UTF-16 TextGrid")?
    } else {
        let data = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(bytes);
        std::str::from_utf8(data)
            .map_err(|_| "TextGrid is not UTF-8 or BOM-marked UTF-16")?
            .to_owned()
    };
    if text.contains('\0') {
        return Err("NUL in TextGrid".into());
    }
    Ok(text)
}

struct Parser<'a> {
    lines: std::str::SplitInclusive<'a, char>,
    line_ending: &'a str,
}

impl<'a> Parser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            lines: text.split_inclusive('\n'),
            line_ending: "",
        }
    }

    fn read_line(&mut self) -> Option<&'a str> {
        let line = self.lines.next()?;
        if let Some(content) = line.strip_suffix("\r\n") {
            self.line_ending = "\r\n";
            Some(content)
        } else if let Some(content) = line.strip_suffix('\n') {
            self.line_ending = "\n";
            Some(content)
        } else {
            self.line_ending = "";
            Some(line)
        }
    }

    fn next(&mut self) -> Result<&'a str, String> {
        while let Some(line) = self.read_line() {
            if !line.trim().is_empty() {
                return Ok(line.trim_start());
            }
        }
        Err("truncated TextGrid".into())
    }

    fn expect(&mut self, expected: &str) -> Result<(), String> {
        if self.next()?.trim() != expected {
            return Err(format!("unsupported TextGrid: expected {expected}"));
        }
        Ok(())
    }

    fn assignment(&mut self, key: &str) -> Result<&'a str, String> {
        let (name, value) = self
            .next()?
            .split_once('=')
            .ok_or_else(|| format!("missing {key} assignment"))?;
        if name.trim() != key {
            return Err(format!("unsupported TextGrid: expected {key}"));
        }
        Ok(value.trim_start())
    }

    fn quoted(&mut self, key: &str) -> Result<String, String> {
        let value = self.assignment(key)?;
        let mut rest = value
            .strip_prefix('"')
            .ok_or_else(|| format!("{key} is not a quoted string"))?;
        let mut output = String::new();
        loop {
            let mut chars = rest.chars().peekable();
            while let Some(ch) = chars.next() {
                if ch == '"' {
                    if chars.peek() == Some(&'"') {
                        chars.next();
                        output.push('"');
                    } else {
                        if chars.any(|tail| !tail.is_whitespace()) {
                            return Err(format!("unexpected text after {key}"));
                        }
                        if output.contains('\\') {
                            return Err("Praat backslash escapes are unsupported".into());
                        }
                        return Ok(output);
                    }
                } else {
                    output.push(ch);
                }
                if output.len() > MAX_STRING_BYTES {
                    return Err(format!("{key} exceeds string limit"));
                }
            }
            if self.line_ending.is_empty() {
                return Err("unterminated quoted string".into());
            }
            output.push_str(self.line_ending);
            if output.len() > MAX_STRING_BYTES {
                return Err(format!("{key} exceeds string limit"));
            }
            rest = self.read_line().ok_or("unterminated quoted string")?;
        }
    }

    fn number(&mut self, key: &str) -> Result<f64, String> {
        let value: f64 = self
            .assignment(key)?
            .trim()
            .parse()
            .map_err(|_| format!("invalid {key}"))?;
        if !value.is_finite() || !(0.0..=MAX_TIME_SECONDS).contains(&value) {
            return Err(format!("invalid {key} time"));
        }
        Ok(value)
    }

    fn count(&mut self, key: &str, limit: usize) -> Result<usize, String> {
        let count: usize = self
            .assignment(key)?
            .trim()
            .parse()
            .map_err(|_| format!("invalid {key} count"))?;
        if count > limit {
            return Err(format!("{key} count exceeds limit"));
        }
        Ok(count)
    }

    fn index(&mut self, name: &str, expected: usize) -> Result<(), String> {
        let line = self.next()?.trim();
        let prefix = format!("{name} [");
        let index = line
            .strip_prefix(&prefix)
            .and_then(|value| value.strip_suffix("]:"))
            .and_then(|value| value.parse::<usize>().ok());
        if index != Some(expected) {
            return Err(format!("invalid {name} index"));
        }
        Ok(())
    }
}

pub(crate) fn parse(text: &str) -> Result<Vec<Tier>, String> {
    let mut parser = Parser::new(text);
    if parser.quoted("File type")? != "ooTextFile" || parser.quoted("Object class")? != "TextGrid" {
        return Err("not a labelled long TextGrid".into());
    }
    let grid_start = parser.number("xmin")?;
    let grid_end = parser.number("xmax")?;
    if grid_end < grid_start {
        return Err("invalid TextGrid time domain".into());
    }
    let flag = parser.next()?.trim();
    if flag == "tiers? <absent>" {
        if parser.lines.any(|line| !line.trim().is_empty()) {
            return Err("unexpected data after absent tiers".into());
        }
        return Ok(Vec::new());
    }
    if flag != "tiers? <exists>" {
        return Err("unsupported tier flag".into());
    }
    let tier_count = parser.count("size", MAX_TIERS)?;
    parser.expect("item []:")?;
    let mut tiers = Vec::with_capacity(tier_count);
    let mut interval_total = 0_usize;
    for i in 1..=tier_count {
        parser.index("item", i)?;
        let class = parser.quoted("class")?;
        let name = parser.quoted("name")?;
        if name.len() > 256
            || name == "."
            || name == ".."
            || name.contains(['/', '\\'])
            || name.chars().any(char::is_control)
        {
            return Err("invalid tier label".into());
        }
        let start = parser.number("xmin")?;
        let end = parser.number("xmax")?;
        if start < grid_start || end > grid_end || end < start {
            return Err("tier time is outside TextGrid".into());
        }
        if class == "TextTier" {
            if parser.count("points: size", MAX_INTERVALS)? != 0 {
                return Err("nonempty point tiers are unsupported".into());
            }
            tiers.push(Tier {
                name,
                intervals: Vec::new(),
            });
            continue;
        }
        if class != "IntervalTier" {
            return Err("unsupported TextGrid tier type".into());
        }
        let count = parser.count("intervals: size", MAX_INTERVALS)?;
        interval_total = interval_total
            .checked_add(count)
            .ok_or("interval count overflow")?;
        if interval_total > MAX_INTERVALS {
            return Err("interval count exceeds per-file limit".into());
        }
        let mut intervals: Vec<Interval> = Vec::with_capacity(count);
        for j in 1..=count {
            parser.index("intervals", j)?;
            let first = parser.number("xmin")?;
            let last = parser.number("xmax")?;
            let mark = parser.quoted("text")?;
            if first < start || last > end || last <= first {
                return Err("invalid interval time".into());
            }
            if intervals
                .last()
                .is_some_and(|previous| first < previous.end)
            {
                return Err("overlapping or out-of-order intervals in tier".into());
            }
            if name.is_empty() && !mark.is_empty() {
                return Err("nonempty interval has no tier label".into());
            }
            intervals.push(Interval {
                start: first,
                end: last,
                mark,
            });
        }
        tiers.push(Tier { name, intervals });
    }
    if parser.lines.any(|line| !line.trim().is_empty()) {
        return Err("unexpected data after TextGrid".into());
    }
    Ok(tiers)
}
