//! A speech script is spoken as an ordered list of speech texts, one speech
//! synthesis each. Splitting only chooses where a speech text ends; the words
//! and their order stay the script's own.
use crate::{Error, Result};

/// Text tokens per speech text once a script is split. The validated 49.2 s
/// generation spoke 115 tokens, so this is about a minute of Chinese speech and
/// half of the 1500-frame ceiling.
pub const TEXT_TOKENS: usize = 128;
/// Budget for the first speech text, about ten seconds of speech. A script that
/// derives its voice anchor records that anchor from this speech text.
pub const OPENING_TEXT_TOKENS: usize = 24;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct SpeechText {
    pub text: String,
    /// Token count of this speech text's own part of the target prompt.
    pub tokens: usize,
}

/// Token budgets for the speech texts of one script.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScriptBudget {
    /// Budget for the first speech text. Zero gives it the same budget as the rest.
    pub opening: usize,
    /// Budget for every later speech text.
    pub rest: usize,
}
impl Default for ScriptBudget {
    fn default() -> Self {
        Self {
            opening: OPENING_TEXT_TOKENS,
            rest: TEXT_TOKENS,
        }
    }
}

/// Tokenize text the way upstream's templates.py renders a prompt: parse its
/// special tokens, render it, then tokenize again without adding any.
pub(crate) fn render(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
    let error = |e: tokenizers::Error| Error::Tokenizer(e.to_string());
    let encoded = tokenizer.encode(text, true).map_err(error)?;
    let rendered = tokenizer.decode(encoded.get_ids(), false).map_err(error)?;
    Ok(tokenizer
        .encode(rendered, false)
        .map_err(error)?
        .get_ids()
        .to_vec())
}

/// Splits speech scripts with the model's own tokenizer.
pub struct Splitter<'a> {
    tokenizer: &'a tokenizers::Tokenizer,
    /// Tokens the rendered form adds in front of every text, its `<bos>`.
    overhead: usize,
}
impl<'a> Splitter<'a> {
    pub fn new(tokenizer: &'a tokenizers::Tokenizer) -> Result<Self> {
        Ok(Self {
            overhead: render(tokenizer, "")?.len(),
            tokenizer,
        })
    }
    pub fn split(&self, script: &str, budget: ScriptBudget) -> Result<Vec<SpeechText>> {
        split(script, budget, &mut |text| {
            Ok(render(self.tokenizer, text)?
                .len()
                .saturating_sub(self.overhead))
        })
    }
}

/// Split a speech script into speech texts. `count` gives the token count of a
/// candidate speech text. A script within the budget is one speech text and is
/// returned unchanged, so a single speech synthesis sees exactly what it saw before.
///
/// Speech texts end after sentence punctuation. A sentence too long for its
/// budget ends after clause punctuation, a line break or a space, and what is
/// still too long is cut between characters. Nothing is dropped: removing
/// whitespace from the speech texts and concatenating them gives the script
/// without whitespace.
pub fn split(
    script: &str,
    budget: ScriptBudget,
    count: &mut dyn FnMut(&str) -> Result<usize>,
) -> Result<Vec<SpeechText>> {
    if budget.rest == 0 {
        return Err(Error::Config("speech text budget must be positive".into()));
    }
    let tokens = count(script)?;
    if tokens <= budget.rest {
        return Ok(vec![SpeechText {
            text: script.to_owned(),
            tokens,
        }]);
    }
    let end = script.trim_end().len();
    let mut start = script.len() - script.trim_start().len();
    if start >= end {
        return Err(Error::Config("请输入要合成的文本".into()));
    }
    let bounds = boundaries(script);
    let mut texts = Vec::new();
    while start < end {
        let opening = texts.is_empty() && budget.opening > 0 && budget.opening < budget.rest;
        let limit = if opening { budget.opening } else { budget.rest };
        let rest = &script[start..end];
        let tokens = count(rest)?;
        if tokens <= limit {
            texts.push(SpeechText {
                text: rest.to_owned(),
                tokens,
            });
            break;
        }
        // The opening keeps its first sentence whole, up to the full budget.
        let (text, tokens, next) = take(
            script,
            start,
            end,
            limit,
            opening.then_some(budget.rest),
            &bounds,
            tokens,
            count,
        )?;
        texts.push(SpeechText { text, tokens });
        start = next;
    }
    Ok(texts)
}

/// The next speech text, which begins at `start`. `tokens` counts the whole remainder.
fn take(
    script: &str,
    start: usize,
    end: usize,
    budget: usize,
    first_sentence: Option<usize>,
    bounds: &[(usize, Strength)],
    tokens: usize,
    count: &mut dyn FnMut(&str) -> Result<usize>,
) -> Result<(String, usize, usize)> {
    for strength in [
        Strength::Sentence,
        Strength::Clause,
        Strength::LineBreak,
        Strength::Word,
    ] {
        let mut chosen = None;
        let mut first = true;
        for &(offset, level) in bounds {
            if level != strength || offset <= start || offset > end {
                continue;
            }
            let candidate = script[start..offset].trim_end();
            let tokens = count(candidate)?;
            let allowed = match first.then_some(level) {
                Some(Strength::Sentence) => first_sentence.unwrap_or(budget),
                _ => budget,
            };
            if tokens > allowed {
                break;
            }
            chosen = Some((offset, candidate, tokens));
            first = false;
        }
        if let Some((offset, text, tokens)) = chosen {
            return Ok((
                text.to_owned(),
                tokens,
                after_whitespace(script, offset, end),
            ));
        }
    }
    cut(script, start, end, budget, tokens, count)
}

/// No boundary fits, so the remainder is cut between characters. The pieces are
/// of near-equal length: a long number or clause then leaves no one-character tail.
fn cut(
    script: &str,
    start: usize,
    end: usize,
    budget: usize,
    tokens: usize,
    count: &mut dyn FnMut(&str) -> Result<usize>,
) -> Result<(String, usize, usize)> {
    let offsets: Vec<usize> = script[start..end]
        .char_indices()
        .map(|(i, _)| start + i)
        .skip(1)
        .collect();
    let mut best = match offsets.first() {
        Some(&first) => first,
        None => return Ok((script[start..end].to_owned(), tokens, end)),
    };
    let target = tokens.div_ceil(tokens.div_ceil(budget));
    let (mut low, mut high) = (0, offsets.len());
    while low < high {
        let middle = low + (high - low) / 2;
        if count(script[start..offsets[middle]].trim_end())? <= target {
            best = offsets[middle];
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    let text = script[start..best].trim_end().to_owned();
    Ok((
        text.clone(),
        count(&text)?,
        after_whitespace(script, best, end),
    ))
}

fn after_whitespace(script: &str, from: usize, end: usize) -> usize {
    let rest = &script[from..end];
    from + rest.len() - rest.trim_start().len()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Strength {
    Word,
    LineBreak,
    Clause,
    Sentence,
}

const SENTENCE_END: &[char] = &['。', '！', '？', '；', '…'];
const CLOSING: &[char] = &[
    '”', '’', '」', '』', '）', '》', '〉', '】', '〕', ')', ']', '}', '"', '\'',
];

/// Byte offsets after which a speech text may end, each with the strength of
/// the break. Whitespace after an offset belongs to the next speech text.
fn boundaries(script: &str) -> Vec<(usize, Strength)> {
    let chars: Vec<(usize, char)> = script.char_indices().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        // Brackets and tags keep their contents together, vocal events included.
        if let Some(close) = enclosed(&chars, i) {
            i = close + 1;
            continue;
        }
        let Some((strength, past)) = break_after(&chars, i) else {
            i += 1;
            continue;
        };
        let mut end = past;
        while chars
            .get(end)
            .is_some_and(|x| SENTENCE_END.contains(&x.1) || CLOSING.contains(&x.1))
        {
            end += 1;
        }
        out.push((offset_from(&chars, end), strength));
        i = end;
    }
    out
}

fn offset_from(chars: &[(usize, char)], end: usize) -> usize {
    let (offset, c) = chars[end - 1];
    offset + c.len_utf8()
}

/// Whether the character at `i` can end a speech text, and the index just past
/// its punctuation.
fn break_after(chars: &[(usize, char)], i: usize) -> Option<(Strength, usize)> {
    let c = chars[i].1;
    if SENTENCE_END.contains(&c) {
        return Some((Strength::Sentence, i + 1));
    }
    if matches!(c, '，' | '、' | '：') {
        return Some((Strength::Clause, i + 1));
    }
    if c == '—' {
        let run = 1 + chars[i + 1..].iter().take_while(|x| x.1 == '—').count();
        return Some((Strength::Clause, i + run));
    }
    if c == '\n' {
        let run = 1 + chars[i + 1..]
            .iter()
            .take_while(|x| x.1 == '\n' || x.1 == '\r')
            .count();
        let lines = chars[i..i + run].iter().filter(|x| x.1 == '\n').count();
        let strength = if lines >= 2 {
            Strength::Sentence
        } else {
            Strength::LineBreak
        };
        return Some((strength, i + run));
    }
    if c.is_whitespace() {
        let run = 1 + chars[i + 1..]
            .iter()
            .take_while(|x| x.1.is_whitespace() && x.1 != '\n')
            .count();
        return Some((Strength::Word, i + run));
    }
    if c == ',' || c == ':' {
        // English clause punctuation breaks only before the space after it, so
        // "1,000", "10:30" and URLs stay whole.
        return chars
            .get(i + 1)
            .filter(|x| x.1.is_whitespace())
            .map(|_| (Strength::Clause, i + 1));
    }
    if !".!?;".contains(c) {
        return None;
    }
    let run = 1 + chars[i + 1..]
        .iter()
        .take_while(|x| ".!?;".contains(x.1))
        .count();
    let past = i + run;
    match chars.get(past).map(|x| x.1) {
        None => Some((Strength::Sentence, past)),
        // "3.14", "U.S.A" and "x!y" stay whole.
        Some(n) if n.is_ascii_alphanumeric() => None,
        Some(n) if n.is_whitespace() => match next_word(chars, past) {
            // "e.g. next" continues its sentence.
            Some(w) if c == '.' && w.is_ascii_lowercase() => None,
            _ => Some((Strength::Sentence, past)),
        },
        Some(_) => Some((Strength::Sentence, past)),
    }
}

fn next_word(chars: &[(usize, char)], from: usize) -> Option<char> {
    chars[from..]
        .iter()
        .map(|x| x.1)
        .find(|c| !c.is_whitespace())
}

/// The index of the bracket or tag closer pairing with the opener at `i`.
fn enclosed(chars: &[(usize, char)], i: usize) -> Option<usize> {
    let c = chars[i].1;
    let (open, close, max) = match c {
        '(' => ('(', ')', 64),
        '[' => ('[', ']', 64),
        // A tag such as <ins_bos>; a bare "<" in prose has no closer nearby.
        '<' => ('<', '>', 32),
        _ => return None,
    };
    if c == '<' && !chars.get(i + 1).is_some_and(|x| x.1.is_ascii_alphabetic()) {
        return None;
    }
    let mut depth = 0usize;
    for (j, x) in chars.iter().enumerate().skip(i).take(max) {
        if x.1 == '\n' {
            return None;
        }
        if x.1 == open {
            depth += 1;
        } else if x.1 == close {
            depth -= 1;
            if depth == 0 {
                return Some(j);
            }
        }
    }
    None
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn characters(text: &str) -> Result<usize> {
        Ok(text.chars().filter(|c| !c.is_whitespace()).count())
    }

    fn budget(rest: usize) -> ScriptBudget {
        ScriptBudget { opening: 0, rest }
    }

    fn split_by(
        script: &str,
        budget: ScriptBudget,
        count: &mut dyn FnMut(&str) -> Result<usize>,
    ) -> Vec<String> {
        split(script, budget, count)
            .unwrap()
            .into_iter()
            .map(|text| text.text)
            .collect()
    }

    fn keeps_every_word(script: &str, texts: &[String]) {
        let words = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
        assert_eq!(words(&texts.concat()), words(script));
    }

    #[test]
    fn a_script_within_budget_is_one_unchanged_speech_text() {
        let script = "  你好，今天天气很好。\n";
        let texts = split(script, ScriptBudget::default(), &mut characters).unwrap();
        assert_eq!(
            texts,
            vec![SpeechText {
                text: script.to_owned(),
                tokens: 10,
            }]
        );
    }

    #[test]
    fn packs_sentences_up_to_the_budget() {
        let script = "一二。三四。五六。七八九十。";
        let texts = split_by(script, budget(8), &mut characters);
        assert_eq!(texts, ["一二。三四。", "五六。七八九十。"]);
        keeps_every_word(script, &texts);
        for text in split(script, budget(8), &mut characters).unwrap() {
            assert!(text.tokens <= 8);
        }
    }

    #[test]
    fn a_long_sentence_falls_back_to_clauses_then_characters() {
        let script = "一二三四五六七八九十，一二三。";
        let texts = split_by(script, budget(6), &mut characters);
        assert_eq!(texts, ["一二三四五", "六七八九十，", "一二三。"]);
        keeps_every_word(script, &texts);
    }

    #[test]
    fn an_unbreakable_run_is_cut_into_even_pieces() {
        let texts = split_by("一二三四五六七八九十甲乙", budget(10), &mut characters);
        assert_eq!(texts, ["一二三四五六", "七八九十甲乙"]);
    }

    #[test]
    fn decimals_abbreviations_urls_and_brackets_stay_whole() {
        let script = "值3.14。(e.g. 好)。网址a.org/b。共1,000个。10:30开始。";
        let texts = split_by(script, budget(10), &mut characters);
        assert_eq!(
            texts,
            [
                "值3.14。",
                "(e.g. 好)。",
                "网址a.org/b。",
                "共1,000个。",
                "10:30开始。"
            ]
        );
        keeps_every_word(script, &texts);
    }

    #[test]
    fn closing_quotes_stay_with_their_sentence() {
        let texts = split_by("「你好。」“再见。”好的。", budget(5), &mut characters);
        assert_eq!(texts, ["「你好。」", "“再见。”", "好的。"]);
    }

    #[test]
    fn blank_lines_end_a_speech_text_and_single_breaks_only_help() {
        let script = "第一行没有标点\n第二行也没有\n\n第三段。";
        let texts = split_by(script, budget(8), &mut characters);
        assert_eq!(texts, ["第一行没有标点", "第二行也没有", "第三段。"]);
        keeps_every_word(script, &texts);
    }

    #[test]
    fn english_breaks_at_sentences_then_spaces() {
        let mut words = |text: &str| Ok(text.split_whitespace().count());
        let script = "One sentence here.  Two sentence here!\nWe counted 1,000 people here. Then at 10:30 we left, tired.\n";
        let texts = split_by(script, budget(5), &mut words);
        assert_eq!(
            texts,
            [
                "One sentence here.",
                "Two sentence here!",
                "We counted 1,000 people here.",
                "Then at 10:30 we left,",
                "tired."
            ]
        );
        keeps_every_word(script, &texts);
    }

    #[test]
    fn the_opening_keeps_its_first_sentence_and_stays_short() {
        let script = "六字句子六字。短。后面的句子很长很长很长。";
        let texts = split_by(
            script,
            ScriptBudget {
                opening: 4,
                rest: 10,
            },
            &mut characters,
        );
        assert_eq!(
            texts,
            ["六字句子六字。", "短。", "后面的句子很", "长很长很长。"]
        );
        keeps_every_word(script, &texts);
    }

    #[test]
    fn the_opening_takes_whole_sentences_up_to_its_budget() {
        let script = "好。再来一句。后面。还有很多很多很多很多很多的话。";
        let texts = split_by(
            script,
            ScriptBudget {
                opening: 7,
                rest: 8,
            },
            &mut characters,
        );
        assert_eq!(
            texts,
            [
                "好。再来一句。",
                "后面。",
                "还有很多很多很多",
                "很多很多的话。"
            ]
        );
        keeps_every_word(script, &texts);
    }

    #[test]
    fn whitespace_only_scripts_and_zero_budgets_are_rejected() {
        let error = split(
            "                                      ",
            budget(8),
            &mut |text| Ok(text.len()),
        )
        .unwrap_err();
        assert!(error.to_string().contains("请输入要合成的文本"));
        let error = split("一。", budget(0), &mut characters).unwrap_err();
        assert!(error.to_string().contains("budget"));
    }

    #[test]
    fn counting_errors_propagate() {
        let error = split("一二三四五六七八九。", budget(2), &mut |_| {
            Err(Error::Config("tokenizer".into()))
        })
        .unwrap_err();
        assert!(error.to_string().contains("tokenizer"));
    }

    #[test]
    fn cuts_never_split_a_character() {
        let script = "🙂等效😀求解🙂等效😀求解🙂等效。";
        let texts = split_by(script, budget(4), &mut characters);
        assert_eq!(texts, ["🙂等效😀", "求解🙂等", "效😀求解", "🙂等效。"]);
        keeps_every_word(script, &texts);
    }

    /// `models/Breeze-TTS-2/tokenizer.json`, relative to the crate directory.
    pub(crate) fn checkpoint_tokenizer() -> Option<tokenizers::Tokenizer> {
        tokenizers::Tokenizer::from_file("../../models/Breeze-TTS-2/tokenizer.json").ok()
    }

    #[test]
    #[ignore = "needs the Breeze-TTS-2 tokenizer; skips when it is absent"]
    fn the_real_tokenizer_counts_a_speech_text_as_its_share_of_the_prompt() {
        let Some(tokenizer) = checkpoint_tokenizer() else {
            eprintln!("skipped: models/Breeze-TTS-2/tokenizer.json is absent");
            return;
        };
        let splitter = Splitter::new(&tokenizer).unwrap();
        for text in [
            "你好，今天天气很好。",
            "The quick brown fox jumps over the lazy dog. It was a sunny day.",
            "值3.14。(e.g. 好)。网址a.org/b。共1,000个。10:30开始。",
            "  你好，今天天气很好。\n",
            "行内声音事件：(laugh) 然后[笑]继续说。",
        ] {
            let tokens = render(&tokenizer, text).unwrap().len() - splitter.overhead;
            let share = |prefix: &str| {
                render(&tokenizer, &format!("{prefix}{text}"))
                    .unwrap()
                    .len()
                    - render(&tokenizer, prefix).unwrap().len()
            };
            assert_eq!(tokens, share("[S0]"));
            assert_eq!(tokens, share("[S0]<ins_bos>温柔的女声<ins_eos>"));
        }
    }

    #[test]
    #[ignore = "needs the Breeze-TTS-2 tokenizer; skips when it is absent"]
    fn the_real_tokenizer_splits_a_long_script_within_its_budget() {
        let Some(tokenizer) = checkpoint_tokenizer() else {
            eprintln!("skipped: models/Breeze-TTS-2/tokenizer.json is absent");
            return;
        };
        let splitter = Splitter::new(&tokenizer).unwrap();
        let paragraph = "人工智能正在改变我们的工作方式。它可以帮助医生更快地阅读影像，也可以帮助工程师检查代码。当然，新的工具也带来新的问题。例如，我们如何确认答案是对的？又如何保护每个人的隐私？这些问题没有简单的答案。我们需要更多的讨论，也需要更清楚的规则。让我们从一个小例子开始。假设你要写一封邮件，说明项目推迟了三天。你希望语气礼貌，但不啰嗦。模型可以给出三个版本，你再挑选最合适的那个。它不会替你承担责任。决定仍然是你做的。";
        let one = splitter.split(paragraph, ScriptBudget::default()).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].text, paragraph);
        assert!(one[0].tokens <= TEXT_TOKENS);

        let script = paragraph.repeat(6);
        let texts = splitter.split(&script, ScriptBudget::default()).unwrap();
        let tokens: Vec<_> = texts.iter().map(|text| text.tokens).collect();
        eprintln!("speech text tokens: {tokens:?}");
        assert!(texts.len() >= 5);
        assert!(tokens[0] <= OPENING_TEXT_TOKENS);
        assert!(tokens.iter().all(|tokens| *tokens <= TEXT_TOKENS));
        assert!(
            texts
                .iter()
                .all(|text| text.text.ends_with(['。', '？', '！']))
        );
        let spoken: String = texts.iter().map(|text| text.text.as_str()).collect();
        assert_eq!(spoken, script);
    }
}
