//! Desktop speech-enhancement session.
//!
//! Callers that leave the session disabled never sample. Command-line and HTTP
//! synthesis do not use this module. A draw is one enhancer answer; tests pass
//! answers directly and never load a checkpoint.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

/// Same default MiniCPM-V checkpoint the image prompt rewrite uses.
pub const DEFAULT_CHECKPOINT: &str = "models/MiniCPM-V-4.7-35B-A3B-int4";

const EVENTS_EN: [&str; 4] = ["(laugh)", "(cough)", "(clears throat)", "(sigh)"];
const EVENTS_ZH: [&str; 4] = ["[笑]", "[咳嗽]", "[清嗓子]", "[叹气]"];
const TEMPLATE_TOKENS: [&str; 13] = [
    "[S0]",
    "[S1]",
    "[S2]",
    "[S3]",
    "[S4]",
    "[S5]",
    "[S6]",
    "[S7]",
    "[S8]",
    "[S9]",
    "<ins_bos>",
    "<ins_eos>",
    "<|AUDIO|>",
];

/// The words to speak and the optional voice instruction. Reference transcripts
/// are not part of a request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Source {
    pub speech_text: String,
    pub voice_instruction: String,
}

/// One rewritten pair. Synthesis consumes only these two strings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pair {
    pub speech_text: String,
    pub voice_instruction: String,
}

/// One enhancer draw. `Err` is a failed call, not a pair.
#[derive(Clone, Debug)]
pub struct Sample {
    pub answer: Result<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Consideration {
    Speak(Pair),
    Choose {
        alternatives: [Pair; 3],
        enhancer_prompt: String,
    },
    Failed(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeechChoice {
    pub alternatives: [Pair; 3],
    pub winner: usize,
    pub enhancer_prompt: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Hearing {
    alternatives: [Pair; 3],
    heard: [bool; 3],
    enhancer_prompt: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Session {
    #[serde(default)]
    enabled: bool,
    #[serde(default = "default_enhancer_prompt")]
    prompt: String,
    #[serde(default)]
    previous: Option<String>,
    #[serde(default)]
    choices: Vec<SpeechChoice>,
    #[serde(default)]
    draft: Option<String>,
    #[serde(default)]
    sample_cursor: u64,
    #[serde(skip)]
    hearing: Option<Hearing>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftRequest {
    pub system: String,
    pub user: String,
}

pub fn default_enhancer_prompt() -> String {
    "你是语音合成前的改写器。只输出一个 JSON 对象，键为 speech_text 和 voice_instruction。\
speech_text 保持用户朗读文字的原词和词序，只能插入官方行内声音事件：英文正文用 (laugh)、(cough)、(clears throat)、(sigh)，中文正文用 [笑]、[咳嗽]、[清嗓子]、[叹气]。\
voice_instruction 用与朗读文字相同的语言扩写声音描述；用户没写时必须是空字符串，用户写了就必须仍然非空。\
不要输出 [S0] 到 [S9]、<ins_bos>、<ins_eos>、<|AUDIO|>、<|audio_eos|>。不要编造新的叙述。"
        .into()
}

pub fn draft_system_prompt() -> String {
    "你根据历次语音选择改写增强提示。只输出下一版增强提示全文，不要解释。\
新提示仍要求：朗读文字保持原词和词序，只插入官方行内声音事件；声音描述与朗读文字同语言；空描述保持为空。\
不要要求模型改写参考文字，也不要把朗读文字扩成另一段叙述。".into()
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    pub fn new() -> Self {
        Self {
            enabled: false,
            prompt: default_enhancer_prompt(),
            previous: None,
            choices: Vec::new(),
            draft: None,
            sample_cursor: 0,
            hearing: None,
        }
    }

    pub fn load(bytes: &[u8]) -> Self {
        serde_json::from_slice(bytes).unwrap_or_else(|_| Self::new())
    }

    pub fn store(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn enhancer_prompt(&self) -> &str {
        &self.prompt
    }

    pub fn previous_prompt(&self) -> Option<&str> {
        self.previous.as_deref()
    }

    pub fn choices(&self) -> &[SpeechChoice] {
        &self.choices
    }

    pub fn draft(&self) -> Option<&str> {
        self.draft.as_deref()
    }

    pub fn can_choose(&self) -> bool {
        self.hearing
            .as_ref()
            .is_some_and(|hearing| hearing.heard.iter().all(|heard| *heard))
    }

    pub fn hearing(&self) -> bool {
        self.hearing.is_some()
    }

    /// Three independent enhancer seeds. Disabled sessions are not sampled by
    /// the desktop; this cursor only advances when a caller asks for seeds.
    pub fn next_sample_seeds(&mut self) -> [u64; 3] {
        let base = self.sample_cursor;
        self.sample_cursor = self.sample_cursor.wrapping_add(3);
        [base, base.wrapping_add(1), base.wrapping_add(2)]
    }

    pub fn consider<F>(
        &self,
        source: &Source,
        samples: [Sample; 3],
        within_budget: F,
    ) -> Consideration
    where
        F: Fn(&str, &str) -> bool,
    {
        consider(&self.prompt, source, samples, within_budget)
    }

    /// Replaces an unfinished hearing and does not record a speech choice.
    pub fn start_hearing(&mut self, alternatives: [Pair; 3], enhancer_prompt: &str) {
        self.hearing = Some(Hearing {
            alternatives,
            heard: [false; 3],
            enhancer_prompt: enhancer_prompt.to_string(),
        });
    }

    pub fn mark_heard(&mut self, index: usize) -> Result<(), &'static str> {
        let hearing = self.hearing.as_mut().ok_or("没有进行中的盲选")?;
        let heard = hearing.heard.get_mut(index).ok_or("没有这一段")?;
        *heard = true;
        Ok(())
    }

    pub fn choose(&mut self, index: usize) -> Result<Pair, &'static str> {
        let hearing = self.hearing.as_ref().ok_or("没有进行中的盲选")?;
        if hearing.heard.iter().any(|heard| !heard) {
            return Err("三段都播放到结束才能选择");
        }
        if index > 2 {
            return Err("没有这一段");
        }
        let choice = SpeechChoice {
            alternatives: hearing.alternatives.clone(),
            winner: index,
            enhancer_prompt: hearing.enhancer_prompt.clone(),
        };
        let winner = hearing.alternatives[index].clone();
        self.choices.push(choice);
        self.hearing = None;
        Ok(winner)
    }

    pub fn abandon(&mut self) {
        self.hearing = None;
    }

    pub fn draft_request(&self) -> Result<DraftRequest, &'static str> {
        if self.choices.is_empty() {
            return Err("至少有一条语音选择才能起草");
        }
        let mut user = String::new();
        user.push_str("当前增强提示：\n");
        user.push_str(&self.prompt);
        user.push('\n');
        for (index, choice) in self.choices.iter().enumerate() {
            user.push_str(&format!("\n语音选择 {}，产生它的增强提示：\n", index + 1));
            user.push_str(&choice.enhancer_prompt);
            for (label, pair) in ["甲", "乙", "丙"].into_iter().zip(&choice.alternatives) {
                user.push_str(&format!(
                    "\n{label}\n朗读文字：{}\n声音描述：{}\n",
                    pair.speech_text, pair.voice_instruction
                ));
            }
            user.push_str(&format!("选中：{}\n", ["甲", "乙", "丙"][choice.winner]));
        }
        Ok(DraftRequest {
            system: draft_system_prompt(),
            user,
        })
    }

    pub fn stage_draft(&mut self, text: String) {
        self.draft = Some(text);
    }

    pub fn accept_displayed(&mut self, displayed: &str) -> Result<(), &'static str> {
        if self.draft.is_none() {
            return Err("没有可接受的草稿");
        }
        self.previous = Some(std::mem::replace(&mut self.prompt, displayed.to_string()));
        self.draft = None;
        Ok(())
    }

    pub fn discard_draft(&mut self) {
        self.draft = None;
    }

    pub fn restore_prompt(&mut self) -> Result<(), &'static str> {
        let previous = self.previous.take().ok_or("没有可恢复的上一版增强提示")?;
        self.prompt = previous;
        Ok(())
    }
}

pub fn consider<F>(
    enhancer_prompt: &str,
    source: &Source,
    samples: [Sample; 3],
    within_budget: F,
) -> Consideration
where
    F: Fn(&str, &str) -> bool,
{
    let draws: Vec<Result<Pair, String>> = samples
        .into_iter()
        .map(|sample| validate(source, sample, &within_budget))
        .collect();
    let valid: Vec<Pair> = draws
        .iter()
        .filter_map(|draw| draw.as_ref().ok())
        .cloned()
        .collect();
    if valid.len() != 3 {
        let reasons: Vec<_> = draws
            .iter()
            .filter_map(|draw| draw.as_ref().err().map(String::as_str))
            .collect();
        return Consideration::Failed(format!("有效改写不足三份。{}", reasons.join(" ")));
    }
    let mut distinct: Vec<Pair> = Vec::new();
    for pair in &valid {
        if !distinct.contains(pair) {
            distinct.push(pair.clone());
        }
    }
    if distinct.len() == 1 {
        return Consideration::Speak(distinct.remove(0));
    }
    if distinct.len() < 3 {
        return Consideration::Failed("只有两种不同结果，无法盲选".into());
    }
    Consideration::Choose {
        alternatives: [valid[0].clone(), valid[1].clone(), valid[2].clone()],
        enhancer_prompt: enhancer_prompt.to_string(),
    }
}

pub fn sample_user_message(source: &Source) -> String {
    format!(
        "朗读文字：\n{}\n\n声音描述：\n{}",
        source.speech_text,
        source.voice_instruction.trim()
    )
}

/// Read-only check. A missing directory is an error and is not created.
pub fn checkpoint_available(dir: &Path) -> Result<(), String> {
    if !dir.is_dir() {
        return Err(format!("找不到 35B 检查点：{}", dir.display()));
    }
    for name in ["config.json", "tokenizer.json"] {
        if !dir.join(name).is_file() {
            return Err(format!("35B 检查点缺少 {name}：{}", dir.display()));
        }
    }
    Ok(())
}

fn validate<F>(source: &Source, sample: Sample, within_budget: &F) -> Result<Pair, String>
where
    F: Fn(&str, &str) -> bool,
{
    let answer = sample.answer?;
    let value = extract_pair(&answer).ok_or("一次采样得不到朗读文字和声音描述")?;
    let speech_text = value
        .get("speech_text")
        .and_then(Value::as_str)
        .ok_or("一次采样得不到朗读文字和声音描述")?
        .to_string();
    let voice_instruction = value
        .get("voice_instruction")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if let Some(token) = template_token(&speech_text).or_else(|| template_token(&voice_instruction))
    {
        return Err(format!("改写含有模板记号 {token}"));
    }
    if !preserves_speech(&source.speech_text, &speech_text) {
        return Err("改写改变了朗读文字".into());
    }
    let source_instruction = source.voice_instruction.trim();
    if source_instruction.is_empty() {
        if !voice_instruction.is_empty() {
            return Err("空的声音描述被写成了内容".into());
        }
    } else if voice_instruction.is_empty() {
        return Err("声音描述被清空".into());
    } else if !same_language(&source.speech_text, &voice_instruction) {
        return Err("声音描述的语言与朗读文字不一致".into());
    }
    if !within_budget(&speech_text, &voice_instruction) {
        return Err("超过文本 token 上限".into());
    }
    Ok(Pair {
        speech_text,
        voice_instruction,
    })
}

fn template_token(text: &str) -> Option<&'static str> {
    TEMPLATE_TOKENS
        .into_iter()
        .find(|token| text.contains(token))
        .or_else(|| text.contains("<|audio_eos|>").then_some("<|audio_eos|>"))
}

fn preserves_speech(original: &str, rewritten: &str) -> bool {
    let events = allowed_events(original);
    let mut pending = original;
    let mut rest = rewritten;
    while !pending.is_empty() || !rest.is_empty() {
        if let Some(event) = events.iter().copied().find(|event| rest.starts_with(event)) {
            if pending.starts_with(event) {
                pending = &pending[event.len()..];
            }
            rest = &rest[event.len()..];
            continue;
        }
        let Some(next) = pending.chars().next() else {
            return false;
        };
        let Some(got) = rest.chars().next() else {
            return false;
        };
        if next != got {
            return false;
        }
        pending = &pending[next.len_utf8()..];
        rest = &rest[got.len_utf8()..];
    }
    true
}

fn allowed_events(speech_text: &str) -> Vec<&'static str> {
    let letters = script(speech_text);
    match (letters.han && !letters.latin, letters.latin && !letters.han) {
        (true, false) => EVENTS_ZH.to_vec(),
        (false, true) => EVENTS_EN.to_vec(),
        _ => {
            let mut events = EVENTS_ZH.to_vec();
            events.extend(EVENTS_EN);
            events
        }
    }
}

fn same_language(speech_text: &str, voice_instruction: &str) -> bool {
    let speech = script(speech_text);
    let instruction = script(voice_instruction);
    match (speech.han && !speech.latin, speech.latin && !speech.han) {
        (true, false) => {
            instruction.han && !instruction.latin && !instruction.kana && !instruction.hangul
        }
        (false, true) => {
            instruction.latin && !instruction.han && !instruction.kana && !instruction.hangul
        }
        _ => true,
    }
}

struct Letters {
    han: bool,
    latin: bool,
    kana: bool,
    hangul: bool,
}

fn script(text: &str) -> Letters {
    let mut letters = Letters {
        han: false,
        latin: false,
        kana: false,
        hangul: false,
    };
    for ch in text.chars() {
        let code = ch as u32;
        letters.han |= matches!(code, 0x3400..=0x9fff);
        letters.latin |= ch.is_ascii_alphabetic();
        letters.kana |= matches!(code, 0x3040..=0x30ff);
        letters.hangul |= matches!(code, 0xac00..=0xd7af);
    }
    letters
}

pub fn without_thinking(text: &str) -> String {
    strip_think(text)
}

fn extract_pair(answer: &str) -> Option<Value> {
    let text = strip_think(answer);
    let bytes = text.as_bytes();
    let mut index = text.len();
    while index > 0 {
        index -= 1;
        if bytes[index] != b'{' || !text.is_char_boundary(index) {
            continue;
        }
        let Some(end) = balanced_object(&text, index) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(&text[index..end]) else {
            continue;
        };
        if value.get("speech_text").and_then(Value::as_str).is_some()
            && value.get("voice_instruction").is_some()
        {
            return Some(value);
        }
    }
    None
}

fn strip_think(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        out.push_str(&rest[..start]);
        let after = &rest[start + "<think>".len()..];
        if let Some(end) = after.find("</think>") {
            rest = &after[end + "</think>".len()..];
        } else {
            rest = "";
        }
    }
    out.push_str(rest);
    out
}

fn balanced_object(text: &str, start: usize) -> Option<usize> {
    let mut depth = 0;
    let mut chars = text[start..].char_indices();
    let mut in_string = false;
    let mut escape = false;
    while let Some((offset, ch)) = chars.next() {
        if in_string {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(start + offset + ch.len_utf8());
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> Source {
        Source {
            speech_text: "明天上午九点开会".into(),
            voice_instruction: "年轻女性，温柔".into(),
        }
    }

    fn sample(speech: &str, instruction: &str) -> Sample {
        Sample {
            answer: Ok(format!(
                r#"{{"speech_text":{speech:?},"voice_instruction":{instruction:?}}}"#
            )),
        }
    }

    fn always(_speech: &str, _instruction: &str) -> bool {
        true
    }

    fn consider_three(samples: [Sample; 3]) -> Consideration {
        consider(&default_enhancer_prompt(), &source(), samples, always)
    }

    #[test]
    fn switch_starts_off_and_is_remembered() {
        let mut session = Session::new();
        assert!(!session.enabled());
        session.set_enabled(true);
        let restored = Session::load(&session.store());
        assert!(restored.enabled());
        assert_eq!(restored.enhancer_prompt(), session.enhancer_prompt());
    }

    #[test]
    fn meeting_text_keeps_its_words_and_may_gain_a_chinese_event() {
        let rewritten = sample("明天上午九点开会[轻笑]", "一位年轻女性，声音温柔");
        let result = consider_three([rewritten.clone(), rewritten.clone(), rewritten]);
        assert!(matches!(result, Consideration::Failed(reason) if reason.contains("朗读文字")));

        let kept = sample("明天上午九点开会[笑]", "一位年轻女性，声音温柔");
        let result = consider_three([kept.clone(), kept.clone(), kept]);
        match result {
            Consideration::Speak(pair) => {
                assert_eq!(pair.speech_text, "明天上午九点开会[笑]");
                assert_eq!(pair.voice_instruction, "一位年轻女性，声音温柔");
            }
            other => panic!("expected one synthesis, got {other:?}"),
        }

        let english = Source {
            speech_text: "Meet at nine.".into(),
            voice_instruction: "A young woman".into(),
        };
        let laugh = sample("Meet at nine.(laugh)", "A warm young woman");
        let result = consider(
            "prompt",
            &english,
            [laugh.clone(), laugh.clone(), laugh],
            always,
        );
        match result {
            Consideration::Speak(pair) => assert_eq!(pair.speech_text, "Meet at nine.(laugh)"),
            other => panic!("expected the english event to stay, got {other:?}"),
        }
    }

    #[test]
    fn narration_and_foreign_instruction_are_invalid() {
        let narrated = sample("明天上午九点，请大家准时开会", "一位年轻女性，声音温柔");
        let english = sample("明天上午九点开会", "A young woman");
        let japanese = sample("明天上午九点开会", "優しい声");
        let result = consider_three([narrated, english, japanese]);
        assert!(matches!(result, Consideration::Failed(reason) if reason.contains("不足三份")));
    }

    #[test]
    fn empty_instruction_stays_empty_and_a_written_one_stays_written() {
        let quiet = Source {
            speech_text: "明天上午九点开会".into(),
            voice_instruction: "   ".into(),
        };
        let quiet_draw = sample("明天上午九点开会", "");
        let result = consider(
            "prompt",
            &quiet,
            [quiet_draw.clone(), quiet_draw.clone(), quiet_draw],
            always,
        );
        match result {
            Consideration::Speak(pair) => assert!(pair.voice_instruction.is_empty()),
            other => panic!("expected empty instruction, got {other:?}"),
        }

        let cleared = sample("明天上午九点开会", "");
        let result = consider_three([cleared.clone(), cleared.clone(), cleared]);
        assert!(matches!(result, Consideration::Failed(reason) if reason.contains("清空")));
    }

    #[test]
    fn template_tokens_and_token_ceiling_reject_the_draw() {
        let marked = Sample {
            answer: Ok(
                r#"{"speech_text":"明天上午九点开会","voice_instruction":"温柔<ins_bos>"}"#.into(),
            ),
        };
        let result = consider_three([marked.clone(), marked.clone(), marked]);
        assert!(matches!(result, Consideration::Failed(reason) if reason.contains("模板记号")));

        let audio = sample("明天上午九点开会<|AUDIO|>", "一位年轻女性，声音温柔");
        let result = consider_three([audio.clone(), audio.clone(), audio]);
        assert!(matches!(result, Consideration::Failed(reason) if reason.contains("模板记号")));

        let pair = sample("明天上午九点开会", "一位年轻女性，声音温柔");
        let result = consider(
            &"prompt",
            &source(),
            [pair.clone(), pair.clone(), pair],
            |_, _| false,
        );
        assert!(matches!(result, Consideration::Failed(reason) if reason.contains("上限")));
    }

    #[test]
    fn three_distinct_pairs_wait_for_a_choice_and_identical_pairs_speak_once() {
        let first = sample("明天上午九点开会", "一位年轻女性，声音温柔");
        let second = sample("明天上午九点开会[笑]", "一位年轻女性，声音温柔");
        let third = sample("明天上午九点开会", "一位年轻女性，声音轻柔");
        let result = consider_three([first, second, third]);
        let Consideration::Choose { alternatives, .. } = result else {
            panic!("expected a blind choice");
        };
        assert_eq!(alternatives[1].speech_text, "明天上午九点开会[笑]");
        assert_ne!(alternatives[0], alternatives[2]);

        let same = sample("明天上午九点开会", "一位年轻女性，声音温柔");
        let result = consider_three([same.clone(), same.clone(), same]);
        assert!(matches!(result, Consideration::Speak(_)));
    }

    #[test]
    fn two_distinct_results_and_a_failed_call_do_not_speak_the_original() {
        let one = sample("明天上午九点开会", "一位年轻女性，声音温柔");
        let two = sample("明天上午九点开会[笑]", "一位年轻女性，声音温柔");
        let result = consider_three([one.clone(), two, one]);
        assert!(matches!(result, Consideration::Failed(reason) if reason.contains("两种")));

        let broken = Sample {
            answer: Err("找不到 35B 检查点：missing".into()),
        };
        let result = consider_three([
            broken,
            sample("明天上午九点开会", "年轻女性，温柔"),
            sample("明天上午九点开会", "年轻女性，温柔"),
        ]);
        match result {
            Consideration::Failed(reason) => {
                assert!(reason.contains("不足三份"));
                assert!(reason.contains("找不到 35B"));
            }
            Consideration::Speak(pair) => panic!("fell back to synthesis: {pair:?}"),
            Consideration::Choose { .. } => panic!("opened a panel without three draws"),
        }
    }

    #[test]
    fn a_choice_keeps_the_prompt_that_produced_it_and_an_abandoned_panel_does_not() {
        let mut session = Session::new();
        let alternatives = [
            Pair {
                speech_text: "明天上午九点开会".into(),
                voice_instruction: "甲".into(),
            },
            Pair {
                speech_text: "明天上午九点开会[笑]".into(),
                voice_instruction: "乙".into(),
            },
            Pair {
                speech_text: "明天上午九点开会".into(),
                voice_instruction: "丙".into(),
            },
        ];
        session.start_hearing(alternatives.clone(), "第一份提示");
        session.start_hearing(alternatives.clone(), "第二份提示");
        assert!(session.choices().is_empty());
        assert!(session.choose(0).is_err());
        assert!(session.choices().is_empty());
        for index in 0..3 {
            session.mark_heard(index).unwrap();
        }
        let winner = session.choose(1).unwrap();
        assert_eq!(winner.voice_instruction, "乙");
        assert_eq!(session.choices()[0].enhancer_prompt, "第二份提示");
        assert_eq!(session.choices()[0].winner, 1);
        session.abandon();
        assert_eq!(session.choices().len(), 1);
    }

    #[test]
    fn drafting_sees_every_choice_and_acceptance_keeps_the_previous_prompt() {
        let mut session = Session::new();
        assert!(session.draft_request().is_err());
        session.choices.push(SpeechChoice {
            alternatives: [pair("甲"), pair("乙"), pair("丙")],
            winner: 0,
            enhancer_prompt: "旧提示全文".into(),
        });
        let request = session.draft_request().unwrap();
        assert!(request.user.contains("旧提示全文"));
        assert!(request.user.contains("甲"));
        assert!(request.user.contains("选中：甲"));
        let original = session.enhancer_prompt().to_string();
        session.stage_draft("模型草稿".into());
        session.accept_displayed("人改过的提示").unwrap();
        assert_eq!(session.enhancer_prompt(), "人改过的提示");
        assert_eq!(session.previous_prompt(), Some(original.as_str()));
        assert_eq!(session.choices()[0].enhancer_prompt, "旧提示全文");
        session.discard_draft();
        assert_eq!(session.enhancer_prompt(), "人改过的提示");
        session.restore_prompt().unwrap();
        assert_eq!(session.enhancer_prompt(), original);
        assert!(session.restore_prompt().is_err());
        assert!(session.accept_displayed("没有草稿").is_err());
    }

    #[test]
    fn the_sample_message_has_no_reference_transcript_slot() {
        let message = sample_user_message(&source());
        assert!(message.contains("明天上午九点开会"));
        assert!(message.contains("年轻女性，温柔"));
        assert!(!message.contains("参考文字"));
    }

    #[test]
    fn a_missing_checkpoint_is_reported_without_creating_it() {
        let path = std::env::temp_dir().join(format!(
            "omni-enhancement-missing-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&path);
        let error = checkpoint_available(&path).unwrap_err();
        assert!(error.contains("找不到 35B"));
        assert!(!path.exists());
        let mut session = Session::new();
        session.stage_draft("草稿".into());
        session.accept_displayed("新提示").unwrap();
        assert!(!path.exists());
    }

    fn pair(instruction: &str) -> Pair {
        Pair {
            speech_text: "明天上午九点开会".into(),
            voice_instruction: instruction.into(),
        }
    }
}
