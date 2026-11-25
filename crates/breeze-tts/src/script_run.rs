//! One speech script speaks its speech texts in order, one speech synthesis each.
//! The script owns the voice anchor: a later speech text is conditioned on that
//! one reference recording, never on the speech text before it, so nothing
//! accumulates from speech text to speech text.
use crate::reference::ReferenceEncoder;
use crate::*;
use std::time::Instant;

/// A speech script prepared once: its ordered speech texts, the shared
/// sampling options and whether its voice anchor comes from the opening audio.
/// Splitting and that decision happen here, before any synthesis slot or
/// playback estimate asks for them; preparation leaves the position limits of
/// each speech text to the synthesis that runs it.
#[derive(Clone, Debug)]
pub struct PreparedScript {
    /// The script's speech texts in order.
    pub texts: Vec<SpeechText>,
    /// Sampling, limits and voice-conditioning shared by every speech text.
    /// `synthesis.text` stays the whole speech script; the run gives each
    /// speech text its own text.
    pub synthesis: SynthesisOptions,
    /// Record the voice anchor from the opening audio: the script has several
    /// speech texts and no supplied reference recording.
    pub needs_voice_anchor: bool,
}

impl PreparedScript {
    /// The prepared script of already-split speech texts: the same sampling
    /// options, and the anchor decision of their count and reference.
    pub fn new(texts: Vec<SpeechText>, synthesis: SynthesisOptions) -> Self {
        let needs_voice_anchor = synthesis.reference.is_none() && texts.len() > 1;
        Self {
            texts,
            synthesis,
            needs_voice_anchor,
        }
    }
}

impl<B: Backend> Model<B> {
    /// Split a speech script and settle its voice-conditioning once, for every
    /// way the script can be spoken afterwards.
    pub fn prepare_script(&self, options: &ScriptOptions) -> Result<PreparedScript> {
        let texts = self.split_script(&options.synthesis.text, options.budget)?;
        Ok(PreparedScript::new(texts, options.synthesis.clone()))
    }

    /// Speak a prepared speech script. Its speech texts run in the prepared
    /// order with the prepared voice-conditioning; nothing is split again.
    /// Pass the reference encoder whenever `needs_voice_anchor` holds; without
    /// it such a script is refused, because its speech texts would each pick
    /// their own voice.
    pub fn speak_prepared(
        &self,
        prepared: PreparedScript,
        cancel: &AtomicBool,
        encoder: Option<&ReferenceEncoder<B>>,
        mut on_event: impl FnMut(ScriptEvent<'_>) -> Result<()>,
    ) -> Result<ScriptReport> {
        if prepared.needs_voice_anchor && encoder.is_none() {
            return Err(Error::Config(
                "长文本分段合成需要参考音频编码器来固定音色".into(),
            ));
        }
        let mut run = ScriptRun::new(
            prepared.texts,
            prepared.synthesis,
            prepared.needs_voice_anchor,
        );
        while !run.is_finished() {
            if cancel.load(Ordering::Acquire) {
                run.stop();
                break;
            }
            let synthesis = run.next_options().expect("unfinished script");
            let index = run.completed();
            let report = self.synthesize(&synthesis, cancel, |pcm| {
                run.record(pcm);
                on_event(ScriptEvent::Audio(pcm))
            })?;
            let recorded = run.complete(report, cancel, |pcm, cancel| match encoder {
                Some(encoder) => encoder
                    .encode(pcm, cancel, |_| {})
                    .map(|encoded| encoded.map(|encoded| encoded.codes)),
                None => Ok(None),
            })?;
            on_event(ScriptEvent::SpeechText(
                index,
                run.last_report().expect("pushed"),
                !run.is_finished(),
            ))?;
            if let Some(recorded @ 1..) = recorded {
                on_event(ScriptEvent::Anchor(recorded))?;
            }
        }
        Ok(run.report())
    }
}

/// Opening audio needed before it becomes the voice anchor: the shortest
/// reference `docs/breeze-tts.md` recommends, three seconds.
pub const ANCHOR_SAMPLES: usize = 3 * SAMPLE_RATE as usize;

#[derive(Clone, Debug)]
pub struct ScriptOptions {
    /// Sampling, limits and voice-conditioning shared by every speech text.
    /// `synthesis.text` is the whole speech script; `synthesis.max_frames` is
    /// each speech text's own frame ceiling.
    pub synthesis: SynthesisOptions,
    pub budget: ScriptBudget,
}
impl Default for ScriptOptions {
    fn default() -> Self {
        Self {
            synthesis: SynthesisOptions::default(),
            budget: ScriptBudget::default(),
        }
    }
}

/// Where the reference recording shared by a script's speech texts came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorOrigin {
    /// The speaker supplied it.
    Supplied,
    /// It is the script's own opening audio.
    Opening,
    /// The script has one speech text, or nothing followed its opening audio.
    None,
}

#[derive(Debug, serde::Serialize)]
pub struct ScriptReport {
    /// The script's speech texts in order. `speech_texts[i]` is the synthesis
    /// of `texts[i]`, for every speech text that ran.
    pub texts: Vec<SpeechText>,
    /// Every speech synthesis that ran, in order.
    pub speech_texts: Vec<SynthesisReport>,
    pub frames: usize,
    pub audio_seconds: f64,
    pub elapsed_seconds: f64,
    pub first_audio_seconds: Option<f64>,
    pub stopped: bool,
    /// The speech text that reached its frame ceiling. The script stops there,
    /// because its unread words cannot be located in the text.
    pub reached_limit: Option<usize>,
    pub anchor: AnchorOrigin,
    /// Speech texts recorded into the voice anchor.
    pub anchor_speech_texts: usize,
}

/// What a speech script hands its caller, in order.
pub enum ScriptEvent<'a> {
    /// PCM of the speech text now running.
    Audio(&'a [f32]),
    /// A speech synthesis ended, after all of its audio was handed off. The
    /// third field says whether the script actually continues with a next
    /// speech text; it is false once the script ends here, was stopped, or the
    /// speech text reached its frame ceiling.
    SpeechText(usize, &'a SynthesisReport, bool),
    /// The script now has its voice anchor, recorded from the first `n` speech texts.
    Anchor(usize),
}

/// The state of one speech script between its speech syntheses.
pub struct ScriptRun {
    texts: Vec<SpeechText>,
    base: SynthesisOptions,
    derive: bool,
    recording: Vec<f32>,
    anchor: Option<ReferenceAudio>,
    reports: Vec<SynthesisReport>,
    anchor_speech_texts: usize,
    stopped: bool,
    reached_limit: Option<usize>,
    started: Instant,
}
impl ScriptRun {
    /// `derive` records the voice anchor from the opening audio. Callers only
    /// set it when the script has several speech texts and no supplied reference.
    pub fn new(texts: Vec<SpeechText>, base: SynthesisOptions, derive: bool) -> Self {
        Self {
            texts,
            base,
            derive,
            recording: Vec::new(),
            anchor: None,
            reports: Vec::new(),
            anchor_speech_texts: 0,
            stopped: false,
            reached_limit: None,
            started: Instant::now(),
        }
    }
    pub fn speech_texts(&self) -> usize {
        self.texts.len()
    }
    pub fn completed(&self) -> usize {
        self.reports.len()
    }
    pub fn is_finished(&self) -> bool {
        self.stopped || self.reached_limit.is_some() || self.reports.len() == self.texts.len()
    }
    /// End the script as stopped, before the next speech text starts.
    pub fn stop(&mut self) {
        self.stopped = true;
    }
    /// Options of the next speech text, or `None` once the script is over.
    pub fn next_options(&mut self) -> Option<SynthesisOptions> {
        if self.is_finished() {
            return None;
        }
        let mut options = self.base.clone();
        options.text = self.texts[self.reports.len()].text.clone();
        options.reference = self.base.reference.clone().or_else(|| self.anchor.clone());
        Some(options)
    }
    /// Keep the audio handed off for the speech text now running.
    pub fn record(&mut self, pcm: &[f32]) {
        if self.derive && self.anchor.is_none() {
            self.recording.extend_from_slice(pcm);
        }
    }
    pub fn last_report(&self) -> Option<&SynthesisReport> {
        self.reports.last()
    }
    /// Fold in the result of the speech text that ran. `Some(n)` means another
    /// speech text follows; `n` speech texts were just recorded into a new voice
    /// anchor, or none. `None` ends the script: it ended, was stopped, or a
    /// speech text reached its frame ceiling.
    pub fn complete(
        &mut self,
        report: SynthesisReport,
        cancel: &AtomicBool,
        encode: impl FnOnce(&[f32], &AtomicBool) -> Result<Option<Vec<[u32; CODEBOOKS]>>>,
    ) -> Result<Option<usize>> {
        let index = self.reports.len();
        self.reports.push(report);
        let report = self.reports.last().expect("pushed");
        let stopped = report.stopped;
        let ceiling = report.reached_limit;
        if stopped {
            self.stopped = true;
        }
        if ceiling {
            self.reached_limit = Some(index);
        }
        if self.stopped || self.reached_limit.is_some() {
            return Ok(None);
        }
        let more = self.reports.len() < self.texts.len();
        // The last speech text has nothing left to condition on.
        if !more || !self.wants_anchor() {
            return Ok(more.then_some(0));
        }
        let Some(codes) = encode(&self.recording, cancel)? else {
            self.stopped = true;
            return Ok(None);
        };
        let anchor = ReferenceAudio {
            text: transcript(&self.texts[..=index]),
            codes,
        };
        anchor.validate()?;
        self.anchor = Some(anchor);
        self.anchor_speech_texts = index + 1;
        self.recording = Vec::new();
        Ok(Some(index + 1))
    }
    fn wants_anchor(&self) -> bool {
        self.derive && self.anchor.is_none() && self.recording.len() >= ANCHOR_SAMPLES
    }
    pub fn report(self) -> ScriptReport {
        let anchor = match (&self.base.reference, self.anchor_speech_texts) {
            (Some(_), _) => AnchorOrigin::Supplied,
            (None, 0) => AnchorOrigin::None,
            (None, _) => AnchorOrigin::Opening,
        };
        ScriptReport {
            frames: self.reports.iter().map(|r| r.frames).sum(),
            audio_seconds: self.reports.iter().map(|r| r.audio_seconds).sum(),
            elapsed_seconds: self.started.elapsed().as_secs_f64(),
            first_audio_seconds: self.reports.first().and_then(|r| r.first_audio_seconds),
            stopped: self.stopped,
            reached_limit: self.reached_limit,
            anchor,
            anchor_speech_texts: self.anchor_speech_texts,
            texts: self.texts,
            speech_texts: self.reports,
        }
    }
}

fn transcript(texts: &[SpeechText]) -> String {
    let mut transcript = String::new();
    for text in texts {
        // An English sentence keeps a space after it; a Chinese one does not.
        if transcript.ends_with(|c: char| c.is_ascii()) {
            transcript.push(' ');
        }
        transcript.push_str(&text.text);
    }
    transcript
}

impl<B: Backend> Model<B> {
    /// Speak a speech script. Each speech text is one speech synthesis whose PCM
    /// is handed off before the next speech text starts. Pass the reference
    /// encoder whenever the prepared script needs a voice anchor; without it
    /// such a script is refused, because its speech texts would each pick
    /// their own voice.
    pub fn synthesize_script(
        &self,
        options: &ScriptOptions,
        cancel: &AtomicBool,
        encoder: Option<&ReferenceEncoder<B>>,
        on_event: impl FnMut(ScriptEvent<'_>) -> Result<()>,
    ) -> Result<ScriptReport> {
        self.speak_prepared(self.prepare_script(options)?, cancel, encoder, on_event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speech_texts(n: usize) -> Vec<SpeechText> {
        (0..n)
            .map(|i| SpeechText {
                text: format!("第{i}句。"),
                tokens: 4,
            })
            .collect()
    }

    fn ran(frames: usize) -> SynthesisReport {
        SynthesisReport {
            frames,
            audio_seconds: frames as f64 * 0.08,
            elapsed_seconds: 1.0,
            first_audio_seconds: Some(0.5),
            stopped: false,
            reached_limit: false,
            codes: vec![],
            profile: None,
        }
    }

    fn speak(run: &mut ScriptRun, seconds: f64) {
        let _ = run.next_options().unwrap();
        run.record(&vec![0.0; (seconds * SAMPLE_RATE as f64) as usize]);
    }

    #[test]
    fn later_speech_texts_share_the_anchor_recorded_once_from_the_opening() {
        let mut run = ScriptRun::new(speech_texts(4), SynthesisOptions::default(), true);
        let mut encoded = None;
        speak(&mut run, 2.0);
        assert_eq!(
            run.complete(ran(25), &AtomicBool::new(false), |_, _| {
                panic!("two seconds is too short to record")
            })
            .unwrap(),
            Some(0)
        );
        assert!(run.next_options().unwrap().reference.is_none());
        speak(&mut run, 2.0);
        let recorded = run
            .complete(ran(25), &AtomicBool::new(false), |pcm, _| {
                encoded = Some(pcm.to_vec());
                Ok(Some(vec![[7; CODEBOOKS]; 50]))
            })
            .unwrap();
        assert_eq!(recorded, Some(2));
        assert_eq!(encoded.as_ref().unwrap().len(), 4 * SAMPLE_RATE as usize);
        let anchor = run.next_options().unwrap().reference.expect("anchor");
        assert_eq!(anchor.text, "第0句。第1句。");
        assert_eq!(anchor.codes, vec![[7; CODEBOOKS]; 50]);
        assert_eq!(
            run.next_options().unwrap().reference.unwrap().text,
            anchor.text
        );
        run.record(&[0.0; SAMPLE_RATE as usize]);
        assert_eq!(
            run.complete(ran(25), &AtomicBool::new(false), |_, _| {
                panic!("the anchor is already recorded")
            })
            .unwrap(),
            Some(0)
        );
        let _ = run.next_options().unwrap();
        assert_eq!(
            run.complete(ran(25), &AtomicBool::new(false), |_, _| Ok(None))
                .unwrap(),
            None
        );
        assert!(run.next_options().is_none());
        let report = run.report();
        assert_eq!(report.texts, speech_texts(4));
        assert_eq!(report.anchor, AnchorOrigin::Opening);
        assert_eq!(report.anchor_speech_texts, 2);
        assert_eq!(report.speech_texts.len(), 4);
        assert_eq!(report.frames, 100);
        assert_eq!(report.audio_seconds, 8.0);
        assert_eq!(report.first_audio_seconds, Some(0.5));
        assert!(!report.stopped);
        assert_eq!(report.reached_limit, None);
    }

    #[test]
    fn a_frame_ceiling_stops_the_script_and_keeps_what_ran() {
        let mut run = ScriptRun::new(speech_texts(3), SynthesisOptions::default(), false);
        speak(&mut run, 0.0);
        run.complete(ran(25), &AtomicBool::new(false), |_, _| Ok(None))
            .unwrap();
        speak(&mut run, 0.0);
        let mut ceiling = ran(1500);
        ceiling.reached_limit = true;
        assert_eq!(
            run.complete(ceiling, &AtomicBool::new(false), |_, _| Ok(None))
                .unwrap(),
            None
        );
        assert!(run.next_options().is_none());
        let report = run.report();
        assert_eq!(report.reached_limit, Some(1));
        assert_eq!(report.anchor, AnchorOrigin::None);
        assert_eq!(report.speech_texts.len(), 2);
        assert_eq!(report.frames, 1525);
    }

    #[test]
    fn a_stopped_speech_text_ends_the_script() {
        let mut run = ScriptRun::new(speech_texts(3), SynthesisOptions::default(), true);
        speak(&mut run, 1.0);
        let mut stopped = ran(4);
        stopped.stopped = true;
        assert_eq!(
            run.complete(stopped, &AtomicBool::new(false), |_, _| {
                panic!("a stopped speech text records no anchor")
            })
            .unwrap(),
            None
        );
        assert!(run.report().stopped);
    }

    #[test]
    fn a_supplied_reference_condition_every_speech_text() {
        let reference = ReferenceAudio {
            text: "参考。".into(),
            codes: vec![[1; CODEBOOKS]; 3],
        };
        let base = SynthesisOptions {
            reference: Some(reference.clone()),
            ..Default::default()
        };
        let mut run = ScriptRun::new(speech_texts(2), base, false);
        assert_eq!(
            run.next_options().unwrap().reference.unwrap().text,
            reference.text
        );
        speak(&mut run, 10.0);
        assert_eq!(
            run.complete(ran(125), &AtomicBool::new(false), |_, _| {
                panic!("a supplied reference is never re-recorded")
            })
            .unwrap(),
            Some(0)
        );
        assert_eq!(
            run.next_options().unwrap().reference.unwrap().codes,
            reference.codes
        );
        run.complete(ran(1), &AtomicBool::new(false), |_, _| Ok(None))
            .unwrap();
        assert_eq!(run.report().anchor, AnchorOrigin::Supplied);
    }

    #[test]
    fn a_single_speech_text_or_a_short_opening_records_no_anchor() {
        let mut run = ScriptRun::new(speech_texts(1), SynthesisOptions::default(), true);
        speak(&mut run, 30.0);
        assert_eq!(
            run.complete(ran(375), &AtomicBool::new(false), |_, _| {
                panic!("one speech text needs no anchor")
            })
            .unwrap(),
            None
        );
        assert_eq!(run.report().anchor, AnchorOrigin::None);
        let mut run = ScriptRun::new(speech_texts(2), SynthesisOptions::default(), true);
        speak(&mut run, 1.0);
        assert_eq!(
            run.complete(ran(12), &AtomicBool::new(false), |_, _| Ok(None))
                .unwrap(),
            Some(0)
        );
        assert_eq!(
            run.complete(ran(1), &AtomicBool::new(false), |_, _| {
                panic!("the opening stayed too short")
            })
            .unwrap(),
            None
        );
        assert_eq!(run.report().anchor, AnchorOrigin::None);
    }

    #[test]
    fn cancelling_while_recording_the_anchor_stops_the_script() {
        let mut run = ScriptRun::new(speech_texts(3), SynthesisOptions::default(), true);
        speak(&mut run, 5.0);
        assert_eq!(
            run.complete(ran(62), &AtomicBool::new(true), |_, _| Ok(None))
                .unwrap(),
            None
        );
        assert!(run.report().stopped);
    }

    #[test]
    fn transcripts_join_english_with_a_space_and_chinese_without_one() {
        let texts = |parts: &[&str]| {
            parts
                .iter()
                .map(|text| SpeechText {
                    text: text.to_string(),
                    tokens: 1,
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(
            transcript(&texts(&["第一句。", "第二句。"])),
            "第一句。第二句。"
        );
        assert_eq!(
            transcript(&texts(&["One here.", "Two here."])),
            "One here. Two here."
        );
        assert_eq!(
            transcript(&texts(&["一是One.", "二是Two."])),
            "一是One. 二是Two."
        );
        assert_eq!(
            transcript(&texts(&["One 是一。", "Two 是二。"])),
            "One 是一。Two 是二。"
        );
    }

    #[test]
    fn a_prepared_script_keeps_the_ordered_texts_and_the_anchor_decision() {
        let long = speech_texts(3);
        let base = SynthesisOptions::default();
        let prepared = PreparedScript::new(long.clone(), base.clone());
        assert_eq!(prepared.texts, long);
        assert_eq!(prepared.synthesis.text, base.text);
        assert!(prepared.needs_voice_anchor);

        let short = PreparedScript::new(speech_texts(1), base.clone());
        assert!(!short.needs_voice_anchor);

        let reference = ReferenceAudio {
            text: "参考。".into(),
            codes: vec![[1; CODEBOOKS]; 3],
        };
        let mut supplied = base.clone();
        supplied.reference = Some(reference);
        let supplied = PreparedScript::new(long, supplied);
        assert!(!supplied.needs_voice_anchor);
    }

    #[test]
    #[ignore = "needs the Breeze-TTS-2 tokenizer; skips when it is absent"]
    fn preparation_splits_the_real_tokenizer_way_for_every_script_kind() {
        let Some(tokenizer) = crate::script::tests::checkpoint_tokenizer() else {
            eprintln!("skipped: models/Breeze-TTS-2/tokenizer.json is absent");
            return;
        };
        let splitter = crate::script::Splitter::new(&tokenizer).unwrap();
        let short = "你好，今天天气很好。";
        let paragraph = "人工智能正在改变我们的工作方式。它可以帮助医生更快地阅读影像，也可以帮助工程师检查代码。当然，新的工具也带来新的问题。例如，我们如何确认答案是对的？又如何保护每个人的隐私？这些问题没有简单的答案。我们需要更多的讨论，也需要更清楚的规则。让我们从一个小例子开始。假设你要写一封邮件，说明项目推迟了三天。你希望语气礼貌，但不啰嗦。模型可以给出三个版本，你再挑选最合适的那个。它不会替你承担责任。决定仍然是你做的。";
        let long = paragraph.repeat(6);
        let reference = ReferenceAudio {
            text: "参考。".into(),
            codes: vec![[1; CODEBOOKS]; 3],
        };
        for (script, reference) in [
            (short, None),
            (long.as_str(), None),
            (short, Some(reference.clone())),
            (long.as_str(), Some(reference)),
        ] {
            let mut synthesis = SynthesisOptions {
                text: script.to_string(),
                reference,
                ..Default::default()
            };
            let texts = splitter.split(script, ScriptBudget::default()).unwrap();
            let prepared = PreparedScript::new(texts.clone(), synthesis.clone());
            assert_eq!(prepared.texts, texts);
            assert_eq!(prepared.synthesis.text, script);
            assert_eq!(
                prepared.needs_voice_anchor,
                synthesis.reference.is_none() && texts.len() > 1
            );
            synthesis.text = texts[0].text.clone();
            assert_eq!(prepared.texts[0], texts[0]);
        }
    }
}
