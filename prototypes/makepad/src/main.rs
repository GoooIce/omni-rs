use makepad_widgets::*;
#[cfg(not(target_arch = "wasm32"))]
mod asr;
#[cfg(not(target_arch = "wasm32"))]
mod audio_spool;
#[cfg(not(target_arch = "wasm32"))]
mod capture_buffer;
#[cfg(not(target_arch = "wasm32"))]
mod clip_playback;
#[cfg(not(target_arch = "wasm32"))]
mod desktop_runtime;
#[cfg(not(target_arch = "wasm32"))]
mod engine;
#[cfg(not(target_arch = "wasm32"))]
mod engine_harness;
#[cfg(not(target_arch = "wasm32"))]
mod engine_ui;
#[cfg(not(target_arch = "wasm32"))]
mod export;
#[cfg(not(target_arch = "wasm32"))]
mod gpu;
#[cfg(not(target_arch = "wasm32"))]
mod image_engine;
#[cfg(not(target_arch = "wasm32"))]
mod image_ui;
#[cfg(not(target_arch = "wasm32"))]
mod microphone;
#[cfg(not(target_arch = "wasm32"))]
mod model_loader;
#[cfg(not(target_arch = "wasm32"))]
mod model_location;
#[cfg(not(target_arch = "wasm32"))]
mod music_engine;
#[cfg(not(target_arch = "wasm32"))]
mod music_ui;
#[cfg(not(target_arch = "wasm32"))]
mod playback;
#[cfg(not(target_arch = "wasm32"))]
mod playback_buffer;
mod protocol;
#[cfg(not(target_arch = "wasm32"))]
mod resampler;
#[cfg(not(target_arch = "wasm32"))]
mod speech_enhancer;
mod transcription_ui;
#[cfg(not(target_arch = "wasm32"))]
mod translation_engine;
#[cfg(not(target_arch = "wasm32"))]
mod translation_ui;
#[cfg(not(target_arch = "wasm32"))]
mod tts_engine;
#[cfg(not(target_arch = "wasm32"))]
mod tts_ui;
#[cfg(target_arch = "wasm32")]
mod web_bridge;
#[cfg(not(target_arch = "wasm32"))]
mod workspace_ui;
#[cfg(not(target_arch = "wasm32"))]
use desktop_runtime::Runtime;
use transcription_ui::State as TranscriptionState;
#[cfg(not(target_arch = "wasm32"))]
use workspace_ui::State as WorkspaceState;

app_main!(App, font_set: International);

// The pinned Button API separates input admission from the disabled material.
// Keep both states together for every action in the shared workspace.
trait ActionButtonState {
    fn set_action_enabled(&self, cx: &mut Cx, enabled: bool);
}

impl ActionButtonState for ButtonRef {
    fn set_action_enabled(&self, cx: &mut Cx, enabled: bool) {
        if self
            .borrow()
            .is_some_and(|button| button.enabled() != enabled)
        {
            self.set_enabled(cx, enabled);
            self.set_disabled(cx, !enabled);
        }
    }
}

script_mod! {
    use mod.prelude.widgets.*

    // Explicit, eager font resources keep native and Web text metrics aligned.
    let speech_font = FontFamily{
        latin := FontMember{res: crate_resource("makepad_widgets:resources/IBMPlexSans-Text.ttf")}
        cjk := FontMember{res: crate_resource("self:resources/fonts/NotoSansSC-VF.ttf") weight: 400}
        hangul := FontMember{res: crate_resource("self:resources/fonts/NotoSansKR-Hangul.ttf") weight: 400}
        arabic := FontMember{res: crate_resource("self:resources/fonts/NotoSansArabic-VF.ttf") weight: 400}
        thai := FontMember{res: crate_resource("self:resources/fonts/NotoSansThai-VF.ttf") weight: 400}
        devanagari := FontMember{res: crate_resource("self:resources/fonts/NotoSansDevanagari-VF.ttf") weight: 400}
    }
    let speech_heading = FontFamily{
        latin := FontMember{res: crate_resource("makepad_widgets:resources/IBMPlexSans-SemiBold.ttf")}
        cjk := FontMember{res: crate_resource("self:resources/fonts/NotoSansSC-VF.ttf") weight: 600}
    }
    let SpeechLabel = Label{
        padding: 0 margin: 0
        draw_text.color: #e8eeeb
        draw_text.text_style.font_family: speech_font
        draw_text.text_style.font_size: 13
    }
    let MutedLabel = SpeechLabel{
        width: Fill
        draw_text.color: #a3b1aa
        draw_text.text_style.font_size: 12
    }
    let SectionLabel = SpeechLabel{
        draw_text.color: #edf3ef
        draw_text.text_style.font_family: speech_heading
        draw_text.text_style.font_size: 18
    }
    let SpeechInput = TextInputFlat{
        margin: 0 padding: Inset{left: 10 right: 10 top: 9 bottom: 9}
        draw_bg +: {
            border_radius: 5 border_size: 1
            color: #121719 color_hover: #151c1d color_focus: #151d1c
            color_down: #121719 color_empty: #121719 color_disabled: #191e20
            border_color: #35413d border_color_hover: #60766a
            border_color_focus: #a5d7bd border_color_down: #60766a
            border_color_empty: #35413d border_color_disabled: #2a3330
            color_dither: 0
        }
        draw_text +: {
            color: #edf3ef color_hover: #edf3ef color_focus: #edf3ef color_down: #edf3ef
            color_empty: #91a099 color_empty_hover: #abb8b0 color_empty_focus: #91a099
            color_disabled: #7e8d84
            text_style.font_family: speech_font
            text_style.font_size: 14
        }
        draw_cursor.color: #a5d7bd
        draw_selection.color: #3b6955
        draw_selection.color_focus: #3b6955
    }
    let SpeechButton = ButtonFlat{
        margin: 0 height: 34
        padding: Inset{left: 10 right: 10 top: 6 bottom: 6}
        draw_bg +: {
            border_size: 1 border_radius: 5
            color: #x27322e color_hover: #34463c color_down: #x17251e color_focus: #34463c
            color_disabled: #202824
            border_color: #46564c border_color_hover: #70897a border_color_down: #a5d7bd
            border_color_focus: #a5d7bd border_color_disabled: #2d3831
            color_dither: 0
        }
        draw_text +: {
            color: #e5ede7 color_hover: #ffffff color_down: #ffffff color_focus: #ffffff
            color_disabled: #728277
            text_style.font_family: speech_font
            text_style.font_size: 13
        }
    }
    let PrimaryButton = SpeechButton{
        draw_bg +: {
            color: #b7dfc7 color_hover: #d0efdb color_down: #93c8a9 color_focus: #d0efdb
            color_disabled: #31463a
            border_color: #b7dfc7 border_color_hover: #e0f7e8 border_color_down: #e0f7e8
            border_color_focus: #e0f7e8 border_color_disabled: #3c5143
        }
        draw_text +: {
            color: #142d20 color_hover: #142d20 color_down: #142d20 color_focus: #142d20
            color_disabled: #8ca795
            text_style.font_family: speech_heading
        }
    }
    let QuietButton = SpeechButton{
        draw_bg +: {
            color: #1c2420 color_hover: #2b3b31 color_down: #x17251e color_focus: #2b3b31
            color_disabled: #1c2420
            border_color: #x39463e border_color_disabled: #2c3730
        }
    }
    let WorkspacePanel = RoundedView{
        width: Fill height: Fit flow: Down padding: 16 spacing: 10
        draw_bg.color: #1c2420 draw_bg.border_radius: 8
        draw_bg.border_color: #344137 draw_bg.border_size: 1
    }
    let Divider = RoundedView{
        width: Fill height: 1 padding: 0
        draw_bg.color: #344137 draw_bg.border_radius: 0
    }

    startup() do #(App::script_component(vm)){
        ui: Root{
            main_window := Window{
                window.inner_size: vec2(1000, 810)
                window.title: "知音 · 本地工作台"
                body +: {
                    show_bg: true
                    draw_bg +: {pixel: fn() {return #111814}}
                    RoundedView{
                        width: Fill height: Fill flow: Down padding: 20 spacing: 12
                        draw_bg.color: #111814 draw_bg.border_radius: 0
                        View{
                            width: Fill height: Fit flow: Right spacing: 16 align: Align{y: 0.5}
                            app_heading := View{
                                width: Fill height: Fit flow: Down spacing: 4
                                app_title := SectionLabel{text: "知音" draw_text.text_style.font_size: 20}
                                app_subtitle := MutedLabel{text: "声音、文字、图像与音乐"}
                            }
                            nav_group := View{
                                visible: false width: Fit height: Fit flow: Down spacing: 4
                                navigation := View{
                                    width: Fit height: Fit flow: Right spacing: 6
                                    language_tab := SpeechButton{text: "● 语言" height: 38}
                                    creative_tab := SpeechButton{text: "创作" height: 38}
                                }
                                tab_indicator := RoundedView{
                                    width: 48 height: 3 margin: {left: 0}
                                    draw_bg.color: #b7dfc7 draw_bg.border_radius: 2
                                }
                            }
                        }
                        running_bar := RoundedView{
                            visible: false width: Fill height: Fit flow: Right spacing: 10
                            padding: 10 align: Align{y: 0.5}
                            draw_bg.color: #27372d draw_bg.border_radius: 5
                            running_status := SpeechLabel{width: Fill text: "空闲" draw_text.color: #b7dfc7}
                            running_stop := SpeechButton{text: "停止推理" height: 32}
                            playback_stop := SpeechButton{text: "停止播放" height: 32}
                        }
                        pages := ScrollYView{
                            width: Fill height: Fill flow: Down spacing: 0
                            page_stage := View{
                                width: Fill height: Fit flow: Down clip_x: true clip_y: false
                                page_track := View{
                                    width: Fill height: Fit flow: Right spacing: 0
                            language_page := View{
                                width: Fill height: Fit flow: Down spacing: 12
                                language_columns := View{
                                    width: Fill height: Fit flow: Right spacing: 14
                                    asr_page := WorkspacePanel{
                                        width: Fill{weight: 170}
                                        spacing: 8
                                        View{
                                            width: Fill height: Fit flow: Right align: Align{y: 0.5}
                                            SectionLabel{width: Fill text: "听见，变成文字"}
                                            asr_details_toggle := QuietButton{text: "识别设置" height: 30}
                                        }
                                        status := MutedLabel{text: "正在准备转写模型…" draw_text.color: #b7dfc7}
                                        model_progress := RoundedView{
                                            visible: false width: Fill height: 4 padding: 0
                                            draw_bg.color: #344137 draw_bg.border_radius: 2
                                            model_progress_fill := RoundedView{width: 0 height: Fill draw_bg.color: #b7dfc7 draw_bg.border_radius: 2}
                                        }
                                        model_recovery := View{
                                            visible: false width: Fill height: Fit flow: Right spacing: 8
                                            choose_model := SpeechButton{text: "选择模型…"}
                                            retry_model := SpeechButton{text: "重试加载"}
                                        }
                                        View{
                                            width: Fill height: Fit flow: Right spacing: 8 align: Align{y: 0.5}
                                            audio_file := SpeechLabel{
                                                width: Fill text: "尚未选择音频"
                                                draw_text.color: #edf3ef
                                                draw_text.text_style.font_size: 15
                                            }
                                            choose_audio := SpeechButton{text: "选择音频" height: 34}
                                            sample := QuietButton{text: "官方示例" height: 34}
                                        }
                                        View{
                                            width: Fill height: Fit flow: Right spacing: 8
                                            transcribe := PrimaryButton{text: "开始转写" height: 38}
                                            microphone := SpeechButton{text: "麦克风" height: 38}
                                            stop := QuietButton{text: "停止" height: 38}
                                        }
                                        input_status_row := View{
                                            visible: false width: Fill height: Fit
                                            input_status := MutedLabel{text: ""}
                                        }
                                        Divider{}
                                        View{
                                            width: Fill height: Fit flow: Right spacing: 6 align: Align{y: 0.5}
                                            asr_language_caption := SpeechLabel{text: "语言" width: Fit}
                                            asr_lang_auto := QuietButton{text: "● 自动" height: 30}
                                            asr_lang_zh := QuietButton{text: "中文" height: 30}
                                            asr_lang_en := QuietButton{text: "英文" height: 30}
                                            asr_lang_yue := QuietButton{text: "粤语" height: 30}
                                            asr_lang_ja := QuietButton{text: "日语" height: 30}
                                            asr_lang_ko := QuietButton{text: "韩语" height: 30}
                                        }
                                        View{
                                            width: Fill height: Fit flow: Right spacing: 8 align: Align{y: 0.5}
                                            transcript_title := SpeechLabel{width: Fill text: "转写文字"}
                                            copy := QuietButton{text: "复制" height: 30}
                                            asr_to_tts := QuietButton{text: "送去朗读 →" height: 30}
                                        }
                                        transcript := SpeechInput{
                                            width: Fill height: 188 is_multiline: true is_read_only: true
                                            empty_text: "转写出现在这里。"
                                            draw_text.text_style.font_size: 16
                                            draw_bg.color: #24312b
                                            draw_bg.color_empty: #24312b
                                            draw_bg.color_hover: #24312b
                                            draw_bg.color_focus: #24312b
                                        }
                                        asr_translation_group := View{
                                            visible: false width: Fill height: Fit flow: Down spacing: 8
                                                asr_translation_readiness := MutedLabel{text: "尚未载入" draw_text.color: #d7b56a}
                                                View{
                                                width: Fill height: Fit flow: Right spacing: 8 align: Align{y: 0.5}
                                                SpeechLabel{text: "翻译为"}
                                                asr_translation_language := SpeechInput{width: Fill height: 34 text: "English" empty_text: "目标语言全称"}
                                                asr_translate := QuietButton{text: "载入翻译模型" height: 38}
                                                asr_translation_stop := QuietButton{text: "停止"}
                                            }
                                            asr_translation_well := View{
                                                width: Fill height: Fit
                                                asr_translation := SpeechInput{
                                                    width: Fill height: 56 is_multiline: true is_read_only: true
                                                    empty_text: "译文写在这里。"
                                                }
                                            }
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 8
                                                asr_translation_copy := QuietButton{text: "复制译文" height: 30}
                                                asr_translation_to_tts := QuietButton{text: "译文朗读 →" height: 30}
                                                asr_translation_model := QuietButton{text: "翻译模型" height: 30 visible: false}
                                            }
                                            asr_translation_note := View{
                                                visible: false width: Fill height: Fit
                                                asr_translation_status := MutedLabel{text: ""}
                                            }
                                        }
                                        asr_details := View{
                                            visible: false width: Fill height: Fit flow: Down spacing: 10
                                            asr_details_pad := View{width: Fill height: 0}
                                            Divider{}
                                            asr_details_title := SpeechLabel{text: "识别设置"}
                                            SpeechLabel{text: "音频路径"}
                                            audio_path_name := SpeechLabel{
                                                visible: false
                                                width: Fill text: "尚未选择音频"
                                                draw_text.color: #edf3ef
                                                draw_text.text_style.font_size: 15
                                            }
                                            audio_path := SpeechInput{
                                                width: Fill height: 72 is_multiline: true
                                                empty_text: "完整 WAV 路径。也可以用上面的选择音频。"
                                            }
                                            asr_other_language := SpeechLabel{text: "其他识别语言"}
                                            asr_language := SpeechInput{width: Fill height: 34 empty_text: "泰语，或其他没有按钮的名称"}
                                            asr_language_tail := View{width: Fill height: 36}
                                            MutedLabel{visible: false text: "这里只填按钮上没有的语言。下次转写生效。"}
                                            SpeechLabel{text: "识别提示词"}
                                            context := SpeechInput{width: Fill height: 56 is_multiline: true empty_text: "人名、术语。下次转写才用。"}
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 6
                                                speed_balanced := QuietButton{text: "● 均衡" height: 30}
                                                speed_low := QuietButton{text: "低延迟" height: 30}
                                                speed_fast := QuietButton{text: "快速" height: 30}
                                                speed_save := QuietButton{text: "省算力" height: 30}
                                            }
                                            speed_hint := MutedLabel{text: "均衡：320 ms 一块，前瞻 160 ms，下次生效。"}
                                            ping := QuietButton{text: "响应测试" height: 30}
                                            hint := MutedLabel{text: "开始之后，停止会保留已经确认的文字。"}
                                            model_title := SpeechLabel{text: "转写模型"}
                                            model_form := View{
                                                width: Fill height: Fit flow: Right spacing: 8
                                                model_path := SpeechInput{width: Fill height: 36 empty_text: "模型目录 / Web manifest URL"}
                                                load := SpeechButton{text: "载入" height: 36}
                                            }
                                            metrics := MutedLabel{text: "音频 —  ·  推理 —  ·  RTF —"}
                                            heartbeat := MutedLabel{text: "界面就绪"}
                                        }
                                    }
                                    language_stack_gap := View{visible: false width: Fill height: 64}
                                    tts_page := WorkspacePanel{
                                        spacing: 8
                                        View{
                                            width: Fill height: Fit flow: Right spacing: 8 align: Align{y: 0.5}
                                            SectionLabel{width: Fill text: "让文字开口"}
                                            tts_load := QuietButton{text: "载入模型" height: 30}
                                        }
                                        tts_status := MutedLabel{text: "语音模型尚未载入" draw_text.color: #d7b56a}
                                        tts_progress := RoundedView{
                                            visible: false width: Fill height: 4 padding: 0
                                            draw_bg.color: #344137 draw_bg.border_radius: 2
                                            tts_progress_fill := RoundedView{width: 0 height: Fill draw_bg.color: #b7dfc7 draw_bg.border_radius: 2}
                                        }
                                        tts_recovery := View{
                                            visible: false width: Fill height: Fit flow: Right spacing: 8
                                            tts_choose := SpeechButton{text: "选择模型…"}
                                            tts_retry := SpeechButton{text: "重试加载"}
                                        }
                                        View{
                                            width: Fill height: Fit flow: Right spacing: 8
                                            tts_speech_mode := QuietButton{text: "● 朗读与克隆" height: 30}
                                            tts_design_mode := QuietButton{text: "音色设计" height: 30}
                                        }
                                        tts_hearing := View{
                                            visible: false width: Fill height: Fit flow: Down spacing: 8
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 8
                                                tts_hear_0 := SpeechButton{text: "甲" height: 32}
                                                tts_hear_1 := SpeechButton{text: "乙" height: 32}
                                                tts_hear_2 := SpeechButton{text: "丙" height: 32}
                                            }
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 8
                                                tts_pick_0 := SpeechButton{text: "选择甲" height: 32}
                                                tts_pick_1 := SpeechButton{text: "选择乙" height: 32}
                                                tts_pick_2 := SpeechButton{text: "选择丙" height: 32}
                                            }
                                            MutedLabel{text: "听完三段后再选择。这里不显示改写后的文字。"}
                                        }
                                        tts_text_title := SpeechLabel{text: "要朗读的文字"}
                                            tts_text := SpeechInput{
                                                width: Fill height: 116 is_multiline: true
                                                empty_text: "写上要读的中文或英文。"
                                            draw_text.text_style.font_size: 16
                                        }
                                        View{
                                            width: Fill height: Fit flow: Right spacing: 8
                                            tts_synthesize := PrimaryButton{text: "合成并播放" height: 38}
                                            tts_stop := QuietButton{text: "停止" height: 38}
                                            tts_replay := QuietButton{text: "试听" height: 38}
                                            tts_save := QuietButton{text: "保存" height: 38}
                                        }
                                        tts_design_results := View{
                                            visible: false width: Fill height: Fit flow: Down spacing: 8
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 8
                                                tts_design_next := SpeechButton{text: "换个候选" height: 32}
                                                tts_design_use := SpeechButton{text: "使用此音色朗读" height: 32}
                                            }
                                            tts_design_candidate := MutedLabel{text: "候选 1"}
                                        }
                                        tts_details_toggle := QuietButton{text: "声音设置" height: 32}
                                        tts_details := View{
                                            visible: false width: Fill height: Fit flow: Down spacing: 9
                                            tts_enhance := QuietButton{text: "提示词增强" height: 32}
                                            tts_instruction_title := SpeechLabel{text: "声音描述（可选）"}
                                            tts_instruction := SpeechInput{
                                                width: Fill height: 66 is_multiline: true
                                                empty_text: "例如：年轻女性，声音温柔清晰，语速自然。"
                                            }
                                            tts_draft := SpeechButton{text: "改进增强提示" height: 32}
                                            tts_draft_text := SpeechInput{
                                                width: Fill height: 88 is_multiline: true
                                                empty_text: "起草后的增强提示显示在这里，可以修改。"
                                            }
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 8
                                                tts_draft_accept := SpeechButton{text: "接受" height: 32}
                                                tts_draft_discard := QuietButton{text: "不接受" height: 32}
                                                tts_draft_restore := QuietButton{text: "恢复上一版" height: 32}
                                            }
                                            tts_design_presets := View{
                                                visible: false width: Fill height: Fit flow: Down spacing: 8
                                                View{
                                                    width: Fill height: Fit flow: Right spacing: 8
                                                    tts_voice_gentle := SpeechButton{text: "温柔女声" height: 30}
                                                    tts_voice_warm := SpeechButton{text: "沉稳男声" height: 30}
                                                }
                                                View{
                                                    width: Fill height: Fit flow: Right spacing: 8
                                                    tts_voice_bright := SpeechButton{text: "清朗青年" height: 30}
                                                    tts_voice_english := SpeechButton{text: "英文旁白" height: 30}
                                                }
                                                MutedLabel{text: "描述的语言建议与试听文字一致。"}
                                            }
                                            tts_reference_group := View{
                                                width: Fill height: Fit flow: Down spacing: 8
                                                View{
                                                    width: Fill height: Fit flow: Right spacing: 8
                                                    tts_reference_choose := SpeechButton{text: "参考 WAV…" height: 32}
                                                    tts_reference_clear := QuietButton{text: "清除参考" height: 32}
                                                }
                                                tts_reference_name := MutedLabel{text: "建议 3–10 秒清晰人声"}
                                                tts_reference_text := SpeechInput{
                                                    width: Fill height: 64 is_multiline: true
                                                    empty_text: "填写参考音频中实际说出的文字，用于克隆声音。"
                                                }
                                            }
                                            tts_metrics := MutedLabel{text: "音频 —  ·  首声 —  ·  缓冲 —"}
                                            MutedLabel{text: "Breeze-TTS-2 · 24 kHz · 权重与输出限研究及非商业用途"}
                                        }
                                        tts_translate_toggle := QuietButton{text: "翻译这段朗读" height: 32}
                                        tts_translation_panel := View{
                                            visible: false width: Fill height: Fit flow: Down spacing: 8
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 8 align: Align{y: 0.5}
                                                SpeechLabel{text: "翻译为"}
                                                tts_translation_language := SpeechInput{width: Fill height: 34 text: "English" empty_text: "目标语言全称"}
                                                tts_translate := SpeechButton{text: "翻译"}
                                                tts_translation_stop := QuietButton{text: "停止"}
                                            }
                                            tts_translation := SpeechInput{
                                                width: Fill height: 56 is_multiline: true is_read_only: true
                                                empty_text: "译文写在这里。"
                                            }
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 8
                                                tts_translation_use := SpeechButton{text: "用译文朗读" height: 30}
                                                tts_translation_model := QuietButton{text: "翻译模型" height: 30}
                                            }
                                            tts_translation_status := MutedLabel{text: ""}
                                        }
                                    }
                                }
                            }
                            creative_page := View{
                                visible: false width: Fill height: Fit flow: Down spacing: 12
                                creative_columns := View{
                                    width: Fill height: Fit flow: Right spacing: 14
                                    music_page := WorkspacePanel{
                                        View{
                                            width: Fill height: Fit flow: Right spacing: 8 align: Align{y: 0.5}
                                            SectionLabel{width: Fill text: "写一首歌"}
                                            music_stop := QuietButton{text: "停止" height: 30}
                                        }
                                        MutedLabel{text: "从风格和歌词开始，也可以从录音转谱。"}
                                        music_status := MutedLabel{text: "歌曲在生成时再载入" draw_text.color: #d7b56a}
                                        music_progress := RoundedView{
                                            visible: false width: Fill height: 4 padding: 0
                                            draw_bg.color: #344137 draw_bg.border_radius: 2
                                            music_progress_fill := RoundedView{width: 0 height: Fill draw_bg.color: #b7dfc7 draw_bg.border_radius: 2}
                                        }
                                        SpeechLabel{text: "风格与声音 · 用英文"}
                                        music_style := SpeechInput{
                                            width: Fill height: 76 is_multiline: true
                                            empty_text: "acoustic pop, piano, 90 BPM"
                                        }
                                        SpeechLabel{text: "歌词"}
                                            music_lyrics := SpeechInput{
                                                width: Fill height: 112 is_multiline: true
                                            empty_text: "用 [Verse]、[Chorus] 标记段落。\n纯音乐可以填写 [Instrumental]。"
                                        }
                                        View{
                                            width: Fill height: Fit flow: Right spacing: 8 align: Align{y: 0.5}
                                            SpeechLabel{text: "最长秒数"}
                                            music_seconds := SpeechInput{width: 64 height: 34 text: "60"}
                                            SpeechLabel{text: "种子"}
                                            music_seed := SpeechInput{width: 96 height: 34 text: "831001"}
                                        }
                                        View{
                                            width: Fill height: Fit flow: Right spacing: 8
                                            music_generate := PrimaryButton{text: "生成歌曲" height: 38}
                                            music_replay := QuietButton{text: "试听" height: 38}
                                            music_save_audio := QuietButton{text: "保存 WAV" height: 38}
                                        }
                                        music_song_status := MutedLabel{text: "48 kHz 双声道 · 达到时长上限时会标记截断"}
                                        View{
                                            width: Fill height: Fit flow: Right spacing: 8
                                            music_details_toggle := QuietButton{text: "从录音转谱" height: 32}
                                            music_save_song := QuietButton{text: "生成记录" height: 32}
                                        }
                                        music_details := View{
                                            visible: false width: Fill height: Fit flow: Down spacing: 10
                                            Divider{}
                                            SpeechLabel{text: "参考音频"}
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 8
                                                music_source := SpeechInput{width: Fill height: 36 empty_text: "WAV 文件路径"}
                                                music_choose_audio := SpeechButton{text: "选择…" height: 36}
                                            }
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 8
                                                music_transcribe := SpeechButton{text: "音频转谱"}
                                                music_melody := SpeechButton{text: "旋律"}
                                                music_analyze := SpeechButton{text: "提取特征"}
                                            }
                                            music_analysis := MutedLabel{text: "转谱最长 5 分钟；分析提取音乐特征向量。"}
                                            music_save_analysis := QuietButton{text: "保存特征 JSON…" height: 30}
                                            Divider{}
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 8 align: Align{y: 0.5}
                                                SpeechLabel{width: Fill text: "ABC 乐谱"}
                                                music_use_score := SpeechButton{text: "使用此乐谱：关" height: 30}
                                            }
                                            music_score := SpeechInput{
                                                width: Fill height: 144 is_multiline: true
                                                empty_text: "检查或编辑转谱结果，也可粘贴已有的 YuE2 双声部 ABC 乐谱。"
                                            }
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 8
                                                music_restore_score := SpeechButton{text: "载入转谱结果" height: 30}
                                                music_save_abc := QuietButton{text: "保存 ABC…" height: 30}
                                            }
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 8
                                                music_save_midi := QuietButton{text: "转谱 MIDI…" height: 30}
                                                music_save_score := QuietButton{text: "转谱事件…" height: 30}
                                            }
                                            music_score_status := SpeechInput{
                                                width: Fill height: 76 is_multiline: true is_read_only: true
                                                empty_text: "MIDI 保留原始转谱；ABC 编辑用于下一次生成。"
                                            }
                                            Divider{}
                                            SpeechLabel{text: "音乐模型目录"}
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 8
                                                music_models := SpeechInput{width: Fill height: 34 empty_text: "四份音乐模型的父目录"}
                                                music_choose_models := SpeechButton{text: "选择…"}
                                            }
                                            MutedLabel{text: "MERT2 / SheetSage2 / YuE2 / VAE · 每次只加载所需模型"}
                                        }
                                        music_export_status := MutedLabel{text: ""}
                                    }
                                    creative_stack_gap := View{visible: false width: Fill height: 64}
                                    image_page := WorkspacePanel{
                                        width: Fill{weight: 170}
                                        View{
                                            width: Fill height: Fit flow: Right spacing: 8 align: Align{y: 0.5}
                                            SectionLabel{width: Fill text: "画出想象"}
                                            image_load := QuietButton{text: "载入模型" height: 30}
                                        }
                                        image_status := MutedLabel{text: "图像模型尚未载入" draw_text.color: #d7b56a}
                                        image_progress := RoundedView{
                                            visible: false width: Fill height: 4 padding: 0
                                            draw_bg.color: #344137 draw_bg.border_radius: 2
                                            image_progress_fill := RoundedView{width: 0 height: Fill draw_bg.color: #b7dfc7 draw_bg.border_radius: 2}
                                        }
                                        image_recovery := View{
                                            visible: false width: Fill height: Fit flow: Right spacing: 8
                                            image_choose := SpeechButton{text: "选择模型…"}
                                            image_retry := SpeechButton{text: "重试加载"}
                                        }
                                        image_preview_group := RoundedView{
                                            width: Fill height: 280 padding: 8 flow: Down spacing: 0
                                            draw_bg.color: #121719 draw_bg.border_radius: 8
                                            draw_bg.border_color: #5d7d6c draw_bg.border_size: 1
                                            image_preview_caption := MutedLabel{text: "载入后，画面出现在这里。"}
                                            image_preview_single := View{
                                                width: Fill height: Fill
                                                image_preview := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                            }
                                            image_observation := View{
                                                visible: false width: Fill height: Fill flow: Down spacing: 4
                                                image_official_caption := MutedLabel{text: "官方文本条件"}
                                                image_official_steps := View{
                                                    width: Fill height: Fill flow: Right spacing: 4
                                                    image_step_1 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_step_2 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_step_3 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_step_4 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_step_5 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_step_6 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_step_7 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_step_8 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                }
                                                image_1b_caption := MutedLabel{text: "MiniCPM-V-4.7-1B 投影"}
                                                image_1b_steps := View{
                                                    width: Fill height: Fill flow: Right spacing: 4
                                                    image_1b_step_1 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_1b_step_2 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_1b_step_3 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_1b_step_4 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_1b_step_5 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_1b_step_6 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_1b_step_7 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_1b_step_8 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                }
                                                image_35b_caption := MutedLabel{text: "MiniCPM-V-4.7-35B-A3B-int4 投影"}
                                                image_35b_steps := View{
                                                    width: Fill height: Fill flow: Right spacing: 4
                                                    image_35b_step_1 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_35b_step_2 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_35b_step_3 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_35b_step_4 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_35b_step_5 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_35b_step_6 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_35b_step_7 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                    image_35b_step_8 := Image{width: Fill height: Fill fit: ImageFit.Smallest}
                                                }
                                            }
                                        }
                                        image_description := View{
                                            width: Fill height: Fit flow: Down spacing: 10
                                            SpeechLabel{text: "画面描述 / 编辑指令"}
                                            image_prompt := SpeechInput{
                                                width: Fill height: 88 is_multiline: true
                                                empty_text: "雨后的玻璃音乐厅，暖色灯光。"
                                                draw_text.text_style.font_size: 16
                                            }
                                            View{
                                                width: Fill height: 48 flow: Right spacing: 10
                                                image_reference_thumb := View{
                                                    visible: false width: Fit height: Fit
                                                    image_reference_preview := Image{width: 48 height: 48 fit: ImageFit.Smallest}
                                                }
                                                View{
                                                    width: Fill height: Fit flow: Down spacing: 4
                                                    View{
                                                        width: Fill height: Fit flow: Right spacing: 8
                                                        image_reference_choose := QuietButton{text: "参考图" height: 28}
                                                        image_reference_clear := QuietButton{text: "清除" height: 28}
                                                        image_example := QuietButton{text: "示例描述" height: 28}
                                                    }
                                                    image_reference_name := MutedLabel{text: "可添加 1–4 张 PNG、JPEG 或 WebP"}
                                                }
                                            }
                                        }
                                        View{
                                            width: Fill height: Fit flow: Right spacing: 8
                                            image_generate := PrimaryButton{text: "生成图像" height: 38}
                                            image_stop := QuietButton{text: "停止" height: 38}
                                            image_save := QuietButton{text: "保存 PNG" height: 38}
                                        }
                                        image_settings_toggle := QuietButton{text: "尺寸与更多参数" height: 32}
                                        image_advanced := View{
                                            visible: false width: Fill height: Fit flow: Down spacing: 9
                                            image_observe_toggle := QuietButton{text: "去噪观察" height: 32}
                                            Divider{}
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 8 align: Align{y: 0.5}
                                                SpeechLabel{text: "尺寸"}
                                                image_width := SpeechInput{width: Fill height: 34 text: "2048"}
                                                SpeechLabel{text: "×"}
                                                image_height := SpeechInput{width: Fill height: 34 text: "2048"}
                                            }
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 6
                                                image_size_square := SpeechButton{text: "1:1" width: Fill height: 30}
                                                image_size_landscape := SpeechButton{text: "4:3" width: Fill height: 30}
                                                image_size_portrait := SpeechButton{text: "3:4" width: Fill height: 30}
                                            }
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 6
                                                image_size_photo_landscape := SpeechButton{text: "3:2" width: Fill height: 30}
                                                image_size_photo_portrait := SpeechButton{text: "2:3" width: Fill height: 30}
                                                image_size_widescreen := SpeechButton{text: "16:9" width: Fill height: 30}
                                                image_size_vertical := SpeechButton{text: "9:16" width: Fill height: 30}
                                            }
                                            image_dimensions_hint := MutedLabel{text: "自定义 256–2752 · 总像素不超过 430 万"}
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 8 align: Align{y: 0.5}
                                                SpeechLabel{text: "种子"}
                                                image_seed := SpeechInput{width: Fill height: 34 text: "42"}
                                                image_random_seed := SpeechButton{text: "换种子"}
                                            }
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 8 align: Align{y: 0.5}
                                                image_cfg_toggle := SpeechButton{text: "负向引导 · 关闭"}
                                                SpeechLabel{text: "强度"}
                                                image_cfg_scale := SpeechInput{width: Fill height: 34 text: "2.0"}
                                            }
                                            image_negative_prompt := SpeechInput{
                                                width: Fill height: 62 is_multiline: true
                                                empty_text: "不希望出现的内容，如文字、水印…"
                                            }
                                            image_cfg_hint := MutedLabel{text: "关闭时负向词不生效；默认 CFG=1。"}
                                            View{
                                                width: Fill height: Fit flow: Right spacing: 8 align: Align{y: 0.5}
                                                SpeechLabel{text: "参考图分辨率"}
                                                image_reference_resolution := SpeechInput{width: Fill height: 34 text: "自动" empty_text: "自动 / 288–1024"}
                                            }
                                            image_sampling_hint := MutedLabel{text: "8 步 · 32 像素对齐 · 负向引导关闭"}
                                            image_metrics := MutedLabel{text: "文本 —  ·  生成 —  ·  解码 —"}
                                            MutedLabel{text: "Qwen-Image-2.1 · 权重限研究及非商业用途"}
                                        }
                                    }
                                }
                            }
                                }
                                slide_lip_left := RoundedView{
                                    visible: false width: 28 height: 200
                                    draw_bg.color: #111814 draw_bg.border_radius: 0
                                }
                                slide_lip_right := RoundedView{
                                    visible: false width: 28 height: 200
                                    draw_bg.color: #111814 draw_bg.border_radius: 0
                                }
                            }
                            scroll_tail := View{width: Fill height: 20}
                        }
                        View{
                            width: Fill height: Fit flow: Right spacing: 10 align: Align{y: 0.5}
                            MutedLabel{width: Fill text: "本地处理 · 换页保留输入和结果" draw_text.text_style.font_size: 13}
                            backend := SpeechLabel{text: "Makepad" draw_text.text_style.font_size: 13 draw_text.color: #7f9286}
                        }
                    }
                }
            }
        }
    }
}

#[derive(Script, ScriptHook)]
pub struct App {
    #[live]
    ui: WidgetRef,
    #[rust]
    timer: Timer,
    #[rust]
    ticks: usize,
    #[rust]
    pings: usize,
    #[rust]
    transcription: TranscriptionState,
    #[cfg(not(target_arch = "wasm32"))]
    #[rust]
    runtime: Runtime,
    #[cfg(not(target_arch = "wasm32"))]
    #[rust]
    tts: Option<tts_ui::State>,
    #[cfg(not(target_arch = "wasm32"))]
    #[rust]
    workspace: WorkspaceState,
    #[cfg(not(target_arch = "wasm32"))]
    #[rust]
    image: Option<image_ui::State>,
    #[cfg(not(target_arch = "wasm32"))]
    #[rust]
    music: Option<music_ui::State>,
    #[cfg(not(target_arch = "wasm32"))]
    #[rust]
    translation: Option<translation_ui::State>,
}

impl App {
    fn inference_busy(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        // A cold translation retains its accepted request across load -> run.
        return self.runtime.busy() || self.translation.as_ref().is_some_and(|s| s.busy);
        #[cfg(target_arch = "wasm32")]
        self.transcription.busy()
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn refresh_running(&mut self, cx: &mut Cx) {
        let active = self.runtime.snapshot().or_else(|| {
            self.translation
                .as_ref()
                .and_then(|state| state.running())
                .map(|stopped| ("translation", stopped))
        });
        let playing = self.tts.as_ref().is_some_and(|t| t.playing())
            || self.music.as_ref().is_some_and(|m| m.playing());
        self.ui
            .view(cx, ids!(running_bar))
            .set_visible(cx, active.is_some() || playing);
        self.ui
            .button(cx, ids!(running_stop))
            .set_visible(cx, active.is_some());
        self.ui
            .button(cx, ids!(running_stop))
            .set_action_enabled(cx, active.is_some_and(|(_, stopped)| !stopped));
        self.ui
            .button(cx, ids!(playback_stop))
            .set_visible(cx, playing);
        let label = active
            .map(|(name, stopped)| {
                let name = match name {
                    "r2t2" => "语音转写",
                    "breeze" => "语音合成",
                    "qwen-image" => "图像生成",
                    "translation" => "翻译",
                    "music" => "音乐创作",
                    name => name,
                };
                format!(
                    "{}：{name}{}",
                    if stopped {
                        "正在停止"
                    } else {
                        "正在运行"
                    },
                    if playing { " · 音频播放中" } else { "" }
                )
            })
            .unwrap_or_else(|| "音频播放中 · 可继续使用其他模型".into());
        self.ui.label(cx, ids!(running_status)).set_text(cx, &label);
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn activate_current_page(&mut self, cx: &mut Cx) {
        // Sharing a workspace never starts the neighboring tool's model.
        // ASR keeps its existing startup/recovery continuation; TTS and image
        // load only from their explicit controls, music from a submitted action.
        if self.inference_busy() || self.runtime.dialog_busy() || !self.workspace.language() {
            return;
        }
        self.resume_transcription_model(cx);
    }
}

impl MatchEvent for App {
    fn handle_startup(&mut self, cx: &mut Cx) {
        #[cfg(target_arch = "wasm32")]
        {
            self.timer = cx.start_interval(0.25);
            // The browser prototype remains ASR-only, with its manual model
            // form available even though native settings start collapsed.
            self.ui.view(cx, ids!(tts_page)).set_visible(cx, false);
            self.ui.view(cx, ids!(asr_details)).set_visible(cx, true);
            self.ui
                .button(cx, ids!(asr_details_toggle))
                .set_visible(cx, false);
            self.ui.button(cx, ids!(asr_to_tts)).set_visible(cx, false);
            self.ui
                .label(cx, ids!(app_subtitle))
                .set_text(cx, "浏览器内语音转写");
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
            self.tts = Some(tts_ui::State::new(&root, self.runtime.clone()));
            self.image = Some(image_ui::State::new(&root, self.runtime.clone()));
            self.initialize_music(cx, &root);
            self.refresh_workspace_layout(cx, 1000.0);
            self.translation = Some(translation_ui::State::new(&root, self.runtime.clone()));
            self.ui
                .view(cx, ids!(asr_translation_group))
                .set_visible(cx, true);
            self.ui.view(cx, ids!(app_heading)).set_visible(cx, false);
            self.ui.view(cx, ids!(nav_group)).set_visible(cx, true);
            self.ui.view(cx, ids!(navigation)).set_visible(cx, true);
            self.ui.label(cx, ids!(app_title)).set_text(cx, "知音");
            self.ui
                .label(cx, ids!(app_subtitle))
                .set_text(cx, "声音、文字、图像与音乐");
            self.open_transcription(cx, &root);
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.open_transcription(cx);
        }
    }

    fn handle_actions(&mut self, cx: &mut Cx, actions: &Actions) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.ui.button(cx, ids!(running_stop)).clicked(actions) {
                self.runtime.stop();
                self.stop_translation(cx);
            }
            if self.ui.button(cx, ids!(playback_stop)).clicked(actions) {
                if let Some(t) = &mut self.tts {
                    t.stop_playback();
                }
                if let Some(m) = &mut self.music {
                    m.stop_playback();
                }
            }
            self.handle_workspace_actions(cx, actions);
            self.handle_translation_actions(cx, actions);
            self.handle_image_actions(cx, actions);
            self.handle_tts_actions(cx, actions);
            self.handle_music_actions(cx, actions);
        }
        self.handle_transcription_actions(cx, actions);
    }
}

impl AppMain for App {
    fn script_mod(vm: &mut ScriptVm) -> ScriptValue {
        makepad_widgets::script_mod(vm);
        self::script_mod(vm)
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event) {
        if self.timer.is_event(event).is_some() {
            self.ticks += 1;
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.settle_asr_clip(cx);
                if !self.asr_clip_pending() {
                    cx.stop_timer(self.timer);
                }
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.step_workspace_slide(cx, event);
        // The pinned Makepad version caches TextInput layout before web fonts
        // arrive. Re-layout the prefilled paths when resource loading completes.
        #[cfg(target_arch = "wasm32")]
        if matches!(event, Event::NetworkResponses(_)) {
            for path in [ids!(model_path), ids!(audio_path)] {
                let input = self.ui.text_input(cx, path);
                input.set_text(cx, &input.text());
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.poll_tts(cx);
            self.poll_image(cx);
            self.poll_translation(cx);
            self.poll_music(cx);
        }
        self.poll_transcription(cx, event);
        self.match_event(cx, event);
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.activate_current_page(cx);
            self.refresh_running(cx);
            self.refresh_workspace_controls(cx);
            self.place_workspace_indicator(cx, event);
        }
        self.refresh_transcription_controls(cx);
        #[cfg(not(target_arch = "wasm32"))]
        self.refresh_tts_controls(cx);
        #[cfg(not(target_arch = "wasm32"))]
        self.refresh_image_controls(cx);
        #[cfg(not(target_arch = "wasm32"))]
        self.refresh_translation_controls(cx);
        #[cfg(not(target_arch = "wasm32"))]
        self.refresh_music_controls(cx);
        self.ui.handle_event(cx, event, &mut Scope::empty());
    }
}
