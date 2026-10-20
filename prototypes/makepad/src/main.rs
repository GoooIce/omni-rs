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
use protocol::Update;
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

#[cfg(not(target_arch = "wasm32"))]
struct NavSlide {
    from_x: f64,
    from_w: f64,
    started: std::time::Instant,
    to_creative: bool,
}

#[derive(Script, ScriptHook)]
pub struct App {
    #[live]
    ui: WidgetRef,
    #[rust]
    timer: Timer,
    #[cfg(not(target_arch = "wasm32"))]
    #[rust]
    nav_timer: Timer,
    #[cfg(not(target_arch = "wasm32"))]
    #[rust]
    nav_slide: Option<NavSlide>,
    #[cfg(not(target_arch = "wasm32"))]
    #[rust]
    nav_placed: bool,
    #[rust]
    ticks: usize,
    #[rust]
    pings: usize,
    #[rust]
    ready: bool,
    #[rust]
    busy: bool,
    #[rust]
    loading: bool,
    #[rust]
    microphone: bool,
    #[rust]
    stopping: bool,
    #[rust]
    mode: usize,
    #[rust]
    transcript: String,
    #[rust]
    language_choice: String,
    #[rust]
    controls: Option<(bool, bool, bool, bool, bool, bool)>,
    #[cfg(not(target_arch = "wasm32"))]
    #[rust]
    engine: Option<engine::Engine>,
    #[cfg(not(target_arch = "wasm32"))]
    #[rust]
    runtime: Runtime,
    #[cfg(not(target_arch = "wasm32"))]
    #[rust]
    model_dialog: Option<desktop_runtime::Dialog>,
    #[cfg(not(target_arch = "wasm32"))]
    #[rust]
    needs_directory: bool,
    #[cfg(not(target_arch = "wasm32"))]
    #[rust]
    pending_model_load: bool,
    #[cfg(not(target_arch = "wasm32"))]
    #[rust]
    picking_model: bool,
    #[cfg(not(target_arch = "wasm32"))]
    #[rust]
    picking_audio: bool,
    #[cfg(not(target_arch = "wasm32"))]
    #[rust]
    remember_model: bool,
    #[cfg(not(target_arch = "wasm32"))]
    #[rust]
    settings_path: Option<std::path::PathBuf>,
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
        self.busy
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
        if self.needs_directory {
            self.choose_model(cx);
        } else if self.pending_model_load {
            self.start_model_load(cx);
        }
    }
    fn refresh_controls(&mut self, cx: &mut Cx) {
        let busy = self.busy || self.inference_busy();
        let state = (
            self.busy,
            self.loading,
            self.ready,
            self.transcript.is_empty(),
            self.stopping,
            self.inference_busy(),
        );
        if self.controls == Some(state) {
            return;
        }
        self.controls = Some(state);
        self.ui.button(cx, ids!(load)).set_action_enabled(cx, !busy);
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.ui
                .button(cx, ids!(choose_model))
                .set_action_enabled(cx, !busy);
            self.ui
                .button(cx, ids!(retry_model))
                .set_action_enabled(cx, !busy);
        }
        self.ui
            .button(cx, ids!(transcribe))
            .set_action_enabled(cx, self.ready && !busy);
        self.ui
            .button(cx, ids!(sample))
            .set_action_enabled(cx, !busy);
        self.ui
            .button(cx, ids!(choose_audio))
            .set_action_enabled(cx, !busy);
        for id in [
            ids!(asr_lang_auto),
            ids!(asr_lang_zh),
            ids!(asr_lang_en),
            ids!(asr_lang_yue),
            ids!(asr_lang_ja),
            ids!(asr_lang_ko),
        ] {
            self.ui.button(cx, id).set_action_enabled(cx, !busy);
        }
        for id in [
            ids!(speed_balanced),
            ids!(speed_low),
            ids!(speed_fast),
            ids!(speed_save),
        ] {
            self.ui.button(cx, id).set_action_enabled(cx, !busy);
        }
        self.ui
            .button(cx, ids!(microphone))
            .set_action_enabled(cx, self.ready && !busy);
        self.ui.button(cx, ids!(stop)).set_visible(cx, self.busy);
        self.ui
            .button(cx, ids!(stop))
            .set_action_enabled(cx, self.busy && !self.stopping);
        self.ui
            .button(cx, ids!(copy))
            .set_action_enabled(cx, !self.transcript.is_empty());
    }

    fn status(&self, cx: &mut Cx, text: &str) {
        let shown = text.replacen("模型已就绪", "转写已就绪", 1);
        self.ui.label(cx, ids!(status)).set_text(cx, &shown);
    }

    fn show_input_status(&self, cx: &mut Cx, text: &str) {
        self.ui
            .view(cx, ids!(input_status_row))
            .set_visible(cx, true);
        self.ui.label(cx, ids!(input_status)).set_text(cx, text);
    }

    fn audio_path_value(&self, cx: &mut Cx) -> String {
        self.ui
            .text_input(cx, ids!(audio_path))
            .text()
            .replace(['\n', '\r'], "")
    }

    fn set_audio_path_text(&self, cx: &mut Cx, path: &str) {
        let clean: Vec<char> = path
            .chars()
            .filter(|ch| *ch != '\n' && *ch != '\r')
            .collect();
        let mut split_at = None;
        let mut best_tail = 0usize;
        for (index, ch) in clean.iter().enumerate() {
            if *ch != '/' || index == 0 {
                continue;
            }
            let tail = clean.len() - index - 1;
            if tail > best_tail && tail <= 48 {
                best_tail = tail;
                split_at = Some(index);
            }
        }
        let shown = if let Some(index) = split_at {
            let head: String = clean[..=index].iter().collect();
            let tail: String = clean[index + 1..].iter().collect();
            format!("{head}\n{tail}")
        } else {
            clean.into_iter().collect()
        };
        self.ui
            .text_input(cx, ids!(audio_path))
            .set_text(cx, &shown);
    }

    fn refresh_audio_file(&self, cx: &mut Cx) {
        let path = self.audio_path_value(cx);
        let trimmed = path.trim();
        let name = if trimmed.is_empty() {
            "尚未选择音频".to_owned()
        } else {
            std::path::Path::new(trimmed)
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .filter(|name| !name.is_empty())
                .unwrap_or_else(|| trimmed.to_owned())
        };
        self.ui.label(cx, ids!(audio_file)).set_text(cx, &name);
        self.ui.label(cx, ids!(audio_path_name)).set_text(cx, &name);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn tab_indicator_target(&self, cx: &mut Cx) -> Option<(f64, f64)> {
        let nav = self.ui.widget(cx, ids!(navigation)).area().rect(cx);
        let id = if self.workspace.indicator_language() {
            ids!(language_tab)
        } else {
            ids!(creative_tab)
        };
        let tab = self.ui.widget(cx, id).area().rect(cx);
        if nav.size.x < 1.0 || tab.size.x < 1.0 {
            return None;
        }
        Some(((tab.pos.x - nav.pos.x).max(0.0), tab.size.x))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn place_tab_indicator(&self, cx: &mut Cx, x: f64, width: f64) {
        if let Some(mut view) = self.ui.view(cx, ids!(tab_indicator)).borrow_mut() {
            view.walk.margin.left = x;
            view.walk.width = Size::Fixed(width.max(28.0));
            view.redraw(cx);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn follow_tab_indicator(&mut self, cx: &mut Cx, animate: bool) -> bool {
        let Some((x, width)) = self.tab_indicator_target(cx) else {
            return false;
        };
        let current = self
            .ui
            .view(cx, ids!(tab_indicator))
            .borrow()
            .map(|view| {
                let width = match view.walk.width {
                    Size::Fixed(width) => width,
                    _ => 48.0,
                };
                (view.walk.margin.left, width)
            })
            .unwrap_or((0.0, 48.0));
        if animate && self.nav_placed {
            self.nav_slide = Some(NavSlide {
                from_x: current.0,
                from_w: current.1,
                started: std::time::Instant::now(),
                to_creative: !self.workspace.indicator_language(),
            });
            self.nav_timer = cx.start_interval(1.0 / 60.0);
            return true;
        }
        if self.nav_placed && (current.0 - x).abs() < 0.5 && (current.1 - width).abs() < 0.5 {
            return false;
        }
        self.place_tab_indicator(cx, x, width);
        self.nav_placed = true;
        false
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn step_tab_indicator(&mut self, cx: &mut Cx) {
        let Some(slide) = &self.nav_slide else {
            cx.stop_timer(self.nav_timer);
            return;
        };
        let started = slide.started;
        let from_x = slide.from_x;
        let from_w = slide.from_w;
        let to_creative = slide.to_creative;
        let Some((x, width)) = self.tab_indicator_target(cx) else {
            return;
        };
        let t = (started.elapsed().as_secs_f64() / 0.28).clamp(0.0, 1.0);
        let page_eased = 1.0 - (1.0 - t) * (1.0 - t);
        let line_eased = (page_eased * 1.7).min(1.0);
        self.place_tab_indicator(
            cx,
            from_x + (x - from_x) * line_eased,
            from_w + (width - from_w) * line_eased,
        );
        self.slide_pages(cx, page_eased, to_creative);
        if t > 0.0 {
            self.reveal_pending_page(cx);
        }
        if t >= 1.0 {
            self.nav_slide = None;
            cx.stop_timer(self.nav_timer);
            self.apply_pending_page(cx);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn slide_pages(&mut self, cx: &mut Cx, eased: f64, to_creative: bool) {
        let pages = self.ui.widget(cx, ids!(pages)).area().rect(cx);
        let width = pages.size.x;
        if width < 40.0 {
            return;
        }
        let shift = if to_creative {
            -width * eased
        } else {
            -width * (1.0 - eased)
        };
        self.ui.view(cx, ids!(language_page)).set_visible(cx, true);
        self.ui.view(cx, ids!(creative_page)).set_visible(cx, true);
        for id in [ids!(language_page), ids!(creative_page)] {
            if let Some(mut view) = self.ui.view(cx, id).borrow_mut() {
                view.walk.width = Size::Fixed(width);
                view.redraw(cx);
            }
        }
        if let Some(mut track) = self.ui.view(cx, ids!(page_track)).borrow_mut() {
            track.walk.width = Size::Fixed(width * 2.0);
            track.walk.margin.left = shift;
            track.redraw(cx);
        }
        self.place_slide_lips(
            cx,
            true,
            pages.pos.x,
            pages.pos.y,
            width,
            pages.size.y,
            shift,
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn place_slide_lips(
        &mut self,
        cx: &mut Cx,
        show: bool,
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        shift: f64,
    ) {
        let mut slivers: Vec<(f64, f64)> = Vec::new();
        if show {
            let gap = 14.0;
            let wide = (width - gap) * 170.0 / 270.0;
            let narrow = width - gap - wide;
            let view_left = x;
            let view_right = x + width;
            let lang_x = x + shift;
            let crea_x = lang_x + width;
            for (card_x, card_w) in [
                (lang_x, wide),
                (lang_x + wide + gap, narrow),
                (crea_x, narrow),
                (crea_x + narrow + gap, wide),
            ] {
                let card_right = card_x + card_w;
                let overlaps = card_right > view_left + 1.0 && card_x < view_right - 1.0;
                let full = card_x >= view_left - 1.0 && card_right <= view_right + 1.0;
                if overlaps && !full {
                    let left = card_x.max(view_left);
                    let right = card_right.min(view_right);
                    if right - left > 1.0 {
                        slivers.push((left, right));
                    }
                }
            }
        }
        slivers.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let slots = [ids!(slide_lip_left), ids!(slide_lip_right)];
        for (index, id) in slots.into_iter().enumerate() {
            let sliver = if index == 0 {
                slivers.first()
            } else {
                slivers.get(slivers.len().saturating_sub(1))
            };
            let distinct = index == 0 || slivers.len() > 1;
            self.ui
                .view(cx, id)
                .set_visible(cx, sliver.is_some() && distinct);
            if let Some((left, right)) = sliver.filter(|_| distinct) {
                if let Some(mut view) = self.ui.view(cx, id).borrow_mut() {
                    view.walk.abs_pos = Some(dvec2(*left, y));
                    view.walk.width = Size::Fixed((*right - *left).max(1.0));
                    view.walk.height = Size::Fixed(height.max(200.0));
                    view.redraw(cx);
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn rest_page_track(&mut self, cx: &mut Cx) {
        for id in [ids!(language_page), ids!(creative_page)] {
            if let Some(mut view) = self.ui.view(cx, id).borrow_mut() {
                view.walk.width = Size::fill();
                view.redraw(cx);
            }
        }
        if let Some(mut track) = self.ui.view(cx, ids!(page_track)).borrow_mut() {
            track.walk.width = Size::fill();
            track.walk.margin.left = 0.0;
            track.redraw(cx);
        }
        let pages = self.ui.widget(cx, ids!(pages)).area().rect(cx);
        self.place_slide_lips(
            cx,
            false,
            pages.pos.x,
            pages.pos.y,
            pages.size.x,
            pages.size.y,
            0.0,
        );
    }

    fn refresh_speed_choices(&self, cx: &mut Cx) {
        #[cfg(not(target_arch = "wasm32"))]
        let choices = [
            (ids!(speed_balanced), 0usize),
            (ids!(speed_low), 1),
            (ids!(speed_fast), 2),
            (ids!(speed_save), 3),
        ];
        #[cfg(target_arch = "wasm32")]
        let choices = [
            (ids!(speed_low), 0usize),
            (ids!(speed_fast), 1),
            (ids!(speed_save), 2),
        ];
        for (id, mode) in choices {
            let name = protocol::mode_label(mode);
            let label = if self.mode == mode {
                format!("● {name}")
            } else {
                name.to_owned()
            };
            self.ui.button(cx, id).set_text(cx, &label);
        }
        self.ui
            .label(cx, ids!(speed_hint))
            .set_text(cx, protocol::mode_hint(self.mode));
    }

    fn asr_language_raw(&self, cx: &mut Cx) -> String {
        let custom = self.ui.text_input(cx, ids!(asr_language)).text();
        let custom = custom.trim();
        if !custom.is_empty() {
            return custom.to_string();
        }
        if self.language_choice.is_empty() {
            "自动".to_owned()
        } else {
            self.language_choice.clone()
        }
    }

    fn refresh_language_choices(&self, cx: &mut Cx) {
        let raw = self.asr_language_raw(cx);
        let selected = match protocol::asr_language(&raw) {
            Ok(None) => Some("自动"),
            Ok(Some(name)) => match name.as_str() {
                "Chinese" => Some("中文"),
                "English" => Some("英文"),
                "Cantonese" => Some("粤语"),
                "Japanese" => Some("日语"),
                "Korean" => Some("韩语"),
                _ => None,
            },
            Err(_) => None,
        };
        let caption = match selected {
            Some(_) => "语言".to_owned(),
            None if protocol::asr_language(&raw).is_err() => "语言 · 无法识别".to_owned(),
            None => format!("语言 · {}", raw.trim()),
        };
        self.ui
            .label(cx, ids!(asr_language_caption))
            .set_text(cx, &caption);
        for (id, name) in [
            (ids!(asr_lang_auto), "自动"),
            (ids!(asr_lang_zh), "中文"),
            (ids!(asr_lang_en), "英文"),
            (ids!(asr_lang_yue), "粤语"),
            (ids!(asr_lang_ja), "日语"),
            (ids!(asr_lang_ko), "韩语"),
        ] {
            let label = if selected == Some(name) {
                format!("● {name}")
            } else {
                name.to_owned()
            };
            self.ui.button(cx, id).set_text(cx, &label);
        }
    }

    fn set_transcript_language(&self, cx: &mut Cx, language: &str) {
        let title = if language.is_empty() {
            "转写文字".to_owned()
        } else {
            format!("转写文字 · {language}")
        };
        self.ui
            .label(cx, ids!(transcript_title))
            .set_text(cx, &title);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn set_model_progress(&self, cx: &mut Cx, progress: f64) {
        crate::engine_ui::set_progress(
            &self.ui,
            cx,
            ids!(model_progress),
            ids!(model_progress_fill),
            progress,
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn start_model_load(&mut self, cx: &mut Cx) {
        if self.runtime.busy() {
            self.pending_model_load = true;
            self.busy = false;
            self.loading = false;
            return;
        }
        self.pending_model_load = false;
        self.needs_directory = false;
        self.busy = true;
        self.loading = true;
        self.ready = false;
        self.ui
            .view(cx, ids!(model_recovery))
            .set_visible(cx, false);
        self.set_model_progress(cx, 0.0);
        self.ui.view(cx, ids!(model_progress)).set_visible(cx, true);
        self.status(cx, "正在加载 Confucius4-R2T2…");
        self.command(cx, "load");
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn choose_audio(&mut self, cx: &mut Cx) {
        if self.picking_model || self.picking_audio {
            return;
        }
        let Some(dialog) = self.runtime.dialog("asr") else {
            return;
        };
        self.model_dialog = Some(dialog);
        self.picking_audio = true;
        self.busy = true;
        let current = std::path::PathBuf::from(self.audio_path_value(cx));
        let mut dialog = FileDialog::new()
            .set_title("选择要转写的 WAV".into())
            .add_filter("WAV 音频".into(), vec!["wav".into()]);
        if let Some(parent) = current.parent().filter(|path| path.is_dir()) {
            dialog = dialog.set_location(parent.to_owned());
        }
        cx.open_select_file_dialog(dialog);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn choose_model(&mut self, cx: &mut Cx) {
        if self.picking_model || self.picking_audio {
            return;
        }
        let Some(dialog) = self.runtime.dialog("asr") else {
            return;
        };
        self.model_dialog = Some(dialog);
        self.needs_directory = false;
        self.picking_model = true;
        self.busy = true;
        self.loading = true;
        let current = std::path::PathBuf::from(self.ui.text_input(cx, ids!(model_path)).text());
        crate::engine_ui::choose_directory(
            cx,
            &current,
            "选择 Confucius4-R2T2 模型目录（包含 config.json、tokenizer.json 和 safetensors 权重）",
        );
    }

    fn apply_update(&mut self, cx: &mut Cx, update: Update) {
        match update.kind.as_str() {
            "ready" => {
                #[cfg(not(target_arch = "wasm32"))]
                let asr_visible = self.workspace.language();
                #[cfg(target_arch = "wasm32")]
                let asr_visible = true;
                if asr_visible && !update.backend.is_empty() {
                    self.ui
                        .label(cx, ids!(backend))
                        .set_text(cx, &update.backend);
                }
                self.ready = true;
                self.busy = false;
                self.loading = false;
                self.status(cx, &update.message);
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.ui
                        .view(cx, ids!(model_progress))
                        .set_visible(cx, false);
                    self.ui
                        .view(cx, ids!(model_recovery))
                        .set_visible(cx, false);
                    if self.remember_model {
                        self.remember_model = false;
                        let path = self.ui.text_input(cx, ids!(model_path)).text();
                        let result = self
                            .settings_path
                            .as_deref()
                            .ok_or_else(|| "找不到用户设置目录".to_owned())
                            .and_then(|settings| {
                                model_location::remember(settings, std::path::Path::new(&path))
                            });
                        if let Err(error) = result {
                            self.status(cx, &format!("模型已就绪 · 目录未能保存：{error}"));
                        }
                    }
                }
            }
            "loading" | "progress" | "notice" => {
                self.status(cx, &update.message);
                #[cfg(not(target_arch = "wasm32"))]
                if update.kind == "loading" {
                    self.set_model_progress(cx, update.progress);
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            "missing" => {
                self.ready = false;
                self.status(cx, &update.message);
                self.ui
                    .view(cx, ids!(model_progress))
                    .set_visible(cx, false);
                self.ui.view(cx, ids!(model_recovery)).set_visible(cx, true);
                self.busy = false;
                self.loading = false;
                self.needs_directory = true;
            }
            "capture" => {
                if self.microphone && self.busy {
                    let bars = (update.level.clamp(0.0, 1.0).sqrt() * 12.0).round() as usize;
                    self.show_input_status(
                        cx,
                        &format!(
                            "{} · 已采集 {:.1}s · 待处理 {:.1}s · [{}{}]",
                            update.message,
                            update.captured_seconds,
                            update.backlog_seconds,
                            "|".repeat(bars),
                            "·".repeat(12 - bars)
                        ),
                    );
                }
            }
            "event" | "done" => {
                // Avoid relaying out the full transcript on empty deltas.
                // Engine messages still preserve every committed delta.
                if cfg!(target_arch = "wasm32") || self.transcript != update.text {
                    self.transcript = update.text;
                    self.ui
                        .text_input(cx, ids!(transcript))
                        .set_text(cx, &self.transcript);
                }
                if !update.language.is_empty() {
                    self.set_transcript_language(cx, &update.language);
                }
                self.ui.label(cx, ids!(metrics)).set_text(
                    cx,
                    &format!(
                        "音频 {:.2} s  ·  推理 {:.2} s  ·  RTF {:.3}  ·  {} 段  ·  本段 {:.0} ms",
                        update.audio_seconds,
                        update.total_compute_ms / 1000.0,
                        update.rtf,
                        update.event_count,
                        update.compute_ms
                    ),
                );
                if update.kind == "done" {
                    self.busy = false;
                    self.stopping = false;
                    if self.microphone {
                        self.show_input_status(
                            cx,
                            &format!("录音已结束 · 已采集 {:.2} s", update.captured_seconds),
                        );
                    }
                    self.microphone = false;
                    self.status(
                        cx,
                        if !update.message.is_empty() {
                            &update.message
                        } else if update.cancelled {
                            "已停止 · 保留已识别内容"
                        } else {
                            "转写完成 · 可再次运行观察预热后的速度"
                        },
                    );
                } else {
                    self.status(
                        cx,
                        if self.stopping {
                            "正在打断识别 · 保留已确认文字…"
                        } else if self.microphone {
                            "正在录音并转写 · 点击停止结束录音"
                        } else {
                            "正在转写 · 界面可以继续操作"
                        },
                    );
                }
            }
            "error" | "fatal" | "load_cancelled" => {
                self.busy = false;
                if self.loading || update.kind == "fatal" {
                    self.ready = false;
                }
                self.loading = false;
                self.stopping = false;
                if self.microphone {
                    self.show_input_status(cx, "麦克风已关闭");
                }
                self.microphone = false;
                self.status(cx, &format!("错误：{}", update.message));
                #[cfg(not(target_arch = "wasm32"))]
                if !self.ready {
                    self.ui
                        .view(cx, ids!(model_progress))
                        .set_visible(cx, false);
                    self.ui.view(cx, ids!(model_recovery)).set_visible(cx, true);
                }
            }
            _ => (),
        }
    }

    fn command(&mut self, cx: &mut Cx, command: &str) {
        let path = self.ui.text_input(cx, ids!(model_path)).text();
        let audio = self.audio_path_value(cx);
        let context = self.ui.text_input(cx, ids!(context)).text();
        let language = if command == "transcribe" || command == "microphone" {
            match protocol::asr_language(&self.asr_language_raw(cx)) {
                Ok(language) => language,
                Err(error) => {
                    self.busy = false;
                    self.microphone = false;
                    self.status(cx, &error);
                    return;
                }
            }
        } else {
            None
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            let Some(engine) = &self.engine else {
                return;
            };
            if command == "stop" {
                engine.cancel();
                return;
            }
            let request = if command == "load" {
                engine::Command::Load(path)
            } else if command == "microphone" {
                engine::Command::Microphone {
                    context,
                    options: protocol::decode_options(self.mode),
                    language,
                }
            } else {
                engine::Command::Transcribe {
                    path: audio,
                    context,
                    options: protocol::decode_options(self.mode),
                    language,
                }
            };
            if let Err(error) = engine.submit(request) {
                self.apply_update(
                    cx,
                    Update::status("error", format!("推理线程不可用：{error}")),
                );
            }
        }
        #[cfg(target_arch = "wasm32")]
        cx.os.from_wasm(web_bridge::FromWasmCommand {
            json: serde_json::json!({
                "type": command, "url": path, "audio": audio, "context": context,
                "config": protocol::decode_options(self.mode),
                "language": language,
            })
            .to_string(),
        });
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
            self.settings_path = model_location::settings_path();
            let directory = model_location::initial_directory(&root, self.settings_path.as_deref());
            self.ui
                .text_input(cx, ids!(model_path))
                .set_text(cx, &directory.to_string_lossy());
            self.ui
                .label(cx, ids!(model_title))
                .set_text(cx, "转写模型");
            self.ui
                .label(cx, ids!(transcript_title))
                .set_text(cx, "转写文字");
            let sample = root.join("artifacts/Confucius4-R2T2-upstream/resources/test.wav");
            let sample = std::fs::canonicalize(&sample).unwrap_or(sample);
            self.set_audio_path_text(cx, &sample.to_string_lossy());
            self.refresh_audio_file(cx);
            self.language_choice = "自动".to_owned();
            self.refresh_language_choices(cx);
            self.refresh_speed_choices(cx);
            self.engine = Some(engine::Engine::spawn_shared(
                SignalToUI::set_ui_signal,
                self.runtime.clone(),
            ));
            self.ui.label(cx, ids!(backend)).set_text(
                cx,
                if cfg!(target_os = "macos") {
                    "Metal · 本机推理"
                } else {
                    "WGPU · 本机推理"
                },
            );
            self.start_model_load(cx);
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.mode = 1;
            self.ui
                .button(cx, ids!(speed_balanced))
                .set_visible(cx, false);
            self.ui
                .text_input(cx, ids!(model_path))
                .set_text(cx, "/models/web/manifest.json");
            self.set_audio_path_text(cx, "/artifacts/Confucius4-R2T2-upstream/resources/test.wav");
            self.ui
                .label(cx, ids!(backend))
                .set_text(cx, "WebGPU · 浏览器推理");
            self.ui.button(cx, ids!(sample)).set_text(cx, "选择音频");
            self.ui
                .button(cx, ids!(choose_audio))
                .set_visible(cx, false);
            self.refresh_audio_file(cx);
            self.language_choice = "自动".to_owned();
            self.refresh_language_choices(cx);
            self.refresh_speed_choices(cx);
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
        #[cfg(not(target_arch = "wasm32"))]
        {
            for action in actions {
                if !self.picking_model && !self.picking_audio {
                    continue;
                }
                match action.downcast_ref::<FileDialogAction>() {
                    Some(FileDialogAction::FileSelected { paths, .. }) if self.picking_audio => {
                        self.model_dialog.take();
                        self.picking_audio = false;
                        self.busy = false;
                        if let Some(path) = paths.first() {
                            self.set_audio_path_text(cx, &path.to_string_lossy());
                            self.refresh_audio_file(cx);
                            self.status(
                                cx,
                                &format!(
                                    "已选择 {} · 可以开始转写",
                                    self.ui.label(cx, ids!(audio_file)).text()
                                ),
                            );
                        }
                    }
                    Some(FileDialogAction::FileCancelled { .. }) if self.picking_audio => {
                        self.model_dialog.take();
                        self.picking_audio = false;
                        self.busy = false;
                    }
                    Some(FileDialogAction::FolderSelected(path)) if self.picking_model => {
                        self.model_dialog.take();
                        self.picking_model = false;
                        let directory = model_location::selected_directory(path);
                        self.ui
                            .text_input(cx, ids!(model_path))
                            .set_text(cx, &directory.to_string_lossy());
                        self.remember_model = true;
                        self.ready = false;
                        self.busy = false;
                        self.loading = false;
                        self.pending_model_load = true;
                        if self.workspace.language() && !self.inference_busy() {
                            self.start_model_load(cx);
                        }
                    }
                    Some(FileDialogAction::FolderCancelled) if self.picking_model => {
                        self.model_dialog.take();
                        self.picking_model = false;
                        self.busy = false;
                        self.loading = false;
                        self.status(cx, "尚未选择模型 · 选择模型目录后自动加载");
                    }
                    _ => (),
                }
            }
            if !self.workspace.language() {
                return;
            }
            if self.ui.button(cx, ids!(choose_audio)).clicked(actions) && !self.busy {
                self.choose_audio(cx);
            }
            if self.ui.button(cx, ids!(choose_model)).clicked(actions) && !self.busy {
                self.choose_model(cx);
            }
            if self.ui.button(cx, ids!(retry_model)).clicked(actions) && !self.busy {
                self.start_model_load(cx);
            }
        }
        let speed_choice = [
            (ids!(speed_balanced), 0usize),
            (
                ids!(speed_low),
                if cfg!(target_arch = "wasm32") { 0 } else { 1 },
            ),
            (
                ids!(speed_fast),
                if cfg!(target_arch = "wasm32") { 1 } else { 2 },
            ),
            (
                ids!(speed_save),
                if cfg!(target_arch = "wasm32") { 2 } else { 3 },
            ),
        ];
        for (id, mode) in speed_choice {
            if self.ui.button(cx, id).clicked(actions) && !self.busy && !self.inference_busy() {
                if mode < protocol::mode_count() {
                    self.mode = mode;
                    self.refresh_speed_choices(cx);
                }
            }
        }
        if self.ui.button(cx, ids!(ping)).clicked(actions) {
            self.pings += 1;
            self.ui.label(cx, ids!(heartbeat)).set_text(
                cx,
                &format!("界面响应 {} 次 · 心跳 {}", self.pings, self.ticks),
            );
        }
        if self.ui.button(cx, ids!(copy)).clicked(actions) {
            #[cfg(not(target_arch = "wasm32"))]
            cx.copy_to_clipboard(&self.transcript);
            // The pinned upstream Web backend leaves CopyToClipboard unhandled.
            #[cfg(target_arch = "wasm32")]
            cx.os.from_wasm(web_bridge::FromWasmCommand {
                json: serde_json::json!({"type": "copy", "text": self.transcript}).to_string(),
            });
        }
        if self.ui.button(cx, ids!(stop)).clicked(actions) && self.busy && !self.stopping {
            self.stopping = true;
            self.command(cx, "stop");
            self.status(cx, "正在打断识别 · 保留已确认文字…");
        }
        if self
            .ui
            .text_input(cx, ids!(audio_path))
            .changed(actions)
            .is_some()
        {
            self.refresh_audio_file(cx);
        }
        if let Some(text) = self.ui.text_input(cx, ids!(asr_language)).changed(actions) {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                self.refresh_language_choices(cx);
            } else if let Some(choice) = ["自动", "中文", "英文", "粤语", "日语", "韩语"]
                .into_iter()
                .find(|choice| *choice == trimmed)
            {
                self.language_choice = choice.to_owned();
                self.ui.text_input(cx, ids!(asr_language)).set_text(cx, "");
                self.refresh_language_choices(cx);
            } else {
                self.refresh_language_choices(cx);
                if let Err(error) = protocol::asr_language(trimmed) {
                    self.status(cx, &error);
                }
            }
        }
        for (id, value) in [
            (ids!(asr_lang_auto), "自动"),
            (ids!(asr_lang_zh), "中文"),
            (ids!(asr_lang_en), "英文"),
            (ids!(asr_lang_yue), "粤语"),
            (ids!(asr_lang_ja), "日语"),
            (ids!(asr_lang_ko), "韩语"),
        ] {
            if self.ui.button(cx, id).clicked(actions) && !self.busy && !self.inference_busy() {
                self.language_choice = value.to_owned();
                self.ui.text_input(cx, ids!(asr_language)).set_text(cx, "");
                self.refresh_language_choices(cx);
            }
        }
        if self.ui.button(cx, ids!(sample)).clicked(actions) && !self.busy {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let sample = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../artifacts/Confucius4-R2T2-upstream/resources/test.wav");
                let sample = std::fs::canonicalize(&sample).unwrap_or(sample);
                self.set_audio_path_text(cx, &sample.to_string_lossy());
                self.refresh_audio_file(cx);
                self.status(cx, "已填入官方示例 · 可以开始转写");
            }
            #[cfg(target_arch = "wasm32")]
            self.command(cx, "choose");
        }
        #[cfg(target_arch = "wasm32")]
        if self.ui.button(cx, ids!(load)).clicked(actions) && !self.busy {
            self.busy = true;
            self.loading = true;
            self.ready = false;
            self.status(cx, "正在载入模型…");
            self.command(cx, "load");
        }
        if self.ui.button(cx, ids!(transcribe)).clicked(actions)
            && !self.busy
            && !self.inference_busy()
        {
            if !self.ready {
                self.status(cx, "请先载入模型");
                return;
            }
            if let Err(error) = protocol::asr_language(&self.asr_language_raw(cx)) {
                self.status(cx, &error);
                return;
            }
            self.busy = true;
            self.transcript.clear();
            self.ui.text_input(cx, ids!(transcript)).set_text(cx, "");
            self.set_transcript_language(cx, "");
            self.status(cx, "开始转写 · 首次运行可能需要较长时间");
            self.command(cx, "transcribe");
        }
        if self.ui.button(cx, ids!(microphone)).clicked(actions)
            && !self.busy
            && !self.inference_busy()
            && self.ready
        {
            if let Err(error) = protocol::asr_language(&self.asr_language_raw(cx)) {
                self.status(cx, &error);
                return;
            }
            self.busy = true;
            self.microphone = true;
            self.stopping = false;
            self.transcript.clear();
            self.ui.text_input(cx, ids!(transcript)).set_text(cx, "");
            self.set_transcript_language(cx, "");
            self.show_input_status(cx, "正在打开默认输入设备…");
            self.status(cx, "正在开启麦克风 · 首次使用请允许录音权限");
            self.command(cx, "microphone");
        }
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
        if self.nav_timer.is_event(event).is_some() {
            self.step_tab_indicator(cx);
        }
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
            let updates: Vec<_> = self
                .engine
                .as_ref()
                .map(|e| e.updates.try_iter().collect())
                .unwrap_or_default();
            for update in updates {
                self.apply_update(cx, update);
            }
        }
        #[cfg(target_arch = "wasm32")]
        if let Event::ToWasmMsg(message) = event {
            use makepad_widgets::makepad_platform::makepad_wasm_bridge::ToWasm;
            if message.id == web_bridge::ToWasmUpdate::live_id() {
                let message = web_bridge::ToWasmUpdate::read_to_wasm(&mut message.as_ref());
                match serde_json::from_str::<Update>(&message.json) {
                    Ok(update) if update.kind == "file" => {
                        self.set_audio_path_text(cx, &update.text);
                        self.refresh_audio_file(cx);
                    }
                    Ok(update) => self.apply_update(cx, update),
                    Err(error) => self.apply_update(
                        cx,
                        Update::status("error", format!("消息解析失败：{error}")),
                    ),
                }
            }
        }
        self.match_event(cx, event);
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.activate_current_page(cx);
            self.refresh_running(cx);
            self.refresh_workspace_controls(cx);
            if let Event::WindowGeomChange(geometry) = event {
                self.refresh_workspace_layout(cx, geometry.new_geom.inner_size.x);
                self.follow_tab_indicator(cx, false);
            }
            if self.nav_slide.is_none() {
                self.follow_tab_indicator(cx, false);
            }
        }
        self.refresh_controls(cx);
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
