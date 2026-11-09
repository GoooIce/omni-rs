use omni_lfm3::chat::{ChatTemplate, RenderOptions, Tokenizer};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::PathBuf;

#[test]
fn configured_template_renders_ordered_conversation() {
    let template = ChatTemplate::from_source(
        "{{ bos_token }}{% for m in messages %}[{{ m.role }}]{{ m.content }}{% endfor %}",
        "<s>",
    );
    let messages = vec![
        json!({"role":"user", "content":"你好"}),
        json!({"role":"assistant", "content":"hello"}),
    ];
    assert_eq!(
        template
            .render(&messages, None, RenderOptions::default())
            .unwrap(),
        "<s>[user]你好[assistant]hello"
    );
}

#[test]
fn continuation_omits_bos_and_repeated_tool_system_prompt() {
    let template = ChatTemplate::from_source(
        "{{ bos_token }}{% if tools %}[tools={{ tools|length }}]{% endif %}{% for m in messages %}[{{ m.role }}]{{ m.content }}{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}",
        "<s>",
    );
    let tools = vec![json!({"name":"calculator"})];
    let suffix = vec![json!({"role":"tool", "content":"2"})];
    assert_eq!(
        template
            .render_suffix(&suffix, Some(&tools), RenderOptions::default())
            .unwrap(),
        "[tool]2[assistant]"
    );
    assert!(
        template
            .render_suffix(
                &[json!({"role":"system", "content":"unexpected"})],
                None,
                RenderOptions::default()
            )
            .is_err()
    );
}

#[derive(Deserialize)]
struct Case {
    name: String,
    messages: Vec<Value>,
    tools: Option<Vec<Value>>,
    #[serde(default)]
    preserve_thinking: bool,
    #[serde(default)]
    enable_thinking: bool,
    expected: String,
    tokens: Vec<u32>,
}

#[test]
#[ignore = "requires the official model template and tokenizer files"]
fn official_template_and_tokenizer_match_reference_examples() {
    let model = PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required"));
    let template = ChatTemplate::load(&model).unwrap();
    let tokenizer = Tokenizer::load(&model).unwrap();
    let cases: Vec<Case> =
        serde_json::from_str(include_str!("fixtures/chat_examples.json")).unwrap();
    for case in cases {
        let rendered = template
            .render(
                &case.messages,
                case.tools.as_deref(),
                RenderOptions {
                    add_generation_prompt: true,
                    preserve_thinking: case.preserve_thinking,
                    enable_thinking: case.enable_thinking,
                },
            )
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert_eq!(rendered, case.expected, "{}", case.name);
        assert_eq!(
            tokenizer.encode(&rendered).unwrap(),
            case.tokens,
            "{}",
            case.name
        );
    }
}

#[test]
#[ignore = "requires the official model template and tokenizer files"]
fn cached_multiturn_suffix_matches_complete_official_template() {
    let model = PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required"));
    let template = ChatTemplate::load(&model).unwrap();
    let tokenizer = Tokenizer::load(&model).unwrap();
    let first = json!({"role":"user", "content":"What is 1+1?"});
    let assistant = json!({"role":"assistant", "content":"2"});
    let next = json!({"role":"user", "content":"And 2+2?"});
    let opts = RenderOptions::default();
    let first_prompt = template
        .render(std::slice::from_ref(&first), None, opts)
        .unwrap();
    let suffix = template
        .render_suffix(std::slice::from_ref(&next), None, opts)
        .unwrap();
    let rebuilt = format!("{first_prompt}2<|im_end|>\n{suffix}");
    let complete = template
        .render(&[first, assistant, next], None, opts)
        .unwrap();
    assert_eq!(rebuilt, complete);
    assert_eq!(
        tokenizer.encode(&rebuilt).unwrap(),
        tokenizer.encode(&complete).unwrap()
    );
}
