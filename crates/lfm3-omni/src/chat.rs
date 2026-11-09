//! Render the checkpoint's own conversation template and tokenize its output.
use minijinja::{
    AutoEscape, Environment, Error as TemplateError, ErrorKind, context,
    value::{Serde, Value},
};
use omni_r2t2::{Error, Result};
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub struct RenderOptions {
    pub add_generation_prompt: bool,
    pub preserve_thinking: bool,
    pub enable_thinking: bool,
}
impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            preserve_thinking: false,
            enable_thinking: false,
        }
    }
}

/// Uses the supplied checkpoint template, including tool definitions and calls.
pub struct ChatTemplate {
    source: String,
    bos_token: String,
}
impl ChatTemplate {
    pub fn load(directory: &Path) -> Result<Self> {
        let source = std::fs::read_to_string(directory.join("chat_template.jinja"))?;
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("tokenizer_config.json"))?)?;
        let bos = config
            .get("bos_token")
            .and_then(|v| {
                v.as_str()
                    .or_else(|| v.get("content").and_then(|c| c.as_str()))
            })
            .ok_or_else(|| Error::Config("tokenizer_config.json lacks bos_token".into()))?;
        Ok(Self::from_source(&source, bos))
    }
    pub fn from_source(source: &str, bos_token: &str) -> Self {
        // Hugging Face's generation block only marks supervised tokens. It does
        // not change rendering; keep its whitespace control in the substitute.
        let source = source
            .replace("{%- generation -%}", "{%- if true -%}")
            .replace("{%- endgeneration -%}", "{%- endif -%}");
        Self {
            source,
            bos_token: bos_token.into(),
        }
    }
    pub fn render(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        options: RenderOptions,
    ) -> Result<String> {
        let mut env = Environment::new();
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<String, TemplateError> {
                Err(TemplateError::new(ErrorKind::InvalidOperation, message))
            },
        );
        // Transformers' template filter uses json.dumps with Unicode preserved
        // and default ", "/": " separators, not HTML-escaped compact JSON.
        env.add_filter(
            "tojson",
            |value: Value| -> std::result::Result<String, TemplateError> {
                let value = serde_json::to_value(value)
                    .map_err(|e| TemplateError::new(ErrorKind::InvalidOperation, e.to_string()))?;
                Ok(python_json(&value))
            },
        );
        let template = env
            .template_from_str(&self.source)
            .map_err(template_error)?;
        template.render(context! {
            bos_token => &self.bos_token, messages => Serde(messages), tools => Serde(tools),
            add_generation_prompt => options.add_generation_prompt,
            preserve_thinking => options.preserve_thinking, enable_thinking => options.enable_thinking,
        }).map_err(template_error)
    }

    /// Render new messages after an already cached assistant turn. The BOS and
    /// tool system prompt are already in that cache, but `tools` is still
    /// available to the template when it expands tool references.
    pub fn render_suffix(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        options: RenderOptions,
    ) -> Result<String> {
        if messages.is_empty()
            || messages
                .iter()
                .any(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
        {
            return Err(Error::Tokenizer(
                "a conversation continuation needs non-system messages".into(),
            ));
        }
        let prefix = self.render(
            &[],
            tools,
            RenderOptions {
                add_generation_prompt: false,
                ..options
            },
        )?;
        let rendered = self.render(messages, tools, options)?;
        rendered
            .strip_prefix(&prefix)
            .map(str::to_owned)
            .ok_or_else(|| {
                Error::Tokenizer(
                    "chat template cannot render this continuation without repeating its prefix"
                        .into(),
                )
            })
    }
}
fn template_error(error: TemplateError) -> Error {
    Error::Tokenizer(format!("chat template: {error:#}"))
}

fn python_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Array(items) => format!(
            "[{}]",
            items.iter().map(python_json).collect::<Vec<_>>().join(", ")
        ),
        serde_json::Value::Object(items) => format!(
            "{{{}}}",
            items
                .iter()
                .map(|(k, v)| format!(
                    "{}: {}",
                    serde_json::to_string(k).expect("JSON string"),
                    python_json(v)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => serde_json::to_string(value).expect("valid JSON value"),
    }
}

/// Checkpoint tokenizer. Rendered templates already contain their special tokens.
pub struct Tokenizer(tokenizers::Tokenizer);
impl Tokenizer {
    pub fn load(directory: &Path) -> Result<Self> {
        tokenizers::Tokenizer::from_file(directory.join("tokenizer.json"))
            .map(Self)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }
    pub fn encode(&self, rendered: &str) -> Result<Vec<u32>> {
        self.0
            .encode(rendered, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.0
            .decode(ids, skip_special_tokens)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }
}
