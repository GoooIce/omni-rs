//! ChatML conversation rendering and image-placeholder expansion, mirroring the
//! official MiniCPM-V-4.7 chat template byte for byte.

use std::fmt;

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const IMAGE_PAD: &str = "<|image_pad|>";
pub const VIDEO_PAD: &str = "<|video_pad|>";
pub const IMAGE_START: &str = "<image>";
pub const IMAGE_END: &str = "</image>";
pub const SLICE_START: &str = "<slice>";
pub const SLICE_END: &str = "</slice>";
pub const IMAGE_ID_START: &str = "<image_id>";
pub const IMAGE_ID_END: &str = "</image_id>";

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum ChatError {
    #[error("no messages provided")]
    Empty,
    #[error("system message must be first")]
    SystemNotFirst,
    #[error("system message cannot contain images or videos")]
    SystemMedia,
    #[error("no user query found in messages")]
    NoUserQuery,
    #[error("unexpected message role: {0}")]
    UnexpectedRole(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Part {
    Text(String),
    Image,
    Video,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Part>,
}

impl Message {
    /// A plain text message in one argument.
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![Part::Text(text.into())],
        }
    }
}

/// The template renders each content item, then trims the joined string.
fn render_parts(parts: &[Part], system: bool) -> Result<String, ChatError> {
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            Part::Text(text) => out.push(text.clone()),
            Part::Image => {
                if system {
                    return Err(ChatError::SystemMedia);
                }
                out.push(IMAGE_PAD.to_string());
            }
            Part::Video => {
                if system {
                    return Err(ChatError::SystemMedia);
                }
                out.push(VIDEO_PAD.to_string());
            }
        }
    }
    Ok(out.join("\n").trim().to_string())
}

/// The last user message that is not a wrapped tool response anchors assistant
/// reasoning formatting; every assistant turn before it gets an empty think block.
fn last_query_index(messages: &[Message]) -> Result<usize, ChatError> {
    for (index, message) in messages.iter().enumerate().rev() {
        if message.role != Role::User {
            continue;
        }
        let content = render_parts(&message.content, false)?;
        let wrapped =
            content.starts_with("<tool_response>") && content.ends_with("</tool_response>");
        if !wrapped {
            return Ok(index);
        }
    }
    Err(ChatError::NoUserQuery)
}

fn split_think(content: &str) -> (String, String) {
    if let Some((before, after)) = content.split_once("</think>") {
        let reasoning = before
            .trim_end_matches('\n')
            .rsplit("<think>")
            .next()
            .unwrap_or("")
            .trim_start_matches('\n');
        let answer = after.trim_start_matches('\n');
        (reasoning.to_string(), answer.to_string())
    } else {
        (String::new(), content.to_string())
    }
}

/// Render a conversation exactly as the official `apply_chat_template` does.
///
/// `enable_thinking` defaults to enabled: the generation prompt opens a
/// `<think>` block for the model to fill; `Some(false)` closes it empty.
pub fn render(
    messages: &[Message],
    add_generation_prompt: bool,
    enable_thinking: Option<bool>,
) -> Result<String, ChatError> {
    if messages.is_empty() {
        return Err(ChatError::Empty);
    }
    let mut out = String::new();
    if messages[0].role == Role::System {
        let content = render_parts(&messages[0].content, true)?;
        out.push_str(IM_START);
        out.push_str("system\n");
        out.push_str(&content);
        out.push_str(IM_END);
        out.push('\n');
    }
    let anchor = last_query_index(messages)?;
    for (index, message) in messages.iter().enumerate() {
        let content = render_parts(&message.content, message.role == Role::System)?;
        match message.role {
            Role::System if index == 0 => {}
            Role::System => return Err(ChatError::SystemNotFirst),
            Role::User => {
                out.push_str(IM_START);
                out.push_str("user\n");
                out.push_str(&content);
                out.push_str(IM_END);
                out.push('\n');
            }
            Role::Assistant => {
                let (reasoning, answer) = split_think(&content);
                out.push_str(IM_START);
                out.push_str("assistant\n");
                if index > anchor && !reasoning.trim().is_empty() {
                    out.push_str("<think>\n");
                    out.push_str(&reasoning);
                    out.push_str("\n</think>\n\n");
                } else {
                    out.push_str("<think>\n\n</think>\n\n");
                }
                out.push_str(&answer);
                out.push_str(IM_END);
                out.push('\n');
            }
        }
    }
    if add_generation_prompt {
        out.push_str(IM_START);
        out.push_str("assistant\n");
        if enable_thinking == Some(false) {
            out.push_str("<think>\n\n</think>\n\n");
        } else {
            out.push_str("<think>\n");
        }
    }
    Ok(out)
}

/// One visual crop's patch grid, in patches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    pub height: usize,
    pub width: usize,
}

impl Grid {
    pub fn tokens(&self, divisor: usize) -> usize {
        self.height * self.width / divisor
    }
}

/// Expand one image into marker-wrapped placeholder text, mirroring the official
/// processor: the thumbnail first, then `rows` rows of `cols` slice crops joined
/// by newlines. `divisor` is 16 in the default "16x" downsample mode.
pub(crate) fn image_replacement(
    crops: &[Grid],
    rows: usize,
    cols: usize,
    divisor: usize,
    token: &str,
) -> String {
    let thumbnail = crops.first().map_or(0, |g| g.tokens(divisor));
    let mut out = format!("{IMAGE_START}{}{IMAGE_END}", token.repeat(thumbnail),);
    if rows > 0 && cols > 0 {
        let per_slice = crops.get(1).map_or(0, |g| g.tokens(divisor));
        let slice = format!("{SLICE_START}{}{SLICE_END}", token.repeat(per_slice));
        let mut row_lengths = Vec::with_capacity(rows);
        for _ in 0..rows {
            row_lengths.push(slice.repeat(cols));
        }
        out.push_str(&row_lengths.join("\n"));
    }
    out
}

/// Local image ids prefix each replacement, e.g. `<image_id>0</image_id>`.
pub(crate) fn image_id_prefix(local_index: usize) -> String {
    format!("{IMAGE_ID_START}{local_index}{IMAGE_ID_END}")
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fixtures(name: &str) -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(format!("{name}.json"));
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn parse_part(value: &Value) -> Part {
        match value["type"].as_str().unwrap_or("text") {
            "image" => Part::Image,
            "video" => Part::Video,
            _ => Part::Text(value["text"].as_str().unwrap().to_string()),
        }
    }

    fn parse_message(value: &Value) -> Message {
        let role = match value["role"].as_str().unwrap() {
            "system" => Role::System,
            "user" => Role::User,
            _ => Role::Assistant,
        };
        let content = match &value["content"] {
            Value::String(text) => vec![Part::Text(text.clone())],
            parts => parts.as_array().unwrap().iter().map(parse_part).collect(),
        };
        Message { role, content }
    }

    #[test]
    fn renders_match_official_template() {
        let cases = fixtures("chat-template");
        let object = cases.as_object().expect("fixture object");
        assert!(!object.is_empty());
        for (name, case) in object {
            let messages: Vec<Message> = case["messages"]
                .as_array()
                .unwrap()
                .iter()
                .map(parse_message)
                .collect();
            let rendered = render(
                &messages,
                case["add_generation_prompt"].as_bool().unwrap(),
                case["enable_thinking"].as_bool(),
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(rendered, case["rendered"].as_str().unwrap(), "{name}");
        }
    }

    #[test]
    fn image_expansions_match_official_processor() {
        let cases = fixtures("image-pad-expansion");
        let object = cases.as_object().expect("fixture object");
        assert!(object.len() >= 4);
        for (name, case) in object {
            let crops: Vec<Grid> = case["target_sizes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|ts| {
                    let sizes = ts.as_array().unwrap();
                    Grid {
                        height: sizes[0].as_u64().unwrap() as usize,
                        width: sizes[1].as_u64().unwrap() as usize,
                    }
                })
                .collect();
            let grid = case["grid"].as_array().unwrap();
            let replacement = image_replacement(
                &crops,
                grid[0].as_u64().unwrap() as usize,
                grid[1].as_u64().unwrap() as usize,
                16,
                IMAGE_PAD,
            );
            assert_eq!(replacement, case["replacement"].as_str().unwrap(), "{name}");
        }
    }

    #[test]
    fn template_rejects_invalid_conversations() {
        assert_eq!(render(&[], true, None).unwrap_err(), ChatError::Empty);
        let only_tool = vec![Message::text(
            Role::User,
            "<tool_response>x</tool_response>",
        )];
        assert_eq!(
            render(&only_tool, true, None).unwrap_err(),
            ChatError::NoUserQuery
        );
        let late_system = vec![
            Message::text(Role::User, "hi"),
            Message::text(Role::System, "late"),
        ];
        assert_eq!(
            render(&late_system, true, None).unwrap_err(),
            ChatError::SystemNotFirst
        );
        let media_system = vec![Message {
            role: Role::System,
            content: vec![Part::Image],
        }];
        assert_eq!(
            render(&media_system, true, None).unwrap_err(),
            ChatError::SystemMedia
        );
    }

    #[test]
    fn token_counts_follow_grid_area_over_divisor() {
        assert_eq!(
            Grid {
                height: 70,
                width: 70
            }
            .tokens(16),
            306
        );
        assert_eq!(
            Grid {
                height: 36,
                width: 36
            }
            .tokens(16),
            81
        );
        assert_eq!(
            Grid {
                height: 28,
                width: 70
            }
            .tokens(16),
            122
        );
    }
}
