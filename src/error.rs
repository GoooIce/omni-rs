#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("invalid audio: {0}")]
    Audio(String),
    #[error("checkpoint: {0}")]
    Checkpoint(String),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("inference: {0}")]
    Inference(String),
    #[error("the streaming session has already finished")]
    Finished,
    #[error("the streaming session was interrupted")]
    Interrupted,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
