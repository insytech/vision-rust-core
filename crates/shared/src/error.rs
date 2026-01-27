//! Common error types for vision-rust-core

use thiserror::Error;

#[derive(Error, Debug)]
pub enum VisionError {
    #[error("Image processing error: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Encoding error: {0}")]
    EncodingError(String),

    #[error("Array shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },
}

pub type VisionResult<T> = Result<T, VisionError>;
