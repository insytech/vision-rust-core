//! Shared utilities for vision-rust-core
//!
//! This crate contains common functionality used across vision-storage,
//! vision-training, and vision-ai-node.

pub mod image_ops;
pub mod io_utils;
pub mod error;

pub use error::{VisionError, VisionResult};
