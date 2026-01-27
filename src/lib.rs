//! vision-rust-core workspace root
//!
//! This crate serves as the workspace root and contains benchmarks.
//! The actual functionality is in the individual crates:
//! - `vision-storage`: PyO3 bindings for storage-server
//! - `vision-training`: PyO3 bindings for training-server
//! - `vision-ai-node`: Neon bindings for vision-ai Node.js server
//! - `shared`: Common utilities

pub use shared;
