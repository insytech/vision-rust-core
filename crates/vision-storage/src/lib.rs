//! vision-storage: Rust optimizations for storage-server
//!
//! High-performance replacements for CPU-bound operations:
//! - Thumbnail generation (5-10x faster than PIL)
//! - Encoding detection (50-100x faster than chardet)
//! - COCO to LabelMe parsing (10-20x faster)
//! - ZIP operations (4-6x faster)

use pyo3::prelude::*;

pub mod thumbnail;
pub mod encoding;
pub mod coco_parser;
pub mod zip_ops;

use thumbnail::generate_thumbnail;
use encoding::{verify_encoding, verify_encoding_bytes};
use coco_parser::parse_coco_to_labelme;
use zip_ops::{extract_zip, compress_directory};

/// Python module for vision-storage
#[pymodule]
fn vision_storage(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Thumbnail functions
    m.add_function(wrap_pyfunction!(generate_thumbnail, m)?)?;

    // Encoding functions
    m.add_function(wrap_pyfunction!(verify_encoding, m)?)?;
    m.add_function(wrap_pyfunction!(verify_encoding_bytes, m)?)?;

    // COCO parser functions
    m.add_function(wrap_pyfunction!(parse_coco_to_labelme, m)?)?;

    // ZIP functions
    m.add_function(wrap_pyfunction!(extract_zip, m)?)?;
    m.add_function(wrap_pyfunction!(compress_directory, m)?)?;

    Ok(())
}
