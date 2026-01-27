//! vision-training: Rust optimizations for training-server
//!
//! High-performance replacements for CPU-bound operations:
//! - EfficientAD inference pipeline (4-10x faster)
//! - Parallel file downloads (3-4x faster)
//! - LabelMe to YOLO conversion (10x faster)
//! - Directory walking (5-10x faster)

use pyo3::prelude::*;

pub mod efficientad;
pub mod downloader;
pub mod labelme_yolo;
pub mod walker;

/// Python module for vision-training
#[pymodule]
fn vision_training(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // EfficientAD functions
    m.add_function(wrap_pyfunction!(efficientad::compute_anomaly_maps, m)?)?;
    m.add_function(wrap_pyfunction!(efficientad::find_bounding_boxes, m)?)?;
    m.add_function(wrap_pyfunction!(efficientad::generate_overlay, m)?)?;
    m.add_function(wrap_pyfunction!(efficientad::generate_mask, m)?)?;
    m.add_function(wrap_pyfunction!(efficientad::array_to_base64, m)?)?;
    m.add_function(wrap_pyfunction!(efficientad::normalize_heatmap, m)?)?;
    m.add_function(wrap_pyfunction!(efficientad::compute_percentiles, m)?)?;

    // Downloader functions
    m.add_function(wrap_pyfunction!(downloader::download_files, m)?)?;
    m.add_function(wrap_pyfunction!(downloader::download_file, m)?)?;

    // LabelMe/YOLO conversion
    m.add_function(wrap_pyfunction!(labelme_yolo::convert_labelme_to_yolo, m)?)?;
    m.add_function(wrap_pyfunction!(labelme_yolo::convert_labelme_dir_to_yolo, m)?)?;
    m.add_function(wrap_pyfunction!(labelme_yolo::fix_json_image_paths, m)?)?;

    // Walker functions
    m.add_function(wrap_pyfunction!(walker::find_images, m)?)?;
    m.add_function(wrap_pyfunction!(walker::find_json_files, m)?)?;
    m.add_function(wrap_pyfunction!(walker::walk_directory, m)?)?;

    Ok(())
}
