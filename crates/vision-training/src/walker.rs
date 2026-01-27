//! Fast directory walking using walkdir
//!
//! Replaces Python os.walk, glob, Path.rglob with native Rust.
//! Expected speedup: 5-10x

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;
use walkdir::WalkDir;

/// Find all image files in a directory recursively
#[pyfunction]
#[pyo3(signature = (root, max_depth=None))]
pub fn find_images(root: &str, max_depth: Option<usize>) -> PyResult<Vec<String>> {
    let extensions = ["jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif"];
    walk_with_extensions(root, &extensions, max_depth)
}

/// Find all JSON files in a directory recursively
#[pyfunction]
#[pyo3(signature = (root, max_depth=None))]
pub fn find_json_files(root: &str, max_depth: Option<usize>) -> PyResult<Vec<String>> {
    walk_with_extensions(root, &["json"], max_depth)
}

/// Walk directory and find files with specific extensions
#[pyfunction]
#[pyo3(signature = (root, extensions, max_depth=None))]
pub fn walk_directory(
    root: &str,
    extensions: Vec<String>,
    max_depth: Option<usize>,
) -> PyResult<Vec<String>> {
    let ext_refs: Vec<&str> = extensions.iter().map(|s| s.as_str()).collect();
    walk_with_extensions(root, &ext_refs, max_depth)
}

fn walk_with_extensions(
    root: &str,
    extensions: &[&str],
    max_depth: Option<usize>,
) -> PyResult<Vec<String>> {
    let root_path = Path::new(root);

    if !root_path.exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            format!("Directory not found: {}", root)
        ));
    }

    let walker = match max_depth {
        Some(depth) => WalkDir::new(root_path).max_depth(depth),
        None => WalkDir::new(root_path),
    };

    let paths: Vec<String> = walker
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            if extensions.is_empty() {
                return true;
            }
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| extensions.iter().any(|ex| ex.eq_ignore_ascii_case(ext)))
                .unwrap_or(false)
        })
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();

    Ok(paths)
}

/// Count files by extension in a directory
#[pyfunction]
pub fn count_by_extension<'py>(
    py: Python<'py>,
    root: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let root_path = Path::new(root);

    let counts: std::collections::HashMap<String, usize> = WalkDir::new(root_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .fold(std::collections::HashMap::new(), |mut acc, entry| {
            let ext = entry
                .path()
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("no_extension")
                .to_lowercase();
            *acc.entry(ext).or_insert(0) += 1;
            acc
        });

    let dict = PyDict::new_bound(py);
    for (ext, count) in counts {
        dict.set_item(ext, count)?;
    }

    Ok(dict)
}

/// Get total size of all files in directory
#[pyfunction]
pub fn get_directory_size(root: &str) -> PyResult<u64> {
    let root_path = Path::new(root);

    let total: u64 = WalkDir::new(root_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum();

    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use tempfile::tempdir;

    #[test]
    fn test_find_images() {
        let temp = tempdir().unwrap();
        let dir = temp.path();

        // Create test files
        File::create(dir.join("test1.jpg")).unwrap();
        File::create(dir.join("test2.png")).unwrap();
        File::create(dir.join("test3.txt")).unwrap();

        let images = walk_with_extensions(dir.to_str().unwrap(), &["jpg", "png"], None).unwrap();
        assert_eq!(images.len(), 2);
    }
}
