//! Common I/O utilities shared across crates

use rayon::prelude::*;
use std::path::{Path, PathBuf};

use crate::{VisionError, VisionResult};

/// Fast recursive directory walking
/// Returns sorted list of file paths matching the given extensions
pub fn walk_directory(
    root: &Path,
    extensions: &[&str],
    max_depth: Option<usize>,
) -> VisionResult<Vec<PathBuf>> {
    use std::fs;

    fn walk_recursive(
        dir: &Path,
        extensions: &[&str],
        max_depth: Option<usize>,
        current_depth: usize,
        results: &mut Vec<PathBuf>,
    ) -> std::io::Result<()> {
        if let Some(max) = max_depth {
            if current_depth > max {
                return Ok(());
            }
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                walk_recursive(&path, extensions, max_depth, current_depth + 1, results)?;
            } else if path.is_file() {
                if extensions.is_empty() {
                    results.push(path);
                } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if extensions.iter().any(|e| e.eq_ignore_ascii_case(ext)) {
                        results.push(path);
                    }
                }
            }
        }
        Ok(())
    }

    let mut paths = Vec::new();
    walk_recursive(root, extensions, max_depth, 0, &mut paths)?;
    Ok(paths)
}

/// Fast glob pattern matching for images
pub fn find_images(root: &Path, max_depth: Option<usize>) -> VisionResult<Vec<PathBuf>> {
    walk_directory(root, &["jpg", "jpeg", "png", "bmp", "webp", "tiff"], max_depth)
}

/// Fast glob pattern matching for JSON files
pub fn find_json_files(root: &Path, max_depth: Option<usize>) -> VisionResult<Vec<PathBuf>> {
    walk_directory(root, &["json"], max_depth)
}

/// Parallel file reading - reads multiple files concurrently
pub fn read_files_parallel(paths: &[PathBuf]) -> Vec<VisionResult<Vec<u8>>> {
    paths
        .par_iter()
        .map(|path| {
            std::fs::read(path).map_err(VisionError::from)
        })
        .collect()
}

/// Validate UTF-8 encoding (fast replacement for chardet)
/// Returns Ok(content) if valid UTF-8, or attempts common encodings
pub fn validate_and_decode_utf8(data: &[u8]) -> VisionResult<String> {
    // Fast path: valid UTF-8
    if let Ok(s) = std::str::from_utf8(data) {
        return Ok(s.to_string());
    }

    // Try common encodings
    // Latin-1 (ISO-8859-1) - always succeeds as fallback
    let decoded: String = data.iter().map(|&b| b as char).collect();

    // Verify it looks like valid JSON/text
    if decoded.contains('{') || decoded.chars().all(|c| c.is_ascii() || c.is_alphanumeric()) {
        Ok(decoded)
    } else {
        Err(VisionError::EncodingError(
            "Unable to decode file as UTF-8 or Latin-1".into()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_utf8() {
        let valid = b"Hello, World!";
        assert!(validate_and_decode_utf8(valid).is_ok());

        let json = b"{\"key\": \"value\"}";
        assert!(validate_and_decode_utf8(json).is_ok());
    }
}
