//! Fast ZIP compression and extraction
//!
//! Replaces Python zipfile with Rust zip crate.
//! Expected speedup: 4-6x

use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use walkdir::WalkDir;
use zip::write::FileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

/// Extract a ZIP file to a directory
///
/// # Arguments
/// * `zip_path` - Path to the ZIP file
/// * `dest_dir` - Destination directory
///
/// # Returns
/// * List of extracted file paths
#[pyfunction]
pub fn extract_zip(zip_path: &str, dest_dir: &str) -> PyResult<Vec<String>> {
    let file = File::open(zip_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let mut archive = ZipArchive::new(file)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let dest_path = Path::new(dest_dir);
    fs::create_dir_all(dest_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let mut extracted_files = Vec::new();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let outpath = match file.enclosed_name() {
            Some(path) => dest_path.join(path),
            None => continue,
        };

        if file.name().ends_with('/') {
            fs::create_dir_all(&outpath)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        } else {
            if let Some(parent) = outpath.parent() {
                if !parent.exists() {
                    fs::create_dir_all(parent)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
                }
            }

            let mut outfile = File::create(&outpath)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

            std::io::copy(&mut file, &mut outfile)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

            extracted_files.push(outpath.to_string_lossy().to_string());
        }

        // Set permissions on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Some(mode) = file.unix_mode() {
                fs::set_permissions(&outpath, fs::Permissions::from_mode(mode)).ok();
            }
        }
    }

    Ok(extracted_files)
}

/// Compress a directory to a ZIP file
///
/// # Arguments
/// * `source_dir` - Source directory to compress
/// * `zip_path` - Output ZIP file path
/// * `compression_level` - Compression level (0-9, default 6)
///
/// # Returns
/// * Number of files compressed
#[pyfunction]
#[pyo3(signature = (source_dir, zip_path, compression_level=6))]
pub fn compress_directory(
    source_dir: &str,
    zip_path: &str,
    compression_level: u32,
) -> PyResult<usize> {
    let source_path = Path::new(source_dir);
    let zip_file = File::create(zip_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let mut zip = ZipWriter::new(zip_file);

    let options = FileOptions::<()>::default()
        .compression_method(CompressionMethod::Deflated)
        .compression_level(Some(compression_level as i64));

    // Collect all files first (for potential parallel reading)
    let files: Vec<_> = WalkDir::new(source_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .collect();

    let file_count = files.len();

    // Read files in parallel
    let file_data: Vec<_> = files
        .par_iter()
        .map(|entry| {
            let path = entry.path();
            let relative_path = path.strip_prefix(source_path).unwrap_or(path);
            let mut contents = Vec::new();
            if let Ok(mut f) = File::open(path) {
                f.read_to_end(&mut contents).ok();
            }
            (relative_path.to_path_buf(), contents)
        })
        .collect();

    // Write to ZIP sequentially (ZIP format requires sequential writes)
    for (relative_path, contents) in file_data {
        let name = relative_path.to_string_lossy();
        zip.start_file(name.as_ref(), options.clone())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        zip.write_all(&contents)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    }

    zip.finish()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    Ok(file_count)
}

/// Compress files to a ZIP in memory and return bytes
///
/// Useful for streaming responses
#[pyfunction]
#[pyo3(signature = (file_paths, base_dir, compression_level=6))]
pub fn compress_files_to_bytes(
    py: Python<'_>,
    file_paths: Vec<String>,
    base_dir: &str,
    compression_level: u32,
) -> PyResult<Py<pyo3::types::PyBytes>> {
    use std::io::Cursor;

    let base_path = Path::new(base_dir);
    let buffer = Cursor::new(Vec::new());
    let mut zip = ZipWriter::new(buffer);

    let options = FileOptions::<()>::default()
        .compression_method(CompressionMethod::Deflated)
        .compression_level(Some(compression_level as i64));

    // Read files in parallel
    let file_data: Vec<_> = file_paths
        .par_iter()
        .filter_map(|path_str| {
            let path = Path::new(path_str);
            let relative_path = path.strip_prefix(base_path).unwrap_or(path);
            let mut contents = Vec::new();
            File::open(path).ok()?.read_to_end(&mut contents).ok()?;
            Some((relative_path.to_path_buf(), contents))
        })
        .collect();

    for (relative_path, contents) in file_data {
        let name = relative_path.to_string_lossy();
        zip.start_file(name.as_ref(), options.clone())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        zip.write_all(&contents)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    }

    let cursor = zip.finish()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    Ok(pyo3::types::PyBytes::new_bound(py, &cursor.into_inner()).unbind())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_compress_and_extract() {
        let temp_dir = tempdir().unwrap();
        let source_dir = temp_dir.path().join("source");
        let extract_dir = temp_dir.path().join("extract");
        let zip_path = temp_dir.path().join("test.zip");

        // Create test files
        fs::create_dir_all(&source_dir).unwrap();
        let mut f1 = File::create(source_dir.join("file1.txt")).unwrap();
        f1.write_all(b"Hello, World!").unwrap();

        // Compress
        let count = compress_directory(
            source_dir.to_str().unwrap(),
            zip_path.to_str().unwrap(),
            6,
        ).unwrap();
        assert_eq!(count, 1);

        // Extract
        let files = extract_zip(
            zip_path.to_str().unwrap(),
            extract_dir.to_str().unwrap(),
        ).unwrap();
        assert_eq!(files.len(), 1);

        // Verify content
        let content = fs::read_to_string(extract_dir.join("file1.txt")).unwrap();
        assert_eq!(content, "Hello, World!");
    }
}
