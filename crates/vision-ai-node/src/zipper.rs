//! Fast ZIP compression for Node.js
//!
//! Replaces archiver with Rust zip crate.
//! Expected speedup: 4-6x

use neon::prelude::*;
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use walkdir::WalkDir;
use zip::write::FileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

/// Create ZIP from a directory
///
/// # Arguments
/// * `sourceDir` - Directory to compress
/// * `zipPath` - Output ZIP file path
/// * `compressionLevel` - Compression level 0-9 (default 6)
///
/// # Returns
/// * Number of files compressed
pub fn create_zip(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let source_dir = cx.argument::<JsString>(0)?.value(&mut cx);
    let zip_path = cx.argument::<JsString>(1)?.value(&mut cx);
    let compression_level = cx
        .argument_opt(2)
        .and_then(|v| v.downcast::<JsNumber, _>(&mut cx).ok())
        .map(|n| n.value(&mut cx) as u32)
        .unwrap_or(6);

    let count = create_zip_impl(&source_dir, &zip_path, compression_level)
        .map_err(|e| cx.throw_error::<_, ()>(e).unwrap_err())?;

    Ok(cx.number(count as f64))
}

fn create_zip_impl(source_dir: &str, zip_path: &str, compression_level: u32) -> Result<usize, String> {
    let source_path = Path::new(source_dir);
    let zip_file = File::create(zip_path).map_err(|e| e.to_string())?;

    let mut zip = ZipWriter::new(zip_file);
    let options = FileOptions::<()>::default()
        .compression_method(CompressionMethod::Deflated)
        .compression_level(Some(compression_level as i64));

    // Collect files
    let files: Vec<_> = WalkDir::new(source_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .collect();

    let file_count = files.len();

    // Read files in parallel
    let file_data: Vec<_> = files
        .par_iter()
        .filter_map(|entry| {
            let path = entry.path();
            let relative_path = path.strip_prefix(source_path).ok()?;
            let mut contents = Vec::new();
            File::open(path).ok()?.read_to_end(&mut contents).ok()?;
            Some((relative_path.to_path_buf(), contents))
        })
        .collect();

    // Write to ZIP sequentially
    for (relative_path, contents) in file_data {
        let name = relative_path.to_string_lossy();
        zip.start_file(name.as_ref(), options.clone())
            .map_err(|e| e.to_string())?;
        zip.write_all(&contents).map_err(|e| e.to_string())?;
    }

    zip.finish().map_err(|e| e.to_string())?;
    Ok(file_count)
}

/// Extract ZIP to directory
///
/// # Arguments
/// * `zipPath` - ZIP file to extract
/// * `destDir` - Destination directory
///
/// # Returns
/// * Array of extracted file paths
pub fn extract_zip(mut cx: FunctionContext) -> JsResult<JsArray> {
    let zip_path = cx.argument::<JsString>(0)?.value(&mut cx);
    let dest_dir = cx.argument::<JsString>(1)?.value(&mut cx);

    let files = extract_zip_impl(&zip_path, &dest_dir)
        .map_err(|e| cx.throw_error::<_, ()>(e).unwrap_err())?;

    let js_arr = JsArray::new(&mut cx, files.len());
    for (i, path) in files.iter().enumerate() {
        let js_str = cx.string(path);
        js_arr.set(&mut cx, i as u32, js_str)?;
    }

    Ok(js_arr)
}

fn extract_zip_impl(zip_path: &str, dest_dir: &str) -> Result<Vec<String>, String> {
    let file = File::open(zip_path).map_err(|e| e.to_string())?;
    let mut archive = ZipArchive::new(file).map_err(|e| e.to_string())?;

    let dest_path = Path::new(dest_dir);
    fs::create_dir_all(dest_path).map_err(|e| e.to_string())?;

    let mut extracted = Vec::new();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i).map_err(|e| e.to_string())?;

        let outpath = match file.enclosed_name() {
            Some(path) => dest_path.join(path),
            None => continue,
        };

        if file.name().ends_with('/') {
            fs::create_dir_all(&outpath).map_err(|e| e.to_string())?;
        } else {
            if let Some(parent) = outpath.parent() {
                if !parent.exists() {
                    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                }
            }

            let mut outfile = File::create(&outpath).map_err(|e| e.to_string())?;
            std::io::copy(&mut file, &mut outfile).map_err(|e| e.to_string())?;
            extracted.push(outpath.to_string_lossy().to_string());
        }
    }

    Ok(extracted)
}

/// Create ZIP from specific file list
///
/// # Arguments
/// * `files` - Array of {path, name} objects
/// * `zipPath` - Output ZIP path
/// * `compressionLevel` - Compression level (default 6)
///
/// # Returns
/// * Number of files added
pub fn create_zip_from_files(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let files_arr = cx.argument::<JsArray>(0)?;
    let zip_path = cx.argument::<JsString>(1)?.value(&mut cx);
    let compression_level = cx
        .argument_opt(2)
        .and_then(|v| v.downcast::<JsNumber, _>(&mut cx).ok())
        .map(|n| n.value(&mut cx) as u32)
        .unwrap_or(6);

    // Extract file info from JS array
    let mut files: Vec<(String, String)> = Vec::new();
    for i in 0..files_arr.len(&mut cx) {
        let obj: Handle<JsObject> = files_arr.get(&mut cx, i)?;
        let path: Handle<JsString> = obj.get(&mut cx, "path")?;
        let name: Handle<JsString> = obj.get(&mut cx, "name")?;
        files.push((path.value(&mut cx), name.value(&mut cx)));
    }

    let count = create_zip_from_files_impl(&files, &zip_path, compression_level)
        .map_err(|e| cx.throw_error::<_, ()>(e).unwrap_err())?;

    Ok(cx.number(count as f64))
}

fn create_zip_from_files_impl(
    files: &[(String, String)],
    zip_path: &str,
    compression_level: u32,
) -> Result<usize, String> {
    let zip_file = File::create(zip_path).map_err(|e| e.to_string())?;
    let mut zip = ZipWriter::new(zip_file);

    let options = FileOptions::<()>::default()
        .compression_method(CompressionMethod::Deflated)
        .compression_level(Some(compression_level as i64));

    // Read files in parallel
    let file_data: Vec<_> = files
        .par_iter()
        .filter_map(|(path, name)| {
            let mut contents = Vec::new();
            File::open(path).ok()?.read_to_end(&mut contents).ok()?;
            Some((name.clone(), contents))
        })
        .collect();

    let count = file_data.len();

    for (name, contents) in file_data {
        zip.start_file(&name, options.clone())
            .map_err(|e| e.to_string())?;
        zip.write_all(&contents).map_err(|e| e.to_string())?;
    }

    zip.finish().map_err(|e| e.to_string())?;
    Ok(count)
}
