//! Fast parallel file downloader using tokio + reqwest
//!
//! Replaces Python httpx/aiohttp with native async Rust.
//! Expected speedup: 3-4x

use futures::stream::{self, StreamExt};
use pyo3::prelude::*;
use std::path::Path;
use std::sync::{Arc, OnceLock};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::runtime::Runtime;

/// Global shared runtime to avoid creating one per call
static RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn get_runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        Runtime::new().expect("Failed to create Tokio runtime")
    })
}

/// Download multiple files in parallel
///
/// # Arguments
/// * `urls` - List of (url, destination_path) tuples
/// * `max_concurrent` - Maximum concurrent downloads (default 50)
/// * `timeout_secs` - Timeout per request in seconds (default 60)
///
/// # Returns
/// * List of (path, success, error_message) tuples
#[pyfunction]
#[pyo3(signature = (urls, max_concurrent=50, timeout_secs=60))]
pub fn download_files(
    urls: Vec<(String, String)>,
    max_concurrent: usize,
    timeout_secs: u64,
) -> PyResult<Vec<(String, bool, String)>> {
    let rt = get_runtime();

    rt.block_on(async {
        download_files_async(urls, max_concurrent, timeout_secs).await
    })
}

async fn download_files_async(
    urls: Vec<(String, String)>,
    max_concurrent: usize,
    timeout_secs: u64,
) -> PyResult<Vec<(String, bool, String)>> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let client = Arc::new(client);

    let results: Vec<(String, bool, String)> = stream::iter(urls)
        .map(|(url, dest_path)| {
            let client = Arc::clone(&client);
            async move {
                match download_one(&client, &url, &dest_path).await {
                    Ok(()) => (dest_path, true, String::new()),
                    Err(e) => (dest_path, false, e),
                }
            }
        })
        .buffer_unordered(max_concurrent)
        .collect()
        .await;

    Ok(results)
}

async fn download_one(
    client: &reqwest::Client,
    url: &str,
    dest_path: &str,
) -> Result<(), String> {
    // Create parent directories
    let path = Path::new(dest_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("Failed to create directory: {}", e))?;
    }

    // Download
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("HTTP {}: {}", response.status(), url));
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| format!("Failed to read response: {}", e))?;

    // Write to file
    let mut file = fs::File::create(dest_path)
        .await
        .map_err(|e| format!("Failed to create file: {}", e))?;

    file.write_all(&bytes)
        .await
        .map_err(|e| format!("Failed to write file: {}", e))?;

    Ok(())
}

/// Download a single file
///
/// # Arguments
/// * `url` - URL to download
/// * `dest_path` - Destination file path
/// * `timeout_secs` - Timeout in seconds (default 60)
///
/// # Returns
/// * True if successful
#[pyfunction]
#[pyo3(signature = (url, dest_path, timeout_secs=60))]
pub fn download_file(
    url: &str,
    dest_path: &str,
    timeout_secs: u64,
) -> PyResult<bool> {
    let rt = get_runtime();

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    rt.block_on(async {
        match download_one(&client, url, dest_path).await {
            Ok(()) => Ok(true),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(e)),
        }
    })
}

#[cfg(test)]
mod tests {
    // Integration tests would require network access
}
