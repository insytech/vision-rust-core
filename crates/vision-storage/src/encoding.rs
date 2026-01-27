//! Fast encoding detection and validation
//!
//! Replaces chardet with native Rust UTF-8 validation.
//! Expected speedup: 50-100x

use pyo3::prelude::*;
use std::path::Path;

/// Verify file encoding and return decoded content
///
/// Much faster than chardet - uses native Rust UTF-8 validation
/// with fallback to Latin-1 for legacy files.
///
/// # Arguments
/// * `file_path` - Path to the file to verify
///
/// # Returns
/// * Tuple of (content: str, encoding: str)
#[pyfunction]
pub fn verify_encoding(file_path: &str) -> PyResult<(String, String)> {
    let path = Path::new(file_path);
    let bytes = std::fs::read(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    verify_encoding_impl(&bytes)
}

/// Verify encoding from bytes directly
///
/// # Arguments
/// * `data` - Raw bytes to verify
///
/// # Returns
/// * Tuple of (content: str, encoding: str)
#[pyfunction]
pub fn verify_encoding_bytes(data: &[u8]) -> PyResult<(String, String)> {
    verify_encoding_impl(data)
}

fn verify_encoding_impl(data: &[u8]) -> PyResult<(String, String)> {
    // Skip BOM if present
    let data = if data.starts_with(&[0xEF, 0xBB, 0xBF]) {
        &data[3..] // UTF-8 BOM
    } else if data.starts_with(&[0xFF, 0xFE]) {
        // UTF-16 LE BOM - convert
        return decode_utf16_le(&data[2..]);
    } else if data.starts_with(&[0xFE, 0xFF]) {
        // UTF-16 BE BOM - convert
        return decode_utf16_be(&data[2..]);
    } else {
        data
    };

    // Fast path: valid UTF-8 (most common case)
    if let Ok(content) = std::str::from_utf8(data) {
        return Ok((content.to_string(), "utf-8".to_string()));
    }

    // Try to repair UTF-8 by replacing invalid sequences
    let content = String::from_utf8_lossy(data);
    if !content.contains('\u{FFFD}') || content.matches('\u{FFFD}').count() < data.len() / 100 {
        // Less than 1% replacement chars - probably UTF-8 with minor corruption
        return Ok((content.into_owned(), "utf-8-lossy".to_string()));
    }

    // Fallback: Latin-1 (ISO-8859-1) - always succeeds
    let content: String = data.iter().map(|&b| b as char).collect();
    Ok((content, "latin-1".to_string()))
}

fn decode_utf16_le(data: &[u8]) -> PyResult<(String, String)> {
    if data.len() % 2 != 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid UTF-16 LE data length"
        ));
    }

    let utf16: Vec<u16> = data
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    let content = String::from_utf16(&utf16)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok((content, "utf-16-le".to_string()))
}

fn decode_utf16_be(data: &[u8]) -> PyResult<(String, String)> {
    if data.len() % 2 != 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid UTF-16 BE data length"
        ));
    }

    let utf16: Vec<u16> = data
        .chunks_exact(2)
        .map(|chunk| u16::from_be_bytes([chunk[0], chunk[1]]))
        .collect();

    let content = String::from_utf16(&utf16)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok((content, "utf-16-be".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utf8() {
        let data = b"Hello, World!";
        let (content, encoding) = verify_encoding_impl(data).unwrap();
        assert_eq!(content, "Hello, World!");
        assert_eq!(encoding, "utf-8");
    }

    #[test]
    fn test_utf8_with_unicode() {
        let data = "Héllo, 世界!".as_bytes();
        let (content, encoding) = verify_encoding_impl(data).unwrap();
        assert_eq!(content, "Héllo, 世界!");
        assert_eq!(encoding, "utf-8");
    }

    #[test]
    fn test_json_content() {
        let data = br#"{"key": "value", "number": 123}"#;
        let (content, encoding) = verify_encoding_impl(data).unwrap();
        assert!(content.contains("key"));
        assert_eq!(encoding, "utf-8");
    }

    #[test]
    fn test_utf8_bom() {
        let data = b"\xef\xbb\xbfHello";
        let (content, encoding) = verify_encoding_impl(data).unwrap();
        assert_eq!(content, "Hello");
        assert_eq!(encoding, "utf-8");
    }
}
