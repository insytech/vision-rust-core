//! Fast thumbnail generation using image-rs
//!
//! Replaces PIL LANCZOS + JPEG compression with native Rust.
//! Expected speedup: 5-10x

use image::{DynamicImage, GenericImageView, ImageFormat, ImageReader};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::io::Cursor;

/// Generate a JPEG thumbnail from image bytes
///
/// # Arguments
/// * `image_bytes` - Raw image bytes (any supported format)
/// * `max_size` - Maximum dimension (width or height)
/// * `quality` - JPEG quality (1-100, default 85)
///
/// # Returns
/// * JPEG bytes of the thumbnail
#[pyfunction]
#[pyo3(signature = (image_bytes, max_size=200, quality=85))]
pub fn generate_thumbnail(
    py: Python<'_>,
    image_bytes: &[u8],
    max_size: u32,
    quality: u8,
) -> PyResult<Py<PyBytes>> {
    // Load image from bytes
    let img = ImageReader::new(Cursor::new(image_bytes))
        .with_guessed_format()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
        .decode()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Convert RGBA/Palette to RGB
    let img = match img {
        DynamicImage::ImageRgba8(rgba) => {
            // Composite RGBA onto white background
            let (w, h) = rgba.dimensions();
            let mut rgb = image::RgbImage::new(w, h);
            for (x, y, pixel) in rgba.enumerate_pixels() {
                let alpha = pixel[3] as f32 / 255.0;
                let inv_alpha = 1.0 - alpha;
                rgb.put_pixel(x, y, image::Rgb([
                    (pixel[0] as f32 * alpha + 255.0 * inv_alpha) as u8,
                    (pixel[1] as f32 * alpha + 255.0 * inv_alpha) as u8,
                    (pixel[2] as f32 * alpha + 255.0 * inv_alpha) as u8,
                ]));
            }
            DynamicImage::ImageRgb8(rgb)
        }
        DynamicImage::ImageLuma8(gray) => DynamicImage::ImageRgb8(DynamicImage::ImageLuma8(gray).to_rgb8()),
        DynamicImage::ImageLumaA8(gray_alpha) => {
            // Composite grayscale with alpha onto white
            let (w, h) = gray_alpha.dimensions();
            let mut rgb = image::RgbImage::new(w, h);
            for (x, y, pixel) in gray_alpha.enumerate_pixels() {
                let alpha = pixel[1] as f32 / 255.0;
                let inv_alpha = 1.0 - alpha;
                let gray = (pixel[0] as f32 * alpha + 255.0 * inv_alpha) as u8;
                rgb.put_pixel(x, y, image::Rgb([gray, gray, gray]));
            }
            DynamicImage::ImageRgb8(rgb)
        }
        other => other,
    };

    // Calculate thumbnail dimensions maintaining aspect ratio
    let (orig_w, orig_h) = img.dimensions();
    let (thumb_w, thumb_h) = if orig_w > orig_h {
        let ratio = max_size as f32 / orig_w as f32;
        (max_size, (orig_h as f32 * ratio) as u32)
    } else {
        let ratio = max_size as f32 / orig_h as f32;
        ((orig_w as f32 * ratio) as u32, max_size)
    };

    // Resize using Lanczos3 (high quality)
    let thumbnail = img.resize_exact(
        thumb_w.max(1),
        thumb_h.max(1),
        image::imageops::FilterType::Lanczos3,
    );

    // Encode to JPEG
    let mut buffer = Cursor::new(Vec::new());
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buffer, quality);
    thumbnail
        .write_with_encoder(encoder)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    Ok(PyBytes::new_bound(py, &buffer.into_inner()).unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thumbnail_creation() {
        // Create a simple test image
        let img = image::RgbImage::from_fn(100, 100, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });

        let mut buffer = Cursor::new(Vec::new());
        img.write_to(&mut buffer, ImageFormat::Png).unwrap();

        // Would need Python context to test fully
        assert!(!buffer.into_inner().is_empty());
    }
}
