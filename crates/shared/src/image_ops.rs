//! Common image operations shared across crates
//!
//! Ported from vision-device/rust_optimization for reuse in servers.

use base64::{engine::general_purpose::STANDARD, Engine};
use image::{DynamicImage, ImageFormat, ImageBuffer, Rgb, Rgba};
use ndarray::{Array3, ArrayView2, ArrayView3, Zip};
use std::io::Cursor;

use crate::{VisionError, VisionResult};

/// Apply jet colormap to a normalized heatmap (0.0-1.0)
/// Returns RGB image as (H, W, 3) array
/// Accepts ArrayView to avoid unnecessary cloning
#[inline]
pub fn apply_jet_colormap(heatmap: ArrayView2<f32>) -> Array3<u8> {
    let (height, width) = heatmap.dim();
    let mut result = Array3::<u8>::zeros((height, width, 3));

    Zip::indexed(heatmap)
        .for_each(|(y, x), &v| {
            let v_clamped = v.clamp(0.0, 1.0);
            let (r, g, b) = jet_color(v_clamped);
            result[[y, x, 0]] = r;
            result[[y, x, 1]] = g;
            result[[y, x, 2]] = b;
        });

    result
}

/// Jet colormap implementation
#[inline(always)]
fn jet_color(v: f32) -> (u8, u8, u8) {
    let r = (1.5 - (4.0 * v - 3.0).abs()).clamp(0.0, 1.0);
    let g = (1.5 - (4.0 * v - 2.0).abs()).clamp(0.0, 1.0);
    let b = (1.5 - (4.0 * v - 1.0).abs()).clamp(0.0, 1.0);

    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Overlay heatmap on image with alpha blending
/// Accepts ArrayView to avoid unnecessary cloning
#[inline]
pub fn overlay_heatmap(
    image: ArrayView3<u8>,
    heatmap: ArrayView2<f32>,
    alpha: f32,
) -> VisionResult<Array3<u8>> {
    let (h, w, c) = image.dim();
    let (hh, hw) = heatmap.dim();

    if h != hh || w != hw {
        return Err(VisionError::ShapeMismatch {
            expected: format!("({}, {})", h, w),
            actual: format!("({}, {})", hh, hw),
        });
    }

    if c != 3 {
        return Err(VisionError::InvalidInput("Image must have 3 channels".into()));
    }

    let colored = apply_jet_colormap(heatmap);
    let mut result = Array3::<u8>::zeros((h, w, 3));
    let inv_alpha = 1.0 - alpha;

    // Use Zip for better vectorization
    Zip::indexed(&mut result)
        .for_each(|(y, x, ch), r_val| {
            let bg = image[[y, x, ch]] as f32;
            let fg = colored[[y, x, ch]] as f32;
            *r_val = (bg * inv_alpha + fg * alpha) as u8;
        });

    Ok(result)
}

/// Encode array as PNG and return base64 string
/// Accepts ArrayView to avoid unnecessary cloning
#[inline]
pub fn array_to_base64_png(array: ArrayView3<u8>) -> VisionResult<String> {
    let (height, width, channels) = array.dim();

    let img: DynamicImage = match channels {
        3 => {
            let buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(
                width as u32,
                height as u32,
                |x, y| {
                    Rgb([
                        array[[y as usize, x as usize, 0]],
                        array[[y as usize, x as usize, 1]],
                        array[[y as usize, x as usize, 2]],
                    ])
                },
            );
            DynamicImage::ImageRgb8(buffer)
        }
        4 => {
            let buffer: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_fn(
                width as u32,
                height as u32,
                |x, y| {
                    Rgba([
                        array[[y as usize, x as usize, 0]],
                        array[[y as usize, x as usize, 1]],
                        array[[y as usize, x as usize, 2]],
                        array[[y as usize, x as usize, 3]],
                    ])
                },
            );
            DynamicImage::ImageRgba8(buffer)
        }
        _ => return Err(VisionError::InvalidInput(
            format!("Unsupported channel count: {}", channels)
        )),
    };

    let mut buffer = Cursor::new(Vec::new());
    img.write_to(&mut buffer, ImageFormat::Png)?;

    Ok(STANDARD.encode(buffer.into_inner()))
}

/// Resize image using bilinear interpolation
/// Accepts ArrayView to avoid unnecessary cloning
#[inline]
pub fn resize_bilinear(
    image: ArrayView3<u8>,
    new_height: usize,
    new_width: usize,
) -> Array3<u8> {
    let (h, w, c) = image.dim();
    let mut result = Array3::<u8>::zeros((new_height, new_width, c));

    let scale_y = h as f32 / new_height as f32;
    let scale_x = w as f32 / new_width as f32;
    let h_max = (h as i32 - 1) as usize;
    let w_max = (w as i32 - 1) as usize;

    for ny in 0..new_height {
        let sy = (ny as f32 + 0.5) * scale_y - 0.5;
        let y0_i = sy.floor() as i32;
        let y1_i = y0_i + 1;
        let fy = sy - y0_i as f32;

        let y0 = y0_i.clamp(0, h_max as i32) as usize;
        let y1 = y1_i.clamp(0, h_max as i32) as usize;

        for nx in 0..new_width {
            let sx = (nx as f32 + 0.5) * scale_x - 0.5;
            let x0_i = sx.floor() as i32;
            let x1_i = x0_i + 1;
            let fx = sx - x0_i as f32;

            let x0 = x0_i.clamp(0, w_max as i32) as usize;
            let x1 = x1_i.clamp(0, w_max as i32) as usize;

            for ch in 0..c {
                let v00 = image[[y0, x0, ch]] as f32;
                let v01 = image[[y0, x1, ch]] as f32;
                let v10 = image[[y1, x0, ch]] as f32;
                let v11 = image[[y1, x1, ch]] as f32;

                let v = v00 * (1.0 - fx) * (1.0 - fy)
                    + v01 * fx * (1.0 - fy)
                    + v10 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                result[[ny, nx, ch]] = v.round() as u8;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_jet_colormap() {
        // Test edge values
        assert_eq!(jet_color(0.0), (0, 0, 127));
        assert_eq!(jet_color(1.0), (127, 0, 0));
    }

    #[test]
    fn test_overlay_shape_mismatch() {
        let image = Array3::<u8>::zeros((100, 100, 3));
        let heatmap = Array2::<f32>::zeros((50, 50));

        let result = overlay_heatmap(image.view(), heatmap.view(), 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_jet_colormap() {
        let mut heatmap = Array2::<f32>::zeros((10, 10));
        heatmap[[5, 5]] = 0.5;
        let result = apply_jet_colormap(heatmap.view());
        assert_eq!(result.dim(), (10, 10, 3));
    }

    #[test]
    fn test_resize_bilinear() {
        let image = Array3::<u8>::from_elem((100, 100, 3), 128);
        let resized = resize_bilinear(image.view(), 50, 50);
        assert_eq!(resized.dim(), (50, 50, 3));
    }
}
