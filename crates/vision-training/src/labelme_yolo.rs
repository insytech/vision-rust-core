//! LabelMe to YOLO format conversion
//!
//! Replaces subprocess call to labelme2yolo with native Rust.
//! Expected speedup: 10x

use pyo3::prelude::*;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// LabelMe JSON format
#[derive(Debug, Deserialize)]
struct LabelMeAnnotation {
    shapes: Vec<LabelMeShape>,
    #[serde(rename = "imageWidth")]
    image_width: u32,
    #[serde(rename = "imageHeight")]
    image_height: u32,
    #[serde(rename = "imagePath")]
    image_path: String,
}

#[derive(Debug, Deserialize)]
struct LabelMeShape {
    label: String,
    points: Vec<Vec<f64>>,
    shape_type: String,
}

/// Convert a single LabelMe JSON to YOLO format
///
/// # Arguments
/// * `json_path` - Path to LabelMe JSON file
/// * `output_path` - Path for output YOLO txt file
/// * `class_map` - Dict mapping class names to indices
///
/// # Returns
/// * True if successful
#[pyfunction]
pub fn convert_labelme_to_yolo(
    json_path: &str,
    output_path: &str,
    class_map: HashMap<String, usize>,
) -> PyResult<bool> {
    let content = fs::read_to_string(json_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let annotation: LabelMeAnnotation = serde_json::from_str(&content)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let img_w = annotation.image_width as f64;
    let img_h = annotation.image_height as f64;

    let mut yolo_lines = Vec::new();

    for shape in &annotation.shapes {
        // Get class index
        let class_idx = match class_map.get(&shape.label) {
            Some(&idx) => idx,
            None => continue, // Skip unknown classes
        };

        match shape.shape_type.as_str() {
            "rectangle" => {
                if shape.points.len() >= 2 {
                    let (x1, y1) = (shape.points[0][0], shape.points[0][1]);
                    let (x2, y2) = (shape.points[1][0], shape.points[1][1]);

                    // Convert to YOLO format (center_x, center_y, width, height) normalized
                    let cx = ((x1 + x2) / 2.0) / img_w;
                    let cy = ((y1 + y2) / 2.0) / img_h;
                    let w = (x2 - x1).abs() / img_w;
                    let h = (y2 - y1).abs() / img_h;

                    yolo_lines.push(format!("{} {:.6} {:.6} {:.6} {:.6}", class_idx, cx, cy, w, h));
                }
            }
            "polygon" => {
                if shape.points.len() >= 3 {
                    // For polygon, compute bounding box
                    let xs: Vec<f64> = shape.points.iter().map(|p| p[0]).collect();
                    let ys: Vec<f64> = shape.points.iter().map(|p| p[1]).collect();

                    let x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
                    let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min);
                    let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                    let cx = ((x_min + x_max) / 2.0) / img_w;
                    let cy = ((y_min + y_max) / 2.0) / img_h;
                    let w = (x_max - x_min) / img_w;
                    let h = (y_max - y_min) / img_h;

                    yolo_lines.push(format!("{} {:.6} {:.6} {:.6} {:.6}", class_idx, cx, cy, w, h));
                }
            }
            _ => {} // Skip other shape types
        }
    }

    // Write output
    let output = yolo_lines.join("\n");
    fs::write(output_path, output)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    Ok(true)
}

/// Convert entire directory of LabelMe JSONs to YOLO format
///
/// # Arguments
/// * `input_dir` - Directory containing LabelMe JSON files
/// * `output_dir` - Output directory for YOLO txt files
/// * `class_list` - List of class names (index = class id)
///
/// # Returns
/// * Number of files converted
#[pyfunction]
pub fn convert_labelme_dir_to_yolo(
    input_dir: &str,
    output_dir: &str,
    class_list: Vec<String>,
) -> PyResult<usize> {
    let input_path = Path::new(input_dir);
    let output_path = Path::new(output_dir);

    // Create output directory
    fs::create_dir_all(output_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    // Build class map
    let class_map: HashMap<String, usize> = class_list
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .collect();

    // Find all JSON files
    let json_files: Vec<_> = walkdir::WalkDir::new(input_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().is_file()
                && e.path().extension().map(|ext| ext == "json").unwrap_or(false)
        })
        .collect();

    // Convert in parallel
    let results: Vec<bool> = json_files
        .par_iter()
        .map(|entry| {
            let json_path = entry.path();
            let stem = json_path.file_stem().unwrap_or_default();
            let txt_path = output_path.join(format!("{}.txt", stem.to_string_lossy()));

            convert_single_file(json_path, &txt_path, &class_map).unwrap_or(false)
        })
        .collect();

    Ok(results.iter().filter(|&&b| b).count())
}

fn convert_single_file(
    json_path: &Path,
    txt_path: &Path,
    class_map: &HashMap<String, usize>,
) -> Result<bool, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(json_path)?;
    let annotation: LabelMeAnnotation = serde_json::from_str(&content)?;

    let img_w = annotation.image_width as f64;
    let img_h = annotation.image_height as f64;

    let mut yolo_lines = Vec::new();

    for shape in &annotation.shapes {
        let class_idx = match class_map.get(&shape.label) {
            Some(&idx) => idx,
            None => continue,
        };

        if shape.shape_type == "rectangle" && shape.points.len() >= 2 {
            let (x1, y1) = (shape.points[0][0], shape.points[0][1]);
            let (x2, y2) = (shape.points[1][0], shape.points[1][1]);

            let cx = ((x1 + x2) / 2.0) / img_w;
            let cy = ((y1 + y2) / 2.0) / img_h;
            let w = (x2 - x1).abs() / img_w;
            let h = (y2 - y1).abs() / img_h;

            yolo_lines.push(format!("{} {:.6} {:.6} {:.6} {:.6}", class_idx, cx, cy, w, h));
        } else if shape.shape_type == "polygon" && shape.points.len() >= 3 {
            let xs: Vec<f64> = shape.points.iter().map(|p| p[0]).collect();
            let ys: Vec<f64> = shape.points.iter().map(|p| p[1]).collect();

            let x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
            let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min);
            let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let cx = ((x_min + x_max) / 2.0) / img_w;
            let cy = ((y_min + y_max) / 2.0) / img_h;
            let w = (x_max - x_min) / img_w;
            let h = (y_max - y_min) / img_h;

            yolo_lines.push(format!("{} {:.6} {:.6} {:.6} {:.6}", class_idx, cx, cy, w, h));
        }
    }

    fs::write(txt_path, yolo_lines.join("\n"))?;
    Ok(true)
}

/// Fix image paths in LabelMe JSON files
///
/// Updates the imagePath field to match the actual filename.
///
/// # Arguments
/// * `json_dir` - Directory containing JSON files
/// * `image_dir` - Directory containing images (to verify paths)
///
/// # Returns
/// * Number of files fixed
#[pyfunction]
pub fn fix_json_image_paths(json_dir: &str, image_dir: &str) -> PyResult<usize> {
    let json_path = Path::new(json_dir);
    let img_path = Path::new(image_dir);

    // Build set of available images
    let images: std::collections::HashSet<String> = walkdir::WalkDir::new(img_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| e.file_name().to_str().map(String::from))
        .collect();

    // Process JSON files
    let json_files: Vec<_> = walkdir::WalkDir::new(json_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().is_file()
                && e.path().extension().map(|ext| ext == "json").unwrap_or(false)
        })
        .collect();

    let fixed_count: usize = json_files
        .par_iter()
        .filter_map(|entry| {
            let path = entry.path();
            fix_single_json(path, &images).ok()
        })
        .filter(|&fixed| fixed)
        .count();

    Ok(fixed_count)
}

fn fix_single_json(
    path: &Path,
    images: &std::collections::HashSet<String>,
) -> Result<bool, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let mut json: serde_json::Value = serde_json::from_str(&content)?;

    let stem = path.file_stem().unwrap_or_default().to_string_lossy();

    // Try to find matching image
    let extensions = ["jpg", "jpeg", "png", "bmp", "webp"];
    let mut new_path = None;

    for ext in &extensions {
        let candidate = format!("{}.{}", stem, ext);
        if images.contains(&candidate) {
            new_path = Some(candidate);
            break;
        }
    }

    if let Some(np) = new_path {
        if let Some(obj) = json.as_object_mut() {
            let old_path = obj.get("imagePath").and_then(|v| v.as_str()).unwrap_or("");
            if old_path != np {
                obj.insert("imagePath".to_string(), serde_json::Value::String(np));
                fs::write(path, serde_json::to_string_pretty(&json)?)?;
                return Ok(true);
            }
        }
    }

    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yolo_conversion() {
        let labelme_json = r#"{
            "shapes": [
                {
                    "label": "defect",
                    "points": [[10, 20], [100, 200]],
                    "shape_type": "rectangle"
                }
            ],
            "imageWidth": 640,
            "imageHeight": 480,
            "imagePath": "test.jpg"
        }"#;

        let annotation: LabelMeAnnotation = serde_json::from_str(labelme_json).unwrap();
        assert_eq!(annotation.shapes.len(), 1);
        assert_eq!(annotation.image_width, 640);
    }
}
