//! Fast COCO to LabelMe format parser
//!
//! Replaces Python JSON parsing with serde_json zero-copy parsing.
//! Expected speedup: 10-20x

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::Deserialize;
use std::collections::HashMap;

/// COCO format structures
#[derive(Debug, Deserialize)]
struct CocoDataset {
    images: Vec<CocoImage>,
    annotations: Vec<CocoAnnotation>,
    categories: Vec<CocoCategory>,
}

#[derive(Debug, Deserialize)]
struct CocoImage {
    id: i64,
    file_name: String,
    width: u32,
    height: u32,
}

#[derive(Debug, Deserialize)]
struct CocoAnnotation {
    id: i64,
    image_id: i64,
    category_id: i64,
    bbox: Vec<f64>, // [x, y, width, height]
    #[serde(default)]
    segmentation: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct CocoCategory {
    id: i64,
    name: String,
}

/// Parse COCO JSON to LabelMe format
///
/// # Arguments
/// * `coco_json` - COCO format JSON string
///
/// # Returns
/// * List of dicts, each representing a LabelMe annotation file
#[pyfunction]
pub fn parse_coco_to_labelme(py: Python<'_>, coco_json: &str) -> PyResult<Py<PyList>> {
    // Parse COCO JSON using serde (much faster than Python json)
    let coco: CocoDataset = serde_json::from_str(coco_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Build lookup maps
    let image_map: HashMap<i64, &CocoImage> = coco.images.iter().map(|img| (img.id, img)).collect();
    let category_map: HashMap<i64, &str> = coco
        .categories
        .iter()
        .map(|cat| (cat.id, cat.name.as_str()))
        .collect();

    // Group annotations by image
    let mut annotations_by_image: HashMap<i64, Vec<&CocoAnnotation>> = HashMap::new();
    for ann in &coco.annotations {
        annotations_by_image
            .entry(ann.image_id)
            .or_default()
            .push(ann);
    }

    // Convert to LabelMe format
    let result = PyList::empty_bound(py);

    for image in &coco.images {
        let labelme_dict = PyDict::new_bound(py);

        // Basic image info
        labelme_dict.set_item("version", "5.0.1")?;
        labelme_dict.set_item("flags", PyDict::new_bound(py))?;
        labelme_dict.set_item("imagePath", &image.file_name)?;
        labelme_dict.set_item("imageData", py.None())?;
        labelme_dict.set_item("imageHeight", image.height)?;
        labelme_dict.set_item("imageWidth", image.width)?;

        // Convert shapes
        let shapes = PyList::empty_bound(py);

        if let Some(anns) = annotations_by_image.get(&image.id) {
            for ann in anns {
                let shape = PyDict::new_bound(py);

                // Get category name
                let label = category_map
                    .get(&ann.category_id)
                    .unwrap_or(&"unknown");
                shape.set_item("label", *label)?;

                // Convert bbox [x, y, w, h] to points [[x1,y1], [x2,y2]]
                if ann.bbox.len() >= 4 {
                    let x = ann.bbox[0];
                    let y = ann.bbox[1];
                    let w = ann.bbox[2];
                    let h = ann.bbox[3];

                    let points = PyList::new_bound(py, &[
                        vec![x, y],
                        vec![x + w, y + h],
                    ]);
                    shape.set_item("points", points)?;
                    shape.set_item("shape_type", "rectangle")?;
                } else {
                    continue;
                }

                shape.set_item("flags", PyDict::new_bound(py))?;
                shape.set_item("group_id", py.None())?;
                shape.set_item("description", "")?;

                shapes.append(shape)?;
            }
        }

        labelme_dict.set_item("shapes", shapes)?;
        result.append(labelme_dict)?;
    }

    Ok(result.unbind())
}

/// Parse COCO JSON file and return image-annotation mapping
///
/// More efficient for large datasets - returns raw data for Python to process
#[pyfunction]
pub fn parse_coco_fast(py: Python<'_>, coco_json: &str) -> PyResult<Py<PyDict>> {
    let coco: CocoDataset = serde_json::from_str(coco_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let result = PyDict::new_bound(py);

    // Images dict: {id: {file_name, width, height}}
    let images_dict = PyDict::new_bound(py);
    for img in &coco.images {
        let img_info = PyDict::new_bound(py);
        img_info.set_item("file_name", &img.file_name)?;
        img_info.set_item("width", img.width)?;
        img_info.set_item("height", img.height)?;
        images_dict.set_item(img.id, img_info)?;
    }
    result.set_item("images", images_dict)?;

    // Categories dict: {id: name}
    let categories_dict = PyDict::new_bound(py);
    for cat in &coco.categories {
        categories_dict.set_item(cat.id, &cat.name)?;
    }
    result.set_item("categories", categories_dict)?;

    // Annotations grouped by image_id
    let annotations_dict = PyDict::new_bound(py);
    for ann in &coco.annotations {
        let key = ann.image_id;

        // Get or create list for this image
        if annotations_dict.get_item(key)?.is_none() {
            annotations_dict.set_item(key, PyList::empty_bound(py))?;
        }

        if let Some(ann_list) = annotations_dict.get_item(key)? {
            let ann_list = ann_list.downcast::<PyList>()?;
            let ann_dict = PyDict::new_bound(py);
            ann_dict.set_item("id", ann.id)?;
            ann_dict.set_item("category_id", ann.category_id)?;
            ann_dict.set_item("bbox", &ann.bbox)?;
            ann_list.append(ann_dict)?;
        }
    }
    result.set_item("annotations", annotations_dict)?;

    Ok(result.unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coco_parsing() {
        let coco_json = r#"{
            "images": [{"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 100, 200]}],
            "categories": [{"id": 1, "name": "defect"}]
        }"#;

        let coco: CocoDataset = serde_json::from_str(coco_json).unwrap();
        assert_eq!(coco.images.len(), 1);
        assert_eq!(coco.annotations.len(), 1);
        assert_eq!(coco.categories.len(), 1);
    }
}
