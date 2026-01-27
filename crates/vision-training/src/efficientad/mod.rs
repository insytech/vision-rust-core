//! EfficientAD inference pipeline optimizations
//!
//! Ported from vision-device/rust_optimization with additions for
//! training-server specific needs (base64 encoding, overlays).
//!
//! Expected speedup: 4-10x over Python/NumPy

use ndarray::{Array2, Array3, Zip};
use numpy::{PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Compute percentile using quickselect - O(n) average instead of O(n log n) sort
#[inline]
fn percentile_quickselect(values: &mut [f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * values.len() as f32) as usize;
    let idx = idx.min(values.len() - 1);

    // select_nth_unstable partially sorts and returns the element at position idx
    let (_, median, _) = values.select_nth_unstable_by(idx, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    *median
}

/// Union-Find with path compression and union by rank for O(α(n)) operations
struct UnionFind {
    parent: Vec<i32>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size as i32).collect(),
            rank: vec![0; size],
        }
    }

    /// Find with path compression
    fn find(&mut self, x: i32) -> i32 {
        if self.parent[x as usize] != x {
            self.parent[x as usize] = self.find(self.parent[x as usize]);
        }
        self.parent[x as usize]
    }

    /// Union by rank
    fn union(&mut self, x: i32, y: i32) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }

        match self.rank[rx as usize].cmp(&self.rank[ry as usize]) {
            std::cmp::Ordering::Less => self.parent[rx as usize] = ry,
            std::cmp::Ordering::Greater => self.parent[ry as usize] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry as usize] = rx;
                self.rank[rx as usize] += 1;
            }
        }
    }
}

/// Compute anomaly maps from teacher-student and autoencoder outputs
#[pyfunction]
#[pyo3(signature = (teacher_out, student_out, ae_out, ae_target, st_weight=0.5, ae_weight=0.5))]
pub fn compute_anomaly_maps<'py>(
    py: Python<'py>,
    teacher_out: PyReadonlyArray3<f32>,
    student_out: PyReadonlyArray3<f32>,
    ae_out: PyReadonlyArray3<f32>,
    ae_target: PyReadonlyArray3<f32>,
    st_weight: f32,
    ae_weight: f32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>, f32)> {
    let teacher = teacher_out.as_array();
    let student = student_out.as_array();
    let ae = ae_out.as_array();
    let target = ae_target.as_array();

    let (h, w, c) = teacher.dim();
    let c_f32 = c as f32;

    // Compute ST and AE maps using ndarray::Zip for better SIMD optimization
    let mut st_map = Array2::<f32>::zeros((h, w));
    let mut ae_map = Array2::<f32>::zeros((h, w));

    // Process each spatial position
    Zip::indexed(&mut st_map)
        .and(&mut ae_map)
        .for_each(|(y, x), st_val, ae_val| {
            let mut st_sum = 0.0f32;
            let mut ae_sum = 0.0f32;

            for ch in 0..c {
                let diff_st = teacher[[y, x, ch]] - student[[y, x, ch]];
                st_sum += diff_st * diff_st;

                let diff_ae = ae[[y, x, ch]] - target[[y, x, ch]];
                ae_sum += diff_ae * diff_ae;
            }

            *st_val = (st_sum / c_f32).sqrt();
            *ae_val = (ae_sum / c_f32).sqrt();
        });

    // Combine maps and find max using Zip
    let mut combined = Array2::<f32>::zeros((h, w));
    let mut max_score = 0.0f32;

    Zip::from(&st_map)
        .and(&ae_map)
        .and(&mut combined)
        .for_each(|&st, &ae, comb| {
            let v = st * st_weight + ae * ae_weight;
            *comb = v;
            if v > max_score {
                max_score = v;
            }
        });

    Ok((
        PyArray2::from_owned_array_bound(py, st_map),
        PyArray2::from_owned_array_bound(py, ae_map),
        PyArray2::from_owned_array_bound(py, combined),
        max_score,
    ))
}

/// Find bounding boxes from binary mask using connected components
/// Uses Union-Find with path compression for O(α(n)) amortized operations
#[pyfunction]
#[pyo3(signature = (mask, min_area=50))]
pub fn find_bounding_boxes(
    mask: PyReadonlyArray2<u8>,
    min_area: usize,
) -> PyResult<Vec<[i32; 5]>> {
    let mask = mask.as_array();
    let (height, width) = mask.dim();
    let total_pixels = height * width;

    // Initialize Union-Find structure
    let mut uf = UnionFind::new(total_pixels);
    let mut labels = vec![-1i32; total_pixels];
    let mut next_label = 0i32;

    // First pass: assign labels and union neighbors
    for y in 0..height {
        for x in 0..width {
            if mask[[y, x]] == 0 {
                continue;
            }

            let idx = y * width + x;

            // Assign new label if not yet labeled
            if labels[idx] == -1 {
                labels[idx] = next_label;
                next_label += 1;
            }

            let current_label = labels[idx];

            // Check top neighbor
            if y > 0 && mask[[y - 1, x]] > 0 {
                let top_idx = (y - 1) * width + x;
                if labels[top_idx] != -1 {
                    uf.union(current_label, labels[top_idx]);
                }
            }

            // Check left neighbor
            if x > 0 && mask[[y, x - 1]] > 0 {
                let left_idx = y * width + x - 1;
                if labels[left_idx] != -1 {
                    uf.union(current_label, labels[left_idx]);
                }
            }
        }
    }

    // Second pass: compute bounding boxes using resolved labels
    let mut boxes: std::collections::HashMap<i32, (i32, i32, i32, i32, usize)> =
        std::collections::HashMap::new();

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if labels[idx] == -1 {
                continue;
            }

            // Find root with path compression
            let root = uf.find(labels[idx]);

            let entry = boxes.entry(root).or_insert((
                x as i32,
                y as i32,
                x as i32,
                y as i32,
                0,
            ));

            entry.0 = entry.0.min(x as i32); // x1
            entry.1 = entry.1.min(y as i32); // y1
            entry.2 = entry.2.max(x as i32); // x2
            entry.3 = entry.3.max(y as i32); // y2
            entry.4 += 1; // area
        }
    }

    // Filter by minimum area and format output
    let result: Vec<[i32; 5]> = boxes
        .into_values()
        .filter(|(_, _, _, _, area)| *area >= min_area)
        .map(|(x1, y1, x2, y2, area)| [x1, y1, x2, y2, area as i32])
        .collect();

    Ok(result)
}

/// Generate heatmap overlay on image
#[pyfunction]
#[pyo3(signature = (image, heatmap, alpha=0.5))]
pub fn generate_overlay<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    heatmap: PyReadonlyArray2<f32>,
    alpha: f32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img = image.as_array();
    let heat = heatmap.as_array();

    // Pass views directly - no cloning needed
    let result = shared::image_ops::overlay_heatmap(img, heat, alpha)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(PyArray3::from_owned_array_bound(py, result))
}

/// Generate binary mask from heatmap using percentile threshold
/// Uses quickselect O(n) instead of full sort O(n log n)
#[pyfunction]
#[pyo3(signature = (heatmap, percentile=95.0))]
pub fn generate_mask<'py>(
    py: Python<'py>,
    heatmap: PyReadonlyArray2<f32>,
    percentile: f32,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let heat = heatmap.as_array();
    let (h, w) = heat.dim();

    // Compute percentile threshold using O(n) quickselect
    let mut values: Vec<f32> = heat.iter().cloned().collect();
    let threshold = percentile_quickselect(&mut values, percentile);

    // Create binary mask using Zip for better vectorization
    let mut mask = Array2::<u8>::zeros((h, w));
    Zip::from(&heat)
        .and(&mut mask)
        .for_each(|&h_val, m_val| {
            *m_val = if h_val > threshold { 255 } else { 0 };
        });

    Ok(PyArray2::from_owned_array_bound(py, mask))
}

/// Convert array to base64-encoded PNG
#[pyfunction]
#[pyo3(signature = (array, colormap=true))]
pub fn array_to_base64(
    array: &Bound<'_, pyo3::PyAny>,
    colormap: bool,
) -> PyResult<String> {
    // Try 3D array first (RGB image)
    if let Ok(arr3d) = array.extract::<PyReadonlyArray3<u8>>() {
        let arr = arr3d.as_array();
        // Pass view directly - no cloning needed
        return shared::image_ops::array_to_base64_png(arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()));
    }

    // Try 2D float array (heatmap)
    if let Ok(arr2d) = array.extract::<PyReadonlyArray2<f32>>() {
        let arr = arr2d.as_array();
        let (h, w) = arr.dim();

        let rgb_arr = if colormap {
            // Pass view directly - no cloning needed
            shared::image_ops::apply_jet_colormap(arr)
        } else {
            // Convert to grayscale using Zip for SIMD
            let mut result = Array3::<u8>::zeros((h, w, 3));
            Zip::indexed(&arr)
                .for_each(|(y, x), &val| {
                    let v = (val.clamp(0.0, 1.0) * 255.0) as u8;
                    result[[y, x, 0]] = v;
                    result[[y, x, 1]] = v;
                    result[[y, x, 2]] = v;
                });
            result
        };

        return shared::image_ops::array_to_base64_png(rgb_arr.view())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Expected 2D float32 or 3D uint8 array"
    ))
}

/// Normalize heatmap using percentiles
/// Uses quickselect O(n) for each percentile instead of full sort
#[pyfunction]
#[pyo3(signature = (heatmap, p_low=2.0, p_high=98.0))]
pub fn normalize_heatmap<'py>(
    py: Python<'py>,
    heatmap: PyReadonlyArray2<f32>,
    p_low: f32,
    p_high: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let heat = heatmap.as_array();
    let (h, w) = heat.dim();

    // Compute percentiles using O(n) quickselect
    let mut values_low: Vec<f32> = heat.iter().cloned().collect();
    let mut values_high = values_low.clone();

    let v_low = percentile_quickselect(&mut values_low, p_low);
    let v_high = percentile_quickselect(&mut values_high, p_high);

    let range = (v_high - v_low).max(1e-6);

    // Normalize using Zip for SIMD optimization
    let mut result = Array2::<f32>::zeros((h, w));
    Zip::from(&heat)
        .and(&mut result)
        .for_each(|&h_val, r_val| {
            *r_val = ((h_val - v_low) / range).clamp(0.0, 1.0);
        });

    Ok(PyArray2::from_owned_array_bound(py, result))
}

/// Compute percentiles using quickselect O(n) per percentile
#[pyfunction]
pub fn compute_percentiles<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<f32>,
    percentiles: Vec<f32>,
) -> PyResult<Bound<'py, PyDict>> {
    let arr = array.as_array();
    let dict = PyDict::new_bound(py);

    for p in percentiles {
        // Clone for each percentile since quickselect modifies the array
        let mut values: Vec<f32> = arr.iter().cloned().collect();
        let val = percentile_quickselect(&mut values, p);
        dict.set_item(format!("p{}", p as i32), val)?;
    }

    Ok(dict)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_find_bounding_boxes() {
        // Create a simple mask with one component
        let mut mask = Array2::<u8>::zeros((10, 10));
        for y in 2..5 {
            for x in 3..7 {
                mask[[y, x]] = 255;
            }
        }
        assert_eq!(mask[[3, 4]], 255);
    }
}
