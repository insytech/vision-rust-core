//! LabelMe to YOLO format conversion
//!
//! Replaces subprocess call to labelme2yolo with native Rust.
//! Expected speedup: 10x
//!
//! # Encodings
//!
//! The converter is told which **label encoding** to emit; it never infers one.
//! Before this module took an `encoding` parameter it emitted the 5-value
//! axis-aligned form for everything, which meant an oriented-bounding-box model
//! was trained on labels with the rotation thrown away -- plausible numbers,
//! silently wrong model. The two encodings are:
//!
//! * [`ENC_BBOX5`] -- `class cx cy w h`, normalised. Detection and defects.
//! * [`ENC_OBB9`] -- `class x1 y1 x2 y2 x3 y3 x4 y4`, normalised (the DOTA /
//!   Ultralytics-OBB form). Measurement. A polygon is reduced by
//!   [`min_area_rect`], **not** by min/max over its vertices.
//!
//! # ABI
//!
//! Both pyfunctions changed arity and return type when the encoding parameter
//! was added. A `.so` built before that change imports fine and fails at *call*
//! time, so the Python side pins the crate version (exported as the module's
//! `__version__`, see `lib.rs`) and refuses to use a module that does not match.
//!
//! Tasks: B2.1, B2.2, B2.3, B2.4.

use pyo3::prelude::*;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// `class cx cy w h`, normalised. The historical output of this module.
pub const ENC_BBOX5: &str = "bbox5";
/// `class x1 y1 x2 y2 x3 y3 x4 y4`, normalised. DOTA / Ultralytics OBB.
pub const ENC_OBB9: &str = "obb9";

/// LabelMe JSON format
#[derive(Debug, Deserialize)]
struct LabelMeAnnotation {
    shapes: Vec<LabelMeShape>,
    #[serde(rename = "imageWidth")]
    image_width: u32,
    #[serde(rename = "imageHeight")]
    image_height: u32,
    #[serde(rename = "imagePath")]
    #[allow(dead_code)]
    image_path: String,
}

#[derive(Debug, Deserialize)]
struct LabelMeShape {
    label: String,
    points: Vec<Vec<f64>>,
    shape_type: String,
    /// LabelMe's instance grouping. Already present in the client's `IShape`
    /// and dropped here until B2.1. It is how LabelMe binds keypoints to an
    /// instance, so preserving it now is what keeps a future keypoint encoding
    /// from being a converter rewrite. Nothing reads it yet -- it is carried,
    /// and `group_id_round_trips` is what proves it is still carried.
    #[serde(default)]
    #[allow(dead_code)]
    group_id: Option<i64>,
}

/// How one file's shapes came out.
///
/// `read == emitted + dropped + malformed + unknown_class`, always -- that
/// reconciliation is the point of counting at all. The three non-emitted
/// buckets are kept apart because the caller treats them differently, and
/// collapsing them is what made the original `_ => {}` arm indefensible:
///
/// * `malformed` is an **error**. The operator drew a shape this encoding does
///   handle, and it came out unusable (a polygon with two vertices). Something
///   is wrong with the data and training on it silently would hide that.
/// * `dropped` is a **report**. The shape type is one this encoding simply does
///   not represent -- a circle in a box dataset. Long-standing, and raising on
///   it would fail existing detection datasets that have always contained
///   stray annotations.
/// * `unknown_class` is **normal**. The label is not in the class map.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ShapeCounts {
    /// Shapes written as a label line.
    pub emitted: usize,
    /// Shape types the active encoding cannot represent at all.
    pub dropped: usize,
    /// Shape types the encoding *does* handle, with unusable geometry.
    pub malformed: usize,
    /// Shapes whose label is not in the class map.
    pub unknown_class: usize,
}

impl ShapeCounts {
    fn add(&mut self, other: ShapeCounts) {
        self.emitted += other.emitted;
        self.dropped += other.dropped;
        self.malformed += other.malformed;
        self.unknown_class += other.unknown_class;
    }
}

/// Why a shape produced no label line.
enum Rejected {
    /// The encoding cannot represent this shape type.
    UnsupportedType,
    /// The right shape type, with geometry that cannot be used.
    Malformed,
}

fn unsupported_encoding_message(encoding: &str) -> String {
    format!(
        "unsupported label encoding {encoding:?}; this converter emits {ENC_BBOX5:?} or {ENC_OBB9:?}"
    )
}

fn unsupported_encoding(encoding: &str) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(unsupported_encoding_message(encoding))
}

/// Andrew's monotone chain. Returns the convex hull counter-clockwise, without
/// the duplicated last point.
fn convex_hull(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut pts: Vec<(f64, f64)> = points.to_vec();
    pts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }

    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };

    let mut hull: Vec<(f64, f64)> = Vec::with_capacity(pts.len() * 2);
    for &p in pts.iter() {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower_len = hull.len() + 1;
    for &p in pts.iter().rev() {
        while hull.len() >= lower_len && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
        {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

/// The minimum-area enclosing rectangle, as four corners.
///
/// Rotating calipers over the convex hull: the minimum-area rectangle is always
/// flush with one hull edge, so trying each edge as the rectangle's axis and
/// keeping the smallest area is exact, not approximate.
///
/// This is what makes `obb9` an oriented box. Taking min/max over the raw
/// vertices instead -- which is what the 5-value path does -- returns the
/// axis-aligned bounds and discards the rotation entirely.
pub fn min_area_rect(points: &[(f64, f64)]) -> [(f64, f64); 4] {
    let hull = convex_hull(points);
    if hull.len() < 3 {
        // Degenerate (a point or a segment): fall back to the axis-aligned box,
        // which for these inputs is also the minimum-area one.
        let xs: Vec<f64> = points.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = points.iter().map(|p| p.1).collect();
        let x0 = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let x1 = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y0 = ys.iter().cloned().fold(f64::INFINITY, f64::min);
        let y1 = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
    }

    let mut best_area = f64::INFINITY;
    let mut best: [(f64, f64); 4] = [(0.0, 0.0); 4];

    for i in 0..hull.len() {
        let a = hull[i];
        let b = hull[(i + 1) % hull.len()];
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len = (dx * dx + dy * dy).sqrt();
        if len < f64::EPSILON {
            continue;
        }
        let u = (dx / len, dy / len); // along the edge
        let v = (-u.1, u.0); // perpendicular

        let (mut u0, mut u1) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut v0, mut v1) = (f64::INFINITY, f64::NEG_INFINITY);
        for &p in &hull {
            let pu = p.0 * u.0 + p.1 * u.1;
            let pv = p.0 * v.0 + p.1 * v.1;
            u0 = u0.min(pu);
            u1 = u1.max(pu);
            v0 = v0.min(pv);
            v1 = v1.max(pv);
        }

        let area = (u1 - u0) * (v1 - v0);
        if area < best_area {
            best_area = area;
            let corner = |cu: f64, cv: f64| (u.0 * cu + v.0 * cv, u.1 * cu + v.1 * cv);
            best = [
                corner(u0, v0),
                corner(u1, v0),
                corner(u1, v1),
                corner(u0, v1),
            ];
        }
    }
    best
}

/// Turn one shape into a label line, or say why it could not be.
///
/// The single place either encoding is produced. Both pyfunctions route through
/// it so the `bbox5` bytes cannot drift between the single-file and the
/// directory path -- they were two separately-maintained copies of the same
/// arithmetic before.
fn encode_shape(
    shape: &LabelMeShape,
    class_idx: usize,
    img_w: f64,
    img_h: f64,
    encoding: &str,
) -> Result<String, Rejected> {
    // Both encodings are built from the same two shape types, so the
    // supported/malformed split is decided once, here, rather than per encoding.
    let corners: Vec<(f64, f64)> = match shape.shape_type.as_str() {
        "rectangle" => {
            if shape.points.len() < 2 {
                return Err(Rejected::Malformed);
            }
            let (x1, y1) = (shape.points[0][0], shape.points[0][1]);
            let (x2, y2) = (shape.points[1][0], shape.points[1][1]);
            let (xa, xb) = (x1.min(x2), x1.max(x2));
            let (ya, yb) = (y1.min(y2), y1.max(y2));
            vec![(xa, ya), (xb, ya), (xb, yb), (xa, yb)]
        }
        "polygon" => {
            if shape.points.len() < 3 {
                return Err(Rejected::Malformed);
            }
            shape.points.iter().map(|p| (p[0], p[1])).collect()
        }
        _ => return Err(Rejected::UnsupportedType),
    };

    match encoding {
        ENC_BBOX5 => {
            let xs: Vec<f64> = corners.iter().map(|p| p.0).collect();
            let ys: Vec<f64> = corners.iter().map(|p| p.1).collect();
            let x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
            let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min);
            let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let cx = ((x_min + x_max) / 2.0) / img_w;
            let cy = ((y_min + y_max) / 2.0) / img_h;
            let w = (x_max - x_min) / img_w;
            let h = (y_max - y_min) / img_h;
            Ok(format!(
                "{} {:.6} {:.6} {:.6} {:.6}",
                class_idx, cx, cy, w, h
            ))
        }
        ENC_OBB9 => {
            let rect = min_area_rect(&corners);
            let mut line = format!("{}", class_idx);
            for (x, y) in rect {
                line.push_str(&format!(" {:.6} {:.6}", x / img_w, y / img_h));
            }
            Ok(line)
        }
        _ => Err(Rejected::UnsupportedType),
    }
}

fn encode_annotation(
    annotation: &LabelMeAnnotation,
    class_map: &HashMap<String, usize>,
    encoding: &str,
) -> (Vec<String>, ShapeCounts) {
    let img_w = annotation.image_width as f64;
    let img_h = annotation.image_height as f64;

    let mut lines = Vec::new();
    let mut counts = ShapeCounts::default();

    for shape in &annotation.shapes {
        let class_idx = match class_map.get(&shape.label) {
            Some(&idx) => idx,
            None => {
                counts.unknown_class += 1;
                continue;
            }
        };
        match encode_shape(shape, class_idx, img_w, img_h, encoding) {
            Ok(line) => {
                lines.push(line);
                counts.emitted += 1;
            }
            Err(Rejected::UnsupportedType) => counts.dropped += 1,
            Err(Rejected::Malformed) => counts.malformed += 1,
        }
    }
    (lines, counts)
}

/// Convert a single LabelMe JSON to YOLO format
///
/// # Arguments
/// * `json_path` - Path to LabelMe JSON file
/// * `output_path` - Path for output YOLO txt file
/// * `class_map` - Dict mapping class names to indices
/// * `encoding` - `"bbox5"` or `"obb9"`
///
/// # Returns
/// * `(emitted, dropped, malformed, unknown_class)` shape counts
#[pyfunction]
pub fn convert_labelme_to_yolo(
    json_path: &str,
    output_path: &str,
    class_map: HashMap<String, usize>,
    encoding: &str,
) -> PyResult<(usize, usize, usize, usize)> {
    if encoding != ENC_BBOX5 && encoding != ENC_OBB9 {
        return Err(unsupported_encoding(encoding));
    }

    let content = fs::read_to_string(json_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let annotation: LabelMeAnnotation = serde_json::from_str(&content)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let (lines, counts) = encode_annotation(&annotation, &class_map, encoding);

    fs::write(output_path, lines.join("\n"))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    Ok((
        counts.emitted,
        counts.dropped,
        counts.malformed,
        counts.unknown_class,
    ))
}

/// Convert entire directory of LabelMe JSONs to YOLO format
///
/// # Arguments
/// * `input_dir` - Directory containing LabelMe JSON files
/// * `output_dir` - Output directory for YOLO txt files
/// * `class_list` - List of class names (index = class id)
/// * `encoding` - `"bbox5"` or `"obb9"`
///
/// # Returns
/// * `(files_converted, emitted, dropped, malformed, unknown_class)`
#[pyfunction]
pub fn convert_labelme_dir_to_yolo(
    input_dir: &str,
    output_dir: &str,
    class_list: Vec<String>,
    encoding: &str,
) -> PyResult<(usize, usize, usize, usize, usize)> {
    if encoding != ENC_BBOX5 && encoding != ENC_OBB9 {
        return Err(unsupported_encoding(encoding));
    }

    let input_path = Path::new(input_dir);
    let output_path = Path::new(output_dir);

    fs::create_dir_all(output_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let class_map: HashMap<String, usize> = class_list
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .collect();

    let json_files: Vec<_> = walkdir::WalkDir::new(input_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().is_file()
                && e.path().extension().map(|ext| ext == "json").unwrap_or(false)
        })
        .collect();

    let results: Vec<(bool, ShapeCounts)> = json_files
        .par_iter()
        .map(|entry| {
            let json_path = entry.path();
            let stem = json_path.file_stem().unwrap_or_default();
            let txt_path = output_path.join(format!("{}.txt", stem.to_string_lossy()));

            convert_single_file(json_path, &txt_path, &class_map, encoding)
                .unwrap_or((false, ShapeCounts::default()))
        })
        .collect();

    let mut totals = ShapeCounts::default();
    let mut files = 0usize;
    for (ok, counts) in results {
        if ok {
            files += 1;
        }
        totals.add(counts);
    }

    Ok((
        files,
        totals.emitted,
        totals.dropped,
        totals.malformed,
        totals.unknown_class,
    ))
}

fn convert_single_file(
    json_path: &Path,
    txt_path: &Path,
    class_map: &HashMap<String, usize>,
    encoding: &str,
) -> Result<(bool, ShapeCounts), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(json_path)?;
    let annotation: LabelMeAnnotation = serde_json::from_str(&content)?;

    let (lines, counts) = encode_annotation(&annotation, class_map, encoding);

    fs::write(txt_path, lines.join("\n"))?;
    Ok((true, counts))
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

    /// The byte-equality fixture for B2.3c.
    ///
    /// One axis-aligned rectangle plus one rectangle rotated exactly 30 deg
    /// (100x40 px centred at (320,240) in a 640x480 image).
    const GOLDEN_BBOX5_LABELME: &str = r#"{
        "shapes": [
            {"label": "part", "points": [[10, 20], [100, 200]], "shape_type": "rectangle"},
            {"label": "hole",
             "points": [[353.30127018922195, 282.32050807568877],
                        [266.69872981077805, 232.32050807568877],
                        [286.69872981077805, 197.67949192431123],
                        [373.30127018922195, 247.67949192431123]],
             "shape_type": "polygon"}
        ],
        "imageWidth": 640,
        "imageHeight": 480,
        "imagePath": "golden.jpg"
    }"#;

    /// **Captured from this converter before the `obb9` encoding existed**, by
    /// running the then-current `convert_labelme_to_yolo` over
    /// `GOLDEN_BBOX5_LABELME` and printing the file it wrote. It is evidence
    /// that `bbox5` did not move, not a restatement of the current code.
    ///
    /// Note the final digit of the last field: `0.176335`. Deriving it by hand
    /// gives `0.176336`. That is exactly why it was captured and not computed.
    const GOLDEN_BBOX5_OUTPUT: &str = "0 0.085938 0.229167 0.140625 0.375000\n\
                                       1 0.500000 0.500000 0.166566 0.176335";

    fn class_map() -> HashMap<String, usize> {
        let mut cm = HashMap::new();
        cm.insert("part".to_string(), 0usize);
        cm.insert("hole".to_string(), 1usize);
        cm
    }

    fn convert(json: &str, encoding: &str) -> (String, (usize, usize, usize, usize)) {
        let dir = tempfile::tempdir().unwrap();
        let json_path = dir.path().join("g.json");
        let txt_path = dir.path().join("g.txt");
        fs::write(&json_path, json).unwrap();
        let counts = convert_labelme_to_yolo(
            json_path.to_str().unwrap(),
            txt_path.to_str().unwrap(),
            class_map(),
            encoding,
        )
        .unwrap();
        (fs::read_to_string(&txt_path).unwrap(), counts)
    }

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

    /// B2.1 -- `group_id` survives deserialization, including two instances
    /// that share a label and differ only in `group_id`.
    #[test]
    fn group_id_round_trips() {
        let json = r#"{
            "shapes": [
                {"label": "part", "points": [[0,0],[10,10]], "shape_type": "rectangle", "group_id": 1},
                {"label": "part", "points": [[20,20],[30,30]], "shape_type": "rectangle", "group_id": 2},
                {"label": "part", "points": [[40,40],[50,50]], "shape_type": "rectangle"}
            ],
            "imageWidth": 640, "imageHeight": 480, "imagePath": "a.jpg"
        }"#;
        let ann: LabelMeAnnotation = serde_json::from_str(json).unwrap();
        assert_eq!(ann.shapes[0].label, ann.shapes[1].label);
        assert_eq!(ann.shapes[0].group_id, Some(1));
        assert_eq!(ann.shapes[1].group_id, Some(2));
        assert_ne!(ann.shapes[0].group_id, ann.shapes[1].group_id);
        // absent is None, not an error
        assert_eq!(ann.shapes[2].group_id, None);
    }

    /// B2.3c -- the same fixture on a `bbox5` job is byte-identical to what
    /// the converter produced before `obb9` existed.
    #[test]
    fn bbox5_output_is_byte_identical_to_the_captured_golden() {
        let (out, counts) = convert(GOLDEN_BBOX5_LABELME, ENC_BBOX5);
        assert_eq!(out, GOLDEN_BBOX5_OUTPUT);
        assert_eq!(out.as_bytes(), GOLDEN_BBOX5_OUTPUT.as_bytes());
        assert_eq!(counts, (2, 0, 0, 0));
    }

    /// B2.3b -- exactly 9 whitespace-separated fields per line.
    #[test]
    fn obb9_emits_nine_fields() {
        let (out, counts) = convert(GOLDEN_BBOX5_LABELME, ENC_OBB9);
        assert_eq!(counts, (2, 0, 0, 0));
        for line in out.lines() {
            assert_eq!(
                line.split_whitespace().count(),
                9,
                "expected a 9-value DOTA line, got {line:?}"
            );
        }
    }

    /// B2.3a -- a polygon rotated 30 deg emits **rotated** corners, not the
    /// axis-aligned min/max bounds. The fixture is a 100x40 rectangle rotated
    /// 30 deg, so the minimum-area rectangle must recover 100x40 -- while the
    /// axis-aligned bounds of the same points are 106.60 x 84.64.
    #[test]
    fn obb9_does_not_collapse_a_rotated_polygon_to_an_axis_aligned_box() {
        let (out, _) = convert(GOLDEN_BBOX5_LABELME, ENC_OBB9);
        let hole = out.lines().nth(1).unwrap();
        let v: Vec<f64> = hole
            .split_whitespace()
            .skip(1)
            .map(|s| s.parse::<f64>().unwrap())
            .collect();
        let pts: Vec<(f64, f64)> = (0..4).map(|i| (v[2 * i] * 640.0, v[2 * i + 1] * 480.0)).collect();

        let side = |a: (f64, f64), b: (f64, f64)| ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
        let mut sides = [
            side(pts[0], pts[1]),
            side(pts[1], pts[2]),
            side(pts[2], pts[3]),
            side(pts[3], pts[0]),
        ];
        sides.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // the true rectangle, recovered: 40 x 40 x 100 x 100
        assert!((sides[0] - 40.0).abs() < 0.01, "sides={sides:?}");
        assert!((sides[1] - 40.0).abs() < 0.01, "sides={sides:?}");
        assert!((sides[2] - 100.0).abs() < 0.01, "sides={sides:?}");
        assert!((sides[3] - 100.0).abs() < 0.01, "sides={sides:?}");

        // and at least one corner is genuinely off-axis: no two corners share
        // an x or a y, which is what an axis-aligned box would give.
        let axis_aligned = pts
            .iter()
            .any(|p| pts.iter().filter(|q| (q.0 - p.0).abs() < 1e-6).count() > 1);
        assert!(!axis_aligned, "corners are axis-aligned: {pts:?}");

        // the axis-aligned collapse would have been 106.60 x 84.64
        assert!(
            sides.iter().all(|s| (s - 106.60).abs() > 1.0),
            "looks like the axis-aligned bounds: {sides:?}"
        );
    }

    /// B2.4 -- an unrepresentable shape is counted, not silently discarded.
    #[test]
    fn unsupported_shapes_are_counted_not_dropped_silently() {
        let json = r#"{
            "shapes": [
                {"label": "part", "points": [[10,20],[100,200]], "shape_type": "rectangle"},
                {"label": "part", "points": [[5,5],[9,9]], "shape_type": "circle"},
                {"label": "part", "points": [[1,1],[2,2]], "shape_type": "polygon"},
                {"label": "ghost", "points": [[1,1],[2,2]], "shape_type": "rectangle"}
            ],
            "imageWidth": 640, "imageHeight": 480, "imagePath": "a.jpg"
        }"#;
        let (out, (emitted, dropped, malformed, unknown)) = convert(json, ENC_BBOX5);
        assert_eq!(dropped, 1, "the circle must be reported");
        assert_eq!(malformed, 1, "the 2-point polygon is an error, not a report");
        assert_eq!(unknown, 1, "the unmapped label is counted separately");
        assert_eq!(emitted, 1);
        assert_eq!(out.lines().count(), 1);
        // read == emitted + dropped + malformed + unknown_class, always
        assert_eq!(4, emitted + dropped + malformed + unknown);
    }

    /// B2.2 -- an encoding this converter cannot produce is an error, never a
    /// quiet substitution of the one it can.
    #[test]
    fn an_unknown_encoding_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let json_path = dir.path().join("g.json");
        fs::write(&json_path, GOLDEN_BBOX5_LABELME).unwrap();
        let txt_path = dir.path().join("g.txt");
        let result = convert_labelme_to_yolo(
            json_path.to_str().unwrap(),
            txt_path.to_str().unwrap(),
            class_map(),
            "pose5+3k",
        );
        assert!(result.is_err());
        // The rejection happens before anything is written -- the encoding is
        // validated first precisely so a refused job leaves no partial labels.
        assert!(!txt_path.exists(), "a refused encoding still wrote a label file");
        // asserted on the message builder, which needs no interpreter
        let msg = unsupported_encoding_message("pose5+3k");
        assert!(msg.contains("pose5+3k"), "{msg}");
        assert!(msg.contains("obb9"), "{msg}");
    }

    /// The directory path emits the same bytes as the single-file path, and
    /// reconciles its counts the same way.
    #[test]
    fn dir_and_single_file_paths_agree() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        fs::write(src.path().join("g.json"), GOLDEN_BBOX5_LABELME).unwrap();

        let (files, emitted, dropped, malformed, unknown) = convert_labelme_dir_to_yolo(
            src.path().to_str().unwrap(),
            dst.path().to_str().unwrap(),
            vec!["part".to_string(), "hole".to_string()],
            ENC_BBOX5,
        )
        .unwrap();
        assert_eq!((files, emitted, dropped, malformed, unknown), (1, 2, 0, 0, 0));
        assert_eq!(
            fs::read_to_string(dst.path().join("g.txt")).unwrap(),
            GOLDEN_BBOX5_OUTPUT
        );
    }

    #[test]
    fn min_area_rect_recovers_an_axis_aligned_box() {
        let pts = [(10.0, 20.0), (110.0, 20.0), (110.0, 60.0), (10.0, 60.0)];
        let r = min_area_rect(&pts);
        let xs: Vec<f64> = r.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = r.iter().map(|p| p.1).collect();
        assert!((xs.iter().cloned().fold(f64::INFINITY, f64::min) - 10.0).abs() < 1e-6);
        assert!((xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - 110.0).abs() < 1e-6);
        assert!((ys.iter().cloned().fold(f64::INFINITY, f64::min) - 20.0).abs() < 1e-6);
        assert!((ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - 60.0).abs() < 1e-6);
    }
}
