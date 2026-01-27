# vision-training

Rust optimizations for training-server (PyO3 bindings).

## Functions

### EfficientAD Pipeline
- `compute_anomaly_maps()` - ST/AE distance maps computation
- `find_bounding_boxes()` - Connected components with Union-Find
- `generate_overlay()` - Heatmap overlay with alpha blending
- `generate_mask()` - Binary mask from percentile threshold
- `array_to_base64()` - PNG encoding to base64
- `normalize_heatmap()` - Percentile-based normalization
- `compute_percentiles()` - Fast percentile computation

### File Operations
- `download_files()` - Parallel file downloads with tokio
- `download_file()` - Single file download
- `convert_labelme_to_yolo()` - LabelMe to YOLO conversion
- `convert_labelme_dir_to_yolo()` - Batch directory conversion
- `fix_json_image_paths()` - Fix imagePath in LabelMe JSONs

### Directory Walking
- `find_images()` - Fast recursive image search
- `find_json_files()` - Fast recursive JSON search
- `walk_directory()` - Generic directory walking

## Installation

```bash
pip install vision_training-*.whl
```
