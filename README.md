# Vision Rust Core

High-performance Rust extensions for Vision AI servers.

## Overview

This workspace contains Rust modules that replace CPU-bound and I/O-intensive operations in the Vision AI platform:

| Crate | Target | Bindings | Description |
|-------|--------|----------|-------------|
| `vision-storage` | storage-server | PyO3 | Thumbnails, encoding, COCO parsing, ZIP |
| `vision-training` | training-server | PyO3 | EfficientAD pipeline, downloads, conversions |
| `vision-ai-node` | vision-ai | Neon | SIMD JSON, ZIP, transformations |
| `shared` | Internal | - | Common utilities |

## Performance Gains

| Operation | Python/JS | Rust | Speedup |
|-----------|-----------|------|---------|
| Encoding detection (chardet) | ~200ms | ~2ms | **100x** |
| Thumbnail generation | ~200ms | ~20ms | **10x** |
| COCO JSON parsing (10k) | ~3s | ~150ms | **20x** |
| ZIP compression | ~4s | ~700ms | **6x** |
| JSON.parse (MQTT) | ~5ms | ~1ms | **5x** |
| EfficientAD inference | ~500ms | ~50ms | **10x** |

## Quick Start

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Install maturin (for Python bindings)
pip install maturin
```

### Build All Crates

```bash
cd vision-rust-core

# Build workspace
cargo build --release

# Run tests
cargo test
```

### Build Python Modules

```bash
# vision-storage
cd crates/vision-storage
maturin build --release
pip install target/wheels/*.whl

# vision-training
cd ../vision-training
maturin build --release
pip install target/wheels/*.whl
```

### Build Node.js Module

```bash
cd crates/vision-ai-node
npm install
npm run build
```

## Usage Examples

### vision-storage (Python)

```python
import vision_storage as vs

# Fast thumbnail generation
thumbnail_bytes = vs.generate_thumbnail(image_bytes, max_size=200, quality=85)

# Fast encoding detection (replaces chardet)
content, encoding = vs.verify_encoding("/path/to/file.json")

# COCO to LabelMe conversion
labelme_annotations = vs.parse_coco_to_labelme(coco_json_string)

# ZIP operations
vs.extract_zip("/path/to/file.zip", "/dest/dir")
vs.compress_directory("/source/dir", "/output.zip", compression_level=6)
```

### vision-training (Python)

```python
import vision_training as vt
import numpy as np

# EfficientAD inference pipeline
st_map, ae_map, combined, max_score = vt.compute_anomaly_maps(
    teacher_out, student_out, ae_out, ae_target,
    st_weight=0.5, ae_weight=0.5
)

# Find anomaly regions
boxes = vt.find_bounding_boxes(binary_mask, min_area=50)

# Generate overlay
overlay = vt.generate_overlay(image, heatmap, alpha=0.5)

# Fast parallel downloads
results = vt.download_files([
    ("http://example.com/file1.jpg", "/dest/file1.jpg"),
    ("http://example.com/file2.jpg", "/dest/file2.jpg"),
], max_concurrent=50)

# LabelMe to YOLO conversion
vt.convert_labelme_dir_to_yolo(
    "/input/labelme",
    "/output/yolo",
    ["class1", "class2", "class3"]
)

# Fast directory walking
images = vt.find_images("/data/images", max_depth=3)
```

### vision-ai-node (Node.js)

```typescript
import {
  jsonParse,
  jsonStringify,
  createZip,
  transformRevision
} from 'vision-ai-node';

// SIMD JSON parsing (for MQTT messages)
const metrics = jsonParse(mqttPayload);

// Fast ZIP creation
const fileCount = createZip('/source/dir', '/output.zip', 6);

// Batch transformation
const transformed = transformRevision(revisionData);
```

## Architecture

```
vision-rust-core/
├── Cargo.toml                 # Workspace configuration
├── crates/
│   ├── shared/                # Common utilities
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── error.rs       # Error types
│   │   │   ├── image_ops.rs   # Image operations
│   │   │   └── io_utils.rs    # I/O utilities
│   │   └── Cargo.toml
│   │
│   ├── vision-storage/        # storage-server (PyO3)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── thumbnail.rs
│   │   │   ├── encoding.rs
│   │   │   ├── coco_parser.rs
│   │   │   └── zip_ops.rs
│   │   ├── Cargo.toml
│   │   └── pyproject.toml
│   │
│   ├── vision-training/       # training-server (PyO3)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── efficientad/
│   │   │   ├── downloader.rs
│   │   │   ├── labelme_yolo.rs
│   │   │   └── walker.rs
│   │   ├── Cargo.toml
│   │   └── pyproject.toml
│   │
│   └── vision-ai-node/        # vision-ai (Neon)
│       ├── src/
│       │   ├── lib.rs
│       │   ├── json_simd.rs
│       │   ├── zipper.rs
│       │   └── transformer.rs
│       ├── Cargo.toml
│       ├── package.json
│       └── index.d.ts
│
└── README.md
```

## Development

### Running Tests

```bash
# All tests
cargo test

# Specific crate
cargo test -p vision-storage
cargo test -p vision-training
```

### Benchmarks

```bash
# Build with optimizations
cargo build --release

# Run benchmarks (if implemented)
cargo bench
```

### Code Formatting

```bash
cargo fmt
cargo clippy
```

## Integration with Servers

### storage-server

Add to `requirements.txt`:
```
vision-storage @ file:///path/to/vision-rust-core/crates/vision-storage/target/wheels/vision_storage-*.whl
```

Replace in Python code:
```python
# Before
from api.utils.thumbnail import generate_thumbnail
from api.utils.analysis.verifyEncoding import verify_encoding

# After
from vision_storage import generate_thumbnail, verify_encoding
```

### training-server

Add to `requirements.txt`:
```
vision-training @ file:///path/to/vision-rust-core/crates/vision-training/target/wheels/vision_training-*.whl
```

### vision-ai

Add to `package.json`:
```json
{
  "dependencies": {
    "vision-ai-node": "file:../vision-rust-core/crates/vision-ai-node"
  }
}
```

## License

MIT
