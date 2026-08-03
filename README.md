# Vision Rust Core

Rust extensions for the Vision AI platform.

## Overview

| Crate | Target | Bindings | Status |
|-------|--------|----------|--------|
| `vision-training` | training-server | PyO3 | **Deployed.** Wheel built and installed; ABI pinned |
| `shared` | Internal | - | Common utilities |

`vision-storage` was removed on 2026-08-03. No `Dockerfile`, `requirements.txt`
or deploy script ever installed its wheel, so every call site in
`storage-server` and `training-server` always took the Python fallback, and its
one measurable win — `generate_thumbnail` — is now covered by `PIL.draft()` in
`storage-server/src/api/utils/thumbnail.py`.

`vision-ai-node` was removed on 2026-08-03, the whole crate this time. Its
`json_simd.rs` had already gone: benchmarked at **0.27x** for parse and
**0.14x** for stringify, i.e. V8's native `JSON.parse` / `JSON.stringify` are
4x and 7x faster. What remained — `zipper.rs` and `transformer.rs` — was never
loaded by any deployment: no `package.json` or `pnpm-lock.yaml` declared the
module, and `vision-ai/server/.dockerignore` excludes `node_modules`, so the
`require` inside `rust-optimized.ts` always threw and the `archiver` fallback
always ran. The ZIP path did measure **2.69x** faster than `archiver`
(120 x 60 KB files, level 9: 66.2 ms vs 178.2 ms), but realising it would mean
shipping a prebuilt `.node` per architecture into a `node:21-alpine` image that
has no Rust toolchain — for a single manual installer download. The
`rust-optimized.ts` wrapper and its tests went with it; the endpoint now calls
`archiver` directly, with no branch.

## Why `vision-training` is kept

Not for speed — its conversion is only ~1.2x faster than the Python path. It is
kept for **capability**: it is the only converter that emits `obb9` (the nine
DOTA values for oriented boxes). The `labelme2yolo` CLI can only emit `bbox5`.

Its Python module version (`vision_training.__version__`) is an ABI contract
that `training-server` checks at import time; bumping the crate version without
rebuilding and reinstalling the wheel disables the Rust path with a logged
mismatch instead of failing silently. See
`vision-device/docs/deploy-gates-model-type-pipeline.md`.

## Quick Start

### Prerequisites

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
pip install maturin
```

### Build and test

```bash
cd vision-rust-core
cargo build --release

# `cargo test` links against libpython, so it needs the interpreter's libdir.
LD_LIBRARY_PATH=/root/miniconda3/lib cargo test --workspace
```

### Build the Python module

The crate builds against CPython's stable ABI (`pyo3/abi3-py310`), so the
result is a **single** wheel -- `vision_training-<ver>-cp310-abi3-<plat>.whl` --
that installs on 3.10 (the training server) through 3.13 (dev boxes). Build it
with the *oldest* supported interpreter; a newer one still produces a
`cp310-abi3` tag but pins a newer glibc into the platform tag.

```bash
cd crates/vision-training
maturin build --release --interpreter /root/miniconda3/envs/vision-training-v2/bin/python
pip install ../../target/wheels/vision_training-*.whl
```

Keep exactly one `vision_training-*.whl` in whatever directory you install
from: `Dockerfile.gpu` installs by glob, and a leftover per-interpreter wheel
next to the abi3 one makes which `.so` wins depend on shell glob order.

## Usage

### vision-training (Python)

```python
import vision_training as vt
import numpy as np

# EfficientAD inference pipeline
st_map, ae_map, combined, max_score = vt.compute_anomaly_maps(
    teacher_out, student_out, ae_out, ae_target,
    st_weight=0.5, ae_weight=0.5
)

# LabelMe to YOLO conversion (the reason this crate exists: obb9)
vt.convert_labelme_dir_to_yolo(
    "/input/labelme",
    "/output/yolo",
    ["class1", "class2", "class3"]
)

# Parallel downloads, directory walking
results = vt.download_files([("http://example.com/a.jpg", "/dest/a.jpg")], max_concurrent=50)
images = vt.find_images("/data/images", max_depth=3)
```

Some exports (`find_bounding_boxes`, `generate_overlay`, `generate_mask`) were
benchmarked **slower** than the OpenCV/NumPy equivalents (0.04x, 0.1x, 0.2x).
They remain exported because the shipped wheel's ABI is pinned; do not reach
for them in new code.

## Architecture

```
vision-rust-core/
├── Cargo.toml                 # Workspace configuration
├── crates/
│   ├── shared/                # Common utilities
│   │   └── src/{lib,error,image_ops,io_utils}.rs
│   │
│   └── vision-training/       # training-server (PyO3)
│       ├── src/
│       │   ├── lib.rs
│       │   ├── efficientad/
│       │   ├── downloader.rs
│       │   ├── labelme_yolo.rs
│       │   └── walker.rs
│       ├── Cargo.toml
│       └── pyproject.toml
│
└── README.md
```

## Development

```bash
LD_LIBRARY_PATH=/root/miniconda3/lib cargo test --workspace
cargo fmt
cargo clippy
cargo bench
```

## Integration with servers

### training-server

Add to `requirements.txt`:
```
vision-training @ file:///path/to/vision-rust-core/target/wheels/vision_training-*.whl
```

## License

MIT
