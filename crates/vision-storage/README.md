# vision-storage

Rust optimizations for storage-server (PyO3 bindings).

## Functions

- `verify_encoding(file_path)` - Fast UTF-8/encoding detection (50-100x faster than chardet)
- `verify_encoding_bytes(data)` - Encoding detection from bytes
- `generate_thumbnail(image_bytes, max_size, quality)` - Fast thumbnail generation (5-10x faster than PIL)
- `parse_coco_to_labelme(coco_json)` - COCO to LabelMe conversion (10-20x faster)
- `extract_zip(zip_path, dest_dir)` - Fast ZIP extraction (4-6x faster)
- `compress_directory(source_dir, zip_path, level)` - Fast ZIP compression (4-6x faster)

## Installation

```bash
pip install vision_storage-*.whl
```

## Usage

```python
from vision_storage import (
    verify_encoding,
    generate_thumbnail,
    parse_coco_to_labelme,
    extract_zip,
    compress_directory
)

# Encoding detection
content, encoding = verify_encoding("/path/to/file.json")

# Thumbnail generation
thumbnail_bytes = generate_thumbnail(image_bytes, max_size=200, quality=85)

# COCO to LabelMe
labelme_list = parse_coco_to_labelme(coco_json_string)

# ZIP operations
files = extract_zip("/path/to/file.zip", "/dest/dir")
compress_directory("/source/dir", "/output.zip", compression_level=6)
```
