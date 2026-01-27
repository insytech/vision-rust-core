//! vision-ai-node: Rust optimizations for vision-ai Node.js server
//!
//! High-performance replacements using Neon bindings:
//! - SIMD JSON parsing (3-5x faster)
//! - ZIP operations (4-6x faster)
//! - Data transformations (2-4x faster)

use neon::prelude::*;

mod json_simd;
mod zipper;
mod transformer;

#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
    // JSON functions
    cx.export_function("jsonParse", json_simd::json_parse)?;
    cx.export_function("jsonStringify", json_simd::json_stringify)?;
    cx.export_function("jsonParseArray", json_simd::json_parse_array)?;

    // ZIP functions
    cx.export_function("createZip", zipper::create_zip)?;
    cx.export_function("extractZip", zipper::extract_zip)?;
    cx.export_function("createZipFromFiles", zipper::create_zip_from_files)?;

    // Transformer functions
    cx.export_function("transformRevision", transformer::transform_revision)?;
    cx.export_function("transformFiles", transformer::transform_files)?;
    cx.export_function("transformClasses", transformer::transform_classes)?;

    Ok(())
}
