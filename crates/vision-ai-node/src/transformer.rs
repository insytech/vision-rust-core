//! Data transformation optimizations
//!
//! Replaces JavaScript .map() operations on large arrays.
//! Expected speedup: 2-4x for large datasets

use neon::prelude::*;

/// Transform revision data for API response
pub fn transform_revision(mut cx: FunctionContext) -> JsResult<JsObject> {
    let revision_obj = cx.argument::<JsObject>(0)?;

    // Extract revision fields
    let id: Handle<JsString> = revision_obj.get(&mut cx, "id")?;
    let name: Handle<JsValue> = revision_obj.get(&mut cx, "name")?;
    let status: Handle<JsValue> = revision_obj.get(&mut cx, "status")?;

    // Transform classes if present
    let classes: Handle<JsValue> = revision_obj.get(&mut cx, "Class")?;
    let transformed_classes = if classes.is_a::<JsArray, _>(&mut cx) {
        let arr = classes.downcast::<JsArray, _>(&mut cx).unwrap();
        let len = arr.len(&mut cx);
        let result = JsArray::new(&mut cx, len as usize);
        for i in 0..len {
            let class_obj: Handle<JsObject> = arr.get(&mut cx, i)?;
            let transformed = cx.empty_object();
            let cid: Handle<JsValue> = class_obj.get(&mut cx, "id")?;
            let cname: Handle<JsValue> = class_obj.get(&mut cx, "name")?;
            let ccolor: Handle<JsValue> = class_obj.get(&mut cx, "color")?;
            transformed.set(&mut cx, "id", cid)?;
            transformed.set(&mut cx, "name", cname)?;
            transformed.set(&mut cx, "color", ccolor)?;
            result.set(&mut cx, i, transformed)?;
        }
        result
    } else {
        JsArray::new(&mut cx, 0)
    };

    // Transform files if present
    let files: Handle<JsValue> = revision_obj.get(&mut cx, "File")?;
    let transformed_files = if files.is_a::<JsArray, _>(&mut cx) {
        let arr = files.downcast::<JsArray, _>(&mut cx).unwrap();
        let len = arr.len(&mut cx);
        let result = JsArray::new(&mut cx, len as usize);
        for i in 0..len {
            let file_obj: Handle<JsObject> = arr.get(&mut cx, i)?;
            let transformed = cx.empty_object();
            let fid: Handle<JsValue> = file_obj.get(&mut cx, "id")?;
            let fname: Handle<JsValue> = file_obj.get(&mut cx, "name")?;
            let furl: Handle<JsValue> = file_obj.get(&mut cx, "url")?;
            transformed.set(&mut cx, "id", fid)?;
            transformed.set(&mut cx, "name", fname)?;
            transformed.set(&mut cx, "url", furl)?;
            result.set(&mut cx, i, transformed)?;
        }
        result
    } else {
        JsArray::new(&mut cx, 0)
    };

    // Build result object
    let result = cx.empty_object();
    result.set(&mut cx, "id", id)?;
    result.set(&mut cx, "name", name)?;
    result.set(&mut cx, "status", status)?;
    result.set(&mut cx, "classes", transformed_classes)?;
    result.set(&mut cx, "files", transformed_files)?;

    Ok(result)
}

/// Transform files array
pub fn transform_files(mut cx: FunctionContext) -> JsResult<JsArray> {
    let files_arr = cx.argument::<JsArray>(0)?;
    let len = files_arr.len(&mut cx);
    let result = JsArray::new(&mut cx, len as usize);

    for i in 0..len {
        let file_obj: Handle<JsObject> = files_arr.get(&mut cx, i)?;
        let transformed = cx.empty_object();

        let id: Handle<JsValue> = file_obj.get(&mut cx, "id")?;
        let name: Handle<JsValue> = file_obj.get(&mut cx, "name")?;
        let url: Handle<JsValue> = file_obj.get(&mut cx, "url")?;
        let file_type: Handle<JsValue> = file_obj.get(&mut cx, "fileType")?;

        transformed.set(&mut cx, "id", id)?;
        transformed.set(&mut cx, "name", name)?;
        transformed.set(&mut cx, "url", url)?;
        transformed.set(&mut cx, "fileType", file_type)?;

        let thumbnail: Handle<JsValue> = file_obj.get(&mut cx, "thumbnailUrl")?;
        if !thumbnail.is_a::<JsNull, _>(&mut cx) && !thumbnail.is_a::<JsUndefined, _>(&mut cx) {
            transformed.set(&mut cx, "thumbnailUrl", thumbnail)?;
        }

        result.set(&mut cx, i, transformed)?;
    }

    Ok(result)
}

/// Transform classes array
pub fn transform_classes(mut cx: FunctionContext) -> JsResult<JsArray> {
    let classes_arr = cx.argument::<JsArray>(0)?;
    let len = classes_arr.len(&mut cx);
    let result = JsArray::new(&mut cx, len as usize);

    for i in 0..len {
        let class_obj: Handle<JsObject> = classes_arr.get(&mut cx, i)?;
        let transformed = cx.empty_object();

        let id: Handle<JsValue> = class_obj.get(&mut cx, "id")?;
        let name: Handle<JsValue> = class_obj.get(&mut cx, "name")?;
        let color: Handle<JsValue> = class_obj.get(&mut cx, "color")?;

        transformed.set(&mut cx, "id", id)?;
        transformed.set(&mut cx, "name", name)?;
        transformed.set(&mut cx, "color", color)?;

        let index: Handle<JsValue> = class_obj.get(&mut cx, "index")?;
        if let Ok(idx_str) = index.downcast::<JsString, _>(&mut cx) {
            let idx_val = idx_str.value(&mut cx);
            if let Ok(num) = idx_val.parse::<f64>() {
                let js_num = cx.number(num);
                transformed.set(&mut cx, "index", js_num)?;
            } else {
                transformed.set(&mut cx, "index", index)?;
            }
        } else {
            transformed.set(&mut cx, "index", index)?;
        }

        result.set(&mut cx, i, transformed)?;
    }

    Ok(result)
}
