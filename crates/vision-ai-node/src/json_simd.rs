//! SIMD-accelerated JSON parsing
//!
//! Replaces Node.js JSON.parse/stringify with simd-json.
//! Expected speedup: 3-5x for large payloads

use neon::prelude::*;

/// Parse JSON string using SIMD acceleration
pub fn json_parse(mut cx: FunctionContext) -> JsResult<JsValue> {
    let json_str = cx.argument::<JsString>(0)?.value(&mut cx);

    // Use simd-json for parsing
    let mut bytes = json_str.into_bytes();

    let value: serde_json::Value = match simd_json::serde::from_slice(&mut bytes) {
        Ok(v) => v,
        Err(e) => return cx.throw_error(format!("JSON parse error: {}", e)),
    };

    // Convert serde_json::Value to Neon JsValue
    serde_value_to_js(&mut cx, &value)
}

/// Stringify JavaScript value to JSON
pub fn json_stringify(mut cx: FunctionContext) -> JsResult<JsString> {
    let value = cx.argument::<JsValue>(0)?;
    let pretty = cx
        .argument_opt(1)
        .and_then(|v| v.downcast::<JsBoolean, _>(&mut cx).ok())
        .map(|b| b.value(&mut cx))
        .unwrap_or(false);

    // Convert JsValue to serde_json::Value
    let serde_value = js_to_serde_value(&mut cx, value)?;

    // Serialize
    let json_str = if pretty {
        serde_json::to_string_pretty(&serde_value)
    } else {
        serde_json::to_string(&serde_value)
    }
    .map_err(|e| cx.throw_error::<_, String>(format!("JSON stringify error: {}", e)).unwrap_err())?;

    Ok(cx.string(json_str))
}

/// Parse JSON array with SIMD - optimized for arrays of objects
pub fn json_parse_array(mut cx: FunctionContext) -> JsResult<JsArray> {
    let json_str = cx.argument::<JsString>(0)?.value(&mut cx);
    let mut bytes = json_str.into_bytes();

    let value: serde_json::Value = match simd_json::serde::from_slice(&mut bytes) {
        Ok(v) => v,
        Err(e) => return cx.throw_error(format!("JSON parse error: {}", e)),
    };

    match value {
        serde_json::Value::Array(arr) => {
            let js_arr = JsArray::new(&mut cx, arr.len());
            for (i, item) in arr.iter().enumerate() {
                let js_val = serde_value_to_js(&mut cx, item)?;
                js_arr.set(&mut cx, i as u32, js_val)?;
            }
            Ok(js_arr)
        }
        _ => cx.throw_error("Expected JSON array"),
    }
}

/// Convert serde_json::Value to Neon JsValue
fn serde_value_to_js<'a>(
    cx: &mut FunctionContext<'a>,
    value: &serde_json::Value,
) -> JsResult<'a, JsValue> {
    match value {
        serde_json::Value::Null => Ok(cx.null().upcast()),
        serde_json::Value::Bool(b) => Ok(cx.boolean(*b).upcast()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(cx.number(i as f64).upcast())
            } else if let Some(f) = n.as_f64() {
                Ok(cx.number(f).upcast())
            } else {
                Ok(cx.number(0.0).upcast())
            }
        }
        serde_json::Value::String(s) => Ok(cx.string(s).upcast()),
        serde_json::Value::Array(arr) => {
            let js_arr = JsArray::new(cx, arr.len());
            for (i, item) in arr.iter().enumerate() {
                let js_val = serde_value_to_js(cx, item)?;
                js_arr.set(cx, i as u32, js_val)?;
            }
            Ok(js_arr.upcast())
        }
        serde_json::Value::Object(obj) => {
            let js_obj = cx.empty_object();
            for (key, val) in obj {
                let js_val = serde_value_to_js(cx, val)?;
                js_obj.set(cx, key.as_str(), js_val)?;
            }
            Ok(js_obj.upcast())
        }
    }
}

/// Convert Neon JsValue to serde_json::Value
fn js_to_serde_value(
    cx: &mut FunctionContext,
    value: Handle<JsValue>,
) -> NeonResult<serde_json::Value> {
    if value.is_a::<JsNull, _>(cx) || value.is_a::<JsUndefined, _>(cx) {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = value.downcast::<JsBoolean, _>(cx) {
        Ok(serde_json::Value::Bool(b.value(cx)))
    } else if let Ok(n) = value.downcast::<JsNumber, _>(cx) {
        let num = n.value(cx);
        Ok(serde_json::Number::from_f64(num)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null))
    } else if let Ok(s) = value.downcast::<JsString, _>(cx) {
        Ok(serde_json::Value::String(s.value(cx)))
    } else if let Ok(arr) = value.downcast::<JsArray, _>(cx) {
        let len = arr.len(cx);
        let mut vec = Vec::with_capacity(len as usize);
        for i in 0..len {
            let item: Handle<JsValue> = arr.get(cx, i)?;
            vec.push(js_to_serde_value(cx, item)?);
        }
        Ok(serde_json::Value::Array(vec))
    } else if let Ok(obj) = value.downcast::<JsObject, _>(cx) {
        let keys = obj.get_own_property_names(cx)?;
        let len = keys.len(cx);
        let mut map = serde_json::Map::new();
        for i in 0..len {
            let key: Handle<JsString> = keys.get(cx, i)?;
            let key_str = key.value(cx);
            let val: Handle<JsValue> = obj.get(cx, key)?;
            map.insert(key_str, js_to_serde_value(cx, val)?);
        }
        Ok(serde_json::Value::Object(map))
    } else {
        Ok(serde_json::Value::Null)
    }
}
