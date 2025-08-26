use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString, PyTuple};
use serde_json::{self, Number, Value};

// Define SecurityError to match DataSON's Python SecurityError
pyo3::create_exception!(
    _datason_rust,
    SecurityError,
    pyo3::exceptions::PyRuntimeError,
    "DataSON security limit exceeded"
);

fn py_to_value(
    obj: &PyAny,
    depth: usize,
    max_depth: usize,
    total: &mut usize,
    max_total: usize,
    max_str: usize,
) -> PyResult<Value> {
    if depth > max_depth {
        return Err(SecurityError::new_err("Maximum serialization depth exceeded"));
    }
    if *total > max_total {
        return Err(SecurityError::new_err("Object too large"));
    }
    if obj.is_none() {
        *total += 1;
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.extract::<bool>() {
        *total += 1;
        return Ok(Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        *total += 8;
        return Ok(Value::Number(Number::from(i)));
    }
    if let Ok(f) = obj.extract::<f64>() {
        if !f.is_finite() {
            return Err(PyErr::new::<PyTypeError, _>("Non-finite float"));
        }
        *total += 8;
        let n = Number::from_f64(f).ok_or_else(|| PyTypeError::new_err("Invalid float"))?;
        return Ok(Value::Number(n));
    }
    if let Ok(s) = obj.extract::<String>() {
        if s.len() > max_str {
            return Err(SecurityError::new_err("String length exceeds limit"));
        }
        *total += s.len();
        if *total > max_total {
            return Err(SecurityError::new_err("Object too large"));
        }
        return Ok(Value::String(s));
    }
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            let val = py_to_value(v, depth + 1, max_depth, total, max_total, max_str)?;
            map.insert(key, val);
        }
        return Ok(Value::Object(map));
    }
    if let Ok(list) = obj.downcast::<PyList>() {
        let mut vec = Vec::with_capacity(list.len());
        for item in list.iter() {
            let val = py_to_value(item, depth + 1, max_depth, total, max_total, max_str)?;
            vec.push(val);
        }
        return Ok(Value::Array(vec));
    }
    if let Ok(tuple) = obj.downcast::<PyTuple>() {
        let mut vec = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            let val = py_to_value(item, depth + 1, max_depth, total, max_total, max_str)?;
            vec.push(val);
        }
        return Ok(Value::Array(vec));
    }
    Err(PyTypeError::new_err("unsupported type"))
}

fn value_to_py(
    py: Python,
    value: &Value,
    depth: usize,
    max_depth: usize,
    total: &mut usize,
    max_total: usize,
    max_str: usize,
) -> PyResult<PyObject> {
    if depth > max_depth {
        return Err(SecurityError::new_err("Maximum deserialization depth exceeded"));
    }
    if *total > max_total {
        return Err(SecurityError::new_err("Object too large"));
    }
    Ok(match value {
        Value::Null => {
            *total += 1;
            py.None()
        }
        Value::Bool(b) => {
            *total += 1;
            b.into_py(py)
        }
        Value::Number(n) => {
            *total += 8;
            if let Some(i) = n.as_i64() {
                i.into_py(py)
            } else if let Some(f) = n.as_f64() {
                if !f.is_finite() {
                    return Err(PyTypeError::new_err("Non-finite float"));
                }
                f.into_py(py)
            } else if let Some(u) = n.as_u64() {
                (u as i64).into_py(py)
            } else {
                return Err(PyTypeError::new_err("Invalid number"));
            }
        }
        Value::String(s) => {
            if s.len() > max_str {
                return Err(SecurityError::new_err("String length exceeds limit"));
            }
            *total += s.len();
            PyString::new(py, s).into_py(py)
        }
        Value::Array(arr) => {
            let mut vec = Vec::with_capacity(arr.len());
            for v in arr {
                let obj = value_to_py(py, v, depth + 1, max_depth, total, max_total, max_str)?;
                vec.push(obj);
            }
            PyList::new(py, vec).into_py(py)
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                if k.len() > max_str {
                    return Err(SecurityError::new_err("String length exceeds limit"));
                }
                let val = value_to_py(py, v, depth + 1, max_depth, total, max_total, max_str)?;
                dict.set_item(k, val)?;
            }
            dict.into_py(py)
        }
    })
}

#[pyfunction]
fn dumps_core(
    py: Python,
    obj: PyObject,
    ensure_ascii: bool,
    allow_nan: bool,
    max_depth: usize,
    max_total_bytes: usize,
    max_string_length: usize,
) -> PyResult<Py<PyBytes>> {
    let mut total = 0usize;
    let value = py_to_value(
        obj.as_ref(py),
        0,
        max_depth,
        &mut total,
        max_total_bytes,
        max_string_length,
    )?;
    let s = if ensure_ascii {
        serde_json::to_string(&value)?
    } else {
        serde_json::to_string(&value)?
    };
    if s.len() > max_total_bytes {
        return Err(SecurityError::new_err("Serialized object too large"));
    }
    Ok(PyBytes::new(py, s.as_bytes()).into())
}

#[pyfunction]
fn loads_core(
    py: Python,
    data: PyObject,
    max_depth: usize,
    max_total_bytes: usize,
    max_string_length: usize,
) -> PyResult<PyObject> {
    let bytes: Vec<u8> = if let Ok(s) = data.extract::<&str>(py) {
        s.as_bytes().to_vec()
    } else {
        data.extract::<Vec<u8>>(py)?
    };
    if bytes.len() > max_total_bytes {
        return Err(SecurityError::new_err("Input too large"));
    }
    let value: Value = serde_json::from_slice(&bytes)?;
    let mut total = 0usize;
    let obj = value_to_py(
        py,
        &value,
        0,
        max_depth,
        &mut total,
        max_total_bytes,
        max_string_length,
    )?;
    Ok(obj)
}

#[pymodule]
fn _datason_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dumps_core, m)?)?;
    m.add_function(wrap_pyfunction!(loads_core, m)?)?;
    m.add("SecurityError", _py.get_type::<SecurityError>())?;
    Ok(())
}
