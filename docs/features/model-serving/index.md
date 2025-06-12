# Model Serving Integration

Comprehensive guides for using **datason** with popular model serving and experiment tracking frameworks.

These recipes show how to plug datason into each framework so your ML objects round-trip cleanly between training code, servers, and web APIs.

## 🚀 BentoML

```python
import bentoml
from bentoml.io import JSON
import datason
from datason.config import get_api_config

svc = bentoml.Service("my_model_service")
API_CONFIG = get_api_config()

@svc.api(input=JSON(), output=JSON())
def predict(parsed_json: dict) -> dict:
    # Parse incoming JSON with datason
    data = datason.auto_deserialize(parsed_json, config=API_CONFIG)
    prediction = run_model(data)
    # Serialize response for BentoML
    return datason.serialize(prediction, config=API_CONFIG)
```

## ⚡ Ray Serve

```python
from ray import serve
import datason
from datason.config import get_api_config

API_CONFIG = get_api_config()

@serve.deployment
class ModelDeployment:
    async def __call__(self, request):
        payload = await request.json()
        data = datason.auto_deserialize(payload, config=API_CONFIG)
        result = run_model(data)
        return datason.serialize(result, config=API_CONFIG)
```

## 🎛️ Streamlit / Gradio

```python
import streamlit as st
import datason
from datason.config import get_api_config

API_CONFIG = get_api_config()

uploaded = st.file_uploader("Upload JSON")
if uploaded:
    raw = uploaded.read().decode()
    data = datason.loads(raw, config=API_CONFIG)
    prediction = run_model(data)
    st.json(datason.dumps(prediction, config=API_CONFIG))
```

For Gradio, use the same pattern inside your `fn` callbacks.

## 📦 MLflow Artifacts

```python
import mlflow
import datason

with mlflow.start_run():
    result = train_model()
    mlflow.log_dict(datason.serialize(result), "results.json")
```

## ☁️ Seldon Core / KServe

```python
from seldon_core.user_model import SeldonComponent
import datason
from datason.config import get_api_config

API_CONFIG = get_api_config()

class Model(SeldonComponent):
    def predict(self, features, **kwargs):
        data = datason.auto_deserialize(features, config=API_CONFIG)
        output = run_model(data)
        return datason.serialize(output, config=API_CONFIG)
```

These integrations ensure consistent JSON handling across your ML stack—BentoML or Ray Serve for serving, Streamlit/Gradio for demos, and MLflow for experiment tracking—all powered by datason's ML-friendly serialization.
