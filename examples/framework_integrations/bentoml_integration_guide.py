#!/usr/bin/env python3
"""BentoML integration example using datason for JSON handling."""

import bentoml
from bentoml.io import JSON

import datason
from datason.config import get_api_config

API_CONFIG = get_api_config()


# Dummy model function
def run_model(payload: dict) -> dict:
    return {"echo": payload, "prediction": 1}


svc = bentoml.Service("datason_bentoml_demo")


@svc.api(input=JSON(), output=JSON())
def predict(parsed_json: dict) -> dict:
    data = datason.auto_deserialize(parsed_json, config=API_CONFIG)
    result = run_model(data)
    return datason.serialize(result, config=API_CONFIG)


if __name__ == "__main__":
    # `bentoml serve` will launch this service
    print("Run with: bentoml serve bentoml_integration_guide:svc --reload")
