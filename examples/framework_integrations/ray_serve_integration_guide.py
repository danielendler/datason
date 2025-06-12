#!/usr/bin/env python3
"""Ray Serve deployment using datason for request/response handling."""

from ray import serve

import datason
from datason.config import get_api_config

API_CONFIG = get_api_config()


# Dummy model function
def run_model(payload: dict) -> dict:
    return {"echo": payload, "score": 0.5}


@serve.deployment
class ModelDeployment:
    async def __call__(self, request):
        payload = await request.json()
        data = datason.auto_deserialize(payload, config=API_CONFIG)
        result = run_model(data)
        return datason.serialize(result, config=API_CONFIG)


app = ModelDeployment.bind()

if __name__ == "__main__":
    print("Run with: serve run ray_serve_integration_guide:app")
