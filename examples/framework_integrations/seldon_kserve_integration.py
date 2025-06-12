#!/usr/bin/env python3
"""Seldon Core / KServe example using datason."""

from seldon_core.user_model import SeldonComponent

import datason
from datason.config import get_api_config

API_CONFIG = get_api_config()


# Dummy model function
def run_model(payload: dict) -> dict:
    return {"prediction": 123, "input": payload}


class Model(SeldonComponent):
    def predict(self, features, **kwargs):
        data = datason.auto_deserialize(features, config=API_CONFIG)
        output = run_model(data)
        return datason.serialize(output, config=API_CONFIG)


if __name__ == "__main__":
    print("Use this class in your Seldon deployment")
