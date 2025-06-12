#!/usr/bin/env python3
"""Streamlit and Gradio example using datason for safe JSON."""

import datason
from datason.config import get_api_config

API_CONFIG = get_api_config()


# Dummy model function
def run_model(payload: dict) -> dict:
    return {"result": 42, "input": payload}


# Streamlit demo
try:
    import streamlit as st
except ImportError:
    st = None

if st is not None:
    uploaded = st.file_uploader("Upload JSON")
    if uploaded:
        raw = uploaded.read().decode()
        data = datason.loads(raw, config=API_CONFIG)
        output = run_model(data)
        st.json(datason.dumps(output, config=API_CONFIG))

# Gradio demo
try:
    import gradio as gr
except ImportError:
    gr = None


def gradio_predict(data: dict) -> dict:
    parsed = datason.auto_deserialize(data, config=API_CONFIG)
    result = run_model(parsed)
    return datason.serialize(result, config=API_CONFIG)


if gr is not None:
    demo = gr.Interface(fn=gradio_predict, inputs="json", outputs="json")
    if __name__ == "__main__":
        demo.launch()
elif __name__ == "__main__":
    print("Install streamlit or gradio to run this demo")
