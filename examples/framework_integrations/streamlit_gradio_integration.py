#!/usr/bin/env python3
"""
Streamlit and Gradio Integration with DataSON

A comprehensive example showing how to integrate DataSON with Streamlit and Gradio
for interactive ML applications, data visualization, and AI-powered dashboards.

Features:
- Modern DataSON API (dump_api, load_smart, dump_ml)
- Interactive data processing with DataSON
- ML model serving through web interfaces
- Real-time data validation and transformation
- File upload/download with DataSON serialization
"""

import json
import time
from typing import Any, Dict, List, Union

# Framework availability checks
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

try:
    import gradio as gr

    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None

import numpy as np
import pandas as pd

import datason as ds
from datason.config import get_api_config

API_CONFIG = get_api_config()


class DataSONInteractiveDemo:
    """Interactive demo class for DataSON with UI frameworks."""

    def __init__(self):
        self.model_name = "interactive_ml_model"
        self.prediction_history = []
        self.processing_stats = {"total_requests": 0, "successful_parses": 0}

    def process_data_with_datason(self, data: Any, mode: str = "smart") -> Dict[str, Any]:
        """Process data using different DataSON modes."""
        self.processing_stats["total_requests"] += 1

        try:
            if mode == "basic":
                processed_data = ds.load_basic(data)
            elif mode == "smart":
                processed_data = ds.load_smart(data, config=API_CONFIG)
            elif mode == "api":
                processed_data = ds.dump_api(data)
            elif mode == "ml":
                processed_data = ds.dump_ml(data)
            elif mode == "secure":
                processed_data = ds.dump_secure(data)
            else:
                processed_data = data

            self.processing_stats["successful_parses"] += 1

            return {
                "status": "success",
                "processed_data": processed_data,
                "mode_used": mode,
                "processing_time": time.time(),
                "data_size": len(str(processed_data)),
            }

        except Exception as e:
            return {"status": "error", "error": str(e), "mode_used": mode, "original_data_type": type(data).__name__}

    def predict_with_metadata(self, features: Union[List, np.ndarray]) -> Dict[str, Any]:
        """Mock ML prediction with comprehensive metadata."""
        # Simulate prediction
        if isinstance(features, np.ndarray):
            features = features.tolist()

        prediction = np.random.choice([0, 1, 2])
        confidence = 0.6 + 0.4 * np.random.random()

        result = {
            "prediction": int(prediction),
            "confidence": float(confidence),
            "feature_summary": {
                "count": len(features),
                "mean": float(np.mean(features)),
                "std": float(np.std(features)),
                "min": float(np.min(features)),
                "max": float(np.max(features)),
            },
            "model_info": {"name": self.model_name, "version": "2.1.0", "timestamp": time.time()},
        }

        # Store in history
        self.prediction_history.append(result)
        if len(self.prediction_history) > 50:  # Keep last 50 predictions
            self.prediction_history.pop(0)

        return result

    def create_sample_datasets(self) -> Dict[str, Any]:
        """Create sample datasets for testing."""
        return {
            "simple_dict": {"name": "John Doe", "age": 30, "scores": [85, 92, 78, 96]},
            "ml_data": {
                "features": np.random.randn(10).tolist(),
                "metadata": {"experiment_id": "exp_001", "model_type": "classifier", "timestamp": time.time()},
            },
            "complex_nested": {
                "user_profile": {
                    "id": 12345,
                    "preferences": {"theme": "dark", "notifications": True},
                    "activity": {
                        "last_login": "2024-01-15T10:30:00Z",
                        "session_count": 42,
                        "favorite_features": ["data_viz", "ml_models", "dashboards"],
                    },
                },
                "data_points": [{"x": i, "y": np.sin(i / 10) + np.random.normal(0, 0.1)} for i in range(20)],
            },
            "pandas_dataframe": pd.DataFrame(
                {"A": range(5), "B": np.random.randn(5), "C": ["a", "b", "c", "d", "e"]}
            ).to_dict(),
            "mixed_types": {
                "text": "Hello DataSON!",
                "numbers": [1, 2.5, 3, 4.7],
                "boolean": True,
                "null_value": None,
                "datetime_string": "2024-01-15T12:00:00Z",
                "numpy_array": np.array([1, 2, 3, 4, 5]).tolist(),
            },
        }


# ================================
# STREAMLIT INTEGRATION
# ================================


def create_streamlit_app():
    """Create a comprehensive Streamlit app with DataSON integration."""
    if not STREAMLIT_AVAILABLE:
        print("❌ Streamlit not available. Install with: pip install streamlit")
        return

    st.set_page_config(
        page_title="DataSON + Streamlit Demo", page_icon="🚀", layout="wide", initial_sidebar_state="expanded"
    )

    st.title("🚀 DataSON + Streamlit Integration Demo")
    st.markdown("Comprehensive demonstration of DataSON features in Streamlit")

    # Initialize demo instance
    if "demo" not in st.session_state:
        st.session_state.demo = DataSONInteractiveDemo()

    demo = st.session_state.demo

    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    processing_mode = st.sidebar.selectbox(
        "DataSON Processing Mode",
        ["smart", "basic", "api", "ml", "secure"],
        index=0,
        help="Choose how DataSON should process your data",
    )

    show_metadata = st.sidebar.checkbox("Show Processing Metadata", value=True)
    auto_format = st.sidebar.checkbox("Auto-format JSON Output", value=True)

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🧪 Data Processing", "🤖 ML Predictions", "📁 File Operations", "📊 Analytics", "🔍 API Explorer"]
    )

    # Tab 1: Data Processing
    with tab1:
        st.header("🧪 Interactive Data Processing")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Data")

            # Sample data selector
            sample_datasets = demo.create_sample_datasets()
            selected_sample = st.selectbox(
                "Choose Sample Data", list(sample_datasets.keys()), format_func=lambda x: x.replace("_", " ").title()
            )

            if st.button("Load Sample Data", key="load_sample"):
                st.session_state.input_data = ds.dumps_json(sample_datasets[selected_sample], indent=2)

            # Text area for custom input
            input_data = st.text_area(
                "Or Enter Your Own JSON Data",
                value=st.session_state.get("input_data", '{"example": "data"}'),
                height=200,
                key="custom_input",
            )

            if st.button("Process Data", type="primary"):
                try:
                    # Parse input JSON
                    parsed_input = json.loads(input_data)

                    # Process with DataSON
                    result = demo.process_data_with_datason(parsed_input, processing_mode)
                    st.session_state.processing_result = result

                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
                except Exception as e:
                    st.error(f"Processing error: {e}")

        with col2:
            st.subheader("Processing Result")

            if "processing_result" in st.session_state:
                result = st.session_state.processing_result

                if result["status"] == "success":
                    st.success(f"✅ Processed with mode: {result['mode_used']}")

                    if show_metadata:
                        st.info(f"📊 Data size: {result['data_size']} characters")

                    # Display processed data
                    if auto_format:
                        formatted_output = ds.dumps_json(result["processed_data"], indent=2)
                        st.code(formatted_output, language="json")
                    else:
                        st.json(result["processed_data"])

                    # Download option
                    st.download_button(
                        label="📥 Download Result",
                        data=ds.dumps_json(result["processed_data"], indent=2),
                        file_name=f"datason_result_{int(time.time())}.json",
                        mime="application/json",
                    )

                else:
                    st.error(f"❌ Processing failed: {result['error']}")

    # Tab 2: ML Predictions
    with tab2:
        st.header("🤖 ML Model Predictions")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Feature Input")

            # Feature input methods
            input_method = st.radio("Input Method", ["Manual Entry", "Random Generation", "File Upload"])

            if input_method == "Manual Entry":
                feature_count = st.slider("Number of Features", 1, 20, 10)
                features = []

                for i in range(feature_count):
                    value = st.number_input(f"Feature {i + 1}", value=0.0, key=f"feature_{i}")
                    features.append(value)

            elif input_method == "Random Generation":
                feature_count = st.slider("Number of Features", 1, 50, 10)
                if st.button("Generate Random Features"):
                    features = np.random.randn(feature_count).tolist()
                    st.session_state.random_features = features

                features = st.session_state.get("random_features", [0.0] * feature_count)
                st.write("Generated Features:", features[:10])  # Show first 10

            else:  # File Upload
                uploaded_file = st.file_uploader("Upload Feature File (JSON/CSV)", type=["json", "csv"])
                features = [0.0] * 10  # Default

                if uploaded_file:
                    try:
                        if uploaded_file.type == "application/json":
                            content = uploaded_file.read().decode()
                            features = ds.load_smart(content, config=API_CONFIG)
                            if isinstance(features, dict) and "features" in features:
                                features = features["features"]
                        else:  # CSV
                            df = pd.read_csv(uploaded_file)
                            features = df.iloc[0].tolist()  # First row as features
                    except Exception as e:
                        st.error(f"File processing error: {e}")

            if st.button("🔮 Make Prediction", type="primary"):
                prediction_result = demo.predict_with_metadata(features)
                st.session_state.prediction_result = prediction_result

        with col2:
            st.subheader("Prediction Result")

            if "prediction_result" in st.session_state:
                result = st.session_state.prediction_result

                # Main prediction display
                st.metric(
                    label="Prediction", value=result["prediction"], delta=f"Confidence: {result['confidence']:.2%}"
                )

                # Feature statistics
                st.subheader("Feature Analysis")
                stats = result["feature_summary"]

                col_stats1, col_stats2 = st.columns(2)
                with col_stats1:
                    st.metric("Mean", f"{stats['mean']:.3f}")
                    st.metric("Std Dev", f"{stats['std']:.3f}")

                with col_stats2:
                    st.metric("Min", f"{stats['min']:.3f}")
                    st.metric("Max", f"{stats['max']:.3f}")

                # Download prediction
                prediction_data = ds.dump_api(result)
                st.download_button(
                    label="📥 Download Prediction",
                    data=ds.dumps_json(prediction_data, indent=2),
                    file_name=f"prediction_{int(time.time())}.json",
                    mime="application/json",
                )

    # Tab 3: File Operations
    with tab3:
        st.header("📁 File Operations with DataSON")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Upload & Process")

            uploaded_files = st.file_uploader(
                "Upload Data Files", type=["json", "jsonl", "csv"], accept_multiple_files=True
            )

            if uploaded_files:
                for uploaded_file in uploaded_files:
                    st.write(f"📄 Processing: {uploaded_file.name}")

                    try:
                        if uploaded_file.type == "application/json":
                            content = uploaded_file.read().decode()
                            processed = ds.load_smart(content, config=API_CONFIG)
                        else:
                            # Handle other formats
                            content = uploaded_file.read().decode()
                            processed = ds.loads(content)

                        st.success(f"✅ Processed {uploaded_file.name}")

                        # Show preview
                        if st.checkbox(f"Show preview of {uploaded_file.name}"):
                            st.json(processed)

                        # Store for download
                        processed_key = f"processed_{uploaded_file.name}"
                        st.session_state[processed_key] = processed

                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")

        with col2:
            st.subheader("Generate & Download")

            # Create sample files
            if st.button("Generate Sample ML Dataset"):
                ml_dataset = {
                    "metadata": {
                        "dataset_name": "sample_ml_data",
                        "created_at": time.time(),
                        "features": 20,
                        "samples": 100,
                    },
                    "data": {"X": np.random.randn(100, 20).tolist(), "y": np.random.randint(0, 3, 100).tolist()},
                    "feature_names": [f"feature_{i}" for i in range(20)],
                }

                serialized_dataset = ds.dump_ml(ml_dataset)

                st.download_button(
                    label="📥 Download ML Dataset",
                    data=ds.dumps_json(serialized_dataset, indent=2),
                    file_name="ml_dataset.json",
                    mime="application/json",
                )

    # Tab 4: Analytics
    with tab4:
        st.header("📊 Processing Analytics")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Usage Statistics")
            stats = demo.processing_stats

            st.metric("Total Requests", stats["total_requests"])
            st.metric("Successful Parses", stats["successful_parses"])

            if stats["total_requests"] > 0:
                success_rate = stats["successful_parses"] / stats["total_requests"]
                st.metric("Success Rate", f"{success_rate:.1%}")

        with col2:
            st.subheader("Prediction History")

            if demo.prediction_history:
                history_df = pd.DataFrame(
                    [
                        {
                            "prediction": p["prediction"],
                            "confidence": p["confidence"],
                            "feature_count": p["feature_summary"]["count"],
                            "timestamp": p["model_info"]["timestamp"],
                        }
                        for p in demo.prediction_history[-10:]  # Last 10
                    ]
                )

                st.dataframe(history_df)

                # Simple chart
                st.line_chart(history_df.set_index("timestamp")["confidence"])

    # Tab 5: API Explorer
    with tab5:
        st.header("🔍 DataSON API Explorer")

        st.subheader("Available Functions")

        api_functions = {
            "dump_api": "Clean JSON output for APIs",
            "dump_ml": "ML-optimized serialization",
            "dump_secure": "Security-focused with PII redaction",
            "load_smart": "Intelligent parsing with type detection",
            "load_basic": "Fast basic parsing",
            "dumps_json": "Standard JSON string output",
        }

        for func_name, description in api_functions.items():
            with st.expander(f"📚 {func_name}"):
                st.write(description)

                # Example usage
                if func_name == "dump_api":
                    example_data = {"user_id": 123, "name": "John", "active": True}
                    result = ds.dump_api(example_data)
                    st.code(f"ds.{func_name}({example_data}) = {result}")

    # Footer
    st.markdown("---")
    st.markdown("🚀 **DataSON + Streamlit Integration** - Powered by modern serialization")


# ================================
# GRADIO INTEGRATION
# ================================


def create_gradio_interface():
    """Create a comprehensive Gradio interface with DataSON integration."""
    if not GRADIO_AVAILABLE:
        print("❌ Gradio not available. Install with: pip install gradio")
        return None

    demo_instance = DataSONInteractiveDemo()

    def process_json_data(json_input: str, mode: str) -> tuple:
        """Process JSON data and return result with metadata."""
        try:
            parsed_data = json.loads(json_input)
            result = demo_instance.process_data_with_datason(parsed_data, mode)

            if result["status"] == "success":
                output_json = ds.dumps_json(result["processed_data"], indent=2)
                status = f"✅ Success with {result['mode_used']} mode"
                metadata = f"Data size: {result['data_size']} chars"
            else:
                output_json = f"Error: {result['error']}"
                status = "❌ Processing failed"
                metadata = f"Original type: {result.get('original_data_type', 'unknown')}"

            return output_json, status, metadata

        except json.JSONDecodeError as e:
            return f"JSON Parse Error: {e}", "❌ Invalid JSON", "Fix JSON syntax"
        except Exception as e:
            return f"Error: {e}", "❌ Unexpected error", str(e)

    def predict_from_features(features_text: str) -> tuple:
        """Make ML prediction from comma-separated features."""
        try:
            # Parse features
            features = [float(x.strip()) for x in features_text.split(",")]

            # Make prediction
            result = demo_instance.predict_with_metadata(features)

            # Format output
            prediction_text = f"Prediction: {result['prediction']}"
            confidence_text = f"Confidence: {result['confidence']:.2%}"
            details = ds.dumps_json(result, indent=2)

            return prediction_text, confidence_text, details

        except ValueError as e:
            return f"Error: {e}", "Invalid input format", "Use comma-separated numbers"
        except Exception as e:
            return f"Error: {e}", "Processing failed", str(e)

    def batch_process_file(file) -> str:
        """Process uploaded file with DataSON."""
        if file is None:
            return "No file uploaded"

        try:
            # Read file content
            if file.name.endswith(".json"):
                content = file.read().decode("utf-8")
                data = json.loads(content)
                processed = ds.load_smart(data, config=API_CONFIG)
            else:
                return "Unsupported file type. Please upload JSON files."

            # Return processed data
            return ds.dumps_json(processed, indent=2)

        except Exception as e:
            return f"File processing error: {e}"

    # Create Gradio interface with multiple tabs
    with gr.Blocks(
        title="DataSON + Gradio Demo",
        theme=gr.themes.Soft(),
        css=".gradio-container {background: linear-gradient(45deg, #f0f0f0, #ffffff);}",
    ) as interface:
        gr.Markdown("# 🚀 DataSON + Gradio Integration Demo")
        gr.Markdown("Comprehensive demonstration of DataSON features in Gradio")

        with gr.Tabs():
            # Tab 1: Data Processing
            with gr.TabItem("🧪 Data Processing"):
                gr.Markdown("## Interactive JSON Data Processing")

                with gr.Row():
                    with gr.Column():
                        json_input = gr.Textbox(
                            label="Input JSON Data",
                            placeholder='{"example": "data", "numbers": [1, 2, 3]}',
                            lines=8,
                            value='{"name": "DataSON", "version": "2.0", "features": ["fast", "smart", "secure"]}',
                        )

                        processing_mode = gr.Dropdown(
                            choices=["smart", "basic", "api", "ml", "secure"], value="smart", label="Processing Mode"
                        )

                        process_btn = gr.Button("🔄 Process Data", variant="primary")

                    with gr.Column():
                        processed_output = gr.Textbox(label="Processed Output", lines=8, interactive=False)

                        status_output = gr.Textbox(label="Status", interactive=False)

                        metadata_output = gr.Textbox(label="Metadata", interactive=False)

                process_btn.click(
                    fn=process_json_data,
                    inputs=[json_input, processing_mode],
                    outputs=[processed_output, status_output, metadata_output],
                )

            # Tab 2: ML Predictions
            with gr.TabItem("🤖 ML Predictions"):
                gr.Markdown("## Machine Learning Predictions")

                with gr.Row():
                    with gr.Column():
                        features_input = gr.Textbox(
                            label="Features (comma-separated)",
                            placeholder="1.2, 3.4, 5.6, 7.8, 9.0",
                            value="1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0",
                        )

                        predict_btn = gr.Button("🔮 Make Prediction", variant="primary")

                        # Quick sample buttons
                        gr.Markdown("### Quick Samples:")
                        samples = ["1,2,3,4,5", "0.1,0.5,0.9,1.2,1.8", "-1,0,1,2,3,4,5"]

                        for i, sample in enumerate(samples):
                            btn = gr.Button(f"Sample {i + 1}")
                            btn.click(lambda s=sample: s, outputs=features_input)

                    with gr.Column():
                        prediction_output = gr.Textbox(label="Prediction", interactive=False)

                        confidence_output = gr.Textbox(label="Confidence", interactive=False)

                        details_output = gr.JSON(label="Detailed Results")

                predict_btn.click(
                    fn=predict_from_features,
                    inputs=[features_input],
                    outputs=[prediction_output, confidence_output, details_output],
                )

            # Tab 3: File Processing
            with gr.TabItem("📁 File Processing"):
                gr.Markdown("## File Upload and Processing")

                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(label="Upload JSON File", file_types=[".json"])

                        process_file_btn = gr.Button("📄 Process File", variant="primary")

                    with gr.Column():
                        file_output = gr.Textbox(label="Processed File Content", lines=15, interactive=False)

                process_file_btn.click(fn=batch_process_file, inputs=[file_input], outputs=[file_output])

            # Tab 4: API Documentation
            with gr.TabItem("📚 API Reference"):
                gr.Markdown("## DataSON API Functions")

                api_docs = """
                ### Core Functions

                - **`dump_api(obj)`** - Clean JSON output for web APIs
                - **`dump_ml(obj)`** - ML-optimized serialization
                - **`dump_secure(obj)`** - Security-focused with PII redaction
                - **`load_smart(data)`** - Intelligent parsing with type detection
                - **`load_basic(data)`** - Fast basic parsing
                - **`dumps_json(obj)`** - Standard JSON string output

                ### Configuration

                - **`get_api_config()`** - Get API-friendly configuration
                - **`get_ml_config()`** - Get ML-optimized configuration

                ### Features

                ✅ **Smart Type Detection** - Automatically detects and converts data types
                ✅ **UUID Handling** - Converts UUIDs to strings for API compatibility
                ✅ **Security Features** - Built-in PII redaction and data sanitization
                ✅ **ML Optimization** - Optimized for machine learning workflows
                ✅ **Cross-Version Compatibility** - Works across Python 3.8-3.11+
                """

                gr.Markdown(api_docs)

        gr.Markdown("---")
        gr.Markdown("🚀 **DataSON Integration** - Modern serialization for interactive applications")

    return interface


# ================================
# MAIN DEMO RUNNER
# ================================


def run_ui_frameworks_demo():
    """Run comprehensive UI frameworks demonstration."""
    print("🚀 DataSON + UI Frameworks Integration Demo")
    print("=" * 60)

    # Check framework availability
    streamlit_status = "✅ Available" if STREAMLIT_AVAILABLE else "❌ Not installed"
    gradio_status = "✅ Available" if GRADIO_AVAILABLE else "❌ Not installed"

    print("\n📦 Framework Status:")
    print(f"  - Streamlit: {streamlit_status}")
    print(f"  - Gradio: {gradio_status}")

    if not (STREAMLIT_AVAILABLE or GRADIO_AVAILABLE):
        print("\n💡 Install UI frameworks:")
        print("  pip install streamlit gradio")

        # Show DataSON features without UI frameworks
        print("\n🔄 Demonstrating DataSON features:")
        demo = DataSONInteractiveDemo()

        sample_data = {"test": "data", "values": [1, 2, 3]}

        for mode in ["smart", "api", "ml", "secure"]:
            result = demo.process_data_with_datason(sample_data, mode)
            print(f"  {mode.upper()}: {result['status']}")

        return {"status": "demo_completed_without_ui"}

    print("\n🎯 DataSON Features for Interactive Apps:")
    print("  - 🌐 dump_api() for clean web interface data")
    print("  - 🧠 load_smart() for user input processing")
    print("  - 🔒 dump_secure() for safe data handling")
    print("  - 📊 Real-time data validation and transformation")
    print("  - 📁 File upload/download with DataSON serialization")

    print("\n🚀 Launch Instructions:")

    if STREAMLIT_AVAILABLE:
        print("\n📊 **Streamlit App:**")
        print(f"  streamlit run {__file__}")
        print("  Features: Multi-tab interface, file operations, ML predictions")

    if GRADIO_AVAILABLE:
        print("\n🎨 **Gradio Interface:**")
        print(f"  python {__file__} --gradio")
        print("  Features: Interactive processing, batch operations, API docs")

    return {"streamlit_available": STREAMLIT_AVAILABLE, "gradio_available": GRADIO_AVAILABLE, "demo_ready": True}


if __name__ == "__main__":
    import sys

    if "--gradio" in sys.argv and GRADIO_AVAILABLE:
        # Launch Gradio interface
        print("🚀 Launching Gradio interface...")
        interface = create_gradio_interface()
        if interface:
            interface.launch(server_name="127.0.0.1", server_port=7860, share=False, debug=True)

    elif "--streamlit" in sys.argv or (STREAMLIT_AVAILABLE and not any(arg.startswith("--") for arg in sys.argv[1:])):
        # This is called when running with streamlit
        create_streamlit_app()

    else:
        # Run the demo overview
        results = run_ui_frameworks_demo()
        print(f"\n🎯 Demo Results: {results}")

        print("\n💡 Usage Examples:")
        print(f"  python {__file__}                    # Show overview")
        print(f"  python {__file__} --gradio           # Launch Gradio interface")
        print(f"  streamlit run {__file__}             # Launch Streamlit app")

        if GRADIO_AVAILABLE and not any("--" in arg for arg in sys.argv[1:]):
            print("\n🎨 Auto-launching Gradio demo...")
            interface = create_gradio_interface()
            if interface:
                interface.launch(share=False, debug=False)
