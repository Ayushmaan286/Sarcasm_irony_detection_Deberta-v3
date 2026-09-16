"""
Context-Aware Sarcasm and Irony Detection - Streamlit Application

This app is a thin UI layer only. All model architecture, checkpoint
loading, preprocessing, and prediction logic live in inference.py and
are not duplicated or modified here.
"""

import os
import importlib.util
import streamlit as st

INFERENCE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "inference",
    "inference.py",
)

@st.cache_resource
def load_inference_module():
    spec = importlib.util.spec_from_file_location("inference", INFERENCE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    inference = load_inference_module()
    load_error = None
except Exception as e:
    inference = None
    load_error = str(e)


st.set_page_config(
    page_title="Context-Aware Sarcasm and Irony Detection",
    layout="centered",
)

st.title("Context-Aware Sarcasm and Irony Detection")
st.caption("Joint Sarcasm and Irony Classification using DeBERTa-v3")

if load_error is not None:
    st.error(f"Failed to load the inference pipeline: {load_error}")
    st.stop()

st.divider()


st.subheader("Context")
st.caption("Optional. Provide the preceding conversational context, if any.")
context_input = st.text_area(
    label="Context",
    label_visibility="collapsed",
    height=100,
    placeholder="e.g. Someone says they had a terrible day.",
)

st.subheader("Current Text")
st.caption("Required. The utterance to classify.")
text_input = st.text_area(
    label="Current Text",
    label_visibility="collapsed",
    height=100,
    placeholder="e.g. What a wonderful day!",
)

predict_clicked = st.button("Predict", type="primary")


def render_result(task_result):
    label = task_result["label"]
    confidence = task_result["confidence"]
    probabilities = task_result["probabilities"]

    st.markdown(f"**{label}**")
    st.write(f"Confidence: {confidence * 100:.1f}%")
    for class_label, prob in probabilities.items():
        st.write(f"{class_label}: {prob * 100:.1f}%")


if predict_clicked:
    if not text_input or not text_input.strip():
        st.error("Current text is required. Please enter a value before predicting.")
    else:
        try:
            result = inference.predict(context_input, text_input)

            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Sarcasm Detection")
                render_result(result["sarcasm"])

            with col2:
                st.subheader("Irony Detection")
                render_result(result["irony"])

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
