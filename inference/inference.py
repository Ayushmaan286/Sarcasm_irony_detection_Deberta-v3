"""
Standalone inference pipeline for the Context-Aware Joint Sarcasm
and Irony Detection model (DeBERTa-v3-base).

This module is the prediction engine only. It contains no training,
evaluation-suite, or UI code. It is intended to be imported by the
Streamlit application in a later stage.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from pathlib import Path


# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

MODEL_NAME = "microsoft/deberta-v3-base"
PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_DIR / "models" / "best_checkpoint.pt"
MAX_LENGTH = 256

SARCASM_LABELS = {0: "Not Sarcastic", 1: "Sarcastic"}
IRONY_LABELS = {0: "Not Ironic", 1: "Ironic"}


# --------------------------------------------------------------------
# Device
# --------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------
# Model architecture (must match training exactly)
# --------------------------------------------------------------------

class SarcasmIronyModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.sarcasm_head = nn.Linear(hidden_size, 2)
        self.irony_head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        sarcasm_logits = self.sarcasm_head(pooled_output)
        irony_logits = self.irony_head(pooled_output)

        return sarcasm_logits, irony_logits


# --------------------------------------------------------------------
# Tokenizer + model loading
# --------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = SarcasmIronyModel()
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()


# --------------------------------------------------------------------
# Input construction
# --------------------------------------------------------------------

def build_model_input(context, text):
    """
    Constructs the exact baseline model input format:

    [CONTEXT]
    {context}

    [CURRENT]
    {text}
    """
    if context is None or not context.strip():
        context = "No context available."
    else:
        context = context.strip()

    return f"[CONTEXT]\n{context}\n\n[CURRENT]\n{text}"


def _run_inference(model_input):
    """Tokenizes model_input and runs a forward pass. Returns raw logits."""
    encoded = tokenizer(
        model_input,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        sarcasm_logits, irony_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    return sarcasm_logits, irony_logits


# --------------------------------------------------------------------
# Public prediction function
# --------------------------------------------------------------------

def predict(context, text):
    """
    Runs standalone inference for sarcasm and irony detection.

    Args:
        context (str or None): conversational context. If None/empty,
            defaults to "No context available."
        text (str): the current utterance. Required.

    Returns:
        dict: structured prediction result for sarcasm and irony.
    """
    if text is None or not text.strip():
        raise ValueError("`text` (the current utterance) is required and cannot be empty.")

    model_input = build_model_input(context, text)
    sarcasm_logits, irony_logits = _run_inference(model_input)

    sarcasm_probs = torch.softmax(sarcasm_logits, dim=-1).squeeze(0)
    irony_probs = torch.softmax(irony_logits, dim=-1).squeeze(0)

    sarcasm_pred = torch.argmax(sarcasm_probs).item()
    irony_pred = torch.argmax(irony_probs).item()

    return {
        "sarcasm": {
            "label": SARCASM_LABELS[sarcasm_pred],
            "confidence": float(sarcasm_probs[sarcasm_pred].item()),
            "probabilities": {
                SARCASM_LABELS[0]: float(sarcasm_probs[0].item()),
                SARCASM_LABELS[1]: float(sarcasm_probs[1].item()),
            },
        },
        "irony": {
            "label": IRONY_LABELS[irony_pred],
            "confidence": float(irony_probs[irony_pred].item()),
            "probabilities": {
                IRONY_LABELS[0]: float(irony_probs[0].item()),
                IRONY_LABELS[1]: float(irony_probs[1].item()),
            },
        },
    }
