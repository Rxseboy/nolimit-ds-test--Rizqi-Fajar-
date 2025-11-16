"""
Inference utilities for Tweet Sentiment Classification Project.
Handles prediction and probability outputs.
"""


import torch
import numpy as np
from typing import Dict
from transformers import AutoModelForSequenceClassification


from .config import model_cfg, paths
from .preprocessing import tokenizer


# Label mapping (same logic as the model)
ID2LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


# Cached model instance (lazy load)
_cached_model = None


def load_inference_model():
    """Load the trained model only once (lazy loading)."""
    global _cached_model

    if _cached_model is None:
        _cached_model = AutoModelForSequenceClassification.from_pretrained(
            paths.TRAINED_MODEL_DIR,
            num_labels=model_cfg.NUM_LABELS,
        )
        _cached_model.to(model_cfg.DEVICE)
        _cached_model.eval()

    return _cached_model


# -------------------
# Main Prediction API
# -------------------


def predict(text: str) -> Dict:
    """
    Perform sentiment prediction on a single text.
    Returns:
    { "label": <sentiment>, "probabilities": {...} }
    """
    model = load_inference_model()

    encoded = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=model_cfg.MAX_LENGTH,
    )

    # Move all tensors to the correct device
    encoded = {k: v.to(model_cfg.DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        out = model(**encoded)
        logits = out.logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    pred_id = int(np.argmax(probs))

    return {
        "label": ID2LABEL[pred_id],
        "probabilities": {
            "Negative": float(probs[0]),
            "Neutral": float(probs[1]),
            "Positive": float(probs[2]),
        },
    }