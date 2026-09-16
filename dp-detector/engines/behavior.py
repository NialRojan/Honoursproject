from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from engines.base import Finding

MODEL_DIR = Path(__file__).parent.parent / "models" / "behavior_bert"
DEPTH_THRESHOLD = 2
_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        if not MODEL_DIR.exists():
            raise RuntimeError(
                "Behavior engine model not trained yet — run training/train_behavior.py (Task 10) first"
            )
        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        _model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        _model.eval()


def _classify(text: str) -> float:
    inputs = _tokenizer(text[:512], return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return probs[1].item()


def analyze(capture) -> list:
    depth = len(capture.click_trace)
    if depth <= DEPTH_THRESHOLD:
        return []

    _load()
    last_step = capture.click_trace[-1]
    dark_prob = _classify(last_step.visible_text_after)
    depth_factor = min(1.0, depth / (DEPTH_THRESHOLD + 2))
    confidence = round(dark_prob * depth_factor, 2)

    if dark_prob > 0.5:
        return [Finding(
            engine="behavior", pattern_type="roach_motel", confidence=confidence,
            evidence=f"cancellation required {depth} clicks; final step text classified as manipulative "
                     f"(p={dark_prob:.2f})",
        )]
    return []
