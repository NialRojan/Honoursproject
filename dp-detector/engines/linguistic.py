import re
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from engines.base import Finding

MODEL_DIR = Path(__file__).parent.parent / "models" / "linguistic_bert"
_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        _model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        _model.eval()


def _split_blocks(text: str) -> list:
    blocks = [b.strip() for b in re.split(r"[\n\.\!\?]", text) if b.strip()]
    return [b for b in blocks if len(b) > 3]


def analyze(capture) -> list:
    _load()
    findings = []
    for block in _split_blocks(capture.rendered_text):
        inputs = _tokenizer(block, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        dark_prob = probs[1].item()
        if dark_prob > 0.6:
            findings.append(Finding(
                engine="linguistic", pattern_type="manipulative_text", confidence=round(dark_prob, 2),
                evidence=f"classified as manipulative: \"{block[:80]}\"",
            ))
    return findings
