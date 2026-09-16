from pathlib import Path
import pytest
from capture.capture import capture
from engines.linguistic import analyze

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
MODEL_DIR = Path(__file__).parent.parent / "models" / "linguistic_bert"

pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason="run training/train_linguistic.py first (Task 7)")


def test_detects_confirm_shaming_text():
    cap = capture(str(FIXTURES_DIR / "confirm_shaming.html"), is_fixture=True)
    findings = analyze(cap)
    assert any(f.pattern_type == "manipulative_text" for f in findings)
