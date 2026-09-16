from pathlib import Path
import pytest
from capture.capture import capture
from engines.behavior import analyze

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
MODEL_DIR = Path(__file__).parent.parent / "models" / "behavior_bert"

pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason="run training/train_behavior.py first (Task 10)")


def test_flags_deep_cancel_flow():
    cap = capture(str(FIXTURES_DIR / "deep_cancel_flow.html"), is_fixture=True)
    findings = analyze(cap)
    assert any(f.pattern_type == "roach_motel" for f in findings)


def test_does_not_flag_easy_cancel_flow():
    cap = capture(str(FIXTURES_DIR / "easy_cancel_flow.html"), is_fixture=True)
    findings = analyze(cap)
    assert not any(f.pattern_type == "roach_motel" for f in findings)
