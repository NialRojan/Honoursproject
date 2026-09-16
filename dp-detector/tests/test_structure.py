from pathlib import Path
from capture.capture import capture
from engines.structure import analyze

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

def _types(findings):
    return {f.pattern_type for f in findings}

def test_detects_preselected_consent():
    cap = capture(str(FIXTURES_DIR / "precheck_consent.html"), is_fixture=True)
    assert "preselected_consent" in _types(analyze(cap))

def test_detects_hidden_cancel_link():
    cap = capture(str(FIXTURES_DIR / "hidden_cancel.html"), is_fixture=True)
    assert "hidden_cancel_link" in _types(analyze(cap))

def test_detects_forced_continuity_pricing():
    cap = capture(str(FIXTURES_DIR / "forced_continuity.html"), is_fixture=True)
    assert "forced_continuity_pricing" in _types(analyze(cap))

def test_no_false_positive_on_easy_cancel_flow():
    cap = capture(str(FIXTURES_DIR / "easy_cancel_flow.html"), is_fixture=True)
    assert _types(analyze(cap)) == set()
