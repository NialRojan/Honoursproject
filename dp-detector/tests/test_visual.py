from pathlib import Path

from capture.capture import capture
from engines.visual import analyze

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

def test_detects_preselection_on_precheck_consent():
    cap = capture(str(FIXTURES_DIR / "precheck_consent.html"), is_fixture=True)
    findings = analyze(cap)
    assert any(f.pattern_type == "preselection" for f in findings)

def test_no_crash_on_clean_fixture():
    cap = capture(str(FIXTURES_DIR / "easy_cancel_flow.html"), is_fixture=True)
    findings = analyze(cap)
    assert isinstance(findings, list)

def test_detects_false_hierarchy_on_disguised_button():
    cap = capture(str(FIXTURES_DIR / "disguised_button.html"), is_fixture=True)
    findings = analyze(cap)
    assert any(f.pattern_type == "false_hierarchy" for f in findings)
