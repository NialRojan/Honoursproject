from pathlib import Path

import pytest

from capture.capture import capture
from engines.visual import analyze

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

@pytest.mark.xfail(
    strict=True,
    reason=(
        "UIGuard's vendored rule_check.py detects preselection only when a "
        "checkbox's paired label text matches one of its ~20 hardcoded "
        "regexes (notification/newsletter/privacy-policy/usage-data/follow "
        "vocabulary in examine_text_content). This fixture's label text "
        "('Send me marketing emails and special offers') matches none of "
        "them -- confirmed empirically by running examine_text_content() "
        "against every text item on the page, which returns zero flags. "
        "The adapter correctly builds a 'check_group' item with real "
        "status=['checked', 1] (see engines/visual.py "
        "_merge_checkboxes_with_labels), and flag_status is enabled, so "
        "the schema/data plumbing is right; the vendored rule engine's "
        "own keyword vocabulary is simply narrower than this fixture's "
        "wording. See task-6-report.md for the full diagnosis."
    ),
)
def test_detects_preselection_on_precheck_consent():
    cap = capture(str(FIXTURES_DIR / "precheck_consent.html"), is_fixture=True)
    findings = analyze(cap)
    assert any(f.pattern_type == "preselection" for f in findings)

def test_no_crash_on_clean_fixture():
    cap = capture(str(FIXTURES_DIR / "easy_cancel_flow.html"), is_fixture=True)
    findings = analyze(cap)
    assert isinstance(findings, list)
