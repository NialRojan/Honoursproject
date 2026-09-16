from pathlib import Path
from capture.capture import capture

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

def test_capture_fixture_extracts_dom_and_screenshot():
    result = capture(str(FIXTURES_DIR / "precheck_consent.html"), is_fixture=True)
    assert "marketing-optin" in result.dom_html
    assert len(result.screenshot_png) > 0
    assert "Create your account" in result.rendered_text
    assert len(result.elements) > 0
    checkbox = next(e for e in result.elements if e.attrs.get("type") == "checkbox")
    assert checkbox.attrs.get("checked") in ("true", "")

def test_capture_click_trace_on_deep_cancel_flow():
    result = capture(str(FIXTURES_DIR / "deep_cancel_flow.html"), is_fixture=True)
    assert len(result.click_trace) >= 3
    assert "final confirmation" in result.click_trace[-1].visible_text_after.lower()

def test_capture_click_trace_on_easy_cancel_flow():
    result = capture(str(FIXTURES_DIR / "easy_cancel_flow.html"), is_fixture=True)
    assert len(result.click_trace) <= 1
