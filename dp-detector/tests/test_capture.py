import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from capture.capture import EXTRACT_ELEMENTS_JS, capture

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

class _BadGatewayHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(502)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<html><body><h1>502 Bad Gateway</h1></body></html>")

    def log_message(self, format, *args):
        pass

def test_extract_elements_handles_unparseable_computed_opacity():
    # getComputedStyle(el).opacity returned "" for some elements on real sites
    # (observed in production), making parseFloat(...) evaluate to NaN. NaN
    # then crashes /analyze's JSON response (Starlette's JSONResponse uses
    # allow_nan=False). Force the exact browser condition via a Proxy over
    # getComputedStyle rather than relying on finding the triggering site.
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content("<html><body><div id='x'>hi</div></body></html>")
            page.evaluate("""
                () => {
                    const original = window.getComputedStyle;
                    window.getComputedStyle = (el, ...rest) => {
                        const style = original(el, ...rest);
                        return new Proxy(style, {
                            get(target, prop) {
                                if (prop === 'opacity') return '';
                                const v = target[prop];
                                return typeof v === 'function' ? v.bind(target) : v;
                            }
                        });
                    };
                }
            """)
            raw = page.evaluate(EXTRACT_ELEMENTS_JS)
        finally:
            browser.close()

    assert all(not (isinstance(r["opacity"], float) and math.isnan(r["opacity"])) for r in raw)

def test_extract_elements_reflects_live_checked_state_not_just_attribute():
    # Real sites (React/Vue/etc.) set checkbox state via the `.checked` DOM
    # property, which does not update the reflected `checked` HTML attribute.
    # Structure/Visual's preselected-consent detection keys off `"checked" in
    # el.attrs`, so a JS-checked box was invisible to both engines on any
    # real, JS-rendered page even though it's visibly checked on screen.
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content("""
                <html><body>
                  <input type="checkbox" id="js-checked">
                  <input type="checkbox" id="js-unchecked" checked>
                </body></html>
            """)
            page.evaluate("""
                () => {
                    document.getElementById('js-checked').checked = true;
                    document.getElementById('js-unchecked').checked = false;
                }
            """)
            raw = page.evaluate(EXTRACT_ELEMENTS_JS)
        finally:
            browser.close()

    by_id = {r["attrs"].get("id"): r for r in raw if r["tag"] == "input"}
    assert "checked" in by_id["js-checked"]["attrs"]
    assert "checked" not in by_id["js-unchecked"]["attrs"]

def test_capture_raises_on_non_2xx_http_status():
    server = HTTPServer(("127.0.0.1", 0), _BadGatewayHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with pytest.raises(RuntimeError, match="502"):
            capture(f"http://127.0.0.1:{port}/", is_fixture=False)
    finally:
        server.shutdown()
        thread.join()
