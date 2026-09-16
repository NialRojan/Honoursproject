# Phase 1 Multimodal Dark Pattern Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A locally-run tool that, given a URL or a saved fixture page, captures it with a headless browser, runs it through four dark-pattern-detection engines (Structure, Visual, Linguistic, Behavior), fuses the findings into a 0-100 risk score, and shows the result in a web frontend with the flagged elements highlighted on the screenshot.

**Architecture:** Single Python (FastAPI) process. A Playwright-based capture module produces one `Capture` object (DOM, screenshot, text, elements, click-trace) from either a live URL or a local fixture file, using the same code path for both. Four engine modules each implement `analyze(capture) -> list[Finding]` against a shared contract; a deterministic fusion step combines their output into a risk score. A static HTML/JS frontend renders the result.

**Tech Stack:** Python 3.14, FastAPI, Playwright (sync API), PyTorch 2.14 + transformers 5.17 (verified installable on Python 3.14 — no pinned virtualenv needed), BeautifulSoup4, opencv-python-headless + scikit-learn (UIGuard's vendored dependencies), easyocr (pure-pip OCR, no system package install required), pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-multimodal-dark-pattern-engine-design.md`

## Global Constraints

- All engine modules expose exactly one function: `analyze(capture: Capture) -> list[Finding]`, using the `Finding` and `Capture` dataclasses defined in Task 2 and Task 4 — no task may redefine these shapes.
- Python 3.14 throughout; confirmed via `pip install --dry-run` that playwright 1.63.0, torch 2.14.0, and transformers 5.17.0 all resolve cleanly. No pinned virtualenv fallback is needed.
- No live-network calls in the automated `pytest` suite. Live-URL mode is exercised manually only (Task 15).
- Every vendored/derived piece of third-party code gets an attribution entry in `vendor/NOTICE` (MIT: Pattern Highlighter, AppRay; Apache-2.0: UIGuard, ec-darkpattern).
- AppRay's LLM-guided exploration is explicitly **not** used — Behavior engine uses AppRay's trained classifier architecture only, fed by our own scripted (non-LLM) click-trace capture, per the confirmed scope decision.
- UIGuard is used via its `rule_check.py` decision logic only, with `flag_icon = flag_TM = flag_status = flag_grouping = False`, fed by our own Playwright-based element/bbox/text extraction — **not** its Faster R-CNN + PaddleOCR pipeline, per the confirmed scope decision.
- All work happens inside a `dp-detector/` subdirectory of this repo; commit after every task.

---

## File Structure

```
dp-detector/
├── requirements.txt
├── .gitignore
├── README.md
├── app.py
├── capture/
│   ├── __init__.py
│   └── capture.py
├── engines/
│   ├── __init__.py
│   ├── base.py
│   ├── structure.py
│   ├── visual.py
│   ├── linguistic.py
│   └── behavior.py
├── fusion.py
├── vendor/
│   ├── NOTICE
│   └── uiguard/
│       ├── __init__.py
│       ├── rule_check.py     # vendored, unmodified, from UIST23-UIGuard
│       ├── utils.py            # vendored, unmodified
│       └── grouping.py          # vendored, unmodified (from DBSCAN/grouping.py)
├── training/
│   ├── bert_finetune.py
│   ├── train_linguistic.py
│   ├── prepare_behavior_dataset.py
│   ├── train_behavior.py
│   └── data/                    # downloaded/generated datasets (gitignored)
├── models/                       # trained checkpoints (gitignored)
│   ├── linguistic_bert/
│   └── behavior_bert/
├── fixtures/
│   ├── precheck_consent.html
│   ├── hidden_cancel.html
│   ├── forced_continuity.html
│   ├── confirm_shaming.html
│   ├── disguised_button.html
│   ├── deep_cancel_flow.html
│   └── easy_cancel_flow.html
├── static/
│   ├── index.html
│   ├── app.js
│   └── style.css
└── tests/
    ├── test_health.py
    ├── test_base.py
    ├── test_fixtures.py
    ├── test_capture.py
    ├── test_structure.py
    ├── test_visual.py
    ├── test_linguistic.py
    ├── test_behavior.py
    ├── test_fusion.py
    └── test_integration.py
```

---

### Task 1: Project scaffolding

**Files:**
- Create: `dp-detector/requirements.txt`
- Create: `dp-detector/.gitignore`
- Create: `dp-detector/app.py`
- Test: `dp-detector/tests/test_health.py`

**Interfaces:**
- Produces: a runnable `app.py` with a FastAPI instance named `app`, and `GET /health` returning `{"status": "ok"}`.

- [ ] **Step 1: Write the failing test**

```python
# dp-detector/tests/test_health.py
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 2: Create the virtual environment and dependency list**

```bash
cd "dp-detector"
python3 -m venv .venv
source .venv/bin/activate
```

```
# dp-detector/requirements.txt
fastapi==0.115.6
uvicorn[standard]==0.34.0
httpx==0.28.1
playwright==1.63.0
beautifulsoup4==4.12.3
torch==2.14.0
transformers==5.17.0
pandas==2.2.3
scikit-learn==1.5.2
opencv-python-headless==4.10.0.84
matplotlib==3.9.2
easyocr==1.7.2
pillow==11.0.0
requests==2.32.3
pytest==8.3.4
```

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

```
# dp-detector/.gitignore
.venv/
__pycache__/
*.pyc
training/data/
models/
.pytest_cache/
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd dp-detector && pytest tests/test_health.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app'` or import error, since `app.py` doesn't exist yet.

- [ ] **Step 4: Write minimal implementation**

```python
# dp-detector/app.py
from fastapi import FastAPI

app = FastAPI(title="Dark Pattern Detection Engine")


@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd dp-detector && pytest tests/test_health.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add dp-detector/requirements.txt dp-detector/.gitignore dp-detector/app.py dp-detector/tests/test_health.py
git commit -m "feat: scaffold FastAPI app with health endpoint"
```

---

### Task 2: Engine base interfaces

**Files:**
- Create: `dp-detector/engines/__init__.py`
- Create: `dp-detector/engines/base.py`
- Test: `dp-detector/tests/test_base.py`

**Interfaces:**
- Produces: `Finding` dataclass — fields `engine: str`, `pattern_type: str`, `confidence: float`, `evidence: str`, `element_ref: str | None = None`. Every later engine task returns `list[Finding]` using this exact shape.

- [ ] **Step 1: Write the failing test**

```python
# dp-detector/tests/test_base.py
from engines.base import Finding

def test_finding_defaults():
    f = Finding(engine="structure", pattern_type="preselected_consent", confidence=0.9, evidence="checkbox pre-checked")
    assert f.element_ref is None
    assert f.confidence == 0.9

def test_finding_with_element_ref():
    f = Finding(engine="visual", pattern_type="disguised_ad", confidence=0.7, evidence="ad styled as content", element_ref="#ad-1")
    assert f.element_ref == "#ad-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dp-detector && pytest tests/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engines'`

- [ ] **Step 3: Write minimal implementation**

```python
# dp-detector/engines/__init__.py
```

```python
# dp-detector/engines/base.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class Finding:
    engine: str
    pattern_type: str
    confidence: float
    evidence: str
    element_ref: Optional[str] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dp-detector && pytest tests/test_base.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dp-detector/engines/__init__.py dp-detector/engines/base.py dp-detector/tests/test_base.py
git commit -m "feat: add Finding dataclass shared engine contract"
```

---

### Task 3: Fixture pages

**Files:**
- Create: `dp-detector/fixtures/precheck_consent.html`
- Create: `dp-detector/fixtures/hidden_cancel.html`
- Create: `dp-detector/fixtures/forced_continuity.html`
- Create: `dp-detector/fixtures/confirm_shaming.html`
- Create: `dp-detector/fixtures/disguised_button.html`
- Create: `dp-detector/fixtures/deep_cancel_flow.html`
- Create: `dp-detector/fixtures/easy_cancel_flow.html`
- Test: `dp-detector/tests/test_fixtures.py`

**Interfaces:**
- Produces: 7 static HTML files under `fixtures/`, each with a `data-fixture-id` attribute on `<body>` matching its filename stem, consumed by Task 4's capture module and Task 13's `/fixtures` endpoint.

- [ ] **Step 1: Write the failing test**

```python
# dp-detector/tests/test_fixtures.py
from pathlib import Path
from bs4 import BeautifulSoup

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
EXPECTED = [
    "precheck_consent", "hidden_cancel", "forced_continuity",
    "confirm_shaming", "disguised_button", "deep_cancel_flow", "easy_cancel_flow",
]

def test_all_fixtures_exist_and_are_valid_html():
    for name in EXPECTED:
        path = FIXTURES_DIR / f"{name}.html"
        assert path.exists(), f"missing fixture {path}"
        soup = BeautifulSoup(path.read_text(), "html.parser")
        assert soup.body is not None
        assert soup.body.get("data-fixture-id") == name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dp-detector && pytest tests/test_fixtures.py -v`
Expected: FAIL — fixture files don't exist yet.

- [ ] **Step 3: Write the fixtures**

```html
<!-- dp-detector/fixtures/precheck_consent.html -->
<!DOCTYPE html>
<html><head><title>Sign Up</title></head>
<body data-fixture-id="precheck_consent">
  <h1>Create your account</h1>
  <form>
    <label><input type="email" required> Email</label>
    <label>
      <input type="checkbox" id="marketing-optin" checked>
      Send me marketing emails and special offers
    </label>
    <button type="submit">Create Account</button>
  </form>
</body></html>
```

```html
<!-- dp-detector/fixtures/hidden_cancel.html -->
<!DOCTYPE html>
<html><head><title>Account Settings</title></head>
<body data-fixture-id="hidden_cancel">
  <h1>Account Settings</h1>
  <a href="#manage" style="display:inline-block;padding:10px 20px;background:#0066ff;color:#fff;">Manage Subscription</a>
  <a href="#cancel" id="cancel-link" style="display:none;">Cancel my subscription</a>
</body></html>
```

```html
<!-- dp-detector/fixtures/forced_continuity.html -->
<!DOCTYPE html>
<html><head><title>Free Trial</title></head>
<body data-fixture-id="forced_continuity">
  <h1>Start your free trial</h1>
  <p>Get full access for $0 today. $23.99/month after 30 days.</p>
  <button>Start Free Trial</button>
</body></html>
```

```html
<!-- dp-detector/fixtures/confirm_shaming.html -->
<!DOCTYPE html>
<html><head><title>Newsletter</title></head>
<body data-fixture-id="confirm_shaming">
  <h1>Subscribe and save 20%</h1>
  <button id="accept">Yes, sign me up</button>
  <button id="decline">No thanks, I don't want to save money</button>
</body></html>
```

```html
<!-- dp-detector/fixtures/disguised_button.html -->
<!DOCTYPE html>
<html><head><title>Cookie Consent</title></head>
<body data-fixture-id="disguised_button">
  <h1>We value your privacy</h1>
  <button id="accept-all" style="font-size:20px;padding:16px 32px;background:#00aa00;color:#fff;">Accept All</button>
  <a id="reject-all" href="#reject" style="font-size:9px;color:#cccccc;">reject</a>
</body></html>
```

```html
<!-- dp-detector/fixtures/deep_cancel_flow.html -->
<!DOCTYPE html>
<html><head><title>Manage Account</title></head>
<body data-fixture-id="deep_cancel_flow">
  <div id="step-0">
    <button id="upgrade-btn">Upgrade now (1 click)</button>
    <a id="start-cancel" href="#" onclick="document.getElementById('step-0').style.display='none';document.getElementById('step-1').style.display='block';return false;">Cancel account</a>
  </div>
  <div id="step-1" style="display:none;">
    <p>Wait! Are you sure? You'll lose all your data.</p>
    <a id="step-1-continue" href="#" onclick="document.getElementById('step-1').style.display='none';document.getElementById('step-2').style.display='block';return false;">Continue cancelling</a>
  </div>
  <div id="step-2" style="display:none;">
    <p>Have you tried contacting support first?</p>
    <a id="step-2-continue" href="#" onclick="document.getElementById('step-2').style.display='none';document.getElementById('step-3').style.display='block';return false;">Continue cancelling</a>
  </div>
  <div id="step-3" style="display:none;">
    <p>This is your final confirmation.</p>
    <button id="confirm-cancel">Confirm cancellation</button>
  </div>
</body></html>
```

```html
<!-- dp-detector/fixtures/easy_cancel_flow.html -->
<!DOCTYPE html>
<html><head><title>Manage Account</title></head>
<body data-fixture-id="easy_cancel_flow">
  <button id="upgrade-btn">Upgrade now</button>
  <button id="confirm-cancel">Cancel account</button>
</body></html>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dp-detector && pytest tests/test_fixtures.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dp-detector/fixtures/
git commit -m "feat: add self-authored dark-pattern fixture pages"
```

---

### Task 4: Capture module

**Files:**
- Create: `dp-detector/capture/__init__.py`
- Create: `dp-detector/capture/capture.py`
- Test: `dp-detector/tests/test_capture.py`

**Interfaces:**
- Consumes: fixture files from Task 3.
- Produces: `Element` (fields: `selector: str`, `tag: str`, `text: str`, `bbox: tuple[float, float, float, float]`, `display: str`, `visibility: str`, `font_size: float`, `opacity: float`, `attrs: dict`), `ClickTraceStep` (fields: `step_index: int`, `clicked_selector: str`, `visible_text_after: str`), `Capture` (fields: `url: str`, `dom_html: str`, `screenshot_png: bytes`, `rendered_text: str`, `elements: list[Element]`, `click_trace: list[ClickTraceStep]`). Function `capture(source: str, is_fixture: bool = False) -> Capture`, consumed by every engine task and by Task 13's API.

- [ ] **Step 1: Write the failing test**

```python
# dp-detector/tests/test_capture.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dp-detector && pytest tests/test_capture.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'capture'`

- [ ] **Step 3: Write minimal implementation**

```python
# dp-detector/capture/__init__.py
```

```python
# dp-detector/capture/capture.py
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright

CANCEL_TEXT_RE = re.compile(r"cancel|unsubscribe|delete.*account", re.IGNORECASE)
MAX_CLICK_DEPTH = 5

EXTRACT_ELEMENTS_JS = """
() => {
    const els = [];
    const all = document.querySelectorAll('body, body *');
    let idx = 0;
    for (const el of all) {
        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        const attrs = {};
        for (const attr of el.attributes) { attrs[attr.name] = attr.value; }
        const ownText = Array.from(el.childNodes)
            .filter(n => n.nodeType === 3)
            .map(n => n.textContent.trim())
            .join(' ').trim();
        idx += 1;
        const selector = el.id ? ('#' + el.id) : (el.tagName.toLowerCase() + ':nth-child(' + idx + ')');
        els.push({
            selector: selector,
            tag: el.tagName.toLowerCase(),
            text: ownText || el.innerText || '',
            bbox: [rect.x, rect.y, rect.x + rect.width, rect.y + rect.height],
            display: style.display,
            visibility: style.visibility,
            font_size: parseFloat(style.fontSize) || 0,
            opacity: parseFloat(style.opacity),
            attrs: attrs,
        });
    }
    return els;
}
"""


@dataclass
class Element:
    selector: str
    tag: str
    text: str
    bbox: tuple
    display: str
    visibility: str
    font_size: float
    opacity: float
    attrs: dict = field(default_factory=dict)


@dataclass
class ClickTraceStep:
    step_index: int
    clicked_selector: str
    visible_text_after: str


@dataclass
class Capture:
    url: str
    dom_html: str
    screenshot_png: bytes
    rendered_text: str
    elements: list
    click_trace: list


def _extract_elements(page) -> list:
    raw = page.evaluate(EXTRACT_ELEMENTS_JS)
    return [
        Element(
            selector=r["selector"], tag=r["tag"], text=r["text"], bbox=tuple(r["bbox"]),
            display=r["display"], visibility=r["visibility"], font_size=r["font_size"],
            opacity=r["opacity"], attrs=r["attrs"],
        )
        for r in raw
    ]


def _walk_click_trace(page) -> list:
    trace = []
    for depth in range(MAX_CLICK_DEPTH):
        candidates = page.query_selector_all("a, button")
        target = None
        for c in candidates:
            try:
                text = c.inner_text()
            except Exception:
                continue
            if text and CANCEL_TEXT_RE.search(text):
                box = c.bounding_box()
                if box is not None:
                    target = c
                    break
        if target is None:
            break
        try:
            target.click()
            page.wait_for_timeout(200)
        except Exception:
            break
        visible_text = page.inner_text("body")
        trace.append(ClickTraceStep(step_index=depth, clicked_selector=target.evaluate("el => el.id ? '#' + el.id : el.tagName") , visible_text_after=visible_text))
    return trace


def capture(source: str, is_fixture: bool = False) -> Capture:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        if is_fixture:
            file_url = "file://" + str(Path(source).resolve())
            page.goto(file_url)
            url = file_url
        else:
            page.goto(source, wait_until="networkidle")
            url = source

        dom_html = page.content()
        screenshot_png = page.screenshot()
        rendered_text = page.inner_text("body")
        elements = _extract_elements(page)
        click_trace = _walk_click_trace(page)

        browser.close()

    return Capture(
        url=url, dom_html=dom_html, screenshot_png=screenshot_png,
        rendered_text=rendered_text, elements=elements, click_trace=click_trace,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dp-detector && pytest tests/test_capture.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dp-detector/capture/
git commit -m "feat: add Playwright-based page capture for URLs and fixtures"
```

---

### Task 5: Structure engine

**Files:**
- Create: `dp-detector/engines/structure.py`
- Test: `dp-detector/tests/test_structure.py`

**Interfaces:**
- Consumes: `Capture`, `Element` from Task 4; `Finding` from Task 2.
- Produces: `analyze(capture: Capture) -> list[Finding]` with `engine="structure"`, `pattern_type` one of `"preselected_consent"`, `"hidden_cancel_link"`, `"forced_required_consent"`, `"forced_continuity_pricing"`.

- [ ] **Step 1: Write the failing test**

```python
# dp-detector/tests/test_structure.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dp-detector && pytest tests/test_structure.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engines.structure'`

- [ ] **Step 3: Write minimal implementation**

```python
# dp-detector/engines/structure.py
import re
from engines.base import Finding

MARKETING_KEYWORDS_RE = re.compile(r"marketing|special offers|newsletter|promot", re.IGNORECASE)

# Adapted from Pattern Highlighter (DAPDE Project, MIT License)
# https://github.com/Dapde/Pattern-Highlighter — chrome/scripts/constants.js, "Forced Continuity" pattern
FORCED_CONTINUITY_RES = [
    re.compile(
        r"(?:(?:€|EUR|GBP|£|\$|USD)\s*\d+(?:\.\d{2})?|\d+(?:\.\d{2})?\s*(?:euros?|€|EUR|GBP|£|pounds?(?:\s*sterling)?|\$|USD|dollars?))"
        r"\s*(?:(?:(?:per|\/|a)\s*month)|(?:p|\/)m)\s*(?:after|from\s*(?:month|day)\s*\d+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:(?:€|EUR|GBP|£|\$|USD)\s*\d+(?:\.\d{2})?|\d+(?:\.\d{2})?\s*(?:euros?|€|EUR|GBP|£|pounds?(?:\s*sterling)?|\$|USD|dollars?))"
        r"\s*(?:after\s*(?:the)?\s*\d+(?:th|nd|rd)?\s*(?:months?|days?)|from\s*(?:month|day)\s*\d+)",
        re.IGNORECASE,
    ),
]


def _is_hidden(el) -> bool:
    return el.display == "none" or el.visibility == "hidden" or el.font_size < 2 or el.opacity == 0


def analyze(capture) -> list:
    findings = []

    for el in capture.elements:
        if el.tag == "input" and el.attrs.get("type") == "checkbox" and "checked" in el.attrs:
            findings.append(Finding(
                engine="structure", pattern_type="preselected_consent", confidence=0.85,
                evidence=f"checkbox '{el.selector}' is pre-checked", element_ref=el.selector,
            ))
            if el.attrs.get("required") is not None:
                findings.append(Finding(
                    engine="structure", pattern_type="forced_required_consent", confidence=0.8,
                    evidence=f"checkbox '{el.selector}' is both pre-checked and required", element_ref=el.selector,
                ))

    for el in capture.elements:
        if el.tag == "a" and CANCEL_RE_MATCH(el.text) and _is_hidden(el):
            findings.append(Finding(
                engine="structure", pattern_type="hidden_cancel_link", confidence=0.8,
                evidence=f"cancel-related link '{el.selector}' is hidden (display={el.display}, visibility={el.visibility})",
                element_ref=el.selector,
            ))

    for pattern in FORCED_CONTINUITY_RES:
        if pattern.search(capture.rendered_text):
            findings.append(Finding(
                engine="structure", pattern_type="forced_continuity_pricing", confidence=0.75,
                evidence="page text matches forced-continuity pricing pattern (price/month after trial)",
            ))
            break

    return findings


def CANCEL_RE_MATCH(text: str) -> bool:
    return bool(re.search(r"cancel|unsubscribe|delete.*account", text, re.IGNORECASE))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dp-detector && pytest tests/test_structure.py -v`
Expected: PASS

- [ ] **Step 5: Add vendor attribution and commit**

```
# dp-detector/vendor/NOTICE  (create this file now, appended to in later tasks)
This project reuses logic and/or code from the following projects:

- Pattern Highlighter (DAPDE Project) — MIT License
  https://github.com/Dapde/Pattern-Highlighter
  Used: forced-continuity pricing regular expressions, adapted into
  engines/structure.py from chrome/scripts/constants.js.
```

```bash
mkdir -p dp-detector/vendor
git add dp-detector/engines/structure.py dp-detector/tests/test_structure.py dp-detector/vendor/NOTICE
git commit -m "feat: add Structure engine with DOM rule checks"
```

---

### Task 6: Visual engine (UIGuard adapter)

**Files:**
- Create: `dp-detector/vendor/uiguard/__init__.py`
- Create: `dp-detector/vendor/uiguard/rule_check.py` (vendored from UIST23-UIGuard `src/rule_check.py`, Apache-2.0, unmodified)
- Create: `dp-detector/vendor/uiguard/utils.py` (vendored from UIST23-UIGuard `src/utils.py`, unmodified)
- Create: `dp-detector/vendor/uiguard/grouping.py` (vendored from UIST23-UIGuard `src/DBSCAN/grouping.py`, unmodified)
- Modify: `dp-detector/vendor/uiguard/rule_check.py` — change its two internal imports from `from utils import ...` / `from DBSCAN.grouping import get_grouping` to relative imports `from .utils import ...` / `from .grouping import get_grouping`, since it now lives in a package
- Create: `dp-detector/engines/visual.py`
- Modify: `dp-detector/vendor/NOTICE`
- Test: `dp-detector/tests/test_visual.py`

**Interfaces:**
- Consumes: `Capture`, `Element` from Task 4; `Finding` from Task 2.
- Produces: `analyze(capture: Capture) -> list[Finding]` with `engine="visual"`, `pattern_type` values taken from UIGuard's `dpCode2dpRealName` mapping (e.g. `"preselection"`, `"interface_inference"`, `"forced_action"`).

- [ ] **Step 1: Vendor the three UIGuard source files**

```bash
mkdir -p dp-detector/vendor/uiguard
touch dp-detector/vendor/uiguard/__init__.py
curl -sL https://raw.githubusercontent.com/chenjshnn/UIST23-UIGuard/main/src/rule_check.py -o dp-detector/vendor/uiguard/rule_check.py
curl -sL https://raw.githubusercontent.com/chenjshnn/UIST23-UIGuard/main/src/utils.py -o dp-detector/vendor/uiguard/utils.py
curl -sL https://raw.githubusercontent.com/chenjshnn/UIST23-UIGuard/main/src/DBSCAN/grouping.py -o dp-detector/vendor/uiguard/grouping.py
```

Then edit `dp-detector/vendor/uiguard/rule_check.py`: change the line `from utils import merge_bbox, is_horizontal_overlap` to `from .utils import merge_bbox, is_horizontal_overlap`, and the line `from DBSCAN.grouping import get_grouping` to `from .grouping import get_grouping`.

- [ ] **Step 2: Write the failing test**

```python
# dp-detector/tests/test_visual.py
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd dp-detector && pytest tests/test_visual.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engines.visual'`

- [ ] **Step 4: Write the adapter**

```python
# dp-detector/engines/visual.py
import tempfile
from pathlib import Path

from engines.base import Finding
from vendor.uiguard import rule_check

rule_check.flag_icon = False
rule_check.flag_TM = False
rule_check.flag_status = False
rule_check.flag_grouping = False

TAG_TO_CATEGORY = {
    "button": "Button",
    "img": "ImageView",
    "input:checkbox": "CheckBox",
    "input:radio": "RadioButton",
    "input:text": "EditText",
    "input:email": "EditText",
    "a": "Button",
}


def _category_for(el) -> str:
    if el.tag == "input":
        return TAG_TO_CATEGORY.get(f"input:{el.attrs.get('type', 'text')}", "EditText")
    if el.tag in TAG_TO_CATEGORY:
        return TAG_TO_CATEGORY[el.tag]
    if el.text.strip():
        return "pText"
    return "TextView"


def _build_item_list(capture) -> list:
    items = []
    for el in capture.elements:
        x1, y1, x2, y2 = el.bbox
        if x2 <= x1 or y2 <= y1:
            continue
        items.append({
            "category": _category_for(el),
            "bbox": [x1, y1, x2, y2],
            "text": el.text,
            "score": 1.0,
        })
    return items


def analyze(capture) -> list:
    item_list = _build_item_list(capture)
    if not item_list:
        return []

    max_x = max(item["bbox"][2] for item in item_list)
    max_y = max(item["bbox"][3] for item in item_list)

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "screenshot.png"
        img_path.write_bytes(capture.screenshot_png)

        final_results = rule_check.predict_type(
            all_properties=item_list,
            img_h=int(max_y),
            img_w=int(max_x),
            vis_save_root=tmpdir,
            img_path=str(img_path),
            vis=False,
        )

    findings = []
    for dp_code, instances in (final_results or {}).items():
        for inst in instances:
            findings.append(Finding(
                engine="visual",
                pattern_type=_readable_name(dp_code),
                confidence=0.7,
                evidence=f"UIGuard rule matched code {dp_code} at bbox {inst.get('bbox')}",
            ))
    return findings


DP_CODE_TO_NAME = {
    "II-AM-G-SMALL": "interface_inference",
    "FA-G-WATCHAD": "forced_action",
    "SN-FC": "forced_continuity",
    "II-AM-TWE": "aesthetic_manipulation",
    "II-AM-DA": "disguised_ad",
    "FA-SOCIALPYRAMID": "social_pyramid",
    "FA-GAMIFICATION": "forced_action",
    "NG-RATE": "nag_to_rate",
    "NG-UPGRADE": "nag_to_upgrade",
    "FA-G-COUNTDOWNAD": "forced_action",
    "II-AM-FH": "false_hierarchy",
    "FA-G-PRO": "forced_action",
    "II-PRE-FOLLOW": "preselection",
    "II-PRE-NOTIFICATION": "preselection",
    "II-PRE-PRIVACY": "preselection",
    "II-PRE-USAGE-DATA": "preselection",
    "II-PRE": "preselection",
    "FA-Privacy": "forced_action",
}


def _readable_name(dp_code: str) -> str:
    return DP_CODE_TO_NAME.get(dp_code, dp_code.lower())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd dp-detector && pytest tests/test_visual.py -v`
Expected: PASS. If `predict_type` raises on missing optional fields (e.g. `status`, `iconLabel`) for some items, add the missing key with a neutral default (`""` or `None`) to the dict built in `_build_item_list` and re-run — do not change vendored code to work around it.

- [ ] **Step 6: Update attribution and commit**

Append to `dp-detector/vendor/NOTICE`:

```

- UIGuard — Apache License 2.0
  https://github.com/chenjshnn/UIST23-UIGuard
  Used: vendor/uiguard/rule_check.py, utils.py, grouping.py, vendored
  unmodified except for converting two imports to relative imports.
  Original paper: Chen et al., "Unveiling the Tricks: Automated
  Detection of Dark Patterns in Mobile Applications", UIST 2023.
```

```bash
git add dp-detector/vendor/uiguard dp-detector/engines/visual.py dp-detector/tests/test_visual.py dp-detector/vendor/NOTICE
git commit -m "feat: add Visual engine adapting UIGuard's rule_check logic"
```

---

### Task 7: Shared BERT fine-tuning helper + Linguistic engine training

**Files:**
- Create: `dp-detector/training/__init__.py`
- Create: `dp-detector/training/bert_finetune.py`
- Create: `dp-detector/training/train_linguistic.py`

**Interfaces:**
- Produces: `finetune_bert_classifier(dataset_tsv_path: str, output_dir: str, text_col: str, label_col: str, base_model: str = "bert-base-uncased", epochs: int = 3) -> None`, saved model+tokenizer at `output_dir`. Consumed by both this task and Task 10.
- Produces: a trained checkpoint at `dp-detector/models/linguistic_bert/`, consumed by Task 8.

- [ ] **Step 1: Write the shared fine-tuning helper**

```python
# dp-detector/training/__init__.py
```

```python
# dp-detector/training/bert_finetune.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import torch


class _TsvTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def finetune_bert_classifier(dataset_tsv_path: str, output_dir: str, text_col: str, label_col: str,
                              base_model: str = "bert-base-uncased", epochs: int = 3) -> None:
    df = pd.read_csv(dataset_tsv_path, sep="\t")
    df = df.dropna(subset=[text_col, label_col])
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df[label_col])

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)

    train_ds = _TsvTextDataset(train_df[text_col], train_df[label_col], tokenizer)
    val_ds = _TsvTextDataset(val_df[text_col], val_df[label_col], tokenizer)

    args = TrainingArguments(
        output_dir=output_dir + "_checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to=[],
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
```

- [ ] **Step 2: Write the linguistic training script**

```python
# dp-detector/training/train_linguistic.py
import urllib.request
from pathlib import Path

from training.bert_finetune import finetune_bert_classifier

DATA_DIR = Path(__file__).parent / "data"
DATASET_URL = "https://raw.githubusercontent.com/yamanalab/ec-darkpattern/master/dataset/dataset.tsv"
DATASET_PATH = DATA_DIR / "linguistic_dataset.tsv"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "linguistic_bert"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_PATH.exists():
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
    finetune_bert_classifier(
        dataset_tsv_path=str(DATASET_PATH),
        output_dir=str(OUTPUT_DIR),
        text_col="text",
        label_col="label",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the training script**

Run: `cd dp-detector && python -m training.train_linguistic`
Expected: downloads `dataset.tsv` (2,362 rows), fine-tunes `bert-base-uncased` for 3 epochs, and writes a checkpoint to `models/linguistic_bert/`. This step has no pytest assertion — its output (the saved model directory) is verified by Task 8's test. Expect several minutes on GPU.

- [ ] **Step 4: Update attribution and commit**

Append to `dp-detector/vendor/NOTICE`:

```

- ec-darkpattern dataset — Apache License 2.0
  https://github.com/yamanalab/ec-darkpattern
  Used: dataset/dataset.tsv, downloaded at training time by
  training/train_linguistic.py, to fine-tune the Linguistic engine's
  BERT classifier. Underlying text originally from Mathur et al.,
  "Dark Patterns at Scale" (CSCW 2019).
```

```bash
git add dp-detector/training/__init__.py dp-detector/training/bert_finetune.py dp-detector/training/train_linguistic.py dp-detector/vendor/NOTICE
git commit -m "feat: add shared BERT fine-tuning helper and Linguistic training script"
```
(`models/` and `training/data/` stay untracked per `.gitignore`.)

---

### Task 8: Linguistic engine inference

**Files:**
- Create: `dp-detector/engines/linguistic.py`
- Test: `dp-detector/tests/test_linguistic.py`

**Interfaces:**
- Consumes: `Capture` from Task 4; `Finding` from Task 2; checkpoint at `models/linguistic_bert/` from Task 7.
- Produces: `analyze(capture: Capture) -> list[Finding]` with `engine="linguistic"`, `pattern_type="manipulative_text"`.

- [ ] **Step 1: Write the test (skips gracefully if the model isn't trained yet)**

```python
# dp-detector/tests/test_linguistic.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dp-detector && pytest tests/test_linguistic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engines.linguistic'` (or skipped if the model isn't trained; train it first so this step is meaningful).

- [ ] **Step 3: Write the inference wrapper**

```python
# dp-detector/engines/linguistic.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dp-detector && pytest tests/test_linguistic.py -v`
Expected: PASS (after Task 7's training has produced `models/linguistic_bert/`).

- [ ] **Step 5: Commit**

```bash
git add dp-detector/engines/linguistic.py dp-detector/tests/test_linguistic.py
git commit -m "feat: add Linguistic engine BERT inference wrapper"
```

---

### Task 9: Behavior engine — dataset preparation from AppRay

**Files:**
- Create: `dp-detector/training/prepare_behavior_dataset.py`

**Interfaces:**
- Produces: `dp-detector/training/data/behavior_dataset.tsv` with columns `text`, `label` (1 = dark pattern, 0 = benign), consumed by Task 10.

- [ ] **Step 1: Write the data-prep script**

```python
# dp-detector/training/prepare_behavior_dataset.py
import argparse
import json
import zipfile
from pathlib import Path

import easyocr
import requests
from PIL import Image

DATA_DIR = Path(__file__).parent / "data"
DARK_ZIP_URL = "https://zenodo.org/records/13268006/files/AppRay-Dark.zip?download=1"
LIGHT_ZIP_URL = "https://zenodo.org/records/13268006/files/AppRay-Light.zip?download=1"
ANNOTATIONS_URL = "https://zenodo.org/records/13268006/files/all_images_patterns.json?download=1"
OUTPUT_TSV = DATA_DIR / "behavior_dataset.tsv"


def _download(url: str, dest: Path):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def _unzip(zip_path: Path, dest_dir: Path):
    if dest_dir.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)


def main(sample_size: int):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dark_zip = DATA_DIR / "AppRay-Dark.zip"
    light_zip = DATA_DIR / "AppRay-Light.zip"
    annotations_path = DATA_DIR / "all_images_patterns.json"

    _download(DARK_ZIP_URL, dark_zip)
    _download(LIGHT_ZIP_URL, light_zip)
    _download(ANNOTATIONS_URL, annotations_path)

    dark_dir = DATA_DIR / "AppRay-Dark"
    light_dir = DATA_DIR / "AppRay-Light"
    _unzip(dark_zip, dark_dir)
    _unzip(light_zip, light_dir)

    annotations = json.loads(annotations_path.read_text())
    reader = easyocr.Reader(["en"], gpu=True)

    rows = []
    dark_items = list(annotations.items())[:sample_size]
    for image_name, meta in dark_items:
        image_path = next(dark_dir.rglob(image_name), None)
        if image_path is None:
            continue
        img = Image.open(image_path).convert("RGB")
        for pattern in meta.get("patterns", []):
            box = pattern.get("bbox") or pattern
            try:
                crop = img.crop((box["xmin"], box["ymin"], box["xmax"], box["ymax"]))
            except (KeyError, TypeError):
                continue
            text = " ".join(reader.readtext(np_from_pil(crop), detail=0))
            if text.strip():
                rows.append((text.strip(), 1))

    light_images = list(light_dir.rglob("*.png"))[:sample_size] + list(light_dir.rglob("*.jpg"))[:sample_size]
    for image_path in light_images:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        crop = img.crop((0, 0, w, min(200, h)))
        text = " ".join(reader.readtext(np_from_pil(crop), detail=0))
        if text.strip():
            rows.append((text.strip(), 0))

    with open(OUTPUT_TSV, "w", encoding="utf-8") as f:
        f.write("text\tlabel\n")
        for text, label in rows:
            f.write(f"{text.replace(chr(9), ' ')}\t{label}\n")

    print(f"Wrote {len(rows)} rows to {OUTPUT_TSV}")


def np_from_pil(img):
    import numpy as np
    return np.array(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=300,
                         help="Max number of labeled images to process per class (default 300, keeps first Phase-1 run fast; raise for a fuller dataset).")
    args = parser.parse_args()
    main(args.sample_size)
```

- [ ] **Step 2: Run the script**

Run: `cd dp-detector && python -m training.prepare_behavior_dataset --sample-size 300`
Expected: downloads AppRay-Dark.zip (~504MB) and AppRay-Light.zip (~510MB) once (cached on re-run), OCRs the first 300 labeled dark-pattern crops and 300 benign crops, writes `training/data/behavior_dataset.tsv`. No pytest assertion — verify manually with `wc -l dp-detector/training/data/behavior_dataset.tsv` (expect several hundred rows) before moving to Task 10. If `all_images_patterns.json`'s actual key/field names differ from `patterns`/`bbox`/`xmin` (the schema wasn't verified against a live download during planning — only Zenodo's file listing was checked), inspect the downloaded JSON directly (`python -c "import json; print(json.dumps(json.load(open('dp-detector/training/data/all_images_patterns.json')), indent=2)[:2000])"`) and adjust the key names in `main()` accordingly before re-running.

- [ ] **Step 3: Update attribution and commit**

Append to `dp-detector/vendor/NOTICE`:

```

- AppRay-Dark / AppRay-Light datasets — from AppRay, MIT License
  https://github.com/chenjshnn/AppRay
  Dataset: https://zenodo.org/records/13268006
  Used: labeled UI screenshots + bounding-box annotations, OCR'd via
  training/prepare_behavior_dataset.py to build a text classification
  dataset for the Behavior engine. Full AppRay LLM-guided exploration
  and image-fusion model are not used — see design spec decision log.
```

```bash
git add dp-detector/training/prepare_behavior_dataset.py dp-detector/vendor/NOTICE
git commit -m "feat: add Behavior engine dataset preparation from AppRay data"
```

---

### Task 10: Behavior engine training

**Files:**
- Create: `dp-detector/training/train_behavior.py`

**Interfaces:**
- Consumes: `finetune_bert_classifier` from Task 7; `training/data/behavior_dataset.tsv` from Task 9.
- Produces: a trained checkpoint at `dp-detector/models/behavior_bert/`, consumed by Task 11.

- [ ] **Step 1: Write the training script**

```python
# dp-detector/training/train_behavior.py
from pathlib import Path

from training.bert_finetune import finetune_bert_classifier

DATASET_PATH = Path(__file__).parent / "data" / "behavior_dataset.tsv"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "behavior_bert"


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found — run `python -m training.prepare_behavior_dataset` first (Task 9)."
        )
    finetune_bert_classifier(
        dataset_tsv_path=str(DATASET_PATH),
        output_dir=str(OUTPUT_DIR),
        text_col="text",
        label_col="label",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the training script**

Run: `cd dp-detector && python -m training.train_behavior`
Expected: fine-tunes `bert-base-uncased` on the OCR'd AppRay text dataset, writes checkpoint to `models/behavior_bert/`. No pytest assertion — verified by Task 11's test.

- [ ] **Step 3: Commit**

```bash
git add dp-detector/training/train_behavior.py
git commit -m "feat: add Behavior engine BERT training script"
```

---

### Task 11: Behavior engine inference

**Files:**
- Create: `dp-detector/engines/behavior.py`
- Test: `dp-detector/tests/test_behavior.py`

**Interfaces:**
- Consumes: `Capture`, `ClickTraceStep` from Task 4; `Finding` from Task 2; checkpoint at `models/behavior_bert/` from Task 10.
- Produces: `analyze(capture: Capture) -> list[Finding]` with `engine="behavior"`, `pattern_type="roach_motel"`.

- [ ] **Step 1: Write the test (skips gracefully if the model isn't trained yet)**

```python
# dp-detector/tests/test_behavior.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dp-detector && pytest tests/test_behavior.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engines.behavior'`

- [ ] **Step 3: Write the inference wrapper**

```python
# dp-detector/engines/behavior.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dp-detector && pytest tests/test_behavior.py -v`
Expected: PASS (after Task 10's training has produced `models/behavior_bert/`).

- [ ] **Step 5: Commit**

```bash
git add dp-detector/engines/behavior.py dp-detector/tests/test_behavior.py
git commit -m "feat: add Behavior engine click-trace + BERT inference"
```

---

### Task 12: Fusion module

**Files:**
- Create: `dp-detector/fusion.py`
- Test: `dp-detector/tests/test_fusion.py`

**Interfaces:**
- Consumes: `list[Finding]` from Task 2.
- Produces: `fuse(findings: list[Finding]) -> dict` returning `{"risk_score": int, "findings": list[Finding]}`, consumed by Task 13.

- [ ] **Step 1: Write the failing test**

```python
# dp-detector/tests/test_fusion.py
from engines.base import Finding
from fusion import fuse

def test_no_findings_gives_zero_score():
    result = fuse([])
    assert result["risk_score"] == 0
    assert result["findings"] == []

def test_score_increases_with_findings_and_caps_at_100():
    findings = [
        Finding(engine="structure", pattern_type="a", confidence=1.0, evidence="x"),
        Finding(engine="visual", pattern_type="b", confidence=1.0, evidence="x"),
        Finding(engine="linguistic", pattern_type="c", confidence=1.0, evidence="x"),
        Finding(engine="behavior", pattern_type="d", confidence=1.0, evidence="x"),
    ]
    result = fuse(findings)
    assert result["risk_score"] == 100

def test_partial_confidence_scales_score():
    findings = [Finding(engine="structure", pattern_type="a", confidence=0.5, evidence="x")]
    result = fuse(findings)
    assert 0 < result["risk_score"] < 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dp-detector && pytest tests/test_fusion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fusion'`

- [ ] **Step 3: Write minimal implementation**

```python
# dp-detector/fusion.py
ENGINE_WEIGHT = {
    "structure": 25,
    "visual": 25,
    "linguistic": 25,
    "behavior": 25,
}


def fuse(findings: list) -> dict:
    score = 0.0
    for f in findings:
        score += f.confidence * ENGINE_WEIGHT.get(f.engine, 10)
    return {"risk_score": min(100, round(score)), "findings": findings}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dp-detector && pytest tests/test_fusion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dp-detector/fusion.py dp-detector/tests/test_fusion.py
git commit -m "feat: add deterministic fusion of engine findings into risk score"
```

---

### Task 13: FastAPI wiring

**Files:**
- Modify: `dp-detector/app.py`
- Test: `dp-detector/tests/test_integration.py`

**Interfaces:**
- Consumes: `capture()` from Task 4; `analyze()` from Tasks 5/6/8/11; `fuse()` from Task 12.
- Produces: `POST /analyze` accepting `{"url": str}` or `{"fixture_id": str}`, returning `{"risk_score": int, "findings": [...], "screenshot_base64": str, "elements": [...], "engine_errors": {...}}`. `GET /fixtures` returning `{"fixtures": [str, ...]}`.

- [ ] **Step 1: Write the failing test**

```python
# dp-detector/tests/test_integration.py
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_fixtures_endpoint_lists_fixtures():
    response = client.get("/fixtures")
    assert response.status_code == 200
    assert "precheck_consent" in response.json()["fixtures"]

def test_analyze_fixture_returns_valid_shape():
    response = client.post("/analyze", json={"fixture_id": "precheck_consent"})
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["risk_score"] <= 100
    assert isinstance(body["findings"], list)
    assert isinstance(body["screenshot_base64"], str) and len(body["screenshot_base64"]) > 0
    assert isinstance(body["elements"], list)

def test_analyze_missing_fixture_returns_error():
    response = client.post("/analyze", json={"fixture_id": "does_not_exist"})
    assert response.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dp-detector && pytest tests/test_integration.py -v`
Expected: FAIL — `/fixtures` and `/analyze` don't exist yet (404 on both, but a different failure mode than the assertions expect).

- [ ] **Step 3: Wire the endpoints**

```python
# dp-detector/app.py
import base64
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from capture.capture import capture
from engines import structure, visual, linguistic, behavior
from fusion import fuse

app = FastAPI(title="Dark Pattern Detection Engine")

FIXTURES_DIR = Path(__file__).parent / "fixtures"
ENGINES = {
    "structure": structure.analyze,
    "visual": visual.analyze,
    "linguistic": linguistic.analyze,
    "behavior": behavior.analyze,
}


class AnalyzeRequest(BaseModel):
    url: str | None = None
    fixture_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/fixtures")
def list_fixtures():
    names = sorted(p.stem for p in FIXTURES_DIR.glob("*.html"))
    return {"fixtures": names}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if req.fixture_id:
        fixture_path = FIXTURES_DIR / f"{req.fixture_id}.html"
        if not fixture_path.exists():
            raise HTTPException(status_code=404, detail=f"unknown fixture_id '{req.fixture_id}'")
        source, is_fixture = str(fixture_path), True
    elif req.url:
        source, is_fixture = req.url, False
    else:
        raise HTTPException(status_code=400, detail="must provide either 'url' or 'fixture_id'")

    try:
        cap = capture(source, is_fixture=is_fixture)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"capture failed: {e}")

    all_findings = []
    engine_errors = {}
    for name, analyze_fn in ENGINES.items():
        try:
            all_findings.extend(analyze_fn(cap))
        except Exception as e:
            engine_errors[name] = str(e)

    result = fuse(all_findings)

    return {
        "risk_score": result["risk_score"],
        "findings": [asdict(f) for f in result["findings"]],
        "screenshot_base64": base64.b64encode(cap.screenshot_png).decode("ascii"),
        "elements": [asdict(e) for e in cap.elements],
        "engine_errors": engine_errors,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dp-detector && pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite**

Run: `cd dp-detector && pytest -v`
Expected: all tests PASS (Linguistic/Behavior tests only if Tasks 7/9/10 training has been run; otherwise they skip cleanly).

- [ ] **Step 6: Commit**

```bash
git add dp-detector/app.py dp-detector/tests/test_integration.py
git commit -m "feat: wire capture, engines, and fusion into /analyze and /fixtures endpoints"
```

---

### Task 14: Frontend

**Files:**
- Create: `dp-detector/static/index.html`
- Create: `dp-detector/static/app.js`
- Create: `dp-detector/static/style.css`
- Modify: `dp-detector/app.py` — serve static files and the index page

**Interfaces:**
- Consumes: `GET /fixtures` and `POST /analyze` JSON contracts from Task 13.

- [ ] **Step 1: Write the frontend**

```html
<!-- dp-detector/static/index.html -->
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Dark Pattern Detector</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <header>
    <h1>Dark Pattern Detector</h1>
    <div id="controls">
      <input id="url-input" type="text" placeholder="https://example.com">
      <button id="analyze-url-btn">Analyze URL</button>
      <select id="fixture-select"></select>
      <button id="analyze-fixture-btn">Analyze Fixture</button>
    </div>
    <div id="risk-badge" class="hidden">Risk Score: <span id="risk-score-value">-</span>/100</div>
  </header>
  <main>
    <div id="screenshot-container">
      <img id="screenshot" style="display:none;">
      <div id="overlay"></div>
    </div>
    <div id="findings-panel">
      <h2>Findings</h2>
      <ul id="findings-list"></ul>
    </div>
  </main>
  <script src="/static/app.js"></script>
</body>
</html>
```

```javascript
// dp-detector/static/app.js
const fixtureSelect = document.getElementById("fixture-select");
const urlInput = document.getElementById("url-input");
const screenshot = document.getElementById("screenshot");
const overlay = document.getElementById("overlay");
const findingsList = document.getElementById("findings-list");
const riskBadge = document.getElementById("risk-badge");
const riskScoreValue = document.getElementById("risk-score-value");

const ENGINE_COLORS = {
  structure: "#e74c3c",
  visual: "#3498db",
  linguistic: "#9b59b6",
  behavior: "#f39c12",
};

async function loadFixtures() {
  const res = await fetch("/fixtures");
  const data = await res.json();
  fixtureSelect.innerHTML = data.fixtures.map(f => `<option value="${f}">${f}</option>`).join("");
}

function elementBySelector(elements, selector) {
  return elements.find(e => e.selector === selector);
}

function render(data) {
  riskBadge.classList.remove("hidden");
  riskScoreValue.textContent = data.risk_score;

  screenshot.src = "data:image/png;base64," + data.screenshot_base64;
  screenshot.style.display = "block";

  overlay.innerHTML = "";
  findingsList.innerHTML = "";

  data.findings.forEach((f, i) => {
    const li = document.createElement("li");
    li.style.borderLeftColor = ENGINE_COLORS[f.engine] || "#999";
    li.innerHTML = `<strong>${f.engine}</strong>: ${f.pattern_type} (confidence ${f.confidence})<br><small>${f.evidence}</small>`;
    findingsList.appendChild(li);

    if (f.element_ref) {
      const el = elementBySelector(data.elements, f.element_ref);
      if (el) {
        const [x1, y1, x2, y2] = el.bbox;
        const box = document.createElement("div");
        box.className = "finding-box";
        box.style.left = x1 + "px";
        box.style.top = y1 + "px";
        box.style.width = (x2 - x1) + "px";
        box.style.height = (y2 - y1) + "px";
        box.style.borderColor = ENGINE_COLORS[f.engine] || "#999";
        overlay.appendChild(box);
      }
    }
  });
}

async function analyze(payload) {
  findingsList.innerHTML = "<li>Analyzing...</li>";
  const res = await fetch("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    findingsList.innerHTML = `<li>Error: ${res.status}</li>`;
    return;
  }
  render(await res.json());
}

document.getElementById("analyze-url-btn").addEventListener("click", () => {
  if (urlInput.value.trim()) analyze({ url: urlInput.value.trim() });
});

document.getElementById("analyze-fixture-btn").addEventListener("click", () => {
  analyze({ fixture_id: fixtureSelect.value });
});

loadFixtures();
```

```css
/* dp-detector/static/style.css */
body { font-family: system-ui, sans-serif; margin: 0; padding: 0; background: #f5f5f5; }
header { background: #222; color: #fff; padding: 16px 24px; }
header h1 { margin: 0 0 12px 0; font-size: 20px; }
#controls { display: flex; gap: 8px; align-items: center; }
#url-input { flex: 1; padding: 6px; }
#risk-badge { margin-top: 8px; font-size: 18px; }
#risk-badge.hidden { display: none; }
main { display: flex; gap: 16px; padding: 16px; }
#screenshot-container { position: relative; flex: 2; background: #fff; border: 1px solid #ddd; }
#screenshot { width: 100%; display: block; }
#overlay { position: absolute; top: 0; left: 0; }
.finding-box { position: absolute; border: 2px solid; background: rgba(255,0,0,0.08); pointer-events: none; }
#findings-panel { flex: 1; background: #fff; border: 1px solid #ddd; padding: 12px; max-height: 80vh; overflow-y: auto; }
#findings-list { list-style: none; padding: 0; margin: 0; }
#findings-list li { border-left: 4px solid #999; padding: 8px; margin-bottom: 8px; background: #fafafa; }
```

- [ ] **Step 2: Mount static files in app.py**

Add near the bottom of `dp-detector/app.py` (after all `@app.get`/`@app.post` routes, so the catch-all mount doesn't shadow the API routes):

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
```

- [ ] **Step 3: Manually verify**

Run: `cd dp-detector && uvicorn app:app --reload`, open `http://localhost:8000/`, select a fixture (e.g. `disguised_button`), click "Analyze Fixture", confirm the screenshot renders with colored boxes and the findings list populates with a nonzero risk score.

- [ ] **Step 4: Commit**

```bash
git add dp-detector/static/ dp-detector/app.py
git commit -m "feat: add frontend for viewing analysis results"
```

---

### Task 15: README and end-to-end manual verification

**Files:**
- Create: `dp-detector/README.md`

- [ ] **Step 1: Write the README**

```markdown
# Dark Pattern Detection Engine — Phase 1 Prototype

Implements the four-engine architecture (Structure, Visual, Linguistic,
Behavior) from the design spec at
`../docs/superpowers/specs/2026-09-16-multimodal-dark-pattern-engine-design.md`.

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m playwright install chromium

## Train the Linguistic and Behavior engines (one-time, before first use)

    python -m training.train_linguistic
    python -m training.prepare_behavior_dataset --sample-size 300
    python -m training.train_behavior

The Structure and Visual engines need no training — they're rule-based.

## Run the tests

    pytest -v

## Run the server

    uvicorn app:app --reload

Open http://localhost:8000/, pick a fixture from the dropdown (or paste a
real URL) and click Analyze.

## Manual verification checklist

- [ ] `disguised_button` fixture flags a visual finding (false hierarchy / interface inference)
- [ ] `precheck_consent` fixture flags both a structure finding (preselected_consent) and a visual finding (preselection)
- [ ] `confirm_shaming` fixture flags a linguistic finding (manipulative_text)
- [ ] `deep_cancel_flow` fixture flags a behavior finding (roach_motel); `easy_cancel_flow` does not
- [ ] A real public URL (e.g. a known cookie-consent-heavy news site) analyzes without crashing and returns a risk score
```

- [ ] **Step 2: Run through the manual verification checklist above against the running server**

- [ ] **Step 3: Commit**

```bash
git add dp-detector/README.md
git commit -m "docs: add Phase 1 setup, training, and verification instructions"
```

---

## Self-Review Notes

- **Spec coverage**: capture (Task 4), all four engines (5/6/8/11), fusion (12), API (13), frontend (14), fixtures (3), error isolation (13's `engine_errors`), README/manual verification (15) — all spec sections have a task. The spec's assumption that Pattern Highlighter does DOM/AX-tree structural analysis with MiniLM embeddings turned out, on inspecting the actual cloned repo, to be a mix-up with a different tool (the UC Berkeley extension); Pattern Highlighter's real code is four regex-based text patterns. Task 5 corrects this by hand-authoring the genuine DOM-structural checks (pre-checked consent, hidden cancel link, forced required consent) and reusing only Pattern Highlighter's real, verified Forced Continuity regexes.
- **Type consistency**: `Finding`, `Capture`, `Element`, `ClickTraceStep` are defined once (Tasks 2 and 4) and referenced identically by name across all later tasks.
- **Known open risk**: Task 9's exact JSON key names (`patterns`, `bbox`, `xmin`/`ymin`/`xmax`/`ymax`) are based on Zenodo's file-listing description, not a verified live download performed during planning — Step 2 of Task 9 includes an explicit inspect-and-adjust instruction rather than assuming the schema is correct.
