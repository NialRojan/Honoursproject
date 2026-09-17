import re
from dataclasses import dataclass, field
from pathlib import Path

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
        // Frameworks (React, Vue, etc.) set checkbox/radio state via the
        // `.checked` DOM property, which does not update the reflected HTML
        // attribute. Sync attrs.checked to the live property so downstream
        // "checked" in el.attrs checks see real on-screen state, not just
        // whatever was in the initial static markup.
        if (el.tagName === 'INPUT' && (el.type === 'checkbox' || el.type === 'radio')) {
            if (el.checked) { attrs.checked = ''; } else { delete attrs.checked; }
        }
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
            opacity: parseFloat(style.opacity) || 1,
            background_color: style.backgroundColor,
            text_color: style.color,
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
    background_color: str
    text_color: str
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
            opacity=r["opacity"], background_color=r["background_color"],
            text_color=r["text_color"], attrs=r["attrs"],
        )
        for r in raw
    ]


def _walk_click_trace(page) -> list:
    trace = []
    previous_text = page.inner_text("body")
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
        new_text = page.inner_text("body")
        if new_text == previous_text:
            break
        try:
            clicked_selector = target.evaluate("el => el.id ? '#' + el.id : el.tagName")
        except Exception:
            clicked_selector = "unknown"
        trace.append(ClickTraceStep(
            step_index=depth,
            clicked_selector=clicked_selector,
            visible_text_after=new_text,
        ))
        previous_text = new_text
    return trace


def capture(source: str, is_fixture: bool = False) -> Capture:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            if is_fixture:
                file_url = "file://" + str(Path(source).resolve())
                page.goto(file_url)
                url = file_url
            else:
                response = page.goto(source, wait_until="networkidle")
                if response is not None and not response.ok:
                    raise RuntimeError(
                        f"upstream returned HTTP {response.status} for {source}"
                    )
                url = source

            dom_html = page.content()
            screenshot_png = page.screenshot()
            rendered_text = page.inner_text("body")
            elements = _extract_elements(page)
            click_trace = _walk_click_trace(page)
        finally:
            browser.close()

    return Capture(
        url=url, dom_html=dom_html, screenshot_png=screenshot_png,
        rendered_text=rendered_text, elements=elements, click_trace=click_trace,
    )
