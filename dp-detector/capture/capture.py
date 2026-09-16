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
        trace.append(ClickTraceStep(
            step_index=depth,
            clicked_selector=target.evaluate("el => el.id ? '#' + el.id : el.tagName"),
            visible_text_after=new_text,
        ))
        previous_text = new_text
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
