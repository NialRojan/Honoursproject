import re
from engines.base import Finding

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


def CANCEL_RE_MATCH(text: str) -> bool:
    return bool(re.search(r"cancel|unsubscribe|delete.*account", text, re.IGNORECASE))


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
