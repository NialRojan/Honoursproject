import tempfile
from pathlib import Path

import cv2

# The vendored rule_check.py and grouping.py modules call cv2.namedWindow(...)
# at import time (leftover debug/visualization setup from the original
# UIGuard research code). That call fails under headless OpenCV builds with
# no GUI backend (no X server / Cocoa / GTK), which raises cv2.error before
# we ever get to call predict_type(). We don't modify the vendored files, so
# we neutralize the call here, on the shared cv2 module object, before
# importing them.
cv2.namedWindow = lambda *a, **kw: None

from engines.base import Finding
from vendor.uiguard import rule_check

rule_check.flag_icon = False
rule_check.flag_TM = False
# flag_status gates rule_check's consumption of a "status": ["checked"/"unchecked", score]
# field on "check_group" items (see final_check() around lines 621/640/674 of
# rule_check.py). It does NOT gate any of the heavy CV/icon-classifier code
# paths (those are flag_icon/flag_TM). Since _build_item_list below derives
# real checked/unchecked state from the DOM "checked" attribute -- not a
# placeholder -- it is safe and correct to enable this flag.
rule_check.flag_status = True
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

CHECKBOX_CATEGORIES = {"CheckBox"}
TEXT_CATEGORIES = {"pText", "TextView"}


def _category_for(el) -> str:
    if el.tag == "input":
        return TAG_TO_CATEGORY.get(f"input:{el.attrs.get('type', 'text')}", "EditText")
    if el.tag in TAG_TO_CATEGORY:
        return TAG_TO_CATEGORY[el.tag]
    if el.text.strip():
        return "pText"
    return "TextView"


def _contains(outer_bbox, inner_bbox, eps=1.0) -> bool:
    ox1, oy1, ox2, oy2 = outer_bbox
    ix1, iy1, ix2, iy2 = inner_bbox
    return (ox1 <= ix1 + eps and oy1 <= iy1 + eps
            and ox2 >= ix2 - eps and oy2 >= iy2 - eps)


def _area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _merge_checkboxes_with_labels(raw_items: list) -> list:
    """Pair each checkbox item with its nearest enclosing label/text item,
    using bbox containment (our Playwright-derived bboxes make this exact,
    unlike UIGuard's own distance-based utils.merge_checkbox_w_text, which
    is tuned for looser OCR-detected text boxes and mis-paired on our
    tighter DOM-derived boxes). Produces "check_group" items carrying a
    real "status" ("checked"/"unchecked") the vendored rule_check.py's
    preselection rules (final_check) consume when flag_status is True.
    """
    consumed = set()
    check_groups = []

    for i, item in enumerate(raw_items):
        if item["category"] not in CHECKBOX_CATEGORIES or i in consumed:
            continue

        best_idx = None
        best_area = None
        for j, cand in enumerate(raw_items):
            if j == i or j in consumed or cand["category"] not in TEXT_CATEGORIES:
                continue
            if not _contains(cand["bbox"], item["bbox"]):
                continue
            area = _area(cand["bbox"])
            if best_area is None or area < best_area:
                best_area = area
                best_idx = j

        status = ["checked", 1] if item.get("checked") else ["unchecked", 1]

        if best_idx is None:
            item["status"] = status
            continue

        label = raw_items[best_idx]
        check_groups.append({
            "category": "check_group",
            "bbox": label["bbox"],
            "text": label["text"],
            "status": status,
            "score": 1.0,
        })
        consumed.add(i)
        consumed.add(best_idx)

    merged = [item for idx, item in enumerate(raw_items) if idx not in consumed]
    merged.extend(check_groups)
    return merged


def _build_item_list(capture) -> list:
    raw_items = []
    for el in capture.elements:
        x1, y1, x2, y2 = el.bbox
        if x2 <= x1 or y2 <= y1:
            continue
        category = _category_for(el)
        item = {
            "category": category,
            "bbox": [x1, y1, x2, y2],
            "text": el.text,
            "score": 1.0,
        }
        if category in CHECKBOX_CATEGORIES:
            item["checked"] = "checked" in el.attrs
        raw_items.append(item)

    merged = _merge_checkboxes_with_labels(raw_items)
    for item in merged:
        item.pop("checked", None)
    return merged


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
