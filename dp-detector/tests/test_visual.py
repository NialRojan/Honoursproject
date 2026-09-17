from pathlib import Path

import pytest

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


def test_check_am_fh_survives_cluster_with_no_determinable_luminance(monkeypatch):
    # Reproduces the real crash found analyzing nytimes.com's real, complex
    # DOM: get_grouping (vendor/uiguard/grouping.py) does
    # `max(elecons_bg) - min(elecons_bg)` with no guard for an empty list.
    # elecons_bg/elecons_fg only collect a cluster's luminance values when
    # our _luminance() could actually determine one -- and it legitimately
    # returns None (not a fabricated fallback) for fully transparent or
    # unparseable colors. A real page can produce a qualifying Button
    # cluster where every item's luminance is undeterminable, which used to
    # raise ValueError("max() iterable argument is empty") inside
    # get_grouping (called from rule_check.check_AM_FH) -- discarding every
    # other finding predict_type() had already computed for the page, since
    # the crash propagates out of the whole call.
    #
    # get_grouping's real DBSCAN clustering is too fuzzy to reliably
    # hand-craft a two-element input that lands in one cluster, so this
    # test forces a known cluster by monkeypatching DBSCAN itself (not
    # get_grouping, which would bypass the exact vulnerable code this test
    # needs to exercise) -- everything else in get_grouping's real logic,
    # including the buggy max()/min() calls, still runs for real.
    import numpy as np

    import engines.visual as visual
    from vendor.uiguard import grouping

    class _OneClusterDBSCAN:
        def __init__(self, *a, **kw):
            pass

        def fit(self, X):
            self.labels_ = np.zeros(len(X), dtype=int)
            return self

    monkeypatch.setattr(grouping, "DBSCAN", _OneClusterDBSCAN)

    # y >= 50 and <= 5 words: required to survive get_text_button's filter
    # (grouping.py) so these items actually reach the clustering step.
    item_list = [
        {"category": "Button", "bbox": [10, 60, 110, 90], "text": "First button", "score": 1.0, "bg_lum": None, "fg_lum": None},
        {"category": "Button", "bbox": [10, 100, 110, 130], "text": "Second button", "score": 1.0, "bg_lum": None, "fg_lum": None},
    ]

    # The unpatched vendored function crashes on this input.
    with pytest.raises(ValueError):
        grouping.get_grouping(item_list, 200, 200)

    # Our wrapper degrades gracefully instead of losing everything.
    result = visual.rule_check.check_AM_FH({"existing": "results"}, item_list, 200, 200)
    assert result == {"existing": "results"}


def test_analyze_deduplicates_findings_at_the_same_bbox(monkeypatch):
    # Found analyzing nytimes.com: UIGuard's rules (e.g. examine_compound's
    # "image labeled ad") can independently match several distinct DOM
    # nodes that occupy the same/overlapping bounding box (e.g. an ad image
    # plus a separate "Ad" badge label rendered at the same position),
    # producing multiple Findings a user perceives as duplicates of one
    # single ad. analyze() should collapse same-pattern-type findings at
    # the same bbox into one, regardless of why the underlying vendored
    # detection produced repeats.
    import engines.visual as visual

    def fake_predict_type(**kwargs):
        bbox = [0, 100, 200, 130]
        return {
            "II-AM-DA": [
                {"bbox": bbox, "subType": "x", "flag": "flag_notClearAd"},
                {"bbox": bbox, "subType": "x", "flag": "flag_notClearAd"},
                {"bbox": bbox, "subType": "x", "flag": "flag_notClearAd"},
            ]
        }

    monkeypatch.setattr(visual.rule_check, "predict_type", fake_predict_type)

    cap = capture(str(FIXTURES_DIR / "precheck_consent.html"), is_fixture=True)
    findings = visual.analyze(cap)
    disguised_ad_findings = [f for f in findings if f.pattern_type == "disguised_ad"]
    assert len(disguised_ad_findings) == 1
