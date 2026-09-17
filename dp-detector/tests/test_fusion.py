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

def test_many_findings_from_one_engine_do_not_exceed_its_weight():
    # Regression guard: a single engine firing many times (e.g. linguistic
    # flagging dozens of text blocks on a long page) must not be able to
    # dominate the score on its own -- its contribution is capped at its
    # own weight (25), same as if it had fired once at top confidence.
    findings = [
        Finding(engine="linguistic", pattern_type="manipulative_text", confidence=0.9, evidence="x")
        for _ in range(50)
    ]
    result = fuse(findings)
    assert result["risk_score"] == 22  # round(25 * 0.9) == round(22.5) == 22 (banker's rounding)

def test_score_reflects_highest_confidence_per_engine_not_volume():
    findings = [
        Finding(engine="linguistic", pattern_type="a", confidence=0.3, evidence="x"),
        Finding(engine="linguistic", pattern_type="b", confidence=0.9, evidence="x"),
    ]
    result = fuse(findings)
    assert result["risk_score"] == 22  # round(25 * 0.9), the max, not sum
