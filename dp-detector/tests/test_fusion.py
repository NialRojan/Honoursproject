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
