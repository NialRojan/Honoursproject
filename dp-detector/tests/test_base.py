from engines.base import Finding


def test_finding_defaults():
    f = Finding(engine="structure", pattern_type="preselected_consent", confidence=0.9, evidence="checkbox pre-checked")
    assert f.element_ref is None
    assert f.confidence == 0.9


def test_finding_with_element_ref():
    f = Finding(engine="visual", pattern_type="disguised_ad", confidence=0.7, evidence="ad styled as content", element_ref="#ad-1")
    assert f.element_ref == "#ad-1"
