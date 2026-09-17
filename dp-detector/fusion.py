ENGINE_WEIGHT = {
    "structure": 25,
    "visual": 25,
    "linguistic": 25,
    "behavior": 25,
}


def fuse(findings: list) -> dict:
    # Each engine's contribution is capped at its own weight, scaled by its
    # single highest-confidence finding -- not summed across every finding
    # it produces. Otherwise an engine that fires many times on one page
    # (e.g. linguistic flagging dozens of text blocks) can dominate the
    # score on volume alone, saturating it at 100 regardless of severity.
    max_confidence_per_engine = {}
    for f in findings:
        current = max_confidence_per_engine.get(f.engine, 0.0)
        if f.confidence > current:
            max_confidence_per_engine[f.engine] = f.confidence

    score = sum(
        confidence * ENGINE_WEIGHT.get(engine, 10)
        for engine, confidence in max_confidence_per_engine.items()
    )
    return {"risk_score": min(100, round(score)), "findings": findings}
