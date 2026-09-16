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
