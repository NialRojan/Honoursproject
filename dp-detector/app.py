import base64
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from capture.capture import capture
from engines import structure, visual, linguistic, behavior
from fusion import fuse

app = FastAPI(title="Dark Pattern Detection Engine")

FIXTURES_DIR = Path(__file__).parent / "fixtures"
ENGINES = {
    "structure": structure.analyze,
    "visual": visual.analyze,
    "linguistic": linguistic.analyze,
    "behavior": behavior.analyze,
}


class AnalyzeRequest(BaseModel):
    url: str | None = None
    fixture_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/fixtures")
def list_fixtures():
    names = sorted(p.stem for p in FIXTURES_DIR.glob("*.html"))
    return {"fixtures": names}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if req.url and req.fixture_id:
        raise HTTPException(
            status_code=400,
            detail="provide either 'url' or 'fixture_id', not both",
        )
    if req.fixture_id:
        known_fixtures = {p.stem for p in FIXTURES_DIR.glob("*.html")}
        if req.fixture_id not in known_fixtures:
            raise HTTPException(status_code=404, detail=f"unknown fixture_id '{req.fixture_id}'")
        fixture_path = FIXTURES_DIR / f"{req.fixture_id}.html"
        source, is_fixture = str(fixture_path), True
    elif req.url:
        source, is_fixture = req.url, False
    else:
        raise HTTPException(status_code=400, detail="must provide either 'url' or 'fixture_id'")

    try:
        cap = capture(source, is_fixture=is_fixture)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"capture failed: {e}")

    all_findings = []
    engine_errors = {}
    for name, analyze_fn in ENGINES.items():
        try:
            all_findings.extend(analyze_fn(cap))
        except Exception as e:
            engine_errors[name] = str(e)

    result = fuse(all_findings)

    return {
        "risk_score": result["risk_score"],
        "findings": [asdict(f) for f in result["findings"]],
        "screenshot_base64": base64.b64encode(cap.screenshot_png).decode("ascii"),
        "elements": [asdict(e) for e in cap.elements],
        "engine_errors": engine_errors,
    }


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
