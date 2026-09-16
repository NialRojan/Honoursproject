# Multimodal Dark Pattern Detection Engine — Phase 1 Design

## Context

This project implements the four-engine architecture proposed in the user's
own paper, "Technical Countermeasures for Deceptive User Interfaces: A
Comprehensive Detection and Mitigation Framework" (Rojan, Ambarkar,
Parshionkar; IJERT Vol. 15 Issue 05, 2026). The paper proposes a
Code/Structure, Visual, Linguistic, and Behavior engine, fused into a
real-time risk score, but stops at the proposal stage — no implementation
or evaluation exists yet.

This work is split into two phases:

- **Phase 1 (this spec)**: build a working prototype of the four-engine
  architecture end to end, using real, license-clear, publicly available
  models identified from the literature (both the paper's own references
  and a broader survey of current work).
- **Phase 2 (separate, future spec, not designed here)**: dataset
  aggregation toward the paper's proposed scale, adversarial-robustness
  hardening, partial on-device distillation, an agent-facing risk-score
  API, and deployment tooling. Federated learning and interdisciplinary
  regulatory alignment are explicitly excluded from Phase 2 as
  implementation deliverables — the former requires a real distributed
  user base to be meaningful (not simulate-able in a way that proves
  anything), the latter is a policy/collaboration activity, not code.

## Goal

A locally-run tool: point it at a URL (or a saved sample page), and it
returns a dark-pattern report — a risk score, a list of findings, and a
screenshot with flagged elements highlighted — via a simple web frontend.

## Models used per engine (Phase 1)

Selected from a comparison across the paper's own 36 references plus a
broader current-literature search, filtered to what's actually licensed
for reuse (see decision log below for what was excluded and why):

| Engine | Model | Source | License |
|---|---|---|---|
| Structure | Pattern Highlighter's detection rules (DOM + Accessibility-Tree feature checks), ported to Python | github.com/Dapde/Pattern-Highlighter | MIT |
| Visual | UIGuard (property extraction + knowledge-driven checker) | github.com/chenjshnn/UIST23-UIGuard | Apache-2.0 |
| Linguistic | Fine-tuned BERT baseline | github.com/yamanalab/ec-darkpattern | Apache-2.0 |
| Behavior | AppRay's contrastive multi-label classifier (exploration step replaced — see below) | github.com/chenjshnn/AppRay | MIT |

**Decision log (why these, not the alternatives found in the broader search):**
- YOLOv12x (mAP@50 92.8%) and the UC Berkeley Deceptive Pattern Detector
  (MiniLM + XGBoost) are technically stronger/newer than UIGuard and
  Pattern Highlighter respectively, but both have **no license** on their
  GitHub repos (all-rights-reserved by default) — not legally reusable.
  Revisit if the authors add a license.
- DPAgent-in-the-Middle (micro F1 0.816, also repairs detected patterns)
  is the strongest Behavior-engine result found, but has no released code
  (very recent, June 2026 paper).
- DPDGPT (F1 0.88) and DPGuard (WWW'25) both depend on commercial/proprietary
  multimodal LLMs rather than a downloadable model, or have no license —
  excluded for the same reasons.
- **AppRay's exploration step normally calls an LLM** to navigate the app.
  Phase 1 uses AppRay's trained classifier only, fed by our own scripted
  (non-LLM) shallow click-trace capture — this avoids requiring an API key
  or incurring per-run LLM cost, at the expense of AppRay's smarter
  navigation. This is a deliberate scope cut, not an oversight.

## Architecture

Single Python (FastAPI) application, one process, so a local GPU is
directly available to the Visual/Linguistic models without cross-process
overhead. No separate frontend build step.

```
dp-detector/
├── app.py                 # FastAPI: POST /analyze, GET /fixtures, serves static/
├── capture/capture.py      # Playwright: URL or fixture -> DOM+screenshot+text+trace
├── engines/
│   ├── base.py              # Finding dataclass + shared Engine interface
│   ├── structure.py          # Pattern-Highlighter-derived DOM/AX-tree rules
│   ├── visual.py               # UIGuard-derived property extraction + checker
│   ├── linguistic.py            # yamanalab BERT inference
│   └── behavior.py               # scripted click-trace -> AppRay classifier
├── fusion.py                # findings -> single 0-100 risk score
├── fixtures/                 # self-authored sample pages w/ known patterns
├── static/                    # index.html, app.js, style.css
├── tests/
└── requirements.txt
```

## Components

**Capture** (`capture/capture.py`): headless Chromium via Playwright.
Same code path for a live URL and a saved fixture — fixture HTML is loaded
into a real page via Playwright rather than replayed as static JSON, so
both modes exercise identical downstream logic. Output is one `Capture`
object:
```
Capture:
  url: str
  dom_html: str
  screenshot_png: bytes
  rendered_text: str
  elements: list[{selector, bbox, tag, text, computed_style}]
  click_trace: list[{from_selector, to_selector, depth}]
```

**Engines** (`engines/*.py`): each engine exposes one function,
`analyze(capture: Capture) -> list[Finding]`, where
```
Finding:
  engine: str            # "structure" | "visual" | "linguistic" | "behavior"
  pattern_type: str       # e.g. "pre_checked_consent", "roach_motel"
  confidence: float        # 0.0-1.0
  evidence: str             # human-readable justification
  element_ref: str | None    # CSS selector of the flagged element, if applicable
```
- **Structure**: ports Pattern Highlighter's rule functions (pre-checked
  checkboxes, hidden cancel/unsubscribe links, required-field forced
  consent) from JS to Python, operating on `dom_html`/`elements`.
- **Visual**: adapts UIGuard's property-extraction + knowledge-driven
  checker to take our `screenshot_png` + `elements` as input instead of
  its own capture pipeline.
- **Linguistic**: tokenizes `rendered_text` per visible text block, runs
  the yamanalab BERT checkpoint, returns a Finding per flagged block.
- **Behavior**: analyzes `click_trace` for Roach-Motel-style asymmetry
  (e.g. 1-click signup vs N-click cancellation), then passes the
  candidate flow through AppRay's trained classifier for the final
  judgment.

**Fusion** (`fusion.py`): a deterministic weighted combination of all
Findings into one 0-100 risk score for the page. Explicitly *not* the
paper's proposed multimodal-transformer fusion layer — that requires
training data this prototype doesn't have. Findings pass through
unfiltered alongside the score, so the frontend can show exactly what
each engine independently found even where fusion logic is simple.

**API** (`app.py`):
- `POST /analyze` — body `{url}` or `{fixture_id}` → runs capture → all
  four engines → fusion → returns `{risk_score, findings[], screenshot_base64, elements[]}`.
- `GET /fixtures` — lists available saved fixtures for the frontend dropdown.

**Frontend** (`static/`): one page, vanilla HTML/JS (no build step).
Screenshot shown with absolutely-positioned overlay boxes per finding,
color-coded by engine; a findings list alongside; a risk-score badge at
the top. Clicking a finding highlights its box; hovering a box highlights
its finding.

## Fixtures

3-5 small, self-authored HTML pages reproducing known dark-pattern shapes
(pre-checked newsletter checkbox, fake countdown timer, confirm-shaming
cancel button, deep-nested cancellation flow) — not scraped from real
sites, to keep the automated test suite license-clean and stable. Live-URL
mode still works against any real site the user points it at manually,
outside the automated test suite.

## Error handling

- Capture failure (bad URL, timeout, page blocks headless browsers) →
  `/analyze` returns a clear error, no partial pipeline run.
- Each engine call is wrapped individually: if one throws, it contributes
  zero findings and the response includes an `engine_errors: {structure: "..."}`
  field for that engine, rather than crashing the whole request or silently
  treating a failure as "no patterns found."

## Testing

- Unit tests per engine: a fixture-derived `Capture` with a known expected
  `Finding`, written test-first (TDD) before each rule/inference call is
  wired up.
- One integration test running the full pipeline against a local fixture
  end-to-end, asserting the API returns a valid response shape and a
  risk score in [0, 100].
- No live-network calls in the automated test suite — live-URL mode is
  manually invoked only.

## Known implementation risk

The environment runs Python 3.14. Playwright and PyTorch wheel releases
sometimes lag brand-new Python versions. This will be checked at the start
of implementation; if incompatible, the project will use a pinned virtual
environment (e.g. Python 3.12) rather than blocking on it mid-build.
