# Dark Pattern Detection Engine — Phase 1 Prototype

Implements the four-engine architecture (Structure, Visual, Linguistic,
Behavior) from the design spec at
`../docs/superpowers/specs/2026-09-16-multimodal-dark-pattern-engine-design.md`.

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m playwright install chromium

## Train the Linguistic and Behavior engines (one-time, before first use)

    python -m training.train_linguistic
    python -m training.prepare_behavior_dataset --sample-size 300
    python -m training.train_behavior

The Structure and Visual engines need no training — they're rule-based.

**Note:** The Behavior engine dataset preparation and training (Tasks 9–10) depend on external datasets hosted at Zenodo. If you encounter issues with `python -m training.prepare_behavior_dataset`, Zenodo may be temporarily unavailable. Once the external dependency is resolved, re-run the training commands to complete the Behavior engine setup.

## Run the tests

    pytest -v

## Run the server

    uvicorn app:app --reload

Open http://localhost:8000/, pick a fixture from the dropdown (or paste a
real URL) and click Analyze.

## Manual verification checklist

- [ ] `disguised_button` fixture flags a visual finding (false hierarchy / interface inference)
- [ ] `precheck_consent` fixture flags both a structure finding (preselected_consent) and a visual finding (preselection)
- [ ] `confirm_shaming` fixture flags a linguistic finding (manipulative_text)
- [ ] `deep_cancel_flow` fixture flags a behavior finding (roach_motel); `easy_cancel_flow` does not
- [ ] A real public URL (e.g. a known cookie-consent-heavy news site) analyzes without crashing and returns a risk score
