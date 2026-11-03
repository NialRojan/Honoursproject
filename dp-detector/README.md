# Dark Pattern Detection Engine — Phase 1 Prototype

Implements the four-engine architecture (Structure, Visual, Linguistic,
Behavior).

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

The real `all_images_patterns.json` schema (verified against a live download once Zenodo recovered) differs from what was assumed while it was down: it's a JSON array bundled inside `AppRay-Dark.zip` (not a separate top-level Zenodo file, and not a dict keyed by image name), where each item has `image`/`image_size`/`app_name`/`dark_pattern` fields — see `training/prepare_behavior_dataset.py` for the exact structure.

## Run the tests

    pytest -v

## Run the server

    uvicorn app:app --reload

Open http://localhost:8000/, pick a fixture from the dropdown (or paste a
real URL) and click Analyze.

## Manual verification checklist

**Note:** When the Behavior engine's model isn't trained, `/analyze` still returns a 200 response — the engine's failure is surfaced visibly as a red-bordered entry in `engine_errors` (both in the API response and in the frontend), not silently dropped.

- [x] `disguised_button` fixture flags a visual finding (false hierarchy / interface inference)
- [x] `precheck_consent` fixture flags both a structure finding (preselected_consent) and a visual finding (preselection)
- [x] `confirm_shaming` fixture flags a linguistic finding (manipulative_text)
- [x] `deep_cancel_flow` fixture flags a behavior finding (roach_motel); `easy_cancel_flow` does not
- [x] A real public URL analyzes without crashing and returns a risk score

**Caution:** when analyzing a live URL (as opposed to a bundled fixture), `_walk_click_trace` (in `capture/capture.py`) actively clicks any cancel/unsubscribe-like controls it finds on the page, up to `MAX_CLICK_DEPTH` times, in order to build the Behavior engine's click trace. Do not point it at a logged-in or otherwise stateful page you don't want to trigger real actions (e.g. an actual account cancellation) on.
