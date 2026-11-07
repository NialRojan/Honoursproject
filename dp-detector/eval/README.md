# Evaluation harness (Phases 0–1)

Reproduces every baseline number. Run from `dp-detector/` with the project venv.

## 1. Text baselines on the Yada data (random vs grouped splits)

    python -m eval.run --methods lr bert        # ~6 min on an RTX 4060

- Data: `training/data/linguistic_dataset.tsv` (Yada et al. 2022, 2,397 texts).
- Splits: 5-fold, seed 42. *Grouped* folds never split texts that share a `page_id`
  or a normalised template (digits -> `#`, punctuation stripped).
- Temperature is fit on a validation slice cut from each training fold only.
- Writes `results/predictions.csv` and `results/summary.csv`
  (accuracy, P/R/F1 with group-bootstrap CI, AUROC, FPR, ECE, Brier) plus paired McNemar.

## 2. ContextDP-web baselines (162 web screenshots, 7 pattern types)

Data: AidUI's ContextDP-web (github.com/SageSELab/AidUI, `evaluation/evaluation_dataset/web`).
`result.json` is committed; download the images into `eval/data/contextdp_web/images/`
(raw.githubusercontent.com/SageSELab/AidUI/master/evaluation/evaluation_dataset/web/<file_name>).

    # Zero-shot open VLM (needs: ollama pull qwen2.5vl:7b)
    python -m eval.vlm_baseline --model qwen2.5vl:7b
    # DPGuard-style prompt (Appendix D of Shi et al. 2025) on the same VLM
    python -m eval.dpguard_style
    # AidUI in Docker (see below), then score
    docker pull smhasanmansur/aidui-img && docker build -t aidui-open-ocr eval/aidui
    eval/aidui/run.sh            # ONLY3=1 eval/aidui/run.sh re-runs just the analysis stage
    python -m eval.score_aidui
    # everything side by side
    python -m eval.compare_contextdp

### Deviations from the published methods (state these in any write-up)
- **AidUI uses Tesseract, not Google Cloud Vision.** The original UIED stage calls the
  Google Vision API with a key that no longer works; `eval/aidui/ocr_open.py` is a
  drop-in replacement. Scores may differ from the original OCR.
- **The "DPGuard" row is a prompt re-implementation, not DPGuard.** It uses the published
  final prompt with a local open VLM and omits DPGuard's ResNet101 pre-filter and its
  commercial MLLM. DPGuard's 21 categories have no class for ContextDP's urgency/scarcity
  types, so only default choice and attention distraction are comparable per type.
- Scoring is image-level (is type T present in the screenshot), not AidUI's box-level IoU.
  AidUI predictions of types ContextDP-web does not label (nagging, disguised ads) are ignored.
- UIGuard is not run: it is built for mobile screens (Rico) and overlaps ContextDP on 2 types.

## Memory note
AidUI's container and the 7B VLM must not run at the same time on a 14 GB machine
(the kernel OOM-killed Ollama once). Run them one after the other.

## Tests
    python -m pytest tests/test_eval_*.py
