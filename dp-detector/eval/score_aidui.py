"""Score AidUI (open-OCR) predictions on ContextDP-web with the shared scorer.

Predictions are restricted to the 7 pattern types the ContextDP-web screenshots are
labelled for (AidUI can emit 10 types, e.g. NAGGING, that these images carry no labels
for; counting them would penalise it for patterns nobody annotated).
"""
import json
from pathlib import Path

from eval.contextdp import TYPES, load_labels, score_image_level

R = Path("results")
raw = json.load(open(R / "aidui_out" / "aidui_raw_predictions.json"))
name_map = json.load(open(R / "aidui_stage" / "name_map.json"))
preds = {name_map[k]: {t for t in v if t in TYPES} for k, v in raw.items()}
dropped = sorted({t for v in raw.values() for t in v if t not in TYPES})
labels = load_labels()
s = score_image_level(labels, preds)
s["model"] = "AidUI (open OCR: Tesseract), threshold 0.75"
s["types_not_scored"] = dropped
json.dump(s, open(R / "aidui_summary.json", "w"), indent=1)
json.dump({k: sorted(v) for k, v in preds.items()}, open(R / "aidui_predictions.json", "w"), indent=0)
print(s["model"], f"({s['n_images']} images, {s['n_positive']} with a pattern); not scored: {dropped}")
for t, v in sorted(s["per_type"].items()):
    print(f"  {t:22s} P={v['precision']:.2f} R={v['recall']:.2f} F1={v['f1']:.2f} (n={v['support']})")
print(f"  micro F1={s['micro']['f1']:.3f}  macro F1={s['macro_f1']:.3f}  "
      f"binary P={s['binary']['precision']:.2f} R={s['binary']['recall']:.2f} F1={s['binary']['f1']:.3f}")
