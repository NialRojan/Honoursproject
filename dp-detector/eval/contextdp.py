"""ContextDP-web (AidUI, ICSE'23) loader and image-level scoring.

162 web screenshots, COCO-format labels. A screenshot is *positive* for a pattern
type if it carries at least one annotation of that type; "NO DP" boxes and
unannotated screenshots are negatives. All baselines are scored identically here:
image-level presence per type, micro/macro over types, plus binary any-pattern.
"""
import json
from pathlib import Path

from eval.metrics import prf

DATA_DIR = Path(__file__).resolve().parent / "data" / "contextdp_web"
TYPES = ["ACTIVITY MESSAGE", "COUNTDOWN TIMER", "LIMITED TIME MESSAGE", "LOW STOCK MESSAGE",
         "HIGH DEMAND MESSAGE", "DEFAULT CHOICE", "ATTENTION DISTRACTION"]


def load_labels(path=DATA_DIR / "result.json") -> dict:
    d = json.load(open(path))
    cat = {c["id"]: c["name"] for c in d["categories"]}
    labels = {i["file_name"].split("/", 1)[1]: set() for i in d["images"]}
    by_id = {i["id"]: i["file_name"].split("/", 1)[1] for i in d["images"]}
    for a in d["annotations"]:
        name = cat[a["category_id"]]
        if name != "NO DP":
            labels[by_id[a["image_id"]]].add(name)
    return labels


def score_image_level(labels: dict, preds: dict) -> dict:
    imgs = sorted(labels)
    types = sorted({t for s in labels.values() for t in s} | {t for s in preds.values() for t in s})
    per_type = {}
    tp_all = fp_all = fn_all = 0
    for t in types:
        y = [int(t in labels[i]) for i in imgs]
        p = [int(t in preds.get(i, set())) for i in imgs]
        pr, rc, f1 = prf(y, p)
        per_type[t] = {"precision": pr, "recall": rc, "f1": f1, "support": sum(y)}
        tp_all += sum(1 for a, b in zip(y, p) if a and b)
        fp_all += sum(1 for a, b in zip(y, p) if not a and b)
        fn_all += sum(1 for a, b in zip(y, p) if a and not b)
    mp = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
    mr = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
    mf = 2 * mp * mr / (mp + mr) if mp + mr else 0.0
    yb = [int(bool(labels[i])) for i in imgs]
    pb = [int(bool(preds.get(i, set()))) for i in imgs]
    bp, br, bf = prf(yb, pb)
    supported = [v["f1"] for v in per_type.values() if v["support"] > 0]
    return {"per_type": per_type,
            "micro": {"precision": mp, "recall": mr, "f1": mf},
            "macro_f1": sum(supported) / len(supported) if supported else 0.0,
            "binary": {"precision": bp, "recall": br, "f1": bf},
            "n_images": len(imgs), "n_positive": sum(yb)}
