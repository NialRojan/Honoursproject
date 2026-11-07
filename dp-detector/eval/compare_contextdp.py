"""Side-by-side ContextDP-web comparison of every baseline + trivial references."""
import json
from pathlib import Path

import numpy as np

from eval.contextdp import TYPES, load_labels, score_image_level
from eval.metrics import bootstrap_ci, prf
from eval.stats import mcnemar_exact

R = Path("results")
labels = load_labels()
names = sorted(labels)
y = np.array([int(bool(labels[n])) for n in names])
sources = {
    "AidUI (open OCR)": {k: set(v) for k, v in json.load(open(R / "aidui_predictions.json")).items()},
    "Qwen2.5-VL-7B zero-shot": {k: set(v) for k, v in json.load(open(R / "vlm_qwen2.5vl_7b_predictions.json")).items()},
    "always 'pattern present' (trivial)": {n: set(TYPES) for n in names},
    "never (trivial)": {n: set() for n in names},
}
rows, binpreds = [], {}
for m, p in sources.items():
    s = score_image_level(labels, p)
    b = np.array([int(bool(p.get(n))) for n in names])
    binpreds[m] = b
    lo, hi = bootstrap_ci(y, b, lambda a, c: prf(a, c)[2], np.arange(len(y)), n=2000, seed=0)
    rows.append((m, s["binary"]["precision"], s["binary"]["recall"], s["binary"]["f1"], lo, hi,
                 s["micro"]["f1"], s["macro_f1"], int(b.sum())))
out = ["| Method | Binary P | Binary R | Binary F1 [95% CI] | Micro F1 | Macro F1 | Images flagged |",
       "|---|---|---|---|---|---|---|"]
for m, bp, br, bf, lo, hi, mi, ma, k in rows:
    out.append(f"| {m} | {bp:.2f} | {br:.2f} | {bf:.3f} [{lo:.2f}, {hi:.2f}] | {mi:.3f} | {ma:.3f} | {k}/162 |")
a, v = "AidUI (open OCR)", "Qwen2.5-VL-7B zero-shot"
n01, n10, p = mcnemar_exact(y, binpreds[a], binpreds[v])
out.append(f"\nMcNemar (binary decisions) AidUI vs Qwen: AidUI-only-right={n01}, Qwen-only-right={n10}, p={p:.3f}")
neg = y == 0
for m in (a, v):
    out.append(f"False-positive rate on the {int(neg.sum())} pattern-free screenshots, {m}: "
               f"{binpreds[m][neg].mean():.2f}")
text = "\n".join(out)
print(text)
(R / "contextdp_comparison.md").write_text(text + "\n")
