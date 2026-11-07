"""Zero-shot open-weight VLM baseline on ContextDP-web.

python -m eval.vlm_baseline --model qwen2.5vl:7b [--limit N]

Free and reproducible: a local Ollama model, temperature 0, fixed seed, constrained
JSON output. The prompt lists the 7 ContextDP pattern types with short definitions
(AidUI taxonomy). Predictions are cached per image so a run can resume.
"""
import argparse
import base64
import io
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image

from eval.contextdp import DATA_DIR, TYPES, load_labels, score_image_level

OLLAMA = "http://localhost:11434/api/chat"
MAX_W = 1280

DEFINITIONS = {
    "ACTIVITY MESSAGE": "a notice about other shoppers' activity, e.g. '12 people are viewing this', 'John just bought'",
    "COUNTDOWN TIMER": "a visible timer counting down to create urgency",
    "LIMITED TIME MESSAGE": "text claiming an offer or price ends soon, e.g. 'Sale ends tonight', 'Limited time offer'",
    "LOW STOCK MESSAGE": "text claiming scarce stock, e.g. 'Only 3 left', 'Almost sold out'",
    "HIGH DEMAND MESSAGE": "text claiming the product is popular or selling fast, e.g. 'In high demand', 'Selling fast'",
    "DEFAULT CHOICE": "an option pre-selected or pre-checked for the user, e.g. a ticked add-on or newsletter box",
    "ATTENTION DISTRACTION": "one option made visually dominant to steer the user, e.g. a bright 'Accept' next to a faint 'No thanks'",
}


def prompt() -> str:
    lines = "\n".join(f"- {t}: {DEFINITIONS[t]}" for t in TYPES)
    return ("You are auditing a web page screenshot for dark patterns (deceptive UI). "
            "Decide which of the following pattern types are visibly present. Only report a "
            "type if you can see clear evidence; if none apply, return an empty list.\n"
            f"{lines}\n"
            'Answer as JSON: {"patterns": [<type names exactly as written above>]}')


def encode(path: Path) -> str:
    im = Image.open(path).convert("RGB")
    if im.width > MAX_W:
        im = im.resize((MAX_W, round(im.height * MAX_W / im.width)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def ask(model: str, image_b64: str, seed: int = 42) -> list:
    body = {
        "model": model, "stream": False,
        "messages": [{"role": "user", "content": prompt(), "images": [image_b64]}],
        "format": {"type": "object", "properties": {"patterns": {
            "type": "array", "items": {"type": "string", "enum": TYPES}}},
            "required": ["patterns"]},
        "options": {"temperature": 0, "seed": seed, "num_ctx": 8192},
    }
    req = urllib.request.Request(OLLAMA, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=600))
    return json.loads(out["message"]["content"]).get("patterns", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5vl:7b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    labels = load_labels()
    names = sorted(labels)[: a.limit or None]
    tag = a.model.replace(":", "_").replace("/", "_")
    cache_path = Path(a.out) / f"vlm_{tag}_predictions.json"
    Path(a.out).mkdir(exist_ok=True)
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    t0 = time.time()
    for i, n in enumerate(names):
        if n in cache:
            continue
        cache[n] = ask(a.model, encode(DATA_DIR / "images" / n))
        json.dump(cache, open(cache_path, "w"), indent=0)
        print(f"{i + 1}/{len(names)} {n[:40]:40s} -> {cache[n]}  ({time.time() - t0:.0f}s)", flush=True)

    sub = {n: labels[n] for n in names}
    preds = {n: set(cache[n]) for n in names}
    s = score_image_level(sub, preds)
    s["model"] = a.model
    json.dump(s, open(Path(a.out) / f"vlm_{tag}_summary.json", "w"), indent=1)
    print(f"\n{a.model} on ContextDP-web ({s['n_images']} images, {s['n_positive']} with a pattern)")
    for t, v in sorted(s["per_type"].items()):
        print(f"  {t:22s} P={v['precision']:.2f} R={v['recall']:.2f} F1={v['f1']:.2f} (n={v['support']})")
    print(f"  micro F1={s['micro']['f1']:.3f}  macro F1={s['macro_f1']:.3f}  "
          f"binary P={s['binary']['precision']:.2f} R={s['binary']['recall']:.2f} F1={s['binary']['f1']:.3f}")


if __name__ == "__main__":
    main()
