import argparse
import json
import zipfile
from pathlib import Path

import easyocr
import requests
from PIL import Image

DATA_DIR = Path(__file__).parent / "data"
DARK_ZIP_URL = "https://zenodo.org/records/13268006/files/AppRay-Dark.zip?download=1"
LIGHT_ZIP_URL = "https://zenodo.org/records/13268006/files/AppRay-Light.zip?download=1"
ANNOTATIONS_URL = "https://zenodo.org/records/13268006/files/all_images_patterns.json?download=1"
OUTPUT_TSV = DATA_DIR / "behavior_dataset.tsv"


def _download(url: str, dest: Path):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def _unzip(zip_path: Path, dest_dir: Path):
    if dest_dir.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)


def main(sample_size: int):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dark_zip = DATA_DIR / "AppRay-Dark.zip"
    light_zip = DATA_DIR / "AppRay-Light.zip"
    annotations_path = DATA_DIR / "all_images_patterns.json"

    _download(DARK_ZIP_URL, dark_zip)
    _download(LIGHT_ZIP_URL, light_zip)
    _download(ANNOTATIONS_URL, annotations_path)

    dark_dir = DATA_DIR / "AppRay-Dark"
    light_dir = DATA_DIR / "AppRay-Light"
    _unzip(dark_zip, dark_dir)
    _unzip(light_zip, light_dir)

    # NOTE: the JSON schema below (top-level dict keyed by image name, each
    # value having a "patterns" list of {"bbox": {"xmin", "ymin", "xmax",
    # "ymax"}}) was never verified against a live download of
    # all_images_patterns.json during planning -- only Zenodo's file listing
    # was checked, not its contents (Zenodo was unreachable at plan-writing
    # time and still is as of this fix wave). If the actual key/field names
    # differ, inspect the downloaded JSON directly (see the README / Task 9
    # notes) and adjust the key names in this function accordingly before
    # re-running.
    annotations = json.loads(annotations_path.read_text())
    reader = easyocr.Reader(["en"], gpu=True)

    rows = []
    dark_items = list(annotations.items())[:sample_size]
    for image_name, meta in dark_items:
        image_path = next(dark_dir.rglob(image_name), None)
        if image_path is None:
            continue
        img = Image.open(image_path).convert("RGB")
        for pattern in meta.get("patterns", []):
            box = pattern.get("bbox") or pattern
            try:
                crop = img.crop((box["xmin"], box["ymin"], box["xmax"], box["ymax"]))
            except (KeyError, TypeError):
                continue
            text = " ".join(reader.readtext(np_from_pil(crop), detail=0))
            if text.strip():
                rows.append((text.strip(), 1))

    light_images = list(light_dir.rglob("*.png"))[:sample_size] + list(light_dir.rglob("*.jpg"))[:sample_size]
    for image_path in light_images:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        crop = img.crop((0, 0, w, min(200, h)))
        text = " ".join(reader.readtext(np_from_pil(crop), detail=0))
        if text.strip():
            rows.append((text.strip(), 0))

    with open(OUTPUT_TSV, "w", encoding="utf-8") as f:
        f.write("text\tlabel\n")
        for text, label in rows:
            f.write(f"{text.replace(chr(9), ' ')}\t{label}\n")

    print(f"Wrote {len(rows)} rows to {OUTPUT_TSV}")


def np_from_pil(img):
    import numpy as np
    return np.array(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=300,
                         help="Max number of labeled images to process per class (default 300, keeps first Phase-1 run fast; raise for a fuller dataset).")
    args = parser.parse_args()
    main(args.sample_size)
