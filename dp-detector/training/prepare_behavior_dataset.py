import argparse
import json
import zipfile
from pathlib import Path

import easyocr
import requests
from PIL import Image

DATA_DIR = Path(__file__).parent / "data"
DARK_ZIP_URL = "https://zenodo.org/api/records/13268006/files/AppRay-Dark.zip/content"
LIGHT_ZIP_URL = "https://zenodo.org/api/records/13268006/files/AppRay-Light.zip/content"
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


def _is_junk(path: Path) -> bool:
    """macOS zip archives carry __MACOSX/._* AppleDouble resource-fork
    files that share the real files' extensions but aren't valid images."""
    return "__MACOSX" in path.parts or path.name.startswith("._")


def _find_annotations(dark_dir: Path) -> Path:
    for candidate in dark_dir.rglob("all_images_patterns.json"):
        if not _is_junk(candidate):
            return candidate
    raise FileNotFoundError(f"all_images_patterns.json not found under {dark_dir}")


def main(sample_size: int):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dark_zip = DATA_DIR / "AppRay-Dark.zip"
    light_zip = DATA_DIR / "AppRay-Light.zip"

    _download(DARK_ZIP_URL, dark_zip)
    _download(LIGHT_ZIP_URL, light_zip)

    dark_dir = DATA_DIR / "AppRay-Dark"
    light_dir = DATA_DIR / "AppRay-Light"
    _unzip(dark_zip, dark_dir)
    _unzip(light_zip, light_dir)

    # Real schema (confirmed against a live download, and matching the
    # Zenodo record's own description): all_images_patterns.json is a JSON
    # *array* bundled inside AppRay-Dark.zip (not a separate top-level
    # Zenodo file, and not a dict keyed by image name as originally
    # guessed). Each item:
    #   {"image": "<filename>.png", "image_size": {...}, "app_name": "...",
    #    "dark_pattern": [["<DP-type>", {"xmin","ymin","xmax","ymax"}, <is_duplicate>], ...]}
    annotations = json.loads(_find_annotations(dark_dir).read_text())
    reader = easyocr.Reader(["en"], gpu=False)  # GPU driver is in an ERR! state on this host

    rows = []
    dark_items = annotations[:sample_size]
    for item in dark_items:
        image_path = next(
            (p for p in dark_dir.rglob(item["image"]) if not _is_junk(p)), None
        )
        if image_path is None:
            continue
        img = Image.open(image_path).convert("RGB")
        for _dp_type, box, _is_duplicate in item.get("dark_pattern", []):
            try:
                crop = img.crop((
                    int(box["xmin"]), int(box["ymin"]),
                    int(box["xmax"]), int(box["ymax"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue
            text = " ".join(reader.readtext(np_from_pil(crop), detail=0))
            if text.strip():
                rows.append((text.strip(), 1))

    light_candidates = [
        p for p in list(light_dir.rglob("*.png")) + list(light_dir.rglob("*.jpg"))
        if not _is_junk(p)
    ]
    for image_path in light_candidates[:sample_size]:
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
