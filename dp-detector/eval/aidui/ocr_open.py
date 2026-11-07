# Drop-in replacement for AidUI's UIED/detect_text/ocr.py (Python 3.5 compatible).
# The original posts every screenshot to the Google Cloud Vision API. This version uses
# the local Tesseract CLI and returns word boxes in Google's textAnnotations format.
import csv
import io
import subprocess


def ocr_detection_google(imgpath):
    out = subprocess.run(["tesseract", imgpath, "stdout", "--psm", "11", "tsv"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    rows = csv.DictReader(io.StringIO(out.decode("utf-8", "ignore")), delimiter="\t",
                          quoting=csv.QUOTE_NONE)
    words = []
    for r in rows:
        txt = (r.get("text") or "").strip()
        try:
            conf = float(r.get("conf", -1))
        except ValueError:
            continue
        if not txt or conf < 30:
            continue
        l, t = int(r["left"]), int(r["top"])
        rgt, b = l + int(r["width"]), t + int(r["height"])
        words.append({"description": txt, "boundingPoly": {"vertices": [
            {"x": l, "y": t}, {"x": rgt, "y": t}, {"x": rgt, "y": b}, {"x": l, "y": b}]}})
    return words or None
