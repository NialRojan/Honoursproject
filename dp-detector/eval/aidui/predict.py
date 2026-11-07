# Runs INSIDE the AidUI container (env dl_dp_obj_det_env). Mirrors AidUI's main.py
# decision pipeline (threshold 0.75) but skips its built-in evaluation and writes the
# predicted labels per image so the shared scorer can be used.
import glob
import json
import warnings

import text_analysis.pattern_matching.matching as pattern_matching
import visual_analysis.histogram_analysis.histogram_analysis as histogram_analysis
import spatial_analysis.proximity_analysis.proximity_analysis as proximity_analysis
import spatial_analysis.size_analysis.size_analysis as size_analysis
import object_detection.object_detection as object_detection
import dp_resolver.resolver as resolver

warnings.filterwarnings("ignore")
THRESHOLD = 0.75

img_files = sorted(glob.glob("UIED/data/input/*.*"))
ocr_files = sorted(glob.glob("UIED/data/output/ocr/*.json"))
assert len(img_files) == len(ocr_files), (len(img_files), len(ocr_files))

out = {}
for img, ocr in zip(img_files, ocr_files):
    name = img.split("/")[-1]
    assert name.rsplit(".", 1)[0] == ocr.split("/")[-1].rsplit(".", 1)[0], (img, ocr)
    res = pattern_matching.match_patterns(ocr)
    res = histogram_analysis.analyze_histogram(res, img)
    res = proximity_analysis.analyze_proximity(res, img)
    res = size_analysis.analyze_size(res)
    od = object_detection.get_object_detection_result(ocr)
    pred = resolver.resolve_dp({"analysis_result": res, "object_detection_result": od}, THRESHOLD)
    out[name] = [l for l in pred["labels"] if l != "NO DP"]
    print(name, out[name], flush=True)

json.dump(out, open("/out/aidui_raw_predictions.json", "w"), indent=0)
