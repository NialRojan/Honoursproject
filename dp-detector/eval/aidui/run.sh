#!/bin/bash
# Host wrapper. Usage: eval/aidui/run.sh [ONLY3=1]
P="$(cd "$(dirname "$0")/../.." && pwd)"
docker run --rm --entrypoint bash -e ONLY3="${ONLY3:-0}" \
  -v "$P/results/aidui_stage/images:/imgs:ro" -v "$P/results/aidui_out:/out" \
  -v "$P/eval/aidui:/pred:ro" \
  -v "$P/results/aidui_stage/uied_out:/AidUI/UIED/data/output" \
  -v "$P/results/aidui_stage/od_out:/AidUI/object_detection/object_detection_frcnn_mscoco_boilerplate/inference_output" \
  -v "$P/results/aidui_stage/uied_out:/uied_out" -v "$P/results/aidui_stage/od_out:/od_out" \
  aidui-open-ocr /pred/run_batch.sh
