#!/bin/bash
# Runs INSIDE the container. Mounts: /imgs (ro) renamed screenshots, /out results,
# /pred this folder, /uied_out and /od_out persist stage 1-2 outputs on the host.
# ONLY3=1 skips stages 1-2 and re-runs the analysis from the saved outputs.
set -e
eval "$(conda shell.bash hook)"
cd /AidUI
OD=object_detection/object_detection_frcnn_mscoco_boilerplate
rm -rf input tmp_input UIED/data/input
mkdir -p input UIED/data/input $OD/inference_data/test
cp /imgs/*.* input/

if [ "$ONLY3" != "1" ]; then
  find /uied_out /od_out -mindepth 1 -delete
  rm -f $OD/inference_data/test/*
  cp input/*.* $OD/inference_data/test/
  echo "== stage 1: visual cue (icon) detection"
  conda activate dl_dp_obj_det_env
  (cd $OD && python object_detection_frcnn_mscoco_inference.py)
  echo "== stage 2: UIED text/segment extraction (batches of 15)"
  conda activate dp_uied3
  mkdir tmp_input && cp input/*.* tmp_input/
  cd tmp_input
  while [ -n "$(ls -A . 2>/dev/null)" ]; do
    ls | head -15 | xargs -I{} mv {} ../UIED/data/input/
    (cd ../UIED && python run_uied.py)
    rm -f ../UIED/data/input/*
  done
  cd ..
fi

cp input/*.* UIED/data/input/
echo "== stage 3: DP analysis + resolve"
conda activate dl_dp_obj_det_env
PYTHONPATH=/AidUI python /pred/predict.py
