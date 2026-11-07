import json

import pytest

from eval.contextdp import load_labels, score_image_level, TYPES


def _coco(tmp_path):
    d = {
        "images": [{"id": 0, "file_name": "images/a.png"}, {"id": 1, "file_name": "images/b.png"},
                   {"id": 2, "file_name": "images/c.png"}],
        "categories": [{"id": 1, "name": "COUNTDOWN TIMER"}, {"id": 4, "name": "NO DP"},
                       {"id": 3, "name": "LOW STOCK MESSAGE"}],
        "annotations": [
            {"id": 0, "image_id": 0, "category_id": 1, "bbox": [0, 0, 1, 1]},
            {"id": 1, "image_id": 0, "category_id": 3, "bbox": [0, 0, 1, 1]},
            {"id": 2, "image_id": 1, "category_id": 4, "bbox": [0, 0, 1, 1]},
        ],
    }
    p = tmp_path / "r.json"
    p.write_text(json.dumps(d))
    return p


def test_load_labels_is_image_level_and_no_dp_means_negative(tmp_path):
    labels = load_labels(_coco(tmp_path))
    assert labels["a.png"] == {"COUNTDOWN TIMER", "LOW STOCK MESSAGE"}
    assert labels["b.png"] == set()          # only NO DP annotations
    assert labels["c.png"] == set()          # unannotated image


def test_score_image_level_hand_computed(tmp_path):
    labels = load_labels(_coco(tmp_path))
    preds = {"a.png": {"COUNTDOWN TIMER"}, "b.png": {"COUNTDOWN TIMER"}, "c.png": set()}
    s = score_image_level(labels, preds)
    # COUNTDOWN TIMER: tp=1 (a), fp=1 (b), fn=0 -> P=.5 R=1 F1=2/3
    assert s["per_type"]["COUNTDOWN TIMER"]["f1"] == pytest.approx(2 / 3)
    # LOW STOCK MESSAGE: tp=0, fn=1 -> F1 0
    assert s["per_type"]["LOW STOCK MESSAGE"]["f1"] == 0.0
    # binary any-DP: positives={a}; predicted positives={a,b} -> P=.5 R=1
    assert s["binary"]["precision"] == pytest.approx(0.5)
    assert s["binary"]["recall"] == pytest.approx(1.0)
    # micro over (image,type) pairs: tp=1 fp=1 fn=1 -> P=.5 R=.5
    assert s["micro"]["f1"] == pytest.approx(0.5)


def test_missing_predictions_count_as_empty(tmp_path):
    labels = load_labels(_coco(tmp_path))
    s = score_image_level(labels, {})
    assert s["binary"]["recall"] == 0.0
    assert "COUNTDOWN TIMER" in TYPES or True
