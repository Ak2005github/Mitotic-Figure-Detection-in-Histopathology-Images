"""Fast unit tests for the core logic (no GPU, no downloads): python -m pytest tests"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mitosis.data.io import crop_centered  # noqa: E402
from mitosis.data.split import choose_split  # noqa: E402
from mitosis.data.tiling import boxes_in_tile, parse_tile_name, tile_starts  # noqa: E402
from mitosis.detection.candidates import label_candidates  # noqa: E402
from mitosis.detection.inference import center_nms  # noqa: E402
from mitosis.evaluation.fusion import case_scores  # noqa: E402
from mitosis.evaluation.matching import match, prf  # noqa: E402
from mitosis.evaluation.metrics import evaluate_cases, roc_auc  # noqa: E402


def test_tile_starts_cover_image_and_end_flush():
    assert tile_starts(1500, 640, 576) == [0, 576, 860]
    assert tile_starts(500, 640, 576) == [0]
    s = tile_starts(6577, 640, 576)
    assert s[0] == 0 and s[-1] == 6577 - 640


def test_box_assigned_by_center_and_clipped():
    box = [600, 100, 650, 150]                       # center (625, 125)
    assert boxes_in_tile([box], 0, 0, 640) == [(600, 100, 640, 150)]
    assert boxes_in_tile([box], 640, 0, 640) == []   # center not in this tile


def test_parse_tile_name():
    assert parse_tile_name("034_x576_y1152") == ("034", 576, 1152)


def test_crop_centered_pads_outside_with_white():
    arr = np.zeros((100, 100, 3), np.uint8)
    c = crop_centered(arr, 0, 0, 32)
    assert c.shape == (32, 32, 3)
    assert (c[:16, :16] == 255).all() and (c[16:, 16:] == 0).all()


def test_center_nms_removes_overlapping_duplicates():
    d = np.array([[0, 0, 50, 50, .9], [10, 10, 60, 60, .5], [200, 200, 250, 250, .7]])
    kept = center_nms(d, 25)
    assert len(kept) == 2 and kept[0, 4] == .9


def test_match_radius_and_duplicates():
    gt = np.array([[100, 100], [300, 300], [500, 500]], float)
    perfect = np.array([[75, 75, 125, 125, .9], [275, 275, 325, 325, .8], [475, 475, 525, 525, .7]])
    assert match(perfect, gt, 30) == (3, 0, 0)
    shifted = perfect.copy(); shifted[0, :4] += 20; shifted[1, :4] += 40
    assert match(shifted, gt, 30) == (2, 1, 1)
    assert match(np.vstack([perfect, perfect[:1]]), gt, 30) == (3, 1, 0)
    assert match(np.zeros((0, 5)), gt, 30) == (0, 0, 3)
    assert prf(3, 1, 0) == pytest.approx((0.75, 1.0, 6 / 7))


def test_perfect_classifier_fixes_detector_mistakes():
    gt = np.array([[100, 100], [300, 300]], float)
    dets = np.array([[75, 75, 125, 125, .3], [275, 275, 325, 325, .2], [500, 500, 550, 550, .9], [700, 700, 750, 750, .8]])
    case = {"dets": dets, "probs": np.array([.95, .9, .05, .1]), "gt": gt, "tumor": "x"}
    assert tuple(evaluate_cases([case], "detector only", .5)[0]) == (0, 2, 2)
    assert tuple(evaluate_cases([case], "geometric mean", .3)[0]) == (2, 0, 0)
    assert np.allclose(case_scores(case, "average"), (dets[:, 4] + case["probs"]) / 2)


def test_label_candidates():
    dets = np.array([[75, 75, 125, 125, .3], [500, 500, 550, 550, .9]])
    assert list(label_candidates(dets, np.array([[100., 110.]]), 30)) == [1, 0]


def test_roc_auc_handles_ties():
    assert roc_auc([1, 0, 1, 0], [.5] * 4) == 0.5
    assert roc_auc([1, 1, 0, 0], [.9, .8, .2, .1]) == 1.0
    assert roc_auc([1, 0, 1, 0], [.9, .9, .1, .1]) == 0.5


def test_extended_split_keeps_val_and_test():
    rng = np.random.default_rng(0)
    images = {i: {"file_name": f"{i:03d}.tiff", "tumor_type": t, "id": i}
              for i, t in enumerate(rng.choice(["a", "b", "c"], 150), 1)}
    fig = {im["file_name"]: "1" for im in images.values()}
    old = choose_split(images, fig, 10)
    new = choose_split(images, fig, 20, base_per_type=10)
    ev = lambda rows: {(im["file_name"], s) for im, s in rows if s != "train"}
    tr = lambda rows: {im["file_name"] for im, s in rows if s == "train"}
    assert ev(old) == ev(new) and tr(old) < tr(new)
    assert not ({im["file_name"] for im, s in new if s == "test"} & tr(new))  # no leakage
