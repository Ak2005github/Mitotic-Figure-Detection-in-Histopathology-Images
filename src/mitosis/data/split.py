"""Balanced, image-level train/val/test split.

The split is done by IMAGE, never by tile: tiles from one image are highly
correlated, so splitting tiles would leak tissue into the test set.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

TRAIN, VAL, TEST = "train", "val", "test"


def choose_split(images: dict, fig_ids: dict, per_type: int, test_frac: float = 0.2,
                 val_frac: float = 0.1, seed: int = 0, holdout=(), base_per_type: int = 0):
    """Sample `per_type` images per tumor type (0 = all) and assign splits.

    base_per_type: val/test sizes are computed as if only `base_per_type`
    images were sampled; every extra image goes to train. The shuffle order
    does not depend on per_type, so this reproduces the exact val/test images
    of an earlier `per_type=base_per_type` run while the training set grows.
    holdout: tumor types sent entirely to test (cross-domain evaluation).
    """
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for im in images.values():
        if im["file_name"] in fig_ids:  # 50 images in the JSON have no download link
            by_type[im["tumor_type"]].append(im)

    rows = []
    for tumor in sorted(by_type):
        lst = sorted(by_type[tumor], key=lambda x: x["file_name"])
        rng.shuffle(lst)
        if per_type > 0:
            lst = lst[:per_type]
        if tumor in holdout:
            rows += [(im, TEST) for im in lst]
            continue
        n = min(len(lst), base_per_type) if base_per_type else len(lst)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        for i, im in enumerate(lst):
            split = TEST if i < n_test else VAL if i < n_test + n_val else TRAIN
            rows.append((im, split))
    return rows


def write_split_csv(path: Path, rows, boxes, mitosis_cat: int = 1):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["file_name", "image_id", "tumor_type", "split", "width", "height", "n_mitoses"])
        for im, s in rows:
            wr.writerow([im["file_name"], im["id"], im["tumor_type"], s, im["width"], im["height"],
                         len(boxes[im["id"]][mitosis_cat])])


def read_split_csv(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def verify_same_eval_split(old_csv: Path, rows):
    """Raise if val/test differ from an earlier split or old train images are missing."""
    old = {(r["file_name"], r["split"]) for r in read_split_csv(old_csv)}
    old_eval = {x for x in old if x[1] != TRAIN}
    new_eval = {(im["file_name"], s) for im, s in rows if s != TRAIN}
    old_train = {n for n, s in old if s == TRAIN}
    new_train = {im["file_name"] for im, s in rows if s == TRAIN}
    if old_eval != new_eval or not old_train <= new_train:
        raise ValueError(f"Split mismatch with {old_csv}: val/test differ or old train images missing")
    return len(new_eval), len(old_train), len(new_train)
