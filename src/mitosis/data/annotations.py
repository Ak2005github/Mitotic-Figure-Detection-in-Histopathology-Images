"""MIDOG++ annotation and file-list handling.

Notes on the MIDOG++ annotation file (databases/MIDOG++.json)
  * Boxes are [x1, y1, x2, y2] (always 50 x 50 px around the cell), NOT the
    usual COCO [x, y, w, h].
  * Category 1 = mitotic figure, category 2 = "imposter" (a look-alike,
    non-mitotic cell), useful as hard negatives.
"""

import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MITOSIS, IMPOSTER = 1, 2


def load_figshare_ids(repo: Path) -> dict:
    """file_name -> figshare file id, read from the official Setup.ipynb."""
    nb = json.loads((Path(repo) / "Setup.ipynb").read_text())
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if src.lstrip().startswith("filename"):
            mapping = ast.literal_eval(src.split("=", 1)[1].strip())
            return {name: fid for fid, name in mapping.items()}
    raise ValueError("Could not find the figshare file list in Setup.ipynb")


def load_annotations(ann_path: Path):
    """Returns (images by id, boxes[image_id][category] -> list of [x1, y1, x2, y2])."""
    data = json.loads(Path(ann_path).read_text())
    images = {im["id"]: im for im in data["images"]}
    boxes = defaultdict(lambda: {MITOSIS: [], IMPOSTER: []})
    for a in data["annotations"]:
        boxes[a["image_id"]][a["category_id"]].append(a["bbox"])
    return images, boxes


def box_centers(boxes) -> np.ndarray:
    b = np.asarray(boxes, float).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)


def mitosis_centers(ann_path: Path):
    """(file_name -> image_id, image_id -> (N, 2) array of mitosis centers)."""
    images, boxes = load_annotations(ann_path)
    by_name = {im["file_name"]: i for i, im in images.items()}
    centers = {i: box_centers(boxes[i][MITOSIS]) for i in images}
    return by_name, centers, boxes
