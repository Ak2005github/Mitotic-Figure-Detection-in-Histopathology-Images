"""Cell-centred crops for the second-stage classifier."""

from collections import Counter
from pathlib import Path

from PIL import Image

from .annotations import IMPOSTER, MITOSIS
from .io import crop_centered


def save_cell_crops(arr, im_boxes, stem, split, out: Path, size=128):
    """Crops around every labelled mitosis (positive) and imposter (hard negative)."""
    n = Counter()
    for cat, label in ((MITOSIS, "mitosis"), (IMPOSTER, "imposter")):
        d = Path(out) / "crops" / split / label
        for j, (x1, y1, x2, y2) in enumerate(im_boxes[cat]):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if not (0 <= cx < arr.shape[1] and 0 <= cy < arr.shape[0]):
                continue  # 2 MIDOG++ annotations have their center outside the image
            Image.fromarray(crop_centered(arr, cx, cy, size)).save(d / f"{stem}_{j}.jpg", quality=95)
            n[label] += 1
    return n
