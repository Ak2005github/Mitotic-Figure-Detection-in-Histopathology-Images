"""Stage-1 candidates on full val/test images, cached to disk.

Each case holds the detector's candidates (at a low threshold), a crop around
each candidate (for the classifier), the ground-truth mitosis centers and a
label per candidate (1 if a true mitosis is within the match radius).
"""

import pickle
from pathlib import Path

import numpy as np

from ..data.annotations import mitosis_centers
from ..data.io import crop_centered, read_rgb
from ..data.split import read_split_csv
from .inference import predict_large_image


def label_candidates(dets, gt, radius):
    if len(gt) == 0 or len(dets) == 0:
        return np.zeros(len(dets), int)
    c = np.stack([(dets[:, 0] + dets[:, 2]) / 2, (dets[:, 1] + dets[:, 3]) / 2], 1)
    d = np.linalg.norm(c[:, None, :] - gt[None, :, :], axis=2)
    return (d.min(1) <= radius).astype(int)


def generate_candidates(model, data_dir: Path, splits=("val", "test"), inf_cfg=None, crop_size=128,
                        radius=30, device=None):
    inf_cfg = inf_cfg or {}
    data_dir = Path(data_dir)
    by_name, centers, boxes = mitosis_centers(data_dir / "MIDOG++.json")
    cases = {s: [] for s in splits}
    for r in read_split_csv(data_dir / "split.csv"):
        if r["split"] not in cases:
            continue
        path = data_dir / "full" / r["split"] / r["file_name"]
        if not path.exists():
            print(f"  ! missing {path}, skipped")
            continue
        arr = read_rgb(path)
        dets, info = predict_large_image(model, arr, tile=inf_cfg.get("tile", 640), overlap=inf_cfg.get("overlap", 64),
                                         conf=inf_cfg.get("candidate_conf", 0.05), batch=inf_cfg.get("batch", 16),
                                         nms_radius=inf_cfg.get("nms_radius", 25), device=device,
                                         bg_threshold=inf_cfg.get("background_threshold", 225))
        iid = int(r["image_id"])
        gt = centers.get(iid, np.zeros((0, 2)))
        cases[r["split"]].append({
            "name": r["file_name"], "tumor": r["tumor_type"], "split": r["split"], "path": str(path),
            "dets": dets, "gt": gt, "gt_boxes": np.asarray(boxes[iid][1], float).reshape(-1, 4),
            "crops": np.stack([crop_centered(arr, (x1 + x2) / 2, (y1 + y2) / 2, crop_size)
                               for x1, y1, x2, y2, _ in dets]) if len(dets) else np.zeros((0, crop_size, crop_size, 3), np.uint8),
            "cand_labels": label_candidates(dets, gt, radius), "tiles": info,
        })
        print(f"[{r['split']}] {r['file_name']} ({r['tumor_type']}): {len(dets)} candidates, {len(gt)} true mitoses")
    return cases


def save_candidates(cases, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cases, f)


def load_candidates(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)
