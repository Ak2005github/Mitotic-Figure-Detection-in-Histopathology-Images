"""Center-distance matching (MIDOG criterion).

A prediction is a true positive if its center is within `radius` px of an
unmatched ground-truth mitosis (30 px is about 7.5 micrometres at 0.25 um/px).
Greedy, highest confidence first.
"""

import numpy as np


def match(dets: np.ndarray, gt_centers: np.ndarray, radius: float = 30.0):
    """dets: (N, 5) already thresholded. Returns (tp, fp, fn)."""
    if len(gt_centers) == 0:
        return 0, len(dets), 0
    if len(dets) == 0:
        return 0, 0, len(gt_centers)
    dets = dets[np.argsort(-dets[:, 4])]
    centers = np.stack([(dets[:, 0] + dets[:, 2]) / 2, (dets[:, 1] + dets[:, 3]) / 2], 1)
    used = np.zeros(len(gt_centers), bool)
    tp = 0
    for c in centers:
        d = np.linalg.norm(gt_centers - c, axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= radius:
            used[j] = True
            tp += 1
    return tp, len(dets) - tp, len(gt_centers) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f
