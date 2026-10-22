"""Full-image detection metrics, threshold selection and ROC AUC."""

from collections import defaultdict

import numpy as np

from .fusion import case_scores
from .matching import match, prf

DEFAULT_THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.025), 3)


def evaluate_cases(cases, variant="detector only", thr=0.5, radius=30.0):
    """Returns (tp/fp/fn totals, per-tumor totals, true counts, predicted counts)."""
    tot = np.zeros(3, int)
    per_type = defaultdict(lambda: np.zeros(3, int))
    true_c, pred_c = [], []
    for c in cases:
        s = case_scores(c, variant)
        d = c["dets"][s >= thr] if len(c["dets"]) else c["dets"]
        m = np.array(match(d, c["gt"], radius))
        tot += m
        per_type[c["tumor"]] += m
        true_c.append(len(c["gt"]))
        pred_c.append(len(d))
    return tot, per_type, true_c, pred_c


def summarise(tot, per_type, true_c, pred_c, n_images=None):
    p, r, f = prf(*tot)
    tc, pc = np.array(true_c), np.array(pred_c)
    corr = float(np.corrcoef(tc, pc)[0, 1]) if len(set(tc)) > 1 and len(set(pc)) > 1 else None
    return {"images": n_images if n_images is not None else len(tc),
            "tp": int(tot[0]), "fp": int(tot[1]), "fn": int(tot[2]),
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "mean_abs_count_error": round(float(np.mean(np.abs(tc - pc))), 2) if len(tc) else None,
            "count_correlation": None if corr is None else round(corr, 4),
            "per_tumor_type": {t: dict(zip(["precision", "recall", "f1"], [round(x, 4) for x in prf(*m)]),
                                       tp=int(m[0]), fp=int(m[1]), fn=int(m[2]))
                               for t, m in sorted(per_type.items())}}


def report(cases, variant, thr, radius=30.0):
    return summarise(*evaluate_cases(cases, variant, thr, radius))


def threshold_sweep(cases, variant="detector only", radius=30.0, thresholds=DEFAULT_THRESHOLDS):
    """[(threshold, precision, recall, f1), ...] and the best (threshold, f1)."""
    rows = [(float(t), *prf(*evaluate_cases(cases, variant, t, radius)[0])) for t in thresholds]
    best = max(rows, key=lambda x: x[3])
    return rows, (best[0], best[3])


def roc_auc(y, s):
    """Tie-aware ROC AUC (average ranks), no sklearn needed."""
    y, s = np.asarray(y), np.asarray(s, float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1  # average rank for ties
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
