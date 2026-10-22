"""Ways to combine the detector confidence (d) and classifier probability (p)."""

import numpy as np

VARIANTS = {
    "detector only": lambda d, p: d,
    "classifier only": lambda d, p: p,
    "geometric mean": lambda d, p: np.sqrt(d * p),
    "average": lambda d, p: (d + p) / 2,
}


def case_scores(case, variant: str) -> np.ndarray:
    d = case["dets"][:, 4] if len(case["dets"]) else np.zeros(0)
    p = case.get("probs", np.ones_like(d))
    return VARIANTS[variant](d, p) if len(d) else np.zeros(0)
