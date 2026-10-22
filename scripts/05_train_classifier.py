"""Step 5: train the EfficientNet-B0 verifier (mitosis vs look-alike).

Positives: crops of true mitoses. Negatives: imposter crops + mined false
positives. The best epoch is chosen by ROC AUC on the VAL candidates cached by
step 3.

  python scripts/05_train_classifier.py --data-dir datasets/midog_yolo \
      --candidates results/detector/candidates.pkl --out models/classifier_effnet_b0.pt
"""

import argparse
import json
import random

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mitosis.classification.train import collect_train_items, train_classifier  # noqa: E402
from mitosis.config import load_config  # noqa: E402
from mitosis.detection.candidates import load_candidates  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, default=Path("results/detector/candidates.pkl"))
    ap.add_argument("--out", type=Path, default=Path("models/classifier_effnet_b0.pt"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-pretrained", action="store_true", help="(testing only) skip ImageNet weights")
    args = ap.parse_args()

    cfg = load_config(args.config)["classifier"]
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cases = load_candidates(args.candidates)
    val_crops = [c for case in cases["val"] for c in case["crops"]]
    val_labels = [l for case in cases["val"] for l in case["cand_labels"]]
    items = collect_train_items(args.data_dir)
    _, history, best_auc = train_classifier(items, val_crops, val_labels, device, cfg, args.out,
                                            pretrained=not args.no_pretrained)
    n_mined = sum(1 for p, _ in items if p.parent.name == "mined")
    info = {"train_crops": len(items), "positives": sum(y for _, y in items), "mined_hard_negatives": n_mined,
            "val_candidates": len(val_crops), "best_val_auc": round(best_auc, 4), "history": history}
    args.out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"Saved {args.out} (best val AUC {best_auc:.4f})")


if __name__ == "__main__":
    main()
