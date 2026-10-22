"""Step 4: hard negative mining for the classifier.

Runs the detector on the training tiles and saves crops of its false positives
(no true mitosis within 30 px) to crops/train/mined/.

  python scripts/04_mine_hard_negatives.py --weights runs/detector/yolov8s/weights/best.pt \
      --data-dir datasets/midog_yolo
"""

import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mitosis.classification.mining import mine_hard_negatives  # noqa: E402
from mitosis.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO
    cfg = load_config(args.config)
    c = cfg["classifier"]
    mine_hard_negatives(YOLO(args.weights), args.data_dir, conf=c["mining_conf"], radius=cfg["evaluation"]["match_radius"],
                        max_ratio=c["max_negative_ratio"], crop_size=cfg["data"]["crop_size"], device=args.device)


if __name__ == "__main__":
    main()
