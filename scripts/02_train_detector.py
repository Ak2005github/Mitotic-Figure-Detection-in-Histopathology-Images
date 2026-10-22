"""Step 2: fine-tune the YOLOv8s mitosis detector on the tiles.

  python scripts/02_train_detector.py --data-dir datasets/midog_yolo
"""

import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mitosis.config import load_config  # noqa: E402
from mitosis.detection.train import train_detector  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--project", default="runs/detector")
    ap.add_argument("--name", default="yolov8s")
    ap.add_argument("--device", default=None, help="0 for the first GPU, cpu for CPU")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None, help="Lower to 8 on out-of-memory errors")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--cache", default=None, help="ram / disk / none")
    args = ap.parse_args()

    cfg = load_config(args.config)["detector"]
    for key in ("epochs", "batch", "hours", "cache"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    best = train_detector(args.data_dir / "data.yaml", cfg, args.project, args.name, args.device)
    print(f"Best detector weights: {best}")


if __name__ == "__main__":
    main()
