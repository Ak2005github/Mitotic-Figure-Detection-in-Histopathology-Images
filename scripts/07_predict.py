"""Step 7: run the trained two-stage model on new images (any size).

Writes per image: detections CSV, overview (red circles), mitosis density
heatmap, plus summary.json with counts and speed.

  python scripts/07_predict.py --detector runs/detector/yolov8s/weights/best.pt \
      --classifier models/classifier_effnet_b0.pt --from-metrics results/two_stage/metrics.json \
      --source path/to/images --out results/predictions
"""

import argparse
import csv
import json
import time

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mitosis.config import load_config  # noqa: E402
from mitosis.data.io import crop_centered, read_rgb  # noqa: E402
from mitosis.detection.inference import predict_large_image  # noqa: E402
from mitosis.evaluation.fusion import VARIANTS  # noqa: E402
from mitosis.visualization import density_heatmap, save_heatmap, save_overview  # noqa: E402

EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--classifier", default=None, help="Omit for detector-only inference")
    ap.add_argument("--source", type=Path, required=True, help="Image file or folder")
    ap.add_argument("--out", type=Path, default=Path("results/predictions"))
    ap.add_argument("--from-metrics", type=Path, default=None, help="metrics.json from step 6: use its chosen variant + threshold")
    ap.add_argument("--variant", default="geometric mean", choices=list(VARIANTS))
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO
    cfg = load_config(args.config)
    inf, c = cfg["inference"], cfg["classifier"]
    variant, thr = args.variant, args.threshold
    if args.from_metrics:
        m = json.loads(args.from_metrics.read_text())["final"]
        variant, thr = m["variant"], m["threshold"]
    net, device = None, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.classifier and variant != "detector only":
        from mitosis.classification.model import load_classifier, predict_probs
        net = load_classifier(args.classifier, device)
    elif variant != "detector only":
        variant = "detector only"
        print("No classifier given: using detector only (set --threshold accordingly)")
    print(f"Scoring: {variant}, threshold {thr}")

    det = YOLO(args.detector)
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.source.iterdir() if p.suffix.lower() in EXTS) if args.source.is_dir() else [args.source]
    summary = []
    for p in files:
        t0 = time.time()
        arr = read_rgb(p)
        dets, info = predict_large_image(det, arr, inf["tile"], inf["overlap"], inf["candidate_conf"], inf["batch"],
                                         inf["nms_radius"], args.device, inf["background_threshold"])
        probs = np.ones(len(dets))
        if net is not None and len(dets):
            crops = np.stack([crop_centered(arr, (x1 + x2) / 2, (y1 + y2) / 2, cfg["data"]["crop_size"]) for x1, y1, x2, y2, _ in dets])
            probs = predict_probs(net, crops, device, inner=c["inner"], net_size=c["net_size"])
        scores = VARIANTS[variant](dets[:, 4], probs) if len(dets) else np.zeros(0)
        keep = dets[scores >= thr]
        dt = time.time() - t0
        with open(args.out / f"{p.stem}_detections.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["x1", "y1", "x2", "y2", "detector_conf", "classifier_prob", "score"])
            for d, pr, s in zip(dets[scores >= thr], probs[scores >= thr], scores[scores >= thr]):
                wr.writerow([*np.round(d[:4], 1), round(float(d[4]), 4), round(float(pr), 4), round(float(s), 4)])
        save_overview(arr, keep, args.out / f"{p.stem}_overview.jpg")
        save_heatmap(density_heatmap(keep, arr.shape), args.out / f"{p.stem}_heatmap.png", f"{p.name}: {len(keep)} mitotic figures")
        mp = arr.shape[0] * arr.shape[1] / 1e6
        summary.append({"image": p.name, "width": arr.shape[1], "height": arr.shape[0], "mitotic_count": int(len(keep)),
                        "seconds": round(dt, 2), "megapixels_per_second": round(mp / dt, 2), **info})
        print(f"{p.name}: {len(keep)} mitoses | {arr.shape[1]}x{arr.shape[0]} px | {dt:.1f}s")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Results in {args.out}")


if __name__ == "__main__":
    main()
