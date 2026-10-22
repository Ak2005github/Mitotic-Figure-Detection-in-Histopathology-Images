"""Step 3: evaluate the detector alone on FULL val/test images (MIDOG-style F1).

Runs tiled inference on every full image once (low threshold), caches the
candidates for steps 5 and 6, picks the confidence threshold with the best F1
on VAL and reports TEST at that threshold. Also reports tile-level mAP.

  python scripts/03_evaluate_detector.py --weights runs/detector/yolov8s/weights/best.pt \
      --data-dir datasets/midog_yolo --out results/detector
"""

import argparse
import csv
import json

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mitosis.config import load_config  # noqa: E402
from mitosis.data.io import read_rgb  # noqa: E402
from mitosis.detection.candidates import generate_candidates, save_candidates  # noqa: E402
from mitosis.evaluation.metrics import report, threshold_sweep  # noqa: E402
from mitosis.visualization import save_overview, save_threshold_curve  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/detector"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-map", action="store_true", help="Skip tile-level mAP")
    args = ap.parse_args()

    from ultralytics import YOLO
    cfg = load_config(args.config)
    radius = cfg["evaluation"]["match_radius"]
    model = YOLO(args.weights)
    args.out.mkdir(parents=True, exist_ok=True)

    cases = generate_candidates(model, args.data_dir, ("val", "test"), cfg["inference"], cfg["data"]["crop_size"],
                                radius, args.device)
    save_candidates(cases, args.out / "candidates.pkl")

    tune_on = "val" if cases["val"] else "test"
    rows, (best_t, best_f1) = threshold_sweep(cases[tune_on], "detector only", radius)
    print(f"\nBest threshold on {tune_on}: {best_t:.3f} (F1 {best_f1:.3f})")
    out = {"threshold": best_t, "tuned_on": tune_on, "match_radius_px": radius}
    for split in ("val", "test"):
        if cases[split]:
            out[split] = report(cases[split], "detector only", best_t, radius)

    with open(args.out / "per_image_counts.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "image", "tumor_type", "true_count", "predicted_count"])
        for split in ("val", "test"):
            for c in cases[split]:
                wr.writerow([split, c["name"], c["tumor"], len(c["gt"]), int((c["dets"][:, 4] >= best_t).sum())])
    save_threshold_curve(rows, best_t, args.out / "threshold_curve.png",
                         f"Detector only, {tune_on} (best F1 {best_f1:.3f} at {best_t:.2f})")
    (args.out / "overviews").mkdir(exist_ok=True)
    for c in cases["test"][:8]:
        save_overview(read_rgb(c["path"]), c["dets"][c["dets"][:, 4] >= best_t],
                      args.out / "overviews" / f"{Path(c['name']).stem}.jpg", c["gt_boxes"])

    if not args.skip_map:
        m = model.val(data=str(args.data_dir / "data.yaml"), split="test", imgsz=cfg["inference"]["tile"],
                      batch=cfg["inference"]["batch"], device=args.device, plots=True, verbose=False,
                      project=str(args.out.resolve()), name="tile_map", exist_ok=True)
        out["test_tiles_map"] = {"mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4),
                                 "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4)}

    (args.out / "metrics.json").write_text(json.dumps(out, indent=2))
    t = out["test"]
    print(f"TEST (detector only): P {t['precision']:.3f}  R {t['recall']:.3f}  F1 {t['f1']:.3f}  "
          f"count r {t['count_correlation']}")
    print(f"Results in {args.out}")


if __name__ == "__main__":
    main()
