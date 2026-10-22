"""Step 6: two-stage evaluation (detector candidates -> classifier) on full images.

Scores every cached candidate with the classifier (8-fold TTA), then compares
score variants: detector only, classifier only, geometric mean, average. The
variant AND threshold are chosen on VAL only; TEST is reported once.

  python scripts/06_evaluate_two_stage.py --data-dir datasets/midog_yolo \
      --candidates results/detector/candidates.pkl --classifier models/classifier_effnet_b0.pt \
      --out results/two_stage
"""

import argparse
import csv
import json

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from mitosis.classification.model import load_classifier, predict_probs  # noqa: E402
from mitosis.config import load_config  # noqa: E402
from mitosis.data.io import read_rgb  # noqa: E402
from mitosis.detection.candidates import load_candidates  # noqa: E402
from mitosis.evaluation.fusion import VARIANTS, case_scores  # noqa: E402
from mitosis.evaluation.metrics import report, threshold_sweep  # noqa: E402
from mitosis.visualization import save_overview  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, default=Path("results/detector/candidates.pkl"))
    ap.add_argument("--classifier", type=Path, default=Path("models/classifier_effnet_b0.pt"))
    ap.add_argument("--out", type=Path, default=Path("results/two_stage"))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    radius, c = cfg["evaluation"]["match_radius"], cfg["classifier"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_classifier(args.classifier, device)
    cases = load_candidates(args.candidates)
    for split in ("val", "test"):
        for case in cases[split]:
            case["probs"] = predict_probs(net, case["crops"], device, inner=c["inner"], net_size=c["net_size"])

    sweep = {}
    for v in VARIANTS:
        _, (t, f1) = threshold_sweep(cases["val"], v, radius)
        sweep[v] = {"threshold": t, "val_f1": round(f1, 4)}
    best2 = max((v for v in VARIANTS if v != "detector only"), key=lambda v: sweep[v]["val_f1"])
    final = best2 if sweep[best2]["val_f1"] > sweep["detector only"]["val_f1"] else "detector only"

    out = {"selection_on_val": sweep, "chosen_variant": final, "variants": {}}
    cinfo = args.classifier.with_suffix(".json")
    if cinfo.exists():
        out["classifier"] = json.loads(cinfo.read_text())
    for v in VARIANTS:
        t = sweep[v]["threshold"]
        out["variants"][v] = {"threshold": t, "val": report(cases["val"], v, t, radius),
                              "test": report(cases["test"], v, t, radius)}
    out["final"] = {"variant": final, **out["variants"][final]}

    t = sweep[final]["threshold"]
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "per_image_counts.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "image", "tumor_type", "true_count", "predicted_count"])
        for split in ("val", "test"):
            for case in cases[split]:
                wr.writerow([split, case["name"], case["tumor"], len(case["gt"]), int((case_scores(case, final) >= t).sum())])
    (args.out / "overviews").mkdir(exist_ok=True)
    for case in cases["test"][:8]:
        keep = case_scores(case, final) >= t
        save_overview(read_rgb(case["path"]), case["dets"][keep],
                      args.out / "overviews" / f"{Path(case['name']).stem}.jpg", case["gt_boxes"])
    (args.out / "metrics.json").write_text(json.dumps(out, indent=2))

    print("Selection on val:", json.dumps(sweep, indent=2))
    print(f"Chosen: {final}")
    for v in VARIANTS:
        m = out["variants"][v]["test"]
        print(f"  TEST {v:<16} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  count r {m['count_correlation']}")
    print(f"Results in {args.out}")


if __name__ == "__main__":
    main()
