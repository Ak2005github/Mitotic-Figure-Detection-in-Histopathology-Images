#!/usr/bin/env bash
# Run the full pipeline end to end (needs a CUDA GPU for reasonable speed).
# Usage: bash run_pipeline.sh [path/to/MIDOGpp] [dataset dir]
set -euo pipefail

MIDOG_REPO=${1:-MIDOGpp}
DATA=${2:-datasets/midog_yolo}
DET=runs/detector/yolov8s/weights/best.pt
CLS=models/classifier_effnet_b0.pt

[ -d "$MIDOG_REPO" ] || git clone --depth 1 https://github.com/DeepMicroscopy/MIDOGpp "$MIDOG_REPO"

python scripts/01_prepare_dataset.py   --repo "$MIDOG_REPO" --out "$DATA"
python scripts/02_train_detector.py    --data-dir "$DATA"
python scripts/03_evaluate_detector.py --weights "$DET" --data-dir "$DATA" --out results/detector
python scripts/04_mine_hard_negatives.py --weights "$DET" --data-dir "$DATA"
python scripts/05_train_classifier.py  --data-dir "$DATA" --candidates results/detector/candidates.pkl --out "$CLS"
python scripts/06_evaluate_two_stage.py --data-dir "$DATA" --candidates results/detector/candidates.pkl \
                                        --classifier "$CLS" --out results/two_stage
python scripts/07_predict.py --detector "$DET" --classifier "$CLS" --from-metrics results/two_stage/metrics.json \
                             --source "$DATA/full/test" --out results/predictions
