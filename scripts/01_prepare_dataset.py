"""Step 1: build the YOLO dataset from MIDOG++ (streaming download, image-level split, tiling, crops).

  git clone https://github.com/DeepMicroscopy/MIDOGpp
  python scripts/01_prepare_dataset.py --repo MIDOGpp --out datasets/midog_yolo

Output: images/ + labels/ (640 px tiles), full/{val,test}/ (full TIFFs for
evaluation), crops/{train,val}/ (classifier crops), split.csv, data.yaml,
MIDOG++.json, manifest.json.
"""

import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mitosis.config import load_config  # noqa: E402
from mitosis.data.builder import build_dataset, plan_split, print_plan  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, required=True, help="Cloned DeepMicroscopy/MIDOGpp repository")
    ap.add_argument("--out", type=Path, default=Path("datasets/midog_yolo"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--per-type", type=int, default=None, help="Override data.per_type")
    ap.add_argument("--base-per-type", type=int, default=None, help="Override data.base_per_type")
    ap.add_argument("--crop-size", type=int, default=None, help="Override data.crop_size (0 = no crops)")
    ap.add_argument("--keep-split", type=Path, default=None, help="Earlier split.csv: verify val/test are identical")
    ap.add_argument("--workers", type=int, default=4, help="Parallel downloads")
    ap.add_argument("--tmp-dir", type=Path, default=None, help="Where TIFFs are downloaded before tiling")
    ap.add_argument("--images-dir", type=Path, default=None, help="Folder with already-downloaded TIFFs")
    ap.add_argument("--dry-run", action="store_true", help="Only print the planned split")
    args = ap.parse_args()

    cfg = load_config(args.config)["data"]
    for key in ("per_type", "base_per_type", "crop_size"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.dry_run:
        print_plan(plan_split(args.repo, cfg, args.keep_split)[2])
        return
    build_dataset(args.repo, args.out, cfg, args.workers, args.tmp_dir, args.images_dir, args.keep_split)
    print(f"Dataset config: {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
