"""Streaming dataset builder: download -> tile -> crop -> delete, one image at a time.

Peak disk usage is about (workers x one TIFF) plus the tiles, instead of the
full 65 GB. Re-running resumes: images already processed are skipped.
"""

import json
import random
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .annotations import load_annotations, load_figshare_ids
from .crops import save_cell_crops
from .download import download
from .io import read_rgb
from .split import TEST, TRAIN, VAL, choose_split, verify_same_eval_split, write_split_csv
from .tiling import tile_image

SPLITS = (TRAIN, VAL, TEST)


def plan_split(repo: Path, cfg: dict, keep_split: Path = None):
    fig_ids = load_figshare_ids(repo)
    images, boxes = load_annotations(Path(repo) / "databases" / "MIDOG++.json")
    rows = choose_split(images, fig_ids, cfg["per_type"], cfg["test_frac"], cfg["val_frac"], cfg["seed"],
                        cfg.get("holdout", []), cfg.get("base_per_type", 0))
    if keep_split:
        n_eval, n_old, n_new = verify_same_eval_split(keep_split, rows)
        print(f"Verified: same {n_eval} val/test images as {keep_split}, train grows {n_old} -> {n_new} images")
    return fig_ids, boxes, rows


def print_plan(rows):
    plan = Counter((im["tumor_type"], s) for im, s in rows)
    print(f"Selected {len(rows)} images (~{len(rows) * 0.13:.0f} GB to download)")
    for tumor in sorted({t for t, _ in plan}):
        print(f"  {tumor:<35} train {plan[(tumor, TRAIN)]:>3}  val {plan[(tumor, VAL)]:>3}  test {plan[(tumor, TEST)]:>3}")


def build_dataset(repo: Path, out: Path, cfg: dict, workers=4, tmp_dir: Path = None,
                  images_dir: Path = None, keep_split: Path = None):
    repo, out = Path(repo), Path(out).resolve()
    fig_ids, boxes, rows = plan_split(repo, cfg, keep_split)
    print_plan(rows)
    crop_size = cfg.get("crop_size", 0)

    for split in SPLITS:
        for sub in ("images", "labels", "full"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)
    if crop_size:
        for split in (TRAIN, VAL):
            for label in ("mitosis", "imposter"):
                (out / "crops" / split / label).mkdir(parents=True, exist_ok=True)
    tmp = Path(tmp_dir) if tmp_dir else out / "_download"
    tmp.mkdir(parents=True, exist_ok=True)
    write_split_csv(out / "split.csv", rows, boxes)
    shutil.copy2(repo / "databases" / "MIDOG++.json", out / "MIDOG++.json")  # needed for evaluation

    done_file = out / "_done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    todo = [(im, s) for im, s in rows if im["file_name"] not in done]
    print(f"{len(done)} images already processed, {len(todo)} to go")
    lock = threading.Lock()

    def process(item):
        im, split = item
        name = im["file_name"]
        local = Path(images_dir) / name if images_dir and (Path(images_dir) / name).exists() else None
        src = local or (tmp / name)
        if local is None:
            download(fig_ids[name], src)
        arr = read_rgb(src)
        rng = random.Random(f"{cfg['seed']}-{name}")
        st = tile_image(arr, boxes[im["id"]], Path(name).stem, split, out, cfg["tile"], cfg["overlap"],
                        cfg["neg_ratio"], rng, cfg.get("background_threshold", 225))
        if crop_size and split in (TRAIN, VAL):
            st.update(save_cell_crops(arr, boxes[im["id"]], Path(name).stem, split, out, crop_size))
        del arr
        if split in (VAL, TEST):  # keep full images for whole-image evaluation
            dest = out / "full" / split / name
            shutil.copy2(src, dest) if local else shutil.move(str(src), dest)
        elif local is None:
            src.unlink(missing_ok=True)
        with lock:
            with open(done_file, "a") as f:
                f.write(name + "\n")
        return name, split, st

    failed, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process, it) for it in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                name, split, st = fut.result()
            except Exception as e:
                print(f"  ! {e} (re-run to retry)")
                failed.append(str(e))
                continue
            el = time.time() - t0
            print(f"[{k}/{len(todo)}] {name} ({split}): {st['tiles']} tiles, {st['mitoses']} mitoses | "
                  f"elapsed {el / 60:.1f} min, ETA {el / k * (len(todo) - k) / 60:.1f} min")
    shutil.rmtree(tmp, ignore_errors=True)

    (out / "data.yaml").write_text(
        "# MIDOG++ mitotic figure detection (YOLO format)\n"
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: mitotic figure\n")
    manifest = write_manifest(out, rows, cfg, crop_size)
    manifest["failed"] = failed
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nDone.", json.dumps(manifest["splits"], indent=2))
    if failed:
        print(f"\nWARNING: {len(failed)} image(s) failed. Run the same command again to retry only those.")
    return manifest


def write_manifest(out: Path, rows, cfg, crop_size):
    """Statistics read back from disk, so they are correct after resumed runs too."""
    manifest = {"config": {k: str(v) for k, v in cfg.items()}, "splits": {}}
    for split in SPLITS:
        lbls = list((out / "labels" / split).glob("*.txt"))
        n_obj = [len(p.read_text().splitlines()) for p in lbls]
        manifest["splits"][split] = {"images": sum(1 for _, s in rows if s == split), "tiles": len(lbls),
                                     "tiles_with_mitosis": sum(1 for n in n_obj if n),
                                     "mitoses_in_tiles": sum(n_obj)}
        if crop_size and split != TEST:
            for label in ("mitosis", "imposter"):
                manifest["splits"][split][f"crops_{label}"] = len(list((out / "crops" / split / label).glob("*.jpg")))
    return manifest
