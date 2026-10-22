"""Hard negative mining: the detector's own false positives on training tiles.

A detection is a false positive if no true mitosis (anywhere in the full
image, so cells cut by tile borders are handled) lies within the match radius.
Duplicates from overlapping tiles are merged and the most confident (hardest)
ones are kept, capped relative to the number of positive crops.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from ..data.annotations import mitosis_centers
from ..data.io import crop_centered
from ..data.tiling import parse_tile_name


def mine_hard_negatives(model, data_dir: Path, out_dir: Path = None, conf=0.1, radius=30, batch=32,
                        max_ratio=1.5, crop_size=128, device=None):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir) if out_dir else data_dir / "crops" / "train" / "mined"
    by_name, centers, _ = mitosis_centers(data_dir / "MIDOG++.json")
    tiles = sorted((data_dir / "images" / "train").glob("*.jpg"))
    n_pos = len(list((data_dir / "crops" / "train" / "mitosis").glob("*.jpg")))
    cands = []  # (score, image_stem, global x, global y, tile path, local x, local y)
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        results = model.predict([str(p) for p in chunk], imgsz=640, conf=conf, device=device, verbose=False)
        for p, r in zip(chunk, results):
            if len(r.boxes) == 0:
                continue
            stem, x0, y0 = parse_tile_name(p.stem)
            g = centers.get(by_name.get(stem + ".tiff"), np.zeros((0, 2)))
            for (bx1, by1, bx2, by2), s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                lx, ly = (bx1 + bx2) / 2, (by1 + by2) / 2
                gx, gy = lx + x0, ly + y0
                if len(g) and np.min(np.hypot(g[:, 0] - gx, g[:, 1] - gy)) <= radius:
                    continue  # true positive
                cands.append((float(s), stem, gx, gy, p, lx, ly))
    cands.sort(key=lambda c: -c[0])
    seen, kept = set(), []
    for c in cands:
        key = (c[1], int(c[2] // 20), int(c[3] // 20))
        if key not in seen:
            seen.add(key)
            kept.append(c)
    if n_pos:
        kept = kept[: int(max_ratio * n_pos)]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    for s, stem, gx, gy, p, lx, ly in kept:
        if p not in cache:
            cache = {p: np.array(Image.open(p).convert("RGB"))}
        Image.fromarray(crop_centered(cache[p], lx, ly, crop_size)).save(out_dir / f"{stem}_{int(gx)}_{int(gy)}.jpg", quality=95)
    print(f"Hard negative mining: {len(cands)} false positives on {len(tiles)} train tiles, "
          f"{len(kept)} unique kept (cap {max_ratio}x the {n_pos} positives)")
    return len(kept)
