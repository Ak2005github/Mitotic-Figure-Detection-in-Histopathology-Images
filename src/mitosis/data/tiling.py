"""Cut large images into overlapping tiles with YOLO-format labels."""

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .annotations import IMPOSTER, MITOSIS


def tile_starts(length: int, tile: int, stride: int):
    """Tile start offsets covering [0, length); the last tile is flush with the border."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def boxes_in_tile(boxes, x0, y0, tile):
    """Boxes whose CENTER lies inside the tile, clipped to the tile (tile coordinates)."""
    out = []
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if x0 <= cx < x0 + tile and y0 <= cy < y0 + tile:
            bx1, by1 = max(x1, x0) - x0, max(y1, y0) - y0
            bx2, by2 = min(x2, x0 + tile) - x0, min(y2, y0 + tile) - y0
            if bx2 - bx1 > 4 and by2 - by1 > 4:
                out.append((bx1, by1, bx2, by2))
    return out


def is_background(crop: np.ndarray, threshold: float = 225) -> bool:
    """Mostly white glass (no tissue)."""
    return crop.mean() > threshold


def tile_image(arr, im_boxes, stem, split, out: Path, tile=640, overlap=64, neg_ratio=0.15,
               rng=None, bg_threshold=225):
    """Write tiles + YOLO labels for one image. Keeps every tile with a mitosis,
    every tile with an imposter (unlabelled hard negative) and a random
    `neg_ratio` of the remaining tissue tiles."""
    h, w = arr.shape[:2]
    stride = tile - overlap
    stats = Counter()
    img_dir, lbl_dir = Path(out) / "images" / split, Path(out) / "labels" / split
    for y0 in tile_starts(h, tile, stride):
        for x0 in tile_starts(w, tile, stride):
            mit = boxes_in_tile(im_boxes[MITOSIS], x0, y0, tile)
            imp = boxes_in_tile(im_boxes[IMPOSTER], x0, y0, tile)
            crop = arr[y0:y0 + tile, x0:x0 + tile]
            if is_background(crop, bg_threshold):
                stats["skipped_background"] += 1
                continue
            if not (mit or imp or rng.random() < neg_ratio):
                stats["skipped_negative"] += 1
                continue
            if crop.shape[0] != tile or crop.shape[1] != tile:  # tiny images: pad
                pad = np.full((tile, tile, 3), 255, np.uint8)
                pad[:crop.shape[0], :crop.shape[1]] = crop
                crop = pad
            name = f"{stem}_x{x0}_y{y0}"
            Image.fromarray(crop).save(img_dir / f"{name}.jpg", quality=95)
            with open(lbl_dir / f"{name}.txt", "w") as f:
                for bx1, by1, bx2, by2 in mit:
                    cx, cy = (bx1 + bx2) / 2 / tile, (by1 + by2) / 2 / tile
                    bw, bh = (bx2 - bx1) / tile, (by2 - by1) / tile
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            stats["tiles"] += 1
            stats["tiles_with_mitosis"] += bool(mit)
            stats["tiles_hard_negative"] += (not mit) and bool(imp)
            stats["mitoses"] += len(mit)
    return stats


def parse_tile_name(stem: str):
    """'034_x576_y1152' -> ('034', 576, 1152)."""
    image_stem, xs, ys = stem.rsplit("_", 2)
    return image_stem, int(xs[1:]), int(ys[1:])
