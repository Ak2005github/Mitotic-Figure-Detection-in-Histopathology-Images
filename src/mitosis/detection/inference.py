"""Tiled (sliding-window) inference on large pathology images.

A detector trained on 640 x 640 tiles cannot see a ~6,500 x 4,900 px image at
once without shrinking cells to a few pixels. Instead we cut the image into
overlapping tiles, skip background, run the detector on batches of tiles,
shift boxes back to image coordinates and merge duplicates from overlapping
tiles with center-distance NMS.
"""

import numpy as np

from ..data.tiling import is_background, tile_starts


def center_nms(dets: np.ndarray, radius: float) -> np.ndarray:
    """dets: (N, 5) [x1, y1, x2, y2, score]. Keep the highest-scoring box and drop
    any other box whose center is within `radius` px of it."""
    if len(dets) == 0:
        return dets
    dets = dets[np.argsort(-dets[:, 4])]
    centers = np.stack([(dets[:, 0] + dets[:, 2]) / 2, (dets[:, 1] + dets[:, 3]) / 2], 1)
    keep, removed = [], np.zeros(len(dets), bool)
    for i in range(len(dets)):
        if removed[i]:
            continue
        keep.append(i)
        removed |= np.linalg.norm(centers - centers[i], axis=1) < radius
    return dets[keep]


def predict_large_image(model, arr, tile=640, overlap=64, conf=0.05, batch=16, nms_radius=25,
                        device=None, bg_threshold=225):
    """Returns ((N, 5) detections in full-image pixels, tile statistics)."""
    h, w = arr.shape[:2]
    stride = tile - overlap
    coords, crops, all_dets = [], [], []

    def flush():
        if not crops:
            return
        results = model.predict(crops, imgsz=tile, conf=conf, device=device, verbose=False)
        for (x0, y0), r in zip(coords, results):
            if len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            xyxy[:, [0, 2]] += x0
            xyxy[:, [1, 3]] += y0
            all_dets.append(np.hstack([xyxy, r.boxes.conf.cpu().numpy()[:, None]]))
        coords.clear()
        crops.clear()

    n_tiles = n_skipped = 0
    for y0 in tile_starts(h, tile, stride):
        for x0 in tile_starts(w, tile, stride):
            crop = arr[y0:y0 + tile, x0:x0 + tile]
            n_tiles += 1
            if is_background(crop, bg_threshold):
                n_skipped += 1
                continue
            if crop.shape[:2] != (tile, tile):
                pad = np.full((tile, tile, 3), 255, np.uint8)
                pad[:crop.shape[0], :crop.shape[1]] = crop
                crop = pad
            crops.append(np.ascontiguousarray(crop[..., ::-1]))  # Ultralytics expects BGR arrays
            coords.append((x0, y0))
            if len(crops) == batch:
                flush()
    flush()
    dets = np.vstack(all_dets) if all_dets else np.zeros((0, 5))
    return center_nms(dets, nms_radius), {"tiles": n_tiles, "background_skipped": n_skipped}
