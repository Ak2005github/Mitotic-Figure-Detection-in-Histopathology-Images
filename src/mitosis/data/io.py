"""Image reading and cropping helpers."""

from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # MIDOG++ regions are ~6,500 x 4,900 px


def read_rgb(path) -> np.ndarray:
    """Read a TIFF/JPEG/PNG as an (H, W, 3) uint8 RGB array."""
    path = Path(path)
    arr = None
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile
            arr = tifffile.imread(str(path))
        except Exception:
            arr = None
    if arr is None:
        arr = np.array(Image.open(path).convert("RGB"))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):  # channel-first
        arr = np.moveaxis(arr, 0, -1)
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = (arr / arr.max() * 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def crop_centered(arr: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    """size x size crop centred on (cx, cy); parts outside the image are white."""
    h, w = arr.shape[:2]
    x0, y0 = int(round(cx)) - size // 2, int(round(cy)) - size // 2
    out = np.full((size, size, 3), 255, np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + size), min(h, y0 + size)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out
