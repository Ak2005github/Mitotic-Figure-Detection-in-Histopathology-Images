"""Overviews (predictions vs ground truth), density heatmaps, threshold curves."""

import numpy as np
from PIL import Image, ImageDraw


def save_overview(arr, dets, path, gt_boxes=None, max_side=2400):
    """Downscaled image: predictions as red circles, ground truth as green circles."""
    h, w = arr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    img = Image.fromarray(arr).resize((int(w * s), int(h * s)))
    draw = ImageDraw.Draw(img)
    r = max(6, int(35 * s))
    for x1, y1, x2, y2 in (gt_boxes if gt_boxes is not None else []):
        cx, cy = (x1 + x2) / 2 * s, (y1 + y2) / 2 * s
        draw.ellipse([cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3], outline=(0, 200, 0), width=3)
    for x1, y1, x2, y2, *_ in dets:
        cx, cy = (x1 + x2) / 2 * s, (y1 + y2) / 2 * s
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(230, 0, 0), width=3)
    img.save(path, quality=90)


def density_heatmap(dets, shape, cell=512):
    h, w = shape[:2]
    grid = np.zeros((int(np.ceil(h / cell)), int(np.ceil(w / cell))))
    for x1, y1, x2, y2, *_ in dets:
        grid[min(int((y1 + y2) / 2 // cell), grid.shape[0] - 1), min(int((x1 + x2) / 2 // cell), grid.shape[1] - 1)] += 1
    return grid


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_heatmap(grid, path, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap="magma", interpolation="nearest")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="mitoses per 512 x 512 px region")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_threshold_curve(rows, best_t, path, title):
    plt = _plt()
    sw = np.array(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, lab in [(1, "precision"), (2, "recall"), (3, "F1")]:
        ax.plot(sw[:, 0], sw[:, i], label=lab, lw=2)
    ax.axvline(best_t, ls="--", c="gray")
    ax.set_xlabel("confidence threshold"); ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
