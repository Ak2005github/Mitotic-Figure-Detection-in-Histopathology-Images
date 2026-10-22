"""Fine-tune YOLOv8 on the tiled MIDOG++ dataset.

Augmentations run on the fly inside Ultralytics, tuned for histopathology:
vertical + horizontal flips (tissue has no "up"), strong HSV jitter
(stain/scanner variation), mild scale + mosaic, and no shear/perspective
(which would distort cell shape).
"""

import time
from pathlib import Path


def add_time_cap(model, hours: float, t0: float):
    """Stop training after `hours` WITHOUT changing the epoch schedule.

    Ultralytics' own `time=` argument re-plans the number of epochs from the
    budget, which stretches the cosine learning-rate schedule (with a generous
    budget the LR barely decays before early stopping)."""
    def stop_after_budget(trainer):
        if time.time() - t0 > hours * 3600:
            print(f"\nTime budget of {hours} h reached, stopping after epoch {trainer.epoch + 1}.")
            trainer.stop = True
    model.add_callback("on_fit_epoch_end", stop_after_budget)


def train_detector(data_yaml, cfg: dict, project="runs/detector", name="yolov8s", device=None):
    import torch
    from ultralytics import YOLO

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(cfg["model"])
    t0 = time.time()
    if cfg.get("hours"):
        add_time_cap(model, cfg["hours"], t0)
    cache = cfg.get("cache", "ram")
    model.train(
        data=str(data_yaml), epochs=cfg["epochs"], imgsz=cfg["imgsz"], batch=cfg["batch"],
        workers=cfg["workers"], device=device, project=str(project), name=name, exist_ok=True,
        cache=False if cache in (None, "none", False) else cache,
        optimizer=cfg["optimizer"], lr0=cfg["lr0"], cos_lr=True, warmup_epochs=cfg["warmup_epochs"],
        amp=True, patience=cfg["patience"], plots=True, verbose=True, seed=cfg.get("seed", 0),
        **cfg["augment"],
    )
    save_dir = Path(model.trainer.save_dir).resolve()
    print(f"\nTraining time: {(time.time() - t0) / 60:.1f} min")
    return save_dir / "weights" / "best.pt"
