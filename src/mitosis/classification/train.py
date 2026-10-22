"""Train the EfficientNet-B0 verifier on mitosis / imposter / mined crops."""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..evaluation.metrics import roc_auc
from .dataset import CropDataset, seed_worker
from .model import build_model, predict_probs


def collect_train_items(data_dir: Path):
    c = Path(data_dir) / "crops" / "train"
    items = [(p, 1) for p in sorted((c / "mitosis").glob("*.jpg"))]
    items += [(p, 0) for p in sorted((c / "imposter").glob("*.jpg"))]
    items += [(p, 0) for p in sorted((c / "mined").glob("*.jpg"))]
    return items


def train_classifier(train_items, val_crops, val_labels, device, cfg: dict, out_path: Path = None,
                     pretrained=True):
    """Class-balanced cross entropy, AdamW + one-cycle LR, AMP. The epoch with
    the best ROC AUC on the VAL candidates is kept."""
    net = build_model(pretrained).to(device)
    n1 = sum(1 for _, y in train_items if y == 1)
    n0 = len(train_items) - n1
    w = torch.tensor([len(train_items) / (2 * max(n0, 1)), len(train_items) / (2 * max(n1, 1))], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ds = CropDataset(train_items, True, inner=cfg["inner"], net_size=cfg["net_size"])
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["batch"], shuffle=True, num_workers=cfg["workers"],
                                     drop_last=len(train_items) > cfg["batch"], pin_memory=device.type == "cuda",
                                     worker_init_fn=seed_worker)
    epochs = cfg["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=max(1, epochs * len(dl)),
                                                pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_auc, best_state, history = -1.0, None, []
    print(f"Classifier: {n1} positive / {n0} negative train crops, {len(val_crops)} val candidates")
    for ep in range(epochs):
        net.train()
        t0, tot = time.time(), 0.0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(net(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
        p = predict_probs(net, val_crops, device, tta=False, inner=cfg["inner"], net_size=cfg["net_size"])
        auc = roc_auc(val_labels, p) if len(val_crops) else float("nan")
        history.append({"epoch": ep + 1, "train_loss": round(tot / len(train_items), 4), "val_auc": round(auc, 4)})
        print(f"  epoch {ep + 1}/{epochs}  loss {tot / len(train_items):.4f}  val AUC {auc:.4f}  ({time.time() - t0:.0f}s)")
        if best_state is None or (not np.isnan(auc) and auc > best_auc):
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, out_path)
    return net, history, best_auc
