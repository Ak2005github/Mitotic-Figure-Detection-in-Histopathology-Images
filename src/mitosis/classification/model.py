"""EfficientNet-B0 binary classifier and test-time-augmented prediction."""

import numpy as np
import torch
import torch.nn as nn

from .dataset import CropDataset


def build_model(pretrained=True):
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m


def load_classifier(path, device):
    net = build_model(pretrained=False)
    net.load_state_dict(torch.load(path, map_location=device))
    return net.to(device).eval()


@torch.no_grad()
def predict_probs(net, crops, device, batch=256, tta=True, inner=112, net_size=224):
    """P(mitosis) for a list/array of 128 px uint8 crops. tta=True averages 8
    flip/rotation views; tta=False uses identity + horizontal flip."""
    net.eval()
    if len(crops) == 0:
        return np.zeros(0)
    ds = CropDataset([(c, 0) for c in crops], train=False, crop=crops[0].shape[0], inner=inner, net_size=net_size)
    probs = []
    for i in range(0, len(ds), batch):
        x = torch.stack([ds[j][0] for j in range(i, min(i + batch, len(ds)))]).to(device)
        views = ([torch.rot90(v, k, (2, 3)) for v in (x, x.flip(3)) for k in range(4)] if tta else [x, x.flip(3)])
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            p = torch.stack([net(v).float().softmax(1)[:, 1] for v in views]).mean(0)
        probs.append(p.cpu())
    return torch.cat(probs).numpy()
