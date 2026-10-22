"""Crop dataset with histopathology augmentation.

Crops are saved at 128 px; a 112 px window is taken (random during training,
centred at test time) and upsampled to the network input size.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, items, train: bool, crop=128, inner=112, net_size=224):
        self.items, self.train = items, train  # items: [(path or uint8 array, label)]
        self.crop, self.inner, self.net_size = crop, inner, net_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        src, y = self.items[i]
        a = src if isinstance(src, np.ndarray) else np.array(Image.open(src).convert("RGB"))
        if self.train:
            ox, oy = np.random.randint(0, self.crop - self.inner + 1, 2)  # +-8 px jitter
            a = a[oy:oy + self.inner, ox:ox + self.inner]
            a = np.rot90(a, np.random.randint(4))                         # rotations
            if np.random.rand() < 0.5:
                a = a[:, ::-1]                                            # flips
        else:
            o = (self.crop - self.inner) // 2
            a = a[o:o + self.inner, o:o + self.inner]
        t = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0
        if self.train:  # stain / colour jitter
            t = (t * (1 + 0.2 * (torch.rand(3, 1, 1) - 0.5)) + 0.1 * (torch.rand(3, 1, 1) - 0.5)).clamp(0, 1)
        t = F.interpolate(t[None], size=self.net_size, mode="bilinear", align_corners=False)[0]
        return (t - MEAN) / STD, torch.tensor(y, dtype=torch.long)


def seed_worker(worker_id):
    """Different numpy augmentation stream per DataLoader worker."""
    np.random.seed((torch.initial_seed() + worker_id) % 2 ** 32)
