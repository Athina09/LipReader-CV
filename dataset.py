import csv
import random

import numpy as np
import torch
from torch.utils.data import Dataset

CROP = 88  # random crop size out of the 96x96 stored crops


def read_manifest(path):
    """CSV with columns: path,label,speaker -> list of dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def split_by_speaker(rows, val_speakers, test_speakers):
    """Speaker-disjoint split, so no face appears in two sets."""
    val_speakers, test_speakers = set(val_speakers), set(test_speakers)
    train = [r for r in rows if r["speaker"] not in val_speakers | test_speakers]
    val = [r for r in rows if r["speaker"] in val_speakers]
    test = [r for r in rows if r["speaker"] in test_speakers]
    return train, val, test


class ClipDataset(Dataset):
    """Loads (29, 96, 96) uint8 .npy clips; returns (1, 29, 88, 88) floats in 0-1."""

    def __init__(self, rows, label_to_idx, train):
        self.rows = rows
        self.label_to_idx = label_to_idx
        self.train = train

    def __len__(self):
        return len(self.rows)

    def augment(self, clip):
        t, h, w = clip.shape
        # random crop (same box for every frame)
        y, x = random.randint(0, h - CROP), random.randint(0, w - CROP)
        clip = clip[:, y:y + CROP, x:x + CROP].astype(np.float32) / 255.0
        # brightness / contrast
        clip = (clip - 0.5) * random.uniform(0.8, 1.2) + 0.5 + random.uniform(-0.1, 0.1)
        # frame dropping: repeat the previous frame for a few random frames
        for i in random.sample(range(1, t), k=random.randint(0, 3)):
            clip[i] = clip[i - 1]
        return np.clip(clip, 0.0, 1.0)

    def center(self, clip):
        t, h, w = clip.shape
        y, x = (h - CROP) // 2, (w - CROP) // 2
        return clip[:, y:y + CROP, x:x + CROP].astype(np.float32) / 255.0

    def __getitem__(self, i):
        row = self.rows[i]
        clip = np.load(row["path"])
        clip = self.augment(clip) if self.train else self.center(clip)
        x = torch.from_numpy(np.ascontiguousarray(clip)).unsqueeze(0)
        return x, self.label_to_idx[row["label"]]
