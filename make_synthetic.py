"""Write a fake dataset (manifest + .npy clips) to test the training code end to end.

Each class is a bright "mouth" blob whose opening oscillates at a class-specific
frequency, so a model that works can tell the classes apart.
"""
import argparse
import csv
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="synthetic")
ap.add_argument("--classes", type=int, default=4)
ap.add_argument("--speakers", type=int, default=5)
ap.add_argument("--reps", type=int, default=6)
args = ap.parse_args()

rng = np.random.default_rng(0)
os.makedirs(args.out, exist_ok=True)
yy, xx = np.mgrid[:96, :96]
rows = []
for s in range(args.speakers):
    base = rng.uniform(60, 140)  # per-speaker brightness
    for c in range(args.classes):
        for r in range(args.reps):
            phase = rng.uniform(0, 2 * np.pi)
            clip = np.empty((29, 96, 96), np.uint8)
            for t in range(29):
                open_ = 4 + 8 * (0.5 + 0.5 * np.sin(2 * np.pi * (c + 1) * t / 29 + phase))
                img = base + 30 * np.exp(-(((xx - 48) / 20.0) ** 2 + ((yy - 48) / open_) ** 2))
                img += rng.normal(0, 8, img.shape)
                clip[t] = np.clip(img, 0, 255)
            path = os.path.join(args.out, f"S{s:02d}_w{c}_r{r}.npy")
            np.save(path, clip)
            rows.append({"path": path, "label": f"w{c}", "speaker": f"S{s:02d}"})

with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["path", "label", "speaker"])
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} clips to {args.out}/")
