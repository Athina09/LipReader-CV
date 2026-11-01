"""Crop every MIRACL-VC1 word clip to (29, 96, 96) and write data/manifest.csv.

Layout read:  <root>/<speaker>/words/<word 01-10>/<rep 01-10>/color_NNN.jpg
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

from crop_mouth import clip_to_crops

WORDS = {
    "01": "Begin", "02": "Choose", "03": "Connection", "04": "Navigation", "05": "Next",
    "06": "Previous", "07": "Start", "08": "Stop", "09": "Hello", "10": "Web",
}

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="the folder holding F01, F02, ... M08")
ap.add_argument("--out", default="data")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
rows, skipped = [], []
clips = sorted(glob.glob(os.path.join(args.root, "*", "words", "*", "*")))
for n, clip_dir in enumerate(clips, 1):
    rep_id = os.path.basename(clip_dir)
    word_id = os.path.basename(os.path.dirname(clip_dir))
    speaker = clip_dir.split(os.sep)[-4]
    files = sorted(glob.glob(os.path.join(clip_dir, "color_*.jpg")))
    frames = [f for f in (cv2.imread(p) for p in files) if f is not None]
    crops, found, _, _ = clip_to_crops(frames) if frames else (None, 0, None, None)
    if crops is None:
        skipped.append(clip_dir)
        continue
    path =os.path.join(args.out, f"{speaker}_{word_id}_{rep_id}.npy")
    np.save(path, crops)
    rows.append({"path": path, "label": WORDS[word_id], "speaker": speaker})
    if n % 50 == 0:
        print(f"{n}/{len(clips)}", flush=True)

with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["path", "label", "speaker"])
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} clips, skipped {len(skipped)} (no face): {skipped[:5]}")
