# LipReader-CV

Word-level lip reading on MIRACL-VC1 with a small 3D CNN.

## Pipeline
1. `crop_mouth.py <clip>` – Face Mesh finds the lips; writes a (29, 96, 96) grayscale mouth clip plus a preview.
2. `prepare_data.py --root <MIRACL words dir>` – crops every clip and writes `data/manifest.csv` (path, label, speaker).
3. `train.py --manifest data/manifest.csv --val-speakers F10 M07 --test-speakers F11 M08` – trains with a speaker-disjoint split, early stopping, and saves `best.pt`.
4. `server.py --weights best.pt` – web UI at http://localhost:5000 (webcam recording or clip upload).

## Setup
Python 3.12 (mediapipe 0.10.14 needs <= 3.12):
```
uv venv --python 3.12 --managed-python lipenv312
uv pip install --python lipenv312/bin/python mediapipe==0.10.14 opencv-python "numpy<2" torch flask kagglehub
```
