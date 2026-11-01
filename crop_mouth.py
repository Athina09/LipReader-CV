import argparse
import glob
import os

import cv2
import numpy as np
import mediapipe as mp

N_FRAMES = 29
SIZE = 96
LIP_IDS = sorted({i for pair in mp.solutions.face_mesh.FACEMESH_LIPS for i in pair})


def read_frames(path):
    """Read a video file, or a folder of image frames, into a list of BGR arrays."""
    if os.path.isdir(path):
        files = sorted(
            f for ext in ("jpg", "jpeg", "png") for f in glob.glob(os.path.join(path, f"*.{ext}"))
        )
        return [cv2.imread(f) for f in files]
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def lip_boxes(frames, scale):
    """Per-frame (cx, cy, side) in pixels, NaN where no face was found."""
    boxes = np.full((len(frames), 3), np.nan)
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True
    ) as mesh:
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            xs = np.array([lm[j].x * w for j in LIP_IDS])
            ys = np.array([lm[j].y * h for j in LIP_IDS])
            side = max(xs.max() - xs.min(), ys.max() - ys.min()) * scale
            boxes[i] = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, side)
    return boxes


def fill_and_smooth(boxes, win=5):
    """Interpolate missing frames from neighbours, then moving-average to stop shake."""
    idx = np.arange(len(boxes))
    good = ~np.isnan(boxes[:, 0])
    out = np.empty_like(boxes)
    for k in range(3):
        out[:, k] = np.interp(idx, idx[good], boxes[good, k])
    pad = win // 2
    kernel = np.ones(win) / win
    for k in range(3):
        padded = np.pad(out[:, k], pad, mode="edge")
        out[:, k] = np.convolve(padded, kernel, mode="valid")
    return out


def crop(frame, box):
    cx, cy, side = box
    half = int(round(side / 2))
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + 2 * half, y0 + 2 * half
    h, w = frame.shape[:2]
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - w), max(0, y1 - h)
    if pl or pt or pr or pb:
        frame = cv2.copyMakeBorder(frame, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
        x0, x1, y0, y1 = x0 + pl, x1 + pl, y0 + pt, y1 + pt
    patch = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def clip_to_crops(frames, scale=1.8):
    """frames -> ((29, 96, 96) uint8 crops, frames-with-face count, boxes, picks)."""
    raw = lip_boxes(frames, scale)
    found = int((~np.isnan(raw[:, 0])).sum())
    if found == 0:
        return None, 0, None, None
    boxes = fill_and_smooth(raw)
    picks = np.linspace(0, len(frames) - 1, N_FRAMES).round().astype(int)
    crops = np.stack([crop(frames[i], boxes[i]) for i in picks]).astype(np.uint8)
    return crops, found, boxes, picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", help="video file or folder of image frames")
    ap.add_argument("--scale", type=float, default=1.8, help="crop size vs lip width")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    frames = read_frames(args.clip)
    if not frames:
        raise SystemExit(f"no frames read from {args.clip}")

    crops, found, boxes, picks = clip_to_crops(frames, args.scale)
    print(f"face found in {found}/{len(frames)} frames")
    if crops is None:
        raise SystemExit("no face found: 0 frames")

    name = os.path.splitext(os.path.basename(os.path.normpath(args.clip)))[0]
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, f"{name}_mouth.npy"), crops)

    cols = 10
    rows = -(-N_FRAMES // cols)
    sheet = np.zeros((rows * SIZE, cols * SIZE), np.uint8)
    for n, c in enumerate(crops):
        r, col = divmod(n, cols)
        sheet[r * SIZE:(r + 1) * SIZE, col * SIZE:(col + 1) * SIZE] = c
    cv2.imwrite(os.path.join(args.out, f"{name}_preview.png"), sheet)

    mid = picks[len(picks) // 2]
    cx, cy, side = boxes[mid]
    dbg = frames[mid].copy()
    p1 = (int(cx - side / 2), int(cy - side / 2))
    p2 = (int(cx + side / 2), int(cy + side / 2))
    cv2.rectangle(dbg, p1, p2, (0, 255, 0), max(2, dbg.shape[1] // 400))
    cv2.imwrite(os.path.join(args.out, f"{name}_debug.jpg"), dbg)

    print(f"saved {crops.shape} to {args.out}/{name}_mouth.npy (+ preview, debug)")


if __name__ == "__main__":
    main()
