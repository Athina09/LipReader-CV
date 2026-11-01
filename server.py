"""Web UI + API for the lip reader.  python server.py --weights best.pt  ->  http://localhost:5000"""
import argparse
import base64
import os
import tempfile

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

from crop_mouth import clip_to_crops, read_frames
from dataset import CROP
from model import LipNet3D

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="best.pt")
ap.add_argument("--port", type=int, default=5000)
args = ap.parse_args()

ckpt = torch.load(args.weights, map_location="cpu")
labels = ckpt["labels"]
model = LipNet3D(len(labels), width=ckpt["width"])
model.load_state_dict(ckpt["model"])
model.eval()

app = Flask(__name__, static_folder="frontend")


def png_b64(img):
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


@app.get("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.post("/predict")
def predict():
    upload = request.files.get("video")
    if upload is None:
        return jsonify(error="no video uploaded"), 400
    suffix = os.path.splitext(upload.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        upload.save(tmp.name)
        frames = read_frames(tmp.name)
    if len(frames) < 2:
        return jsonify(error="could not read frames from that video"), 400

    crops, found, _, _ = clip_to_crops(frames)
    if crops is None:
        return jsonify(error="no face found in the video"), 422

    y = x = (crops.shape[1] - CROP) // 2
    clip = crops[:, y:y + CROP, x:x + CROP].astype(np.float32) / 255.0
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(clip)[None, None]), dim=1)[0]
    order = probs.argsort(descending=True)
    strip = np.hstack(list(crops[::3]))  # 10 evenly spaced tiles
    return jsonify(
        predictions=[{"label": labels[i], "prob": float(probs[i])} for i in order],
        face_frames=found,
        total_frames=len(frames),
        mouth_strip=png_b64(strip),
    )


if __name__ == "__main__":
    app.run(port=args.port)
