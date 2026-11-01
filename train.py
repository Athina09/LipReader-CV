import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ClipDataset, read_manifest, split_by_speaker
from model import LipNet3D


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"  # mps is no faster than the CPU for 3D convs


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            n += len(labels)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--val-speakers", nargs="+", required=True)
    ap.add_argument("--test-speakers", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--patience", type=int, default=10, help="early-stopping epochs")
    ap.add_argument("--save", default="best.pt")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    labels = sorted({r["label"] for r in rows})
    label_to_idx = {l: i for i, l in enumerate(labels)}
    train_rows, val_rows, test_rows = split_by_speaker(rows, args.val_speakers, args.test_speakers)
    print(f"{len(labels)} classes | train {len(train_rows)} val {len(val_rows)} test {len(test_rows)}")

    train_dl = DataLoader(ClipDataset(train_rows, label_to_idx, True), args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(ClipDataset(val_rows, label_to_idx, False), args.batch_size)
    test_dl = DataLoader(ClipDataset(test_rows, label_to_idx, False), args.batch_size)

    device = pick_device()
    model = LipNet3D(len(labels), width=args.width).to(device)
    print(f"device {device} | {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc, stale = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, device, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_dl, device, criterion)
        print(f"epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.2f} | val loss {va_loss:.3f} acc {va_acc:.2f}")
        if va_acc > best_acc:
            best_acc, stale = va_acc, 0
            torch.save({"model": model.state_dict(), "labels": labels, "width": args.width}, args.save)
        else:
            stale += 1
            if stale >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(torch.load(args.save)["model"])
    _, te_acc = run_epoch(model, test_dl, device, criterion)
    print(f"best val acc {best_acc:.2f} | test acc {te_acc:.2f} (speakers never seen in training)")


if __name__ == "__main__":
    main()
