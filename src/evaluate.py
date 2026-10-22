"""Evaluate a trained checkpoint on the CINIC-10 test set.

Produces overall + per-class accuracy, a confusion matrix plot, and a grid
of misclassified examples.

  python evaluate.py --model resnet                       # outputs/checkpoints/resnet18.pt
  python evaluate.py --model vit  --name vit_full
  python evaluate.py --model vit  --name vit_frozen --freeze
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from data import CLASSES, CINIC_MEAN, CINIC_STD, VIT_MEAN, VIT_STD, get_loader
from engine import CKPT_DIR, PLOT_DIR, get_device, logits_of
from resnet import ResNet18


@torch.no_grad()
def collect_predictions(model, loader, device, max_failures=32):
    """Returns (y_true, y_pred) over the whole loader plus up to
    `max_failures` misclassified images (tensor, true, pred)."""
    model.eval()
    y_true, y_pred, failures = [], [], []
    for images, labels in loader:
        outputs = logits_of(model(images.to(device)))
        predicted = outputs.argmax(1).cpu()
        y_true.append(labels)
        y_pred.append(predicted)
        if len(failures) < max_failures:
            for i in (predicted != labels).nonzero().flatten():
                failures.append((images[i], labels[i].item(), predicted[i].item()))
                if len(failures) >= max_failures:
                    break
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy(), failures


def plot_confusion(cm, name):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel("predicted"), ax.set_ylabel("true")
    ax.set_title(f"{name}: confusion matrix (test set)")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    out = PLOT_DIR / f"{name}_confusion.png"
    fig.savefig(out, dpi=150)
    print(f"confusion matrix -> {out}")


def plot_failures(failures, name, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    n = min(len(failures), 16)
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for ax, (img, true, pred) in zip(axes.flat, failures[:n]):
        ax.imshow((img * std + mean).clamp(0, 1).permute(1, 2, 0))
        ax.set_title(f"true: {CLASSES[true]}\npred: {CLASSES[pred]}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{name}: misclassified test examples")
    fig.tight_layout()
    out = PLOT_DIR / f"{name}_failures.png"
    fig.savefig(out, dpi=150)
    print(f"failure examples -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("resnet", "vit"), required=True)
    p.add_argument("--name", default=None,
                   help="checkpoint/plot name (default: resnet18 or vit_full)")
    p.add_argument("--freeze", action="store_true",
                   help="rebuild the ViT with a frozen body (must match training)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=6)
    args = p.parse_args()

    name = args.name or ("resnet18" if args.model == "resnet" else "vit_full")
    device = get_device()

    if args.model == "resnet":
        model = ResNet18(num_classes=10)
        mean, std = CINIC_MEAN, CINIC_STD
    else:
        from train_vit import build_vit
        model = build_vit(freeze=args.freeze)
        mean, std = VIT_MEAN, VIT_STD

    state = torch.load(CKPT_DIR / f"{name}.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)

    loader = get_loader("test", batch_size=args.batch_size,
                        for_vit=(args.model == "vit"),
                        num_workers=args.num_workers)
    y_true, y_pred, failures = collect_predictions(model, loader, device)

    acc = (y_true == y_pred).mean() * 100
    print(f"\n{name} test accuracy: {acc:.2f}%\n")
    cm = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1) * 100
    for cls, a in zip(CLASSES, per_class):
        print(f"  {cls:<12} {a:6.2f}%")

    np.savetxt(PLOT_DIR / f"{name}_confusion.csv", cm, fmt="%d", delimiter=",")
    plot_confusion(cm, name)
    plot_failures(failures, name, mean, std)


if __name__ == "__main__":
    main()
