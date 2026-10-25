import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "outputs" / "checkpoints"
LOG_DIR = ROOT / "outputs" / "logs"
PLOT_DIR = ROOT / "outputs" / "plots"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def logits_of(output):
    return output.logits if hasattr(output, "logits") else output


@torch.no_grad()
def compute_accuracy(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = logits_of(model(images))
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return correct / total * 100


def train(model, train_loader, valid_loader, name, epochs, lr, log_every=100, val_every=1, device=None):
    device = device or get_device()
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr, betas=(0.9, 0.999))
    loss_fn = nn.CrossEntropyLoss()

    loss_rows, val_rows = [], []
    best_acc, step = 0.0, 0
    ckpt_path = CKPT_DIR / f"{name}.pt"
    t0 = time.time()
    print(f"[{name}] device={device} epochs={epochs} lr={lr} "
          f"trainable params={sum(p.numel() for p in trainable):,}")

    for epoch in range(1, epochs + 1):
        model.train()
        running, n_running = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(logits_of(model(images)), labels)
            loss.backward()
            optimizer.step()

            step += 1
            running += loss.item()
            n_running += 1
            if step % log_every == 0:
                avg = running / n_running
                loss_rows.append((step, epoch, avg))
                running, n_running = 0.0, 0
                print(f"[{name}] epoch {epoch} step {step} loss {avg:.4f} ({time.time() - t0:.0f}s)")

        if epoch % val_every == 0 or epoch == epochs:
            acc = compute_accuracy(model, valid_loader, device)
            val_rows.append((step, epoch, acc))
            print(f"[{name}] epoch {epoch} validation accuracy {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), ckpt_path)
                print(f"[{name}] saved best checkpoint -> {ckpt_path}")

    write_csv(LOG_DIR / f"{name}_loss.csv", ("step", "epoch", "loss"), loss_rows)
    write_csv(LOG_DIR / f"{name}_val.csv", ("step", "epoch", "val_acc"), val_rows)
    plot_curves(name, loss_rows, val_rows)
    print(f"[{name}] done in {(time.time() - t0) / 60:.1f} min; best val acc {best_acc:.2f}%")
    return best_acc


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def plot_curves(name, loss_rows, val_rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot([r[0] for r in loss_rows], [r[2] for r in loss_rows])
    ax1.set_xlabel("step")
    ax1.set_ylabel("training loss")
    ax1.set_title(f"{name}: training loss")
    ax2.plot([r[1] for r in val_rows], [r[2] for r in val_rows], marker="o")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("validation accuracy (%)")
    ax2.set_title(f"{name}: validation accuracy")
    fig.tight_layout()
    out = PLOT_DIR / f"{name}_curves.png"
    fig.savefig(out, dpi=150)
    print(f"[{name}] curves -> {out}")
