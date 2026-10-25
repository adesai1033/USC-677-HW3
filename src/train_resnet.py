import argparse

import torch

from data import get_loader
from engine import train
from resnet import ResNet18


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--name", default="resnet18")
    args = p.parse_args()

    torch.manual_seed(0)
    model = ResNet18(num_classes=10)
    train_loader = get_loader("train", batch_size=args.batch_size, augment=args.augment,
                              num_workers=args.num_workers)
    valid_loader = get_loader("valid", batch_size=256, num_workers=args.num_workers)
    train(model, train_loader, valid_loader, name=args.name, epochs=args.epochs, lr=args.lr,
          log_every=args.log_every, val_every=args.val_every)


if __name__ == "__main__":
    main()
