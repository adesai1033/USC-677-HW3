"""Fine-tune a pre-trained ViT on CINIC-10 with a fresh 10-class head.

Two modes:
  full fine-tune (default):  python train_vit.py
  head-only (frozen body):   python train_vit.py --freeze

Note on learning rate: the assignment's suggested 1e-3 works for the
head-only variant, but full fine-tuning a pretrained ViT needs a much
smaller rate (5e-5 here) or the pretrained weights are destroyed in the
first few hundred steps. Mention this choice in the report.
"""

import argparse

import torch
from transformers import ViTForImageClassification

from data import get_loader
from engine import train

MODEL_ID = "google/vit-base-patch16-224-in21k"


def build_vit(freeze: bool = False):
    # num_labels=10 discards the pretrained head and adds a fresh 10-way one.
    model = ViTForImageClassification.from_pretrained(MODEL_ID, num_labels=10)
    if freeze:
        for param in model.vit.parameters():
            param.requires_grad = False
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=None,
                   help="default: 5e-5 full fine-tune, 1e-3 with --freeze")
    p.add_argument("--freeze", action="store_true",
                   help="freeze everything except the classification head")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=6)
    args = p.parse_args()

    lr = args.lr if args.lr is not None else (1e-3 if args.freeze else 5e-5)
    name = "vit_frozen" if args.freeze else "vit_full"

    torch.manual_seed(0)
    model = build_vit(freeze=args.freeze)
    train_loader = get_loader("train", batch_size=args.batch_size,
                              for_vit=True, num_workers=args.num_workers)
    valid_loader = get_loader("valid", batch_size=64,
                              for_vit=True, num_workers=args.num_workers)
    train(model, train_loader, valid_loader, name=name,
          epochs=args.epochs, lr=lr,
          log_every=args.log_every, val_every=1)


if __name__ == "__main__":
    main()
