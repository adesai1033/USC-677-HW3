# CSCI 677 — Homework 3: ResNet-18 & ViT on CINIC-10

## Setup

```bash
source .venv/bin/activate          # Python 3.12 venv (created with uv)
```

Dependencies are pinned in `requirements.txt` (torch, torchvision, transformers,
scikit-learn, matplotlib, tensorboard, tqdm). Training device: `mps` (Apple GPU).

## Layout

```
CINIC-10/            dataset (train/ valid/ test/, 90k images each, 10 classes)
src/                 model + training/eval code
outputs/checkpoints/ saved model weights
outputs/plots/       loss & accuracy curves, confusion matrices
outputs/logs/        training logs / tensorboard runs
report/              final report
```

## Running

```bash
cd src
../.venv/bin/python train_resnet.py                 # main: 20 epochs, batch 32, Adam 1e-3
../.venv/bin/python train_vit.py                    # full fine-tune, 2 epochs, lr 5e-5
../.venv/bin/python train_vit.py --freeze           # head-only variant, lr 1e-3
../.venv/bin/python evaluate.py --model resnet      # test acc, confusion matrix, failures
../.venv/bin/python evaluate.py --model vit --name vit_full
../.venv/bin/python evaluate.py --model vit --name vit_frozen --freeze
```

Checkpoints (best validation accuracy) land in `outputs/checkpoints/`, CSV logs in
`outputs/logs/`, and curves/confusion/failure plots in `outputs/plots/`.

## Dataset

CINIC-10 (https://doi.org/10.7488/ds/2448), extracted to `./CINIC-10/{train,valid,test}`,
loaded with `torchvision.datasets.ImageFolder` using the normalization constants from
the assignment PDF.
