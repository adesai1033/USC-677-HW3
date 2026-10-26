# CSCI 677 — Homework 3: ResNet-18 & ViT on CINIC-10

## Contents

```
report.pdf           final report
src/                 model + training/eval code
requirements.txt     Python dependencies
run_queue.sh         runs the ViT trainings and all three evaluations in sequence
```

## Setup

Python 3.12. Install the dependencies (torch, torchvision, transformers,
scikit-learn, matplotlib, tensorboard, tqdm):

```bash
pip install -r requirements.txt
```

Download CINIC-10 (https://doi.org/10.7488/ds/2448) and extract it next to `src/`
so the images are at `./CINIC-10/{train,valid,test}`. The data is loaded with
`torchvision.datasets.ImageFolder` using the normalization constants from the
assignment PDF.

Create the output folders before running:

```bash
mkdir -p outputs/checkpoints outputs/logs outputs/plots
```

Training device is picked automatically: `mps` (Apple GPU), then `cuda`, then `cpu`.
The results in the report were produced on an Apple M1 Pro with `mps`.

## Running

```bash
cd src
python train_resnet.py                              # main: 20 epochs, batch 32, Adam 1e-3
python train_vit.py                                 # full fine-tune, 2 epochs, lr 5e-5
python train_vit.py --freeze                        # head-only variant, lr 1e-3
python evaluate.py --model resnet                   # test acc, confusion matrix, failures
python evaluate.py --model vit --name vit_full
python evaluate.py --model vit --name vit_frozen --freeze
```

Checkpoints (best validation accuracy) are saved to `outputs/checkpoints/`, CSV logs
to `outputs/logs/`, and curves/confusion/failure plots to `outputs/plots/`.
