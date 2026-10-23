# CSCI 677 — Homework 3: Image Classification with ResNet-18 and ViT

**Name:** Abhijay Desai
**Date:** October 2026

> Skeleton note: all numbers, tables, and figures below are real results from my
> runs. `TODO` marks places to write analysis prose.

## 1. Setup

**Dataset.** CINIC-10: 270,000 32×32 images in 10 classes, split evenly into
train/valid/test (90,000 each). Loaded with `torchvision.datasets.ImageFolder`.
ResNet inputs use the normalization constants from the assignment; ViT inputs
are resized to 224×224 and normalized with mean = std = 0.5, matching the
preprocessing the `google/vit-base-patch16-224-in21k` checkpoint was
pretrained with.

**Hardware.** Apple M1 Pro (16 GB), PyTorch `mps` backend.

**Experiments.**

| | ResNet-18 | ViT full fine-tune | ViT frozen-head |
|---|---|---|---|
| Initialization | from scratch | `google/vit-base-patch16-224-in21k` | same |
| Trainable params | 11,173,962 | 85,806,346 | 7,690 (head only) |
| Input size | 32×32 | 224×224 (upsampled) | 224×224 (upsampled) |
| Epochs | 20 | 2 | 2 |
| Batch size | 32 | 16 | 16 |
| Optimizer | Adam (β₁=0.9, β₂=0.999) | Adam | Adam |
| Learning rate | 1e-3 | 5e-5 | 1e-3 |
| Loss | cross-entropy | cross-entropy | cross-entropy |
| Training time | 67 min | 179 min | 90 min |

**Note on ViT learning rate.** The suggested 1e-3 is used for ResNet and for the
frozen-head variant, but full fine-tuning used 5e-5: TODO — one sentence on why
(large LR destroys pretrained weights early in training; standard practice for
transformer fine-tuning).

**ResNet-18 architecture.** Implemented layer by layer per Figure 1 of the
assignment: 3×3/s1 stem (64 ch), four stages of two BasicBlocks each
(64/128/256/512 ch), stride-2 convolutions with 1×1-conv+BN shortcuts in the
first block of stages 2–4, global average pooling, FC 512→10. Verified stage
output shapes 32→16→8→4 and parameter count against the reference.

## 2. Training curves and validation accuracy

### ResNet-18

![ResNet-18 curves](figures/resnet18_curves.png)

Validation accuracy by epoch (best **74.13%** @ epoch 6):

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| Val acc (%) | 55.19 | 64.57 | 69.30 | 71.52 | 73.73 | **74.13** | 73.18 | 72.64 | 73.71 |

TODO — analysis: validation accuracy plateaus at ~74% by epoch 6 and drifts
slightly down afterward while training loss keeps decreasing → overfitting
after ~epoch 6; no augmentation or weight decay was used in the main setting.

### ViT (full fine-tune)

![ViT full curves](figures/vit_full_curves.png)

| Epoch | 1 | 2 |
|---|---|---|
| Val acc (%) | **93.28** | 92.92 |

TODO — analysis: best accuracy after a single epoch; the small drop in epoch 2
suggests the model already starts to overfit (training loss ≈ 0.13).

### ViT (frozen body, head only)

![ViT frozen curves](figures/vit_frozen_curves.png)

| Epoch | 1 | 2 |
|---|---|---|
| Val acc (%) | 91.81 | **91.86** |

## 3. Test results

Overall top-1 test accuracy (90,000 test images):

| Model | Test accuracy |
|---|---|
| ResNet-18 (scratch) | 73.71% |
| ViT full fine-tune | **93.28%** |
| ViT frozen-head | 91.73% |

### Per-class accuracy

| Class | ResNet-18 | ViT full | ViT frozen |
|---|---|---|---|
| airplane | 84.68 | 95.31 | 95.96 |
| automobile | 86.67 | 93.81 | 92.01 |
| bird | 60.80 | 95.44 | 93.67 |
| cat | 60.81 | 92.48 | 87.79 |
| deer | 69.48 | 90.09 | 90.38 |
| dog | 58.97 | 86.81 | 87.39 |
| frog | 82.16 | 98.44 | 95.59 |
| horse | 84.34 | 95.72 | 94.82 |
| ship | 81.19 | 97.92 | 95.03 |
| truck | 68.04 | 86.78 | 84.64 |

### Confusion matrices

![ResNet-18 confusion](figures/resnet18_confusion.png)
![ViT full confusion](figures/vit_full_confusion.png)
![ViT frozen confusion](figures/vit_frozen_confusion.png)

Largest off-diagonal confusions (true→predicted, counts out of 9,000/class):

- **ResNet-18:** truck→automobile (2,082), cat→dog (1,601), dog→cat (1,191), dog→deer (973), deer→horse (900)
- **ViT full:** truck→automobile (1,046), dog→cat (617), automobile→truck (460), cat→dog (377), airplane→ship (257)
- **ViT frozen:** truck→automobile (1,147), cat→dog (639), dog→cat (624), automobile→truck (533), deer→dog (282)

TODO — analysis: the same semantic confusion pairs dominate all three models
(truck/automobile, cat/dog); pretrained features shrink but do not eliminate
them.

## 4. Failure cases

![ResNet-18 failures](figures/resnet18_failures.png)
![ViT full failures](figures/vit_full_failures.png)
![ViT frozen failures](figures/vit_frozen_failures.png)

Each grid shows the first two misclassified test images of every true class
(rows read left to right in class order).

TODO — pick 3–4 examples from the grids above and discuss: low resolution,
ambiguous pose/background, CINIC's ImageNet-sourced images being noisier than
CIFAR, fine-grained animal confusion, pickup trucks vs cars, etc.

## 5. Comparison and discussion

TODO — write up, covering at least:

1. **ResNet vs ViT:** scratch-trained ResNet-18 (20 epochs) reaches 73.71%;
   a ViT fine-tuned for only 2 epochs reaches 93.28% (+19.6 pts). Pretraining
   on ImageNet-21k provides features that transfer to CINIC-10 even though the
   32×32 images must be upsampled 7× to the ViT's 224×224 input.
2. **Full fine-tune vs frozen head:** training only the 7,690-parameter head
   already yields 91.73% (−1.55 pts vs full fine-tune) at half the training
   time — the pretrained representation, not task-specific adaptation, carries
   most of the performance. The gain from full fine-tuning is uneven across
   classes: largest on cat (+4.7), ship (+2.9) and frog (+2.9), while the
   frozen-head model is slightly better on airplane (−0.7), dog (−0.6) and
   deer (−0.3).
3. **Compute trade-off:** ResNet is far cheaper per epoch (32×32 input,
   11M params) but starts from zero; the ViT costs more per step yet needs
   10× fewer epochs for +19 pts.
4. **Overfitting behavior:** ResNet overfits after ~6 epochs without
   augmentation; ViT full fine-tune after 1 epoch; the frozen variant is the
   most stable.

## 6. Variations tried

- ViT frozen-head vs full fine-tune (required variation) — see above.
- ViT full fine-tune learning rate lowered to 5e-5 (from the suggested 1e-3) —
  TODO: brief justification sentence.
- TODO (optional): if you run any extra variation (augmentation flag
  `--augment` on ResNet, different batch size/LR, SGD), add results here.

## Appendix: reproducing

```
python src/train_resnet.py                  # 20 epochs, batch 32, Adam 1e-3
python src/train_vit.py                     # full fine-tune, 2 epochs, 5e-5
python src/train_vit.py --freeze            # head-only, 2 epochs, 1e-3
python src/evaluate.py --model resnet
python src/evaluate.py --model vit --name vit_full
python src/evaluate.py --model vit --name vit_frozen --freeze
```
