"""CINIC-10 dataset and dataloaders."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
CINIC_DIR = ROOT / "CINIC-10"

# Normalization constants from the assignment PDF.
CINIC_MEAN = (0.47889522, 0.47227842, 0.43047404)
CINIC_STD = (0.24205776, 0.23828046, 0.25874835)

# ViT checkpoints are pretrained at 224x224 with mean/std 0.5
# (see ViTImageProcessor for google/vit-base-patch16-224-in21k).
VIT_SIZE = 224
VIT_MEAN = (0.5, 0.5, 0.5)
VIT_STD = (0.5, 0.5, 0.5)

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def build_transform(for_vit: bool = False, augment: bool = False):
    """32x32 CINIC transform for ResNet, or resized 224x224 for ViT."""
    ops = []
    if augment:
        ops += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    if for_vit:
        ops.append(transforms.Resize(VIT_SIZE))
    mean, std = (VIT_MEAN, VIT_STD) if for_vit else (CINIC_MEAN, CINIC_STD)
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


def get_dataset(split: str, for_vit: bool = False, augment: bool = False):
    return datasets.ImageFolder(
        str(CINIC_DIR / split),
        transform=build_transform(for_vit=for_vit, augment=augment and split == "train"),
    )


def get_loader(
    split: str,
    batch_size: int = 32,
    for_vit: bool = False,
    augment: bool = False,
    num_workers: int = 6,
):
    return DataLoader(
        get_dataset(split, for_vit=for_vit, augment=augment),
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
