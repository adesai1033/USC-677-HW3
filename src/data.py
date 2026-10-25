from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
CINIC_DIR = ROOT / "CINIC-10"

CINIC_MEAN = (0.47889522, 0.47227842, 0.43047404)
CINIC_STD = (0.24205776, 0.23828046, 0.25874835)

VIT_SIZE = 224
VIT_MEAN = (0.5, 0.5, 0.5)
VIT_STD = (0.5, 0.5, 0.5)

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


def build_transform(for_vit=False, augment=False):
    ops = []
    if augment:
        ops += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if for_vit:
        ops.append(transforms.Resize(VIT_SIZE))
    mean, std = (VIT_MEAN, VIT_STD) if for_vit else (CINIC_MEAN, CINIC_STD)
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


def get_dataset(split, for_vit=False, augment=False):
    transform = build_transform(for_vit=for_vit, augment=augment and split == "train")
    return datasets.ImageFolder(str(CINIC_DIR / split), transform=transform)


def get_loader(split, batch_size=32, for_vit=False, augment=False, num_workers=6):
    dataset = get_dataset(split, for_vit=for_vit, augment=augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                      num_workers=num_workers, persistent_workers=num_workers > 0)
