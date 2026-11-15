"""
TrustVision - Data Module (src/data.py)
Shared Data Contract:
- Images converted to 3 channels (RGB).
- Values strictly in [0, 1] range as torch.float32 tensors.
- Resized to 96x96 for CNN, 224x224 for CLIP.
- Splits: 70% train (1260) / 10% cal (180) / 20% test (360) governed by configs/splits.json.
- Deterministic seeding with [42, 123, 999].
"""

import os
import json
import glob
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms


# Canonical NEU-CLS class names (alphabetical / indexed 0 to 5)
NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(NEU_CLASSES)}


def load_splits_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the pre-computed splits.json configuration file."""
    if config_path is None:
        # Default relative to this source file
        base_dir = Path(__file__).resolve().parent.parent
        config_path = base_dir / "configs" / "splits.json"

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


class NEUCLSDataset(Dataset):
    """
    Standardized PyTorch Dataset for NEU-CLS.
    Ensures input tensors are in [0, 1], 3 channels, with custom target resolution.
    Supports synthetic fallback mode for offline/smoke-testing.
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        img_size: int = 96,
        transform: Optional[transforms.Compose] = None,
        use_mock: bool = False,
    ):
        super().__init__()
        self.root_dir = Path(root_dir) if root_dir else None
        self.img_size = img_size
        self.use_mock = use_mock
        self.samples: List[Tuple[Any, int]] = []

        # Standard transform: Grayscale -> RGB, Resize, ToTensor (scales to [0.0, 1.0])
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),  # Outputs [0.0, 1.0] range
            ])

        if not self.use_mock and self.root_dir and self.root_dir.exists():
            self._load_from_disk()
        else:
            self._init_mock_dataset()

    def _load_from_disk(self):
        """Scans root_dir for images sorted deterministically by class."""
        self.samples = []
        for class_idx, class_name in enumerate(NEU_CLASSES):
            # Look for subdirectories or class-named patterns
            class_folder = self.root_dir / class_name
            if class_folder.is_dir():
                image_files = sorted(
                    list(class_folder.glob("*.bmp"))
                    + list(class_folder.glob("*.jpg"))
                    + list(class_folder.glob("*.png"))
                    + list(class_folder.glob("*.JPEG"))
                )
            else:
                # Look for files matching prefix in root
                image_files = sorted(
                    list(self.root_dir.glob(f"{class_name}*.*"))
                    + list(self.root_dir.glob(f"{class_name[:2]}*.*"))
                )

            for img_path in image_files:
                self.samples.append((str(img_path), class_idx))

        if len(self.samples) == 0:
            print(f"[Warning] No images found in {self.root_dir}. Falling back to mock dataset.")
            self._init_mock_dataset()

    def _init_mock_dataset(self):
        """Creates 1,800 mock samples (300 per class) for testing without full data."""
        self.use_mock = True
        self.samples = []
        for class_idx in range(len(NEU_CLASSES)):
            for i in range(300):
                self.samples.append((f"mock_{class_idx}_{i}", class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item, label = self.samples[idx]

        if self.use_mock:
            # Deterministic pseudo-random image for reproducible testing
            g = torch.Generator().manual_seed(idx)
            img_tensor = torch.rand((3, self.img_size, self.img_size), generator=g)
            return img_tensor, label

        # Load real image
        try:
            with Image.open(item) as img:
                # Convert grayscale to 3 channels RGB
                img = img.convert("RGB")
                img_tensor = self.transform(img)
                return img_tensor, label
        except Exception as e:
            # Return zero tensor on error
            print(f"Error loading image {item}: {e}")
            return torch.zeros((3, self.img_size, self.img_size)), label


def get_dataloaders(
    root_dir: Optional[str] = None,
    batch_size: int = 32,
    seed: int = 42,
    img_size: int = 96,
    label_fraction: float = 1.0,
    task_classes: Optional[List[int]] = None,
    num_workers: int = 0,
    config_path: Optional[str] = None,
    use_mock: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, cal, and test DataLoaders strictly respecting configs/splits.json.

    Args:
        root_dir: Path to NEU-CLS dataset.
        batch_size: Mini-batch size.
        seed: Seed partition to use (42, 123, or 999).
        img_size: 96 for CNN, 224 for CLIP.
        label_fraction: 1.0 (100%), 0.10 (10%), 0.05 (5%), or 0.01 (1%).
        task_classes: Optional list of class IDs (e.g. [0, 1, 2] for Continual Learning Task 1).
        num_workers: DataLoader workers.
        config_path: Optional path to splits.json.
        use_mock: Force mock synthetic dataset.

    Returns:
        (train_loader, cal_loader, test_loader)
    """
    splits_config = load_splits_config(config_path)
    seed_str = str(seed)
    if seed_str not in splits_config["seeds"]:
        raise ValueError(f"Seed {seed} not in splits.json. Choose from {list(splits_config['seeds'].keys())}")

    seed_splits = splits_config["seeds"][seed_str]

    # Select appropriate training subset based on fraction
    if abs(label_fraction - 0.01) < 1e-4:
        train_indices = seed_splits["subsets_train"]["1pct"]
    elif abs(label_fraction - 0.05) < 1e-4:
        train_indices = seed_splits["subsets_train"]["5pct"]
    elif abs(label_fraction - 0.10) < 1e-4:
        train_indices = seed_splits["subsets_train"]["10pct"]
    else:
        train_indices = seed_splits["train"]

    cal_indices = seed_splits["cal"]
    test_indices = seed_splits["test"]

    # Base dataset
    full_dataset = NEUCLSDataset(root_dir=root_dir, img_size=img_size, use_mock=use_mock)

    # Optional class filtering for Continual Learning
    def filter_indices(indices: List[int]) -> List[int]:
        if task_classes is None:
            return indices
        filtered = []
        for idx in indices:
            _, label = full_dataset.samples[idx]
            if label in task_classes:
                filtered.append(idx)
        return filtered

    train_indices = filter_indices(train_indices)
    cal_indices = filter_indices(cal_indices)
    test_indices = filter_indices(test_indices)

    train_subset = Subset(full_dataset, train_indices)
    cal_subset = Subset(full_dataset, cal_indices)
    test_subset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    cal_loader = DataLoader(
        cal_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, cal_loader, test_loader
