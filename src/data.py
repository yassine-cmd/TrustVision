"""
TrustVision - Data Module (src/data.py)
Shared Data Contract:
- Images converted to 3 channels (RGB).
- Values strictly in [0, 1] range as torch.float32 tensors.
- Resized to 96x96 for CNN, 224x224 for CLIP with BICUBIC interpolation.
- Splits: 70% train (1260) / 10% cal (180) / 20% test (360) governed by configs/splits.json.
- Deterministic seeding with [42, 123, 999].
- ZERO fake/mock data: Hard assertion on exactly 300 real images per class (1,800 total).
"""

import json
import shutil
import zipfile
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms
import requests


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

# Full folder name keywords (exact words only, no short substrings to avoid matching 'train' or 'valid')
NEU_FOLDER_KEYWORDS = {
    "crazing": ("crazing",),
    "inclusion": ("inclusion",),
    "patches": ("patches", "patch"),
    "pitted_surface": ("pitted_surface", "pitted"),
    "rolled-in_scale": ("rolled-in_scale", "rolled_in_scale", "rolled"),
    "scratches": ("scratches", "scratch"),
}

# Standard NEU-CLS filename prefixes (strictly checked with startswith)
NEU_FILE_PREFIXES = {
    "crazing": ("crazing_", "crazing-", "crazing", "cr_"),
    "inclusion": ("inclusion_", "inclusion-", "inclusion", "in_"),
    "patches": ("patches_", "patches-", "patches", "pa_"),
    "pitted_surface": ("pitted_surface_", "pitted_surface", "pitted_", "pitted", "ps_"),
    "rolled-in_scale": ("rolled-in_scale_", "rolled_in_scale_", "rolled_", "rolled", "rs_"),
    "scratches": ("scratches_", "scratches-", "scratches", "sc_"),
}

FIGSHARE_URL = "https://ndownloader.figshare.com/files/54094775"


def match_neu_class(file_path: Path) -> Optional[int]:
    """
    Robustly matches a file path to its NEU defect class (0 to 5).
    1. Inspects parent folder using full keywords only (never short substrings like 'in' that match 'train').
    2. Inspects filename prefixes strictly with startswith (e.g. RS_1.bmp -> rolled-in_scale).
    """
    filename_lower = file_path.name.lower()
    parent_lower = file_path.parent.name.lower()

    # 1. Match against parent folder name using full keywords only
    for c_idx, c_name in enumerate(NEU_CLASSES):
        for keyword in NEU_FOLDER_KEYWORDS[c_name]:
            if keyword in parent_lower:
                return c_idx

    # 2. Match against filename prefix strictly with startswith
    for c_idx, c_name in enumerate(NEU_CLASSES):
        for prefix in NEU_FILE_PREFIXES[c_name]:
            if filename_lower.startswith(prefix):
                return c_idx

    return None


def download_neu_dataset(dest_dir: Union[str, Path] = "data/NEU-CLS") -> Path:
    """
    Downloads and extracts the NEU-CLS dataset.
    Tries kagglehub first; if unauthenticated, falls back to direct Figshare mirror (~27 MB).
    Organizes images into canonical class directories under dest_dir.
    """
    dest_path = Path(dest_dir)
    if dest_path.exists():
        existing_imgs = list(dest_path.glob("**/*.jpg")) + list(dest_path.glob("**/*.bmp")) + list(dest_path.glob("**/*.png"))
        if len(existing_imgs) >= 1800:
            return dest_path

    dest_path.mkdir(parents=True, exist_ok=True)
    temp_dir = dest_path.parent / "temp_neu_download"
    temp_dir.mkdir(parents=True, exist_ok=True)
    zip_file = temp_dir / "NEU-CLS.zip"

    download_success = False

    # Attempt 1: kagglehub
    try:
        import kagglehub
        print("[Dataset] Attempting download via kagglehub...")
        kh_path = kagglehub.dataset_download("yiyang-feng/neu-metal-surface-defects-database")
        temp_dir = Path(kh_path)
        download_success = True
        print(f"[Dataset] KaggleHub download successful to {temp_dir}")
    except Exception as e:
        print(f"[Dataset] KaggleHub not available or unauthenticated ({e}). Trying Figshare direct mirror...")

    # Attempt 2: Direct Figshare Mirror
    if not download_success:
        try:
            print(f"[Dataset] Downloading NEU-CLS archive from {FIGSHARE_URL} (~27 MB)...")
            response = requests.get(FIGSHARE_URL, stream=True, timeout=30)
            response.raise_for_status()
            with open(zip_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            print("[Dataset] Download complete. Extracting...")
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(temp_dir)
            download_success = True
        except Exception as e:
            raise RuntimeError(f"Failed to download NEU-CLS dataset from both KaggleHub and Figshare mirror: {e}")

    # Reorganize into canonical class subfolders using robust prefix matcher
    for c in NEU_CLASSES:
        (dest_path / c).mkdir(parents=True, exist_ok=True)

    found_images = (
        list(temp_dir.glob("**/*.jpg"))
        + list(temp_dir.glob("**/*.bmp"))
        + list(temp_dir.glob("**/*.png"))
    )

    for img_path in found_images:
        c_idx = match_neu_class(img_path)
        if c_idx is not None:
            c_name = NEU_CLASSES[c_idx]
            target_file = dest_path / c_name / img_path.name
            if not target_file.exists():
                shutil.copy2(img_path, target_file)

    # Cleanup zip if downloaded
    if zip_file.exists():
        zip_file.unlink()

    return dest_path


def load_splits_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the pre-computed splits.json configuration file."""
    if config_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        config_path = base_dir / "configs" / "splits.json"

    if not Path(config_path).exists():
        raise FileNotFoundError(f"splits.json not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


class NEUCLSDataset(Dataset):
    """
    Standardized PyTorch Dataset for NEU-CLS.
    Ensures input tensors are in [0, 1], 3 channels, with custom target resolution and BICUBIC interpolation.
    Hard requirement: Exactly 1,800 real images (300 per class). No synthetic fallback permitted.
    """

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        img_size: int = 96,
        transform: Optional[transforms.Compose] = None,
        auto_download: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.samples: List[Tuple[str, int, str]] = []  # (full_path, class_idx, rel_path)

        # Standard transform: Bicubic resize matching CLIP's preprocessor + float32 [0, 1]
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(
                    (img_size, img_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
            ])

        # Resolve dataset location
        self.root_dir = self._resolve_root_dir(root_dir, auto_download=auto_download)
        if not self.root_dir or not self.root_dir.exists():
            raise FileNotFoundError(
                f"NEU-CLS dataset directory not found at {self.root_dir}. "
                "Please enable auto_download or place images in data/NEU-CLS."
            )

        self._load_from_disk()

        # Hard assertion: EXACTLY 300 images per class
        for c_idx, c_name in enumerate(NEU_CLASSES):
            c_count = sum(1 for s in self.samples if s[1] == c_idx)
            if c_count != 300:
                raise RuntimeError(
                    f"Data Integrity Error: Class '{c_name}' has {c_count} images (expected exactly 300). "
                    "Zero mock data permitted; dataset must be complete."
                )

    def _resolve_root_dir(self, root_dir: Optional[Union[str, Path]], auto_download: bool) -> Optional[Path]:
        if root_dir and Path(root_dir).exists():
            return Path(root_dir)

        candidates = [
            Path("data/NEU-CLS"),
            Path("./NEU-CLS"),
            Path("../data/NEU-CLS"),
            Path("/content/NEU-CLS"),
            Path("/content/data/NEU-CLS"),
        ]
        for c in candidates:
            if c.exists() and len(
                list(c.glob("**/*.jpg"))
                + list(c.glob("**/*.bmp"))
                + list(c.glob("**/*.png"))
            ) >= 1800:
                return c

        if auto_download:
            return download_neu_dataset("data/NEU-CLS")
        return None

    def _load_from_disk(self):
        """Scans root_dir for images sorted deterministically by class and filename."""
        self.samples = []
        valid_exts = {".jpg", ".jpeg", ".bmp", ".png"}
        for class_idx, class_name in enumerate(NEU_CLASSES):
            class_folder = self.root_dir / class_name
            if class_folder.is_dir():
                image_files = sorted([
                    p for p in class_folder.iterdir()
                    if p.is_file() and p.suffix.lower() in valid_exts
                ])
            else:
                # Flat root scan using robust matcher
                all_files = sorted([
                    p for p in self.root_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in valid_exts
                ])
                image_files = [f for f in all_files if match_neu_class(f) == class_idx]

            for img_path in image_files:
                rel_path = f"{class_name}/{img_path.name}"
                self.samples.append((str(img_path), class_idx, rel_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item, label, _ = self.samples[idx]
        with Image.open(item) as img:
            img = img.convert("RGB")
            img_tensor = self.transform(img)
            return img_tensor, label


def get_dataloaders(
    root_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    seed: int = 42,
    img_size: int = 96,
    label_fraction: float = 1.0,
    task_classes: Optional[List[int]] = None,
    num_workers: int = 0,
    config_path: Optional[str] = None,
    auto_download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Creates train, cal, and test DataLoaders strictly respecting configs/splits.json.

    Returns:
        (train_loader, cal_loader, test_loader, test_file_paths)
    """
    splits_config = load_splits_config(config_path)
    seed_str = str(seed)
    if seed_str not in splits_config["seeds"]:
        raise ValueError(f"Seed {seed} not in splits.json.")

    seed_splits = splits_config["seeds"][seed_str]

    allowed_label_fractions = (0.01, 0.05, 0.10, 1.0)
    if not any(abs(label_fraction - f) < 1e-4 for f in allowed_label_fractions):
        raise ValueError(
            f"label_fraction={label_fraction} is not supported. "
            f"Choose one of {allowed_label_fractions}."
        )

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

    full_dataset = NEUCLSDataset(
        root_dir=root_dir,
        img_size=img_size,
        auto_download=auto_download,
    )

    def filter_indices(indices: List[int]) -> List[int]:
        if task_classes is None:
            return indices
        filtered = []
        for idx in indices:
            _, label, _ = full_dataset.samples[idx]
            if label in task_classes:
                filtered.append(idx)
        return filtered

    train_indices = filter_indices(train_indices)
    cal_indices = filter_indices(cal_indices)
    test_indices = filter_indices(test_indices)

    train_subset = Subset(full_dataset, train_indices)
    cal_subset = Subset(full_dataset, cal_indices)
    test_subset = Subset(full_dataset, test_indices)

    test_files = [full_dataset.samples[i][2] for i in test_indices]

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=train_generator,
    )
    cal_loader = DataLoader(
        cal_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, cal_loader, test_loader, test_files
