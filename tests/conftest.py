"""
TrustVision - Global Pytest Fixtures & Session Setup (tests/conftest.py)
========================================================================
Shared, session-scoped fixtures to avoid redundant disk I/O and weight loading.
"""

from pathlib import Path
import pytest
from PIL import Image
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="session")
def sample_steel_pil(project_root: Path) -> Image.Image:
    """Returns a realistic or valid 200x200 sample image for UI and inspection tests."""
    crazing_sample = project_root / "data/NEU-CLS/crazing/crazing_1.jpg"
    if crazing_sample.exists():
        return Image.open(crazing_sample).convert("RGB")
    
    # Deterministic synthetic fallback if data directory not present
    rng = np.random.RandomState(42)
    arr = (rng.rand(200, 200) * 200 + 30).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


@pytest.fixture(scope="session")
def sample_steel_tensor() -> torch.Tensor:
    """Returns a normalized [0, 1] tensor of shape (1, 3, 96, 96)."""
    torch.manual_seed(42)
    return torch.rand(1, 3, 96, 96)
