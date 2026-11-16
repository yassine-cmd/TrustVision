"""
Export frozen 512-d CLIP image embeddings for all 1,800 NEU-CLS images.
Produces: results/clip_embeddings.npz
For Member C (S1 low-label linear probes across seeds 42, 123, 999).
"""

import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.clip_attn import load_clip, extract_all_clip_embeddings
from src.data import NEUCLSDataset


def export_clip_embeddings(
    output_path: str = "results/clip_embeddings.npz",
    splits_path: str = "configs/splits.json",
    batch_size: int = 64,
):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    splits_file = Path(splits_path)

    # 1. Load splits config & compute SHA256
    with open(splits_file, "rb") as f:
        splits_bytes = f.read()
    splits_sha256 = hashlib.sha256(splits_bytes).hexdigest()
    splits_cfg = json.loads(splits_bytes.decode("utf-8"))
    expected_all_files = splits_cfg["all_files"]
    assert len(expected_all_files) == 1800, f"Expected 1800 files in splits.json, got {len(expected_all_files)}"

    # 2. Load dataset in canonical order (img_size=224 for CLIP)
    print("[CLIP Export] Loading full NEU-CLS dataset (1,800 images at 224x224)...")
    full_dataset = NEUCLSDataset(root_dir="data/NEU-CLS", img_size=224, auto_download=True)
    dataset_paths = [s[2] for s in full_dataset.samples]

    # Verify exact path alignment with configs/splits.json
    assert dataset_paths == expected_all_files, (
        "Dataset sample order does not match configs/splits.json all_files!"
    )
    print("[CLIP Export] Verified 100% path alignment with configs/splits.json (all 1,800 samples match 1:1).")

    # 3. Load CLIP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CLIP Export] Loading CLIP model (openai/clip-vit-base-patch32) on {device}...")
    clip_model, _ = load_clip("openai/clip-vit-base-patch32", device=device)

    # 4. Extract embeddings
    data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    print(f"[CLIP Export] Extracting normalized 512-d embeddings ({len(full_dataset)} images)...")
    features, labels = extract_all_clip_embeddings(clip_model, data_loader, device=device)

    # 5. Assertions
    assert features.shape == (1800, 512), f"Expected shape (1800, 512), got {features.shape}"
    assert labels.shape == (1800,), f"Expected shape (1800,), got {labels.shape}"
    assert features.dtype == np.float32 or np.issubdtype(features.dtype, np.floating)

    # Verify L2-normalization
    norms = np.linalg.norm(features, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5, err_msg="Features are not L2-normalized!")
    print(f"[CLIP Export] Embeddings validated: shape={features.shape}, L2 norms in [{norms.min():.5f}, {norms.max():.5f}]")

    # 6. Save compressed
    paths_arr = np.array(dataset_paths)
    np.savez_compressed(
        output_file,
        features=features.astype(np.float32),
        labels=labels.astype(np.int64),
        paths=paths_arr,
        splits_sha256=splits_sha256,
    )
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"[CLIP Export] Successfully saved compressed embeddings to {output_file} ({file_size_mb:.2f} MB)")

    # 7. Verification of saved file
    loaded = np.load(output_file)
    assert set(loaded.files) == {"features", "labels", "paths", "splits_sha256"}
    assert loaded["features"].shape == (1800, 512)
    assert loaded["labels"].shape == (1800,)
    assert len(loaded["paths"]) == 1800
    assert str(loaded["splits_sha256"]) == splits_sha256
    print("[CLIP Export] Verified saved .npz integrity. Ready for Member C!")


if __name__ == "__main__":
    export_clip_embeddings()
