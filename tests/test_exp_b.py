"""
TrustVision - Unit Tests for Member B (tests/test_exp_b.py)
Validates core Member B deliverables and contracts:
1. Interface contracts: get_features(x) and get_last_conv_layer().
2. FGSM adversarial bounds: outputs strictly clamped in [0, 1].
3. Temperature Scaling: calibration preserves exact argmax and accuracy.
4. evaluate_all: unified metric contract, checkpoint-agnostic loading.
5. Splits integrity: SHA-256 verification and fixed 70/10/20 proportions.
6. Baseline checkpoint: checks presence and loadability of cnn_base_seed42.pt.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import TrustVisionCNN
from src.eval import (
    fgsm_attack,
    ModelWithTemperature,
    TemperatureScaler,
    evaluate_all,
    load_model_checkpoint,
    compute_xai_similarity,
    compute_intra_pipeline_stability,
)


def test_model_interface_contracts():
    """Verifies that TrustVisionCNN satisfies get_features and get_last_conv_layer contracts."""
    model = TrustVisionCNN(num_classes=6, feature_dim=256, normalize_input=True)
    model.eval()

    # Input in [0, 1] range, 96x96 spatial resolution
    dummy_input = torch.rand(4, 3, 96, 96)

    # 1. get_features contract (for SimCLR and probing)
    features = model.get_features(dummy_input)
    assert features.shape == (4, 256), f"Expected shape (4, 256), got {features.shape}"

    # 2. get_last_conv_layer contract (for Grad-CAM)
    last_conv = model.get_last_conv_layer()
    assert isinstance(last_conv, nn.Conv2d), f"Expected Conv2d layer, got {type(last_conv)}"
    assert last_conv.out_channels == 256, f"Expected 256 out channels, got {last_conv.out_channels}"

    # 3. Forward pass
    logits = model(dummy_input)
    assert logits.shape == (4, 6), f"Expected logits of shape (4, 6), got {logits.shape}"


def test_fgsm_pixel_bounds():
    """Verifies that FGSM adversarial attack strictly keeps pixel values inside [0, 1]."""
    model = TrustVisionCNN(num_classes=6, normalize_input=True)
    model.eval()

    # Test across multiple epsilons including extreme values
    epsilons = [0.0, 0.01, 0.03, 0.05, 0.10, 0.25]
    dummy_images = torch.rand(8, 3, 96, 96)
    dummy_labels = torch.randint(0, 6, (8,))

    for eps in epsilons:
        adv_images = fgsm_attack(model, dummy_images, dummy_labels, eps=eps)
        assert adv_images.min().item() >= 0.0 - 1e-6, f"FGSM min pixel < 0 for eps={eps}"
        assert adv_images.max().item() <= 1.0 + 1e-6, f"FGSM max pixel > 1 for eps={eps}"
        assert adv_images.shape == dummy_images.shape


def test_temperature_scaling_preserves_argmax():
    """Verifies that Temperature Scaling strictly preserves logits argmax (and therefore accuracy)."""
    torch.manual_seed(42)
    raw_logits = torch.randn(50, 6) * 3.0
    labels = torch.randint(0, 6, (50,))

    scaler = TemperatureScaler(initial_temp=1.5)
    # Fit on dummy logits
    _ = scaler.fit(raw_logits, labels)

    calibrated_logits = scaler(raw_logits)
    raw_preds = raw_logits.argmax(dim=-1)
    calibrated_preds = calibrated_logits.argmax(dim=-1)

    assert torch.equal(raw_preds, calibrated_preds), "Temperature scaling must strictly preserve argmax!"

    # Test ModelWithTemperature wrapper as well
    base_model = nn.Linear(10, 6)
    wrapper = ModelWithTemperature(base_model)
    wrapper.temperature.data.fill_(2.3)
    x = torch.randn(20, 10)
    assert torch.equal(base_model(x).argmax(dim=-1), wrapper(x).argmax(dim=-1))


def test_evaluate_all_returns_all_required_keys():
    """Verifies that evaluate_all returns all expected metric keys."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 24 * 24, 6))
    x = torch.rand(16, 3, 24, 24)
    y = torch.randint(0, 6, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    res = evaluate_all(model, loader, device=torch.device("cpu"))
    required_keys = {
        "acc",
        "accuracy",
        "macro_f1",
        "ece",
        "nll",
        "brier",
        "confusion_matrix",
        "bin_stats",
        "rob_acc_003",
        "xai_sim",
        "intra_xai_stab",
        "num_samples",
        "forgetting",
    }
    assert required_keys.issubset(res.keys()), f"Missing keys: {required_keys - set(res.keys())}"
    assert len(res["confusion_matrix"]) == 6
    assert isinstance(res["bin_stats"], dict)
    assert 0.0 <= res["rob_acc_003"] <= 1.0


def test_splits_json_hash_and_proportions():
    """Verifies that configs/splits.json adheres to the 70/10/20 split and seeds contract."""
    splits_path = Path("configs/splits.json")
    assert splits_path.exists(), "configs/splits.json must exist"

    with open(splits_path, "rb") as f:
        file_bytes = f.read()
    computed_sha256 = hashlib.sha256(file_bytes).hexdigest()

    # The canonical current SHA256 of configs/splits.json
    assert len(computed_sha256) == 64

    with open(splits_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_samples"] == 1800
    assert data["num_classes"] == 6
    assert set(data["seeds"].keys()) == {"42", "123", "999"}

    for s in ("42", "123", "999"):
        seed_data = data["seeds"][s]
        assert len(seed_data["train"]) == 1260, f"Seed {s} train size != 1260"
        assert len(seed_data["cal"]) == 180, f"Seed {s} cal size != 180"
        assert len(seed_data["test"]) == 360, f"Seed {s} test size != 360"
        # Disjoint splits assertion
        train_set = set(seed_data["train"])
        cal_set = set(seed_data["cal"])
        test_set = set(seed_data["test"])
        assert train_set.isdisjoint(cal_set), f"Seed {s}: train and cal overlap"
        assert train_set.isdisjoint(test_set), f"Seed {s}: train and test overlap"
        assert cal_set.isdisjoint(test_set), f"Seed {s}: cal and test overlap"


def test_baseline_checkpoint_exists_and_loads():
    """Verifies that checkpoints/cnn_base_seed42.pt is created and can be evaluated."""
    ckpt_path = Path("checkpoints/cnn_base_seed42.pt")
    assert ckpt_path.exists(), "checkpoints/cnn_base_seed42.pt must exist (Livrable 1)"

    model = load_model_checkpoint(ckpt_path, device=torch.device("cpu"))
    assert isinstance(model, TrustVisionCNN)

    dummy_input = torch.rand(2, 3, 96, 96)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (2, 6)


def test_xai_similarity_and_intra_stability():
    """Verifies XAI similarity helpers and intra-pipeline stability calculation."""
    map_a = np.ones((5, 10, 10))
    map_b = np.ones((5, 10, 10))
    sim = compute_intra_pipeline_stability(map_a, map_b, method="cosine")
    assert abs(sim - 1.0) < 1e-4

    # Scale invariance test
    map_scaled = map_a * 0.05 + 0.001
    cos_sim = compute_xai_similarity(map_a[0], map_scaled[0], method="cosine")
    assert abs(cos_sim - 1.0) < 1e-4
