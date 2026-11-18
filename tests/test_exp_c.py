"""
TrustVision - Unit Tests for Member C (tests/test_exp_c.py)
==========================================================
Validates the S1 (SimCLR) and S4 (EWC) deliverables without downloading the
dataset:

1. NT-Xent loss sanity (finite, lower for matching views).
2. SimCLR augmentation keeps batched tensor shape/values finite.
3. SimCLR pre-training smoke test and checkpoint contract.
4. Frozen-probe gradient path required by FGSM.
5. Probe folding into a plain TrustVisionCNN (main-model export).
6. Classifier head expansion 3 -> 6 and EWC state padding.
7. Fisher diagonal non-negativity and EWC penalty value.
8. Continual evaluation with an empty new-class set (Task-1 model).
9. Frozen splits/label-budget contract from configs/splits.json.
10. Class-balanced replay buffer (split indices are grouped by class).
11. Temperature-scaling calibration helper and degenerate-temperature guard.
12. Cross-effects table fallback when calibration is rejected.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import load_splits_config
from src.ewc import (
    collect_gradcam_maps,
    compute_fisher_diagonal,
    evaluate_continual_model,
    ewc_penalty,
    expand_classifier,
    normalize_fisher_per_tensor,
    pad_ewc_state_for_expanded_head,
    per_class_accuracy,
    stratified_replay_indices,
)
from src.member_c_utils import (
    fit_temperature_and_evaluate,
    select_calibration_temperature,
)
from src.model import TrustVisionCNN
from src.run_member_c import update_cross_effects_table
from src.simclr import (
    build_probe_module,
    build_simclr_augmentation,
    fold_probe_into_backbone,
    nt_xent_loss,
    simclr_pretrain,
)


CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# SimCLR
# ---------------------------------------------------------------------------

def test_nt_xent_loss_is_finite_and_backpropagates():
    torch.manual_seed(0)
    z_i = torch.randn(6, 16, requires_grad=True)
    z_j = torch.randn(6, 16, requires_grad=True)

    loss = nt_xent_loss(z_i, z_j, temperature=0.5)
    assert torch.isfinite(loss)
    loss.backward()
    assert z_i.grad is not None and torch.isfinite(z_i.grad).all()
    assert z_j.grad is not None and torch.isfinite(z_j.grad).all()


def test_nt_xent_prefers_matching_views():
    torch.manual_seed(0)
    z_i = torch.randn(8, 16)
    z_j_close = z_i + 0.001 * torch.randn_like(z_i)
    z_j_far = torch.randn(8, 16)

    loss_close = nt_xent_loss(z_i, z_j_close, temperature=0.5).item()
    loss_far = nt_xent_loss(z_i, z_j_far, temperature=0.5).item()
    assert loss_close < loss_far


def test_simclr_augmentation_preserves_batch_shape():
    torch.manual_seed(0)
    images = torch.rand(2, 3, 96, 96)
    augmented = build_simclr_augmentation(img_size=96)(images)
    assert augmented.shape == images.shape
    assert torch.isfinite(augmented).all()


def test_simclr_pretrain_smoke(tmp_path):
    torch.manual_seed(0)
    images = torch.rand(4, 3, 96, 96)
    labels = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    checkpoint = tmp_path / "simclr_backbone.pt"
    model, history = simclr_pretrain(
        loader,
        epochs=1,
        device=CPU,
        save_path=checkpoint,
        max_batches_per_epoch=1,
        num_classes=6,
        verbose=False,
    )

    assert checkpoint.exists()
    assert len(history["loss"]) == 1
    assert np.isfinite(history["loss"][0])
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["fc.weight"].shape == (6, 256)


def test_probe_forward_keeps_input_gradient_for_fgsm():
    torch.manual_seed(0)
    backbone = TrustVisionCNN(num_classes=6)
    features = torch.randn(30, 256).numpy()
    labels = np.arange(30) % 6

    probe, _, _ = build_probe_module(
        backbone, features, labels, num_classes=6, seed=0
    )

    images = torch.rand(2, 3, 96, 96, requires_grad=True)
    logits = probe(images)
    assert logits.shape == (2, 6)
    logits.sum().backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()


def test_probe_folding_matches_probe_logits():
    torch.manual_seed(0)
    backbone = TrustVisionCNN(num_classes=6)
    features = torch.randn(30, 256).numpy()
    labels = np.arange(30) % 6

    probe, clf, scaler = build_probe_module(
        backbone, features, labels, num_classes=6, seed=0
    )
    probe.eval()
    folded = fold_probe_into_backbone(backbone, clf, scaler).eval()

    with torch.no_grad():
        images = torch.rand(3, 3, 96, 96)
        probe_logits = probe(images)
        folded_logits = folded(images)

    assert torch.allclose(probe_logits, folded_logits, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# EWC
# ---------------------------------------------------------------------------

def test_expand_classifier_preserves_old_rows():
    torch.manual_seed(0)
    model = TrustVisionCNN(num_classes=3)
    old_weight = model.fc.weight.detach().clone()
    old_bias = model.fc.bias.detach().clone()

    expand_classifier(model, new_num_classes=6)

    assert model.fc.out_features == 6
    assert model.num_classes == 6
    assert torch.equal(model.fc.weight[:3], old_weight)
    assert torch.equal(model.fc.bias[:3], old_bias)


def test_pad_ewc_state_for_expanded_head():
    torch.manual_seed(0)
    model = TrustVisionCNN(num_classes=3)
    fisher = {name: torch.ones_like(param) for name, param in model.named_parameters()}
    old_params = {name: param.detach().clone() for name, param in model.named_parameters()}
    old_fc_weight = fisher["fc.weight"].clone()

    expand_classifier(model, new_num_classes=6)
    padded_fisher, padded_old = pad_ewc_state_for_expanded_head(
        model, fisher, old_params, old_num_classes=3
    )

    assert padded_fisher["fc.weight"].shape == (6, 256)
    assert padded_fisher["fc.bias"].shape == (6,)
    assert torch.equal(padded_fisher["fc.weight"][:3], old_fc_weight)
    assert torch.count_nonzero(padded_fisher["fc.weight"][3:]) == 0
    assert torch.equal(padded_old["fc.weight"][3:], model.fc.weight.detach()[3:])


def test_expanded_head_accepts_six_class_checkpoint():
    """Regression test: the 6-class EWC checkpoint must load after 3 -> 6 expansion."""
    torch.manual_seed(0)
    task1_model = TrustVisionCNN(num_classes=3)
    task2_model = TrustVisionCNN(num_classes=6)

    expand_classifier(task1_model, new_num_classes=6)
    task1_model.load_state_dict(task2_model.state_dict())

    assert task1_model.fc.out_features == 6


def test_stratified_replay_indices_cover_every_old_class():
    """Regression test: the split is grouped by class, so a plain range(120) buffer is single-class."""
    dataset = [(torch.zeros(1), idx // 20) for idx in range(60)]  # 20 images per class, grouped
    indices = stratified_replay_indices(dataset, replay_size=12, classes=(0, 1, 2), seed=42)
    counts = Counter(int(dataset[idx][1]) for idx in indices)

    assert len(indices) == 12
    assert set(counts) == {0, 1, 2}
    assert all(count == 4 for count in counts.values())


def test_normalize_fisher_per_tensor_sets_unit_mean_and_keeps_ordering():
    fisher = {
        "shallow": torch.tensor([1.0, 3.0]),
        "deep": torch.tensor([0.0, 4e-6]),
    }
    normalized = normalize_fisher_per_tensor(fisher)

    for name in fisher:
        assert torch.isclose(normalized[name].mean(), torch.tensor(1.0))
    assert torch.all(normalized["deep"] >= 0)
    assert float(normalized["deep"].max()) == pytest.approx(2.0)


def test_ewc_penalty_zero_at_anchor_and_positive_after_update():
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    fisher = {
        "weight": torch.ones_like(model.weight),
        "bias": torch.ones_like(model.bias),
    }
    old_params = {
        "weight": model.weight.detach().clone(),
        "bias": model.bias.detach().clone(),
    }

    assert ewc_penalty(model, fisher, old_params, ewc_lambda=10.0).item() == 0.0

    with torch.no_grad():
        model.weight.add_(0.1)
    penalty = ewc_penalty(model, fisher, old_params, ewc_lambda=10.0).item()
    expected = 0.5 * 10.0 * (0.1 ** 2) * model.weight.numel()
    assert penalty == pytest.approx(expected, rel=1e-5)


def test_compute_fisher_diagonal_is_non_negative():
    torch.manual_seed(0)
    model = TrustVisionCNN(num_classes=3)
    images = torch.rand(4, 3, 96, 96)
    labels = torch.tensor([0, 1, 2, 0])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    fisher = compute_fisher_diagonal(model, loader, device=CPU, max_batches=1)

    assert fisher
    assert all(torch.all(value >= 0) for value in fisher.values())
    assert any(torch.any(value > 0) for value in fisher.values())


def test_per_class_accuracy():
    cm = [[5, 0, 0], [0, 4, 1], [0, 2, 3]]
    recalls = per_class_accuracy(cm)
    assert recalls == pytest.approx([1.0, 0.8, 0.6])


def test_evaluate_continual_model_handles_empty_new_classes():
    torch.manual_seed(0)
    model = TrustVisionCNN(num_classes=3)
    images = torch.rand(4, 3, 96, 96)
    labels = torch.tensor([0, 1, 2, 0])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    metrics = evaluate_continual_model(
        model,
        loader,
        old_classes=(0, 1, 2),
        new_classes=(),
        device=CPU,
    )

    assert metrics["new_acc"] is None
    assert metrics["new_class_recalls"] is None
    assert 0.0 <= metrics["old_acc"] <= 1.0


def test_temperature_scaling_helper_returns_calibrated_metrics():
    torch.manual_seed(0)
    model = TrustVisionCNN(num_classes=6)
    images = torch.rand(4, 3, 96, 96)
    labels = torch.tensor([0, 1, 2, 3])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    metrics = fit_temperature_and_evaluate(
        model, loader, loader, device=CPU, max_iter=5
    )

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["temperature"] > 0.0
    assert metrics["temperature_fitted"] > 0.0
    assert isinstance(metrics["calibration_accepted"], bool)
    assert np.isfinite(metrics["calibration_nll"])
    assert np.isfinite(metrics["calibration_nll_fitted"])
    assert np.isfinite(metrics["ece"])
    assert np.isfinite(metrics["nll"])
    assert np.isfinite(metrics["brier"])


def test_select_calibration_temperature_rejects_degenerate_floor():
    """Regression test for S4 EWC+replay: LBFGS returned T=0.001 and NLL exploded."""
    logits = torch.tensor([[6.0, 0.0], [6.0, 0.0]])
    labels = torch.tensor([0, 1])

    temperature, accepted, nll_t1, nll_fitted = select_calibration_temperature(
        logits, labels, fitted_temperature=0.001
    )

    assert temperature == 1.0
    assert accepted is False
    assert nll_fitted > nll_t1


def test_select_calibration_temperature_rejects_out_of_range_even_if_nll_improves():
    """A perfect calibration split would otherwise push T to the LBFGS floor."""
    logits = torch.tensor([[6.0, 0.0], [6.0, 0.0]])
    labels = torch.tensor([0, 0])

    temperature, accepted, nll_t1, nll_fitted = select_calibration_temperature(
        logits, labels, fitted_temperature=0.001
    )

    assert temperature == 1.0
    assert accepted is False
    assert nll_fitted < nll_t1


def test_select_calibration_temperature_accepts_helpful_scaling():
    logits = torch.tensor([[3.0, 0.0], [3.0, 0.0]])
    labels = torch.tensor([0, 1])

    temperature, accepted, nll_t1, nll_fitted = select_calibration_temperature(
        logits, labels, fitted_temperature=2.0
    )

    assert temperature == 2.0
    assert accepted is True
    assert nll_fitted < nll_t1


def test_collect_gradcam_maps_shape_and_range():
    torch.manual_seed(0)
    model = TrustVisionCNN(num_classes=3)
    images = torch.rand(4, 3, 96, 96)
    labels = torch.tensor([0, 1, 2, 0])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    maps = collect_gradcam_maps(model, loader, device=CPU)

    assert maps.shape == (4, 96, 96)
    assert np.isfinite(maps).all()
    assert float(maps.min()) >= 0.0
    assert float(maps.max()) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Shared splits contract
# ---------------------------------------------------------------------------

def test_member_c_splits_contract():
    config = load_splits_config()
    seed_splits = config["seeds"]["42"]

    assert len(seed_splits["train"]) == 1260
    assert len(seed_splits["cal"]) == 180
    assert len(seed_splits["test"]) == 360
    assert {key: len(value) for key, value in seed_splits["subsets_train"].items()} == {
        "1pct": 12,
        "5pct": 60,
        "10pct": 126,
        "100pct": 1260,
    }
    assert config["continual_learning"]["task1_classes"] == [0, 1, 2]
    assert config["continual_learning"]["task2_classes"] == [3, 4, 5]


def test_cross_effects_table_uses_raw_metrics_when_calibration_is_rejected(tmp_path):
    raw = {
        "acc": 0.8917,
        "macro_f1": 0.8923,
        "ece": 0.0455,
        "nll": 0.3401,
        "brier": 0.1624,
        "rob_acc_003": 0.1194,
        "xai_sim": None,
        "intra_xai_stab": 0.7707,
        "forgetting": 0.0667,
        "old_acc": 0.9333,
        "new_acc": 0.85,
    }
    s1_metrics = {
        "acc": 0.9722,
        "macro_f1": 0.9722,
        "ece": 0.0087,
        "nll": 0.0785,
        "brier": 0.043,
        "rob_acc_003": 0.15,
        "calibrated": {
            "acc": 0.9722,
            "macro_f1": 0.9722,
            "ece": 0.0087,
            "nll": 0.0785,
            "brier": 0.043,
            "temperature": 0.8689,
            "calibration_accepted": True,
        },
    }
    s4_metrics = {
        **raw,
        "calibrated": {
            "acc": 0.8917,
            "macro_f1": 0.8923,
            "ece": 0.1081,
            "nll": 2.9238,
            "brier": 0.2159,
            "temperature": 1.0,
            "temperature_fitted": 0.001,
            "calibration_accepted": False,
            "calibration_nll": 0.23,
            "calibration_nll_fitted": 0.10,
        },
    }
    s1_output = {
        "simclr": {"label_fractions": {"100pct": s1_metrics}},
        "main_model": {"checkpoint": "checkpoints/s1_main_model.pt"},
    }
    s4_output = {
        "best_ewc": {
            "strategy": "ewc_replay",
            "ewc_lambda": 1.0,
            "metrics": s4_metrics,
        }
    }

    table = update_cross_effects_table(
        s1_output, s4_output, table_path=str(tmp_path / "cross_effects_table.json")
    )
    row = next(r for r in table["rows"] if r["stage"] == "S4_EWC_Continual")

    assert row["ece"] == raw["ece"]
    assert row["nll"] == raw["nll"]
    assert "calibration_note" in row
    assert table["member_c"]["s4_best_ewc"]["calibration_accepted"] is False
    assert table["member_c"]["s4_best_ewc"]["ece_fitted_temperature"] == 0.1081
