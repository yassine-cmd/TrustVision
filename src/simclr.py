"""
TrustVision - Member C: SimCLR contrastive pre-training (S1)
=============================================================
Domain-specific self-supervised pre-training for the shared TrustVisionCNN
backbone, followed by frozen-feature linear probing under label budgets.

Key contracts
-------------
* Inputs are RGB tensors in [0, 1] at 96x96 (see src.data).
* Uses the shared backbone through ``get_features(x)``.
* The projection head is discarded after pre-training; only the backbone is
  exported to ``checkpoints/simclr_backbone_6cls.pt`` (S1) or
  ``checkpoints/simclr_backbone_3cls.pt`` (S4 initial encoder).
* No labels are used during contrastive pre-training.

References
----------
Chen et al., "A Simple Framework for Contrastive Learning of Visual
Representations" (SimCLR), ICML 2020.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data import get_dataloaders
from src.eval import evaluate_all
from src.member_c_utils import calibrated_summary, fit_temperature_and_evaluate
from src.model import SimCLRProjectionHead, TrustVisionCNN


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def build_simclr_augmentation(img_size: int = 96, strength: float = 1.0) -> transforms.Compose:
    """
    Grayscale-friendly augmentation pipeline for steel-surface images.

    NEU-CLS images are single-channel but converted to 3-channel RGB. Heavy
    hue/saturation changes are therefore avoided; geometric transforms and
    mild photometric jitter are used instead.
    """
    crop_scale = (max(0.20, 0.50 - 0.10 * strength), 1.0)
    jitter = 0.35 * strength
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size,
                scale=crop_scale,
                ratio=(0.75, 1.3333),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=15 * strength, fill=0.0)],
                p=0.5,
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=0.15 * strength)],
                p=0.6,
            ),
            transforms.RandomGrayscale(p=0.15),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                p=0.3,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# NT-Xent loss
# ---------------------------------------------------------------------------

def nt_xent_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    Normalized Temperature-scaled Cross Entropy (NT-Xent) for two views.

    Args:
        z_i, z_j: (N, D) projection vectors for the two augmented views.
        temperature: Softmax temperature (tau).

    Returns:
        Scalar contrastive loss.
    """
    if z_i.shape != z_j.shape:
        raise ValueError(f"View shape mismatch: {z_i.shape} vs {z_j.shape}")
    if temperature <= 0:
        raise ValueError("temperature must be strictly positive")

    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # (2N, D)

    similarity = torch.matmul(z, z.t()) / temperature
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    similarity = similarity.masked_fill(mask, float("-inf"))

    targets = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ]
    )
    return F.cross_entropy(similarity, targets)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class SimCLRModel(nn.Module):
    """Shared CNN backbone + MLP projection head used only during pre-training."""

    def __init__(
        self,
        backbone: Optional[TrustVisionCNN] = None,
        feature_dim: int = 256,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 6,
    ):
        super().__init__()
        self.backbone = backbone if backbone is not None else TrustVisionCNN(
            num_classes=num_classes, feature_dim=feature_dim, normalize_input=True
        )
        self.projector = SimCLRProjectionHead(
            in_dim=feature_dim, hidden_dim=hidden_dim, out_dim=projection_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.get_features(x)
        return self.projector(features)


class SimCLRLinearProbe(nn.Module):
    """
    Frozen SimCLR backbone + linear classifier.

    Exposes the same logits interface as TrustVisionCNN so it can be passed to
    the shared ``evaluate_all`` function (accuracy, ECE, FGSM robustness, ...).
    """

    def __init__(self, backbone: TrustVisionCNN, num_classes: int = 6):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.fc = nn.Linear(backbone.feature_dim, num_classes)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        # Keep BatchNorm statistics frozen while probing.
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: no torch.no_grad() here: the shared evaluator needs gradients
        # w.r.t. the input to build FGSM adversarial examples. The backbone
        # parameters are frozen, but the graph to the input must stay alive.
        features = self.backbone.get_features(x)
        return self.fc(features)


# ---------------------------------------------------------------------------
# Pre-training
# ---------------------------------------------------------------------------

def simclr_pretrain(
    train_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.5,
    device: Optional[torch.device] = None,
    save_path: Union[str, Path] = "checkpoints/simclr_backbone_6cls.pt",
    img_size: int = 96,
    projection_dim: int = 128,
    max_batches_per_epoch: Optional[int] = None,
    num_classes: int = 6,
    verbose: bool = True,
) -> Tuple[SimCLRModel, Dict[str, List[float]]]:
    """
    Trains the shared backbone with SimCLR on unlabeled images.

    Returns the trained SimCLR wrapper and a loss history. The backbone state
    dict is written to ``save_path`` for downstream probing / continual learning.
    ``num_classes`` only sets the (discarded) classifier head size so the saved
    backbone matches the downstream 3-class or 6-class model contract.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimCLRModel(
        backbone=TrustVisionCNN(num_classes=num_classes),
        projection_dim=projection_dim,
        num_classes=num_classes,
    )
    model.to(device)
    model.train()

    augmentation = build_simclr_augmentation(img_size=img_size, strength=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    history: Dict[str, List[float]] = {"epoch": [], "loss": []}

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break

            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device)

            view_i = augmentation(images)
            view_j = augmentation(images)

            z_i = model(view_i)
            z_j = model(view_j)
            loss = nt_xent_loss(z_i, z_j, temperature=temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        epoch_loss = running_loss / max(1, n_batches)
        history["epoch"].append(epoch)
        history["loss"].append(round(epoch_loss, 4))
        if verbose:
            print(f"  [SimCLR] epoch {epoch:02d}/{epochs:02d} - NT-Xent: {epoch_loss:.4f}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.backbone.state_dict(), save_path)
    if verbose:
        print(f"  [SimCLR] backbone saved to {save_path}")
    return model, history


# ---------------------------------------------------------------------------
# Feature extraction and linear probing
# ---------------------------------------------------------------------------

def extract_features(
    backbone: TrustVisionCNN,
    loader: DataLoader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts frozen 256-d backbone features and labels from a DataLoader."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone.eval()
    backbone.to(device)

    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device)
            feats = backbone.get_features(images)
            features.append(feats.detach().cpu().numpy())
            labels.append(targets.numpy())

    if not features:
        return np.empty((0, backbone.feature_dim), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def build_probe_module(
    backbone: TrustVisionCNN,
    features_train: np.ndarray,
    labels_train: np.ndarray,
    num_classes: int = 6,
    seed: int = 42,
    C: float = 1.0,
) -> Tuple[nn.Module, LogisticRegression, StandardScaler]:
    """Fits a sklearn probe and wraps it with the frozen backbone for evaluate_all."""
    scaler = StandardScaler()
    x_train = scaler.fit_transform(features_train)
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=seed)
    clf.fit(x_train, labels_train)

    class _ProbeModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = copy.deepcopy(backbone)
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.register_buffer("mean", torch.tensor(scaler.mean_, dtype=torch.float32))
            self.register_buffer("scale", torch.tensor(scaler.scale_, dtype=torch.float32))
            self.register_buffer("weight", torch.tensor(clf.coef_, dtype=torch.float32))
            self.register_buffer("bias", torch.tensor(clf.intercept_, dtype=torch.float32))
            self.num_classes = num_classes
            self.fc = nn.Linear(backbone.feature_dim, num_classes)  # compatibility

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Keep the input graph alive for FGSM (backbone weights stay frozen).
            feats = self.backbone.get_features(x)
            x_norm = (feats - self.mean.to(feats.device)) / self.scale.to(feats.device)
            return x_norm @ self.weight.t() + self.bias

    return _ProbeModule(), clf, scaler


def fold_probe_into_backbone(
    backbone: TrustVisionCNN,
    clf: LogisticRegression,
    scaler: StandardScaler,
) -> TrustVisionCNN:
    """
    Folds a standardized sklearn linear probe into a plain ``TrustVisionCNN``.

    The probe computes ``((f - mean) / scale) @ W^T + b``; this is rewritten as
    ``f @ (W / scale)^T + (b - (W / scale) @ mean)`` and stored in ``model.fc``,
    so the exported checkpoint can be loaded by the shared evaluator/Gradio.
    """
    model = copy.deepcopy(backbone)
    weight = torch.tensor(clf.coef_, dtype=torch.float32)
    bias = torch.tensor(clf.intercept_, dtype=torch.float32)
    mean = torch.tensor(scaler.mean_, dtype=torch.float32)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32)

    if weight.shape[0] != model.fc.out_features:
        raise ValueError(
            f"Probe has {weight.shape[0]} classes but backbone has "
            f"{model.fc.out_features} outputs."
        )

    folded_weight = weight / scale.unsqueeze(0)
    folded_bias = bias - folded_weight @ mean

    with torch.no_grad():
        model.fc.weight.copy_(folded_weight)
        model.fc.bias.copy_(folded_bias)
    model.eval()
    return model


def evaluate_simclr_label_fractions(
    backbone_path: Union[str, Path],
    fractions: Sequence[float] = (0.01, 0.05, 0.10, 1.0),
    seed: int = 42,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    num_classes: int = 6,
    root_dir: str = "data/NEU-CLS",
    calibrate: bool = True,
    probe_checkpoint_path: Optional[Union[str, Path]] = None,
    probe_checkpoint_pattern: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates a frozen SimCLR backbone with linear probes at several label
    budgets (1 %, 5 %, 10 %, 100 %). Test metrics come from the shared
    ``evaluate_all`` contract so they are comparable to S2/S3/S4.

    Every probe is additionally calibrated with temperature scaling on the
    shared calibration split. The 100 % probe can be exported as a
    ``TrustVisionCNN`` checkpoint (``probe_checkpoint_path``) for the
    "best model" S1 contract. Set ``probe_checkpoint_pattern`` (e.g.
    ``"checkpoints/s1_simclr_probe_{fraction}pct.pt"``) to export *every*
    budget so the demo can run live inference at 1/5/10/100 %.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = TrustVisionCNN(num_classes=num_classes)
    state = torch.load(backbone_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    backbone.load_state_dict(state, strict=False)
    backbone.to(device).eval()

    results: Dict[str, Dict[str, float]] = {}
    for fraction in fractions:
        train_loader, cal_loader, test_loader, _ = get_dataloaders(
            root_dir=root_dir,
            batch_size=batch_size,
            seed=seed,
            img_size=96,
            label_fraction=fraction,
            num_classes=num_classes,
            auto_download=True,
        )
        features_train, labels_train = extract_features(backbone, train_loader, device=device)
        probe_module, clf, scaler = build_probe_module(
            backbone, features_train, labels_train, num_classes=num_classes, seed=seed
        )
        probe_module.to(device)
        metrics = evaluate_all(probe_module, test_loader, device=device)
        metrics["label_fraction"] = fraction
        metrics["num_train_labels"] = int(len(labels_train))

        is_full_budget = abs(fraction - 1.0) < 1e-6
        if calibrate:
            calibrated = fit_temperature_and_evaluate(
                probe_module, cal_loader, test_loader, device=device
            )
            metrics["calibrated"] = calibrated_summary(calibrated)

        if probe_checkpoint_pattern is not None:
            export_path = Path(
                probe_checkpoint_pattern.format(fraction=int(round(fraction * 100)))
            )
        elif probe_checkpoint_path is not None and is_full_budget:
            export_path = Path(probe_checkpoint_path)
        else:
            export_path = None

        if export_path is not None:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            folded = fold_probe_into_backbone(backbone, clf, scaler)
            torch.save(folded.state_dict(), export_path)
            metrics["probe_checkpoint"] = str(export_path.as_posix())

        results[f"{int(round(fraction * 100))}pct"] = metrics
        ece_display = metrics["ece"]
        if "calibrated" in metrics:
            ece_display = metrics["calibrated"]["ece"]
        print(
            f"  [SimCLR probe] {fraction * 100:5.1f}% labels "
            f"(n={len(labels_train)}): acc={metrics['acc']:.4f}, "
            f"ece={ece_display:.4f}, rob={metrics['rob_acc_003']:.4f}"
        )

    return results


def compare_with_clip_probe(
    simclr_results: Dict[str, Dict[str, float]],
    s0_path: Union[str, Path] = "results/s0_clip.json",
) -> Dict[str, Optional[float]]:
    """Returns the CLIP linear-probe reference metrics when available."""
    s0_path = Path(s0_path)
    if not s0_path.exists():
        return {"acc": None, "ece": None, "rob_acc_003": None}
    import json

    with open(s0_path, "r", encoding="utf-8") as f:
        s0 = json.load(f)
    probe = s0.get("theme4_linear_probe", {})
    calibrated = probe.get("calibrated", {})
    cross = s0.get("cross_effects_input", {})
    return {
        "acc": cross.get("clip_linear_probe_acc", calibrated.get("accuracy")),
        "ece": cross.get("clip_linear_probe_ece_calibrated", calibrated.get("ece")),
        "rob_acc_003": cross.get("clip_linear_probe_rob_acc_003"),
    }
