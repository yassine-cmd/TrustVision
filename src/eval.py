"""
TrustVision - Evaluation & Metrics Module (src/eval.py)
Shared Evaluation Contract:
- Unified evaluation entrypoint: evaluate_all(model, loader, device, fgsm_eps=0.03)
- Produces standardized metric dict: {acc, ece, rob_acc_003, xai_sim}
- FGSM adversarial attack operating in [0, 1] pixel space
- Expected Calibration Error (ECE) and Temperature Scaling
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> Tuple[float, Dict[str, Any]]:
    """
    Computes Expected Calibration Error (ECE) and reliability diagram statistics.

    Args:
        probs: (N, C) predicted probabilities after softmax.
        labels: (N,) ground-truth class labels.
        n_bins: Number of equal-width confidence bins (default 10).

    Returns:
        (ece_score, bin_details_dict)
    """
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_stats = {
        "bin_accuracies": [],
        "bin_confidences": [],
        "bin_counts": [],
        "bin_edges": bin_boundaries.tolist(),
    }

    n_samples = float(labels.size(0))

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            bin_stats["bin_accuracies"].append(accuracy_in_bin)
            bin_stats["bin_confidences"].append(avg_confidence_in_bin)
            bin_stats["bin_counts"].append(int(in_bin.sum().item()))
        else:
            bin_stats["bin_accuracies"].append(0.0)
            bin_stats["bin_confidences"].append(0.0)
            bin_stats["bin_counts"].append(0)

    return float(ece), bin_stats


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 0.03,
) -> torch.Tensor:
    """
    Generates Fast Gradient Sign Method (FGSM) adversarial examples.
    Assumes images are in [0, 1]. Result is strictly clamped to [0, 1].

    Args:
        model: Target neural network.
        images: Batch of clean images in [0, 1].
        labels: True target labels.
        eps: Attack budget (standard contract specifies eps=0.03).

    Returns:
        Adversarially perturbed images clamped to [0, 1].
    """
    if eps == 0.0:
        return images.clone()

    images_adv = images.clone().detach().requires_grad_(True)
    outputs = model(images_adv)
    loss = F.cross_entropy(outputs, labels)

    model.zero_grad()
    loss.backward()

    # Sign of gradient
    data_grad = images_adv.grad.data
    perturbed_images = images_adv + eps * data_grad.sign()

    # Keep valid pixel range [0, 1]
    perturbed_images = torch.clamp(perturbed_images, 0.0, 1.0)
    return perturbed_images.detach()


class ModelWithTemperature(nn.Module):
    """
    Wraps a model to calibrate its output logits via Temperature Scaling:
    softmax(logits / T). Parameter T is optimized on a calibration set.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits / self.temperature

    def set_temperature(
        self,
        cal_loader: DataLoader,
        device: torch.device,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        """Optimizes temperature parameter T via LBFGS on calibration set."""
        self.eval()
        nll_criterion = nn.CrossEntropyLoss()

        logits_list = []
        labels_list = []

        with torch.no_grad():
            for images, targets in cal_loader:
                images = images.to(device)
                logits = self.model(images)
                logits_list.append(logits.cpu())
                labels_list.append(targets)

        logits_all = torch.cat(logits_list).to(device)
        labels_all = torch.cat(labels_list).to(device)

        # LBFGS optimizer for temperature parameter
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(logits_all / self.temperature, labels_all)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return float(self.temperature.item())


def evaluate_all(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    fgsm_eps: float = 0.03,
    xai_reference_maps: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Unified evaluation function required by the TrustVision repo contract.
    Computes standard accuracy, ECE, robust accuracy at eps=0.03, and XAI similarity.

    Args:
        model: Evaluated PyTorch model.
        dataloader: Test or evaluation DataLoader.
        device: Target torch device (CPU or CUDA).
        fgsm_eps: FGSM perturbation magnitude (default 0.03).
        xai_reference_maps: Optional baseline attention/saliency maps for similarity check.

    Returns:
        Dictionary containing:
          - 'acc': Clean top-1 accuracy (0.0 to 1.0)
          - 'ece': Expected Calibration Error (0.0 to 1.0)
          - 'rob_acc_003': Adversarial accuracy under FGSM eps=0.03 (0.0 to 1.0)
          - 'xai_sim': Cosine similarity to reference XAI maps (or None if unavailable)
          - 'num_samples': Total evaluated samples
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    total_samples = 0
    correct_clean = 0
    correct_rob = 0

    all_probs = []
    all_targets = []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)
        total_samples += batch_size

        # 1. Clean forward pass
        with torch.no_grad():
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            correct_clean += (preds == targets).sum().item()

            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

        # 2. Adversarial forward pass (FGSM eps=0.03)
        images_adv = fgsm_attack(model, images, targets, eps=fgsm_eps)
        with torch.no_grad():
            adv_outputs = model(images_adv)
            adv_preds = adv_outputs.argmax(dim=1)
            correct_rob += (adv_preds == targets).sum().item()

    # Aggregate probabilities & labels
    all_probs_cat = torch.cat(all_probs, dim=0)
    all_targets_cat = torch.cat(all_targets, dim=0)

    clean_acc = correct_clean / total_samples if total_samples > 0 else 0.0
    rob_acc = correct_rob / total_samples if total_samples > 0 else 0.0
    ece_score, _ = compute_ece(all_probs_cat, all_targets_cat, n_bins=10)

    # XAI similarity placeholder (filled when saliency maps are evaluated)
    xai_similarity = None

    return {
        "acc": round(clean_acc, 4),
        "ece": round(ece_score, 4),
        "rob_acc_003": round(rob_acc, 4),
        "xai_sim": xai_similarity,
        "num_samples": total_samples,
    }
