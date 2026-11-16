"""
TrustVision - Evaluation & Metrics Module (src/eval.py)
Shared Evaluation Contract:
- Unified evaluation entrypoint: evaluate_all(model, loader, device, fgsm_eps=0.03)
- Produces standardized metric dict: accuracy, macro_f1, confusion_matrix, ece, nll,
  brier, rob_acc_003, xai_sim, forgetting (plus the legacy `acc` alias).
- FGSM adversarial attack operating in [0, 1] pixel space
- Expected Calibration Error (ECE) and Temperature Scaling
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix


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
        "accuracies": [],
        "confidences": [],
        "counts": [],
    }

    n_samples = float(labels.size(0))

    for idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        if idx == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            bin_stats["accuracies"].append(accuracy_in_bin)
            bin_stats["confidences"].append(avg_confidence_in_bin)
            bin_stats["counts"].append(int(in_bin.sum().item()))
        else:
            bin_stats["accuracies"].append(0.0)
            bin_stats["confidences"].append(0.0)
            bin_stats["counts"].append(0)

    return float(ece), bin_stats


def compute_classification_metrics(
    probs: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes unified classification and calibration metrics:
    Accuracy, Macro-F1, ECE, Negative Log-Likelihood (NLL), Brier score, and Confusion Matrix.
    """
    if isinstance(probs, np.ndarray):
        probs = torch.from_numpy(probs).float()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).long()

    probs = probs.detach().cpu()
    labels = labels.detach().cpu()

    confidences, preds = torch.max(probs, dim=1)
    acc = preds.eq(labels).float().mean().item()

    preds_np = preds.numpy()
    labels_np = labels.numpy()
    num_classes = probs.shape[1]
    macro_f1 = float(f1_score(labels_np, preds_np, average="macro", zero_division=0))
    cm = confusion_matrix(labels_np, preds_np, labels=list(range(num_classes))).tolist()

    ece, bin_stats = compute_ece(probs, labels, n_bins=n_bins)

    eps = 1e-12
    probs_clamped = torch.clamp(probs, min=eps, max=1.0)
    nll = F.nll_loss(torch.log(probs_clamped), labels).item()

    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    brier = torch.mean(torch.sum((probs - one_hot) ** 2, dim=1)).item()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "ece": round(ece, 4),
        "nll": round(nll, 4),
        "brier": round(brier, 4),
        "confusion_matrix": cm,
        "bin_stats": bin_stats,
    }


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
    model.zero_grad()
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
        temp = torch.clamp(self.temperature, min=1e-3)
        return logits / temp

    def set_temperature(
        self,
        cal_loader: DataLoader,
        device: torch.device,
        max_iter: int = 200,
        lr: float = 0.1,
    ) -> float:
        """Optimizes temperature parameter T via LBFGS on calibration set."""
        self.eval()
        self.to(device)
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
            temp = torch.clamp(self.temperature, min=1e-3)
            loss = nll_criterion(logits_all / temp, labels_all)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        with torch.no_grad():
            self.temperature.clamp_(min=1e-3)
        return float(self.temperature.item())


class TemperatureScaler(nn.Module):
    """
    Stand-alone temperature scaling calibrator operating directly on logits.
    softmax(logits / T).
    """

    def __init__(self, initial_temp: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.clamp(self.temperature, min=1e-3)
        return logits / temp

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.1,
        max_iter: int = 200,
    ) -> float:
        """Optimizes temperature parameter T via LBFGS on calibration logits."""
        self.eval()
        device = logits.device
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            temp = torch.clamp(self.temperature, min=1e-3)
            loss = nll_criterion(logits / temp, labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        with torch.no_grad():
            self.temperature.clamp_(min=1e-3)
        return float(self.temperature.item())


def _extract_map_batch(
    maps: Union[np.ndarray, torch.Tensor],
    num_samples: int,
) -> np.ndarray:
    """Normalizes XAI map batches to (N, H, W); averages a layer axis if present."""
    if isinstance(maps, torch.Tensor):
        maps = maps.detach().cpu().numpy()
    arr = np.asarray(maps)
    if arr.ndim == 4:
        arr = arr.mean(axis=1)
    if arr.ndim != 3 or arr.shape[0] != num_samples:
        raise ValueError(
            "Expected XAI maps with shape (N, H, W) or (N, L, H, W) where "
            f"N={num_samples}; got {arr.shape}."
        )
    return arr


def load_model_checkpoint(
    checkpoint_or_model: Union[str, Path, nn.Module],
    device: Optional[torch.device] = None,
    num_classes: int = 6,
) -> nn.Module:
    """
    Checkpoint-agnostic loader.
    Accepts either an already instantiated nn.Module, a path to a .pt file
    (state_dict or full serialized object), and maps it safely to target device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(checkpoint_or_model, nn.Module):
        model = checkpoint_or_model
    elif isinstance(checkpoint_or_model, (str, Path)):
        ckpt_path = Path(checkpoint_or_model)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        
        # Local import to prevent circular dependency
        from src.model import TrustVisionCNN
        loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(loaded, dict):
            state_dict = loaded.get("state_dict", loaded)
            if "fc.weight" in state_dict:
                num_classes = state_dict["fc.weight"].shape[0]
            elif "fc.bias" in state_dict:
                num_classes = state_dict["fc.bias"].shape[0]

            model = TrustVisionCNN(num_classes=num_classes)
            has_fc = "fc.weight" in state_dict or "fc.bias" in state_dict
            model.load_state_dict(state_dict, strict=has_fc)
        elif isinstance(loaded, nn.Module):
            model = loaded
        else:
            raise ValueError(f"Unrecognized checkpoint format at {ckpt_path}")
    else:
        raise TypeError(f"Expected nn.Module or path, got {type(checkpoint_or_model)}")

    model.to(device)
    model.eval()
    return model


def compute_intra_pipeline_stability(
    maps_stage_a: Union[np.ndarray, torch.Tensor],
    maps_stage_b: Union[np.ndarray, torch.Tensor],
    method: str = "cosine",
) -> float:
    """
    Computes intra-pipeline XAI stability: average similarity between explanation maps
    of the same images before vs after defense (e.g. S2 baseline vs S3 adversarial or S4 EWC).
    """
    if isinstance(maps_stage_a, torch.Tensor):
        maps_stage_a = maps_stage_a.detach().cpu().numpy()
    if isinstance(maps_stage_b, torch.Tensor):
        maps_stage_b = maps_stage_b.detach().cpu().numpy()

    a_arr = np.asarray(maps_stage_a)
    b_arr = np.asarray(maps_stage_b)

    if a_arr.ndim == 4:
        a_arr = a_arr.mean(axis=1)
    if b_arr.ndim == 4:
        b_arr = b_arr.mean(axis=1)

    if a_arr.shape[0] != b_arr.shape[0]:
        raise ValueError(f"Sample count mismatch: {a_arr.shape[0]} vs {b_arr.shape[0]}")

    n = a_arr.shape[0]
    sims = [
        compute_xai_similarity(a_arr[i], b_arr[i], method=method)
        for i in range(n)
    ]
    return float(np.mean(sims))


def format_stage_row(stage_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formats a unified dictionary row for the cross-effects comparative table (S0 to S4).
    """
    return {
        "stage": stage_name,
        "acc": metrics.get("acc", metrics.get("accuracy")),
        "macro_f1": metrics.get("macro_f1"),
        "ece": metrics.get("ece"),
        "nll": metrics.get("nll"),
        "brier": metrics.get("brier"),
        "rob_acc_003": metrics.get("rob_acc_003"),
        "xai_sim": metrics.get("xai_sim"),
        "intra_xai_stab": metrics.get("intra_xai_stab"),
        "forgetting": metrics.get("forgetting"),
    }


def evaluate_all(
    model: Union[nn.Module, str, Path],
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    fgsm_eps: float = 0.03,
    xai_reference_maps: Optional[np.ndarray] = None,
    xai_saliency_maps: Optional[np.ndarray] = None,
    previous_task_acc: Optional[float] = None,
    intra_stability_reference_maps: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Unified evaluation function required by the TrustVision repo contract.
    Computes the full classification/calibration suite, robust accuracy at eps=0.03,
    XAI agreement, and forgetting. Agnostic to checkpoint (.pt path or nn.Module).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_checkpoint(model, device=device)
    model.eval()

    # If saliency maps provided without reference, try loading clip_attn_test.npy
    if xai_saliency_maps is not None and xai_reference_maps is None:
        ref_path = Path("results/clip_attn_test.npy")
        if ref_path.exists():
            try:
                xai_reference_maps = np.load(str(ref_path))
            except Exception:
                xai_reference_maps = None

    if (xai_saliency_maps is None) != (xai_reference_maps is None) and xai_reference_maps is not None:
        # If user explicitly passed only one and couldn't auto-resolve, require both
        raise ValueError("Provide both xai_saliency_maps and xai_reference_maps, or neither.")

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

            all_probs.append(probs.detach().cpu())
            all_targets.append(targets.detach().cpu())

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

    metrics = compute_classification_metrics(all_probs_cat, all_targets_cat, n_bins=10)

    # XAI agreement (computed only when both map batches are supplied)
    xai_similarity = None
    if xai_saliency_maps is not None and xai_reference_maps is not None:
        saliency = _extract_map_batch(xai_saliency_maps, total_samples)
        reference = _extract_map_batch(xai_reference_maps, total_samples)
        xai_similarity = float(np.mean([
            compute_xai_similarity(saliency[i], reference[i], method="cosine")
            for i in range(total_samples)
        ]))

    # Intra-pipeline XAI stability (e.g. baseline vs adversarial or EWC)
    intra_stab = None
    if xai_saliency_maps is not None and intra_stability_reference_maps is not None:
        intra_stab = compute_intra_pipeline_stability(
            intra_stability_reference_maps, xai_saliency_maps, method="cosine"
        )

    metrics.update({
        "acc": round(clean_acc, 4),
        "rob_acc_003": round(rob_acc, 4),
        "xai_sim": xai_similarity,
        "intra_xai_stab": intra_stab,
        "num_samples": total_samples,
        "forgetting": (
            None if previous_task_acc is None
            else round(float(previous_task_acc) - clean_acc, 4)
        ),
    })
    return metrics


def _resize_explanation_map(map_2d: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Bilinearly resizes a 2D explanation map to the target spatial grid."""
    if map_2d.shape == target_shape:
        return map_2d
    tensor = torch.from_numpy(map_2d).float().view(1, 1, *map_2d.shape)
    resized = F.interpolate(tensor, size=target_shape, mode="bilinear", align_corners=False)
    return resized.view(*target_shape).numpy().astype(np.float64)


def compute_xai_similarity(
    saliency_map: Union[torch.Tensor, np.ndarray],
    reference_map: Union[torch.Tensor, np.ndarray],
    method: str = "cosine",
    eps: float = 1e-8,
) -> float:
    """
    Computes scale-invariant or normalized alignment between an explanation map (e.g. Grad-CAM)
    and a reference attention map (e.g. ViT CLS-to-patch attention from clip_attn_test.npy).

    If the two grids differ (e.g. 96x96 Grad-CAM vs 7x7 ViT patches), the saliency map is
    bilinearly resized onto the reference grid before comparison, as required by the cahier.

    Note for Member B & Cross-Effects:
      In ViT attention maps, [CLS]-to-patch attention excludes the [CLS] self-attention token,
      leaving spatial patch values in [0.001, 0.09] (sum ~ 0.35).
      Grad-CAM heatmaps are typically normalized to [0, 1].
      'cosine' similarity is inherently scale-invariant and ideal for comparing shapes.
      'normalized_l1' min-max normalizes both maps to [0, 1] before computing 1 - mean(|A - B|).

    Args:
        saliency_map: 2D array or tensor of shape (H, W).
        reference_map: 2D array or tensor of shape (H_ref, W_ref), e.g. (7, 7).
        method: 'cosine' (default) or 'normalized_l1'.
        eps: Small numerical constant for division stability.

    Returns:
        Scalar similarity score in [-1.0, 1.0] (for cosine) or [0.0, 1.0] (for normalized_l1).
    """
    if isinstance(saliency_map, torch.Tensor):
        saliency_map = saliency_map.detach().cpu().numpy()
    if isinstance(reference_map, torch.Tensor):
        reference_map = reference_map.detach().cpu().numpy()

    saliency_map = np.asarray(saliency_map, dtype=np.float64)
    reference_map = np.asarray(reference_map, dtype=np.float64)
    if saliency_map.ndim != 2 or reference_map.ndim != 2:
        raise ValueError(
            "compute_xai_similarity expects 2D maps; got "
            f"{saliency_map.shape} and {reference_map.shape}."
        )

    if saliency_map.shape != reference_map.shape:
        saliency_map = _resize_explanation_map(saliency_map, reference_map.shape)

    a = saliency_map.reshape(-1)
    b = reference_map.reshape(-1)

    if method == "cosine":
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < eps or norm_b < eps:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    elif method == "normalized_l1":
        a_norm = (a - a.min()) / (a.max() - a.min() + eps)
        b_norm = (b - b.min()) / (b.max() - b.min() + eps)
        return float(1.0 - np.mean(np.abs(a_norm - b_norm)))
    else:
        raise ValueError(f"Unknown similarity method: {method}. Choose 'cosine' or 'normalized_l1'.")

