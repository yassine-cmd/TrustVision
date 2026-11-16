"""
TrustVision - XAI, Calibration & Adversarial Robustness Module (src/xai_cal_adv.py)
Implements:
1. XAI: Grad-CAM (via get_last_conv_layer), Integrated Gradients with completeness check.
2. Calibration: ECE with non-parametric bootstrap CI, Temperature Scaling fitted on cal split.
3. FGSM Robustness: Evaluation sweep across epsilon values [0, 0.01, 0.03, 0.05, 0.10].
4. Adversarial Defense & Intra-Pipeline Stability: Compares clean vs robust CNNs and their explanations.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.eval import (
    compute_classification_metrics,
    compute_ece,
    fgsm_attack,
    compute_xai_similarity,
    compute_intra_pipeline_stability,
    load_model_checkpoint,
    TemperatureScaler,
    format_stage_row,
)


# ==============================================================================
# 1. EXPLAINABLE AI (XAI) : Grad-CAM & Integrated Gradients
# ==============================================================================

class GradCAM:
    """
    Grad-CAM implementation adhering strictly to get_last_conv_layer contract.
    Computes class-discriminative saliency maps via gradient pooling.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

        if hasattr(self.model, "get_last_conv_layer"):
            self.target_layer = self.model.get_last_conv_layer()
        else:
            raise AttributeError("Model does not expose get_last_conv_layer() contract.")

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def __call__(
        self,
        images: torch.Tensor,
        target_classes: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> np.ndarray:
        """
        Generates Grad-CAM heatmaps for batch of images.

        Args:
            images: Tensor of shape (B, 3, H, W) with values in [0, 1].
            target_classes: Optional target class indices (B,). If None, uses argmax.

        Returns:
            np.ndarray of shape (B, H, W) normalized to [0, 1].
        """
        self.model.zero_grad()
        device = images.device
        logits = self.model(images)

        if target_classes is None:
            target_classes = logits.argmax(dim=1)
        elif isinstance(target_classes, list):
            target_classes = torch.tensor(target_classes, device=device)

        # One-hot score for targeted classes
        one_hot = torch.zeros_like(logits)
        for i, c in enumerate(target_classes):
            one_hot[i, c] = 1.0

        logits.backward(gradient=one_hot, retain_graph=False)

        # Alpha weights = GAP over spatial dims of gradients
        # activations: (B, C, H_c, W_c), gradients: (B, C, H_c, W_c)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input image size (H, W)
        cam_upsampled = F.interpolate(
            cam, size=(images.size(2), images.size(3)), mode="bilinear", align_corners=False
        )
        cam_np = cam_upsampled.squeeze(1).cpu().numpy()

        # Min-Max normalize each map to [0, 1]
        normalized_maps = []
        for i in range(cam_np.shape[0]):
            m = cam_np[i]
            denom = m.max() - m.min()
            if denom > 1e-8:
                m_norm = (m - m.min()) / denom
            else:
                m_norm = np.zeros_like(m)
            normalized_maps.append(m_norm)

        self.model.zero_grad()
        return np.array(normalized_maps, dtype=np.float32)


def integrated_gradients(
    model: nn.Module,
    images: torch.Tensor,
    target_classes: Optional[Union[List[int], torch.Tensor]] = None,
    steps: int = 30,
    baseline: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Computes Integrated Gradients (IG) with black baseline (zeros) and completeness check.

    Args:
        model: Evaluated neural network.
        images: Input images (B, 3, H, W) in [0, 1].
        target_classes: Target classes (B,). If None, uses model argmax.
        steps: Number of Riemann interpolation steps (default 30).
        baseline: Baseline image (default black image, zeros).

    Returns:
        (saliency_maps_np, completeness_stats_dict)
    """
    model.eval()
    device = images.device
    batch_size = images.size(0)

    if baseline is None:
        baseline = torch.zeros_like(images)

    with torch.no_grad():
        baseline_logits = model(baseline)
        target_logits = model(images)

    if target_classes is None:
        target_classes = target_logits.argmax(dim=1)
    elif isinstance(target_classes, list):
        target_classes = torch.tensor(target_classes, device=device)

    # Accumulate gradients along the interpolation path
    total_gradients = torch.zeros_like(images)
    diff = images - baseline

    for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
        interpolated = baseline + alpha * diff
        interpolated.requires_grad_(True)
        model.zero_grad()

        logits = model(interpolated)
        score = sum(logits[i, target_classes[i]] for i in range(batch_size))
        score.backward()

        total_gradients += interpolated.grad.detach()

    # Average gradient * (input - baseline)
    avg_gradients = total_gradients / steps
    ig_attributions = diff * avg_gradients

    # Completeness verification: sum(IG) should approximate F(x) - F(x0)
    ig_sum_per_sample = ig_attributions.view(batch_size, -1).sum(dim=1)
    target_diff = torch.tensor([
        (target_logits[i, target_classes[i]] - baseline_logits[i, target_classes[i]]).item()
        for i in range(batch_size)
    ], device=device)

    abs_errors = torch.abs(ig_sum_per_sample - target_diff)
    denom = torch.abs(target_diff) + 1e-6
    rel_errors = abs_errors / denom

    completeness_stats = {
        "max_abs_completeness_error": float(abs_errors.max().item()),
        "mean_abs_completeness_error": float(abs_errors.mean().item()),
        "mean_rel_completeness_error": float(rel_errors.mean().item()),
    }

    # Aggregate spatial attribution across color channels (L2 norm)
    spatial_maps = torch.norm(ig_attributions, dim=1).detach().cpu().numpy()

    # Min-Max normalize each attribution map to [0, 1]
    normalized_maps = []
    for i in range(batch_size):
        m = spatial_maps[i]
        denom = m.max() - m.min()
        if denom > 1e-8:
            m_norm = (m - m.min()) / denom
        else:
            m_norm = np.zeros_like(m)
        normalized_maps.append(m_norm)

    model.zero_grad()
    return np.array(normalized_maps, dtype=np.float32), completeness_stats


def compare_xai_methods(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int = 100,
) -> Dict[str, Any]:
    """
    Compares Grad-CAM and Integrated Gradients saliency maps across dataloader samples.
    Computes cosine similarity between both explanation paradigms.
    """
    grad_cam_explainer = GradCAM(model)
    all_grad_cam = []
    all_ig = []

    collected = 0
    completeness_records = []

    for images, _ in dataloader:
        images = images.to(device)
        batch_size = images.size(0)

        # Grad-CAM
        cam_maps = grad_cam_explainer(images)
        # Integrated Gradients
        ig_maps, comp_stats = integrated_gradients(model, images, steps=25)
        completeness_records.append(comp_stats["mean_rel_completeness_error"])

        all_grad_cam.append(cam_maps)
        all_ig.append(ig_maps)
        collected += batch_size

        if collected >= max_samples:
            break

    grad_cam_explainer.remove_hooks()

    grad_cam_cat = np.concatenate(all_grad_cam, axis=0)[:max_samples]
    ig_cat = np.concatenate(all_ig, axis=0)[:max_samples]

    sims = [
        compute_xai_similarity(grad_cam_cat[i], ig_cat[i], method="cosine")
        for i in range(len(grad_cam_cat))
    ]

    return {
        "num_compared": len(sims),
        "mean_cosine_similarity": round(float(np.mean(sims)), 4),
        "std_cosine_similarity": round(float(np.std(sims)), 4),
        "mean_ig_completeness_rel_error": round(float(np.mean(completeness_records)), 4),
        "grad_cam_sample_maps": grad_cam_cat[:6],
        "ig_sample_maps": ig_cat[:6],
    }


# ==============================================================================
# 2. CALIBRATION : ECE Bootstrap, Temperature Scaling, Reliability Diagram
# ==============================================================================

def compute_ece_with_bootstrap(
    probs: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    n_bins: int = 10,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE) and non-parametric bootstrap confidence intervals.
    Essential because the calibration set contains only 180 samples.
    """
    if isinstance(probs, np.ndarray):
        probs = torch.from_numpy(probs).float()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).long()

    # Point estimate
    point_ece, bin_stats = compute_ece(probs, labels, n_bins=n_bins)

    rng = np.random.RandomState(seed)
    n_samples = probs.shape[0]
    bootstrap_eces = []

    for _ in range(n_bootstraps):
        boot_idx = rng.randint(0, n_samples, size=n_samples)
        boot_probs = probs[boot_idx]
        boot_labels = labels[boot_idx]
        b_ece, _ = compute_ece(boot_probs, boot_labels, n_bins=n_bins)
        bootstrap_eces.append(b_ece)

    bootstrap_eces = np.array(bootstrap_eces)
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(bootstrap_eces, 100.0 * alpha))
    ci_upper = float(np.percentile(bootstrap_eces, 100.0 * (1.0 - alpha)))

    return {
        "ece": round(point_ece, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "confidence_level": confidence_level,
        "n_bootstraps": n_bootstraps,
        "bin_stats": bin_stats,
    }


def fit_and_evaluate_temperature_scaling(
    model: nn.Module,
    cal_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Fits Temperature Scaling strictly on the 180 calibration images via LBFGS.
    Evaluates effect on test set: verifies that argmax (accuracy) is unchanged,
    and quantifies improvement in ECE and NLL.
    """
    model.eval()
    model.to(device)

    # 1. Collect calibration logits & labels
    cal_logits_list = []
    cal_labels_list = []
    with torch.no_grad():
        for images, targets in cal_loader:
            images = images.to(device)
            cal_logits_list.append(model(images).cpu())
            cal_labels_list.append(targets)

    cal_logits = torch.cat(cal_logits_list)
    cal_labels = torch.cat(cal_labels_list)

    # Fit temperature scaler
    scaler = TemperatureScaler(initial_temp=1.5)
    best_temp = scaler.fit(cal_logits, cal_labels, lr=0.05, max_iter=250)

    # 2. Evaluate on test set (360 images)
    test_logits_list = []
    test_labels_list = []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            test_logits_list.append(model(images).cpu())
            test_labels_list.append(targets)

    test_logits = torch.cat(test_logits_list)
    test_labels = torch.cat(test_labels_list)

    # Uncalibrated metrics
    uncal_probs = F.softmax(test_logits, dim=1)
    uncal_metrics = compute_classification_metrics(uncal_probs, test_labels, n_bins=10)
    uncal_boot = compute_ece_with_bootstrap(uncal_probs, test_labels, n_bins=10, n_bootstraps=1000)

    # Calibrated metrics
    cal_logits_test = scaler(test_logits)
    cal_probs = F.softmax(cal_logits_test, dim=1)
    cal_metrics = compute_classification_metrics(cal_probs, test_labels, n_bins=10)
    cal_boot = compute_ece_with_bootstrap(cal_probs, test_labels, n_bins=10, n_bootstraps=1000)

    # Strict argmax invariance assertion
    raw_preds = test_logits.argmax(dim=1)
    cal_preds = cal_logits_test.argmax(dim=1)
    argmax_invariant = bool(torch.equal(raw_preds, cal_preds))

    return {
        "temperature": round(best_temp, 4),
        "argmax_invariant": argmax_invariant,
        "uncalibrated": {
            "accuracy": uncal_metrics["accuracy"],
            "ece": uncal_boot["ece"],
            "ece_ci": [uncal_boot["ci_lower"], uncal_boot["ci_upper"]],
            "nll": uncal_metrics["nll"],
            "brier": uncal_metrics["brier"],
            "bin_stats": uncal_boot["bin_stats"],
        },
        "calibrated": {
            "accuracy": cal_metrics["accuracy"],
            "ece": cal_boot["ece"],
            "ece_ci": [cal_boot["ci_lower"], cal_boot["ci_upper"]],
            "nll": cal_metrics["nll"],
            "brier": cal_metrics["brier"],
            "bin_stats": cal_boot["bin_stats"],
        },
        "ece_reduction": round(uncal_boot["ece"] - cal_boot["ece"], 4),
        "nll_reduction": round(uncal_metrics["nll"] - cal_metrics["nll"], 4),
    }


# ==============================================================================
# 3. ADVERSARIAL FGSM SWEEP & DEFENSE
# ==============================================================================

def evaluate_fgsm_sweep(
    model: nn.Module,
    test_loader: DataLoader,
    epsilons: List[float] = [0.0, 0.01, 0.03, 0.05, 0.10],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Evaluates model accuracy and NLL under FGSM attack across a range of epsilon budgets.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    sweep_results = {}
    for eps in epsilons:
        correct = 0
        total = 0
        total_loss = 0.0

        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            if eps > 0.0:
                images_adv = fgsm_attack(model, images, targets, eps=eps)
            else:
                images_adv = images

            with torch.no_grad():
                outputs = model(images_adv)
                preds = outputs.argmax(dim=1)
                loss = F.cross_entropy(outputs, targets)

                correct += (preds == targets).sum().item()
                total += targets.size(0)
                total_loss += loss.item() * targets.size(0)

        acc = correct / total if total > 0 else 0.0
        nll = total_loss / total if total > 0 else 0.0

        sweep_results[str(eps)] = {
            "accuracy": round(acc, 4),
            "nll": round(nll, 4),
            "correct": correct,
            "total": total,
        }

    return sweep_results


def compare_clean_vs_adversarial(
    clean_checkpoint: Union[str, Path, nn.Module],
    adv_checkpoint: Union[str, Path, nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    epsilons: List[float] = [0.0, 0.01, 0.03, 0.05, 0.10],
    compute_explanation_stability: bool = True,
    n_xai_samples: int = 60,
) -> Dict[str, Any]:
    """
    Compares clean baseline CNN vs defensively trained CNN:
    1. Clean accuracy
    2. Robust accuracy curves under FGSM
    3. Calibration metrics (ECE)
    4. Explanation stability: cosine similarity of Grad-CAM and IG maps before vs after defense.
    """
    clean_model = load_model_checkpoint(clean_checkpoint, device=device)
    adv_model = load_model_checkpoint(adv_checkpoint, device=device)

    # 1. Sweep curves
    clean_sweep = evaluate_fgsm_sweep(clean_model, test_loader, epsilons=epsilons, device=device)
    adv_sweep = evaluate_fgsm_sweep(adv_model, test_loader, epsilons=epsilons, device=device)

    # 2. XAI Stability (Clean vs Adv)
    xai_stability = {}
    if compute_explanation_stability:
        clean_gradcam = GradCAM(clean_model)
        adv_gradcam = GradCAM(adv_model)

        collected = 0
        all_cam_clean = []
        all_cam_adv = []
        all_ig_clean = []
        all_ig_adv = []

        for images, _ in test_loader:
            images = images.to(device)
            bsz = images.size(0)

            cam_clean = clean_gradcam(images)
            cam_adv = adv_gradcam(images)
            ig_clean, _ = integrated_gradients(clean_model, images, steps=20)
            ig_adv, _ = integrated_gradients(adv_model, images, steps=20)

            all_cam_clean.append(cam_clean)
            all_cam_adv.append(cam_adv)
            all_ig_clean.append(ig_clean)
            all_ig_adv.append(ig_adv)
            collected += bsz

            if collected >= n_xai_samples:
                break

        clean_gradcam.remove_hooks()
        adv_gradcam.remove_hooks()

        cam_clean_arr = np.concatenate(all_cam_clean, axis=0)[:n_xai_samples]
        cam_adv_arr = np.concatenate(all_cam_adv, axis=0)[:n_xai_samples]
        ig_clean_arr = np.concatenate(all_ig_clean, axis=0)[:n_xai_samples]
        ig_adv_arr = np.concatenate(all_ig_adv, axis=0)[:n_xai_samples]

        gradcam_stability = compute_intra_pipeline_stability(cam_clean_arr, cam_adv_arr, method="cosine")
        ig_stability = compute_intra_pipeline_stability(ig_clean_arr, ig_adv_arr, method="cosine")

        xai_stability = {
            "n_samples": n_xai_samples,
            "gradcam_stability_cosine": round(gradcam_stability, 4),
            "ig_stability_cosine": round(ig_stability, 4),
            "gradcam_sample_clean": cam_clean_arr[:4].tolist(),
            "gradcam_sample_adv": cam_adv_arr[:4].tolist(),
        }

    return {
        "clean_model_sweep": clean_sweep,
        "adv_model_sweep": adv_sweep,
        "clean_rob_acc_003": clean_sweep["0.03"]["accuracy"],
        "adv_rob_acc_003": adv_sweep["0.03"]["accuracy"],
        "robustness_gain_003": round(adv_sweep["0.03"]["accuracy"] - clean_sweep["0.03"]["accuracy"], 4),
        "explanation_stability": xai_stability,
    }
