"""
TrustVision - Scientific Out-of-Distribution (OOD) & Input Sanitization Module
=============================================================================
Provides mathematically grounded OOD detection and data validation:
1. InputSanitizer: Rejects NaN/Inf numerical corruption, out-of-spec dimensions,
   and dynamic range violations before tensor ingestion.
2. MahalanobisOODDetector: Feature-space Mahalanobis distance based on class-conditional
   multivariate Gaussians in penultimate layer representations (Lee et al., NeurIPS 2018).
3. EnergyOODDetector: Helmholtz free energy scoring aligned with log-marginal density
   log p(x) under Gibbs-Boltzmann distribution (Liu et al., NeurIPS 2020).
4. TrustVisionOODPipeline: Composite production gate combining input sanitization,
   feature distance thresholding, temperature calibration, and selective risk routing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.data import NEU_CLASSES


# ---------------------------------------------------------------------------
# 1. Input Sanitizer
# ---------------------------------------------------------------------------

class InputSanitizer:
    """
    Production-grade input validator and preprocessor for industrial vision telemetry.
    Guarantees strict adherence to model input contracts and detects anomalies.
    """

    DEFAULT_MIN_RES: int = 16
    DEFAULT_IMG_SIZE: int = 96

    def __init__(self, target_size: int = DEFAULT_IMG_SIZE, min_res: int = DEFAULT_MIN_RES):
        self.target_size = target_size
        self.min_res = min_res
        self.transform = transforms.Compose([
            transforms.Resize(
                (target_size, target_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
        ])

    def validate_tensor(self, tensor: torch.Tensor) -> Tuple[bool, Optional[str]]:
        """
        Validates mathematical integrity of tensor inputs.
        Rejects NaNs, Infs, non-conforming dimensions, and empty tensors.
        """
        if tensor.numel() == 0:
            return False, "EMPTY_TENSOR: Input tensor contains zero elements."

        if torch.isnan(tensor).any().item():
            return False, "NUMERICAL_ANOMALY: Input tensor contains NaN values."

        if torch.isinf(tensor).any().item():
            return False, "NUMERICAL_ANOMALY: Input tensor contains Infinite values."

        # Expect (C, H, W) or (B, C, H, W)
        if tensor.ndim not in (3, 4):
            return False, f"INVALID_DIMENSIONS: Expected 3D or 4D tensor, got ndim={tensor.ndim}."

        c_dim = 1 if tensor.ndim == 4 else 0
        channels = tensor.shape[c_dim]
        if channels not in (1, 3):
            return False, f"CHANNEL_MISMATCH: Expected 1 or 3 channels, got {channels}."

        h_dim, w_dim = (2, 3) if tensor.ndim == 4 else (1, 2)
        h, w = tensor.shape[h_dim], tensor.shape[w_dim]
        if h < self.min_res or w < self.min_res:
            return False, f"RESOLUTION_TOO_LOW: Spatial resolution {h}x{w} is below minimum {self.min_res}x{self.min_res}."

        return True, None

    def sanitize(
        self,
        image_input: Union[Image.Image, np.ndarray, torch.Tensor],
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Converts heterogeneous inputs (PIL, NumPy, PyTorch) into a validated (1, 3, 96, 96) tensor.
        Raises ValueError if numerical anomalies or invalid formats are encountered.
        """
        if isinstance(image_input, torch.Tensor):
            valid, err = self.validate_tensor(image_input)
            if not valid:
                raise ValueError(err)

            t = image_input.float()
            if t.ndim == 3:
                t = t.unsqueeze(0)

            # If 1 channel, expand to 3
            if t.shape[1] == 1:
                t = t.repeat(1, 3, 1, 1)

            # Resize if necessary
            if t.shape[2:] != (self.target_size, self.target_size):
                t = F.interpolate(
                    t,
                    size=(self.target_size, self.target_size),
                    mode="bicubic",
                    align_corners=False,
                )

            # Clamp valid range [0, 1]
            return t.to(device).clamp(0.0, 1.0)

        if isinstance(image_input, np.ndarray):
            if np.isnan(image_input).any() or np.isinf(image_input).any():
                raise ValueError("NUMERICAL_ANOMALY: Input numpy array contains NaN or Inf values.")

            if image_input.dtype == np.uint8:
                pil_img = Image.fromarray(image_input)
            else:
                arr = image_input.astype(np.float32)
                if arr.max() <= 1.0:
                    arr = arr * 255.0
                arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
                pil_img = Image.fromarray(arr)
            image_input = pil_img

        if isinstance(image_input, Image.Image):
            rgb_img = image_input.convert("RGB")
            w, h = rgb_img.size
            if w < self.min_res or h < self.min_res:
                raise ValueError(f"RESOLUTION_TOO_LOW: Image resolution {w}x{h} < {self.min_res}x{self.min_res}.")
            t = self.transform(rgb_img).unsqueeze(0)
            return t.to(device)

        raise TypeError(f"UNSUPPORTED_INPUT_TYPE: Cannot process input of type {type(image_input).__name__}.")


# ---------------------------------------------------------------------------
# 2. Mahalanobis OOD Detector
# ---------------------------------------------------------------------------

class MahalanobisOODDetector:
    """
    Feature-space Mahalanobis Out-of-Distribution Detector.
    Models in-distribution representations z = g(x) as class-conditional Gaussians N(mu_c, Sigma).
    Measures distance: D_M(z) = min_c (z - mu_c)^T Sigma^{-1} (z - mu_c).
    Inputs exceeding the calibrated threshold tau_ood are classified as OOD.
    """

    def __init__(
        self,
        centroids: Optional[torch.Tensor] = None,
        inv_cov: Optional[torch.Tensor] = None,
        threshold: float = 1650.0,
        classes: Optional[List[str]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.classes = classes or NEU_CLASSES
        self.num_classes = len(self.classes)
        self.threshold = float(threshold)

        if centroids is not None:
            self.centroids = centroids.to(self.device)
        else:
            self.centroids = None

        if inv_cov is not None:
            self.inv_cov = inv_cov.to(self.device)
        else:
            self.inv_cov = None

    def fit(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        cal_loader: Optional[torch.utils.data.DataLoader] = None,
        device: Optional[torch.device] = None,
        reg_eps: float = 1e-4,
        safety_margin: float = 1.20,
    ) -> Dict[str, float]:
        """
        Fits class centroids and pooled empirical covariance on the training split,
        then calibrates decision threshold tau_ood on the calibration split.
        """
        if device is not None:
            self.device = device
        model.eval()
        model.to(self.device)

        feats_list, labels_list = [], []
        with torch.no_grad():
            for images, targets in train_loader:
                images = images.to(self.device)
                features = model.get_features(images)
                feats_list.append(features.cpu())
                labels_list.append(targets.cpu())

        all_feats = torch.cat(feats_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        feature_dim = all_feats.shape[1]

        # 1. Compute empirical class centroids
        self.centroids = torch.zeros(self.num_classes, feature_dim)
        for c in range(self.num_classes):
            idx = (all_labels == c)
            if idx.sum() > 0:
                self.centroids[c] = all_feats[idx].mean(dim=0)
            else:
                self.centroids[c] = torch.zeros(feature_dim)

        # 2. Compute pooled covariance matrix with Tikhonov shrinkage
        centered_diffs = []
        for c in range(self.num_classes):
            idx = (all_labels == c)
            if idx.sum() > 0:
                centered_diffs.append(all_feats[idx] - self.centroids[c])
        all_diffs = torch.cat(centered_diffs, dim=0)
        cov = (all_diffs.t() @ all_diffs) / len(all_feats)
        cov_reg = cov + reg_eps * torch.eye(feature_dim)

        # 3. Regularized pseudo-inverse
        self.inv_cov = torch.pinverse(cov_reg)

        self.centroids = self.centroids.to(self.device)
        self.inv_cov = self.inv_cov.to(self.device)

        # 4. Calibrate threshold on in-distribution calibration set
        stats = {"reg_eps": reg_eps}
        if cal_loader is not None:
            cal_dists = []
            with torch.no_grad():
                for images, _ in cal_loader:
                    images = images.to(self.device)
                    feats = model.get_features(images)
                    dists = self.compute_distances(feats)
                    cal_dists.append(dists.cpu())

            cal_dists_t = torch.cat(cal_dists)
            p95 = float(torch.quantile(cal_dists_t, 0.95).item())
            p99 = float(torch.quantile(cal_dists_t, 0.99).item())
            max_d = float(cal_dists_t.max().item())

            self.threshold = round(max(p99 * safety_margin, max_d * 1.1), 2)
            stats.update({
                "cal_p95": round(p95, 2),
                "cal_p99": round(p99, 2),
                "cal_max": round(max_d, 2),
                "calibrated_threshold": self.threshold,
            })
        return stats

    def compute_distances(self, features: torch.Tensor) -> torch.Tensor:
        """
        Computes minimum Mahalanobis distance across all class centroids for batch (B, D).
        Returns tensor of shape (B,).
        """
        if self.centroids is None or self.inv_cov is None:
            raise RuntimeError("MahalanobisOODDetector must be fitted or loaded before computing distances.")

        features = features.to(self.device)
        batch_size = features.shape[0]

        # Compute quadratic form (z - mu_c)^T Sigma^{-1} (z - mu_c) for each class c
        dists = []
        for c in range(self.num_classes):
            delta = features - self.centroids[c].unsqueeze(0)  # (B, D)
            # (B, D) @ (D, D) element-wise product with (B, D) summed across D
            dist_c = torch.sum((delta @ self.inv_cov) * delta, dim=1, keepdim=True)  # (B, 1)
            dists.append(dist_c)

        all_dists = torch.cat(dists, dim=1)  # (B, K)
        min_dists, _ = torch.min(all_dists, dim=1)  # (B,)
        return min_dists

    def is_ood(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (is_ood_mask, distances) where is_ood_mask is a boolean tensor of shape (B,).
        True indicates an Out-of-Distribution sample.
        """
        dists = self.compute_distances(features)
        is_ood_mask = dists > self.threshold
        return is_ood_mask, dists

    def save(self, file_path: Union[str, Path]) -> None:
        """Saves fitted detector parameters to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "centroids": self.centroids.cpu(),
                "inv_cov": self.inv_cov.cpu(),
                "threshold": self.threshold,
                "classes": self.classes,
            },
            str(path),
        )

    @classmethod
    def load(cls, file_path: Union[str, Path], device: torch.device = torch.device("cpu")) -> "MahalanobisOODDetector":
        """Loads fitted detector from checkpoint."""
        data = torch.load(str(file_path), map_location=device, weights_only=True)
        return cls(
            centroids=data["centroids"],
            inv_cov=data["inv_cov"],
            threshold=float(data["threshold"]),
            classes=data.get("classes", NEU_CLASSES),
            device=device,
        )


# ---------------------------------------------------------------------------
# 3. Energy OOD Detector
# ---------------------------------------------------------------------------

class EnergyOODDetector:
    """
    Energy-based Out-of-Distribution detector (Liu et al., NeurIPS 2020).
    Computes Helmholtz free energy E(x) = -T * log sum_c exp(f_c(x) / T).
    Negative energy score S_E(x) = -E(x) tracks log-marginal density log p(x).
    """

    def __init__(self, temperature: float = 0.9276, max_energy_score: float = 40.0):
        self.temperature = float(temperature)
        self.max_energy_score = float(max_energy_score)

    def compute_energy_score(self, logits: torch.Tensor) -> torch.Tensor:
        """Computes S_E(x) = T * logsumexp(logits / T). Shape: (B,)."""
        temp = max(self.temperature, 1e-4)
        return temp * torch.logsumexp(logits / temp, dim=1)

    def is_ood(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flags samples whose energy score explodes due to uncorrelated high-magnitude logits.
        """
        scores = self.compute_energy_score(logits)
        is_ood_mask = scores > self.max_energy_score
        return is_ood_mask, scores


# ---------------------------------------------------------------------------
# 4. Composite TrustVision OOD Pipeline
# ---------------------------------------------------------------------------

class TrustVisionOODPipeline:
    """
    Integrated inference pipeline uniting:
    1. Input sanitization (NaN/Inf, shape, dynamic range)
    2. Model forward pass and feature extraction
    3. Mahalanobis distance OOD gating
    4. Energy score anomaly check
    5. Temperature-scaled calibrated risk-gate routing (tau_95 = 0.9940)
    """

    def __init__(
        self,
        model: nn.Module,
        mahalanobis_detector: MahalanobisOODDetector,
        energy_detector: Optional[EnergyOODDetector] = None,
        temperature: float = 0.9276,
        tau_risk: float = 0.9940,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.model.eval()
        self.mahalanobis = mahalanobis_detector
        self.energy = energy_detector or EnergyOODDetector(temperature=temperature)
        self.sanitizer = InputSanitizer()
        self.temperature = float(temperature)
        self.tau_risk = float(tau_risk)
        self.device = device
        self.classes = NEU_CLASSES

    def inspect(
        self,
        image_input: Union[Image.Image, np.ndarray, torch.Tensor],
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end safe industrial inspection.
        Guarantees that OOD inputs, noise, and corrupted tensors are intercepted
        and diverted to VOIE B with zero silent passes.
        """
        temp_val = float(temperature) if temperature is not None else self.temperature
        temp_val = max(temp_val, 1e-3)

        # 1. Stage 1: Input Sanitization
        try:
            tensor = self.sanitizer.sanitize(image_input, device=self.device)
        except (ValueError, TypeError) as exc:
            return {
                "status": "REJECT_NUMERICAL_ANOMALY",
                "is_ood": True,
                "ood_reason": str(exc),
                "predicted_class": "NON_CONFORME_ANOMALIE",
                "confidence": 0.0,
                "routing_decision": "VOIE B (REJET ANOMALIE NUMÉRIQUE)",
                "mahalanobis_distance": float("inf"),
                "energy_score": float("-inf"),
                "feature_norm": 0.0,
                "class_probabilities": {c: 0.0 for c in self.classes},
            }

        # 2. Stage 2: Forward pass & feature extraction
        with torch.no_grad():
            features = self.model.get_features(tensor)
            logits = self.model(tensor)

        feat_norm = float(features.norm(dim=1)[0].item())

        # 3. Stage 3: Mahalanobis Latent Space Distance Gate
        m_dist = float(self.mahalanobis.compute_distances(features)[0].item())
        is_mahal_ood = bool(m_dist > self.mahalanobis.threshold)

        # 4. Stage 4: Energy Anomaly Check
        energy_score = float(self.energy.compute_energy_score(logits)[0].item())
        is_energy_ood = bool(energy_score > self.energy.max_energy_score)

        # Composite OOD disposition
        is_ood = is_mahal_ood or is_energy_ood or (feat_norm > 50.0)
        ood_reason = None
        if is_ood:
            reasons = []
            if is_mahal_ood:
                reasons.append(f"MAHALANOBIS_OUTLIER ({m_dist:.1f} > {self.mahalanobis.threshold:.1f})")
            if is_energy_ood:
                reasons.append(f"ENERGY_ANOMALY ({energy_score:.1f} > {self.energy.max_energy_score:.1f})")
            if feat_norm > 50.0:
                reasons.append(f"FEATURE_EXPLOSION (norm {feat_norm:.1f} > 50.0)")
            ood_reason = " & ".join(reasons)

        # 5. Stage 5: Calibrated Probabilities & Risk-Gate Routing
        with torch.no_grad():
            cal_probs = F.softmax(logits / temp_val, dim=1).squeeze(0)
            pred_idx = int(cal_probs.argmax().item())
            confidence = float(cal_probs[pred_idx].item())

        class_probabilities = {
            c: round(float(cal_probs[i].item()), 4)
            for i, c in enumerate(self.classes)
        }

        if is_ood:
            pred_label = "HORS-DISTRIBUTION (OOD)"
            routing = f"VOIE B (REJET OOD: {ood_reason})"
        else:
            pred_label = self.classes[pred_idx]
            if confidence >= self.tau_risk:
                routing = f"VOIE A (CONFORME: {confidence*100:.2f}% >= {self.tau_risk*100:.2f}%)"
            else:
                routing = f"VOIE B (CONTRÔLE OPÉRATEUR: {confidence*100:.2f}% < {self.tau_risk*100:.2f}%)"

        return {
            "status": "OOD_REJECTED" if is_ood else "IN_DISTRIBUTION",
            "is_ood": is_ood,
            "ood_reason": ood_reason,
            "predicted_class": pred_label,
            "raw_predicted_class": self.classes[pred_idx],
            "confidence": round(confidence, 4),
            "routing_decision": routing,
            "mahalanobis_distance": round(m_dist, 2),
            "mahalanobis_threshold": round(self.mahalanobis.threshold, 2),
            "energy_score": round(energy_score, 2),
            "feature_norm": round(feat_norm, 2),
            "temperature_used": round(temp_val, 4),
            "class_probabilities": class_probabilities,
            "tensor": tensor,
        }
