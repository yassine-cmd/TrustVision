"""
TrustVision - Automated Test Suite for Scientific OOD Detection & Input Sanitization
===================================================================================
Tests:
1. InputSanitizer numerical anomaly guards (NaN, Inf, dimensional bounds).
2. MahalanobisOODDetector feature distance thresholding & checkpoint serialization.
3. EnergyOODDetector free-energy scoring and anomaly limits.
4. TrustVisionOODPipeline end-to-end risk-gate routing & latency (< 50 ms PLC deadline).
5. Interception of synthetic adversarial/OOD inputs (noise, checkerboard, random pictures).
"""

from pathlib import Path
import time
import numpy as np
import pytest
import torch
from PIL import Image

from src.data import NEU_CLASSES
from src.eval import load_model_checkpoint
from src.ood import (
    InputSanitizer,
    MahalanobisOODDetector,
    EnergyOODDetector,
    TrustVisionOODPipeline,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_BASE = ROOT / "checkpoints/cnn_base_seed42.pt"
CHECKPOINT_OOD_BASE = ROOT / "checkpoints/ood_base_seed42.pt"


# ==============================================================================
# 1. Input Sanitizer Tests
# ==============================================================================

def test_input_sanitizer_rejects_nan():
    sanitizer = InputSanitizer()
    nan_tensor = torch.full((1, 3, 96, 96), float("nan"))
    with pytest.raises(ValueError, match="NUMERICAL_ANOMALY"):
        sanitizer.sanitize(nan_tensor)


def test_input_sanitizer_rejects_inf():
    sanitizer = InputSanitizer()
    inf_tensor = torch.full((1, 3, 96, 96), float("inf"))
    with pytest.raises(ValueError, match="NUMERICAL_ANOMALY"):
        sanitizer.sanitize(inf_tensor)


def test_input_sanitizer_rejects_low_resolution():
    sanitizer = InputSanitizer(min_res=16)
    low_res = torch.rand(1, 3, 8, 8)
    with pytest.raises(ValueError, match="RESOLUTION_TOO_LOW"):
        sanitizer.sanitize(low_res)


def test_input_sanitizer_accepts_valid_pil():
    sanitizer = InputSanitizer()
    pil_img = Image.new("RGB", (128, 128), color=(128, 128, 128))
    tensor = sanitizer.sanitize(pil_img)
    assert tensor.shape == (1, 3, 96, 96)
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0


def test_input_sanitizer_accepts_numpy_uint8():
    sanitizer = InputSanitizer()
    arr = np.full((100, 100, 3), 128, dtype=np.uint8)
    tensor = sanitizer.sanitize(arr)
    assert tensor.shape == (1, 3, 96, 96)
    assert tensor.dtype == torch.float32


# ==============================================================================
# 2. Mahalanobis OOD Detector Tests
# ==============================================================================

def test_mahalanobis_detector_load_and_predict():
    assert CHECKPOINT_OOD_BASE.exists(), "Pre-fitted OOD detector checkpoint missing."
    detector = MahalanobisOODDetector.load(CHECKPOINT_OOD_BASE)
    assert detector.centroids is not None
    assert detector.inv_cov is not None
    assert detector.threshold > 0.0
    assert detector.centroids.shape == (len(NEU_CLASSES), 256)


def test_mahalanobis_detects_pure_noise():
    model = load_model_checkpoint(str(CHECKPOINT_BASE), device=torch.device("cpu"))
    detector = MahalanobisOODDetector.load(CHECKPOINT_OOD_BASE)

    # Pure noise in [0, 1]
    noise = torch.clamp(torch.randn(5, 3, 96, 96) * 0.25 + 0.5, 0.0, 1.0)
    with torch.no_grad():
        feats = model.get_features(noise)
        is_ood_mask, dists = detector.is_ood(feats)

    assert is_ood_mask.all().item(), "Pure Gaussian noise must be classified as OOD."
    assert (dists > detector.threshold * 5.0).all().item(), "Noise distance must far exceed threshold."


def test_mahalanobis_detects_checkerboard():
    model = load_model_checkpoint(str(CHECKPOINT_BASE), device=torch.device("cpu"))
    detector = MahalanobisOODDetector.load(CHECKPOINT_OOD_BASE)

    # 8x8 checkerboard pattern
    chk = torch.zeros(1, 3, 96, 96)
    for i in range(12):
        for j in range(12):
            if (i + j) % 2 == 0:
                chk[:, :, i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] = 1.0

    with torch.no_grad():
        feats = model.get_features(chk)
        is_ood_mask, dists = detector.is_ood(feats)

    assert is_ood_mask[0].item() is True, "Checkerboard pattern must be classified as OOD."


# ==============================================================================
# 3. Energy OOD Detector Tests
# ==============================================================================

def test_energy_ood_detector():
    energy_det = EnergyOODDetector(temperature=0.9276, min_energy_score=2.0, max_energy_score=40.0)
    normal_logits = torch.tensor([[5.0, 1.0, -1.0, 0.0, -2.0, 0.5]])
    is_ood, score = energy_det.is_ood(normal_logits)
    assert is_ood[0].item() is False
    assert 2.0 <= score[0].item() <= 40.0

    # Test standard low-energy OOD (flat/uncertain logits from Liu et al. 2020)
    flat_logits = torch.zeros((1, 6))  # S_E = T * ln(6) ≈ 1.66 < 2.0
    is_ood_flat, score_flat = energy_det.is_ood(flat_logits)
    assert is_ood_flat[0].item() is True
    assert score_flat[0].item() < 2.0

    # Test high-energy anomalous divergence
    exploded_logits = torch.tensor([[70.0, -50.0, 10.0, -30.0, -50.0, -20.0]])
    is_ood_exp, score_exp = energy_det.is_ood(exploded_logits)
    assert is_ood_exp[0].item() is True
    assert score_exp[0].item() > 40.0


# ==============================================================================
# 4. Composite TrustVision OOD Pipeline Tests
# ==============================================================================

def test_ood_pipeline_on_real_sample():
    model = load_model_checkpoint(str(CHECKPOINT_BASE), device=torch.device("cpu"))
    detector = MahalanobisOODDetector.load(CHECKPOINT_OOD_BASE)
    pipeline = TrustVisionOODPipeline(model, detector, temperature=0.9276, tau_risk=0.9940)

    # Real sample from data/NEU-CLS
    real_sample_path = ROOT / "data/NEU-CLS/crazing/crazing_1.jpg"
    if real_sample_path.exists():
        with Image.open(real_sample_path) as img:
            res = pipeline.inspect(img)
            assert res["is_ood"] is False
            assert res["status"] == "IN_DISTRIBUTION"
            assert res["mahalanobis_distance"] <= detector.threshold


def test_ood_pipeline_intercepts_noise():
    model = load_model_checkpoint(str(CHECKPOINT_BASE), device=torch.device("cpu"))
    detector = MahalanobisOODDetector.load(CHECKPOINT_OOD_BASE)
    pipeline = TrustVisionOODPipeline(model, detector, temperature=0.9276, tau_risk=0.9940)

    noise_tensor = torch.clamp(torch.randn(3, 96, 96) * 0.25 + 0.5, 0.0, 1.0)
    res = pipeline.inspect(noise_tensor)
    assert res["is_ood"] is True
    assert res["status"] == "OOD_REJECTED"
    assert "VOIE B" in res["routing_decision"]
    assert res["predicted_class"] == "HORS-DISTRIBUTION (OOD)"


def test_ood_pipeline_handles_nan_gracefully():
    model = load_model_checkpoint(str(CHECKPOINT_BASE), device=torch.device("cpu"))
    detector = MahalanobisOODDetector.load(CHECKPOINT_OOD_BASE)
    pipeline = TrustVisionOODPipeline(model, detector)

    nan_tensor = torch.full((3, 96, 96), float("nan"))
    res = pipeline.inspect(nan_tensor)
    assert res["is_ood"] is True
    assert res["status"] == "REJECT_NUMERICAL_ANOMALY"
    assert "VOIE B" in res["routing_decision"]


def test_ood_pipeline_latency_plc_compliance():
    model = load_model_checkpoint(str(CHECKPOINT_BASE), device=torch.device("cpu"))
    detector = MahalanobisOODDetector.load(CHECKPOINT_OOD_BASE)
    pipeline = TrustVisionOODPipeline(model, detector)

    dummy = torch.rand(3, 96, 96)
    # 5 Warmup passes
    for _ in range(5):
        _ = pipeline.inspect(dummy)

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = pipeline.inspect(dummy)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    assert p50 < 30.0, f"P50 median latency ({p50:.2f} ms) must be well below 30.0 ms."
    assert p90 < 50.0, f"P90 latency ({p90:.2f} ms) must be strictly below 50.0 ms PLC deadline."
