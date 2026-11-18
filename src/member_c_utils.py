"""
TrustVision - Member C shared utilities (src/member_c_utils.py)
==============================================================
Helpers shared by the SimCLR (S1) and EWC (S4) experiment code:

* temperature-scaling calibration on the shared calibration split,
* logits collection for post-hoc metrics,
* formatting helpers for the cross-effects table.

These helpers deliberately reuse ``src.eval.ModelWithTemperature`` and
``src.eval.compute_classification_metrics`` so Member C reports exactly the
same metric contract as S0/S2/S3.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.eval import ModelWithTemperature, compute_classification_metrics

CALIBRATION_T_MIN = 0.1
CALIBRATION_T_MAX = 10.0


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns concatenated (logits, labels) for a full DataLoader pass."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    logits_list = []
    labels_list = []
    for images, targets in loader:
        images = images.to(device)
        logits_list.append(model(images).detach().cpu())
        labels_list.append(targets.cpu())

    if not logits_list:
        raise ValueError("Cannot collect logits from an empty DataLoader.")

    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def _calibration_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> float:
    """Cross-entropy of temperature-scaled logits on the calibration split."""
    return float(F.cross_entropy(logits / temperature, labels).item())


def select_calibration_temperature(
    cal_logits: torch.Tensor,
    cal_labels: torch.Tensor,
    fitted_temperature: float,
    min_relative_gain: float = 1e-4,
) -> Tuple[float, bool, float, float]:
    """
    Keeps a fitted temperature only if it improves calibration NLL.

    The LBFGS fit in ``ModelWithTemperature`` clamps T at 1e-3. On very
    confident models a line-search step can hit that floor, the clamp then
    kills the gradient and the optimizer returns a degenerate T (observed on
    the S4 EWC+replay model: T=0.001, test NLL 2.92 vs 0.34 at T=1).

    We therefore restrict T to [0.1, 10]. A fit outside that range is rejected
    even if it lowers the calibration NLL (a perfectly classified calibration
    split would otherwise justify T -> 0). A fit inside the range is kept only
    if it lowers the calibration NLL by ``min_relative_gain``. This is a
    no-harm rule evaluated on the calibration split only, so the test split is
    never used to choose the temperature.

    Returns ``(temperature, accepted, nll_at_t1, nll_at_fitted)``. When the
    fitted value is out of range, ``nll_at_fitted`` is evaluated at the nearest
    bound (for diagnostics only); the returned temperature is 1.0.
    """
    base_nll = _calibration_nll(cal_logits, cal_labels, 1.0)
    fitted = float(fitted_temperature)
    if not math.isfinite(fitted) or fitted <= 0.0:
        return 1.0, False, base_nll, base_nll

    if not (CALIBRATION_T_MIN <= fitted <= CALIBRATION_T_MAX):
        fitted_nll = _calibration_nll(
            cal_logits,
            cal_labels,
            min(max(fitted, CALIBRATION_T_MIN), CALIBRATION_T_MAX),
        )
        return 1.0, False, base_nll, fitted_nll

    fitted_nll = _calibration_nll(cal_logits, cal_labels, fitted)
    threshold = abs(base_nll) * min_relative_gain
    accepted = fitted_nll < base_nll - threshold
    if not accepted:
        return 1.0, False, base_nll, fitted_nll
    return fitted, True, base_nll, fitted_nll


def fit_temperature_and_evaluate(
    model: torch.nn.Module,
    cal_loader: DataLoader,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
    max_iter: int = 100,
) -> Dict[str, Any]:
    """
    Fits temperature scaling on the calibration split and returns the
    calibrated test metrics (accuracy, macro-F1, ECE, NLL, Brier).

    Temperature scaling preserves the argmax, so accuracy and adversarial
    robustness are unchanged; only the probability quality is affected.
    A fitted temperature that fails the no-harm check (see
    ``select_calibration_temperature``) is replaced by T=1.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    calibrator = ModelWithTemperature(model).to(device)
    fitted_temperature = calibrator.set_temperature(cal_loader, device=device, max_iter=max_iter)

    cal_logits, cal_labels = collect_logits(model, cal_loader, device=device)
    temperature, accepted, cal_nll, cal_nll_fitted = select_calibration_temperature(
        cal_logits, cal_labels, fitted_temperature
    )

    logits, labels = collect_logits(model, test_loader, device=device)
    probs = F.softmax(logits / temperature, dim=1)
    metrics = compute_classification_metrics(probs, labels)
    metrics["temperature"] = round(float(temperature), 4)
    metrics["temperature_fitted"] = round(float(fitted_temperature), 4)
    metrics["calibration_accepted"] = accepted
    metrics["calibration_nll"] = round(cal_nll, 4)
    metrics["calibration_nll_fitted"] = round(cal_nll_fitted, 4)
    return metrics


def calibrated_summary(calibrated: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes a calibrated metric dict to the cross-effects column names."""
    return {
        "acc": calibrated.get("accuracy"),
        "macro_f1": calibrated.get("macro_f1"),
        "ece": calibrated.get("ece"),
        "nll": calibrated.get("nll"),
        "brier": calibrated.get("brier"),
        "temperature": calibrated.get("temperature"),
        "temperature_fitted": calibrated.get("temperature_fitted"),
        "calibration_accepted": calibrated.get("calibration_accepted"),
        "calibration_nll": calibrated.get("calibration_nll"),
        "calibration_nll_fitted": calibrated.get("calibration_nll_fitted"),
    }
