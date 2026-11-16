"""
TrustVision - Gradio Tab 3 Audit Handler (src/gradio_tab3.py)
Provides ultra-fast (< 0.5s) real-time interactive inference for Gradio Tab 3:
- Sliders: Temperature T (0.1 to 3.0), FGSM Epsilon (0.0 to 0.10).
- Model selection: Baseline CNN vs Adversarially Defended CNN.
- Outputs: Perturbed image, calibrated prediction & confidence, Grad-CAM & IG heatmaps.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.data import NEU_CLASSES
from src.eval import load_model_checkpoint, fgsm_attack
from src.xai_cal_adv import GradCAM, integrated_gradients


class TrustVisionTab3Auditor:
    """
    Stateful inference auditor holding pre-loaded models in memory to ensure
    sub-0.5-second responses without dynamic re-training or disk thrashing.
    """

    def __init__(
        self,
        base_checkpoint: str = "checkpoints/cnn_base_seed42.pt",
        adv_checkpoint: Optional[str] = "checkpoints/cnn_adv_seed42.pt",
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.classes = NEU_CLASSES

        print(f"[Gradio Tab 3] Pre-loading checkpoints on {self.device}...")
        self.models = {
            "base": load_model_checkpoint(base_checkpoint, device=self.device),
        }
        self.explainers = {
            "base": GradCAM(self.models["base"]),
        }

        if adv_checkpoint and Path(adv_checkpoint).exists():
            self.models["adv"] = load_model_checkpoint(adv_checkpoint, device=self.device)
            self.explainers["adv"] = GradCAM(self.models["adv"])

        self.transform = transforms.Compose([
            transforms.Resize((96, 96), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
        ])

    def preprocess(self, image_input: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Standardizes input to (1, 3, 96, 96) tensor in [0, 1]."""
        if isinstance(image_input, torch.Tensor):
            t = image_input.float()
            if t.ndim == 3:
                t = t.unsqueeze(0)
            if t.shape[2:] != (96, 96):
                t = F.interpolate(t, size=(96, 96), mode="bilinear", align_corners=False)
            return t.to(self.device).clamp(0.0, 1.0)

        if isinstance(image_input, np.ndarray):
            if image_input.dtype == np.uint8:
                image_input = Image.fromarray(image_input)
            else:
                image_input = Image.fromarray((image_input * 255).astype(np.uint8))

        if isinstance(image_input, Image.Image):
            img_rgb = image_input.convert("RGB")
            t = self.transform(img_rgb).unsqueeze(0)
            return t.to(self.device)

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def audit(
        self,
        image_input: Union[Image.Image, np.ndarray, torch.Tensor],
        temperature: float = 1.0,
        epsilon: float = 0.0,
        model_type: str = "base",
        include_ig: bool = True,
        ig_steps: int = 20,
    ) -> Dict[str, Any]:
        """
        Runs complete trust audit in < 0.5s:
        1. Applies FGSM attack with budget epsilon in [0, 1] pixel space.
        2. Computes calibrated probabilities via Temperature Scaling (logits / T).
        3. Computes Grad-CAM and optionally Integrated Gradients heatmaps.
        """
        t0 = time.time()
        model_key = "adv" if model_type.lower().startswith("adv") and "adv" in self.models else "base"
        model = self.models[model_key]
        explainer = self.explainers[model_key]

        x_clean = self.preprocess(image_input)

        # 1. FGSM Attack
        if epsilon > 0.0:
            with torch.no_grad():
                clean_pred = model(x_clean).argmax(dim=1)
            x_perturbed = fgsm_attack(model, x_clean, clean_pred, eps=epsilon)
        else:
            x_perturbed = x_clean.clone()

        # 2. Forward pass with Temperature Scaling
        temp_val = max(float(temperature), 1e-3)
        with torch.no_grad():
            raw_logits = model(x_perturbed)
            calibrated_logits = raw_logits / temp_val
            cal_probs = F.softmax(calibrated_logits, dim=1).squeeze(0)
            pred_idx = int(cal_probs.argmax().item())
            confidence = float(cal_probs[pred_idx].item())

        # 3. Grad-CAM saliency map
        cam_map = explainer(x_perturbed, target_classes=[pred_idx])[0]

        # 4. Integrated Gradients
        ig_map = None
        if include_ig:
            ig_maps, _ = integrated_gradients(model, x_perturbed, target_classes=[pred_idx], steps=ig_steps)
            ig_map = ig_maps[0]

        elapsed_ms = (time.time() - t0) * 1000.0

        # Perturbed image as HWC numpy array in [0, 1]
        perturbed_np = x_perturbed.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        return {
            "model_evaluated": model_key,
            "predicted_class": self.classes[pred_idx],
            "confidence": round(confidence, 4),
            "temperature_used": round(temp_val, 4),
            "epsilon_used": round(float(epsilon), 4),
            "class_probabilities": {
                c: round(float(cal_probs[i].item()), 4)
                for i, c in enumerate(self.classes)
            },
            "perturbed_image": perturbed_np,
            "grad_cam_map": cam_map,
            "integrated_gradients_map": ig_map,
            "latency_ms": round(elapsed_ms, 2),
            "sub_500ms_contract_met": bool(elapsed_ms < 500.0),
        }
