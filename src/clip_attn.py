"""
TrustVision - CLIP & Attention Module (src/clip_attn.py)
Owner: Member A (Team Lead)
Themes Covered:
  - Theme 4: Vision Transformers & Vision-Language Models (CLIP Zero-Shot & Linear Probe)
  - Theme 1: Attention & Transformers (Pure PyTorch Q, K, V Re-implementation & ViT Layer/Head Attention Extraction)
"""

import os
import json
import math
import hashlib
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from src.eval import compute_ece, compute_classification_metrics, TemperatureScaler


# Standard defect class names and readable variants
DEFAULT_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

READABLE_DEFECTS = {
    "crazing": "crazing cracks",
    "inclusion": "surface inclusions",
    "patches": "surface patches",
    "pitted_surface": "pitted surfaces",
    "rolled-in_scale": "rolled-in scale",
    "scratches": "scratches",
}

# Standard Prompt Templates for Theme 4 comparison (strictly grammatically natural)
PROMPT_TEMPLATES = {
    "P1_Minimal": "{defect}",
    "P2_Generic": "a photo of {defect}",
    "P3_DomainSpecific": "a photo of a steel surface with {defect}",
    "P4_DetailedTexture": "microscopic texture of steel showing {defect}",
    "P5_QualityInspection": "industrial quality inspection image of steel with {defect}",
}


def _ensure_parent_dir(path: str) -> None:
    """Creates the parent directory only when the path actually has one."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _patch_grid(num_patches: int) -> int:
    """Returns the spatial grid side length, asserting a square patch layout."""
    grid = int(math.isqrt(num_patches))
    if grid * grid != num_patches:
        raise ValueError(f"Attention patch count {num_patches} is not a perfect square.")
    return grid


def extract_tensor_features(feat_output: Any) -> torch.Tensor:
    """
    Bulletproof extraction of raw FloatTensor from Hugging Face model output objects.
    Protects against AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'
    when transformers version or return_dict returns an object instead of a bare tensor.
    """
    if isinstance(feat_output, torch.Tensor):
        return feat_output
    if hasattr(feat_output, "text_embeds") and feat_output.text_embeds is not None:
        return feat_output.text_embeds
    if hasattr(feat_output, "image_embeds") and feat_output.image_embeds is not None:
        return feat_output.image_embeds
    if hasattr(feat_output, "pooler_output") and feat_output.pooler_output is not None:
        return feat_output.pooler_output
    if hasattr(feat_output, "last_hidden_state") and feat_output.last_hidden_state is not None:
        return feat_output.last_hidden_state[:, 0, :]
    return feat_output[0]


# ==============================================================================
# Theme 1: Attention & Transformers (Pure PyTorch Q, K, V Re-Implementation)
# ==============================================================================

class ManualScaledDotProductAttention(nn.Module):
    """
    Pure PyTorch Re-implementation of Scaled Dot-Product Attention:
        Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    Exposes explicit Q, K, V projections and computes raw attention weights.
    """

    def __init__(self, embed_dim: int, d_k: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_k = d_k
        self.scale = 1.0 / math.sqrt(d_k)

        self.q_proj = nn.Linear(embed_dim, d_k, bias=True)
        self.k_proj = nn.Linear(embed_dim, d_k, bias=True)
        self.v_proj = nn.Linear(embed_dim, d_k, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input token representations of shape (batch_size, seq_len, embed_dim)
            return_attention_weights: If True, returns attention matrix (B, seq_len, seq_len)

        Returns:
            (context_vectors, attention_weights)
        """
        Q = self.q_proj(x)  # (B, seq_len, d_k)
        K = self.k_proj(x)  # (B, seq_len, d_k)
        V = self.v_proj(x)  # (B, seq_len, d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, seq_len, seq_len)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)  # (B, seq_len, d_k)

        if return_attention_weights:
            return context, attn_weights
        return context, None


def verify_attention_against_clip(
    clip_model: CLIPModel,
    layer_idx: int = 0,
    head_idx: int = 0,
    seq_len: int = 50,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Numerically verifies the pure PyTorch ManualScaledDotProductAttention
    against actual weights from Hugging Face CLIP's Vision Attention layer.
    """
    if device is None:
        device = next(clip_model.parameters()).device

    clip_layer = clip_model.vision_model.encoder.layers[layer_idx].self_attn
    embed_dim = clip_layer.embed_dim
    num_heads = clip_layer.num_heads
    head_dim = clip_layer.head_dim  # d_k = embed_dim // num_heads

    manual_attn = ManualScaledDotProductAttention(embed_dim=embed_dim, d_k=head_dim).to(device)

    with torch.no_grad():
        start_idx = head_idx * head_dim
        end_idx = start_idx + head_dim

        manual_attn.q_proj.weight.copy_(clip_layer.q_proj.weight[start_idx:end_idx, :])
        manual_attn.q_proj.bias.copy_(clip_layer.q_proj.bias[start_idx:end_idx])

        manual_attn.k_proj.weight.copy_(clip_layer.k_proj.weight[start_idx:end_idx, :])
        manual_attn.k_proj.bias.copy_(clip_layer.k_proj.bias[start_idx:end_idx])

        manual_attn.v_proj.weight.copy_(clip_layer.v_proj.weight[start_idx:end_idx, :])
        manual_attn.v_proj.bias.copy_(clip_layer.v_proj.bias[start_idx:end_idx])

    torch.manual_seed(42)
    x = torch.randn(2, seq_len, embed_dim, device=device)

    manual_attn.eval()
    with torch.no_grad():
        out_manual, weights_manual = manual_attn(x)

        # Black-box verification against Hugging Face CLIPAttention's real forward pass
        hf_outputs = clip_layer(x, output_attentions=True)
        hf_weights_head = hf_outputs[1][:, head_idx, :, :]

    diff_weights = (weights_manual - hf_weights_head).abs().max().item()
    is_verified = diff_weights < 1e-4

    return {
        "verified": is_verified,
        "max_numerical_discrepancy": float(diff_weights),
        "comparison_target": "Hugging Face CLIPAttention real forward pass (output_attentions=True)",
        "layer_idx": layer_idx,
        "head_idx": head_idx,
        "embed_dim": embed_dim,
        "head_dim": head_dim,
    }


def extract_cls_patch_attention(
    clip_model: CLIPModel,
    images: torch.Tensor,
    layer_idx: Optional[int] = -1,
    head_idx: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Extracts spatial attention maps from [CLS] to patch tokens by layer and head.
    If layer_idx is None, extracts across all 12 layers returning shape (B, 12, H_p, W_p).
    If layer_idx is an int, extracts that layer returning shape (B, H_p, W_p).
    """
    if device is None:
        device = next(clip_model.parameters()).device
    images = images.to(device)

    clip_model.eval()
    with torch.no_grad():
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        norm_images = (images - mean) / std

        vision_outputs = clip_model.vision_model(
            norm_images,
            output_attentions=True,
        )
        if vision_outputs.attentions is None:
            raise RuntimeError(
                "CLIP vision_model did not return attentions. "
                "Ensure the model was loaded with attn_implementation='eager'."
            )

        if layer_idx is None:
            # Extract across all transformer layers -> shape (B, num_layers, H_p, W_p)
            layer_maps = []
            for l_attn in vision_outputs.attentions:
                if head_idx is None:
                    cls_patches = l_attn[:, :, 0, 1:].mean(dim=1)
                else:
                    cls_patches = l_attn[:, head_idx, 0, 1:]
                num_patches = cls_patches.shape[-1]
                grid = _patch_grid(num_patches)
                layer_maps.append(cls_patches.view(-1, grid, grid))
            maps = torch.stack(layer_maps, dim=1)  # (B, 12, H_p, W_p)
        else:
            layer_attn = vision_outputs.attentions[layer_idx]  # (B, 12_heads, 50_tokens, 50_tokens)
            if head_idx is None:
                cls_to_patches = layer_attn[:, :, 0, 1:].mean(dim=1)  # (B, 49)
            else:
                cls_to_patches = layer_attn[:, head_idx, 0, 1:]  # (B, 49)
            num_patches = cls_to_patches.shape[-1]
            grid = _patch_grid(num_patches)
            maps = cls_to_patches.view(-1, grid, grid)  # (B, H_p, W_p)

    return maps.cpu().numpy()


# ==============================================================================
# Theme 4: Vision Transformers & VLM (CLIP Zero-Shot & Linear Probe)
# ==============================================================================

def load_clip(
    model_name: str = "openai/clip-vit-base-patch32",
    device: Optional[torch.device] = None,
) -> Tuple[CLIPModel, CLIPProcessor]:
    """
    Loads pre-trained CLIP model and processor.
    CRITICAL: Uses attn_implementation="eager" so output_attentions=True returns valid tensors on all platforms.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained(
        model_name,
        attn_implementation="eager",
    ).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def build_text_embeddings_for_prompts(
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    prompt_template: str,
    class_names: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Builds normalized text embeddings for each defect class given a prompt template."""
    if class_names is None:
        class_names = DEFAULT_CLASSES
    if device is None:
        device = next(clip_model.parameters()).device

    formatted_prompts = []
    for cls_name in class_names:
        defect_str = READABLE_DEFECTS.get(cls_name, cls_name)
        if "{defect}" in prompt_template:
            p = prompt_template.replace("{defect}", defect_str)
        else:
            p = f"{prompt_template} {defect_str}".strip()
        formatted_prompts.append(p)

    inputs = processor(text=formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        raw_text_output = clip_model.get_text_features(**inputs)
        text_features = extract_tensor_features(raw_text_output)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


def evaluate_clip_zero_shot(
    clip_model: CLIPModel,
    dataloader: DataLoader,
    text_features: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Evaluates zero-shot accuracy and ECE on dataloader."""
    if device is None:
        device = next(clip_model.parameters()).device

    clip_model.eval()
    total_samples = 0
    correct = 0
    all_probs = []
    all_targets = []

    logit_scale = clip_model.logit_scale.exp().item()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            total_samples += images.size(0)

            norm_images = (images - mean) / std
            raw_img_output = clip_model.get_image_features(norm_images)
            image_features = extract_tensor_features(raw_img_output)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = logit_scale * torch.matmul(image_features, text_features.t())
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            correct += (preds == targets).sum().item()
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

    all_probs_cat = torch.cat(all_probs, dim=0)
    all_targets_cat = torch.cat(all_targets, dim=0)

    metrics = compute_classification_metrics(all_probs_cat, all_targets_cat, n_bins=10)

    return metrics


def evaluate_clip_zero_shot_from_embeddings(
    image_features: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    text_features: torch.Tensor,
    logit_scale: float = 100.0,
) -> Dict[str, Any]:
    """Fast vectorized zero-shot evaluation directly from pre-extracted image features."""
    if isinstance(image_features, np.ndarray):
        image_features = torch.from_numpy(image_features).float()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).long()

    device = text_features.device
    image_features = image_features.to(device)
    labels = labels.to(device)

    logits = logit_scale * torch.matmul(image_features, text_features.t())
    probs = F.softmax(logits, dim=-1)
    metrics = compute_classification_metrics(probs, labels, n_bins=10)
    return metrics


def extract_all_clip_embeddings(
    clip_model: CLIPModel,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts normalized 512-dimensional CLIP image embeddings."""
    if device is None:
        device = next(clip_model.parameters()).device

    clip_model.eval()
    features_list = []
    labels_list = []

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            norm_images = (images - mean) / std

            raw_feats = clip_model.get_image_features(norm_images)
            feats = extract_tensor_features(raw_feats)
            feats = feats / feats.norm(dim=-1, keepdim=True)

            features_list.append(feats.cpu().numpy())
            labels_list.append(targets.cpu().numpy())

    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)


class LinearProbe(nn.Module):
    """Linear classifier trained on frozen CLIP embeddings."""

    def __init__(self, in_features: int = 512, num_classes: int = 6):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    cal_features: np.ndarray,
    cal_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    num_classes: int = 6,
    epochs: int = 300,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
    seed: int = 42,
    checkpoint_path: Optional[str] = "checkpoints/clip_linear_probe.pt",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Trains linear probe until loss converges, fits temperature scaling on cal split,
    and returns comprehensive metrics (acc, macro_f1, ece, nll, brier, confusion_matrix)
    both BEFORE and AFTER temperature calibration.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducible probe initialisation (matches the committed notebook artifact)
    torch.manual_seed(seed)
    np.random.seed(seed)
    probe = LinearProbe(in_features=train_features.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    X_train = torch.tensor(train_features, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_labels, dtype=torch.long).to(device)
    X_cal = torch.tensor(cal_features, dtype=torch.float32).to(device)
    y_cal = torch.tensor(cal_labels, dtype=torch.long).to(device)
    X_test = torch.tensor(test_features, dtype=torch.float32).to(device)
    y_test = torch.tensor(test_labels, dtype=torch.long).to(device)

    loss_history = []
    probe.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        logits = probe(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if (ep + 1) % 25 == 0 or ep == epochs - 1:
            print(f"  [Probe Epoch {ep+1:3d}/{epochs}] CrossEntropyLoss = {loss.item():.4f}")

    probe.eval()
    with torch.no_grad():
        cal_logits = probe(X_cal)
        test_logits = probe(X_test)
        test_probs_uncal = F.softmax(test_logits, dim=-1)

    # 1. Uncalibrated test metrics
    metrics_uncal = compute_classification_metrics(test_probs_uncal, y_test, n_bins=10)

    # 2. Temperature scaling calibration on CAL split (180 samples)
    scaler = TemperatureScaler(initial_temp=1.0).to(device)
    T_opt = scaler.fit(cal_logits, y_cal, lr=0.1, max_iter=200)

    # Sanity check: calibration on CAL split must decrease or maintain NLL
    uncal_cal_nll = criterion(cal_logits, y_cal).item()
    calib_cal_nll = criterion(cal_logits / T_opt, y_cal).item()
    assert calib_cal_nll <= uncal_cal_nll + 1e-5, (
        f"Calibration sanity failed: cal NLL increased from {uncal_cal_nll} to {calib_cal_nll}"
    )

    # 3. Calibrated test metrics
    with torch.no_grad():
        test_probs_calib = F.softmax(scaler(test_logits), dim=-1)
    metrics_calib = compute_classification_metrics(test_probs_calib, y_test, n_bins=10)

    # 4. Selective-classification threshold: 5th percentile of calibrated CAL confidence
    #    (i.e. the threshold that retains 95% coverage on the calibration split).
    with torch.no_grad():
        cal_probs_calib = F.softmax(scaler(cal_logits), dim=-1)
    tau95 = float(torch.quantile(cal_probs_calib.max(dim=1).values, q=0.05).item())

    if checkpoint_path:
        _ensure_parent_dir(checkpoint_path)
        torch.save({
            "linear_probe_state_dict": probe.state_dict(),
            "optimal_temperature": float(T_opt),
            "in_features": train_features.shape[1],
            "num_classes": num_classes,
        }, checkpoint_path)
        print(f"[Deliverable] Saved Linear Probe + Calibrated Temperature to {checkpoint_path}")

    return {
        "uncalibrated": metrics_uncal,
        "calibrated": metrics_calib,
        "optimal_temperature": round(float(T_opt), 4),
        "tau95": round(tau95, 4),
        "loss_curve": [round(l, 4) for l in loss_history[::10]],
        "probe": probe,
        "temperature_scaler": scaler,
    }


# ==============================================================================
# Pipeline Execution & Deliverables Generation (S0)
# ==============================================================================

def evaluate_probe_robustness(
    clip_model: CLIPModel,
    dataloader: DataLoader,
    probe: nn.Module,
    temperature: float,
    device: Optional[torch.device] = None,
    eps: float = 0.03,
) -> float:
    """
    Computes FGSM robust accuracy (eps=0.03) for a linear probe trained on frozen
    CLIP embeddings. The attack differentiates the calibrated probe cross-entropy
    through the frozen image encoder back into the [0, 1] pixel space.
    """
    if device is None:
        device = next(clip_model.parameters()).device

    clip_model.eval()
    probe.eval()
    probe.to(device)

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    temp = max(float(temperature), 1e-3)

    # Freeze parameters so backward only retains gradients w.r.t. the input pixels.
    clip_grad_state = {name: p.requires_grad for name, p in clip_model.named_parameters()}
    probe_grad_state = {name: p.requires_grad for name, p in probe.named_parameters()}
    for p in clip_model.parameters():
        p.requires_grad_(False)
    for p in probe.parameters():
        p.requires_grad_(False)

    correct = 0
    total = 0
    try:
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            adv_images = images.clone().detach().requires_grad_(True)
            norm_images = (adv_images - mean) / std
            features = extract_tensor_features(clip_model.get_image_features(norm_images))
            features = features / features.norm(dim=-1, keepdim=True)
            loss = F.cross_entropy(probe(features) / temp, targets)
            loss.backward()

            with torch.no_grad():
                adv_images = torch.clamp(adv_images + eps * adv_images.grad.sign(), 0.0, 1.0)
                adv_features = extract_tensor_features(
                    clip_model.get_image_features((adv_images - mean) / std)
                )
                adv_features = adv_features / adv_features.norm(dim=-1, keepdim=True)
                preds = (probe(adv_features) / temp).argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.numel()
    finally:
        for name, p in clip_model.named_parameters():
            p.requires_grad_(clip_grad_state[name])
        for name, p in probe.named_parameters():
            p.requires_grad_(probe_grad_state[name])

    return correct / total if total > 0 else 0.0

def run_s0_pipeline(
    data_dir: Optional[str] = None,
    output_json: str = "results/s0_clip.json",
    output_attn_npy: str = "results/clip_attn_test.npy",
    linear_probe_ckpt: str = "checkpoints/clip_linear_probe.pt",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Executes the complete S0 pipeline for Member A on real NEU-CLS data."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[TrustVision S0] Loading CLIP (openai/clip-vit-base-patch32 with eager attention)...")
    clip_model, processor = load_clip(device=device)

    # 1. Verify Q, K, V against Hugging Face CLIP real forward pass
    print("[Theme 1] Verifying pure PyTorch Q, K, V re-implementation against Hugging Face forward pass...")
    qkv_verification = verify_attention_against_clip(clip_model, layer_idx=0, head_idx=0, device=device)
    print(f"  -> Q, K, V Verified: {qkv_verification['verified']} (max error: {qkv_verification['max_numerical_discrepancy']:.2e})")

    # 2. Prepare DataLoaders
    from src.data import get_dataloaders
    train_loader, cal_loader, test_loader, test_files = get_dataloaders(
        root_dir=data_dir,
        batch_size=32,
        seed=42,
        img_size=224,
    )
    print(f"  -> Data Splits Loaded: Train={len(train_loader.dataset)}, Cal={len(cal_loader.dataset)}, Test={len(test_loader.dataset)}")

    # 3. Extract frozen CLIP embeddings for Train, Cal, and Test splits (done once)
    print("[Theme 4] Extracting frozen CLIP embeddings for Train, Cal, and Test splits...")
    train_feats, train_labels = extract_all_clip_embeddings(clip_model, train_loader, device=device)
    cal_feats, cal_labels = extract_all_clip_embeddings(clip_model, cal_loader, device=device)
    test_feats, test_labels = extract_all_clip_embeddings(clip_model, test_loader, device=device)
    logit_scale = clip_model.logit_scale.exp().item()

    # 4. Evaluate Zero-Shot: Prompt selection on CAL split, final evaluation on TEST split
    print("[Theme 4] Zero-Shot: Selecting best prompt template on CAL split (180 samples)...")
    cal_prompt_results = {}
    for prompt_key, template in PROMPT_TEMPLATES.items():
        text_feats = build_text_embeddings_for_prompts(
            clip_model, processor, template, class_names=DEFAULT_CLASSES, device=device
        )
        res_cal = evaluate_clip_zero_shot_from_embeddings(cal_feats, cal_labels, text_feats, logit_scale=logit_scale)
        cal_prompt_results[prompt_key] = res_cal
        print(f"  [CAL] '{prompt_key}': Acc={res_cal['accuracy']*100:.2f}%, F1={res_cal['macro_f1']:.4f}, ECE={res_cal['ece']:.4f}")

    best_prompt = max(cal_prompt_results, key=lambda k: (cal_prompt_results[k]["accuracy"], -cal_prompt_results[k]["ece"]))
    print(f"  -> Best prompt selected on CAL: '{best_prompt}' (Acc={cal_prompt_results[best_prompt]['accuracy']*100:.2f}%)")

    print("[Theme 4] Zero-Shot: Evaluating all prompts on TEST split (360 samples)...")
    test_prompt_results = {}
    for prompt_key, template in PROMPT_TEMPLATES.items():
        text_feats = build_text_embeddings_for_prompts(
            clip_model, processor, template, class_names=DEFAULT_CLASSES, device=device
        )
        res_test = evaluate_clip_zero_shot_from_embeddings(test_feats, test_labels, text_feats, logit_scale=logit_scale)
        test_prompt_results[prompt_key] = res_test
        print(f"  [TEST] '{prompt_key}': Acc={res_test['accuracy']*100:.2f}%, F1={res_test['macro_f1']:.4f}, ECE={res_test['ece']:.4f}")

    selected_test_metrics = test_prompt_results[best_prompt]

    # 5. Linear Probe: Train to convergence & calibrate temperature on CAL
    print("[Theme 4] Training Linear Probe and fitting Temperature Scaling on CAL split...")
    probe_results = train_linear_probe(
        train_features=train_feats,
        train_labels=train_labels,
        cal_features=cal_feats,
        cal_labels=cal_labels,
        test_features=test_feats,
        test_labels=test_labels,
        num_classes=len(DEFAULT_CLASSES),
        epochs=300,
        checkpoint_path=linear_probe_ckpt,
        device=device,
    )
    print(f"  -> Linear Probe (Uncalibrated): Acc={probe_results['uncalibrated']['accuracy']*100:.2f}%, F1={probe_results['uncalibrated']['macro_f1']:.4f}, ECE={probe_results['uncalibrated']['ece']:.4f}")
    print(f"  -> Linear Probe (Calibrated T={probe_results['optimal_temperature']}): Acc={probe_results['calibrated']['accuracy']*100:.2f}%, F1={probe_results['calibrated']['macro_f1']:.4f}, ECE={probe_results['calibrated']['ece']:.4f}")

    # 5b. FGSM robustness (eps=0.03) of the calibrated linear probe
    print("[Theme 4] Evaluating calibrated linear probe under FGSM eps=0.03...")
    rob_acc_003 = evaluate_probe_robustness(
        clip_model=clip_model,
        dataloader=test_loader,
        probe=probe_results["probe"],
        temperature=probe_results["optimal_temperature"],
        device=device,
        eps=0.03,
    )
    print(f"  -> Calibrated Linear Probe Robust Accuracy (FGSM eps=0.03): {rob_acc_003*100:.2f}%")

    # 6. Extract Multi-Layer ViT Patch Attention on Test Set (All 12 layers)
    print("[Theme 1] Extracting multi-layer ViT patch attention maps (all 12 layers) on test set...")
    all_attention_maps = []
    for images, _ in test_loader:
        maps = extract_cls_patch_attention(clip_model, images, layer_idx=None, head_idx=None, device=device)
        all_attention_maps.append(maps)

    test_attention_maps = np.concatenate(all_attention_maps, axis=0)  # (360, 12, 7, 7)
    print(f"  -> Extracted reference attention maps shape: {test_attention_maps.shape}")

    # 7. Save Deliverables & Hash Verification
    _ensure_parent_dir(output_json)
    _ensure_parent_dir(output_attn_npy)

    np.save(output_attn_npy, test_attention_maps)
    print(f"[Deliverable] Saved reference attention maps to {output_attn_npy}")

    splits_path = Path("configs/splits.json")
    splits_sha256 = None
    if splits_path.exists():
        with open(splits_path, "rb") as f:
            splits_sha256 = hashlib.sha256(f.read()).hexdigest()

    s0_summary = {
        "stage": "S0_CLIP",
        "model": "openai/clip-vit-base-patch32",
        "splits_sha256": splits_sha256,
        "theme1_mechanistic_attention": {
            "max_numerical_discrepancy": qkv_verification["max_numerical_discrepancy"],
            "comparison_target": qkv_verification["comparison_target"],
            "verified": qkv_verification["verified"],
        },
        "theme4_zero_shot": {
            "selection_protocol": "Selected on cal split (180 images), evaluated on test split (360 images)",
            "selected_prompt": best_prompt,
            "selected_test_metrics": selected_test_metrics,
            "cal_ablation": cal_prompt_results,
            "test_ablation": test_prompt_results,
        },
        "theme4_linear_probe": {
            "checkpoint": linear_probe_ckpt,
            "uncalibrated": probe_results["uncalibrated"],
            "calibrated": probe_results["calibrated"],
            "optimal_temperature": probe_results["optimal_temperature"],
            "rob_acc_003": round(rob_acc_003, 4),
            "tau95": probe_results["tau95"],
            "loss_curve": probe_results["loss_curve"],
        },
        "reference_attention_maps": {
            "file": output_attn_npy,
            "shape": list(test_attention_maps.shape),
            "description": "Multi-layer ViT CLS-to-patch attention maps averaged over 12 heads for all 12 layers",
            "status": "Reference maps (not ground truth)",
        },
        "cross_effects_input": {
            "clip_zero_shot_acc": selected_test_metrics["accuracy"],
            "clip_zero_shot_macro_f1": selected_test_metrics["macro_f1"],
            "clip_zero_shot_ece": selected_test_metrics["ece"],
            "clip_linear_probe_acc": probe_results["calibrated"]["accuracy"],
            "clip_linear_probe_macro_f1": probe_results["calibrated"]["macro_f1"],
            "clip_linear_probe_ece_uncalibrated": probe_results["uncalibrated"]["ece"],
            "clip_linear_probe_ece_calibrated": probe_results["calibrated"]["ece"],
            "clip_linear_probe_rob_acc_003": round(rob_acc_003, 4),
            "clip_linear_probe_temperature": probe_results["optimal_temperature"],
            "selective_threshold_tau95": probe_results["tau95"],
            "attention_maps_file": output_attn_npy,
            "test_files": test_files,
            "splits_sha256": splits_sha256,
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(s0_summary, f, indent=2)
    print(f"[Deliverable] Saved S0 results to {output_json}")

    return s0_summary


if __name__ == "__main__":
    run_s0_pipeline()

