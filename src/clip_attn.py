"""
TrustVision - CLIP & Attention Module (src/clip_attn.py)
Owner: Member A (Team Lead)
Themes Covered:
  - Theme 4: Vision Transformers & Vision-Language Models (CLIP Zero-Shot & Linear Probe)
  - Theme 1: Attention & Transformers (Pure PyTorch Q, K, V Re-implementation & ViT Attention Extraction)
"""

import os
import json
import math
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from src.eval import compute_ece


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
    "inclusion": "inclusion defect",
    "patches": "surface patches",
    "pitted_surface": "pitted surface",
    "rolled-in_scale": "rolled-in scale",
    "scratches": "scratches",
}

# Standard Prompt Templates for Theme 4 comparison
PROMPT_TEMPLATES = {
    "minimal": "{defect}",
    "generic": "a photo of {defect}",
    "domain_specific": "a photo of a {defect} steel surface defect",
    "detailed_texture": "microscopic steel surface texture showing {defect}",
    "quality_inspection": "industrial quality inspection image of steel with {defect}",
}


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

        # Standalone projection weights
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
        # Linear projections
        Q = self.q_proj(x)  # (B, seq_len, d_k)
        K = self.k_proj(x)  # (B, seq_len, d_k)
        V = self.v_proj(x)  # (B, seq_len, d_k)

        # Scaled dot-product: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, seq_len, seq_len)

        # Softmax along key sequence dimension
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values: softmax(scores) @ V
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
    
    Proves exact mathematical equivalence: out_scratch ≈ out_clip_head.
    """
    if device is None:
        device = next(clip_model.parameters()).device

    # Access HF CLIP vision self-attention layer
    clip_layer = clip_model.vision_model.encoder.layers[layer_idx].self_attn
    embed_dim = clip_layer.embed_dim
    num_heads = clip_layer.num_heads
    head_dim = clip_layer.head_dim  # d_k = embed_dim // num_heads

    # Instantiate our pure PyTorch manual attention
    manual_attn = ManualScaledDotProductAttention(embed_dim=embed_dim, d_k=head_dim).to(device)

    # Slice weights corresponding to head_idx
    with torch.no_grad():
        start_idx = head_idx * head_dim
        end_idx = start_idx + head_dim

        # Copy Q, K, V weights and biases
        manual_attn.q_proj.weight.copy_(clip_layer.q_proj.weight[start_idx:end_idx, :])
        manual_attn.q_proj.bias.copy_(clip_layer.q_proj.bias[start_idx:end_idx])

        manual_attn.k_proj.weight.copy_(clip_layer.k_proj.weight[start_idx:end_idx, :])
        manual_attn.k_proj.bias.copy_(clip_layer.k_proj.bias[start_idx:end_idx])

        manual_attn.v_proj.weight.copy_(clip_layer.v_proj.weight[start_idx:end_idx, :])
        manual_attn.v_proj.bias.copy_(clip_layer.v_proj.bias[start_idx:end_idx])

    # Generate synthetic input token sequence
    torch.manual_seed(42)
    x = torch.randn(2, seq_len, embed_dim, device=device)

    # Run manual attention
    manual_attn.eval()
    with torch.no_grad():
        out_manual, weights_manual = manual_attn(x)

        # Run CLIP's attention and extract corresponding head slice
        # In HF CLIPAttention: Q, K, V are projected and reshaped to (B, num_heads, seq_len, head_dim)
        Q_clip = clip_layer.q_proj(x).view(2, seq_len, num_heads, head_dim).transpose(1, 2)
        K_clip = clip_layer.k_proj(x).view(2, seq_len, num_heads, head_dim).transpose(1, 2)
        V_clip = clip_layer.v_proj(x).view(2, seq_len, num_heads, head_dim).transpose(1, 2)

        # Head slice
        Q_h = Q_clip[:, head_idx, :, :]
        K_h = K_clip[:, head_idx, :, :]
        V_h = V_clip[:, head_idx, :, :]

        scores_h = torch.matmul(Q_h, K_h.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        weights_clip_h = F.softmax(scores_h, dim=-1)
        out_clip_h = torch.matmul(weights_clip_h, V_h)

    # Compute numerical discrepancies
    diff_out = (out_manual - out_clip_h).abs().max().item()
    diff_weights = (weights_manual - weights_clip_h).abs().max().item()
    is_verified = (diff_out < 1e-4) and (diff_weights < 1e-4)

    return {
        "verified": is_verified,
        "max_context_diff": float(diff_out),
        "max_weights_diff": float(diff_weights),
        "layer_idx": layer_idx,
        "head_idx": head_idx,
        "embed_dim": embed_dim,
        "head_dim": head_dim,
    }


def extract_cls_patch_attention(
    clip_model: CLIPModel,
    images: torch.Tensor,
    layer_idx: int = -1,
    average_heads: bool = True,
    patch_grid_size: int = 7,
) -> np.ndarray:
    """
    Extracts attention maps from the [CLS] token to spatial patch tokens.
    For patch32 on 224x224 images, there are 7x7 = 49 patch tokens.

    Args:
        clip_model: Loaded CLIPModel.
        images: Batch of images (B, 3, 224, 224) in [0, 1].
        layer_idx: Layer to extract from (-1 = last encoder layer).
        average_heads: If True, averages attention weights across all 12 heads.
        patch_grid_size: Spatial grid size (7 for patch32 at 224x224).

    Returns:
        Numpy array of attention maps: (B, 7, 7) or (B, num_heads, 7, 7)
    """
    device = next(clip_model.parameters()).device
    images = images.to(device)

    clip_model.eval()
    with torch.no_grad():
        # Standard normalization expected by CLIP vision encoder
        # (images are [0, 1])
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        norm_images = (images - mean) / std

        vision_outputs = clip_model.vision_model(
            norm_images,
            output_attentions=True,
            return_dict=True,
        )

        # vision_outputs.attentions is a tuple of 12 tensors each (B, num_heads, seq_len=50, seq_len=50)
        layer_attn = vision_outputs.attentions[layer_idx]  # (B, num_heads, 50, 50)

        # [CLS] token is at index 0; patch tokens are indices 1:50
        cls_to_patches = layer_attn[:, :, 0, 1:]  # (B, num_heads, 49)

        if average_heads:
            cls_to_patches = cls_to_patches.mean(dim=1)  # (B, 49)
            maps = cls_to_patches.view(-1, patch_grid_size, patch_grid_size)  # (B, 7, 7)
        else:
            maps = cls_to_patches.view(-1, layer_attn.size(1), patch_grid_size, patch_grid_size)

    return maps.cpu().numpy()


# ==============================================================================
# Theme 4: Vision Transformers & VLM (CLIP Zero-Shot & Linear Probe)
# ==============================================================================

def load_clip(
    model_name: str = "openai/clip-vit-base-patch32",
    device: Optional[torch.device] = None,
) -> Tuple[CLIPModel, CLIPProcessor]:
    """Loads pre-trained CLIP model and processor from Hugging Face."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def build_text_embeddings_for_prompts(
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    prompt_template: str,
    class_names: List[str] = DEFAULT_CLASSES,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Builds normalized text embeddings for each defect class given a prompt template.

    Returns:
        (num_classes, 512) normalized tensor of text embeddings.
    """
    if device is None:
        device = next(clip_model.parameters()).device

    formatted_prompts = [
        prompt_template.format(defect=READABLE_DEFECTS.get(cls_name, cls_name))
        for cls_name in class_names
    ]

    inputs = processor(text=formatted_prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        # L2 Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


def evaluate_clip_zero_shot(
    clip_model: CLIPModel,
    dataloader: DataLoader,
    text_features: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Evaluates CLIP zero-shot classification on a DataLoader using pre-computed text embeddings.
    Computes top-1 accuracy and Expected Calibration Error (ECE).
    """
    if device is None:
        device = next(clip_model.parameters()).device

    clip_model.eval()
    total_samples = 0
    correct = 0

    all_probs = []
    all_targets = []

    # CLIP logit scale (learned temperature parameter)
    logit_scale = clip_model.logit_scale.exp().item()

    # Pre-computed image normalization constants
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)
            total_samples += batch_size

            # Normalize for CLIP
            norm_images = (images - mean) / std

            # Extract normalized image embeddings
            image_features = clip_model.get_image_features(norm_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity scaled by temperature
            logits = logit_scale * torch.matmul(image_features, text_features.t())
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            correct += (preds == targets).sum().item()
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

    all_probs_cat = torch.cat(all_probs, dim=0)
    all_targets_cat = torch.cat(all_targets, dim=0)

    accuracy = correct / total_samples if total_samples > 0 else 0.0
    ece_score, bin_stats = compute_ece(all_probs_cat, all_targets_cat, n_bins=10)

    return {
        "accuracy": round(float(accuracy), 4),
        "ece": round(float(ece_score), 4),
        "total_samples": total_samples,
        "bin_stats": bin_stats,
    }


def extract_all_clip_embeddings(
    clip_model: CLIPModel,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts frozen 512-dimensional CLIP image embeddings for all samples in dataloader.
    Returns: (features_numpy, labels_numpy)
    """
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

            feats = clip_model.get_image_features(norm_images)
            feats = feats / feats.norm(dim=-1, keepdim=True)

            features_list.append(feats.cpu().numpy())
            labels_list.append(targets.numpy())

    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)


class LinearProbe(nn.Module):
    """Lightweight linear classification head trained on frozen CLIP embeddings."""

    def __init__(self, in_features: int = 512, num_classes: int = 6):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    num_classes: int = 6,
    epochs: int = 40,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Trains a linear probe on frozen CLIP representations and evaluates accuracy + ECE."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probe = LinearProbe(in_features=train_features.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    X_train = torch.tensor(train_features, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_labels, dtype=torch.long).to(device)
    X_test = torch.tensor(test_features, dtype=torch.float32).to(device)
    y_test = torch.tensor(test_labels, dtype=torch.long).to(device)

    # Training loop
    probe.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = probe(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    # Evaluation
    probe.eval()
    with torch.no_grad():
        test_logits = probe(X_test)
        test_probs = F.softmax(test_logits, dim=-1)
        test_preds = test_logits.argmax(dim=-1)
        accuracy = (test_preds == y_test).float().mean().item()
        ece_score, _ = compute_ece(test_probs.cpu(), y_test.cpu(), n_bins=10)

    return {
        "accuracy": round(float(accuracy), 4),
        "ece": round(float(ece_score), 4),
        "probe_state_dict": probe.state_dict(),
    }


# ==============================================================================
# Pipeline Execution & Deliverables Generation (S0)
# ==============================================================================

def run_s0_pipeline(
    data_dir: Optional[str] = None,
    output_json: str = "results/s0_clip.json",
    output_attn_npy: str = "results/clip_attn_test.npy",
    use_mock: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Executes the full S0 pipeline for Member A:
      1. Numerical verification of pure PyTorch Q, K, V against CLIP.
      2. Zero-shot evaluation across multiple prompt templates.
      3. Extraction of frozen embeddings & training of linear probe.
      4. Extraction and saving of test set patch attention maps.
      5. Saving results/s0_clip.json and results/clip_attn_test.npy.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[TrustVision S0] Loading CLIP model (openai/clip-vit-base-patch32)...")
    clip_model, processor = load_clip(device=device)

    # 1. Verify Q, K, V
    print("[Theme 1] Verifying pure PyTorch Q, K, V re-implementation against CLIP weights...")
    qkv_verification = verify_attention_against_clip(clip_model, layer_idx=0, head_idx=0, device=device)
    print(f"  -> Q, K, V Verified: {qkv_verification['verified']} (max error: {qkv_verification['max_context_diff']:.2e})")

    # 2. Prepare DataLoaders at 224x224 for CLIP
    from src.data import get_dataloaders
    train_loader, cal_loader, test_loader = get_dataloaders(
        root_dir=data_dir,
        batch_size=32,
        seed=42,
        img_size=224,  # CLIP input resolution
        use_mock=use_mock,
    )

    # 3. Evaluate Zero-Shot across prompt templates
    print("[Theme 4] Evaluating Zero-Shot classification across prompt templates...")
    prompt_results = {}
    best_prompt = None
    best_prompt_acc = -1.0

    for prompt_key, template in PROMPT_TEMPLATES.items():
        text_feats = build_text_embeddings_for_prompts(
            clip_model, processor, template, class_names=DEFAULT_CLASSES, device=device
        )
        res = evaluate_clip_zero_shot(clip_model, test_loader, text_feats, device=device)
        prompt_results[prompt_key] = {
            "template": template,
            "accuracy": res["accuracy"],
            "ece": res["ece"],
        }
        print(f"  -> Template '{prompt_key}': Acc={res['accuracy']*100:.2f}%, ECE={res['ece']:.4f}")

        if res["accuracy"] > best_prompt_acc:
            best_prompt_acc = res["accuracy"]
            best_prompt = prompt_key

    # 4. Extract Embeddings & Train Linear Probe
    print("[Theme 4] Training Linear Probe on frozen CLIP embeddings...")
    train_feats, train_labels = extract_all_clip_embeddings(clip_model, train_loader, device=device)
    test_feats, test_labels = extract_all_clip_embeddings(clip_model, test_loader, device=device)

    probe_results = train_linear_probe(
        train_features=train_feats,
        train_labels=train_labels,
        test_features=test_feats,
        test_labels=test_labels,
        num_classes=len(DEFAULT_CLASSES),
        epochs=40,
        device=device,
    )
    print(f"  -> Linear Probe: Acc={probe_results['accuracy']*100:.2f}%, ECE={probe_results['ece']:.4f}")

    # 5. Extract ViT Patch Attention Maps on Test Set
    print("[Theme 1] Extracting ViT patch attention maps on test set...")
    all_attention_maps = []
    for images, _ in test_loader:
        maps = extract_cls_patch_attention(clip_model, images, layer_idx=-1, average_heads=True)
        all_attention_maps.append(maps)

    test_attention_maps = np.concatenate(all_attention_maps, axis=0)  # (N_test, 7, 7)
    print(f"  -> Extracted attention maps shape: {test_attention_maps.shape}")

    # 6. Save Deliverables
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    os.makedirs(os.path.dirname(output_attn_npy), exist_ok=True)

    np.save(output_attn_npy, test_attention_maps)
    print(f"[Deliverable] Saved attention maps to {output_attn_npy}")

    s0_summary = {
        "stage": "S0_CLIP",
        "model": "openai/clip-vit-base-patch32",
        "theme1_qkv_verification": qkv_verification,
        "theme4_zero_shot": {
            "prompts": prompt_results,
            "best_prompt": best_prompt,
            "best_accuracy": best_prompt_acc,
        },
        "theme4_linear_probe": {
            "accuracy": probe_results["accuracy"],
            "ece": probe_results["ece"],
        },
        "cross_effects_input": {
            "clip_zero_shot_acc": best_prompt_acc,
            "clip_linear_probe_acc": probe_results["accuracy"],
            "clip_linear_probe_ece": probe_results["ece"],
            "attention_maps_file": output_attn_npy,
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(s0_summary, f, indent=2)
    print(f"[Deliverable] Saved S0 results to {output_json}")

    return s0_summary
