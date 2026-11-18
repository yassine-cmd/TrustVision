"""
TrustVision - Member C: Continual Learning with EWC (S4)
========================================================
Single-head class-incremental learning on NEU-CLS:

    Task 1 : crazing, inclusion, patches              (classes 0, 1, 2)
    Task 2 : pitted_surface, rolled-in_scale, scratches (classes 3, 4, 5)

The classifier starts with 3 outputs and is expanded to 6 before Task 2.
No task identifier is provided at inference time. Three strategy families are
compared:

* ``naive``  : plain fine-tuning on Task 2 (catastrophic forgetting baseline)
* ``ewc``    : Elastic Weight Consolidation with a (per-tensor normalized)
  diagonal Fisher matrix
* ``replay`` : fine-tuning with a small, class-balanced rehearsal buffer
* ``ewc_replay`` : best EWC lambda combined with the rehearsal buffer

References
----------
Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks",
PNAS 2017.
"""

from __future__ import annotations

import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import get_dataloaders
from src.eval import compute_intra_pipeline_stability, evaluate_all
from src.member_c_utils import calibrated_summary, fit_temperature_and_evaluate
from src.model import TrustVisionCNN
from src.xai_cal_adv import GradCAM


# ---------------------------------------------------------------------------
# Fisher information & EWC penalty
# ---------------------------------------------------------------------------

def compute_fisher_diagonal(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Estimates the diagonal Fisher information matrix from the Task-1 training
    set using the squared gradients of the cross-entropy loss.

    Returns a dict ``parameter_name -> tensor`` with non-negative entries.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    fisher: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    n_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        model.zero_grad(set_to_none=True)
        logits = model(images)
        # Mean loss * batch size approximates the sum of per-sample losses.
        loss = F.cross_entropy(logits, labels, reduction="mean") * batch_size
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None and name in fisher:
                fisher[name] += param.grad.detach() ** 2

        n_samples += batch_size

    if n_samples == 0:
        raise ValueError("Cannot estimate Fisher information from an empty loader.")

    for name in fisher:
        fisher[name] /= float(n_samples)

    model.zero_grad(set_to_none=True)
    return fisher


def ewc_penalty(
    model: nn.Module,
    fisher: Dict[str, torch.Tensor],
    old_params: Dict[str, torch.Tensor],
    ewc_lambda: float,
) -> torch.Tensor:
    """Computes (lambda / 2) * sum_i F_i * (theta_i - theta_i*)^2."""
    if ewc_lambda <= 0:
        return torch.zeros((), device=next(model.parameters()).device)

    penalty = torch.zeros((), device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if name in fisher and name in old_params:
            f = fisher[name].to(param.device)
            old = old_params[name].to(param.device)
            penalty = penalty + (f * (param - old) ** 2).sum()
    return 0.5 * float(ewc_lambda) * penalty


def normalize_fisher_per_tensor(
    fisher: Dict[str, torch.Tensor],
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Rescales every Fisher tensor to mean 1.

    A saturated classifier produces gradients that differ by several orders of
    magnitude between shallow and deep layers; a single global lambda then
    leaves the deep layers effectively unconstrained. Per-tensor normalization
    keeps the EWC objective unchanged while making lambda comparable across
    layers.
    """
    return {
        name: tensor / tensor.mean().clamp_min(eps)
        for name, tensor in fisher.items()
    }


# ---------------------------------------------------------------------------
# Head expansion and EWC state padding
# ---------------------------------------------------------------------------

def expand_classifier(model: TrustVisionCNN, new_num_classes: int = 6) -> TrustVisionCNN:
    """
    Expands ``model.fc`` from its current number of outputs to
    ``new_num_classes`` in place, preserving the old rows and initializing the
    new rows with Kaiming/Xavier initialization.
    """
    old_fc = model.fc
    old_classes = old_fc.out_features
    if new_num_classes == old_classes:
        return model
    if new_num_classes < old_classes:
        raise ValueError("new_num_classes must be >= current number of classes")

    new_fc = nn.Linear(old_fc.in_features, new_num_classes, bias=old_fc.bias is not None)
    new_fc = new_fc.to(device=old_fc.weight.device, dtype=old_fc.weight.dtype)

    with torch.no_grad():
        new_fc.weight[:old_classes] = old_fc.weight
        if old_fc.bias is not None and new_fc.bias is not None:
            new_fc.bias[:old_classes] = old_fc.bias
        nn.init.kaiming_uniform_(new_fc.weight[old_classes:], a=np.sqrt(5))
        if new_fc.bias is not None:
            fan_in = new_fc.in_features
            bound = 1.0 / np.sqrt(fan_in)
            nn.init.uniform_(new_fc.bias[old_classes:], -bound, bound)

    model.fc = new_fc
    model.num_classes = new_num_classes
    return model


def pad_ewc_state_for_expanded_head(
    model: TrustVisionCNN,
    fisher: Dict[str, torch.Tensor],
    old_params: Dict[str, torch.Tensor],
    old_num_classes: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Pads Fisher and anchor parameters after a 3 -> 6 classifier expansion.

    New classifier rows receive zero Fisher values and anchors equal to their
    fresh initialization, so EWC does not constrain the new classes.
    """
    padded_fisher = {name: tensor.clone() for name, tensor in fisher.items()}
    padded_old = {name: tensor.clone() for name, tensor in old_params.items()}

    if "fc.weight" in padded_fisher:
        f = padded_fisher["fc.weight"]
        p = padded_old["fc.weight"]
        new_f = torch.zeros_like(model.fc.weight)
        new_p = model.fc.weight.detach().clone()
        new_f[:old_num_classes] = f[:old_num_classes]
        new_p[:old_num_classes] = p[:old_num_classes]
        padded_fisher["fc.weight"] = new_f
        padded_old["fc.weight"] = new_p

    if "fc.bias" in padded_fisher and model.fc.bias is not None:
        f = padded_fisher["fc.bias"]
        p = padded_old["fc.bias"]
        new_f = torch.zeros_like(model.fc.bias)
        new_p = model.fc.bias.detach().clone()
        new_f[:old_num_classes] = f[:old_num_classes]
        new_p[:old_num_classes] = p[:old_num_classes]
        padded_fisher["fc.bias"] = new_f
        padded_old["fc.bias"] = new_p

    return padded_fisher, padded_old


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _cycle_batches(loader: DataLoader):
    """Endless iterator over a DataLoader (used for replay batches)."""
    while True:
        for batch in loader:
            yield batch


def stratified_replay_indices(
    dataset,
    replay_size: int,
    classes: Sequence[int] = (0, 1, 2),
    seed: int = 42,
) -> List[int]:
    """
    Builds a deterministic, class-balanced replay buffer.

    The split file stores training indices grouped by class, so a naive
    ``range(replay_size)`` buffer can contain a single class. This helper takes
    ``replay_size // K`` images per old class (plus a seeded remainder) so that
    replay actually rehearses every old class.
    """
    if replay_size <= 0:
        return []

    classes = [int(c) for c in classes]
    buckets: Dict[int, List[int]] = {c: [] for c in classes}
    for idx in range(len(dataset)):
        label = int(dataset[idx][1])
        if label in buckets:
            buckets[label].append(idx)

    rng = random.Random(seed)
    per_class = replay_size // len(classes)
    indices: List[int] = []
    for label in classes:
        pool = buckets[label][:]
        rng.shuffle(pool)
        indices.extend(pool[:per_class])

    remainder = replay_size - len(indices)
    if remainder > 0:
        leftovers = [
            idx
            for label in classes
            for idx in buckets[label][per_class:]
            if idx not in indices
        ]
        rng.shuffle(leftovers)
        indices.extend(leftovers[:remainder])

    return indices



def train_task(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    fisher: Optional[Dict[str, torch.Tensor]] = None,
    old_params: Optional[Dict[str, torch.Tensor]] = None,
    ewc_lambda: float = 0.0,
    replay_loader: Optional[DataLoader] = None,
    replay_weight: float = 1.0,
    freeze_old_head_rows: bool = False,
    old_num_classes: int = 3,
    max_batches_per_epoch: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Trains a task with optional EWC regularization and/or replay.

    Loss = CE(task batch)
           + replay_weight * CE(replay batch)   (if replay_loader)
           + (ewc_lambda / 2) * sum F_i (theta_i - theta_i*)^2

    ``freeze_old_head_rows`` keeps the classifier rows of the old classes
    fixed after the 3 -> 6 expansion, which prevents the new logits from
    suppressing the old ones in the single-head protocol.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    replay_iter = _cycle_batches(replay_loader) if replay_loader is not None else None

    history: Dict[str, List[float]] = {"epoch": [], "loss": [], "task_loss": [], "ewc_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_task = 0.0
        running_ewc = 0.0
        n_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            task_loss = F.cross_entropy(logits, labels)
            loss = task_loss

            if replay_iter is not None:
                replay_images, replay_labels = next(replay_iter)
                replay_images = replay_images.to(device)
                replay_labels = replay_labels.to(device)
                replay_loss = F.cross_entropy(model(replay_images), replay_labels)
                loss = loss + float(replay_weight) * replay_loss

            ewc_loss = torch.zeros((), device=device)
            if fisher is not None and old_params is not None and ewc_lambda > 0:
                ewc_loss = ewc_penalty(model, fisher, old_params, ewc_lambda)
                loss = loss + ewc_loss

            loss.backward()
            if freeze_old_head_rows:
                head = getattr(model, "fc", None)
                if head is not None and head.weight.grad is not None:
                    head.weight.grad[:old_num_classes].zero_()
                    if head.bias is not None and head.bias.grad is not None:
                        head.bias.grad[:old_num_classes].zero_()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += float(loss.item())
            running_task += float(task_loss.item())
            running_ewc += float(ewc_loss.item())
            n_batches += 1

        scheduler.step()
        history["epoch"].append(epoch)
        history["loss"].append(round(running_loss / max(1, n_batches), 4))
        history["task_loss"].append(round(running_task / max(1, n_batches), 4))
        history["ewc_loss"].append(round(running_ewc / max(1, n_batches), 4))
        if verbose:
            print(
                f"  [Task] epoch {epoch:02d}/{epochs:02d} "
                f"- loss={history['loss'][-1]:.4f} - CE={history['task_loss'][-1]:.4f} "
                f"- EWC={history['ewc_loss'][-1]:.4f}"
            )

    return history


# ---------------------------------------------------------------------------
# Metrics for single-head continual learning
# ---------------------------------------------------------------------------

def per_class_accuracy(confusion_matrix: Sequence[Sequence[int]]) -> List[float]:
    """Returns per-class recall from a confusion matrix (rows = true labels)."""
    cm = np.asarray(confusion_matrix, dtype=np.float64)
    diag = np.diag(cm)
    totals = cm.sum(axis=1)
    return [float(diag[i] / totals[i]) if totals[i] > 0 else 0.0 for i in range(cm.shape[0])]


def collect_gradcam_maps(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Collects Grad-CAM maps (B, H, W) for a full DataLoader pass."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)
    explainer = GradCAM(model)
    maps: List[np.ndarray] = []
    try:
        for images, _ in loader:
            maps.append(explainer(images.to(device)))
    finally:
        explainer.remove_hooks()

    if not maps:
        raise ValueError("Cannot compute Grad-CAM on an empty DataLoader.")
    return np.concatenate(maps, axis=0)


def evaluate_continual_model(
    model: nn.Module,
    test_loader: DataLoader,
    old_classes: Sequence[int] = (0, 1, 2),
    new_classes: Sequence[int] = (3, 4, 5),
    device: Optional[torch.device] = None,
    previous_old_acc: Optional[float] = None,
) -> Dict[str, Any]:
    """Full metrics + old/new class accuracy + forgetting for a 6-class model.

    ``new_classes`` may be empty when evaluating the Task-1 model (3 outputs),
    in which case ``new_acc`` and ``new_class_recalls`` are ``None``.
    """
    metrics = evaluate_all(model, test_loader, device=device)
    recalls = per_class_accuracy(metrics["confusion_matrix"])
    old_acc = float(np.mean([recalls[i] for i in old_classes]))

    new_acc: Optional[float] = None
    if len(new_classes) > 0:
        new_acc = float(np.mean([recalls[i] for i in new_classes]))

    metrics.update(
        {
            "old_acc": round(old_acc, 4),
            "new_acc": None if new_acc is None else round(new_acc, 4),
            "old_class_recalls": [round(recalls[i], 4) for i in old_classes],
            "new_class_recalls": (
                None if len(new_classes) == 0
                else [round(recalls[i], 4) for i in new_classes]
            ),
        }
    )
    if previous_old_acc is not None:
        metrics["forgetting"] = round(float(previous_old_acc) - old_acc, 4)
    return metrics


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _build_task_model(num_classes: int, device: torch.device, backbone_path: Optional[Path]) -> TrustVisionCNN:
    model = TrustVisionCNN(num_classes=num_classes)
    if backbone_path is not None and backbone_path.exists():
        state = torch.load(backbone_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Load all compatible tensors; fc mismatch is expected when 3 -> 6.
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"  [Init] Loaded SimCLR backbone {backbone_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    return model.to(device)


def run_s4_ewc_experiment(
    seed: int = 42,
    epochs_task1: int = 10,
    epochs_task2: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    ewc_lambdas: Sequence[float] = (1.0, 10.0, 100.0, 1000.0),
    replay_size: int = 120,
    device: Optional[torch.device] = None,
    backbone_path: Union[str, Path] = "checkpoints/simclr_backbone_3cls.pt",
    out_path: Union[str, Path] = "results/s4_ewc.json",
    root_dir: str = "data/NEU-CLS",
    max_batches_per_epoch: Optional[int] = None,
    calibrate_strategies: bool = True,
    fisher_normalization: str = "none",
    include_ewc_replay: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Runs the full S4 protocol: Task-1 training, Fisher estimation, 3 -> 6 head
    expansion, then naive / EWC / replay fine-tuning and evaluation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    old_classes = [0, 1, 2]
    new_classes = [3, 4, 5]
    backbone_path = Path(backbone_path)

    print("[S4] Loading Task-1 (old classes) data...")
    train_loader_1, _, test_loader_1, _ = get_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        seed=seed,
        img_size=96,
        task_classes=old_classes,
        num_classes=3,
        auto_download=True,
    )
    print("[S4] Loading Task-2 (new classes) data...")
    train_loader_2, _, _, _ = get_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        seed=seed,
        img_size=96,
        task_classes=new_classes,
        num_classes=6,
        auto_download=True,
    )
    print("[S4] Loading the full 6-class calibration/test splits for old/new evaluation...")
    _, cal_loader_all, test_loader_all, _ = get_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        seed=seed,
        img_size=96,
        task_classes=None,
        num_classes=6,
        auto_download=True,
    )

    # ------------------------------------------------------------------
    # Task 1 (3-class head)
    # ------------------------------------------------------------------
    model_task1 = _build_task_model(3, device, backbone_path)
    history_task1 = train_task(
        model_task1,
        train_loader_1,
        epochs=epochs_task1,
        lr=lr,
        device=device,
        max_batches_per_epoch=max_batches_per_epoch,
        verbose=verbose,
    )
    metrics_task1 = evaluate_continual_model(
        model_task1,
        test_loader_1,
        old_classes=old_classes,
        new_classes=(),
        device=device,
    )
    task1_ckpt = Path(f"checkpoints/ewc_task1_seed{seed}.pt")
    task1_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_task1.state_dict(), task1_ckpt)
    print(f"[S4] Task-1 checkpoint saved to {task1_ckpt} (old acc={metrics_task1['old_acc']:.4f})")

    # ------------------------------------------------------------------
    # Fisher information on Task-1 data
    # ------------------------------------------------------------------
    print("[S4] Estimating diagonal Fisher information...")
    fisher = compute_fisher_diagonal(model_task1, train_loader_1, device=device)
    old_params = {name: param.detach().clone() for name, param in model_task1.named_parameters()}
    fisher_path = Path(f"checkpoints/ewc_fisher_task1_seed{seed}.pt")
    torch.save({"fisher": fisher, "old_params": old_params}, fisher_path)

    if fisher_normalization == "per_tensor":
        fisher_for_ewc = normalize_fisher_per_tensor(fisher)
        print("[S4] Fisher normalized per tensor (mean = 1) for the EWC penalty.")
    elif fisher_normalization == "none":
        fisher_for_ewc = fisher
    else:
        raise ValueError(
            f"Unknown fisher_normalization={fisher_normalization!r}; "
            "expected 'none' or 'per_tensor'."
        )

    # Replay buffer: a deterministic, class-balanced subset of Task-1 images.
    replay_indices: List[int] = []
    replay_counts: Dict[int, int] = {}
    replay_loader = None
    if replay_size > 0:
        replay_indices = stratified_replay_indices(
            train_loader_1.dataset,
            replay_size,
            classes=old_classes,
            seed=seed,
        )
        replay_counts = Counter(int(train_loader_1.dataset[i][1]) for i in replay_indices)
        replay_subset = torch.utils.data.Subset(train_loader_1.dataset, replay_indices)
        replay_loader = DataLoader(replay_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        print(
            f"[S4] Replay buffer: {len(replay_indices)} images, "
            f"class counts={dict(sorted(replay_counts.items()))}"
        )

    # ------------------------------------------------------------------
    # Task 2 strategies
    # ------------------------------------------------------------------
    strategies: Dict[str, Dict[str, Any]] = {}

    def run_strategy(name: str, lambda_value: float, use_replay: bool) -> Dict[str, Any]:
        model = copy.deepcopy(model_task1).to(device)
        expand_classifier(model, new_num_classes=6)

        strategy_fisher = None
        strategy_old = None
        if lambda_value > 0:
            strategy_fisher, strategy_old = pad_ewc_state_for_expanded_head(
                model, fisher_for_ewc, old_params, old_num_classes=len(old_classes)
            )

        history = train_task(
            model,
            train_loader_2,
            epochs=epochs_task2,
            lr=lr,
            device=device,
            fisher=strategy_fisher,
            old_params=strategy_old,
            ewc_lambda=lambda_value,
            replay_loader=replay_loader if use_replay else None,
            replay_weight=1.0,
            max_batches_per_epoch=max_batches_per_epoch,
            verbose=verbose,
        )
        metrics = evaluate_continual_model(
            model,
            test_loader_all,
            old_classes=old_classes,
            new_classes=new_classes,
            device=device,
            previous_old_acc=metrics_task1["old_acc"],
        )

        if calibrate_strategies:
            calibrated = fit_temperature_and_evaluate(
                model, cal_loader_all, test_loader_all, device=device
            )
            metrics["temperature"] = calibrated["temperature"]
            metrics["calibrated"] = calibrated_summary(calibrated)

        if name == "naive":
            ckpt = Path(f"checkpoints/ewc_naive_seed{seed}.pt")
        elif name == "replay":
            ckpt = Path(f"checkpoints/ewc_replay_seed{seed}.pt")
        elif name == "ewc_replay":
            ckpt = Path(f"checkpoints/ewc_plus_replay_seed{seed}.pt")
        else:
            ckpt = Path(f"checkpoints/ewc_lambda_{lambda_value:g}_seed{seed}.pt")
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)

        result = {
            "strategy": name,
            "ewc_lambda": lambda_value,
            "replay_size": replay_size if use_replay else 0,
            "replay_class_counts": dict(sorted(replay_counts.items())) if use_replay else {},
            "checkpoint": str(ckpt.as_posix()),
            "history": history,
            "metrics": metrics,
        }
        print(
            f"[S4] {name:>6} (lambda={lambda_value:g}): overall={metrics['acc']:.4f}, "
            f"old={metrics['old_acc']:.4f}, new={metrics['new_acc']:.4f}, "
            f"forgetting={metrics.get('forgetting', 0):.4f}, ece={metrics['ece']:.4f}"
            + (
                f", ece_cal={metrics['calibrated']['ece']:.4f}"
                if "calibrated" in metrics else ""
            )
        )
        return result

    print("[S4] Training naive fine-tuning baseline...")
    strategies["naive"] = run_strategy("naive", 0.0, use_replay=False)

    print("[S4] Training replay baseline...")
    strategies["replay"] = run_strategy("replay", 0.0, use_replay=True)

    for lambda_value in ewc_lambdas:
        print(f"[S4] Training EWC with lambda={lambda_value:g}...")
        strategies[f"ewc_lambda_{lambda_value:g}"] = run_strategy(
            f"ewc_lambda_{lambda_value:g}", float(lambda_value), use_replay=False
        )

    # Best pure EWC lambda -> canonical checkpoint.
    ewc_rows = [v for k, v in strategies.items() if k.startswith("ewc_lambda_")]
    best_pure_ewc = max(
        ewc_rows,
        key=lambda row: (row["metrics"]["old_acc"] + row["metrics"]["new_acc"]) / 2.0,
    )
    canonical = Path(f"checkpoints/ewc_seed{seed}.pt")
    torch.save(
        torch.load(best_pure_ewc["checkpoint"], map_location="cpu", weights_only=False),
        canonical,
    )

    # Optional hybrid: the best pure EWC lambda combined with the replay buffer.
    if include_ewc_replay:
        print(
            f"[S4] Training EWC + replay with lambda={best_pure_ewc['ewc_lambda']:g}..."
        )
        strategies["ewc_replay"] = run_strategy(
            "ewc_replay", float(best_pure_ewc["ewc_lambda"]), use_replay=True
        )

    # Best EWC-family strategy (pure lambda sweep or hybrid) feeds the S4 row.
    ewc_family = [
        row for name, row in strategies.items()
        if name.startswith("ewc_lambda_") or name == "ewc_replay"
    ]
    best_ewc = max(
        ewc_family,
        key=lambda row: (row["metrics"]["old_acc"] + row["metrics"]["new_acc"]) / 2.0,
    )

    # ------------------------------------------------------------------
    # Intra-pipeline XAI stability: Task-1 explanations vs best EWC model
    # ------------------------------------------------------------------
    print("[S4] Computing Grad-CAM stability (Task-1 model vs best EWC)...")
    best_model = copy.deepcopy(model_task1)
    expand_classifier(best_model, new_num_classes=6)
    best_state = torch.load(best_ewc["checkpoint"], map_location=device, weights_only=False)
    best_model.load_state_dict(best_state)
    best_model.to(device).eval()

    maps_task1 = collect_gradcam_maps(model_task1, test_loader_1, device=device)
    maps_ewc = collect_gradcam_maps(best_model, test_loader_1, device=device)
    xai_stability = float(
        compute_intra_pipeline_stability(maps_task1, maps_ewc, method="cosine")
    )
    best_ewc["metrics"]["intra_xai_stab"] = round(xai_stability, 4)
    strategies[best_ewc["strategy"]]["metrics"]["intra_xai_stab"] = round(xai_stability, 4)
    print(f"[S4] Intra-pipeline Grad-CAM stability: {xai_stability:.4f}")

    output = {
        "stage": "S4_EWC_CONTINUAL_LEARNING",
        "protocol": "single-head class-incremental, 3 -> 6 classes, no task ID",
        "seed": seed,
        "task1_classes": old_classes,
        "task2_classes": new_classes,
        "task1_metrics": metrics_task1,
        "task1_history": history_task1,
        "task1_checkpoint": str(task1_ckpt.as_posix()),
        "fisher_checkpoint": str(fisher_path.as_posix()),
        "replay_size": replay_size,
        "replay_class_counts": dict(sorted(replay_counts.items())),
        "calibration_split": "10% cal (180 images), shared splits.json",
        "calibration_notes": [
            f"{name}: fitted T={calibrated.get('temperature_fitted')} rejected "
            f"(calibration NLL {calibrated.get('calibration_nll_fitted')} vs "
            f"{calibrated.get('calibration_nll')} at T=1); raw probabilities used."
            for name, row in strategies.items()
            for calibrated in [row["metrics"].get("calibrated") or {}]
            if calibrated.get("calibration_accepted") is False
        ],
        "fisher_normalization": fisher_normalization,
        "strategies": strategies,
        "best_pure_ewc": {
            "strategy": best_pure_ewc["strategy"],
            "ewc_lambda": best_pure_ewc["ewc_lambda"],
            "checkpoint": best_pure_ewc["checkpoint"],
            "metrics": best_pure_ewc["metrics"],
        },
        "best_ewc": {
            "strategy": best_ewc["strategy"],
            "ewc_lambda": best_ewc["ewc_lambda"],
            "checkpoint": best_ewc["checkpoint"],
            "metrics": best_ewc["metrics"],
        },
        "intra_xai_stability": {
            "reference": "Task-1 model Grad-CAM",
            "compared_to": best_ewc["strategy"],
            "method": "cosine",
            "value": round(xai_stability, 4),
        },
        "canonical_checkpoint": str(canonical.as_posix()),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[S4] Saved results to {out_path}")
    return output
