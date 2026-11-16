"""
TrustVision - Training Module (src/train.py)
Trains baseline or adversarial CNN models on the canonical NEU-CLS train split.
"""

import argparse
import random
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import get_dataloaders
from src.model import TrustVisionCNN
from src.eval import evaluate_all, fgsm_attack


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_cnn(
    seed: int = 42,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    save_path: str = "checkpoints/cnn_base_seed42.pt",
    adversarial: bool = False,
    adv_eps: float = 0.03,
    adv_ratio: float = 0.5,
    random_start: bool = True,
    root_dir: str = "data/NEU-CLS",
    label_fraction: float = 1.0,
    task_classes: Optional[List[int]] = None,
    num_classes: Optional[int] = None,
    remap_task_labels: bool = False,
) -> Tuple[TrustVisionCNN, Dict[str, Any]]:
    """
    Trains TrustVisionCNN on NEU-CLS train split (1260 images or subset).
    Supports standard training, adversarial training, custom label fractions (S1),
    and task-specific class subsets / heads (S4 continual learning).

    Args:
        seed: Random seed (42, 123, 999).
        epochs: Number of training epochs.
        batch_size: Batch size for DataLoader.
        lr: Learning rate for Adam optimizer.
        weight_decay: Weight decay regularizer.
        device: CPU or CUDA device.
        save_path: Destination checkpoint path.
        adversarial: If True, uses adversarial training.
        adv_eps: Attack perturbation budget for adversarial training.
        adv_ratio: Proportion of adversarial examples in the batch.
        random_start: Random initial perturbation to avoid catastrophic overfitting.
        root_dir: Path to NEU-CLS dataset.
        label_fraction: Proportion of training labels to use (0.01, 0.05, 0.10, 1.0).
        task_classes: Subset of classes to train on (e.g. [0, 1, 2] for S4 task 1).
        num_classes: Number of classification head outputs (default: len(task_classes) if task_classes else 6).
        remap_task_labels: If True, remaps task_classes to 0..len(task_classes)-1.

    Returns:
        (trained_model, history_dict)
    """
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if num_classes is None:
        num_classes = len(task_classes) if task_classes is not None else 6

    print(
        f"[Train] Initializing dataloaders for seed={seed}, img_size=96, "
        f"label_fraction={label_fraction}, task_classes={task_classes}, num_classes={num_classes}..."
    )
    train_loader, cal_loader, test_loader, _ = get_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        seed=seed,
        img_size=96,
        label_fraction=label_fraction,
        task_classes=task_classes,
        remap_task_labels=remap_task_labels,
        num_classes=num_classes,
        auto_download=True,
    )

    model = TrustVisionCNN(num_classes=num_classes, feature_dim=256, normalize_input=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "epoch": [],
        "loss": [],
        "train_acc": [],
    }

    print(f"[Train] Starting training ({'Adversarial' if adversarial else 'Clean Baseline'}) for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            if adversarial:
                # Fast Adversarial Training with random start
                bsz = images.size(0)
                n_adv = int(bsz * adv_ratio)

                if n_adv > 0:
                    clean_imgs = images[n_adv:]
                    clean_lbls = labels[n_adv:]
                    target_imgs = images[:n_adv]
                    target_lbls = labels[:n_adv]

                    # Random start perturbation inside [-adv_eps, adv_eps]
                    if random_start:
                        alpha = adv_eps
                        delta = torch.empty_like(target_imgs).uniform_(-alpha, alpha)
                        adv_init = torch.clamp(target_imgs + delta, 0.0, 1.0)
                    else:
                        adv_init = target_imgs

                    adv_imgs = fgsm_attack(model, adv_init, target_lbls, eps=adv_eps)

                    if len(clean_imgs) > 0:
                        batch_imgs = torch.cat([clean_imgs, adv_imgs], dim=0)
                        batch_lbls = torch.cat([clean_lbls, target_lbls], dim=0)
                    else:
                        batch_imgs = adv_imgs
                        batch_lbls = target_lbls
                else:
                    batch_imgs = images
                    batch_lbls = labels
            else:
                batch_imgs = images
                batch_lbls = labels

            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_lbls)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_lbls).sum().item()
            total += batch_imgs.size(0)

        scheduler.step()
        epoch_loss = total_loss / total
        epoch_acc = correct / total

        history["epoch"].append(epoch)
        history["loss"].append(round(epoch_loss, 4))
        history["train_acc"].append(round(epoch_acc, 4))

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc * 100:.2f}%")

    # Save checkpoint
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(save_path_obj))
    print(f"[Train] Model saved to {save_path_obj}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TrustVision CNN")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=str, default="checkpoints/cnn_base_seed42.pt")
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument("--adv-eps", type=float, default=0.03)
    parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--task-classes", type=int, nargs="*", default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    args = parser.parse_args()

    task_classes = args.task_classes
    if task_classes is not None and len(task_classes) == 0:
        task_classes = None

    train_cnn(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        adversarial=args.adversarial,
        adv_eps=args.adv_eps,
        label_fraction=args.label_fraction,
        task_classes=task_classes,
        num_classes=args.num_classes,
    )
