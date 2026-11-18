"""
TrustVision - Member C experiment orchestrator (src/run_member_c.py)
====================================================================
Runs the two Member C stages end-to-end and updates the shared artefacts:

* **S1 (SimCLR)** : 6-class SimCLR pre-training, frozen linear probes at
  1 % / 5 % / 10 % / 100 % labels, comparison with a supervised CNN trained
  from scratch on the same budgets, and comparison with the CLIP probe (S0).
* **S4 (EWC)** : SimCLR encoder pre-trained *only on the 3 initial classes*
  (no leakage), single-head 3 -> 6 class-incremental learning, EWC vs naive
  fine-tuning vs replay, lambda sweep, calibration and FGSM robustness.

Outputs
-------
* ``results/s1_simclr.json``
* ``results/s4_ewc.json``
* ``results/cross_effects_table.json`` (S1/S4 rows updated in place)
* ``results/figures/s1_simclr_label_efficiency.png``
* ``results/figures/s4_ewc_lambda_sweep.png``

Usage
-----
    python -m src.run_member_c --stage all
    python -m src.run_member_c --stage s1 --simclr-epochs 20
    python -m src.run_member_c --stage s4 --ewc-lambdas 0.1 1 10 100
    python -m src.run_member_c --stage s4 --fisher-normalization none --ewc-lambdas 1e3 1e4 1e5
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import get_dataloaders
from src.eval import evaluate_all, format_stage_row
from src.ewc import run_s4_ewc_experiment
from src.member_c_utils import calibrated_summary, fit_temperature_and_evaluate
from src.model import TrustVisionCNN
from src.simclr import (
    compare_with_clip_probe,
    evaluate_simclr_label_fractions,
    simclr_pretrain,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fraction_key(fraction: float) -> str:
    return f"{int(round(fraction * 100))}pct"


def _with_calibrated_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Replaces accuracy/ECE/NLL/Brier by their temperature-scaled values."""
    merged = dict(metrics)
    calibrated = metrics.get("calibrated")
    # Old result files predate the no-harm flag and are treated as accepted.
    if calibrated and calibrated.get("calibration_accepted", True):
        for key in ("acc", "macro_f1", "ece", "nll", "brier"):
            if calibrated.get(key) is not None:
                merged[key] = calibrated[key]
    return merged


def _calibration_note(metrics: Dict[str, Any]) -> Optional[str]:
    """Explains a rejected temperature fit in the cross-effects table."""
    calibrated = metrics.get("calibrated") or {}
    if calibrated.get("calibration_accepted") is not False:
        return None
    return (
        f"Fitted T={calibrated.get('temperature_fitted')} rejected: calibration NLL "
        f"{calibrated.get('calibration_nll_fitted')} vs {calibrated.get('calibration_nll')} at T=1; "
        "raw probabilities reported."
    )


# ---------------------------------------------------------------------------
# Supervised from-scratch baseline (S1 comparison)
# ---------------------------------------------------------------------------

def train_supervised_fraction(
    fraction: float,
    seed: int = 42,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    root_dir: str = "data/NEU-CLS",
    max_batches_per_epoch: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[TrustVisionCNN, Dict[str, List[float]], Dict[str, Any]]:
    """Trains the shared CNN from scratch on a label budget (no SimCLR)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    train_loader, cal_loader, test_loader, _ = get_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        seed=seed,
        img_size=96,
        label_fraction=fraction,
        num_classes=6,
        auto_download=True,
    )

    model = TrustVisionCNN(num_classes=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    history: Dict[str, List[float]] = {"epoch": [], "loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        history["epoch"].append(epoch)
        history["loss"].append(round(running_loss / max(1, n_batches), 4))
        if verbose:
            print(
                f"  [Scratch {fraction * 100:5.1f}%] epoch {epoch:02d}/{epochs:02d} "
                f"- CE: {history['loss'][-1]:.4f}"
            )

    metrics = evaluate_all(model, test_loader, device=device)
    calibrated = fit_temperature_and_evaluate(model, cal_loader, test_loader, device=device)
    metrics["temperature"] = calibrated["temperature"]
    metrics["calibrated"] = calibrated_summary(calibrated)
    metrics["label_fraction"] = fraction
    metrics["num_train_labels"] = int(len(train_loader.dataset))
    print(
        f"  [Scratch {fraction * 100:5.1f}%] test acc={metrics['acc']:.4f}, "
        f"macro-F1={metrics['macro_f1']:.4f}, "
        f"ECE={metrics['ece']:.4f} (cal={metrics['calibrated']['ece']:.4f}), "
        f"rob={metrics['rob_acc_003']:.4f}"
    )
    return model, history, metrics


# ---------------------------------------------------------------------------
# S1 - SimCLR
# ---------------------------------------------------------------------------

def run_s1_simclr(
    seed: int = 42,
    simclr_epochs: int = 20,
    supervised_epochs: int = 15,
    batch_size: int = 64,
    supervised_batch_size: int = 32,
    lr: float = 1e-3,
    temperature: float = 0.5,
    fractions: Sequence[float] = (0.01, 0.05, 0.10, 1.0),
    backbone_path: str = "checkpoints/simclr_backbone_6cls.pt",
    probe_path: str = "checkpoints/s1_simclr_probe_100pct.pt",
    probe_checkpoint_pattern: Optional[str] = "checkpoints/s1_simclr_probe_{fraction}pct.pt",
    scratch_path: str = "checkpoints/s1_scratch_100pct.pt",
    scratch_checkpoint_pattern: Optional[str] = "checkpoints/s1_scratch_{fraction}pct.pt",
    main_path: str = "checkpoints/s1_main_model.pt",
    out_path: str = "results/s1_simclr.json",
    root_dir: str = "data/NEU-CLS",
    device: Optional[torch.device] = None,
    force_retrain: bool = False,
    skip_scratch: bool = False,
    max_batches_per_epoch: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Runs the full S1 protocol and returns the JSON payload."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not any(abs(float(fraction) - 1.0) < 1e-6 for fraction in fractions):
        raise ValueError(
            "run_s1_simclr requires the 100% label fraction to select the main model."
        )

    backbone = Path(backbone_path)
    pretrain_history: Optional[Dict[str, List[float]]] = None
    if force_retrain or not backbone.exists():
        print(f"[S1] Pre-training SimCLR on all 6 classes (unlabeled) -> {backbone}")
        train_loader, _, _, _ = get_dataloaders(
            root_dir=root_dir,
            batch_size=batch_size,
            seed=seed,
            img_size=96,
            label_fraction=1.0,
            num_classes=6,
            auto_download=True,
        )
        _, pretrain_history = simclr_pretrain(
            train_loader,
            epochs=simclr_epochs,
            lr=lr,
            temperature=temperature,
            device=device,
            save_path=backbone,
            num_classes=6,
            max_batches_per_epoch=max_batches_per_epoch,
            verbose=verbose,
        )
    else:
        print(f"[S1] Reusing existing SimCLR backbone: {backbone}")

    print("[S1] Evaluating frozen linear probes at 1 / 5 / 10 / 100 % labels...")
    simclr_results = evaluate_simclr_label_fractions(
        backbone,
        fractions=fractions,
        seed=seed,
        batch_size=batch_size,
        device=device,
        num_classes=6,
        root_dir=root_dir,
        calibrate=True,
        probe_checkpoint_path=probe_path,
        probe_checkpoint_pattern=probe_checkpoint_pattern,
    )

    scratch_results: Dict[str, Dict[str, Any]] = {}
    if not skip_scratch:
        print("[S1] Training supervised CNN from scratch on the same label budgets...")
        for fraction in fractions:
            model, history, metrics = train_supervised_fraction(
                fraction,
                seed=seed,
                epochs=supervised_epochs,
                batch_size=supervised_batch_size,
                lr=lr,
                device=device,
                root_dir=root_dir,
                max_batches_per_epoch=max_batches_per_epoch,
                verbose=verbose,
            )
            metrics["history"] = history
            scratch_results[_fraction_key(fraction)] = metrics
            if abs(fraction - 1.0) < 1e-6:
                scratch_ckpt = Path(scratch_path)
            elif scratch_checkpoint_pattern is not None:
                scratch_ckpt = Path(
                    scratch_checkpoint_pattern.format(
                        fraction=int(round(fraction * 100))
                    )
                )
            else:
                scratch_ckpt = None
            if scratch_ckpt is not None:
                scratch_ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), scratch_ckpt)
                metrics["checkpoint"] = str(scratch_ckpt.as_posix())

    # ------------------------------------------------------------------
    # "Le meilleur devient le modèle principal"
    # ------------------------------------------------------------------
    sim_100 = simclr_results["100pct"]
    main_model: Dict[str, Any]
    if skip_scratch or sim_100["acc"] >= scratch_results.get("100pct", {}).get("acc", -1.0):
        shutil.copy2(probe_path, main_path)
        main_model = {
            "source": "SimCLR frozen linear probe (100% labels)",
            "checkpoint": main_path,
            "acc": sim_100["acc"],
            "macro_f1": sim_100["macro_f1"],
            "ece_calibrated": sim_100.get("calibrated", {}).get("ece"),
        }
    else:
        shutil.copy2(scratch_path, main_path)
        scratch_100 = scratch_results["100pct"]
        main_model = {
            "source": "Supervised CNN from scratch (100% labels)",
            "checkpoint": main_path,
            "acc": scratch_100["acc"],
            "macro_f1": scratch_100["macro_f1"],
            "ece_calibrated": scratch_100.get("calibrated", {}).get("ece"),
        }
    print(f"[S1] Main model -> {main_model['source']} (acc={main_model['acc']:.4f})")

    output: Dict[str, Any] = {
        "stage": "S1_SIMCLR",
        "protocol": (
            "6-class SimCLR pre-training on unlabeled train images, frozen "
            "features + standardized logistic regression probes at 1/5/10/100% "
            "labels; supervised CNN from scratch at the same budgets."
        ),
        "seed": seed,
        "simclr": {
            "checkpoint": str(Path(backbone_path).as_posix()),
            "epochs": simclr_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "temperature": temperature,
            "history": pretrain_history,
            "label_fractions": simclr_results,
        },
        "supervised_from_scratch": {
            "epochs": supervised_epochs,
            "batch_size": supervised_batch_size,
            "label_fractions": scratch_results,
        },
        "clip_reference": compare_with_clip_probe(simclr_results),
        "main_model": main_model,
        "status": "COMPLETED",
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[S1] Saved results to {out}")

    plot_s1_label_efficiency(simclr_results, scratch_results)
    return output


def plot_s1_label_efficiency(
    simclr_results: Dict[str, Dict[str, Any]],
    scratch_results: Dict[str, Dict[str, Any]],
    out_path: str = "results/figures/s1_simclr_label_efficiency.png",
) -> Path:
    """Plots accuracy and calibrated ECE vs label budget for S1."""
    keys = [k for k in ("1pct", "5pct", "10pct", "100pct") if k in simclr_results]
    xs = [int(k.replace("pct", "")) for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)

    ax = axes[0]
    ax.plot(
        xs,
        [simclr_results[k]["acc"] * 100 for k in keys],
        marker="o",
        linewidth=2,
        label="SimCLR (frozen linear probe)",
    )
    if scratch_results:
        ax.plot(
            xs,
            [scratch_results[k]["acc"] * 100 for k in keys],
            marker="s",
            linewidth=2,
            label="CNN from scratch",
        )
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x}%" for x in xs])
    ax.set_xlabel("Label budget")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("S1 - Label efficiency", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(
        xs,
        [
            (simclr_results[k].get("calibrated") or {}).get("ece", simclr_results[k]["ece"])
            for k in keys
        ],
        marker="o",
        linewidth=2,
        label="SimCLR (calibrated)",
    )
    if scratch_results:
        ax.plot(
            xs,
            [
                (scratch_results[k].get("calibrated") or {}).get("ece", scratch_results[k]["ece"])
                for k in keys
            ],
            marker="s",
            linewidth=2,
            label="CNN from scratch (calibrated)",
        )
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x}%" for x in xs])
    ax.set_xlabel("Label budget")
    ax.set_ylabel("ECE (lower is better)")
    ax.set_title("S1 - Calibration", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[S1] Figure saved to {out}")
    return out


# ---------------------------------------------------------------------------
# S4 - EWC
# ---------------------------------------------------------------------------

def run_s4_ewc(
    seed: int = 42,
    simclr_epochs: int = 20,
    task1_epochs: int = 10,
    task2_epochs: int = 10,
    batch_size: int = 32,
    simclr_batch_size: int = 64,
    lr: float = 1e-3,
    temperature: float = 0.5,
    ewc_lambdas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    replay_size: int = 120,
    backbone_path: str = "checkpoints/simclr_backbone_3cls.pt",
    out_path: str = "results/s4_ewc.json",
    root_dir: str = "data/NEU-CLS",
    device: Optional[torch.device] = None,
    force_retrain: bool = False,
    max_batches_per_epoch: Optional[int] = None,
    fisher_normalization: str = "per_tensor",
    include_ewc_replay: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Runs the full S4 protocol (3-class encoder, then EWC vs naive vs replay)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = Path(backbone_path)
    if force_retrain or not backbone.exists():
        print("[S4] Pre-training the SimCLR encoder on the 3 INITIAL classes only...")
        train_loader_1, _, _, _ = get_dataloaders(
            root_dir=root_dir,
            batch_size=simclr_batch_size,
            seed=seed,
            img_size=96,
            task_classes=[0, 1, 2],
            num_classes=3,
            auto_download=True,
        )
        simclr_pretrain(
            train_loader_1,
            epochs=simclr_epochs,
            lr=lr,
            temperature=temperature,
            device=device,
            save_path=backbone,
            num_classes=3,
            max_batches_per_epoch=max_batches_per_epoch,
            verbose=verbose,
        )
    else:
        print(f"[S4] Reusing existing 3-class SimCLR encoder: {backbone}")

    output = run_s4_ewc_experiment(
        seed=seed,
        epochs_task1=task1_epochs,
        epochs_task2=task2_epochs,
        batch_size=batch_size,
        lr=lr,
        ewc_lambdas=ewc_lambdas,
        replay_size=replay_size,
        device=device,
        backbone_path=backbone,
        out_path=out_path,
        root_dir=root_dir,
        max_batches_per_epoch=max_batches_per_epoch,
        calibrate_strategies=True,
        fisher_normalization=fisher_normalization,
        include_ewc_replay=include_ewc_replay,
        verbose=verbose,
    )
    output["simclr_encoder"] = {
        "checkpoint": str(backbone.as_posix()),
        "pretrained_on": "3 initial classes only (no leakage of classes 3-5)",
        "epochs": simclr_epochs,
        "temperature": temperature,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    plot_s4_lambda_sweep(output)
    return output


def plot_s4_lambda_sweep(
    s4_output: Dict[str, Any],
    out_path: str = "results/figures/s4_ewc_lambda_sweep.png",
) -> Path:
    """Plots old/new/overall accuracy and forgetting against the EWC lambda."""
    strategies = s4_output["strategies"]
    ewc_keys = sorted(
        [k for k in strategies if k.startswith("ewc_lambda_")],
        key=lambda k: float(strategies[k]["ewc_lambda"]),
    )
    lambdas = [float(strategies[k]["ewc_lambda"]) for k in ewc_keys]
    old_acc = [strategies[k]["metrics"]["old_acc"] * 100 for k in ewc_keys]
    new_acc = [strategies[k]["metrics"]["new_acc"] * 100 for k in ewc_keys]
    overall = [strategies[k]["metrics"]["acc"] * 100 for k in ewc_keys]
    forgetting = [strategies[k]["metrics"]["forgetting"] * 100 for k in ewc_keys]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)

    ax = axes[0]
    ax.plot(lambdas, old_acc, marker="o", linewidth=2, label="Old classes (1-3)")
    ax.plot(lambdas, new_acc, marker="s", linewidth=2, label="New classes (4-6)")
    ax.plot(lambdas, overall, marker="^", linewidth=2, label="Overall (6 classes)")
    naive = strategies["naive"]["metrics"]
    replay = strategies["replay"]["metrics"]
    ax.axhline(naive["old_acc"] * 100, linestyle="--", linewidth=1.2,
               color="tab:red", alpha=0.7, label="Naive old acc")
    ax.axhline(replay["old_acc"] * 100, linestyle=":", linewidth=1.2,
               color="tab:green", alpha=0.8, label="Replay old acc")
    ax.set_xscale("log")
    ax.set_xlabel(r"EWC $\lambda$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("S4 - Old/new accuracy vs lambda", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(lambdas, forgetting, marker="o", linewidth=2, label="EWC forgetting")
    ax.axhline(naive["forgetting"] * 100, linestyle="--", linewidth=1.2,
               color="tab:red", alpha=0.7, label="Naive forgetting")
    ax.axhline(replay["forgetting"] * 100, linestyle=":", linewidth=1.2,
               color="tab:green", alpha=0.8, label="Replay forgetting")
    ax.set_xscale("log")
    ax.set_xlabel(r"EWC $\lambda$")
    ax.set_ylabel("Forgetting (%)")
    ax.set_title("S4 - Catastrophic forgetting", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[S4] Figure saved to {out}")
    return out


# ---------------------------------------------------------------------------
# Cross-effects table
# ---------------------------------------------------------------------------

def update_cross_effects_table(
    s1_output: Dict[str, Any],
    s4_output: Dict[str, Any],
    table_path: str = "results/cross_effects_table.json",
) -> Dict[str, Any]:
    """Replaces the S1/S4 rows of the shared table with real measurements."""
    path = Path(table_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            table = json.load(f)
    else:
        table = {"title": "TrustVision Cross-Effects Comparative Matrix (S0 to S4)",
                 "split_distribution": "70% train (1260) / 10% cal (180) / 20% test (360)",
                 "seed": 42, "columns": [], "rows": []}

    rows_by_stage = {row["stage"]: row for row in table.get("rows", [])}

    s1_metrics = s1_output["simclr"]["label_fractions"]["100pct"]
    row_s1 = format_stage_row("S1_SimCLR", _with_calibrated_metrics(s1_metrics))
    row_s1["status"] = "COMPLETED"
    s1_xai = s1_metrics.get("xai_sim", s1_output.get("xai_sim"))
    if s1_xai is not None:
        row_s1["xai_sim"] = s1_xai
    s1_note = _calibration_note(s1_metrics)
    if s1_note:
        row_s1["calibration_note"] = s1_note
    rows_by_stage["S1_SimCLR"] = row_s1

    s4_metrics = s4_output["best_ewc"]["metrics"]
    row_s4 = format_stage_row("S4_EWC_Continual", _with_calibrated_metrics(s4_metrics))
    row_s4["status"] = "COMPLETED"
    s4_xai = s4_metrics.get("xai_sim", s4_output.get("xai_sim"))
    if s4_xai is not None:
        row_s4["xai_sim"] = s4_xai
    s4_note = _calibration_note(s4_metrics)
    if s4_note:
        row_s4["calibration_note"] = s4_note
    rows_by_stage["S4_EWC_Continual"] = row_s4

    ordered_stages = [row["stage"] for row in table.get("rows", [])]
    new_rows = [rows_by_stage[stage] for stage in ordered_stages if stage in rows_by_stage]
    new_rows += [row for stage, row in rows_by_stage.items() if stage not in ordered_stages]
    table["rows"] = new_rows
    table["member_c"] = {
        "s1_best_probe_100pct": {
            "acc": s1_metrics["acc"],
            "macro_f1": s1_metrics["macro_f1"],
            "ece_calibrated": (s1_metrics.get("calibrated") or {}).get("ece"),
            "xai_sim": s1_xai,
        },
        "s1_main_model": s1_output.get("main_model"),
        "s4_best_ewc": {
            "strategy": s4_output["best_ewc"]["strategy"],
            "lambda": s4_output["best_ewc"]["ewc_lambda"],
            "acc": s4_metrics["acc"],
            "old_acc": s4_metrics["old_acc"],
            "new_acc": s4_metrics["new_acc"],
            "forgetting": s4_metrics.get("forgetting"),
            "ece": s4_metrics.get("ece"),
            "ece_fitted_temperature": (s4_metrics.get("calibrated") or {}).get("ece"),
            "temperature": s4_metrics.get("temperature"),
            "temperature_fitted": (s4_metrics.get("calibrated") or {}).get("temperature_fitted"),
            "calibration_accepted": (s4_metrics.get("calibrated") or {}).get("calibration_accepted"),
            "xai_sim": s4_xai,
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    print(f"[Cross-Effects] Updated S1/S4 rows in {path}")
    return table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TrustVision Member C (S1 + S4) orchestrator")
    parser.add_argument("--stage", choices=["s1", "s4", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simclr-epochs", type=int, default=20)
    parser.add_argument("--supervised-epochs", type=int, default=15)
    parser.add_argument("--task1-epochs", type=int, default=10)
    parser.add_argument("--task2-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--supervised-batch-size", type=int, default=32)
    parser.add_argument("--s4-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.05, 0.10, 1.0])
    parser.add_argument("--ewc-lambdas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--replay-size", type=int, default=120)
    parser.add_argument(
        "--fisher-normalization",
        choices=["none", "per_tensor"],
        default="per_tensor",
        help="Rescale Fisher per tensor (mean=1) so lambda is comparable across layers.",
    )
    parser.add_argument(
        "--skip-ewc-replay",
        action="store_true",
        help="Skip the EWC + replay hybrid strategy.",
    )
    parser.add_argument("--root-dir", default="data/NEU-CLS")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-scratch", action="store_true")
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Member C] Device: {device}")

    s1_output = None
    s4_output = None

    if args.stage in ("s1", "all"):
        s1_output = run_s1_simclr(
            seed=args.seed,
            simclr_epochs=args.simclr_epochs,
            supervised_epochs=args.supervised_epochs,
            batch_size=args.batch_size,
            supervised_batch_size=args.supervised_batch_size,
            lr=args.lr,
            temperature=args.temperature,
            fractions=args.fractions,
            root_dir=args.root_dir,
            device=device,
            force_retrain=args.force_retrain,
            skip_scratch=args.skip_scratch,
            max_batches_per_epoch=args.max_batches_per_epoch,
        )

    if args.stage in ("s4", "all"):
        s4_output = run_s4_ewc(
            seed=args.seed,
            simclr_epochs=args.simclr_epochs,
            task1_epochs=args.task1_epochs,
            task2_epochs=args.task2_epochs,
            batch_size=args.s4_batch_size,
            simclr_batch_size=args.batch_size,
            lr=args.lr,
            temperature=args.temperature,
            ewc_lambdas=args.ewc_lambdas,
            replay_size=args.replay_size,
            root_dir=args.root_dir,
            device=device,
            force_retrain=args.force_retrain,
            max_batches_per_epoch=args.max_batches_per_epoch,
            fisher_normalization=args.fisher_normalization,
            include_ewc_replay=not args.skip_ewc_replay,
        )

    if s1_output is not None and s4_output is not None:
        update_cross_effects_table(s1_output, s4_output)
    elif s1_output is not None:
        print("[Member C] S4 output not produced; cross-effects table left unchanged.")
    elif s4_output is not None:
        print("[Member C] S1 output not produced; cross-effects table left unchanged.")


if __name__ == "__main__":
    main()
