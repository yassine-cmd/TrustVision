"""
TrustVision - Multi-seed confidence intervals for S1 (scripts/run_s1_multiseed.py)
==================================================================================
Runs the S1 label-efficiency protocol on the three locked splits from
``configs/splits.json`` (seeds 42, 123, 999) and reports mean / standard
deviation / 95 % Student-t confidence intervals for the 1 % and 5 % budgets.

Why retrain SimCLR per seed?
----------------------------
The three splits are different (264 images of seed 42's train set belong to
seed 123's test set). Reusing the seed-42 encoder would leak unlabeled test
images into the representation learning of the other seeds, so the encoder is
pre-trained independently on each seed's *unlabeled* training split. This is
the only defensible protocol for a multi-seed confidence interval.

Outputs
-------
* ``results/s1_multiseed_ci.json`` : per-seed metrics + aggregates + paired delta.
* ``results/figures/s1_multiseed_ci.png`` : accuracy with 95 % CI error bars.
* ``checkpoints/simclr_backbone_6cls_seed{123,999}.pt`` : encoders for
  reproducibility (seed 42 reuses the official ``simclr_backbone_6cls.pt``).

Usage
-----
    python -m scripts.run_s1_multiseed
    python -m scripts.run_s1_multiseed --seeds 42 123 999 --fractions 0.01 0.05
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import get_dataloaders
from src.run_member_c import train_supervised_fraction
from src.simclr import evaluate_simclr_label_fractions, simclr_pretrain

# Two-sided 95 % Student-t critical values, keyed by degrees of freedom.
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def fraction_key(fraction: float) -> str:
    return f"{int(round(fraction * 100))}pct"


def t_critical(n: int) -> float:
    df = max(1, n - 1)
    return T_CRIT_95.get(df, 1.96)


def summarize(values: Sequence[float]) -> Dict[str, Any]:
    """Mean, sample std, and 95 % t-based CI for a small seed sample."""
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(clean)
    if n == 0:
        return {"values": [], "n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    avg = mean(clean)
    if n == 1:
        return {"values": clean, "n": 1, "mean": avg, "std": 0.0, "ci95_low": avg, "ci95_high": avg}
    sd = stdev(clean)
    half = t_critical(n) * sd / math.sqrt(n)
    return {
        "values": [round(v, 6) for v in clean],
        "n": n,
        "mean": round(avg, 6),
        "std": round(sd, 6),
        "ci95_low": round(avg - half, 6),
        "ci95_high": round(avg + half, 6),
    }


def calibrated_or_raw(metrics: Dict[str, Any], key: str) -> float:
    """Prefers the calibrated value when the no-harm guard accepted the fit."""
    calibrated = metrics.get("calibrated") or {}
    if calibrated.get("calibration_accepted", True) and calibrated.get(key) is not None:
        return float(calibrated[key])
    return float(metrics[key])


def run_seed(
    seed: int,
    fractions: Sequence[float],
    simclr_epochs: int,
    scratch_epochs: int,
    batch_size: int,
    scratch_batch_size: int,
    lr: float,
    temperature: float,
    root_dir: str,
    force_pretrain: bool,
    device: torch.device,
) -> Dict[str, Any]:
    """Runs SimCLR pre-training (if needed) and both 1/5 % evaluations."""
    if seed == 42:
        backbone_path = Path("checkpoints/simclr_backbone_6cls.pt")
    else:
        backbone_path = Path(f"checkpoints/simclr_backbone_6cls_seed{seed}.pt")

    if force_pretrain or not backbone_path.exists():
        print(f"[multi-seed {seed}] Pre-training SimCLR ({simclr_epochs} epochs)...")
        train_loader, _, _, _ = get_dataloaders(
            root_dir=root_dir,
            batch_size=batch_size,
            seed=seed,
            img_size=96,
            label_fraction=1.0,
            num_classes=6,
            auto_download=True,
        )
        simclr_pretrain(
            train_loader,
            epochs=simclr_epochs,
            lr=lr,
            temperature=temperature,
            device=device,
            save_path=backbone_path,
            num_classes=6,
            verbose=True,
        )
    else:
        print(f"[multi-seed {seed}] Reusing encoder: {backbone_path}")

    print(f"[multi-seed {seed}] Probing {', '.join(fraction_key(f) for f in fractions)}...")
    simclr_metrics = evaluate_simclr_label_fractions(
        backbone_path,
        fractions=fractions,
        seed=seed,
        batch_size=batch_size,
        device=device,
        num_classes=6,
        root_dir=root_dir,
        calibrate=True,
    )

    scratch_metrics: Dict[str, Dict[str, Any]] = {}
    for fraction in fractions:
        print(f"[multi-seed {seed}] Scratch CNN at {fraction_key(fraction)}...")
        _, history, metrics = train_supervised_fraction(
            fraction,
            seed=seed,
            epochs=scratch_epochs,
            batch_size=scratch_batch_size,
            lr=lr,
            device=device,
            root_dir=root_dir,
            verbose=False,
        )
        metrics["history"] = history
        scratch_metrics[fraction_key(fraction)] = metrics

    return {
        "backbone": str(backbone_path.as_posix()),
        "simclr": simclr_metrics,
        "supervised_from_scratch": scratch_metrics,
    }


def collect_aggregates(
    per_seed: Dict[str, Any],
    fractions: Sequence[float],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    aggregates: Dict[str, Any] = {}
    for fraction in fractions:
        key = fraction_key(fraction)
        aggregates[key] = {}
        for method in ("simclr", "scratch"):
            source = "simclr" if method == "simclr" else "supervised_from_scratch"
            rows = [per_seed[str(seed)][source][key] for seed in seeds]
            aggregates[key][method] = {
                "acc": summarize([row["acc"] for row in rows]),
                "macro_f1": summarize([row["macro_f1"] for row in rows]),
                "ece": summarize([calibrated_or_raw(row, "ece") for row in rows]),
                "nll": summarize([calibrated_or_raw(row, "nll") for row in rows]),
                "brier": summarize([calibrated_or_raw(row, "brier") for row in rows]),
                "rob_acc_003": summarize([row["rob_acc_003"] for row in rows]),
            }
    return aggregates


def collect_paired_delta(
    per_seed: Dict[str, Any],
    fractions: Sequence[float],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    """SimCLR minus scratch on the same seed/split (paired comparison)."""
    paired: Dict[str, Any] = {}
    for fraction in fractions:
        key = fraction_key(fraction)
        deltas = [
            per_seed[str(seed)]["simclr"][key]["acc"]
            - per_seed[str(seed)]["supervised_from_scratch"][key]["acc"]
            for seed in seeds
        ]
        paired[key] = {"acc_delta": summarize(deltas)}
    return paired


def plot_ci(aggregates: Dict[str, Any], fractions: Sequence[float], out_path: Path) -> None:
    keys = [fraction_key(f) for f in fractions]
    labels = [k.replace("pct", " %") for k in keys]
    x = np.arange(len(keys))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=200)
    for offset, method, color, label in (
        (-width / 2, "simclr", "#1f77b4", "SimCLR (frozen probe)"),
        (width / 2, "scratch", "#d62728", "CNN from scratch"),
    ):
        means = [aggregates[k][method]["acc"]["mean"] * 100 for k in keys]
        lows = [aggregates[k][method]["acc"]["ci95_low"] * 100 for k in keys]
        highs = [aggregates[k][method]["acc"]["ci95_high"] * 100 for k in keys]
        errors = [[m - lo for m, lo in zip(means, lows)], [hi - m for m, hi in zip(means, highs)]]
        ax.bar(x + offset, means, width, yerr=errors, capsize=5, color=color, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("S1 - Multi-seed label efficiency (mean, 95 % CI, 3 seeds)", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[multi-seed] Figure saved to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="S1 multi-seed confidence intervals (1 % / 5 %).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.05])
    parser.add_argument("--simclr-epochs", type=int, default=20)
    parser.add_argument("--scratch-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--scratch-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--root-dir", default="data/NEU-CLS")
    parser.add_argument("--out", default="results/s1_multiseed_ci.json")
    parser.add_argument("--figure", default="results/figures/s1_multiseed_ci.png")
    parser.add_argument("--force-pretrain", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[multi-seed] Device: {device}")
    start = time.time()

    per_seed: Dict[str, Any] = {}
    for seed in args.seeds:
        per_seed[str(seed)] = run_seed(
            seed,
            args.fractions,
            args.simclr_epochs,
            args.scratch_epochs,
            args.batch_size,
            args.scratch_batch_size,
            args.lr,
            args.temperature,
            args.root_dir,
            args.force_pretrain,
            device,
        )

    aggregates = collect_aggregates(per_seed, args.fractions, args.seeds)
    paired = collect_paired_delta(per_seed, args.fractions, args.seeds)

    reference_check: Dict[str, Any] = {"checked": False}
    reference_path = Path("results/s1_simclr.json")
    if "42" in per_seed and reference_path.exists():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        details = {}
        ok = True
        for fraction in args.fractions:
            key = fraction_key(fraction)
            ref_acc = reference["simclr"]["label_fractions"].get(key, {}).get("acc")
            got_acc = per_seed["42"]["simclr"][key]["acc"]
            match = ref_acc is not None and abs(got_acc - ref_acc) <= 1e-3
            ok &= bool(match)
            details[key] = {
                "json_acc": ref_acc,
                "rerun_acc": round(float(got_acc), 6),
                "match": bool(match),
            }
        reference_check = {"checked": True, "seed42_matches_official_json": bool(ok), "details": details}

    elapsed = round(time.time() - start, 2)
    payload = {
        "stage": "S1_MULTISEED_LABEL_EFFICIENCY",
        "protocol": (
            "Independent SimCLR encoder per locked split (no cross-seed leakage), "
            "frozen standardized logistic probes at 1% and 5% labels, calibrated "
            "on the shared 10% calibration split with the no-harm guard; supervised "
            "CNN from scratch on the exact same label subsets."
        ),
        "seeds": list(args.seeds),
        "fractions": [fraction_key(f) for f in args.fractions],
        "ci_method": "Mean +/- t(0.975, n-1) * s / sqrt(n); n=3 seeds, df=2 (t=4.303).",
        "runtime_seconds": elapsed,
        "backbones": {seed: per_seed[str(seed)]["backbone"] for seed in args.seeds},
        "per_seed": per_seed,
        "aggregates": aggregates,
        "paired_delta_simclr_minus_scratch": paired,
        "seed42_reference_check": reference_check,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[multi-seed] Results saved to {out}")

    plot_ci(aggregates, args.fractions, Path(args.figure))
    print(f"[multi-seed] Runtime: {elapsed:.1f} s")

    for key in payload["fractions"]:
        sim = aggregates[key]["simclr"]["acc"]
        scr = aggregates[key]["scratch"]["acc"]
        print(
            f"  {key}: SimCLR {sim['mean']*100:.2f}% "
            f"[{sim['ci95_low']*100:.2f}, {sim['ci95_high']*100:.2f}] | "
            f"scratch {scr['mean']*100:.2f}% "
            f"[{scr['ci95_low']*100:.2f}, {scr['ci95_high']*100:.2f}]"
        )

    if reference_check.get("checked") and not reference_check["seed42_matches_official_json"]:
        print("[multi-seed] WARNING: seed 42 did not reproduce the official S1 JSON.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
