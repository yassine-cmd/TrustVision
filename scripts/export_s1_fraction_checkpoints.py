"""
TrustVision - Backfill S1 fraction checkpoints (scripts/export_s1_fraction_checkpoints.py)
=========================================================================================
Exports the missing 1 % / 5 % / 10 % S1 checkpoints for the demo notebook:

* ``checkpoints/s1_simclr_probe_{1,5,10}pct.pt`` : frozen SimCLR backbone with the
  standardized logistic probe folded into ``model.fc`` (same contract as the
  100 % probe checkpoint, directly loadable by ``src.eval.load_model_checkpoint``).
* ``checkpoints/s1_scratch_{1,5,10}pct.pt`` : the supervised CNN trained from
  scratch on the same label budgets.

The official 100 % checkpoints and ``results/s1_simclr.json`` are never written
by this script. Every exported checkpoint is re-evaluated on the shared test
split and compared with the JSON metrics that were recorded during the official
run, so the backfill is auditable.

Usage
-----
    python -m scripts.export_s1_fraction_checkpoints
    python -m scripts.export_s1_fraction_checkpoints --fractions 0.01 0.05 0.10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from src.data import get_dataloaders
from src.eval import evaluate_all
from src.model import TrustVisionCNN
from src.run_member_c import train_supervised_fraction
from src.simclr import (
    build_probe_module,
    extract_features,
    fold_probe_into_backbone,
)


def fraction_key(fraction: float) -> str:
    return f"{int(round(fraction * 100))}pct"


def export_probe(
    fraction: float,
    backbone_path: Path,
    out_path: Path,
    seed: int,
    root_dir: str,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Fits the frozen linear probe and exports it as a plain TrustVisionCNN."""
    backbone = TrustVisionCNN(num_classes=6)
    state = torch.load(backbone_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    backbone.load_state_dict(state, strict=False)
    backbone.to(device).eval()

    train_loader, _, test_loader, _ = get_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        seed=seed,
        img_size=96,
        label_fraction=fraction,
        num_classes=6,
        auto_download=True,
    )
    features, labels = extract_features(backbone, train_loader, device=device)
    _, clf, scaler = build_probe_module(
        backbone, features, labels, num_classes=6, seed=seed
    )
    folded = fold_probe_into_backbone(backbone, clf, scaler)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(folded.state_dict(), out_path)

    metrics = evaluate_all(out_path, test_loader, device=device)
    print(
        f"  [probe {fraction_key(fraction):>6s}] exported {out_path.name} "
        f"({len(labels)} labels) -> acc={metrics['acc']:.4f}"
    )
    return metrics


def export_scratch(
    fraction: float,
    out_path: Path,
    seed: int,
    root_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> Dict[str, Any]:
    """Trains the from-scratch CNN and exports its state dict."""
    model, _, metrics = train_supervised_fraction(
        fraction,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        root_dir=root_dir,
        verbose=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(
        f"  [scratch {fraction_key(fraction):>5s}] exported {out_path.name} "
        f"-> acc={metrics['acc']:.4f}"
    )
    return metrics


def compare_to_json(
    label: str,
    fraction: float,
    measured: Dict[str, Any],
    reference: Dict[str, Any],
    tolerance: float = 1e-4,
) -> bool:
    """Prints a pass/fail comparison against the official S1 JSON."""
    key = fraction_key(fraction)
    ref = reference.get("label_fractions", {}).get(key, {})
    ref_acc = ref.get("acc")
    ok = ref_acc is not None and abs(measured["acc"] - ref_acc) <= tolerance
    print(
        f"  [verify {label:>7s} {key:>6s}] measured={measured['acc']:.4f} "
        f"json={ref_acc if ref_acc is not None else 'n/a'} -> "
        f"{'OK' if ok else 'MISMATCH'}"
    )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the 1/5/10 % S1 checkpoints without touching the 100 % official artefacts."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    parser.add_argument("--root-dir", default="data/NEU-CLS")
    parser.add_argument("--backbone", default="checkpoints/simclr_backbone_6cls.pt")
    parser.add_argument("--results", default="results/s1_simclr.json")
    parser.add_argument("--probe-pattern", default="checkpoints/s1_simclr_probe_{fraction}pct.pt")
    parser.add_argument("--scratch-pattern", default="checkpoints/s1_scratch_{fraction}pct.pt")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for probe feature extraction.")
    parser.add_argument("--scratch-batch-size", type=int, default=32, help="Batch size used by the official scratch baseline.")
    parser.add_argument("--scratch-epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--skip-scratch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[S1 export] Device: {device}")

    reference = json.loads(Path(args.results).read_text(encoding="utf-8"))
    backbone_path = Path(args.backbone)
    if not backbone_path.exists():
        raise FileNotFoundError(f"SimCLR backbone not found: {backbone_path}")

    all_ok = True
    report: Dict[str, Any] = {
        "stage": "S1_CHECKPOINT_METRICS",
        "seed": args.seed,
        "protocol": (
            "Checkpoints exported from the official SimCLR backbone (probes) or "
            "retrained with the official scratch recipe (15 epochs, batch 32). "
            "Every checkpoint is re-evaluated on the shared test split; the "
            "official JSON accuracy is kept untouched as the reference."
        ),
        "checkpoints": {},
    }
    for fraction in args.fractions:
        if abs(fraction - 1.0) < 1e-6:
            print(f"[S1 export] Skipping {fraction_key(fraction)}: official 100 % checkpoints are kept as-is.")
            continue
        key = fraction_key(fraction)

        if not args.skip_probes:
            probe_path = Path(args.probe_pattern.format(fraction=int(round(fraction * 100))))
            if probe_path.exists() and not args.overwrite:
                print(f"[S1 export] Reusing existing probe checkpoint: {probe_path}")
                _, _, test_loader, _ = get_dataloaders(
                    root_dir=args.root_dir,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    img_size=96,
                    label_fraction=fraction,
                    num_classes=6,
                    auto_download=True,
                )
                metrics = evaluate_all(probe_path, test_loader, device=device)
            else:
                metrics = export_probe(
                    fraction,
                    backbone_path,
                    probe_path,
                    args.seed,
                    args.root_dir,
                    args.batch_size,
                    device,
                )
            all_ok &= compare_to_json(
                "probe", fraction, metrics, reference["simclr"]
            )
            report["checkpoints"].setdefault(key, {})["probe"] = {
                "path": str(probe_path.as_posix()),
                "acc": round(float(metrics["acc"]), 6),
                "official_json_acc": reference["simclr"]["label_fractions"].get(key, {}).get("acc"),
            }

        if not args.skip_scratch:
            scratch_path = Path(
                args.scratch_pattern.format(fraction=int(round(fraction * 100)))
            )
            if scratch_path.exists() and not args.overwrite:
                print(f"[S1 export] Reusing existing scratch checkpoint: {scratch_path}")
                _, _, test_loader, _ = get_dataloaders(
                    root_dir=args.root_dir,
                    batch_size=args.scratch_batch_size,
                    seed=args.seed,
                    img_size=96,
                    label_fraction=fraction,
                    num_classes=6,
                    auto_download=True,
                )
                metrics = evaluate_all(scratch_path, test_loader, device=device)
            else:
                metrics = export_scratch(
                    fraction,
                    scratch_path,
                    args.seed,
                    args.root_dir,
                    args.scratch_epochs,
                    args.scratch_batch_size,
                    args.lr,
                    device,
                )
            all_ok &= compare_to_json(
                "scratch", fraction, metrics, reference["supervised_from_scratch"]
            )
            report["checkpoints"].setdefault(key, {})["scratch"] = {
                "path": str(scratch_path.as_posix()),
                "acc": round(float(metrics["acc"]), 6),
                "official_json_acc": reference["supervised_from_scratch"]["label_fractions"].get(key, {}).get("acc"),
            }

    report["all_checks_passed"] = bool(all_ok)
    report_path = Path("results/s1_checkpoint_metrics.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[S1 export] Checkpoint metrics saved to {report_path}")
    print(f"[S1 export] All checks passed: {all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
