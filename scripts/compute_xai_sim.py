"""
TrustVision - Compute the missing S1/S4 XAI similarity (scripts/compute_xai_sim.py)
===================================================================================
The cross-effects matrix requires an ``xai_sim`` value for every stage that has
a CNN: agreement between Grad-CAM and the CLIP ViT CLS->patch attention
reference. S2 was already measured at 0.7663; S1 and S4 were null.

This script recomputes the two missing values with the **exact same protocol**
as ``src/run_experiments.py`` for S2 (no retraining, no metric fabrication):

* reference maps: ``results/clip_attn_test.npy`` -> mean over the 12 layers;
* saliency maps: Grad-CAM on the 360 test images at 96x96;
* similarity: ``src.eval.compute_xai_similarity(..., method="cosine")``;
* aggregate: mean over the 360 test images.

It then writes the values into ``results/s1_simclr.json``,
``results/s4_ewc.json`` and ``results/cross_effects_table.json``.

Usage
-----
    python -m scripts.compute_xai_sim
    python -m scripts.compute_xai_sim --check-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from src.data import get_dataloaders
from src.eval import compute_xai_similarity
from src.model import TrustVisionCNN
from src.xai_cal_adv import GradCAM

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    "Grad-CAM (96x96) vs CLIP ViT CLS-to-patch attention averaged over the 12 "
    "layers (results/clip_attn_test.npy), cosine similarity, mean over the 360 "
    "shared test images. Same protocol as the S2 value 0.7663."
)


def compute_xai_sim(
    checkpoint: Path,
    test_loader,
    reference_maps: np.ndarray,
    device: torch.device,
) -> float:
    model = TrustVisionCNN(num_classes=6).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    explainer = GradCAM(model)
    cams = []
    for images, _ in test_loader:
        cams.append(explainer(images.to(device)))
    explainer.remove_hooks()
    cams = np.concatenate(cams, axis=0)
    if len(cams) != len(reference_maps):
        raise ValueError(f"Map count mismatch: {len(cams)} vs {len(reference_maps)}")

    similarities = [
        compute_xai_similarity(cams[i], reference_maps[i], method="cosine")
        for i in range(len(cams))
    ]
    return round(float(np.mean(similarities)), 4)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def update_outputs(
    s1_value: float,
    s4_value: float,
    s1_path: Path,
    s4_path: Path,
    table_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    s1 = json.loads(s1_path.read_text(encoding="utf-8"))
    s4 = json.loads(s4_path.read_text(encoding="utf-8"))
    table = json.loads(table_path.read_text(encoding="utf-8"))

    s1["xai_sim"] = s1_value
    s1["xai_sim_protocol"] = PROTOCOL
    s1["xai_sim_checkpoint"] = "checkpoints/s1_simclr_probe_100pct.pt"
    s1["simclr"]["label_fractions"]["100pct"]["xai_sim"] = s1_value

    s4["xai_sim"] = s4_value
    s4["xai_sim_protocol"] = PROTOCOL
    s4["xai_sim_checkpoint"] = s4["best_ewc"]["checkpoint"]
    s4["best_ewc"]["metrics"]["xai_sim"] = s4_value
    best_key = s4["best_ewc"]["strategy"]
    if best_key in s4.get("strategies", {}):
        s4["strategies"][best_key]["metrics"]["xai_sim"] = s4_value

    for row in table.get("rows", []):
        if row["stage"] == "S1_SimCLR":
            row["xai_sim"] = s1_value
        elif row["stage"] == "S4_EWC_Continual":
            row["xai_sim"] = s4_value
    table.setdefault("member_c", {})["xai_sim"] = {
        "S1_SimCLR": s1_value,
        "S4_EWC_Continual": s4_value,
        "protocol": PROTOCOL,
    }

    write_json(s1_path, s1)
    write_json(s4_path, s4)
    write_json(table_path, table)
    return s1, s4, table


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute and store the S1/S4 XAI similarity.")
    parser.add_argument("--check-only", action="store_true", help="Compute but do not write JSON files.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--root-dir", default="data/NEU-CLS")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[xai-sim] Device: {device}")

    _, _, test_loader, _ = get_dataloaders(
        root_dir=str(ROOT / args.root_dir),
        batch_size=args.batch_size,
        seed=42,
        img_size=96,
        label_fraction=1.0,
        num_classes=6,
        auto_download=False,
    )

    reference_path = ROOT / "results/clip_attn_test.npy"
    reference = np.load(reference_path)
    reference = reference.mean(axis=1)  # (360, 12, 7, 7) -> (360, 7, 7), as in run_experiments.py
    print(f"[xai-sim] Reference maps: {reference.shape}")

    targets = {
        "S1_SimCLR": ROOT / "checkpoints/s1_simclr_probe_100pct.pt",
        "S4_EWC_Continual": ROOT / "checkpoints/ewc_plus_replay_seed42.pt",
    }
    values: Dict[str, float] = {}
    for label, checkpoint in targets.items():
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        print(f"[xai-sim] Scoring {label} ({checkpoint.name})...")
        values[label] = compute_xai_sim(checkpoint, test_loader, reference, device)
        print(f"  {label}: xai_sim = {values[label]:.4f}")

    summary = {
        "protocol": PROTOCOL,
        "reference_file": "results/clip_attn_test.npy",
        "checkpoints": {label: str(path.relative_to(ROOT).as_posix()) for label, path in targets.items()},
        "values": values,
        "s2_reference": 0.7663,
    }
    if args.check_only:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    s1, s4, table = update_outputs(
        values["S1_SimCLR"],
        values["S4_EWC_Continual"],
        ROOT / "results/s1_simclr.json",
        ROOT / "results/s4_ewc.json",
        ROOT / "results/cross_effects_table.json",
    )
    write_json(ROOT / "results/xai_sim.json", summary)

    print("[xai-sim] Updated results/s1_simclr.json, results/s4_ewc.json, results/cross_effects_table.json")
    print(f"  S1 row  acc={s1['simclr']['label_fractions']['100pct']['acc']:.4f} xai={s1['xai_sim']:.4f}")
    print(f"  S4 row  acc={s4['best_ewc']['metrics']['acc']:.4f} xai={s4['xai_sim']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
