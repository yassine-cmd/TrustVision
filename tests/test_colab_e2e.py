"""
TrustVision - Colab Notebooks & Artifacts Manifest Integrity (tests/test_colab_e2e.py)
======================================================================================
Validates:
1. Python AST syntax integrity and path sanitization across all repository notebooks.
2. Cross-effects table consistency and deliverables manifest (checkpoints, metrics, configs).
3. Clean zero-side-effect importability of all src/ pipeline modules.
"""

import ast
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. Notebook AST Syntax & Clean Path Hygiene
# ---------------------------------------------------------------------------

@pytest.mark.smoke
@pytest.mark.parametrize("nb_name", [
    "TrustVision_Master.ipynb",
    "exp_A.ipynb",
    "exp_B.ipynb",
    "exp_C.ipynb",
])
def test_notebook_syntax_and_clean_paths(nb_name):
    nb_path = ROOT / "notebooks" / nb_name
    assert nb_path.exists(), f"Notebook {nb_name} must exist in notebooks/."

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data, f"{nb_name} must have 'cells' array."
    code_cells = [c for c in nb_data["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) > 0, f"{nb_name} must contain code cells."

    forbidden_patterns = [
        "C:\\Users\\",
        "/content/drive/MyDrive/personal",
    ]

    for idx, cell in enumerate(code_cells):
        raw_source = "".join(cell.get("source", []))

        # Check for forbidden hardcoded personal paths
        for pattern in forbidden_patterns:
            assert pattern not in raw_source, (
                f"Forbidden personal path '{pattern}' found in {nb_name} cell #{idx}."
            )

        # Filter Colab / IPython magic lines before AST parsing
        valid_python_lines = []
        for line in raw_source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("%", "!", "?")):
                continue
            valid_python_lines.append(line)

        code_to_parse = "\n".join(valid_python_lines)
        if code_to_parse.strip():
            try:
                ast.parse(code_to_parse, filename=f"{nb_name}#cell_{idx}")
            except SyntaxError as err:
                pytest.fail(f"Syntax error in {nb_name} cell #{idx}: {err}")


# ---------------------------------------------------------------------------
# 2. Production Artifacts Manifest & Cross-Effects Parity
# ---------------------------------------------------------------------------

@pytest.mark.contract
def test_artifact_manifest_and_checkpoints():
    required_checkpoints = [
        "cnn_base_seed42.pt",
        "cnn_adv_seed42.pt",
        "ood_base_seed42.pt",
    ]
    for ckpt in required_checkpoints:
        p = ROOT / "checkpoints" / ckpt
        assert p.exists(), f"Required model checkpoint {ckpt} is missing from checkpoints/."
        assert p.stat().st_size > 1000, f"Checkpoint {ckpt} is empty or corrupted."

    required_results = [
        "cross_effects_table.json",
        "s0_clip.json",
        "s1_simclr.json",
        "s4_ewc.json",
    ]
    for res in required_results:
        p = ROOT / "results" / res
        assert p.exists(), f"Required experimental result {res} is missing from results/."
        assert p.stat().st_size > 50, f"Result file {res} is empty."


@pytest.mark.contract
def test_cross_effects_table_schema_and_values():
    table_path = ROOT / "results/cross_effects_table.json"
    with open(table_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "title" in data
    assert "split_distribution" in data
    assert "rows" in data

    stages_present = [r["stage"] for r in data["rows"]]
    expected_stages = [
        "S0_CLIP_LinearProbe",
        "S1_SimCLR",
        "S2_CNN_Base_Calibrated",
        "S3_CNN_Adversarial_Defense",
        "S4_EWC_Continual",
    ]
    for exp_st in expected_stages:
        assert exp_st in stages_present, f"Stage {exp_st} must be present in cross-effects table."

    # Validate S0 reference metrics
    s0_row = next(r for r in data["rows"] if "S0" in r["stage"])
    assert s0_row["acc"] > 0.95
    assert s0_row["ece"] < 0.05

    # Validate S3 robustness jump
    s3_row = next(r for r in data["rows"] if "S3" in r["stage"])
    assert s3_row["rob_acc_003"] > 0.70, "S3 robust accuracy must exceed 70%."


# ---------------------------------------------------------------------------
# 3. Source Modules Import Contract
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_all_src_pipeline_modules_importable():
    import src.data as d
    import src.model as m
    import src.eval as e
    import src.clip_attn as ca
    import src.xai_cal_adv as xca
    import src.simclr as s
    import src.ewc as ew
    import src.ood as o
    import src.gradio_app as ga

    assert len(d.NEU_CLASSES) == 6
    assert hasattr(m, "TrustVisionCNN")
    assert hasattr(e, "evaluate_all")
    assert hasattr(ca, "extract_cls_patch_attention")
    assert hasattr(xca, "GradCAM")
    assert hasattr(s, "nt_xent_loss")
    assert hasattr(ew, "compute_fisher_diagonal")
    assert hasattr(o, "TrustVisionOODPipeline")
    assert hasattr(ga, "build_demo")
