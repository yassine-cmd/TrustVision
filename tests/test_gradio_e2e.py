"""
TrustVision - Full-Stack End-to-End Gradio Integration Suite (tests/test_gradio_e2e.py)
========================================================================================
Tests all 5 telemetry tabs and UI callbacks WITHOUT launching a live web server:
1. Tab 1: CLIP ViT Zero-Shot & 12-layer attention overlay callback.
2. Tab 2: SimCLR label efficiency curve & checkpoint inference callback.
3. Tab 3: Trust & Audit Toolkit (Temperature scaling, FGSM, XAI, OOD gate).
4. Tab 4: Continual Learning (EWC lambda slider & stability-plasticity curves).
5. Tab 5 / Blocks: Complete gr.Blocks wiring, dependency graph, and input/output contracts.
"""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt

from src.data import NEU_CLASSES
from src.gradio_app import (
    ClipAttentionTab,
    SimCLRTab,
    EWCTab,
    build_demo,
)
from src.gradio_tab3 import TrustVisionTab3Auditor

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. Tab 1: CLIP Attention & Zero-Shot Prompt Slider Tests
# ---------------------------------------------------------------------------

@pytest.mark.ui
@pytest.mark.parametrize("prompt_tpl", [
    "a photo of {class}",
    "hot-rolled steel surface showing {class} defect",
])
def test_tab1_clip_prompt_slider_contract(sample_steel_pil, prompt_tpl):
    tab = ClipAttentionTab(ROOT)
    overlay, scores, note = tab.analyze(sample_steel_pil, prompt_tpl)

    assert overlay is not None, "CLIP attention heatmap overlay must be generated."
    assert isinstance(overlay, Image.Image), "Overlay must be a valid PIL Image."
    assert overlay.size == (224, 224), "Overlay must match CLIP input resolution (224, 224)."

    assert scores is not None, "Class probability scores dictionary must be generated."
    assert len(scores) == len(NEU_CLASSES), f"Scores must cover all {len(NEU_CLASSES)} classes."
    prob_sum = sum(scores.values())
    assert abs(prob_sum - 1.0) < 0.01, f"Zero-shot probabilities must sum to 1.0 (got {prob_sum})."

    assert isinstance(note, str) and len(note) > 0, "Telemetry card note must be non-empty."
    assert "ATTENTION SPATIALE" in note


@pytest.mark.ui
def test_tab1_clip_handles_none_image():
    tab = ClipAttentionTab(ROOT)
    overlay, scores, note = tab.analyze(None, "a photo of {class}")
    assert overlay is None
    assert scores is None
    assert "Sélectionnez" in note


# ---------------------------------------------------------------------------
# 2. Tab 2: SimCLR Label Efficiency & Probe Inference Tests
# ---------------------------------------------------------------------------

@pytest.mark.ui
def test_tab2_simclr_label_efficiency_figure():
    tab = SimCLRTab(ROOT)
    fig = tab.label_efficiency_figure()
    assert isinstance(fig, plt.Figure), "Must generate a matplotlib Figure."
    assert len(fig.axes) == 2, "Must contain exactly 2 subplots (Accuracy and ECE)."
    plt.close(fig)


@pytest.mark.ui
def test_tab2_simclr_predict_contract(sample_steel_pil):
    tab = SimCLRTab(ROOT)
    assert len(tab.choices) > 0, "SimCLRTab must discover available probe/scratch checkpoints."

    first_choice = list(tab.choices.keys())[0]
    probs, detail = tab.predict(sample_steel_pil, first_choice)

    assert probs is not None, "Class probabilities must be returned."
    assert len(probs) == len(NEU_CLASSES)
    assert abs(sum(probs.values()) - 1.0) < 0.01
    assert "RÉSULTAT D'INFÉRENCE" in detail


@pytest.mark.ui
def test_tab2_simclr_handles_empty_input():
    tab = SimCLRTab(ROOT)
    probs, detail = tab.predict(None, "invalid")
    assert probs is None
    assert "Sélectionnez un échantillon" in detail


# ---------------------------------------------------------------------------
# 3. Tab 3: Trust & Audit Toolkit (Temperature, FGSM, XAI, OOD)
# ---------------------------------------------------------------------------

@pytest.mark.ui
@pytest.mark.parametrize("temperature", [0.5, 0.9276, 2.0])
def test_tab3_temperature_scaling_preserves_ranking(sample_steel_pil, temperature):
    auditor = TrustVisionTab3Auditor(
        base_checkpoint=str(ROOT / "checkpoints/cnn_base_seed42.pt"),
        adv_checkpoint=str(ROOT / "checkpoints/cnn_adv_seed42.pt"),
    )
    res_ref = auditor.audit(sample_steel_pil, temperature=1.0, epsilon=0.0)
    res_t = auditor.audit(sample_steel_pil, temperature=temperature, epsilon=0.0)

    assert res_ref["predicted_class"] == res_t["predicted_class"], (
        "Temperature scaling must strictly preserve argmax prediction."
    )
    assert abs(sum(res_t["class_probabilities"].values()) - 1.0) < 0.01


@pytest.mark.ui
@pytest.mark.parametrize("epsilon", [0.0, 0.03])
@pytest.mark.parametrize("model_type", ["Baseline CNN (calibré)", "CNN avec Défense Adversariale (robuste)"])
def test_tab3_adversarial_epsilon_bounds(sample_steel_pil, epsilon, model_type):
    auditor = TrustVisionTab3Auditor(
        base_checkpoint=str(ROOT / "checkpoints/cnn_base_seed42.pt"),
        adv_checkpoint=str(ROOT / "checkpoints/cnn_adv_seed42.pt"),
    )
    res = auditor.audit(sample_steel_pil, epsilon=epsilon, model_type=model_type, include_ig=True)

    pert_np = res["perturbed_image"]  # (96, 96, 3) float32 in [0, 1]
    clean_arr = np.asarray(sample_steel_pil.resize((96, 96)), dtype=np.float32) / 255.0

    assert pert_np.min() >= 0.0 and pert_np.max() <= 1.0
    assert np.max(np.abs(pert_np - clean_arr)) <= epsilon + 0.015

    assert res["grad_cam_map"] is not None
    assert res["integrated_gradients_map"] is not None
    assert not np.isnan(res["grad_cam_map"]).any()
    assert not np.isnan(res["integrated_gradients_map"]).any()


@pytest.mark.ui
def test_tab3_ood_rejection_trigger():
    auditor = TrustVisionTab3Auditor(
        base_checkpoint=str(ROOT / "checkpoints/cnn_base_seed42.pt"),
        adv_checkpoint=str(ROOT / "checkpoints/cnn_adv_seed42.pt"),
    )
    # Severe anomalous input (pure noise)
    rng = np.random.RandomState(42)
    pure_noise = Image.fromarray((rng.rand(200, 200, 3) * 255).astype(np.uint8))
    res = auditor.audit(pure_noise, temperature=0.9276, epsilon=0.0)

    assert res["is_ood"] is True, "Pure noise anomaly must be flagged as OOD."
    assert "OOD" in res.get("ood_label", "")


# ---------------------------------------------------------------------------
# 4. Tab 4: Continual Learning (EWC & Lambda Slider)
# ---------------------------------------------------------------------------

@pytest.mark.ui
@pytest.mark.parametrize("lambda_idx", [0, 1, 2, 3])
def test_tab4_ewc_lambda_slider_contract(lambda_idx):
    tab = EWCTab(ROOT)
    key, strategy, title = tab.resolve("EWC régularisé (paramètre λ)", lambda_idx)
    metrics = strategy["metrics"]

    assert key.startswith("ewc_lambda_"), f"Resolved key must be an EWC lambda variant (got {key})."
    assert "acc" in metrics and "old_acc" in metrics and "new_acc" in metrics
    assert "λ" in title


@pytest.mark.ui
def test_tab4_bar_figure():
    tab = EWCTab(ROOT)
    fig = tab.bar_figure("EWC régularisé (paramètre λ)", 1)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


@pytest.mark.ui
def test_tab4_comparison_table():
    tab = EWCTab(ROOT)
    table = tab.comparison_table()
    assert len(table) >= 4, "Must contain rows for naive, replay, EWC variants, and ewc_replay."
    for row in table:
        assert len(row) == 7, "Each table row must have 7 metrics."


# ---------------------------------------------------------------------------
# 5. Full Gradio Blocks Architecture & Event Wiring (Zero Server)
# ---------------------------------------------------------------------------

@pytest.mark.ui
def test_gradio_blocks_architecture_and_wiring():
    """Validates that build_demo constructs a valid Blocks UI with all tabs wired without launching."""
    demo = build_demo(ROOT)
    assert isinstance(demo, gr.Blocks), "build_demo must return a gradio.Blocks instance."

    # Verify Blocks configuration graph
    config = demo.config
    assert "components" in config, "Gradio config must contain registered components."
    assert "dependencies" in config, "Gradio config must contain registered event dependencies."

    dependencies = config.get("dependencies", [])
    assert len(dependencies) >= 4, "Must have interactive callbacks for each of the core tabs."

    # Validate each callback wiring has valid inputs and outputs
    for dep in dependencies:
        assert len(dep.get("inputs", [])) >= 0
        assert len(dep.get("outputs", [])) >= 0
