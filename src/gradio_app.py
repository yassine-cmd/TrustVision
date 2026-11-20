"""
TrustVision - Master Gradio demo (src/gradio_app.py)
====================================================
Four-tab interactive demo that assembles the deliverables of the three members
without retraining anything:

* **Tab 1 - CLIP & attention (Membre A)** : live CLIP zero-shot prompt scoring
  plus CLS-to-patch attention overlay (``src/clip_attn.py``). The S0 calibrated
  linear probe (99.44 %) is reported next to the weaker zero-shot baseline.
* **Tab 2 - SimCLR & label efficiency (Membre C)** : the real S1 curves from
  ``results/s1_simclr.json`` and live inference with the frozen probes exported
  at 1 / 5 / 10 / 100 % labels (and the from-scratch CNNs).
* **Tab 3 - Trust audit (Membre B)** : temperature scaling, FGSM, Grad-CAM and
  Integrated Gradients through ``TrustVisionTab3Auditor``.
* **Tab 4 - EWC & catastrophic forgetting (Membre C)** : a lambda slider that
  loads the real ``ewc_lambda_*`` checkpoints, shows old/new/forgetting/ECE
  from run 3 and runs live inference with the selected strategy.

Run locally / on Colab:
    pip install "gradio==4.44.1"
    python -m src.gradio_app
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from src.data import NEU_CLASSES
from src.eval import load_model_checkpoint
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]
try:
    gr.set_static_paths([str(ROOT / "factory_line")])
except Exception:
    pass

S1_PROBE_PATTERN = "checkpoints/s1_simclr_probe_{fraction}pct.pt"
S1_SCRATCH_PATTERN = "checkpoints/s1_scratch_{fraction}pct.pt"

EWC_STRATEGIES = {
    "EWC pur (lambda)": None,
    "EWC + Replay (lambda = 10)": "ewc_replay",
    "Replay seul": "replay",
    "Fine-tuning naif": "naive",
}
EWC_LAMBDAS = [0.1, 1.0, 10.0, 100.0]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _read_json(path: Union[str, Path]) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _to_pil(value: Union[Image.Image, np.ndarray, torch.Tensor, None]) -> Optional[Image.Image]:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu()
        if array.ndim == 4:
            array = array[0]
        array = array.permute(1, 2, 0).numpy()
        value = array
    if isinstance(value, np.ndarray):
        array = value
        if array.dtype != np.uint8:
            array = np.clip(array, 0.0, 1.0) * 255.0
            array = array.astype(np.uint8)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        return Image.fromarray(array).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(value)}")


def _overlay_heatmap(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Image.Image:
    """Blends a normalized saliency/attention map over the input image."""
    base = _to_pil(image)
    heat = np.asarray(heatmap, dtype=np.float32)
    heat = np.squeeze(heat)
    if heat.ndim != 2:
        heat = heat.reshape(heat.shape[-2], heat.shape[-1])
    heat = heat - float(np.nanmin(heat))
    heat = heat / (float(np.nanmax(heat)) + 1e-8)
    rgba = matplotlib.colormaps[colormap](heat)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    colored = Image.fromarray(rgb).resize(base.size, Image.BILINEAR)
    return Image.blend(base, colored, alpha=alpha)


def _image_preprocess(image: Union[Image.Image, np.ndarray], size: int = 96) -> torch.Tensor:
    pil = _to_pil(image)
    transform = transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
        ]
    )
    return transform(pil).unsqueeze(0)


def _probabilities(model: torch.nn.Module, image: torch.Tensor) -> Tuple[str, Dict[str, float], torch.Tensor]:
    model.eval()
    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()
    pred_idx = int(probs.argmax().item())
    return NEU_CLASSES[pred_idx], {
        name: round(float(probs[i].item()), 4) for i, name in enumerate(NEU_CLASSES)
    }, probs


def _effective_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Uses calibrated metrics only when the no-harm guard accepted the fit."""
    calibrated = metrics.get("calibrated") or {}
    if calibrated.get("calibration_accepted", True):
        merged = dict(metrics)
        for key in ("acc", "macro_f1", "ece", "nll", "brier", "temperature"):
            if calibrated.get(key) is not None:
                merged[key] = calibrated[key]
        return merged
    return dict(metrics)


def _metric_markdown(metrics: Dict[str, Any], title: str) -> str:
    effective = _effective_metrics(metrics)
    calibration = (
        "calibrée" if (metrics.get("calibrated") or {}).get("calibration_accepted", False) else "brute (fit rejeté)"
    )
    lines = [
        f"### {title}",
        "",
        "| Métrique | Valeur |",
        "| :--- | :---: |",
        f"| Accuracy globale | **{effective['acc'] * 100:.2f} %** |",
        f"| Macro-F1 | {effective['macro_f1']:.4f} |",
        f"| ECE ({calibration}) | {effective['ece']:.4f} |",
        f"| NLL | {effective['nll']:.4f} |",
        f"| Brier | {effective['brier']:.4f} |",
        f"| Rob. FGSM eps=0.03 | {metrics['rob_acc_003'] * 100:.2f} % |",
    ]
    if metrics.get("old_acc") is not None:
        lines.insert(5, f"| Anciennes classes (1-3) | {metrics['old_acc'] * 100:.2f} % |")
    if metrics.get("new_acc") is not None:
        lines.insert(6, f"| Nouvelles classes (4-6) | {metrics['new_acc'] * 100:.2f} % |")
    if metrics.get("forgetting") is not None:
        lines.insert(7, f"| Forgetting | {metrics['forgetting'] * 100:.2f} % |")
    return "\n".join(lines)


def _bar_figure(labels: List[str], values: List[float], title: str, ylabel: str = "%") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=140)
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#17becf"]
    bars = ax.bar(labels, values, color=colors[: len(labels)])
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tab 1 - CLIP & attention (Member A)
# ---------------------------------------------------------------------------

class ClipAttentionTab:
    """Lazy CLIP loader so the demo starts instantly without downloading weights."""

    def __init__(self, root: Path):
        self.root = root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.text_features = None
        self.current_prompt: Optional[str] = None
        self.s0 = _read_json(root / "results/s0_clip.json") if (root / "results/s0_clip.json").exists() else {}

    def _ensure_model(self, prompt: str) -> None:
        from src.clip_attn import build_text_embeddings_for_prompts, load_clip

        if self.model is None:
            self.model, self.processor = load_clip(device=self.device)
        if self.text_features is None or prompt != self.current_prompt:
            self.text_features = build_text_embeddings_for_prompts(
                self.model,
                self.processor,
                prompt,
                class_names=NEU_CLASSES,
                device=self.device,
            )
            self.current_prompt = prompt

    def analyze(self, image, prompt: str) -> Tuple[Optional[Image.Image], Optional[Dict[str, float]], str]:
        if image is None:
            return None, None, "Charge une image NEU-CLS pour lancer l'analyse CLIP."
        try:
            from src.clip_attn import extract_cls_patch_attention, extract_tensor_features

            self._ensure_model(prompt)
            pil = _to_pil(image).resize((224, 224), Image.BICUBIC)
            tensor = transforms.ToTensor()(pil).unsqueeze(0).to(self.device)

            attention = extract_cls_patch_attention(
                self.model, tensor, layer_idx=None, device=self.device
            )[0].mean(axis=0)
            overlay = _overlay_heatmap(pil, attention, alpha=0.45, colormap="viridis")

            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
            with torch.no_grad():
                raw = self.model.get_image_features((tensor - mean) / std)
                features = extract_tensor_features(raw)
                features = features / features.norm(dim=-1, keepdim=True)
                logits = self.model.logit_scale.exp() * features @ self.text_features.t()
                probs = F.softmax(logits, dim=1).squeeze(0).cpu()

            scores = {
                name: round(float(probs[i].item()), 4) for i, name in enumerate(NEU_CLASSES)
            }
            zero_shot = (
                (self.s0.get("theme4_zero_shot", {}).get("selected_test_metrics", {}) or {}).get("accuracy")
            )
            probe = (
                (self.s0.get("theme4_linear_probe", {}) or {}).get("calibrated", {}) or {}
            ).get("accuracy")
            note = (
                "**Attention CLS → patches** (moyenne des 12 couches, 7×7). "
                "CLIP zero-shot est faible sur NEU-CLS : "
                f"**{zero_shot * 100:.2f} %** en test (S0). "
                "La sonde linéaire calibrée, elle, atteint "
                f"**{probe * 100:.2f} %** : le prompt sert à illustrer le VLM, "
                "pas à remplacer le classifieur supervisé."
                if zero_shot is not None and probe is not None
                else "**Attention CLS → patches** (moyenne des 12 couches, 7×7)."
            )
            return overlay, scores, note
        except Exception as exc:  # pragma: no cover - depends on Colab/transformers
            return None, None, (
                "Impossible de charger CLIP dans cet environnement "
                f"(`{type(exc).__name__}: {exc}`).\n\n"
                "Sur Colab : `pip install transformers` puis relance l'onglet. "
                "Les résultats S0 restent disponibles dans `results/s0_clip.json`."
            )


# ---------------------------------------------------------------------------
# Tab 2 - SimCLR label efficiency (Member C)
# ---------------------------------------------------------------------------

class SimCLRTab:
    def __init__(self, root: Path):
        self.root = root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = _read_json(root / "results/s1_simclr.json")
        checkpoint_metrics_path = root / "results/s1_checkpoint_metrics.json"
        self.checkpoint_metrics = (
            _read_json(checkpoint_metrics_path) if checkpoint_metrics_path.exists() else {}
        )
        self.models: Dict[str, torch.nn.Module] = {}
        self.choices = self._build_choices()

    def _display_acc(self, fraction: str, method: str, fallback: float) -> float:
        """Uses the re-evaluated checkpoint accuracy when available."""
        record = (
            self.checkpoint_metrics.get("checkpoints", {})
            .get(fraction, {})
            .get(method, {})
        )
        value = record.get("acc")
        return float(value) if value is not None else float(fallback)

    def _build_choices(self) -> Dict[str, str]:
        choices: Dict[str, str] = {}
        simclr = self.data["simclr"]["label_fractions"]
        scratch = self.data["supervised_from_scratch"]["label_fractions"]
        for fraction in ("1pct", "5pct", "10pct", "100pct"):
            if fraction not in simclr:
                continue
            path = self.root / S1_PROBE_PATTERN.format(fraction=fraction.replace("pct", ""))
            acc = self._display_acc(fraction, "probe", simclr[fraction]["acc"])
            label = f"SimCLR probe {fraction.replace('pct', ' %')} - {acc * 100:.2f} %"
            if path.exists():
                choices[label] = str(path)
            if fraction in scratch:
                scratch_path = self.root / S1_SCRATCH_PATTERN.format(fraction=fraction.replace("pct", ""))
                scratch_acc = self._display_acc(fraction, "scratch", scratch[fraction]["acc"])
                label_s = (
                    f"CNN scratch {fraction.replace('pct', ' %')} - "
                    f"{scratch_acc * 100:.2f} %"
                )
                if scratch_path.exists():
                    choices[label_s] = str(scratch_path)
        return choices

    def label_efficiency_figure(self) -> plt.Figure:
        keys = [k for k in ("1pct", "5pct", "10pct", "100pct") if k in self.data["simclr"]["label_fractions"]]
        xs = [int(k.replace("pct", "")) for k in keys]
        sim = [self.data["simclr"]["label_fractions"][k]["acc"] * 100 for k in keys]
        scratch_data = self.data["supervised_from_scratch"]["label_fractions"]
        scratch = [scratch_data[k]["acc"] * 100 for k in keys]

        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=140)
        axes[0].plot(xs, sim, marker="o", linewidth=2, label="SimCLR (frozen probe)")
        axes[0].plot(xs, scratch, marker="s", linewidth=2, label="CNN from scratch")
        axes[0].set_xscale("log")
        axes[0].set_xticks(xs)
        axes[0].set_xticklabels([f"{x}%" for x in xs])
        axes[0].set_xlabel("Budget d'étiquettes")
        axes[0].set_ylabel("Accuracy test (%)")
        axes[0].set_title("S1 - Efficacité des étiquettes", fontweight="bold", fontsize=10)
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=7)

        sim_ece = [
            (self.data["simclr"]["label_fractions"][k].get("calibrated") or {}).get(
                "ece", self.data["simclr"]["label_fractions"][k]["ece"]
            )
            for k in keys
        ]
        scratch_ece = [
            (scratch_data[k].get("calibrated") or {}).get("ece", scratch_data[k]["ece"])
            for k in keys
        ]
        axes[1].plot(xs, sim_ece, marker="o", linewidth=2, label="SimCLR (calibré)")
        axes[1].plot(xs, scratch_ece, marker="s", linewidth=2, label="Scratch (calibré)")
        axes[1].set_xscale("log")
        axes[1].set_xticks(xs)
        axes[1].set_xticklabels([f"{x}%" for x in xs])
        axes[1].set_xlabel("Budget d'étiquettes")
        axes[1].set_ylabel("ECE")
        axes[1].set_title("S1 - Calibration", fontweight="bold", fontsize=10)
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        return fig

    def predict(self, image, choice: str) -> Tuple[Optional[Dict[str, float]], str]:
        if image is None:
            return None, "Charge une image pour tester le checkpoint sélectionné."
        if not choice or choice not in self.choices:
            return None, "Aucun checkpoint exporté pour ce budget : lance `scripts/export_s1_fraction_checkpoints.py`."
        path = self.choices[choice]
        if choice not in self.models:
            self.models[choice] = load_model_checkpoint(path, device=self.device)
        tensor = _image_preprocess(image, size=96).to(self.device)
        pred, probs, _ = _probabilities(self.models[choice], tensor)
        detail = (
            f"**Prédiction : {pred}** avec `{Path(path).name}`.\n\n"
            "L'accuracy affichée dans le menu est celle du checkpoint ré-évalué "
            "(`results/s1_checkpoint_metrics.json`) lorsqu'elle est disponible. "
            "Rappel : à 1 % et 5 %, la sonde SimCLR domine largement le CNN "
            "from scratch ; à 100 %, les deux se rejoignent."
        )
        return probs, detail


# ---------------------------------------------------------------------------
# Tab 4 - EWC lambda slider (Member C)
# ---------------------------------------------------------------------------

class EWCTab:
    def __init__(self, root: Path):
        self.root = root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = _read_json(root / "results/s4_ewc.json")
        self.strategies = self.data["strategies"]
        self.models: Dict[str, torch.nn.Module] = {}

    def resolve(self, strategy_label: str, lambda_index: int) -> Tuple[str, Dict[str, Any], str]:
        lambda_value = EWC_LAMBDAS[int(round(lambda_index))]
        if strategy_label == "EWC pur (lambda)":
            key = f"ewc_lambda_{lambda_value:g}"
            title = f"EWC pur, lambda = {lambda_value:g}"
        else:
            key = EWC_STRATEGIES[strategy_label]
            title = strategy_label
        if key is None or key not in self.strategies:
            key = "ewc_replay"
            title = "EWC + Replay (lambda = 10)"
        strategy = self.strategies[key]
        return key, strategy, title

    def metrics_markdown(self, strategy_label: str, lambda_index: int) -> str:
        key, strategy, title = self.resolve(strategy_label, lambda_index)
        metrics = strategy["metrics"]
        text = _metric_markdown(metrics, title)
        if key == "ewc_replay":
            text += (
                "\n\n> Variante retenue pour la ligne S4 : meilleure de la famille EWC. "
                "Le **replay seul** reste le meilleur modèle S4 (93,06 %)."
            )
        if metrics.get("calibrated", {}).get("calibration_accepted") is False:
            text += (
                "\n\n> Fit de température rejeté (T hors plage / NLL dégradée) : "
                "les probabilités brutes sont affichées."
            )
        return text

    def bar_figure(self, strategy_label: str, lambda_index: int) -> plt.Figure:
        _, strategy, title = self.resolve(strategy_label, lambda_index)
        metrics = strategy["metrics"]
        return _bar_figure(
            ["Old 1-3", "New 4-6", "Global"],
            [metrics["old_acc"] * 100, metrics["new_acc"] * 100, metrics["acc"] * 100],
            f"{title} - précision par sous-ensemble",
        )

    def comparison_table(self) -> List[List[Any]]:
        rows: List[List[Any]] = []
        for key, strategy in self.strategies.items():
            metrics = _effective_metrics(strategy["metrics"])
            rows.append(
                [
                    key,
                    f"{strategy.get('ewc_lambda', 0):g}",
                    f"{metrics['acc'] * 100:.2f}",
                    f"{metrics['old_acc'] * 100:.2f}",
                    f"{metrics['new_acc'] * 100:.2f}",
                    f"{metrics['forgetting'] * 100:.2f}",
                    f"{metrics['ece']:.4f}",
                ]
            )
        return rows

    def predict(self, image, strategy_label: str, lambda_index: int) -> Tuple[Optional[Dict[str, float]], str]:
        if image is None:
            return None, "Charge une image pour tester la stratégie EWC sélectionnée."
        key, strategy, title = self.resolve(strategy_label, lambda_index)
        checkpoint = strategy.get("checkpoint")
        if not checkpoint:
            return None, "Checkpoint introuvable pour cette stratégie."
        path = self.root / checkpoint
        if not path.exists():
            return None, f"Checkpoint absent : `{checkpoint}`."
        if key not in self.models:
            self.models[key] = load_model_checkpoint(path, device=self.device)
        tensor = _image_preprocess(image, size=96).to(self.device)
        pred, probs, _ = _probabilities(self.models[key], tensor)
        metrics = _effective_metrics(strategy["metrics"])
        detail = (
            f"**Prédiction : {pred}** avec `{Path(checkpoint).name}`.\n\n"
            f"Old {metrics['old_acc'] * 100:.2f} % · New {metrics['new_acc'] * 100:.2f} % · "
            f"Forgetting {metrics['forgetting'] * 100:.2f} %."
        )
        return probs, detail


# ---------------------------------------------------------------------------
# Gradio assembly
# ---------------------------------------------------------------------------

def build_demo(root: Union[str, Path] = ROOT):
    """Builds the four-tab Gradio Blocks app."""
    import gradio as gr

    root = Path(root)
    clip_tab = ClipAttentionTab(root)
    simclr_tab = SimCLRTab(root)
    ewc_tab = EWCTab(root)
    auditor_holder: Dict[str, Any] = {}

    def get_auditor():
        if "auditor" not in auditor_holder:
            from src.gradio_tab3 import TrustVisionTab3Auditor

            auditor_holder["auditor"] = TrustVisionTab3Auditor(
                base_checkpoint=str(root / "checkpoints/cnn_base_seed42.pt"),
                adv_checkpoint=str(root / "checkpoints/cnn_adv_seed42.pt"),
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
        return auditor_holder["auditor"]

    def run_tab3(image, model_type, temperature, epsilon, include_ig):
        if image is None:
            return None, None, None, "Charge une image pour lancer l'audit.", None
        try:
            auditor = get_auditor()
            result = auditor.audit(
                _to_pil(image),
                temperature=float(temperature),
                epsilon=float(epsilon),
                model_type=model_type,
                include_ig=bool(include_ig),
            )
            base = _to_pil(image)
            cam = _overlay_heatmap(base, result["grad_cam_map"], alpha=0.45, colormap="jet")
            ig = (
                _overlay_heatmap(base, result["integrated_gradients_map"], alpha=0.45, colormap="magma")
                if result.get("integrated_gradients_map") is not None
                else None
            )
            probs = result["class_probabilities"]
            detail = (
                f"**{result['predicted_class']}** · confiance {result['confidence'] * 100:.2f} % · "
                f"T={result['temperature_used']} · eps={result['epsilon_used']} · "
                f"Inférence: {result['latency_ms']:.1f} ms ({'< 500 ms' if result['sub_500ms_contract_met'] else '> 500 ms'}) · "
                f"Total audit (XAI+Grad): {result.get('total_pipeline_ms', result['latency_ms']):.1f} ms."
            )
            return probs, _to_pil(result["perturbed_image"]), cam, detail, ig
        except Exception as exc:
            return None, None, None, f"Erreur d'audit : `{type(exc).__name__}: {exc}`", None

    with gr.Blocks(title="TrustVision - Démonstrateur", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# TrustVision - Démonstrateur d'inspection qualité\n"
            "Pipeline S0 → S4 : CLIP, SimCLR, calibration, XAI, FGSM, EWC. "
            "Toutes les métriques affichées proviennent des JSON réels du dépôt ; "
            "aucune valeur n'est simulée."
        )

        # ------------------------------------------------------------------
        # Tab 1 - CLIP
        # ------------------------------------------------------------------
        with gr.Tab("1 · CLIP & Attention (A)"):
            gr.Markdown(
                "Score zero-shot par prompts et carte d'attention **CLS → patches** "
                "(12 couches moyennées). Le premier clic télécharge CLIP (~600 Mo)."
            )
            with gr.Row():
                clip_image = gr.Image(type="pil", label="Image NEU-CLS")
                clip_overlay = gr.Image(type="pil", label="Attention CLS → patches")
            clip_prompt = gr.Textbox(
                value="a photo of a steel surface with {defect}",
                label="Template de prompt ({defect} est remplacé par chaque classe)",
            )
            clip_button = gr.Button("Analyser avec CLIP", variant="primary")
            clip_scores = gr.Label(num_top_classes=3, label="Zero-shot")
            clip_note = gr.Markdown()

            clip_button.click(
                clip_tab.analyze,
                inputs=[clip_image, clip_prompt],
                outputs=[clip_overlay, clip_scores, clip_note],
            )

        # ------------------------------------------------------------------
        # Tab 2 - SimCLR
        # ------------------------------------------------------------------
        with gr.Tab("2 · SimCLR (C)"):
            gr.Markdown(
                "Courbes réelles de `results/s1_simclr.json` et inférence avec les "
                "checkpoints 1 / 5 / 10 / 100 % exportés."
            )
            simclr_plot = gr.Plot(value=simclr_tab.label_efficiency_figure(), label="S1")
            with gr.Row():
                simclr_image = gr.Image(type="pil", label="Image NEU-CLS")
                simclr_probs = gr.Label(num_top_classes=3, label="Prédiction du checkpoint")
            simclr_choice = gr.Dropdown(
                choices=list(simclr_tab.choices.keys()) or ["Aucun checkpoint disponible"],
                value=(list(simclr_tab.choices.keys())[0] if simclr_tab.choices else None),
                label="Budget / modèle",
            )
            simclr_button = gr.Button("Prédire", variant="primary")
            simclr_detail = gr.Markdown()

            simclr_button.click(
                simclr_tab.predict,
                inputs=[simclr_image, simclr_choice],
                outputs=[simclr_probs, simclr_detail],
            )

        # ------------------------------------------------------------------
        # Tab 3 - Trust audit (B)
        # ------------------------------------------------------------------
        with gr.Tab("3 · XAI & FGSM (B)"):
            gr.Markdown(
                "Audit de confiance : choix du modèle, température, budget FGSM, "
                "Grad-CAM et Integrated Gradients. Contrat de latence < 500 ms."
            )
            with gr.Row():
                tab3_image = gr.Image(type="pil", label="Image NEU-CLS")
                tab3_perturbed = gr.Image(type="pil", label="Image perturbée (FGSM)")
            with gr.Row():
                tab3_model = gr.Radio(
                    choices=[("CNN baseline", "base"), ("CNN défense adversariale", "adv")],
                    value="base",
                    label="Modèle",
                )
                tab3_temperature = gr.Slider(0.1, 3.0, value=1.0, step=0.05, label="Température T")
                tab3_epsilon = gr.Slider(0.0, 0.10, value=0.03, step=0.005, label="Epsilon FGSM")
                tab3_ig = gr.Checkbox(value=True, label="Integrated Gradients")
            tab3_button = gr.Button("Lancer l'audit", variant="primary")
            with gr.Row():
                tab3_probs = gr.Label(num_top_classes=3, label="Probabilités calibrées")
                tab3_cam = gr.Image(type="pil", label="Grad-CAM")
                tab3_ig_map = gr.Image(type="pil", label="Integrated Gradients")
            tab3_detail = gr.Markdown()

            tab3_button.click(
                run_tab3,
                inputs=[tab3_image, tab3_model, tab3_temperature, tab3_epsilon, tab3_ig],
                outputs=[tab3_probs, tab3_perturbed, tab3_cam, tab3_detail, tab3_ig_map],
            )

        # ------------------------------------------------------------------
        # Tab 4 - EWC
        # ------------------------------------------------------------------
        with gr.Tab("4 · EWC (C)"):
            gr.Markdown(
                "Curseur lambda : chaque position charge le checkpoint réel "
                "`ewc_lambda_*_seed42.pt` et affiche old/new/forgetting. "
                "Le scénario est **single-head 3 → 6 classes**, sans identifiant de tâche."
            )
            with gr.Row():
                ewc_strategy = gr.Dropdown(
                    choices=list(EWC_STRATEGIES.keys()),
                    value="EWC pur (lambda)",
                    label="Stratégie",
                )
                ewc_lambda = gr.Slider(
                    0,
                    3,
                    value=2,
                    step=1,
                    label="Lambda (0 → 0.1 · 1 → 1 · 2 → 10 · 3 → 100)",
                )
            with gr.Row():
                ewc_metrics = gr.Markdown(ewc_tab.metrics_markdown("EWC pur (lambda)", 2))
                ewc_plot = gr.Plot(value=ewc_tab.bar_figure("EWC pur (lambda)", 2), label="S4")
            ewc_table = gr.Dataframe(
                headers=["Stratégie", "Lambda", "Acc %", "Old %", "New %", "Forgetting %", "ECE"],
                value=ewc_tab.comparison_table(),
                label="Balayage complet (run 3)",
                interactive=False,
            )
            with gr.Row():
                ewc_image = gr.Image(type="pil", label="Image NEU-CLS")
                ewc_probs = gr.Label(num_top_classes=3, label="Prédiction de la stratégie")
            ewc_button = gr.Button("Prédire avec cette stratégie", variant="primary")
            ewc_detail = gr.Markdown()

            for control in (ewc_strategy, ewc_lambda):
                control.change(
                    lambda strategy, index: (
                        ewc_tab.metrics_markdown(strategy, index),
                        ewc_tab.bar_figure(strategy, index),
                    ),
                    inputs=[ewc_strategy, ewc_lambda],
                    outputs=[ewc_metrics, ewc_plot],
                )
            ewc_button.click(
                ewc_tab.predict,
                inputs=[ewc_image, ewc_strategy, ewc_lambda],
                outputs=[ewc_probs, ewc_detail],
            )

        # ------------------------------------------------------------------
        # Tab 5 - 3D Virtual Factory Line (Flagship)
        # ------------------------------------------------------------------
        with gr.Tab("5 · Ligne d'Usine 3D (Flagship)"):
            gr.Markdown(
                "### 🏭 Ligne d'Inspection Virtuelle 3D en Temps Réel (Three.js)\n\n"
                "Jumeau numérique industriel animé à 60 FPS avec scanner optique zénithal, "
                "banc d'aiguillage pneumatique et simulation de rejet basée sur le modèle calibré.\n\n"
                "#### 📐 Architecture & Spécifications Industrielles :\n"
                "* **Porte de confiance calibrée :** &tau;<sub>95</sub> = 0.9940 (dérivée dans `notebooks/exp_A.ipynb` pour garantir 95 % de couverture autonome).\n"
                "* **Voie A (Verte) :** Confiance calibrée &ge; 0.9940 &rarr; Conditionnement automatisé haute cadence.\n"
                "* **Voie B (Orange) :** Confiance &lt; 0.9940 &rarr; Déviation vers poste de revue humaine (`CONTRÔLE`).\n"
                "* **Stress test adversarial FGSM (&epsilon; = 0.03) :** Démonstration en direct du basculement sécurisé vers l'opérateur dès que la confiance chute sous l'effet d'une perturbation.\n"
                "* **Moteur physique & rendu :** Moteur Three.js WebGL 2 autonome, cinématique de chute vérifiée par 7 bancs de tests, éclairage PBR et capteur linéaire 4K."
            )
            gr.HTML(
                """
                <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(59, 130, 246, 0.08) 100%); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 12px; padding: 28px; text-align: center; margin: 20px 0;">
                    <div style="font-size: 20px; font-weight: 700; color: #10b981; margin-bottom: 8px;">
                        Expérience Plein Écran Haute Résolution (60 FPS)
                    </div>
                    <p style="color: #94a3b8; font-size: 14px; max-width: 650px; margin: 0 auto 20px auto; line-height: 1.5;">
                        Pour une ergonomie optimale et un affichage 3D natif sans contrainte de formatage ni barre de défilement, la simulation s'exécute dans un onglet dédié plein écran.
                    </p>
                    <a href="/factory-line" target="_blank" style="display: inline-flex; align-items: center; justify-content: center; gap: 10px; background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: #ffffff; padding: 14px 28px; border-radius: 8px; font-weight: 700; font-size: 15px; text-decoration: none; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.35); transition: transform 0.15s ease;">
                        <span>🚀 Ouvrir la Ligne 3D Plein Écran (Nouvel Onglet)</span>
                    </a>
                    <div style="margin-top: 18px; font-size: 12px; color: #64748b;">
                        Contrôles : <strong>1 à 6</strong> (caméras) · <strong>Espace</strong> (pause) · <strong>.</strong> (pas à pas) · <strong>Clic droit / Molette</strong> (translation / zoom)
                    </div>
                </div>
                """
            )

        gr.Markdown(
            "---\n"
            "**Reproductibilité** : `pytest tests/ -q` · S1 `results/s1_simclr.json` · "
            "S4 `results/s4_ewc.json` · tableau croisé `results/cross_effects_table.json` · "
            "Flagship 3D [`factory_line/index.html`](factory_line/index.html)."
        )

    return demo


def main() -> None:
    demo = build_demo()
    app = FastAPI(title="TrustVision")

    @app.get("/factory-line")
    def serve_factory_line():
        html_path = ROOT / "factory_line/index.html"
        return FileResponse(html_path, media_type="text/html")

    app = gr.mount_gradio_app(app, demo, path="/")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
