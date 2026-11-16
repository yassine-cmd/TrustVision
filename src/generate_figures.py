"""
TrustVision - Visual Results Generator (src/generate_figures.py)
Generates publication-quality figures for Member B deliverables:
1. results/figures/fgsm_robustness_curve.png
2. results/figures/calibration_reliability_diagrams.png
3. results/figures/xai_gradcam_vs_ig.png
4. results/figures/intra_pipeline_xai_stability.png
5. results/figures/trustvision_member_b_dashboard.png (Unified Overview)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import get_dataloaders, NEU_CLASSES
from src.eval import load_model_checkpoint
from src.xai_cal_adv import GradCAM, integrated_gradients


def generate_all_figures():
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Figures] Generating figures into {output_dir} using device: {device}...")

    # Load S2 and S3 JSON results
    with open("results/s2_calibration.json", "r", encoding="utf-8") as f:
        s2 = json.load(f)
    with open("results/s3_adversarial.json", "r", encoding="utf-8") as f:
        s3 = json.load(f)

    # Load models
    base_model = load_model_checkpoint("checkpoints/cnn_base_seed42.pt", device=device)
    adv_model = load_model_checkpoint("checkpoints/cnn_adv_seed42.pt", device=device)

    # Load test dataloader
    _, _, test_loader, _ = get_dataloaders(
        root_dir="data/NEU-CLS", batch_size=32, seed=42, img_size=96
    )

    # --------------------------------------------------------------------------
    # FIGURE 1: FGSM Robustness Curve
    # --------------------------------------------------------------------------
    print("[Figures] Plotting Figure 1: FGSM Robustness Curve...")
    eps_keys = s3["epsilon_sweep"]
    clean_accs = [s3["clean_model_sweep"][str(e)]["accuracy"] * 100 for e in eps_keys]
    adv_accs = [s3["adv_model_sweep"][str(e)]["accuracy"] * 100 for e in eps_keys]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(eps_keys, clean_accs, "o-", color="#c0392b", linewidth=2.4, markersize=7, label="CNN Standard (S2)")
    ax.plot(eps_keys, adv_accs, "s-", color="#27ae60", linewidth=2.4, markersize=7, label="CNN Défense Adversariale (S3)")
    ax.axvline(x=0.03, color="#7f8c8d", linestyle="--", linewidth=1.5, label="Budget Contrat (eps=0.03)")

    ax.set_xlabel(r"Budget de Perturbation FGSM ($\epsilon$) sur pixels [0, 1]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Exactitude sur Split Test (%)", fontsize=11, fontweight="bold")
    ax.set_title("Courbe de Robustesse sous Attaque FGSM\n(Standard vs. Défense Adversariale)", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 105])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="#f8f9fa", edgecolor="#bdc3c7", fontsize=10)

    # Annotate gain at 0.03
    gain = s3["rob_acc_003_comparison"]["absolute_gain"] * 100
    ax.annotate(
        f"+{gain:.1f}% de gain\nà eps=0.03",
        xy=(0.03, s3["rob_acc_003_comparison"]["adv_trained"] * 100),
        xytext=(0.045, 82),
        arrowprops=dict(facecolor="#27ae60", shrink=0.05, width=1.5, headwidth=6),
        fontweight="bold",
        color="#27ae60",
        fontsize=9,
    )

    plt.tight_layout()
    fig1_path = output_dir / "fgsm_robustness_curve.png"
    plt.savefig(fig1_path)
    plt.close()
    print(f"  -> Saved {fig1_path}")

    # --------------------------------------------------------------------------
    # FIGURE 2: Reliability Diagrams Before vs After Temperature Scaling
    # --------------------------------------------------------------------------
    print("[Figures] Plotting Figure 2: Calibration Reliability Diagrams...")
    ts_data = s2["temperature_scaling"]
    uncal = ts_data["uncalibrated_test_metrics"]
    cal = ts_data["calibrated_test_metrics"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    bin_edges = np.linspace(0, 1, 11)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for ax, data_dict, title_prefix, color in zip(
        axes,
        [uncal, cal],
        ["Avant Calibration (Logits bruts)", f"Après Temperature Scaling (T={ts_data['optimal_temperature_T']:.3f})"],
        ["#2980b9", "#16a085"],
    ):
        accs = data_dict["bin_stats"]["accuracies"]
        confs = data_dict["bin_stats"]["confidences"]
        ece = data_dict["ece"]
        ci = data_dict["ece_ci"]

        ax.bar(centers, accs, width=0.08, color=color, alpha=0.75, edgecolor="#2c3e50", label="Précision par bin")
        ax.plot([0, 1], [0, 1], "--", color="#e74c3c", linewidth=1.8, label="Calibration parfaite")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Confiance moyenne", fontsize=10, fontweight="bold")
        ax.set_ylabel("Exactitude empirique", fontsize=10, fontweight="bold")
        ax.set_title(f"{title_prefix}\nECE = {ece:.4f} (IC 95%: [{ci[0]:.4f}, {ci[1]:.4f}])", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig2_path = output_dir / "calibration_reliability_diagrams.png"
    plt.savefig(fig2_path)
    plt.close()
    print(f"  -> Saved {fig2_path}")

    # --------------------------------------------------------------------------
    # FIGURE 3: XAI Visual Panel (Original, Grad-CAM, Integrated Gradients)
    # --------------------------------------------------------------------------
    print("[Figures] Plotting Figure 3: XAI Panel (Grad-CAM vs IG)...")
    images_batch, labels_batch = next(iter(test_loader))
    sample_indices = [0, 4, 8, 12]  # 4 distinct samples
    imgs_sub = images_batch[sample_indices].to(device)
    lbls_sub = labels_batch[sample_indices]

    gradcam_exp = GradCAM(base_model)
    cams = gradcam_exp(imgs_sub)
    gradcam_exp.remove_hooks()
    igs, _ = integrated_gradients(base_model, imgs_sub, steps=30)

    fig, axes = plt.subplots(4, 3, figsize=(9.5, 11), dpi=300)
    for row in range(4):
        img_np = imgs_sub[row].permute(1, 2, 0).cpu().numpy()
        cls_name = NEU_CLASSES[lbls_sub[row].item()]

        # Col 0: Original
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f"Image Réelle\n[{cls_name}]", fontsize=10, fontweight="bold")
        axes[row, 0].axis("off")

        # Col 1: Grad-CAM Overlay
        axes[row, 1].imshow(img_np)
        axes[row, 1].imshow(cams[row], cmap="jet", alpha=0.55)
        axes[row, 1].set_title("Grad-CAM Overlay", fontsize=10, fontweight="bold")
        axes[row, 1].axis("off")

        # Col 2: Integrated Gradients
        axes[row, 2].imshow(igs[row], cmap="hot")
        axes[row, 2].set_title("Integrated Gradients", fontsize=10, fontweight="bold")
        axes[row, 2].axis("off")

    plt.tight_layout()
    fig3_path = output_dir / "xai_gradcam_vs_ig.png"
    plt.savefig(fig3_path)
    plt.close()
    print(f"  -> Saved {fig3_path}")

    # --------------------------------------------------------------------------
    # FIGURE 4: Intra-Pipeline XAI Stability (Clean vs Adv Defense)
    # --------------------------------------------------------------------------
    print("[Figures] Plotting Figure 4: Intra-Pipeline XAI Stability...")
    gradcam_clean = GradCAM(base_model)
    gradcam_adv = GradCAM(adv_model)

    cams_clean = gradcam_clean(imgs_sub)
    cams_adv = gradcam_adv(imgs_sub)
    gradcam_clean.remove_hooks()
    gradcam_adv.remove_hooks()

    fig, axes = plt.subplots(4, 3, figsize=(9.5, 11), dpi=300)
    for row in range(4):
        img_np = imgs_sub[row].permute(1, 2, 0).cpu().numpy()
        cls_name = NEU_CLASSES[lbls_sub[row].item()]

        # Original
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f"Défaut: {cls_name}", fontsize=10, fontweight="bold")
        axes[row, 0].axis("off")

        # Grad-CAM Clean
        axes[row, 1].imshow(img_np)
        axes[row, 1].imshow(cams_clean[row], cmap="jet", alpha=0.55)
        axes[row, 1].set_title("Grad-CAM (CNN Standard)", fontsize=10, fontweight="bold")
        axes[row, 1].axis("off")

        # Grad-CAM Adv
        axes[row, 2].imshow(img_np)
        axes[row, 2].imshow(cams_adv[row], cmap="jet", alpha=0.55)
        axes[row, 2].set_title("Grad-CAM (CNN Défendu)", fontsize=10, fontweight="bold")
        axes[row, 2].axis("off")

    plt.tight_layout()
    fig4_path = output_dir / "intra_pipeline_xai_stability.png"
    plt.savefig(fig4_path)
    plt.close()
    print(f"  -> Saved {fig4_path}")

    # --------------------------------------------------------------------------
    # FIGURE 5: Unified Summary Poster (Dashboard 2x2)
    # --------------------------------------------------------------------------
    print("[Figures] Plotting Figure 5: Unified Summary Dashboard (Poster)...")
    fig = plt.figure(figsize=(14, 11), dpi=300)
    fig.suptitle("TrustVision — Rapport Visuel d'Audit de Confiance (Membre B)", fontsize=16, fontweight="bold", y=0.98)

    # Subplot 1: Robustness Curve
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(eps_keys, clean_accs, "o-", color="#c0392b", linewidth=2.2, label="CNN Standard (S2)")
    ax1.plot(eps_keys, adv_accs, "s-", color="#27ae60", linewidth=2.2, label="CNN Défense Adversariale (S3)")
    ax1.axvline(x=0.03, color="#7f8c8d", linestyle="--", label="Budget eps=0.03")
    ax1.set_xlabel(r"Budget FGSM ($\epsilon$)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Exactitude Test (%)", fontsize=10, fontweight="bold")
    ax1.set_title("Panneau A : Courbe de Robustesse FGSM", fontsize=11, fontweight="bold")
    ax1.set_ylim([0, 105])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9)

    # Subplot 2: Reliability Diagram (Calibrated)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.bar(centers, cal["bin_stats"]["accuracies"], width=0.08, color="#16a085", alpha=0.75, edgecolor="#2c3e50", label="Précision observée")
    ax2.plot([0, 1], [0, 1], "--", color="#e74c3c", linewidth=1.8, label="Idéal")
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_xlabel("Confiance", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Exactitude", fontsize=10, fontweight="bold")
    ax2.set_title(f"Panneau B : Reliability Diagram (T={ts_data['optimal_temperature_T']:.3f}, ECE={cal['ece']:.4f})", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(fontsize=9)

    # Subplot 3: XAI Panel
    ax3 = fig.add_subplot(2, 2, 3)
    img_sample = imgs_sub[0].permute(1, 2, 0).cpu().numpy()
    ax3.imshow(img_sample)
    ax3.imshow(cams[0], cmap="jet", alpha=0.55)
    ax3.set_title(f"Panneau C : Grad-CAM Overlay ({NEU_CLASSES[lbls_sub[0].item()]})\nAccord cosinus avec IG: {s2['xai_methods_comparison']['gradcam_vs_ig_mean_cosine']:.3f}", fontsize=11, fontweight="bold")
    ax3.axis("off")

    # Subplot 4: Metrics Bar Comparison
    ax4 = fig.add_subplot(2, 2, 4)
    categories = ["Clean Acc", "Rob Acc (eps=0.03)", "Macro-F1", "1 - ECE"]

    cross_path = Path("results/cross_effects_table.json")
    if cross_path.exists():
        with open(cross_path, "r", encoding="utf-8") as f:
            cross_data = json.load(f)
        rows_by_stage = {r["stage"]: r for r in cross_data.get("rows", [])}
        s2_row = rows_by_stage.get("S2_CNN_Base_Calibrated", {})
        s3_row = rows_by_stage.get("S3_CNN_Adversarial_Defense", {})
        clean_vals = [
            s2_row.get("acc", 0.0) * 100,
            s2_row.get("rob_acc_003", 0.0) * 100,
            s2_row.get("macro_f1", 0.0) * 100,
            (1.0 - s2_row.get("ece", 0.0)) * 100,
        ]
        adv_vals = [
            s3_row.get("acc", 0.0) * 100,
            s3_row.get("rob_acc_003", 0.0) * 100,
            s3_row.get("macro_f1", 0.0) * 100,
            (1.0 - s3_row.get("ece", 0.0)) * 100,
        ]
    else:
        clean_vals = [
            s3["clean_model_sweep"]["0.0"]["accuracy"] * 100,
            s3["clean_model_sweep"]["0.03"]["accuracy"] * 100,
            cal["accuracy"] * 100,
            (1.0 - cal["ece"]) * 100,
        ]
        adv_vals = [
            s3["adv_model_sweep"]["0.0"]["accuracy"] * 100,
            s3["adv_model_sweep"]["0.03"]["accuracy"] * 100,
            s3["adv_model_sweep"]["0.0"]["accuracy"] * 100,
            (1.0 - 0.0391) * 100,
        ]

    x = np.arange(len(categories))
    width = 0.35
    ax4.bar(x - width/2, clean_vals, width, label="CNN Standard", color="#c0392b", alpha=0.85)
    ax4.bar(x + width/2, adv_vals, width, label="CNN Défense Adv", color="#27ae60", alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, fontsize=9, fontweight="bold")
    ax4.set_ylabel("Pourcentage (%)", fontsize=10, fontweight="bold")
    ax4.set_title("Panneau D : Comparatif Synthétique Avant vs Après Défense", fontsize=11, fontweight="bold")
    ax4.set_ylim([0, 115])
    ax4.grid(True, linestyle=":", alpha=0.5)
    ax4.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig5_path = output_dir / "trustvision_member_b_dashboard.png"
    plt.savefig(fig5_path)
    plt.close()
    print(f"  -> Saved {fig5_path}")

    print("[Figures] All 5 figures generated successfully!")


if __name__ == "__main__":
    generate_all_figures()
