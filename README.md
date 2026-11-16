# TrustVision

Pipeline d'inspection de qualité d'acier sur le dataset NEU-CLS (1 800 images 200x200, 6 classes de défauts, 300 images/classe).  
Module **IA Avancée** · ENSA Fès.

---

## Lots de Travail

- **Membre A (Étape S0)** : Vision Transformers, CLIP Zero-Shot & Linear Probe calibrée, attention mécanistique (`notebooks/exp_A.ipynb`, `src/clip_attn.py`).
- **Membre B (Étapes S2 & S3)** : Boîte à outils d'audit de confiance (`src/eval.py`, `src/xai_cal_adv.py`, `src/gradio_tab3.py`, `notebooks/exp_B.ipynb`).
  - *Thème 9* : Calibration (Temperature Scaling, ECE Bootstrap) & Explicabilité (Grad-CAM, Integrated Gradients).
  - *Thème 11* : Défense adversariale (FGSM, projection $L_\infty$, stabilité intra-pipeline des explications).
- **Membre C (Étapes S1 & S4)** : Contrastive Learning (SimCLR) & Continual Learning (EWC).

---

## Matrice Synthétique d'Effets Croisés (S0 à S4)

Les métriques consolidées sont exportées dans [`results/cross_effects_table.json`](results/cross_effects_table.json) :

| Étape | Modèle / Checkpoint | Acc Clean | Macro-F1 | ECE Calibré | Rob Acc ($\varepsilon=0.03$) | XAI Sim (vs ViT) | Stabilité Intra XAI | Statut |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S0** | CLIP Linear Probe | 99.44% | 0.9945 | 0.0056 | 21.11% | 1.000 | - | COMPLETED |
| **S1** | SimCLR Linear Probe | - | - | - | - | - | - | *En attente Membre C* |
| **S2** | Baseline CNN (Calibré) | 98.06% | 0.9806 | 0.0121 | 21.11% | 0.7663 | - | COMPLETED |
| **S3** | CNN Défense Adversariale | 93.89% | 0.9392 | 0.0379 | 76.11% | - | 0.9090 | COMPLETED |
| **S4** | Continual EWC | - | - | - | - | - | - | *En attente Membre C* |

Les rapports détaillés et visualisations haute résolution sont disponibles dans [`results/`](results/) et [`results/figures/`](results/figures/).

---

## Reproduction Rapide

### 1. Tests Unitaires
```bash
pytest tests/ -v
```

### 2. Entraînement & Expériences
```bash
# Entraînement CNN standard & adversarial
python -m src.train --seed 42 --epochs 15 --save-path checkpoints/cnn_base_seed42.pt
python -m src.train --seed 42 --epochs 15 --adversarial --adv-eps 0.03 --save-path checkpoints/cnn_adv_seed42.pt

# Exécution complète de l'audit et génération des figures
python -m src.run_experiments
python -m src.generate_figures
```

### 3. Inférence Interactive (Gradio Tab 3)
```python
from src.gradio_tab3 import TrustVisionTab3Auditor
auditor = TrustVisionTab3Auditor()
result = auditor.audit(image_pil, temperature=1.2, epsilon=0.03)
```
