# TrustVision

Pipeline d'inspection de qualité d'acier sur le dataset NEU-CLS (1 800 images 200x200, 6 classes de défauts, 300 images/classe).  
Module **IA Avancée** · ENSA Fès.

---

## Architecture de l'équipe et lots de travail

- **Membre A** : Étape S0 — Vision Transformers, VLM (CLIP Zero-Shot vs Calibrated Linear Probe), attention mécanistique (`notebooks/exp_A.ipynb`, `src/clip_attn.py`).
- **Membre B (ce lot)** : Boîte à outils d'audit de confiance appliquée à tous les checkpoints (`src/eval.py`, `src/xai_cal_adv.py`, `src/gradio_tab3.py`, `notebooks/exp_B.ipynb`).
  - **Thème 9** : IA explicable et fiable (Grad-CAM, Integrated Gradients, Temperature Scaling, ECE Bootstrap).
  - **Thème 11** : ML adversarial (Attaques FGSM, entraînement adversarial avec random start, robustesse, stabilité intra-pipeline des explications).
- **Membre C** : Étape S1 — Contrastive Learning (SimCLR) et Étape S4 — Continual Learning & Catastrophic Forgetting (EWC).

---

## Lot Membre B : Livrables & Résultats Clés

### 1. Checkpoints
- `checkpoints/cnn_base_seed42.pt` : CNN baseline (96x96, 4 blocs Conv, 256-d features) entraîné sur le split train (1 260 images).
  - **Clean Test Acc** : **98.06%**
  - **ECE (non calibré)** : **0.0129** (IC 95%: [0.0094, 0.0285])
  - **Robust Acc (FGSM eps=0.03)** : **21.39%**
- `checkpoints/cnn_adv_seed42.pt` : CNN défendu par entraînement adversarial (mélange propre + FGSM eps=0.03 + random start uniforme).
  - **Clean Test Acc** : **95.00%** (faible compromis de -3.06%)
  - **Robust Acc (FGSM eps=0.03)** : **76.94%** (**gain net de +55.55%**)

### 2. Audit de Confiance (Thème 9 & 11)
- **Calibration (Temperature Scaling)** :
  - $T^* = 0.9279$ optimisé par LBFGS uniquement sur le split de calibration (180 images).
  - Préservation stricte de l'argmax vérifiée (`argmax_invariant: True`, exactitude test strictement préservée à 98.06%).
  - Réduction de l'ECE à **0.0121** (IC 95%: [0.0079, 0.0263]).
- **Explicabilité (XAI)** :
  - Accord Grad-CAM vs Integrated Gradients : similarité cosinus de **0.7835 ± 0.0911**.
  - Erreur relative de complétude de l'IG : **0.0019** (complétude vérifiée).
  - Accord Grad-CAM avec l'attention ViT CLIP (S0 `results/clip_attn_test.npy`) : similarité cosinus de **0.7663**.
  - Stabilité intra-pipeline (Grad-CAM propre vs adversarial) : **0.8336** (maintien d'explications stables malgré la perturbation et la défense).

### 3. Matrice Comparatrice d'Effets Croisés (S0 à S4)
Résultats consolidés dans `results/cross_effects_table.json` :

| Étape | Modèle / Checkpoint | Acc Clean | Macro-F1 | ECE | Rob Acc ($\varepsilon=0.03$) | XAI Sim (vs ViT) | Stabilité Intra XAI | Statut |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S0** | CLIP Linear Probe | 99.44% | 0.9945 | 0.0056 | 21.11% | 1.000 | - | COMPLETED |
| **S1** | SimCLR Linear Probe | - | - | - | - | - | - | *En attente Membre C* |
| **S2** | Baseline CNN (Calibré) | 98.06% | 0.9806 | 0.0121 | 21.39% | 0.7663 | - | COMPLETED |
| **S3** | CNN Défense Adversariale | 95.00% | 0.9501 | 0.0391 | 76.94% | - | 0.8336 | COMPLETED |
| **S4** | Continual EWC | - | - | - | - | - | - | *En attente Membre C* |

---

## Reproduction & Commandes

### 1. Exécution des tests unitaires
```bash
pytest tests/test_exp_b.py -v
```

### 2. Entraînement des modèles
```bash
# CNN Baseline
python -m src.train --seed 42 --epochs 15 --save-path checkpoints/cnn_base_seed42.pt

# CNN Adversarial
python -m src.train --seed 42 --epochs 15 --adversarial --adv-eps 0.03 --save-path checkpoints/cnn_adv_seed42.pt
```

### 3. Exécution complète des expériences et génération des JSON
```bash
python -m src.run_experiments
```

### 4. Inférence Interactive Rapide pour Gradio Tab 3 (< 0.5s)
```python
from src.gradio_tab3 import TrustVisionTab3Auditor
auditor = TrustVisionTab3Auditor()
result = auditor.audit(image_pil, temperature=1.2, epsilon=0.03)
# result contient : predicted_class, confidence, perturbed_image, grad_cam_map, integrated_gradients_map, latency_ms
```

