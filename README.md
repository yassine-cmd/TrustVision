# TrustVision

Pipeline d'inspection de qualité d'acier sur le dataset NEU-CLS (1 800 images 200x200, 6 classes de défauts, 300 images/classe).  
Module **IA Avancée** · ENSA Fès.

---

## Lots de Travail

- **Membre A (Étape S0)** : Vision Transformers, CLIP Zero-Shot & Linear Probe calibrée, attention mécanistique (`notebooks/exp_A.ipynb`, `src/clip_attn.py`).
- **Membre B (Étapes S2 & S3)** : Boîte à outils d'audit de confiance (`src/eval.py`, `src/xai_cal_adv.py`, `src/gradio_tab3.py`, `notebooks/exp_B.ipynb`).
  - *Thème 9* : Calibration (Temperature Scaling, ECE Bootstrap) & Explicabilité (Grad-CAM, Integrated Gradients).
  - *Thème 11* : Défense adversariale (FGSM, projection $L_\infty$, stabilité intra-pipeline des explications).
- **Membre C (Étapes S1 & S4)** : Contrastive Learning (SimCLR) & Continual Learning (EWC) (`src/simclr.py`, `src/ewc.py`, `src/run_member_c.py`, `notebooks/exp_C.ipynb`).
  - *Thème 8* : Pré-entraînement auto-supervisé, sondes linéaires à 1 / 5 / 10 / 100 % d'étiquettes, comparaison from scratch et CLIP.
  - *Thème 13* : Scénario single-head 3 → 6 classes sans identifiant de tâche, EWC vs naïf vs replay, balayage de lambda.

---

## Matrice Synthétique d'Effets Croisés (S0 à S4)

Les métriques consolidées sont exportées dans [`results/cross_effects_table.json`](results/cross_effects_table.json) :

| Étape | Modèle / Checkpoint | Acc Clean | Macro-F1 | ECE | Rob Acc ($\varepsilon=0.03$) | XAI Sim (vs ViT) | Stabilité Intra XAI | Forgetting | Statut |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S0** | CLIP Linear Probe | 99.44% | 0.9945 | 0.0056 | 21.11% | — (référence) | - | — | COMPLETED |
| **S1** | SimCLR Linear Probe | 97.22% | 0.9722 | 0.0087 | 15.00% | 0.7118 | - | — | COMPLETED |
| **S2** | Baseline CNN (Calibré) | 98.06% | 0.9806 | 0.0121 | 21.11% | 0.7663 | - | — | COMPLETED |
| **S3** | CNN Défense Adversariale | 93.89% | 0.9392 | 0.0379 | 76.11% | - | 0.9090 | — | COMPLETED |
| **S4** | EWC + Replay (single-head 3→6) | 86.39% | 0.8633 | 0.0765* | 15.00% | 0.7961 | 0.9831 | 7.22% | COMPLETED |

*Note : les ECE S0 et S2 sont post-calibration (T = 0.4184 et 0.9276). La valeur S3 du tableau (0.0379) correspond au modèle brut ; après temperature scaling (T = 0.6772), l'ECE descend à **0.0306**. Pour S4 (`ewc_replay`, λ = 10), le fit de température a atteint le plancher de l'optimiseur (T = 0,001), hors de la plage autorisée [0,1 ; 10], et la NLL de calibration passait de 0,3914 (T = 1) à 1,1885 : la calibration est rejetée par la règle no-harm de `src/member_c_utils.py` et l'ECE brute (**0,0765**) est reportée avec les probabilités non recalibrées. La ligne S4 correspond à la meilleure variante de la famille EWC (`ewc_replay`, λ = 10) ; le **replay seul** atteint 93,06 % et reste le meilleur modèle S4 toutes familles confondues. Les colonnes `XAI Sim` S1 (0,7118) et S4 (0,7961) ont été mesurées avec le protocole exact de S2 : Grad-CAM 96×96 contre l'attention CLS→patches moyenne sur 12 couches (`results/clip_attn_test.npy`), similarité cosinus, moyenne sur les 360 images de test. S0 sert de référence et n'est donc pas comparée à elle-même ; la colonne `Forgetting` ne s'applique qu'à S4.*

Les rapports détaillés et visualisations haute résolution sont disponibles dans [`results/`](results/) et [`results/figures/`](results/figures/).

---

## Membre C — SimCLR (S1) & EWC (S4)

### S1 — Efficacité des étiquettes

Le backbone partagé `TrustVisionCNN` est pré-entraîné sans étiquettes avec la perte contrastive **NT-Xent** (deux vues augmentées : recadrage aléatoire, flips, rotation, jitter photométrique, flou gaussien). Le projecteur MLP est supprimé après pré-entraînement ; seules les représentations 256-d sont conservées.

Les sondes sont des régressions logistiques standardisées ajustées sur les features gelées, aux budgets **1 %, 5 %, 10 % et 100 %** du split d'entraînement. Chaque sonde est calibrée par temperature scaling sur le split de calibration partagé (10 %). Un **CNN entraîné from scratch** sur les mêmes budgets sert de baseline, et la **sonde CLIP** (S0) de référence externe. Le meilleur modèle à 100 % d'étiquettes — ici le **CNN from scratch** (97,78 %, devant la sonde SimCLR à 97,22 %) — est exporté comme modèle principal S1.

Artefacts : `results/s1_simclr.json`, `results/figures/s1_simclr_label_efficiency.png`, `checkpoints/simclr_backbone_6cls.pt`, `checkpoints/s1_main_model.pt`, et les sondes gelées `checkpoints/s1_simclr_probe_{1,5,10,100}pct.pt` ainsi que les CNN `checkpoints/s1_scratch_{1,5,10,100}pct.pt` pour l'inférence live par budget. La similarité XAI de la sonde 100 % est **0,7118** (Grad-CAM vs attention CLIP, protocole S2).

#### IC 95 % multi-seeds (1 % / 5 %)

Le protocole est répété sur les **trois splits verrouillés** de `configs/splits.json` (seeds 42, 123, 999). Un encodeur SimCLR est **ré-entraîné indépendamment sur le train non étiqueté de chaque split** : réutiliser l'encodeur du seed 42 ferait fuiter 264 images de son train dans le test du seed 123. Les CNN « from scratch » sont ré-entraînés sur les mêmes sous-ensembles exacts, avec la recette officielle (15 epochs, batch 32). L'IC est calculée par Student : moyenne ± t(0,975 ; 2) · s / √3, avec **t = 4,303** (n = 3 seeds, df = 2).

| Budget | Méthode | Seeds 42 / 123 / 999 | Moyenne | Écart-type | IC 95 % |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 % | SimCLR (sonde gelée) | 73.06 / 74.44 / 72.22 % | **73.24 %** | 1.12 pt | **[70.46 ; 76.02] %** |
| 1 % | CNN from scratch | 24.72 / 26.94 / 24.17 % | **25.28 %** | 1.47 pt | **[21.63 ; 28.92] %** |
| 5 % | SimCLR (sonde gelée) | 88.33 / 83.06 / 92.78 % | **88.06 %** | 4.87 pt | **[75.97 ; 100.14] %** |
| 5 % | CNN from scratch | 31.11 / 34.17 / 36.39 % | **33.89 %** | 2.65 pt | **[27.30 ; 40.48] %** |

Le gain apparié SimCLR − scratch (même seed, même split) est de **+47.96 points** à 1 % (IC 95 % [46.90 ; 49.02]) et **+54.17 points** à 5 % (IC 95 % [42.77 ; 65.57]). Le seed 42 reproduit exactement le JSON officiel (73.06 % et 88.33 %, vérifié par `seed42_reference_check`).

> La borne supérieure de 100.14 % à 5 % est la valeur brute de l'intervalle de Student, qui n'est pas borné par construction. L'accuracy étant bornée à 100 %, elle se lit « ≤ 100 % ». Avec n = 3, l'IC reste large : c'est une estimation honnête de la variabilité inter-splits, pas un intervalle de confiance asymptotique.

Artefacts : `results/s1_multiseed_ci.json` (métriques par seed, agrégats, delta apparié), `results/figures/s1_multiseed_ci.png`, encodeurs `checkpoints/simclr_backbone_6cls_seed{123,999}.pt` (le seed 42 réutilise l'encodeur officiel).

### S4 — Apprentissage continu sans fuite

Pour éviter toute fuite des nouvelles classes, un **encodeur SimCLR distinct** est pré-entraîné uniquement sur les images des **3 classes initiales** (`crazing`, `inclusion`, `patches`). La tête 3 classes est entraînée sur la tâche 1, la matrice de Fisher diagonale est estimée sur ces données puis **normalisée par tenseur** (moyenne = 1) pour corriger l'écart de plusieurs ordres de grandeur entre couches profondes et superficiales. La tête est ensuite étendue de **3 à 6 sorties**, les nouvelles lignes étant initialisées sans contrainte EWC.

Quatre familles sont comparées sans identifiant de tâche à l'inférence : **fine-tuning naïf**, **EWC** (balayage de lambda sur Fisher normalisée), **replay** (buffer de 120 images, **40 par ancienne classe**) et **EWC + replay** (meilleur lambda EWC combiné au buffer). Les métriques incluent l'exactitude globale, l'exactitude des anciennes/nouvelles classes, le **forgetting**, l'ECE (temperature scaling avec repli $T=1$ si la calibration dégrade la NLL) et la robustesse FGSM ($\varepsilon=0.03$).

**Résultats réels (seed 42, run Colab T4)** :

| Stratégie | Acc globale | Anciennes classes | Nouvelles classes | Forgetting |
| :--- | :---: | :---: | :---: | :---: |
| Fine-tuning naïf | 49.72% | 0.00% | 99.44% | 100.00% |
| Replay (40/classe) | **93.06%** | 93.33% | 92.78% | **6.67%** |
| EWC pur, meilleur λ = 10 | 55.56% | 14.44% | 96.67% | 85.56% |
| EWC + Replay (λ = 10) | 86.39% | 92.78% | 80.00% | 7.22% |

Le représentant retenu de la famille EWC est **EWC + Replay** (macro-F1 **0,8633**, ECE brute **0,0765**, accuracy FGSM **15,00%**, stabilité Grad-CAM intra-pipeline **0,9831**, similarité XAI vs CLIP **0,7961**). Le résultat honnête est que **EWC seul ne suffit pas** sur ce CNN en single-head : le meilleur λ (10) ne préserve que 14,44 % des anciennes classes. Le **replay** est le vrai moteur de la rétention (93,06 % d'exactitude globale, 93,33 % sur les anciennes classes, forgetting 6,67 %) et **domine l'hybride EWC + replay sur tous les axes** dans ce run (86,39 %, 92,78 %, forgetting 7,22 %). **Replay seul reste donc le meilleur modèle S4 en exactitude globale** ; l'hybride est la meilleure variante de la famille EWC. Le sweep λ complet est dans `results/s4_ewc.json` et la figure `results/figures/s4_ewc_lambda_sweep.png`.

Artefacts : `results/s4_ewc.json`, `results/figures/s4_ewc_lambda_sweep.png`, `checkpoints/simclr_backbone_3cls.pt`, `checkpoints/ewc_seed42.pt` (meilleur EWC pur), `checkpoints/ewc_plus_replay_seed42.pt` (hybride retenu), `checkpoints/ewc_lambda_*_seed42.pt`.

> Les checkpoints S4 (`ewc_seed42.pt`, `ewc_plus_replay_seed42.pt`, `ewc_lambda_*`, `ewc_naive_seed42.pt`, `ewc_replay_seed42.pt`, `ewc_task1_seed42.pt`, `ewc_fisher_task1_seed42.pt`) sont inclus dans `checkpoints/` et ont été rechargés localement : les métriques de `results/s4_ewc.json` sont reproduites au chiffre près (accuracy, macro-F1, old/new, forgetting, ECE, NLL, Brier, robustesse FGSM).

### Commandes

```bash
# Pipeline complet S1 + S4 + mise à jour de la matrice d'effets croisés
python -m src.run_member_c --stage all

# Étape S1 seule (sondes 1/5/10/100 %)
python -m src.run_member_c --stage s1

# Étape S4 seule (balayage lambda, Fisher normalisée par tenseur)
python -m src.run_member_c --stage s4 --ewc-lambdas 0.1 1 10 100 --fisher-normalization per_tensor

# Tests dédiés Membre C
pytest tests/test_exp_c.py -v

# Export des checkpoints 1/5/10 % (sans toucher aux 100 % officiels)
python -m scripts.export_s1_fraction_checkpoints

# IC 95 % multi-seeds (42, 123, 999) à 1 % et 5 %
python -m scripts.run_s1_multiseed

# Recalcul des XAI Sim S1/S4 avec le protocole S2
python -m scripts.compute_xai_sim
```

---

## Démonstrateur Gradio — 4 onglets

Le livrable d'intégration est le **notebook maître** [`notebooks/TrustVision_Master.ipynb`](notebooks/TrustVision_Master.ipynb), qui vérifie les artefacts, affiche les matrices S0–S4 et lance l'application [`src/gradio_app.py`](src/gradio_app.py).

| Onglet | Membre | Fonction |
| :--- | :--- | :--- |
| **1 · CLIP & Attention** | A | Prompt zero-shot, top-3 classes, attention **CLS → patches** (12 couches), rappel du score zero-shot (25,56 %) vs sonde calibrée (99,44 %) |
| **2 · SimCLR** | C | Courbes réelles 1/5/10/100 % et inférence avec les sondes gelées / CNN scratch exportés |
| **3 · XAI & FGSM** | B | Température, epsilon FGSM, Grad-CAM, Integrated Gradients, contrat de latence < 500 ms |
| **4 · EWC** | C | Curseur λ (0,1 / 1 / 10 / 100), métriques old/new/forgetting/ECE et inférence avec `ewc_lambda_*` |

```bash
# Dépendances validées (Gradio 4)
pip install "gradio==4.44.1" "pydantic==2.10.6" "fastapi==0.115.6" "starlette==0.41.3"

# Local
python -m src.gradio_app
# Colab : exécuter la section 5 de notebooks/TrustVision_Master.ipynb (share=True)
```

> Sur Colab, le premier clic de l'onglet 1 télécharge CLIP (~600 Mo). Les onglets 2 à 4 fonctionnent hors ligne avec les checkpoints du dépôt.

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

---

## Dataset & Environnement

### NEU-CLS

| Propriété | Valeur |
| :--- | :--- |
| Classes | `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches` |
| Images | 1 800 (300 par classe), niveaux de gris 200×200, converties en RGB puis redimensionnées à 96×96 pour le CNN |
| Split partagé | 1 260 train (70 %) / 180 calibration (10 %) / 360 test (20 %), seed 42 |
| SHA-256 de `configs/splits.json` | `2848c9ae7cb7c2ef4197ce18e5dd4352d28430724b46953d27b61340921a510c` |
| Téléchargement | Automatique via KaggleHub puis miroir Figshare (`src/data.py`), aucun fallback synthétique |

Le même contrat de split est utilisé par S0, S1, S2, S3 et S4. Les seeds 42, 123 et 999 sont pré-calculés dans `configs/splits.json` ; le seed 42 est la référence officielle.

### Versions testées

| Composant | Version |
| :--- | :--- |
| Python | 3.11 |
| PyTorch | 2.15.0+cpu (revalidation locale), build CUDA sur Colab T4 pour les runs officiels S1/S4 |
| torchvision | 0.29.2 |
| NumPy | 2.4.6 |
| scikit-learn | 1.9.1 |
| Pillow | 13.0.0 |
| matplotlib | 3.11.2 |
| Gradio | 4.44.1 |
| Transformers | 4.57.6 (CLIP, onglet 1) |

> Les checkpoints ont été rechargés et ré-évalués localement sur CPU. Les sondes S1 et les CNN « from scratch » 1/5/10 % **reproduisent le JSON officiel au chiffre près** (24,72 / 31,11 / 87,50 % pour les scratch, 73,06 / 88,33 / 90,56 % pour les sondes) ; le détail est consigné dans `results/s1_checkpoint_metrics.json` (`all_checks_passed: true`).

### Temps d'exécution mesurés

| Tâche | Matériel | Durée |
| :--- | :--- | :--- |
| Tests Membre C (`tests/test_exp_c.py`) | CPU | ~15 s |
| Export des checkpoints 1/5/10 % (3 sondes + 3 CNN scratch) | CPU | ~5 min |
| IC multi-seeds 1 %/5 % (3 seeds, encodeurs complets) | CPU | 2 314 s (~39 min) |
| S4 complet (7 stratégies, run 3) | Colab T4 | ~12–20 min (documenté dans `COLAB.md`) |
| S1 complet (pré-entraînement 20 epochs + 4 sondes + 4 scratch) | Colab T4 | non instrumenté dans le run officiel ; le coût dominant est le pré-entraînement contrastif |

---

## Effets croisés : questions & réponses

**1. Pourquoi EWC dégrade-t-il la calibration (ECE 0,0765) par rapport à S2 (0,0121) ?**  
L'élargissement single-head 3 → 6 et l'objectif EWC/replay déforment les logits, surtout sur les anciennes classes. Le fit de température a atteint le plancher LBFGS (T = 0,001) et la NLL de calibration est passée de 0,3914 à 1,1885 : la règle no-harm a donc rejeté la calibration et reporté les probabilités brutes. La conséquence pratique est qu'une recalibration par tâche (ou une calibration post-fusion) est nécessaire après chaque mise à jour continue.

**2. Pourquoi la robustesse FGSM de S4 (15,00 %) est-elle inférieure à S2 (21,11 %) et très loin de S3 (76,11 %) ?**  
Ni EWC ni le replay n'utilisent d'exemples adversariaux : ils préservent la rétention des anciennes classes, pas l'invariance aux perturbations. S3 est le seul modèle entraîné explicitement en adversarial (eps = 0,03), d'où ses 76,11 %. La combinaison replay + entraînement adversarial est la piste naturelle pour obtenir les deux propriétés.

**3. Pourquoi SimCLR (97,22 %) reste-t-il sous CLIP (99,44 %) ?**  
CLIP a été pré-entraîné sur des centaines de millions d'images web ; le SimCLR du projet est entraîné from scratch sur 1 260 images d'acier non étiquetées pendant 20 epochs. L'intérêt de SimCLR n'est pas de battre CLIP à 100 % d'étiquettes, mais l'efficacité des étiquettes : à 1 %, il atteint 73,06 % contre 24,72 % pour un CNN scratch ; à 5 %, 88,33 % contre 31,11 %.

**4. Pourquoi le replay seul (93,06 %) bat-il EWC + replay (86,39 %) ?**  
Sur un petit CNN single-head, le buffer stratifié ancre directement les anciennes entrées, tandis que la pénalité EWC tire les paramètres vers le modèle de la tâche 1 et limite l'apprentissage des nouvelles classes. L'hybride hérite des deux contraintes ; dans ce run, la contrainte EWC n'apporte pas de gain net. Le résultat est reporté tel quel, sans sélection cosmétique.

---

## Pistes d'extension

1. **SSL plus récents** : MoCo v3, BYOL, DINO ou multi-crop SimCLR pour améliorer les représentations à faible budget.
2. **Continual learning plus robuste** : DER++, LwF, Synaptic Intelligence, PackNet, ou détection task-free pour remplacer le single-head 3 → 6.
3. **Robustesse + rétention** : combiner replay et entraînement adversarial (TRADES, PGD) puis mesurer le compromis old/new/robustesse.
4. **Calibration continue** : vector scaling, calibration conditionnelle à la tâche, ou prédiction conforme pour des intervalles garantis.
5. **Industrialisation** : export ONNX/TorchScript, inférence par lot sur convoyeur, journalisation des dérives (data drift) et alertes qualité.
6. **XAI** : attention rollout, SHAP, explications contrefactuelles et évaluation quantitative de la fidélité au-delà de la similarité cosinus.
