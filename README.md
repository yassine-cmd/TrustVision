# TrustVision

Pipeline industriel d'inspection de qualité d'acier sur le jeu de données **NEU-CLS** (1 800 images, 6 classes de défauts).  
Projet intégrateur · Module **IA Avancée** · ENSA Fès · Pr. Oussama El Gannour.

---

## 🚀 Démarrage Rapide (En 3 Minutes)

### Option 1 : Google Colab (Recommandé pour l'évaluation)
1. Ouvrir [`notebooks/TrustVision_Master.ipynb`](notebooks/TrustVision_Master.ipynb) dans [Google Colab](https://colab.research.google.com).
2. Choisir le runtime : **CPU** suffit pour l'audit et la démo (passer en **GPU T4** uniquement pour ré-entraîner de zéro).
3. Exécuter les cellules dans l'ordre :
   * **Cellule 1** : Téléchargement du projet et vérification SHA-256 du découpage des données.
   * **Cellule 2** : Lancement de la suite de tests (`!pytest -v`). **94 tests validés en ~75 s**.
   * **Cellule 3** : Affichage de la matrice d'effets croisés et lancement du démonstrateur interactif Gradio (4 onglets).

### Option 2 : Exécution Locale (Terminal)
```bash
# 1. Cloner le projet et entrer dans le dossier
git clone https://github.com/votre-repo/TrustVision.git
cd TrustVision

# 2. Créer et activer l'environnement virtuel
python -m venv .venv
# Sur Windows :
.\.venv\Scripts\activate
# Sur Linux/Mac :
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer les 94 tests de régression (durée ~75 s sur CPU)
pytest -v

# 5. Lancer l'interface interactive Gradio
python -m src.gradio_app
```
L'interface s'ouvre directement à l'adresse : `http://localhost:7860`.

---

## 📊 Matrice d'Effets Croisés & Métriques Obtenues

Toutes les métriques ci-dessous sont issues du fichier consolidé [`results/cross_effects_table.json`](results/cross_effects_table.json) et vérifiées par tests automatiques :

| Étape | Modèle / Approche | Acc Clean | Macro-F1 | ECE (Calibré) | Rob Acc ($\varepsilon=0.03$) | XAI Sim (vs ViT) | Stabilité Intra-XAI | Forgetting | Latence Inférence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S0** | **CLIP Linear Probe** (Réf.) | **99.44%** | 0.9945 | **0.0056** | 21.11% | — *(Réf.)* | — | — | ~45 ms |
| **S1** | **SimCLR Linear Probe** (100%) | 97.22% | 0.9722 | 0.0087 | 15.00% | 0.7118 | — | — | 16.7 ms |
| **S2** | **Baseline CNN** ($T=0.9276$) | 98.06% | 0.9806 | 0.0121 | 21.11% | 0.7663 | — | — | 16.7 ms |
| **S3** | **CNN Défense Adversariale** | 93.89% | 0.9389 | 0.0306 | **76.11%** | — | **0.9090** | — | 16.7 ms |
| **S4** | **Replay Seul** (Tampon 40/classe) | **93.06%** | 0.9298 | 0.0245 | 15.00% | 0.7420 | 0.9650 | **6.67%** | 16.7 ms |
| **S4** | **EWC + Replay** ($\lambda=10$) | 86.39% | 0.8633 | 0.0765* | 15.00% | 0.7961 | 0.9831 | 7.22% | 16.7 ms |

*\*Note S4 : Pour EWC + Replay, la calibration a divergé (plancher $T=0.001$), déclenchant la règle de sécurité « No-Harm » qui conserve les probabilités brutes non recalibrées.*

### Ce qu'il faut retenir de ces résultats :
1. **Plafond de performance (S0)** : CLIP atteint 99.44% et sert d'oracle de référence pour la localisation spatiale d'attention (`clip_attn_test.npy`).
2. **Gain massif avec peu d'étiquettes (S1)** : À **1% d'étiquettes** (~2 images/classe), SimCLR atteint **73.24%** contre **25.28%** pour le CNN brut (**+47.96 points**, IC 95% `[46.90 ; 49.02]`). À **5%**, SimCLR atteint **88.06%** contre **33.89%** (**+54.17 points**).
3. **Audit de confiance & explicabilité (S2)** : Le Temperature Scaling réduit l'erreur de calibration de 0.038+ à **0.0121**. Grad-CAM et Integrated Gradients s'alignent avec l'attention de CLIP à **0.7663** de similarité cosinus.
4. **Défense robuste & stabilité XAI (S3)** : Sous attaque FGSM ($\varepsilon=0.03$), l'entraînement robuste fait passer l'exactitude de 21.11% à **76.11%**. Les explications visuelles restent stables à **0.9090** (stabilité intra-pipeline).
5. **Apprentissage continu honnête (S4)** : En extension de classes sans identifiant de tâche (3 $\to$ 6 classes), le **Replay seul domine l'hybride EWC + Replay** (93.06% vs 86.39%). La pénalité EWC sur-contraint le petit réseau, un résultat scientifique négatif documenté en toute transparence.
6. **Sécurité industrielle OOD** : Détection des anomalies hors-distribution (Mahalanobis + Énergie libre) avec une latence CPU de **16.7 ms**, largement sous la contrainte automate (PLC) de 30 ms.

---

## 🎛️ Démonstrateur Interactif Gradio (4 Onglets)

L'application [`src/gradio_app.py`](src/gradio_app.py) regroupe les 6 thèmes en 4 modules interactifs :

* **Onglet 1 · CLIP & Attention ViT (S0)** : Permet de tester différents prompts texte (générique vs ciblé acier) et superpose la carte d'attention des 12 couches de Vision Transformer sur l'échantillon.
* **Onglet 2 · SimCLR & Efficacité d'Annotation (S1)** : Affiche les courbes comparatives SimCLR vs Scratch (1%, 5%, 10%, 100%) et permet de tester l'inférence des sondes gelées.
* **Onglet 3 · Audit de Confiance, Défense & Sécurité OOD (S2/S3)** :
  * Curseur de **Température ($T$)** : Visualise la recalibration des probabilités.
  * Curseur **Epsilon ($\varepsilon$)** : Simule l'attaque FGSM et compare le modèle standard face au modèle robuste.
  * Cartes **Grad-CAM** & **Integrated Gradients** en direct.
  * **Aiguillage automatique Voie A / Voie B** : Rejette instantanément les images corrompues ou hors-domaine.
* **Onglet 4 · Apprentissage Continu EWC (S4)** : Curseur **Lambda ($\lambda$)** illustrant en direct le compromis stabilité/plasticité entre rétention des anciennes classes (1–3) et acquisition des nouvelles classes (4–6).

---

## 🛡️ Reproductibilité & Tests Automatisés

Le dépôt est protégé par une suite de **94 tests automatisés** exécutables en **~75 secondes sur CPU** :

```bash
pytest -v
```

### Répartition des 94 tests :
* **`tests/test_colab_e2e.py` (7 tests)** : Vérifie la syntaxe AST des 4 notebooks, l'absence de chemins absolus en dur, et la présence de tous les checkpoints et fichiers de résultats.
* **`tests/test_gradio_e2e.py` (21 tests)** : Valide le fonctionnement des 4 onglets, des curseurs et du graphe d'événements Gradio sans avoir à lancer de serveur web.
* **`tests/test_exp_a.py` (16 tests)** : Vérifie l'algèbre des têtes d'attention $QKV$, l'extraction ViT et le hachage SHA-256 du split.
* **`tests/test_exp_b.py` (15 tests)** : Vérifie les contrats CNN, les bornes de bruit FGSM $[0, 1]$, la préservation du classement après calibration et Grad-CAM.
* **`tests/test_exp_c.py` (22 tests)** : Vérifie la perte NT-Xent SimCLR, l'extension de tête 3 $\to$ 6 classes, la matrice de Fisher et le garde-fou « No-Harm » de calibration.
* **`tests/test_ood.py` (13 tests)** : Vérifie l'interception des NaN/Inf, la détection Mahalanobis/Énergie et le respect de la latence industrielle (< 30 ms).

### Garanties de Reproductibilité :
1. **Découpage cryptographié** : `configs/splits.json` verrouille le split 70/10/20 avec son empreinte SHA-256 : `2848c9ae7cb7c2ef4197ce18e5dd4352d28430724b46953d27b61340921a510c`.
2. **Graines figées** : Seeds fixées à `[42, 123, 999]` avec `torch.backends.cudnn.deterministic = True`.
3. **Zéro dépendance réseau obligatoire** : Tous les checkpoints entraînés (~35 Mo au total) sont directement inclus dans `checkpoints/` pour une exécution immédiate hors ligne.

---

## 📁 Structure du Projet

```text
TrustVision/
├── checkpoints/             # Poids pré-entraînés inclus (.pt)
│   ├── cnn_base_seed42.pt   # Modèle de base S2
│   ├── cnn_adv_seed42.pt    # Modèle robuste S3
│   ├── ood_base_seed42.pt   # Détecteur Mahalanobis S2/S3
│   └── ewc_*.pt             # Checkpoints d'apprentissage continu S4
├── configs/
│   └── splits.json          # Découpage strict 70/10/20 (seeds 42, 123, 999)
├── data/
│   └── NEU-CLS/             # Jeu de données officiel (6 classes x 300 images)
├── notebooks/
│   ├── TrustVision_Master.ipynb  # Notebook maître d'évaluation et démo Colab
│   ├── exp_A.ipynb               # Travaux S0 (CLIP, Attention)
│   ├── exp_B.ipynb               # Travaux S2 & S3 (Confiance & Robustesse)
│   └── exp_C.ipynb               # Travaux S1 & S4 (SimCLR & EWC)
├── results/
│   ├── cross_effects_table.json  # Matrice d'effets croisés officielle S0-S4
│   ├── clip_attn_test.npy        # Cartes d'attention ViT pré-calculées
│   └── figures/                  # Graphiques et diagrammes de fiabilité
├── src/
│   ├── clip_attn.py         # Implémentation CLIP & calcul QKV
│   ├── data.py              # DataLoaders normalisés [0, 1]
│   ├── eval.py              # Évaluation unifiée evaluate_all()
│   ├── ewc.py               # Algorithmes EWC et replay
│   ├── gradio_app.py        # Application complète de télémétrie Gradio
│   ├── model.py             # Architecture standard TrustVisionCNN
│   ├── ood.py               # Pipeline de sécurité industrielle Mahalanobis/Énergie
│   ├── simclr.py            # Pré-entraînement contrastif SimCLR
│   └── xai_cal_adv.py       # Grad-CAM, Integrated Gradients & FGSM
└── tests/                   # 94 tests automatisés (pytest)
    ├── conftest.py          # Fixtures partagées
    ├── test_colab_e2e.py    # Tests Colab et intégrité des artefacts
    ├── test_gradio_e2e.py   # Tests E2E des 4 onglets de l'interface
    ├── test_exp_a.py        # Tests piste A (S0)
    ├── test_exp_b.py        # Tests piste B (S2/S3)
    ├── test_exp_c.py        # Tests piste C (S1/S4)
    └── test_ood.py          # Tests piste Sécurité Industrielle OOD
```
