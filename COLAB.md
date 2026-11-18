# TrustVision - Exécution Membre C sur Google Colab

Ce guide termine **S4 (EWC)** sur GPU Colab, réutilise les résultats réels de **S1 (SimCLR)** déjà produits, puis met à jour la matrice d'effets croisés S0-S4.

## Prérequis

- Un compte Google et un runtime Colab avec **GPU T4**.
- L'archive `D:\TrustVision_colab.zip` (projet sans `.git`, sans `data/`, avec les checkpoints S1 et S4 vérifiés).
- Le notebook `notebooks/exp_C.ipynb` (inclus dans l'archive).

> Ne pas exécuter `pip install -r requirements.txt` : le fichier épingle `torch==2.15.0` et remplacerait le build CUDA de Colab. Le notebook installe uniquement les paquets manquants.

## Étapes

1. Ouvrir <https://colab.research.google.com> puis **Fichier > Importer un notebook** et choisir `notebooks/exp_C.ipynb`.
2. **Exécution > Modifier le type d'exécution > T4 GPU**, puis vérifier avec `nvidia-smi`.
3. Téléverser `TrustVision_colab.zip` :
   - soit dans `/content` via le panneau **Fichiers** (glisser-déposer),
   - soit dans `Mon Drive/TrustVision_colab.zip` (recommandé pour la sauvegarde).
4. Exécuter les cellules **0.1 à 0.4** : montage Drive, extraction dans `/content/TrustVision`, vérification CUDA, installation des paquets manquants, `pytest tests/test_exp_c.py` (22 tests attendus).
5. Section **2** : `RUN_S1 = False` charge `results/s1_simclr.json` (run réel déjà terminé). Passer à `True` seulement pour refaire S1 sur GPU.
6. Section **3** : exécuter `run_s4_ewc(...)`. L'encodeur SimCLR 3 classes est ré-entraîné sur GPU, puis les stratégies naïve / replay (buffer stratifié 40 par classe) / EWC (λ = 0,1 / 1 / 10 / 100 sur Fisher normalisée par tenseur) / EWC + replay sont comparées. Compter ~12-20 min sur T4 (7 stratégies). Le fichier `results/s4_ewc.json` n'est écrit qu'à la fin : ne pas interrompre la cellule.
7. Section **4** : `update_cross_effects_table(s1, s4)` remplit les lignes S1 et S4 avec les mesures réelles.
8. Section **6** : sauvegarde de `results/` et `checkpoints/` dans `Mon Drive/TrustVision_results/run_<horodatage>/` (ou téléchargement d'un zip).

> Le temperature scaling est protégé par une règle **no-harm** : le T ajusté est borné à **[0,1 ; 10]** et, s'il n'améliore pas la NLL du split de calibration, le modèle repasse à **T = 1** et les métriques brutes sont reportées. Le JSON conserve `temperature_fitted`, `calibration_accepted` et les deux NLL. Cas observé sur `ewc_replay` (λ = 10) : T = 0,001 (plancher LBFGS), NLL de calibration 1,1885 contre 0,3914 à T = 1.

## Notebook maître et démonstrateur (4 onglets)

Le notebook `notebooks/TrustVision_Master.ipynb` est le point d'entrée de la soutenance :

1. Importer `notebooks/TrustVision_Master.ipynb` dans Colab (il est inclus dans l'archive).
2. Exécuter les cellules **0.1 à 0.4** : montage Drive, extraction, installation de `gradio`/`transformers`/`pytest`, vérification des artefacts.
3. Exécuter la section **1** : téléchargement NEU-CLS et vérification du SHA-256 des splits.
4. Exécuter la section **5** : `build_demo().launch(share=True)` ouvre les quatre onglets :
   - onglet 1 (A) : CLIP zero-shot + attention CLS → patches (télécharge ~600 Mo au premier clic) ;
   - onglet 2 (C) : courbes S1 et inférence avec `s1_simclr_probe_{1,5,10,100}pct.pt` / `s1_scratch_*` ;
   - onglet 3 (B) : température, FGSM, Grad-CAM, Integrated Gradients ;
   - onglet 4 (C) : curseur lambda chargeant `ewc_lambda_*_seed42.pt`.
5. La section **6** propose un mini-TP ipywidgets pour lire les stratégies S4 sans réentraînement.

> En cas de conflit de version, forcer `pip install "gradio==4.44.1"` : le notebook utilise l'API `gr.Blocks` / `gr.Tab` de Gradio 4.

## IC multi-seeds (optionnel, CPU)

Les IC 95 % à 1 % et 5 % sont déjà calculées dans `results/s1_multiseed_ci.json` (3 seeds, ré-entraînement complet de l'encodeur par split, ~39 min sur CPU). Pour les refaire :

```bash
python -m scripts.run_s1_multiseed --seeds 42 123 999 --fractions 0.01 0.05
```

La commande vérifie automatiquement que le seed 42 reproduit `results/s1_simclr.json` et écrit `results/figures/s1_multiseed_ci.png`.

## Artefacts à récupérer après le run

- `results/s4_ewc.json`
- `results/cross_effects_table.json` (mis à jour)
- `results/figures/s4_ewc_lambda_sweep.png`
- `results/xai_sim.json` (XAI Sim S1/S4)
- `results/s1_multiseed_ci.json` (IC 95 % multi-seeds)
- `results/figures/s1_multiseed_ci.png`
- `checkpoints/simclr_backbone_6cls_seed{123,999}.pt`
- `results/s1_checkpoint_metrics.json` (métriques propres des checkpoints exportés)
- `notebooks/TrustVision_Master.ipynb` + `src/gradio_app.py`
- `checkpoints/ewc_seed42.pt`
- `checkpoints/ewc_plus_replay_seed42.pt`
- `checkpoints/ewc_lambda_*_seed42.pt`
- `checkpoints/simclr_backbone_3cls.pt`
- `checkpoints/s1_simclr_probe_{1,5,10,100}pct.pt`
- `checkpoints/s1_scratch_{1,5,10,100}pct.pt`

Les recopier dans le dépôt local, puis mettre à jour la section Membre C du `README.md` avec les valeurs réelles.
