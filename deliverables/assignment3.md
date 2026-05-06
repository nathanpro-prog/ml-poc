# Assignment 3

## 1. Définition du problème ML

**Moteur 1** — Classification binaire : prédire si une équipe va gagner un match (`WL_BIN`) à partir de ses statistiques de forme récente (rolling 5/10 matchs) et du contexte du match (domicile, repos, back-to-back).
Données : `matches_features.csv` — 27 313 lignes, ~50 features.

**Moteur 2** — Classification binaire : prédire si un joueur remporte un award (MVP, DPOY, ROY, 6MOY) à partir de ses statistiques agrégées sur la saison. Un modèle par award.
Données : `awards_features_labeled.csv` — 4 654 lignes, 25 features, 4 cibles.

---

## 2. Métriques d'évaluation

**Moteur 1** — `ROC-AUC` (principale), `Log Loss`, `Accuracy`.

**Moteur 2** — `Top-1 Accuracy` (le joueur classé #1 est-il le vrai lauréat ?), `Precision@3`, `ROC-AUC`.

Implémenté dans `src/metrics.py`.

---

## 3. Protocole d'évaluation

Split **temporel** : 8 saisons en train (2014-15 → 2021-22), 2 saisons en test (2022-23, 2023-24). Pas de k-fold aléatoire pour éviter le data leakage temporel.

Pour le Moteur 2, `class_weight='balanced'` est activé sur tous les modèles pour gérer le fort déséquilibre (~1 lauréat pour ~500 joueurs par saison).

Implémenté dans `src/data.py`.

---

## 4. Les trois modèles

Les mêmes trois modèles sont utilisés pour les deux moteurs.

### Logistic Regression — baseline

| | |
|---|---|
| **Hypothèses** | Relation linéaire entre les features et la cible. Nécessite une standardisation. |
| **Avantages** | Probabilités bien calibrées, coefficients interprétables, rapide. |
| **Limites** | Ne capture pas les interactions entre features. |
| **Adéquation** | Bonne baseline sur ROC-AUC et Log Loss. Sert de référence pour les deux moteurs. |

### Random Forest

| | |
|---|---|
| **Hypothèses** | Les patterns sont capturables par des combinaisons de seuils. L'agrégation de nombreux arbres réduit la variance. |
| **Avantages** | Robuste, gère les interactions, feature importance native, pas de standardisation. |
| **Limites** | Probabilités moins calibrées, moins interprétable qu'un modèle linéaire. |
| **Adéquation** | Bon équilibre performance/robustesse pour les deux moteurs. |

### XGBoost (M1) / Decision Tree (M2)

Le troisième modèle diffère selon le moteur.

**Moteur 1 — XGBoost**

| | |
|---|---|
| **Hypothèses** | Les résidus des arbres précédents sont corrigibles itérativement. |
| **Avantages** | Meilleure performance attendue sur ROC-AUC, compatible SHAP, gère nativement les NaN. |
| **Limites** | Plus sensible aux hyperparamètres, entraînement plus long. |
| **Adéquation** | Modèle principal du Moteur 1 pour maximiser le ROC-AUC. |

**Moteur 2 — Decision Tree**

| | |
|---|---|
| **Hypothèses** | La sélection d'un lauréat suit des règles de seuils explicites sur les stats. |
| **Avantages** | Règles entièrement lisibles, utile pour comprendre les critères de chaque award. |
| **Limites** | Variance élevée, moins performant que Random Forest. |
| **Adéquation** | Priorité à l'interprétabilité pour le Moteur 2. |

---

## 5. Justification du choix

Les trois modèles forment une progression en complexité (linéaire → ensemble → boosting/arbre) qui permet de mesurer le gain réel apporté par la complexité. Tous partagent l'interface scikit-learn (`fit`, `predict_proba`), ce qui garantit une comparaison équitable via `src/metrics.py`. Pour le Moteur 2, la priorité est l'interprétabilité d'où le remplacement de XGBoost par un Decision Tree.

---

## 6. Fichiers `.py`

| Fichier | Ce qui a été implémenté |
|---|---|
| `src/data.py` | `load_dataset_split()`, split temporel, masques `IS_VALID` |
| `src/metrics.py` | `compute_metrics()`, `engine1_compare_models()`, `engine2_compare_models()` |
| `src/model_io.py` | `save_model()`, `save_engine_models()`, `load_engine_models()` |

---

