# Sélection des Modèles — NBA ML Prediction System
*Session 4 — Modélisation*

---

## 1. Définition des problèmes

### Moteur 1 — Prédiction Victoire / Défaite

| Élément | Détail |
|---|---|
| **Type** | Classification binaire supervisée |
| **Cible** | `WL_BIN` (1 = victoire, 0 = défaite) |
| **Granularité** | Une ligne = un match d'une équipe |
| **Classes** | Équilibrées (50.1% victoires) |
| **Horizon** | Prédiction avant le match (features rolling pré-match uniquement) |

### Moteur 2 — Prédiction des Awards (MVP, DPOY, ROY, 6MOY)

| Élément | Détail |
|---|---|
| **Type** | Ranking / classification binaire par groupe-saison |
| **Cible** | Binaire par award (1 = lauréat, 0 = non-lauréat) |
| **Granularité** | Une ligne = un joueur sur une saison |
| **Classes** | Très déséquilibrées (1 lauréat pour ~480 joueurs) |
| **Contrainte** | Le ranking est relatif à la cohorte de la saison |

---

## 2. Métriques d'évaluation

### Moteur 1

| Métrique | Formule simplifiée | Justification |
|---|---|---|
| **Log Loss** | $-\frac{1}{n}\sum y_i \log(\hat{p}_i)$ | Pénalise les prédictions confiantes mais fausses ; adapté si on veut des probabilités bien calibrées |
| **ROC-AUC** | Aire sous la courbe ROC | Mesure la capacité de séparation W/L indépendamment du seuil ; robuste au split temporel |

- Baseline Log Loss : **0.693** (prédire 0.5 partout)
- Baseline ROC-AUC : **0.5** (aléatoire)
- Objectif : Log Loss < 0.60, ROC-AUC > 0.70

### Moteur 2

| Métrique | Définition | Justification |
|---|---|---|
| **Top-1 Accuracy** | % de saisons où le joueur prédit #1 est le vrai lauréat | L'award ne va qu'à une personne — c'est la métrique principale |
| **Precision@3** | Proportion de vrais finalistes dans le top-3 prédit | Les votes MVP sont souvent serrés ; être dans le top-3 a une valeur réelle |

- Baseline Top-1 Accuracy : **~0.2%** (aléatoire parmi ~480 joueurs)
- Objectif : Top-1 Accuracy ≥ 0.5 sur 2 saisons test (≥ 1 saison sur 2)

---

## 3. Protocole de comparaison

### Split temporel (identique pour les deux moteurs)

```
Train : saisons 2014-15 → 2021-22  (8 saisons)
Test  : saisons 2022-23 → 2023-24  (2 saisons)
```

Justification : éviter le **data leakage temporel** — un modèle entraîné sur
des matchs futurs pour prédire des matchs passés serait artificellement bon.

### Protocole

1. Entraîner chaque modèle sur le train set avec les hyperparamètres par défaut
2. Calculer les métriques sur le test set via `src/metrics.py`
3. Comparer via `engine1_compare_models()` / `engine2_compare_models()`
4. Sélectionner le meilleur modèle par moteur
5. (Optionnel) Affiner les hyperparamètres du meilleur modèle par GridSearchCV

---

## 4. Modèles sélectionnés — Moteur 1 (W/L)

> Priorité : équilibre interprétabilité / performance

### Modèle 1.A — Logistic Regression (baseline)

**Avantages**
- Interprétable directement : les coefficients indiquent l'impact de chaque feature sur la probabilité de victoire
- Probabilités bien calibrées nativement → bon Log Loss
- Rapide à entraîner, pas de risque de surapprentissage avec régularisation L2

**Limites**
- Suppose une relation linéaire entre les features et le log-odds → ne capte pas les interactions (ex: `BACK_TO_BACK` × `ROLL5_PTS`)
- Sensible à la corrélation entre features (ROLL5 et ROLL10 sont corrélés)

**Adéquation avec le problème**
Bon choix de baseline pour établir une borne inférieure de performance.
Les features rolling sont approximativement linéaires avec la victoire,
ce qui rend la LR compétitive malgré sa simplicité.

---

### Modèle 1.B — Random Forest

**Avantages**
- Capture les interactions non-linéaires entre features (ex: fatigue + forme récente)
- Robuste aux valeurs aberrantes et aux features corrélées
- `feature_importances_` permet d'identifier les features les plus prédictives
- Peu sensible aux hyperparamètres par défaut

**Limites**
- Probabilités moins bien calibrées que la LR (nécessite CalibratedClassifierCV pour un bon Log Loss)
- Moins interprétable qu'une LR au niveau individuel (pas de coefficients directs)
- Plus lent à entraîner sur 21 000 matchs

**Adéquation avec le problème**
Très adapté : les données NBA comportent des interactions importantes
(un back-to-back ne pénalise pas toutes les équipes de la même façon).
Le feature importance permettra de valider les hypothèses de l'EDA.

---

### Modèle 1.C — XGBoost

**Avantages**
- Généralement meilleure performance pure que le Random Forest sur données tabulaires
- Compatible SHAP → interprétabilité locale (pourquoi tel match prédit comme victoire ?)
- Gère nativement les valeurs manquantes (utile pour les masques IS_VALID)
- Régularisation intégrée (L1 + L2) → réduit le surapprentissage

**Limites**
- Plus sensible aux hyperparamètres (learning rate, max_depth, n_estimators)
- Temps d'entraînement plus élevé sans GPU
- Boîte noire sans SHAP

**Adéquation avec le problème**
Meilleur candidat pour la performance pure. Avec SHAP, on récupère
l'interprétabilité locale manquante. C'est le modèle qu'on s'attend à
voir gagner sur ROC-AUC.

---

## 5. Modèles sélectionnés — Moteur 2 (Awards)

> Priorité : interprétabilité (expliquer pourquoi un joueur gagne un award)

### Modèle 2.A — Logistic Regression (baseline)

**Avantages**
- Coefficients directement lisibles : "chaque point de PTS_AVG supplémentaire augmente les chances MVP de X%"
- Fonctionne bien sur des features déjà bien engineerées (rankings ligue, moyennes saison)
- Probabilités calibrées → bon Precision@3

**Limites**
- Ne gère pas les interactions entre features (ex: PTS_AVG élevé ET bon PLUS_MINUS)
- Classe très déséquilibrée (1/480) → nécessite `class_weight='balanced'`

**Adéquation avec le problème**
Baseline solide. Les awards MVP/DPOY sont souvent corrélés linéairement
avec des stats dominantes (PTS_AVG, RANK_PTS_AVG), ce qui favorise la LR.

---

### Modèle 2.B — Decision Tree

**Avantages**
- Totalement interprétable : on peut lire l'arbre et extraire des règles métier
  (ex: "si RANK_PTS_AVG ≤ 3 ET PLUS_MINUS_AVG > 8 → candidat MVP")
- Visualisable directement avec `plot_tree()`
- Capture les seuils naturels présents dans les votes NBA

**Limites**
- Variance élevée : sensible aux données d'entraînement, peut surfit
- Probabilités peu calibrées (feuilles pures → 0 ou 1)
- Moins performant que les ensembles sur des datasets déséquilibrés

**Adéquation avec le problème**
Excellent pour l'interprétabilité : les journalistes et fans NBA raisonnent
en règles ("si le meilleur scoreur EST aussi le meilleur +/-...").
Utile comme outil d'analyse même si pas le meilleur en performance.

---

### Modèle 2.C — Random Forest

**Avantages**
- Réduit la variance du Decision Tree par agrégation
- `feature_importances_` → quelles stats comptent le plus pour chaque award ?
- Gère bien le déséquilibre des classes avec `class_weight='balanced'`
- Meilleure performance attendue que le DT seul

**Limites**
- Moins interprétable qu'un Decision Tree unique
- Peut être dominé par des features corrélées (PTS_AVG et RANK_PTS_AVG)

**Adéquation avec le problème**
Meilleur compromis performance/interprétabilité pour le Moteur 2.
Le feature importance par award (MVP vs DPOY vs ROY) donnera des insights
métier intéressants pour l'analyse.

---

## 6. Récapitulatif

| Moteur | Modèle | Rôle | Métrique principale |
|---|---|---|---|
| **M1** | Logistic Regression | Baseline | ROC-AUC |
| **M1** | Random Forest | Robuste / feature importance | ROC-AUC |
| **M1** | XGBoost | Performance + SHAP | ROC-AUC + Log Loss |
| **M2** | Logistic Regression | Baseline interprétable | Top-1 Accuracy |
| **M2** | Decision Tree | Règles métier | Top-1 Accuracy |
| **M2** | Random Forest | Performance + feature importance | Top-1 Accuracy |

---

## 7. Structure du dossier `models/`

```
models/
├── model_selection.md        ← ce document
├── engine1/
│   ├── logistic_regression.pkl    (après entraînement)
│   ├── random_forest.pkl
│   └── xgboost.pkl
└── engine2/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    └── random_forest.pkl
```

Les fichiers `.pkl` seront générés lors de la session de modélisation
via `src/model_io.py` (déjà présent dans le projet).
