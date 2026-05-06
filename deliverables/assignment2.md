# Assignment 2 — Préparation des données

## Nettoyage des données

**Moteur 1 (matchs)** : les rolling averages (`ROLL5_*`, `ROLL10_*`) et `WIN_STREAK` sont vides pour les premiers matchs de chaque saison. Ces NaN sont imputés à 0.

Les stats brutes du match en cours (`PTS`, `FGM`, `FG_PCT`, etc.) ont été supprimées — elles ne sont connues qu'après le match et causeraient un data leakage.

**Moteur 2 (awards)** : pas de valeurs manquantes. Les labels `MVP`, `DPOY`, `ROY`, `6MOY` étaient absents de la NBA API — ils ont été collectés manuellement sur Basketball-Reference et encodés en binaire dans `src/awards_labels.py`.

---

## Transformations appliquées

**Encoding** : aucun — toutes les features sont déjà numériques. `HOME`, `BACK_TO_BACK` et `WL_BIN` sont des entiers 0/1 natifs.

**Scaling** : StandardScaler appliqué sur les features continues, via le paramètre `scale=True/False` dans `src/data.py`. Les colonnes binaires (`HOME`, `BACK_TO_BACK`, `*_IS_VALID`) ne sont pas scalées.

Le scaler est fitté uniquement sur le train set pour éviter toute fuite d'information vers le test.

---

## Nouvelles features créées

**Moteur 1** :
- `ROLL5_*` / `ROLL10_*` : moyenne des 5 et 10 derniers matchs sur PTS, FG_PCT, AST, etc. — capturent la forme récente de l'équipe
- `HOME`, `DAYS_REST`, `BACK_TO_BACK`, `GAMES_LAST_7D`, `WIN_STREAK` : contexte du match (fatigue, avantage terrain, dynamique)
- `*_IS_VALID` : masque binaire ajouté dans `src/data.py` pour chaque rolling feature — vaut 0 si la valeur est imputée (début de saison), 1 si réelle

**Moteur 2** :
- `RANK_PTS_AVG`, `RANK_REB_AVG`, etc. : rang du joueur dans la ligue — comparables entre saisons contrairement aux stats brutes
- `DD2_RATE`, `TD3_RATE` : taux de double et triple-doubles
- `CONSISTENCY` : régularité des performances (écart-type inversé)
- `TOP5_PCT_FLAG` : 1 si le joueur est dans le top 5% de la ligue sur plusieurs stats clés

---

## Justification des choix

**Split temporel (8 saisons train / 2 test)** plutôt qu'aléatoire : un split aléatoire permettrait d'entraîner sur des matchs futurs pour prédire des matchs passés, ce qui est irréaliste et introduit du data leakage via les rolling averages.

**Masques IS_VALID plutôt que suppression des NaN** : supprimer les premières lignes de chaque saison ferait perdre ~3 000 lignes (14% du dataset). L'imputation à 0 avec masque conserve ces données tout en signalant leur particularité au modèle.

**Rankings ligue pour le Moteur 2** : les stats brutes ne sont pas comparables entre saisons à cause de l'évolution du jeu (inflation des scores post-2015). Un rang 1 en 2014 et en 2023 a la même signification.

---

## Alternatives testées et non retenues

**PCA** (section 14.3 du notebook, `plots/16_pca_analysis.png`) : les 3 premières composantes n'expliquent que 68% de la variance, et les composantes n'ont aucun sens métier NBA. Rejetée pour perte d'interprétabilité sans gain significatif.

**MinMaxScaler** : plus sensible aux valeurs extrêmes (performances atypiques fréquentes en NBA). StandardScaler retenu car plus robuste.

**Imputation par la médiane** : la médiane de la saison est calculée sur des matchs futurs — introduit du data leakage. Rejetée au profit de l'imputation à 0 avec masque.

---

## Impact attendu sur les modèles

La suppression des stats brutes force les modèles à apprendre de vraies tendances plutôt que de mémoriser les performances passées — évite un ROC-AUC artificiellement proche de 1.

Les rolling averages sont la principale source de signal pour le Moteur 1 (corrélation ~0.3-0.4 avec WL_BIN observée en EDA). Les rankings ligue sont les features les plus discriminantes pour le Moteur 2.

Le paramètre `scale=False` pour Random Forest et XGBoost est sans effet sur leurs performances (les arbres sont invariants au scaling) mais documente explicitement l'intention.

---

## Datasets — où ils sont et comment les charger

```
nba_data/processed/
├── matches_features.csv          # Moteur 1 — 27 313 lignes × 57 colonnes
├── awards_features.csv           # Moteur 2 sans labels
└── awards_features_labeled.csv   # Moteur 2 avec labels — généré par src/awards_labels.py
```

```python
from src.data import load_engine1_data, load_engine2_data

# Moteur 1
X_train, X_test, y_train, y_test = load_engine1_data(scale=True)

# Moteur 2
X_train, X_test, y_train, y_test, groups_test = load_engine2_data(award="MVP", scale=True)
```
