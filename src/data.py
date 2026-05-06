"""
data.py — Pipeline de préparation des données pour les deux moteurs NBA.

Fonctions exposées :
    load_engine1_data(scale=True)  → X_train, X_test, y_train, y_test
    load_engine2_data(scale=True)  → X_train, X_test, y_train, y_test, groups_test

Paramètre scale :
    True  → StandardScaler appliqué (pour Logistic Regression)
    False → données brutes non scalées (pour Random Forest, XGBoost, Decision Tree)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
_ROOT    = Path(__file__).resolve().parent.parent
_MATCHES = _ROOT / "nba_data" / "processed" / "matches_features.csv"
_AWARDS  = _ROOT / "nba_data" / "processed" / "awards_features_labeled.csv"

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Split temporel : train = 2014-2021 (8 saisons), test = 2022-2023 (2 saisons)
TRAIN_SEASONS_MATCHES = list(range(2014, 2022))   # 2014..2021
TEST_SEASONS_MATCHES  = list(range(2022, 2024))   # 2022..2023

TRAIN_SEASONS_AWARDS = [
    "2014-15", "2015-16", "2016-17", "2017-18",
    "2018-19", "2019-20", "2020-21", "2021-22",
]
TEST_SEASONS_AWARDS = ["2022-23", "2023-24"]

# Features à exclure du Moteur 1 (identifiants, stats brutes du match courant,
# colonnes cible ou redondantes)
_ENGINE1_DROP = [
    "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
    "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "SEASON_YEAR",
    # Stats brutes du match en cours → data leakage
    "MIN", "PTS", "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF",
    "PLUS_MINUS",
]
_ENGINE1_TARGET = "WL_BIN"

# Colonnes rolling pouvant avoir des NaN en début de saison
_ROLL_COLS_ENGINE1 = [
    "ROLL5_PTS",  "ROLL10_PTS",
    "ROLL5_FG_PCT",  "ROLL10_FG_PCT",
    "ROLL5_FG3_PCT", "ROLL10_FG3_PCT",
    "ROLL5_FT_PCT",  "ROLL10_FT_PCT",
    "ROLL5_OREB",  "ROLL10_OREB",
    "ROLL5_DREB",  "ROLL10_DREB",
    "ROLL5_AST",   "ROLL10_AST",
    "ROLL5_STL",   "ROLL10_STL",
    "ROLL5_BLK",   "ROLL10_BLK",
    "ROLL5_TOV",   "ROLL10_TOV",
    "ROLL5_PLUS_MINUS", "ROLL10_PLUS_MINUS",
    "WIN_STREAK",
]

# Colonnes identifiants et cibles du Moteur 2
_ENGINE2_ID_COLS = ["PLAYER_ID", "PLAYER_NAME", "SEASON_YEAR", "TEAM_ID"]
_ENGINE2_TARGETS = ["MVP", "DPOY", "ROY", "6MOY"]

# Colonnes binaires/flags à ne pas scaler
_BINARY_COLS = {"HOME", "BACK_TO_BACK", "TOP5_PCT_FLAG"}


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _add_valid_mask(df: pd.DataFrame, roll_cols: list[str]) -> pd.DataFrame:
    """
    Pour chaque colonne rolling présente dans df :
      - Ajoute <col>_IS_VALID (1 = valeur réelle, 0 = imputée / début de saison)
      - Remplace les NaN par 0

    Cela permet aux modèles de distinguer les vraies valeurs nulles
    des valeurs artificiellement imputées en début de saison.
    """
    for col in roll_cols:
        if col in df.columns:
            df[f"{col}_IS_VALID"] = df[col].notna().astype(int)
    df[roll_cols] = df[roll_cols].fillna(0)
    return df


def _scale(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Applique StandardScaler sur les colonnes numériques continues.
    Les colonnes binaires / flags (HOME, BACK_TO_BACK, *_IS_VALID,
    TOP5_PCT_FLAG) sont exclues du scaling.

    Returns : X_train_scaled, X_test_scaled, scaler
    """
    binary_like = [
        c for c in X_train.columns
        if c in _BINARY_COLS
        or c.endswith("_IS_VALID")
        or c.endswith("_FLAG")
    ]
    num_cols = [
        c for c in X_train.select_dtypes(include="number").columns
        if c not in binary_like
    ]

    scaler  = StandardScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    return X_train, X_test, scaler


# ---------------------------------------------------------------------------
# Moteur 1 — Prédiction victoire / défaite
# ---------------------------------------------------------------------------

def load_engine1_data(
    scale: bool = True,
    matches_path: Path = _MATCHES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Charge et prépare les données pour le Moteur 1 (classification W/L).

    Paramètres
    ----------
    scale : bool (défaut True)
        True  → StandardScaler appliqué sur les features continues.
                 À utiliser pour la Logistic Regression.
        False → données non scalées.
                 À utiliser pour Random Forest et XGBoost
                 (invariants aux transformations monotones).

    Split temporel
    --------------
        train → saisons 2014-2021  (8 saisons)
        test  → saisons 2022-2023  (2 saisons)

    Traitement NaN
    --------------
        Les rolling averages vides en début de saison sont imputées à 0.
        Une colonne <feature>_IS_VALID est ajoutée pour signaler
        au modèle que ces valeurs sont artificielles.

    Returns
    -------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series  (0 = défaite, 1 = victoire)
    """
    df = pd.read_csv(matches_path)

    # Split temporel
    train = df[df["SEASON_YEAR"].isin(TRAIN_SEASONS_MATCHES)].copy()
    test  = df[df["SEASON_YEAR"].isin(TEST_SEASONS_MATCHES)].copy()

    # Masque validité + imputation NaN
    train = _add_valid_mask(train, _ROLL_COLS_ENGINE1)
    test  = _add_valid_mask(test,  _ROLL_COLS_ENGINE1)

    # Séparation X / y
    feature_cols = [
        c for c in train.columns
        if c not in _ENGINE1_DROP and c != _ENGINE1_TARGET
    ]
    X_train = train[feature_cols]
    X_test  = test[feature_cols]
    y_train = train[_ENGINE1_TARGET]
    y_test  = test[_ENGINE1_TARGET]

    # Scaling conditionnel
    if scale:
        X_train, X_test, _ = _scale(X_train, X_test)

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Moteur 2 — Prédiction Awards (MVP, DPOY, ROY, 6MOY)
# ---------------------------------------------------------------------------

def load_engine2_data(
    award: str = "MVP",
    scale: bool = True,
    awards_path: Path = _AWARDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray]:
    """
    Charge et prépare les données pour le Moteur 2 (ranking award).

    Paramètres
    ----------
    award : str
        Un parmi "MVP", "DPOY", "ROY", "6MOY".

    scale : bool (défaut True)
        True  → StandardScaler appliqué.
                 À utiliser pour la Logistic Regression.
        False → données non scalées.
                 À utiliser pour Decision Tree et Random Forest.

    Split temporel
    --------------
        train → saisons 2014-15 à 2021-22  (8 saisons)
        test  → saisons 2022-23 et 2023-24 (2 saisons)

    Returns
    -------
    X_train, X_test   : pd.DataFrame
    y_train, y_test   : pd.Series  (0 = non-lauréat, 1 = lauréat)
    groups_test       : np.ndarray des SEASON_YEAR du test set
                        → passé à engine2_top1_accuracy() et
                          engine2_precision_at_k() dans metrics.py
    """
    if award not in _ENGINE2_TARGETS:
        raise ValueError(
            f"award doit être l'un de {_ENGINE2_TARGETS}, reçu : '{award}'"
        )

    df = pd.read_csv(awards_path)

    # Split temporel
    train = df[df["SEASON_YEAR"].isin(TRAIN_SEASONS_AWARDS)].copy()
    test  = df[df["SEASON_YEAR"].isin(TEST_SEASONS_AWARDS)].copy()

    # Séparation X / y
    drop_cols    = _ENGINE2_ID_COLS + _ENGINE2_TARGETS
    feature_cols = [c for c in train.columns if c not in drop_cols]

    X_train = train[feature_cols]
    X_test  = test[feature_cols]
    y_train = train[award]
    y_test  = test[award]

    groups_test = test["SEASON_YEAR"].values

    # Scaling conditionnel
    if scale:
        X_train, X_test, _ = _scale(X_train, X_test)

    return X_train, X_test, y_train, y_test, groups_test


# ---------------------------------------------------------------------------
# Smoke test  →  python src/data.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("MOTEUR 1 — W/L")
    print("=" * 60)

    for sc, label in [(True, "scale=True  [LR]"), (False, "scale=False [RF/XGB]")]:
        X_tr, X_te, y_tr, y_te = load_engine1_data(scale=sc)
        print(f"\n  {label}")
        print(f"    Train : {X_tr.shape}  |  positifs : {y_tr.mean():.1%}")
        print(f"    Test  : {X_te.shape}  |  positifs : {y_te.mean():.1%}")

    print(f"\n  Features ({X_tr.shape[1]}) :")
    print(f"  {list(X_tr.columns)}")

    print("\n" + "=" * 60)
    print("MOTEUR 2 — AWARDS")
    print("=" * 60)

    for aw in ["MVP", "DPOY", "ROY", "6MOY"]:
        for sc, label in [(True, "scale=True  [LR]"), (False, "scale=False [DT/RF]")]:
            X_tr2, X_te2, y_tr2, y_te2, grp = load_engine2_data(award=aw, scale=sc)
            print(f"\n  {aw} | {label}")
            print(f"    Train : {X_tr2.shape}  |  lauréats : {int(y_tr2.sum())}")
            print(f"    Test  : {X_te2.shape}  |  lauréats : {int(y_te2.sum())}")
            print(f"    Saisons test : {np.unique(grp)}")