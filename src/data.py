"""Student-owned dataset loading contract.

Students must implement ``load_dataset_split`` so that ``scripts/main.py`` can
evaluate every configured model on the same test split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MATCHES_PATH = _PROJECT_ROOT / "nba_data" / "processed" / "matches_features.csv"
_AWARDS_PATH  = _PROJECT_ROOT / "nba_data" / "processed" / "awards_features_labeled.csv"

# ---------------------------------------------------------------------------
# Engine 1 — constants
# ---------------------------------------------------------------------------

# Columns that are identifiers / raw stats / leak into the target
_ENGINE1_DROP = [
    "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
    "GAME_ID", "GAME_DATE", "MATCHUP", "WL",
    "MIN", "PTS", "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF",
    "PLUS_MINUS", "SEASON_YEAR",
]

_ENGINE1_TARGET = "WL_BIN"

# Rolling prefixes — used to generate IS_VALID masks
_ROLL_PREFIXES = [
    "ROLL5_PTS", "ROLL10_PTS",
    "ROLL5_FG_PCT", "ROLL10_FG_PCT",
    "ROLL5_FG3_PCT", "ROLL10_FG3_PCT",
    "ROLL5_FT_PCT", "ROLL10_FT_PCT",
    "ROLL5_OREB", "ROLL10_OREB",
    "ROLL5_DREB", "ROLL10_DREB",
    "ROLL5_AST", "ROLL10_AST",
    "ROLL5_STL", "ROLL10_STL",
    "ROLL5_BLK", "ROLL10_BLK",
    "ROLL5_TOV", "ROLL10_TOV",
    "ROLL5_PLUS_MINUS", "ROLL10_PLUS_MINUS",
]

# ---------------------------------------------------------------------------
# Engine 2 — constants
# ---------------------------------------------------------------------------

_ENGINE2_FEATURES = [
    "PTS_AVG", "REB_AVG", "AST_AVG", "STL_AVG", "BLK_AVG",
    "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS_AVG",
    "FANTASY_AVG", "MIN_AVG",
    "DD2_RATE", "TD3_RATE", "CONSISTENCY",
    "RANK_PTS_AVG", "RANK_AST_AVG", "RANK_REB_AVG",
    "RANK_BLK_AVG", "RANK_STL_AVG", "RANK_PLUS_MINUS_AVG",
    "TOP5_PCT_FLAG", "GP",
    "DD2_TOTAL", "TD3_TOTAL", "TOV_AVG",
]

_ENGINE2_TARGETS = ["MVP", "DPOY", "ROY", "6MOY"]

# Last N seasons held out for test (temporal split)
_TEST_SEASONS_ENGINE1 = 2   # last 2 seasons → ~20 % of 10
_TEST_SEASONS_ENGINE2 = 2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_validity_masks(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary IS_VALID flags for every rolling feature.

    A rolling value is considered valid when it is not NaN (i.e. enough
    games have been played to compute the window).  NaN values are then
    filled with 0 so the feature matrix contains no missing values.
    """
    for col in _ROLL_PREFIXES:
        if col in df.columns:
            mask_col = f"{col}_IS_VALID"
            df[mask_col] = df[col].notna().astype(np.int8)
            df[col] = df[col].fillna(0.0)
    return df


def _temporal_split_engine1(
    df: pd.DataFrame,
    n_test_seasons: int = _TEST_SEASONS_ENGINE1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split matches chronologically by SEASON_YEAR.

    The ``n_test_seasons`` most recent seasons form the test set; the rest
    is the training set.  Order within each split is preserved.
    """
    seasons_sorted = sorted(df["SEASON_YEAR"].unique())
    test_seasons   = set(seasons_sorted[-n_test_seasons:])
    mask_test      = df["SEASON_YEAR"].isin(test_seasons)
    return df[~mask_test].copy(), df[mask_test].copy()


def _temporal_split_engine2(
    df: pd.DataFrame,
    n_test_seasons: int = _TEST_SEASONS_ENGINE2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split awards data chronologically by SEASON_YEAR."""
    seasons_sorted = sorted(df["SEASON_YEAR"].unique())
    test_seasons   = set(seasons_sorted[-n_test_seasons:])
    mask_test      = df["SEASON_YEAR"].isin(test_seasons)
    return df[~mask_test].copy(), df[mask_test].copy()


# ---------------------------------------------------------------------------
# Public API — Engine 1
# ---------------------------------------------------------------------------

def load_dataset_split_engine1(
    data_path: Path | str | None = None,
    n_test_seasons: int = _TEST_SEASONS_ENGINE1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and split the **match-outcome** dataset (Engine 1).

    Parameters
    ----------
    data_path:
        Override the default path to ``matches_features.csv``.
    n_test_seasons:
        Number of most recent seasons to use as the test set.

    Returns
    -------
    X_train, X_test, y_train, y_test
        Feature DataFrames and binary target Series (``WL_BIN``).
        Rolling features include ``*_IS_VALID`` validity masks;
        NaN values in rolling columns are filled with 0.
    """
    path = Path(data_path) if data_path else _MATCHES_PATH
    df   = pd.read_csv(path)

    # Generate IS_VALID masks and fill NaNs before splitting
    df = _add_validity_masks(df)

    train_df, test_df = _temporal_split_engine1(df, n_test_seasons)

    # Feature matrix: drop metadata + target
    feature_cols = [
        c for c in train_df.columns
        if c not in _ENGINE1_DROP and c != _ENGINE1_TARGET
    ]

    X_train = train_df[feature_cols].reset_index(drop=True)
    X_test  = test_df[feature_cols].reset_index(drop=True)
    y_train = train_df[_ENGINE1_TARGET].reset_index(drop=True)
    y_test  = test_df[_ENGINE1_TARGET].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Public API — Engine 2
# ---------------------------------------------------------------------------

def load_dataset_split_engine2(
    target: str = "MVP",
    data_path: Path | str | None = None,
    n_test_seasons: int = _TEST_SEASONS_ENGINE2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and split the **award prediction** dataset (Engine 2).

    Parameters
    ----------
    target:
        Award label to predict.  One of ``"MVP"``, ``"DPOY"``,
        ``"ROY"``, ``"6MOY"``.
    data_path:
        Override the default path to ``awards_features_labeled.csv``.
    n_test_seasons:
        Number of most recent seasons to use as the test set.

    Returns
    -------
    X_train, X_test, y_train, y_test
        Feature DataFrames (25 columns) and binary target Series.
    """
    if target not in _ENGINE2_TARGETS:
        raise ValueError(
            f"Unknown target '{target}'. Choose from {_ENGINE2_TARGETS}."
        )

    path = Path(data_path) if data_path else _AWARDS_PATH
    df   = pd.read_csv(path)

    # Keep only feature columns that are present in the file
    available_features = [c for c in _ENGINE2_FEATURES if c in df.columns]

    train_df, test_df = _temporal_split_engine2(df, n_test_seasons)

    X_train = train_df[available_features].reset_index(drop=True)
    X_test  = test_df[available_features].reset_index(drop=True)
    y_train = train_df[target].reset_index(drop=True)
    y_test  = test_df[target].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Contract entry-point (scripts/main.py calls this)
# ---------------------------------------------------------------------------

def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return the dataset split used for model evaluation.

    Expected return value:
        A tuple ``(X_train, X_test, y_train, y_test)``.

    Constraints:
    - ``X_train`` and ``X_test`` must contain feature data in a format accepted
      by the trained models stored in ``config.MODELS``.
    - ``y_train`` and ``y_test`` must contain the corresponding targets.
    - ``y_test`` must align with the predictions produced by each loaded model.

    Default behaviour:
        Loads Engine 1 (match-outcome prediction, target ``WL_BIN``) with a
        2-season temporal hold-out.  To use Engine 2 call
        ``load_dataset_split_engine2(target=...)`` directly.
    """
    return load_dataset_split_engine1()