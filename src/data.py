"""Student-owned dataset loading contract.

Students must implement ``load_dataset_split`` so that ``scripts/main.py`` can
evaluate every configured model on the same test split.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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

# Rolling stats used to build opponent features and differentials
_OPP_STATS = [
    "ROLL5_PTS", "ROLL5_FG_PCT", "ROLL5_FG3_PCT", "ROLL5_FT_PCT",
    "ROLL5_OREB", "ROLL5_DREB", "ROLL5_AST", "ROLL5_STL",
    "ROLL5_BLK", "ROLL5_TOV", "ROLL5_PLUS_MINUS",
]

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

# Season-normalized z-score versions (added by _add_zscore_features)
_ENGINE2_ZSCORE_FEATURES = [
    "Z_PTS_AVG", "Z_REB_AVG", "Z_AST_AVG", "Z_STL_AVG", "Z_BLK_AVG",
    "Z_PLUS_MINUS_AVG", "Z_FANTASY_AVG", "Z_DD2_RATE", "Z_CONSISTENCY",
    "Z_MIN_AVG", "Z_TOV_AVG",
]

_ENGINE2_TARGETS = ["MVP", "DPOY", "ROY", "6MOY"]

# Last N seasons held out for test (temporal split)
_TEST_SEASONS_ENGINE1 = 2   # last 2 seasons → ~20 % of 10
_TEST_SEASONS_ENGINE2 = 2

# ---------------------------------------------------------------------------
# Engine 2 — eligibility rules
# ---------------------------------------------------------------------------

# Minimum games played for any award candidate (removes injury-shortened seasons)
_GP_MIN = 65

# Maximum minutes per game for 6MOY bench-player classification
# All historical 6MOY winners average ≤ 32.2 min/game
_BENCH_MIN_THRESHOLD = 33.0


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
# Opponent features & z-score helpers
# ---------------------------------------------------------------------------

def _add_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join each team-game row with its opponent's rolling stats.


    For every game (identified by GAME_ID) there are exactly two rows: one
    for the home team (HOME=1) and one for the away team (HOME=0).  This
    function looks up the opponent's rolling stats and appends them as
    ``OPP_ROLL5_*`` columns.  It also computes ``DIFF_*`` columns
    (team stat minus opponent stat) which are strong predictors of outcome.

    Requires ``GAME_ID`` and ``HOME`` to be present in *df* (they are
    dropped later by ``_ENGINE1_DROP``).
    """
    # Build opponent lookup keyed by (GAME_ID, opponent_HOME)
    lookup_cols = ["GAME_ID", "HOME"] + [s for s in _OPP_STATS if s in df.columns]
    lookup = (
        df[lookup_cols]
        .copy()
        .rename(columns={s: f"OPP_{s}" for s in _OPP_STATS if s in df.columns})
    )
    # Flip HOME so (GAME_ID, 1) → (GAME_ID, 0) and vice versa; now the key
    # matches the *team's* (GAME_ID, HOME) and points to the opponent's stats.
    lookup["HOME"] = 1 - lookup["HOME"]

    result = df.merge(lookup, on=["GAME_ID", "HOME"], how="left")
    logger.debug("_add_opponent_features: added %d OPP/DIFF columns", len(_OPP_STATS) * 3)

    # Differential: team_stat − opponent_stat (positive → team is better)
    for stat in _OPP_STATS:
        if stat not in df.columns:
            continue
        opp_col  = f"OPP_{stat}"
        diff_col = "DIFF_" + stat[len("ROLL5_"):]   # e.g. DIFF_PTS
        result[diff_col] = result[stat].fillna(0.0) - result[opp_col].fillna(0.0)

    # IS_VALID masks for opponent features
    for stat in _OPP_STATS:
        opp_col   = f"OPP_{stat}"
        valid_col = f"{opp_col}_IS_VALID"
        if opp_col in result.columns:
            result[valid_col] = result[opp_col].notna().astype(np.int8)
            result[opp_col]   = result[opp_col].fillna(0.0)

    return result


def _add_zscore_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-season z-score normalised columns for key Engine-2 stats.

    For each stat in ``_ENGINE2_ZSCORE_FEATURES``, computes how many
    standard deviations above the season mean the player sits.  This makes
    the features relative within a season rather than absolute across eras,
    which is more informative for award prediction.

    Requires ``SEASON_YEAR`` to be present in *df*.
    """
    base_stats = [f[2:] for f in _ENGINE2_ZSCORE_FEATURES]  # strip "Z_"
    added = 0
    for stat in base_stats:
        if stat not in df.columns:
            continue
        z_col = f"Z_{stat}"
        df[z_col] = df.groupby("SEASON_YEAR")[stat].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8)
        )
        added += 1
    logger.debug("_add_zscore_features: added %d Z_* columns", added)
    return df


def _apply_eligibility_filter(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Apply award-specific eligibility rules to restrict the candidate pool.

    Rules applied in order:
    1. All awards: GP >= ``_GP_MIN`` (minimum games played).
    2. ROY only: IS_ROOKIE == 1 (first season in dataset).
       If the column is absent, derive it on-the-fly from PLAYER_ID.
    3. 6MOY only: IS_BENCH == 1 (MIN_AVG < ``_BENCH_MIN_THRESHOLD``).
       If the column is absent, derive it on-the-fly from MIN_AVG.
    """
    before = len(df)

    # Rule 1: minimum games played
    df = df[df["GP"] >= _GP_MIN].copy()

    # Rule 2: ROY — rookie-year players only
    if target == "ROY":
        if "IS_ROOKIE" not in df.columns:
            first_season = df.groupby("PLAYER_ID")["SEASON_YEAR"].transform("min")
            df["IS_ROOKIE"] = (df["SEASON_YEAR"] == first_season).astype(np.int8)
        df = df[df["IS_ROOKIE"] == 1].copy()

    # Rule 3: 6MOY — bench players only
    elif target == "6MOY":
        if "IS_BENCH" not in df.columns:
            df["IS_BENCH"] = (df["MIN_AVG"] < _BENCH_MIN_THRESHOLD).astype(np.int8)
        df = df[df["IS_BENCH"] == 1].copy()

    logger.info("Eligibility filter [%s]: %d → %d rows (GP>=%d%s)",
                target, before, len(df), _GP_MIN,
                ", IS_ROOKIE" if target == "ROY" else ", IS_BENCH" if target == "6MOY" else "")
    return df.reset_index(drop=True)


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
    logger.info("E1 raw data: %s rows from %s", len(df), path.name)

    # Add opponent differential features (requires GAME_ID & HOME, dropped later)
    df = _add_opponent_features(df)

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

    logger.info("E1 split: train=%s test=%s features=%d win_rate_train=%.3f",
                X_train.shape, X_test.shape, len(feature_cols), y_train.mean())
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
    logger.info("E2 raw data: %s rows from %s (target=%s)", len(df), path.name, target)

    # Add season-normalised z-score features BEFORE filtering so that z-scores
    # reflect the full league distribution (not just the eligible pool).
    df = _add_zscore_features(df)

    # Apply award-specific eligibility rules (GP minimum + ROY/6MOY restrictions)
    df = _apply_eligibility_filter(df, target)

    # Keep base + z-score features that are present in the file
    all_features = _ENGINE2_FEATURES + _ENGINE2_ZSCORE_FEATURES
    available_features = [c for c in all_features if c in df.columns]

    train_df, test_df = _temporal_split_engine2(df, n_test_seasons)

    X_train = train_df[available_features].reset_index(drop=True)
    X_test  = test_df[available_features].reset_index(drop=True)
    y_train = train_df[target].reset_index(drop=True)
    y_test  = test_df[target].reset_index(drop=True)

    logger.info("E2 split: train=%s test=%s features=%d positives_train=%d",
                X_train.shape, X_test.shape, len(available_features), int(y_train.sum()))
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


# ---------------------------------------------------------------------------
# train.py compatibility helpers
# ---------------------------------------------------------------------------

def get_engine1_data(
    data_path: Path | str | None = None,
    n_test_seasons: int = _TEST_SEASONS_ENGINE1,
) -> tuple[tuple, tuple]:
    """Return Engine-1 data as ``(X_train, y_train), (X_test, y_test)``.

    Wrapper around ``load_dataset_split_engine1`` that reorders the return
    value into the tuple-of-tuples format expected by ``src/train.py``.
    """
    X_train, X_test, y_train, y_test = load_dataset_split_engine1(
        data_path=data_path, n_test_seasons=n_test_seasons
    )
    return (X_train, y_train), (X_test, y_test)


def get_engine2_data(
    target: str = "MVP",
    data_path: Path | str | None = None,
    n_test_seasons: int = _TEST_SEASONS_ENGINE2,
) -> tuple[tuple, tuple]:
    """Return Engine-2 data as ``(X_train, y_train), (X_test, y_test)``.

    Wrapper around ``load_dataset_split_engine2`` that reorders the return
    value into the tuple-of-tuples format expected by ``src/train.py``.
    """
    X_train, X_test, y_train, y_test = load_dataset_split_engine2(
        target=target, data_path=data_path, n_test_seasons=n_test_seasons
    )
    return (X_train, y_train), (X_test, y_test)