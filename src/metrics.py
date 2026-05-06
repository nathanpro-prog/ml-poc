"""Student-owned metrics contract.

Students must implement ``compute_metrics`` to return the evaluation metrics
that matter for their project.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Contract entry-point (scripts/main.py calls this)
# ---------------------------------------------------------------------------

def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return the metrics used to compare model performance.

    Expected return value:
        A dictionary mapping metric names to numeric values, for example:
        ``{"log_loss": 0.52, "roc_auc": 0.73}``.

    Constraints:
    - Every value must be numeric and convertible to ``float``.
    - Use the same metric set for every model so results remain comparable.
    - Keep metric names stable because they are written to
      ``results/model_metrics.csv``.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 / 1).  Accepts array-like or
        ``pandas.Series``.
    y_pred:
        Model output.  This function auto-detects its shape:

        * **1-D probabilities** (values in [0, 1]) — used for both
          ``log_loss`` and ``roc_auc``.
        * **1-D hard labels** (0 or 1) — only ``accuracy`` is reliable;
          ``log_loss`` and ``roc_auc`` are set to ``NaN``.
        * **2-D probability matrix** (shape ``[n, 2]``) — column 1 is
          taken as the positive-class probability.

    Returns
    -------
    dict[str, float]
        Keys: ``"log_loss"``, ``"roc_auc"``, ``"accuracy"``.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred)

    # Resolve 2-D probability matrix to 1-D positive probabilities
    if y_pred.ndim == 2:
        if y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
        else:
            raise ValueError(
                f"y_pred has unexpected shape {y_pred.shape}. "
                "Expected 1-D array or 2-D matrix with 2 columns."
            )

    y_pred = y_pred.ravel()

    # Detect whether y_pred contains probabilities or hard labels
    is_proba = not np.all(np.isin(y_pred, [0, 1]))

    if is_proba:
        # Clip to avoid log(0) edge-cases
        y_prob   = np.clip(y_pred, 1e-7, 1 - 1e-7)
        y_labels = (y_prob >= 0.5).astype(int)
        ll       = float(log_loss(y_true, y_prob))
        auc      = float(roc_auc_score(y_true, y_prob))
    else:
        y_labels = y_pred.astype(int)
        ll       = float("nan")
        auc      = float("nan")

    acc = float(accuracy_score(y_true, y_labels))

    return {
        "log_loss": ll,
        "roc_auc":  auc,
        "accuracy": acc,
    }


# ---------------------------------------------------------------------------
# Engine 1 helpers — match outcome prediction
# ---------------------------------------------------------------------------

def engine1_evaluate_model(
    model: Any,
    X_test: Any,
    y_test: Any,
) -> dict[str, float]:
    """Evaluate a single Engine-1 model and return its metrics.

    Uses ``predict_proba`` when available, otherwise falls back to
    ``predict`` (hard labels only).
    """
    if hasattr(model, "predict_proba"):
        y_pred = model.predict_proba(X_test)
    else:
        y_pred = model.predict(X_test)
    return compute_metrics(y_test, y_pred)


def engine1_compare_models(
    models: dict[str, Any],
    X_test: Any,
    y_test: Any,
) -> pd.DataFrame:
    """Compare multiple Engine-1 models on the same test split.

    Parameters
    ----------
    models:
        Mapping of ``{model_name: fitted_model}``.
    X_test, y_test:
        Held-out features and labels.

    Returns
    -------
    pd.DataFrame
        One row per model, columns: ``model``, ``log_loss``,
        ``roc_auc``, ``accuracy``.  Sorted by ``roc_auc`` descending.
    """
    rows = []
    for name, model in models.items():
        metrics = engine1_evaluate_model(model, X_test, y_test)
        rows.append({"model": name, **metrics})

    df = pd.DataFrame(rows)
    return df.sort_values("roc_auc", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Engine 2 helpers — award prediction
# ---------------------------------------------------------------------------

def _precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 3) -> float:
    """Fraction of the top-k ranked players that are true award winners."""
    top_k_idx = np.argsort(y_scores)[::-1][:k]
    return float(np.sum(y_true[top_k_idx]) / k)


def _top1_accuracy(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Whether the player ranked #1 by the model is a true winner."""
    top1_idx = int(np.argmax(y_scores))
    return float(y_true[top1_idx])


def engine2_compute_metrics(
    y_true: Any,
    y_scores: Any,
    k: int = 3,
) -> dict[str, float]:
    """Compute Engine-2 metrics for a single award target.

    Parameters
    ----------
    y_true:
        Binary ground-truth (1 = won the award this season).
    y_scores:
        Predicted probabilities (positive class).
    k:
        Number of top candidates for Precision@k.

    Returns
    -------
    dict[str, float]
        Keys: ``"top1_accuracy"``, ``"precision_at_k"``,
        ``"log_loss"``, ``"roc_auc"``.
    """
    y_true   = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores).ravel()

    # Guard: if only one class is present, AUC is undefined
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        auc = float("nan")
        ll  = float("nan")
    else:
        y_clipped = np.clip(y_scores, 1e-7, 1 - 1e-7)
        auc       = float(roc_auc_score(y_true, y_clipped))
        ll        = float(log_loss(y_true, y_clipped))

    return {
        "top1_accuracy":  _top1_accuracy(y_true, y_scores),
        f"precision_at_{k}": _precision_at_k(y_true, y_scores, k),
        "log_loss":       ll,
        "roc_auc":        auc,
    }


def engine2_evaluate_model(
    model: Any,
    X_test: Any,
    y_test: Any,
    k: int = 3,
) -> dict[str, float]:
    """Evaluate a single Engine-2 model on one award target."""
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)

    return engine2_compute_metrics(y_test, y_scores, k=k)


def engine2_compare_models(
    models: dict[str, Any],
    X_test: Any,
    y_test: Any,
    k: int = 3,
) -> pd.DataFrame:
    """Compare multiple Engine-2 models on the same award test split.

    Parameters
    ----------
    models:
        Mapping of ``{model_name: fitted_model}``.
    X_test, y_test:
        Held-out features and labels for one award target.
    k:
        Precision@k parameter.

    Returns
    -------
    pd.DataFrame
        One row per model, sorted by ``top1_accuracy`` then
        ``precision_at_k`` descending.
    """
    rows = []
    for name, model in models.items():
        metrics = engine2_evaluate_model(model, X_test, y_test, k=k)
        rows.append({"model": name, **metrics})

    df = pd.DataFrame(rows)
    sort_col = f"precision_at_{k}"
    return df.sort_values(
        ["top1_accuracy", sort_col], ascending=False
    ).reset_index(drop=True)