"""
metrics.py — Métriques d'évaluation pour les deux moteurs NBA.

Moteur 1 (classification W/L) :
    engine1_log_loss(y_true, y_prob)
    engine1_roc_auc(y_true, y_prob)
    engine1_metrics(y_true, y_prob)          → dict
    engine1_compare_models(results)          → pd.DataFrame classé

Moteur 2 (ranking awards) :
    engine2_top1_accuracy(y_true, y_prob, groups)
    engine2_precision_at_k(y_true, y_prob, groups, k=3)
    engine2_metrics(y_true, y_prob, groups)  → dict
    engine2_compare_models(results)          → pd.DataFrame classé
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


# ===========================================================================
# MOTEUR 1 — Prédiction Victoire / Défaite (classification binaire)
# ===========================================================================
# Métriques choisies :
#   • Log Loss  → pénalise les prédictions confiantes mais fausses ;
#                 adapté car on veut des probabilités bien calibrées
#                 (utile pour parier / prendre une décision)
#   • ROC-AUC   → mesure la capacité à séparer W et L indépendamment
#                 du seuil de décision ; robuste au déséquilibre de classes
# ===========================================================================

def engine1_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Log Loss pour le Moteur 1.
    y_prob : probabilités de victoire (classe 1), shape (n,)
    Valeur idéale → 0. Baseline (prédire 0.5 partout) ≈ 0.693.
    """
    return log_loss(y_true, y_prob)


def engine1_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    ROC-AUC pour le Moteur 1.
    y_prob : probabilités de victoire (classe 1), shape (n,)
    Valeur idéale → 1.0. Baseline (aléatoire) = 0.5.
    """
    return roc_auc_score(y_true, y_prob)


def engine1_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Calcule Log Loss + ROC-AUC en une seule passe.

    Returns
    -------
    {
        "log_loss": float,   # ↓ minimiser
        "roc_auc":  float,   # ↑ maximiser
    }
    """
    return {
        "log_loss": engine1_log_loss(y_true, y_prob),
        "roc_auc":  engine1_roc_auc(y_true, y_prob),
    }


def engine1_compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """
    Compare plusieurs modèles Moteur 1 côte à côte.

    Paramètre
    ---------
    results : dict
        Clé = nom du modèle, valeur = dict retourné par engine1_metrics().
        Ex: {
            "LogisticRegression": {"log_loss": 0.61, "roc_auc": 0.72},
            "RandomForest":       {"log_loss": 0.58, "roc_auc": 0.75},
            "XGBoost":            {"log_loss": 0.55, "roc_auc": 0.78},
        }

    Returns
    -------
    pd.DataFrame trié par ROC-AUC décroissant (meilleur modèle en premier).
    """
    df = pd.DataFrame(results).T
    df.index.name = "model"
    df = df[["log_loss", "roc_auc"]].astype(float)
    df = df.sort_values("roc_auc", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    return df


# ===========================================================================
# MOTEUR 2 — Prédiction Awards (ranking par saison)
# ===========================================================================
# Métriques choisies :
#   • Top-1 Accuracy  → % de saisons où le joueur prédit #1 est le vrai lauréat
#                       métrique principale car l'award ne va qu'à une personne
#   • Precision@3     → parmi les 3 joueurs les mieux classés par le modèle,
#                       combien sont dans le "vrai top 3" (finalistes réels) ?
#                       utile car les votes réels sont souvent proches
#
# Note : on travaille par groupe (saison) car le ranking est relatif à la
# cohorte de joueurs de cette saison, pas absolu.
# ===========================================================================

def engine2_top1_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
) -> float:
    """
    Top-1 Accuracy par saison pour le Moteur 2.

    Pour chaque saison, vérifie si le joueur avec la probabilité la plus haute
    est bien le lauréat (y_true == 1).

    Paramètres
    ----------
    y_true  : array binaire (1 = lauréat, 0 = non-lauréat)
    y_prob  : probabilités estimées d'être lauréat
    groups  : saison de chaque observation (ex: ['2022-23', '2022-23', ...])

    Returns
    -------
    float : proportion de saisons correctement prédites (entre 0 et 1)
    """
    y_true  = np.asarray(y_true)
    y_prob  = np.asarray(y_prob)
    groups  = np.asarray(groups)

    correct = 0
    seasons = np.unique(groups)

    for season in seasons:
        mask        = groups == season
        top1_idx    = np.argmax(y_prob[mask])
        correct    += int(y_true[mask][top1_idx] == 1)

    return correct / len(seasons)


def engine2_precision_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    k: int = 3,
) -> float:
    """
    Precision@K moyen par saison pour le Moteur 2.

    Pour chaque saison, calcule la proportion de vrais lauréats / finalistes
    parmi les k joueurs les mieux classés par le modèle.

    Note : dans notre contexte, y_true est binaire (1 seul lauréat par saison),
    donc Precision@3 = 1/3 si le lauréat est dans le top-3, 0 sinon.

    Returns
    -------
    float : Precision@K moyen sur toutes les saisons
    """
    y_true  = np.asarray(y_true)
    y_prob  = np.asarray(y_prob)
    groups  = np.asarray(groups)

    precisions = []
    seasons    = np.unique(groups)

    for season in seasons:
        mask      = groups == season
        topk_idx  = np.argsort(y_prob[mask])[::-1][:k]
        hits      = y_true[mask][topk_idx].sum()
        precisions.append(hits / k)

    return float(np.mean(precisions))


def engine2_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    k: int = 3,
) -> dict:
    """
    Calcule Top-1 Accuracy + Precision@K en une seule passe.

    Returns
    -------
    {
        "top1_accuracy":   float,   # ↑ maximiser
        "precision_at_k":  float,   # ↑ maximiser  (k=3 par défaut)
        "k":               int,
    }
    """
    return {
        "top1_accuracy":  engine2_top1_accuracy(y_true, y_prob, groups),
        "precision_at_k": engine2_precision_at_k(y_true, y_prob, groups, k),
        "k": k,
    }


def engine2_compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """
    Compare plusieurs modèles Moteur 2 côte à côte.

    Paramètre
    ---------
    results : dict
        Clé = nom du modèle, valeur = dict retourné par engine2_metrics().
        Ex: {
            "LogisticRegression": {"top1_accuracy": 0.5, "precision_at_k": 0.33, "k": 3},
            "DecisionTree":       {"top1_accuracy": 0.5, "precision_at_k": 0.50, "k": 3},
            "RandomForest":       {"top1_accuracy": 1.0, "precision_at_k": 0.67, "k": 3},
        }

    Returns
    -------
    pd.DataFrame trié par top1_accuracy décroissant.
    """
    df = pd.DataFrame(results).T
    df.index.name = "model"
    df = df[["top1_accuracy", "precision_at_k", "k"]].copy()
    df["top1_accuracy"]  = df["top1_accuracy"].astype(float)
    df["precision_at_k"] = df["precision_at_k"].astype(float)
    df = df.sort_values("top1_accuracy", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    return df


# ===========================================================================
# Smoke test (python src/metrics.py)
# ===========================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # --- Moteur 1 ---
    y_true1 = rng.integers(0, 2, size=100)
    y_prob1 = rng.uniform(0, 1, size=100)
    m1 = engine1_metrics(y_true1, y_prob1)
    print("=== Moteur 1 ===")
    print(f"  Log Loss : {m1['log_loss']:.4f}  (baseline ≈ 0.693)")
    print(f"  ROC-AUC  : {m1['roc_auc']:.4f}  (baseline = 0.5)\n")

    fake_results_1 = {
        "LogisticRegression": {"log_loss": 0.61, "roc_auc": 0.72},
        "RandomForest":       {"log_loss": 0.58, "roc_auc": 0.75},
        "XGBoost":            {"log_loss": 0.55, "roc_auc": 0.78},
    }
    print(engine1_compare_models(fake_results_1).to_string())

    # --- Moteur 2 ---
    print("\n=== Moteur 2 ===")
    seasons  = np.array(["2022-23"] * 50 + ["2023-24"] * 50)
    y_true2  = np.zeros(100, dtype=int)
    y_true2[10] = 1   # lauréat saison 1
    y_true2[73] = 1   # lauréat saison 2
    y_prob2  = rng.uniform(0, 1, size=100)
    y_prob2[10] = 0.99  # modèle parfait saison 1
    m2 = engine2_metrics(y_true2, y_prob2, seasons)
    print(f"  Top-1 Accuracy : {m2['top1_accuracy']:.2f}")
    print(f"  Precision@3    : {m2['precision_at_k']:.2f}\n")

    fake_results_2 = {
        "LogisticRegression": {"top1_accuracy": 0.5,  "precision_at_k": 0.33, "k": 3},
        "DecisionTree":       {"top1_accuracy": 0.5,  "precision_at_k": 0.50, "k": 3},
        "RandomForest":       {"top1_accuracy": 1.0,  "precision_at_k": 0.67, "k": 3},
    }
    print(engine2_compare_models(fake_results_2).to_string())