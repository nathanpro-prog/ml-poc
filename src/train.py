"""
train.py
--------
Entraînement des modèles pour les deux moteurs NBA ML :
  - Moteur 1 : prédiction de victoire (WL_BIN) sur matches_features.csv
  - Moteur 2 : prédiction des awards (MVP, DPOY, ROY, 6MOY) sur awards_features_labeled.csv

Librairies utilisées :
  - scikit-learn  : modèles (LR, DT, RF, XGBoost)
  - optuna        : hyperparameter tuning (TPE sampler)
  - mlflow        : tracking des expériences
  - joblib        : sauvegarde pickle .pkl

Usage :
    python src/train.py --engine 1          # Moteur 1 uniquement
    python src/train.py --engine 2          # Moteur 2 uniquement
    python src/train.py --engine all        # Les deux (défaut)
    python src/train.py --engine all --tune # Activer Optuna (plus lent)
"""

import argparse
import os
import warnings
import logging

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
from optuna.samplers import TPESampler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    log_loss, roc_auc_score,
    accuracy_score, precision_score
)

from data import get_engine1_data, get_engine2_data
from model_io import save_model

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

 
RANDOM_STATE = 42
N_OPTUNA_TRIALS = 30        
MLFLOW_EXPERIMENT_E1 = "NBA_Engine1_WinPrediction"
MLFLOW_EXPERIMENT_E2 = "NBA_Engine2_AwardsPrediction"
MODEL_DIR_E1 = "models/engine1"
MODEL_DIR_E2 = "models/engine2"

AWARDS = ["MVP", "DPOY", "ROY", "6MOY"]



def evaluate_engine1(model, X_test, y_test) -> dict:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return {
        "log_loss": round(log_loss(y_test, y_pred_proba), 4),
        "roc_auc":  round(roc_auc_score(y_test, y_pred_proba), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
    }



def precision_at_k(y_true: pd.Series, y_proba: np.ndarray, k: int = 3) -> float:
    """Parmi les k joueurs les mieux classés, combien sont de vrais lauréats ?"""
    top_k_idx = np.argsort(y_proba)[-k:]
    return float(y_true.iloc[top_k_idx].sum()) / k


def evaluate_engine2(model, X_test, y_test: pd.Series) -> dict:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    top1_idx = np.argmax(y_pred_proba)
    top1_acc = int(y_test.iloc[top1_idx] == 1)
    p_at_3 = precision_at_k(y_test, y_pred_proba, k=3)
    try:
        auc = round(roc_auc_score(y_test, y_pred_proba), 4)
    except ValueError:
        auc = None  # Cas où une seule classe dans y_test
    return {
        "top1_accuracy":  top1_acc,
        "precision_at_3": round(p_at_3, 4),
        "roc_auc":        auc,
    }



def _objective_engine1(trial, model_name: str, X_train, y_train, X_val, y_val) -> float:
    """Minimise le log_loss sur le set de validation."""

    if model_name == "logistic_regression":
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        clf = LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_STATE)

    elif model_name == "random_forest":
        n_est = trial.suggest_int("n_estimators", 50, 400)
        max_d = trial.suggest_int("max_depth", 3, 15)
        min_s = trial.suggest_int("min_samples_split", 2, 20)
        clf = RandomForestClassifier(
            n_estimators=n_est, max_depth=max_d,
            min_samples_split=min_s, random_state=RANDOM_STATE, n_jobs=-1
        )

    elif model_name == "xgboost":
        lr   = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        n_est = trial.suggest_int("n_estimators", 50, 400)
        max_d = trial.suggest_int("max_depth", 2, 8)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        clf = XGBClassifier(
            learning_rate=lr, n_estimators=n_est, max_depth=max_d,
            subsample=subsample, use_label_encoder=False,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
        )

    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_val)[:, 1]
    return log_loss(y_val, proba)


# 
# OPTUNA — MOTEUR 2
# 
def _objective_engine2(trial, model_name: str, X_train, y_train, X_val, y_val) -> float:
    """Maximise precision@3 (retourné en négatif pour minimisation)."""

    if model_name == "logistic_regression":
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        clf = LogisticRegression(C=C, max_iter=1000,
                                  class_weight="balanced", random_state=RANDOM_STATE)

    elif model_name == "decision_tree":
        max_d = trial.suggest_int("max_depth", 2, 10)
        min_s = trial.suggest_int("min_samples_split", 2, 20)
        min_l = trial.suggest_int("min_samples_leaf", 1, 10)
        clf = DecisionTreeClassifier(
            max_depth=max_d, min_samples_split=min_s,
            min_samples_leaf=min_l, class_weight="balanced",
            random_state=RANDOM_STATE
        )

    elif model_name == "random_forest":
        n_est = trial.suggest_int("n_estimators", 50, 300)
        max_d = trial.suggest_int("max_depth", 2, 12)
        clf = RandomForestClassifier(
            n_estimators=n_est, max_depth=max_d,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )

    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_val)[:, 1]
    return -precision_at_k(pd.Series(y_val), proba, k=3)


def _build_engine1_model(name: str, params: dict):
    if name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params, max_iter=1000, random_state=RANDOM_STATE))
        ])
    elif name == "random_forest":
        return RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)
    elif name == "xgboost":
        return XGBClassifier(**params, use_label_encoder=False,
                              eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)


def _build_engine2_model(name: str, params: dict):
    if name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params, max_iter=1000,
                                       class_weight="balanced", random_state=RANDOM_STATE))
        ])
    elif name == "decision_tree":
        return DecisionTreeClassifier(**params, class_weight="balanced", random_state=RANDOM_STATE)
    elif name == "random_forest":
        return RandomForestClassifier(**params, class_weight="balanced",
                                       random_state=RANDOM_STATE, n_jobs=-1)


DEFAULT_PARAMS_E1 = {
    "logistic_regression": {"C": 1.0},
    "random_forest":       {"n_estimators": 200, "max_depth": 8, "min_samples_split": 5},
    "xgboost":             {"learning_rate": 0.05, "n_estimators": 200,
                            "max_depth": 4, "subsample": 0.8},
}

DEFAULT_PARAMS_E2 = {
    "logistic_regression": {"C": 1.0},
    "decision_tree":       {"max_depth": 5, "min_samples_split": 5, "min_samples_leaf": 2},
    "random_forest":       {"n_estimators": 200, "max_depth": 6},
}


def train_engine1(use_optuna: bool = False) -> dict:
    
    logger.info("═══ MOTEUR 1 — Prédiction de victoire ═══")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_E1)

    # Chargement des données via data.py
    (X_train, y_train), (X_test, y_test) = get_engine1_data()
    logger.info(f"Train : {X_train.shape} | Test : {X_test.shape}")

    # Split val sur train pour Optuna (20% du train)
    val_size = int(len(X_train) * 0.2)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_tr,  y_tr  = X_train[:-val_size], y_train[:-val_size]

    models_e1 = ["logistic_regression", "random_forest", "xgboost"]
    results = {}

    for model_name in models_e1:
        logger.info(f"  → Modèle : {model_name}")
        with mlflow.start_run(run_name=f"E1_{model_name}"):

            # ── Optuna tuning ──
            if use_optuna:
                logger.info(f"    Optuna : {N_OPTUNA_TRIALS} trials...")
                study = optuna.create_study(
                    direction="minimize",
                    sampler=TPESampler(seed=RANDOM_STATE)
                )
                study.optimize(
                    lambda trial: _objective_engine1(
                        trial, model_name, X_tr, y_tr, X_val, y_val
                    ),
                    n_trials=N_OPTUNA_TRIALS,
                    show_progress_bar=False
                )
                best_params = study.best_params
                mlflow.log_params({f"optuna_{k}": v for k, v in best_params.items()})
                mlflow.log_metric("optuna_best_val_logloss", study.best_value)
                logger.info(f"    Best params : {best_params}")
            else:
                best_params = DEFAULT_PARAMS_E1[model_name]

            # ── Entraînement final sur train complet ──
            model = _build_engine1_model(model_name, best_params)
            model.fit(X_train, y_train)

            # ── Évaluation ──
            metrics = evaluate_engine1(model, X_test, y_test)
            results[model_name] = metrics
            logger.info(f"    Métriques : {metrics}")

            # ── MLflow logging ──
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path=model_name)

            # ── Sauvegarde pickle ──
            pkl_path = os.path.join(MODEL_DIR_E1, f"{model_name}.pkl")
            save_model(model, pkl_path)

    logger.info("  ✅ Moteur 1 terminé")
    return results


def train_engine2(use_optuna: bool = False) -> dict:
    
    logger.info("═══ MOTEUR 2 — Prédiction des awards ═══")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_E2)

    models_e2 = ["logistic_regression", "decision_tree", "random_forest"]
    all_results = {}

    for award in AWARDS:
        logger.info(f"  ── Award : {award} ──")
        all_results[award] = {}

        (X_train, y_train), (X_test, y_test) = get_engine2_data(target=award)
        logger.info(f"  Train : {X_train.shape} | Positifs train : {y_train.sum()}")

        val_size = max(1, int(len(X_train) * 0.2))
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_tr,  y_tr  = X_train[:-val_size], y_train[:-val_size]

        for model_name in models_e2:
            logger.info(f"    → Modèle : {model_name}")
            with mlflow.start_run(run_name=f"E2_{award}_{model_name}"):

                if use_optuna:
                    logger.info(f"      Optuna : {N_OPTUNA_TRIALS} trials...")
                    study = optuna.create_study(
                        direction="minimize",
                        sampler=TPESampler(seed=RANDOM_STATE)
                    )
                    study.optimize(
                        lambda trial: _objective_engine2(
                            trial, model_name, X_tr, y_tr, X_val, y_val
                        ),
                        n_trials=N_OPTUNA_TRIALS,
                        show_progress_bar=False
                    )
                    best_params = study.best_params
                    mlflow.log_params({f"optuna_{k}": v for k, v in best_params.items()})
                    mlflow.log_metric("optuna_best_val_neg_p3",
                                      study.best_value)
                    logger.info(f"      Best params : {best_params}")
                else:
                    best_params = DEFAULT_PARAMS_E2[model_name]

                model = _build_engine2_model(model_name, best_params)
                model.fit(X_train, y_train)

                metrics = evaluate_engine2(model, X_test, y_test)
                all_results[award][model_name] = metrics
                logger.info(f"      Métriques : {metrics}")

                mlflow.log_param("award", award)
                mlflow.log_params(best_params)
                mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
                mlflow.sklearn.log_model(model, artifact_path=f"{award}_{model_name}")

                pkl_path = os.path.join(MODEL_DIR_E2, f"{award}_{model_name}.pkl")
                save_model(model, pkl_path)

    logger.info("  ✅ Moteur 2 terminé")
    return all_results


def print_summary(results_e1: dict | None, results_e2: dict | None) -> None:
    print("\n" + "═" * 60)
    if results_e1:
        print("📊 MOTEUR 1 — Résultats comparatifs")
        print(f"  {'Modèle':<25} {'Log Loss':>10} {'ROC-AUC':>10} {'Accuracy':>10}")
        print("  " + "─" * 57)
        for name, m in results_e1.items():
            print(f"  {name:<25} {m['log_loss']:>10.4f} {m['roc_auc']:>10.4f} {m['accuracy']:>10.4f}")

    if results_e2:
        print("\n📊 MOTEUR 2 — Résultats comparatifs")
        for award in AWARDS:
            print(f"\n  [{award}]")
            print(f"  {'Modèle':<25} {'Top-1 Acc':>10} {'Prec@3':>10} {'ROC-AUC':>10}")
            print("  " + "─" * 57)
            for name, m in results_e2[award].items():
                auc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "   N/A"
                print(f"  {name:<25} {m['top1_accuracy']:>10} {m['precision_at_3']:>10.4f} {auc_str:>10}")
    print("═" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Entraînement NBA ML Models")
    parser.add_argument(
        "--engine", type=str, default="all",
        choices=["1", "2", "all"],
        help="Moteur à entraîner : 1, 2, ou all (défaut)"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Activer Optuna hyperparameter tuning (plus lent)"
    )
    args = parser.parse_args()

    results_e1 = None
    results_e2 = None

    if args.engine in ("1", "all"):
        results_e1 = train_engine1(use_optuna=args.tune)

    if args.engine in ("2", "all"):
        results_e2 = train_engine2(use_optuna=args.tune)

    print_summary(results_e1, results_e2)


if __name__ == "__main__":
    main()
