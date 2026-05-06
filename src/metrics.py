import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def engine1_log_loss(y_true, y_prob):
    return log_loss(y_true, y_prob)


def engine1_roc_auc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob)


def engine1_metrics(y_true, y_prob):
    return {
        "log_loss": engine1_log_loss(y_true, y_prob),
        "roc_auc": engine1_roc_auc(y_true, y_prob),
    }


def engine2_top1_accuracy(y_true, y_prob, groups):
    """
    y_true  : array binaire (1 = vainqueur)
    y_prob  : array de probabilités prédites
    groups  : array identifiant la saison de chaque ligne
    Retourne le % de saisons où le joueur prédit #1 est bien le vainqueur.
    """
    seasons = np.unique(groups)
    correct = 0
    for season in seasons:
        mask = groups == season
        predicted_winner = np.argmax(y_prob[mask])
        if y_true[mask][predicted_winner] == 1:
            correct += 1
    return correct / len(seasons)


def engine2_precision_at_k(y_true, y_prob, groups, k=3):
    """
    Retourne la précision moyenne dans le top-k prédit par saison.
    """
    seasons = np.unique(groups)
    precisions = []
    for season in seasons:
        mask = groups == season
        top_k_idx = np.argsort(y_prob[mask])[::-1][:k]
        precision = y_true[mask][top_k_idx].sum() / k
        precisions.append(precision)
    return np.mean(precisions)


def engine2_metrics(y_true, y_prob, groups, k=3):
    return {
        "top1_accuracy": engine2_top1_accuracy(y_true, y_prob, groups),
        f"precision_at_{k}": engine2_precision_at_k(y_true, y_prob, groups, k),
    }