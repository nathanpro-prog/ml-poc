Assignment 1 — NBA ML Prediction System
Description du projet

Système ML à 2 moteurs appliqué à 10 saisons NBA (2014-15 → 2023-24)
Moteur 1 : prédire le résultat d'un match (W/L) → détecter des erreurs de cotes bookmakers
Moteur 2 : prédire les vainqueurs des awards de fin de saison (MVP, DPOY, ROY...) → anticiper les votes selon les critères historiques


Définition du problème

Moteur 1 : classification binaire — cible WL (Victoire = 1, Défaite = 0)
Moteur 2 : ranking / classification multi-classe — cible = vainqueur de l'award par saison


Dataset

Source : NBA Stats API via le package Python nba_api
Période : saisons 2014-15 à 2023-24 (regular season + playoffs)
3 fichiers dans data/ :

nba_games_10seasons.csv — ~27 450 lignes, résultats matchs (W/L, box score)
nba_player_game_logs_10seasons.csv — ~296 000 lignes, stats individuelles par match
nba_team_game_logs_10seasons.csv — ~27 450 lignes, stats équipes enrichies avec rankings




Features disponibles

Moteur 1 : FG_PCT, FG3_PCT, PTS, REB, AST, STL, BLK, TOV, PLUS_MINUS
Moteur 2 : moyennes saison de PTS, REB, AST, STL, BLK, FG_PCT, PLUS_MINUS, DD2, TD3, win % de l'équipe
Features à construire : flag HOME (depuis MATCHUP), rolling averages 5 matchs, BACK_TO_BACK, DAYS_REST


EDA — Premiers résultats

Distribution WL : parfaitement équilibrée (50/50 par construction)
Avantage domicile : ~58 % de victoires sur 10 saisons
Features les plus corrélées au résultat : FG_PCT, TOV, OREB, PLUS_MINUS
Tendance notable : les tirs à 3 points ont augmenté de ~22 à ~35 tentatives/match entre 2014 et 2024
Anomalie COVID (2019-21) : avantage domicile quasi nul — à traiter séparément
Notebook complet : notebooks/01_eda_nba.ipynb — exécutable via jupyter notebook


Objectif business

Moteur 1 : comparer les probabilités du modèle aux cotes implicites des bookmakers pour identifier des value bets
Moteur 2 : fournir un pronostic data-driven des awards avant le vote officiel


Contexte machine learning

Moteur 1 : supervisé, split chronologique obligatoire (pas de split aléatoire — risque de leakage), test set = saison 2023-24
Moteur 2 : labels absents des CSV → collecte externe nécessaire (Basketball-Reference), validation leave-one-season-out
Modèles envisagés : Logistic Regression (baseline) → Random Forest → XGBoost / LightGBM


Métriques envisagées

Moteur 1 : Accuracy, ROC-AUC, Log Loss (prioritaire pour la calibration des probabilités), Brier Score
Moteur 2 : Top-1 Accuracy, Precision@3, Spearman Rank Correlation


Hypothèses, risques et limites

Hypothèses : les stats passées sont prédictives à court terme ; les critères de vote sont stables dans le temps
Risques :

Data leakage si split non chronologique
Blessures et transferts non capturés dans les données
Overfitting sur le Moteur 2 (seulement ~50 observations labelisées)
Saisons COVID atypiques (bulle, pas de public)


Limites :

Pas de données blessures / fatigue / back-to-backs (à construire)
Subjectivité des votes non modélisable
Labels Awards à collecter séparément