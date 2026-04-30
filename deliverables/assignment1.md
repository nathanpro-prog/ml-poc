1. Description du projet
Objectif : Créer un double moteur de prédiction :

Moteur de Matchs : Calculer les probabilités de victoire pour détecter des erreurs de côtes.

Moteur d'Awards : Modéliser les critères historiques des votants pour prédire les vainqueurs de fin de saison.

Features spécifiques : Ajout de stats "avancées" comme le Win Shares (combien de victoires un joueur apporte à son équipe) et le Usage Rate (combien de ballons un joueur touche).

2. Problématique ML
Classification : Prédire si une équipe gagne ou perd (0 ou 1).

Régression : Estimer les stats précises (ex: points d'un joueur) pour les paris "Props".

3. Données & Features
Sources : nba_api (Python) pour les stats officielles et NBA Stuffer pour les stats de calendrier/fatigue.

Variables clés :

Stats d'équipes (Net Rating, rythme de jeu).

Stats individuelles (Points, efficacité au tir).

Contexte (Match à domicile, jours de repos, blessures).

4. Analyse Exploratoire (EDA)
Statut : En attente de récupération des données.

Plan : La collecte via l'API sera finalisée à la prochaine séance.

Objectif : Nettoyer le dataset et créer une matrice de corrélation dans le notebook dédié.

5. Stratégie Machine Learning
Modèles envisagés : Random Forest et Régression Logistique (simples et efficaces).

Métrique de succès : Accuracy (pourcentage de bons pronostics).

6. Risques & Limites
Blessures : Un joueur star absent peut fausser la prédiction.

Psychologie : Le modèle ne voit pas la fatigue mentale ou l'enjeu d'un match de fin de saison.