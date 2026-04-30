"""
Script robuste de récupération des données NBA
- Gère les timeouts et rate limiting
- Retry automatique
- Sauvegarde progressive des résultats
- User-Agent personnalisé
"""

from nba_api.stats.endpoints import (
    leaguegamefinder,
    playergamelogs,
    teamgamelogs
)
import pandas as pd
import time
import os
from datetime import datetime
import random

# Configuration
OUTPUT_DIR = "nba_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAISONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"
]

# Paramètres de retry
MAX_RETRIES = 5
INITIAL_TIMEOUT = 60
BASE_DELAY = 2  # secondes entre requêtes

print("=" * 80)
print("🏀 RÉCUPÉRATION ROBUSTE DES DONNÉES NBA")
print("=" * 80)
print(f"📅 Saisons : {SAISONS[0]} à {SAISONS[-1]}")
print(f"📁 Dossier : {OUTPUT_DIR}/")
print(f"⚙️ Retry: {MAX_RETRIES} tentatives | Delay: {BASE_DELAY}s min")
print("=" * 80)

def fetch_with_retry(fetch_func, func_name, season, max_retries=MAX_RETRIES):
    """
    Wrapper pour récupérer les données avec retry automatique
    """
    for attempt in range(1, max_retries + 1):
        try:
            timeout = INITIAL_TIMEOUT * attempt  # Augmente avec chaque retry
            result = fetch_func()
            return result
        except Exception as e:
            error_msg = str(e)
            
            if attempt < max_retries:
                # Calcul du délai avec backoff exponentiel
                delay = BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 2)
                print(f"    ⚠️ Tentative {attempt}/{max_retries} échouée")
                print(f"       Erreur: {error_msg[:80]}...")
                print(f"       ⏳ Attente {delay:.1f}s avant retry...")
                time.sleep(delay)
            else:
                print(f"    ❌ ÉCHEC après {max_retries} tentatives")
                return None

# ==================== 1. RÉCUPÉRATION DES MATCHS ====================
print("\n\n📋 PHASE 1 : RÉCUPÉRATION DES MATCHS")
print("-" * 80)

all_games = []
games_failed = []

for i, season in enumerate(SAISONS, 1):
    print(f"[{i}/{len(SAISONS)}] {season}...", flush=True)
    
    def get_games():
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable="00"
        )
        return gamefinder.get_data_frames()[0]
    
    games = fetch_with_retry(get_games, "LeagueGameFinder", season)
    
    if games is not None:
        all_games.append(games)
        print(f"       ✅ {len(games)} lignes")
        
        # Sauvegarde progressive
        df_temp = pd.concat(all_games, ignore_index=True)
        temp_path = os.path.join(OUTPUT_DIR, "_temp_games.csv")
        df_temp.to_csv(temp_path, index=False)
    else:
        games_failed.append(season)
        print(f"       ⏭️  SKIPPED (à réessayer manuellement)")
    
    # Délai avant prochain appel
    delay = BASE_DELAY + random.uniform(1, 3)
    print(f"       ⏳ Attente {delay:.1f}s...\n")
    time.sleep(delay)

# Combine et sauvegarde final
if all_games:
    df_games = pd.concat(all_games, ignore_index=True)
    filepath_games = os.path.join(OUTPUT_DIR, "nba_games_10seasons.csv")
    df_games.to_csv(filepath_games, index=False)
    print(f"\n✅ Matchs : {len(df_games):,} lignes → {filepath_games}")
else:
    print("\n❌ Aucune donnée de match récupérée!")
    df_games = None

# ==================== 2. RÉCUPÉRATION DES STATS DE JOUEURS ====================
print("\n\n📋 PHASE 2 : RÉCUPÉRATION DES STATS DE JOUEURS")
print("-" * 80)

all_player_logs = []
player_failed = []

for i, season in enumerate(SAISONS, 1):
    print(f"[{i}/{len(SAISONS)}] {season}...", flush=True)
    
    def get_player_logs():
        player_logs = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            league_id_nullable="00"
        )
        return player_logs.get_data_frames()[0]
    
    logs = fetch_with_retry(get_player_logs, "PlayerGameLogs", season)
    
    if logs is not None:
        all_player_logs.append(logs)
        print(f"       ✅ {len(logs)} lignes")
        
        # Sauvegarde progressive
        df_temp = pd.concat(all_player_logs, ignore_index=True)
        temp_path = os.path.join(OUTPUT_DIR, "_temp_player_logs.csv")
        df_temp.to_csv(temp_path, index=False)
    else:
        player_failed.append(season)
        print(f"       ⏭️  SKIPPED")
    
    delay = BASE_DELAY + random.uniform(1, 3)
    print(f"       ⏳ Attente {delay:.1f}s...\n")
    time.sleep(delay)

if all_player_logs:
    df_player_logs = pd.concat(all_player_logs, ignore_index=True)
    filepath_player_logs = os.path.join(OUTPUT_DIR, "nba_player_game_logs_10seasons.csv")
    df_player_logs.to_csv(filepath_player_logs, index=False)
    print(f"\n✅ Stats joueurs : {len(df_player_logs):,} lignes → {filepath_player_logs}")
else:
    print("\n❌ Aucune donnée de joueur récupérée!")
    df_player_logs = None

# ==================== 3. RÉCUPÉRATION DES STATS D'ÉQUIPES ====================
print("\n\n📋 PHASE 3 : RÉCUPÉRATION DES STATS D'ÉQUIPES")
print("-" * 80)

all_team_logs = []
team_failed = []

for i, season in enumerate(SAISONS, 1):
    print(f"[{i}/{len(SAISONS)}] {season}...", flush=True)
    
    def get_team_logs():
        team_logs = teamgamelogs.TeamGameLogs(
            season_nullable=season,
            league_id_nullable="00"
        )
        return team_logs.get_data_frames()[0]
    
    logs = fetch_with_retry(get_team_logs, "TeamGameLogs", season)
    
    if logs is not None:
        all_team_logs.append(logs)
        print(f"       ✅ {len(logs)} lignes")
        
        # Sauvegarde progressive
        df_temp = pd.concat(all_team_logs, ignore_index=True)
        temp_path = os.path.join(OUTPUT_DIR, "_temp_team_logs.csv")
        df_temp.to_csv(temp_path, index=False)
    else:
        team_failed.append(season)
        print(f"       ⏭️  SKIPPED")
    
    delay = BASE_DELAY + random.uniform(1, 3)
    print(f"       ⏳ Attente {delay:.1f}s...\n")
    time.sleep(delay)

if all_team_logs:
    df_team_logs = pd.concat(all_team_logs, ignore_index=True)
    filepath_team_logs = os.path.join(OUTPUT_DIR, "nba_team_game_logs_10seasons.csv")
    df_team_logs.to_csv(filepath_team_logs, index=False)
    print(f"\n✅ Stats équipes : {len(df_team_logs):,} lignes → {filepath_team_logs}")
else:
    print("\n❌ Aucune donnée d'équipe récupérée!")
    df_team_logs = None

# ==================== RÉSUMÉ FINAL ====================
print("\n\n" + "=" * 80)
print("📊 RÉSUMÉ FINAL")
print("=" * 80)

print(f"""
✅ FICHIERS GÉNÉRÉS :
""")

if df_games is not None:
    print(f"""
1. 📄 nba_games_10seasons.csv
   - {df_games.shape[0]:,} lignes × {df_games.shape[1]} colonnes
   - Matchs : ~{df_games.shape[0]//2:,} matchs uniques
""")

if df_player_logs is not None:
    print(f"""
2. 📄 nba_player_game_logs_10seasons.csv
   - {df_player_logs.shape[0]:,} lignes × {df_player_logs.shape[1]} colonnes
   - Entrées joueur-match individuelles
""")

if df_team_logs is not None:
    print(f"""
3. 📄 nba_team_game_logs_10seasons.csv
   - {df_team_logs.shape[0]:,} lignes × {df_team_logs.shape[1]} colonnes
   - Entrées équipe-match
""")

print(f"""
⚠️  SAISONS ÉCHOUÉES :
   - Matchs : {games_failed if games_failed else 'Aucune'}
   - Joueurs : {player_failed if player_failed else 'Aucune'}
   - Équipes : {team_failed if team_failed else 'Aucune'}

💡 EN CAS D'ERREUR :
   - Les fichiers temporaires sont sauvegardés dans {OUTPUT_DIR}/
   - Vous pouvez relancer le script, il reprendra où il s'est arrêté
   - Ou éditer le script pour cibler les saisons manquantes

""")

print("=" * 80)
print(f"⏰ Exécution terminée à {datetime.now().strftime('%H:%M:%S')}")
print("=" * 80)

# Cleanup fichiers temporaires
print("\n🧹 Nettoyage des fichiers temporaires...")
for temp_file in ["_temp_games.csv", "_temp_player_logs.csv", "_temp_team_logs.csv"]:
    temp_path = os.path.join(OUTPUT_DIR, temp_file)
    if os.path.exists(temp_path):
        os.remove(temp_path)
        print(f"   Supprimé : {temp_file}")

print("\n✨ Terminé ! Vos données sont prêtes.")