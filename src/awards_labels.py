import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


awards_data = [
    ("2014-15", "Stephen Curry",           "Kawhi Leonard",        "Andrew Wiggins",        "Lou Williams"),
    ("2015-16", "Stephen Curry",           "Kawhi Leonard",        "Karl-Anthony Towns",    "Jamal Crawford"),
    ("2016-17", "Russell Westbrook",       "Draymond Green",       "Malcolm Brogdon",       "Eric Gordon"),
    ("2017-18", "James Harden",            "Rudy Gobert",          "Ben Simmons",           "Lou Williams"),
    ("2018-19", "Giannis Antetokounmpo",   "Rudy Gobert",          "Luka Dončić",           "Lou Williams"),
    ("2019-20", "Giannis Antetokounmpo",   "Giannis Antetokounmpo","Ja Morant",             "Montrezl Harrell"),
    ("2020-21", "Nikola Jokić",            "Rudy Gobert",          "LaMelo Ball",           "Jordan Clarkson"),
    ("2021-22", "Nikola Jokić",            "Marcus Smart",         "Scottie Barnes",        "Tyler Herro"),
    ("2022-23", "Joel Embiid",             "Jaren Jackson Jr.",    "Paolo Banchero",        "Malcolm Brogdon"),
    ("2023-24", "Nikola Jokić",            "Rudy Gobert",          "Victor Wembanyama",     "Malik Monk"),
]

df_awards = pd.DataFrame(awards_data, columns=["SEASON_YEAR", "MVP", "DPOY", "ROY", "6MOY"])

award_flags = {}
for _, row in df_awards.iterrows():
    season = row["SEASON_YEAR"]
    for award in ["MVP", "DPOY", "ROY", "6MOY"]:
        player = row[award]
        key = (player, season)
        if key not in award_flags:
            award_flags[key] = {"MVP": 0, "DPOY": 0, "ROY": 0, "6MOY": 0}
        award_flags[key][award] = 1

df = pd.read_csv(ROOT / "nba_data/processed/awards_features.csv")
df["MVP"]  = 0
df["DPOY"] = 0
df["ROY"]  = 0
df["6MOY"] = 0

for (player, season), flags in award_flags.items():
    mask = (df["PLAYER_NAME"] == player) & (df["SEASON_YEAR"] == season)
    for award, val in flags.items():
        df.loc[mask, award] = val

df.to_csv(ROOT / "nba_data/processed/awards_features_labeled.csv", index=False)