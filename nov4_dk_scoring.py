#!/usr/bin/env python3
"""
Calculate DraftKings NBA scoring for Nov 4, 2025 box scores.
Score all lineups from the DK entries CSV.
"""
import csv
import os

# DraftKings NBA Scoring Rules
# Point: +1, 3PM: +0.5, Rebound: +1.25, Assist: +1.5
# Steal: +2, Block: +2, Turnover: -0.5
# Double-Double: +1.5 bonus, Triple-Double: +3 bonus

def calc_dk_points(pts, reb, ast, stl, blk, tov, tpm):
    dk = pts * 1.0 + tpm * 0.5 + reb * 1.25 + ast * 1.5 + stl * 2.0 + blk * 2.0 - tov * 0.5
    # Double-double / Triple-double bonuses
    doubles = sum(1 for x in [pts, reb, ast] if x >= 10)
    if doubles >= 3:
        dk += 3.0  # Triple-double
    elif doubles >= 2:
        dk += 1.5  # Double-double
    return dk

# ============================================================
# BOX SCORES - Nov 4, 2025 (from basketball-reference.com)
# ============================================================

# Game 1: ATL 127, ORL 112
orl_players = [
    {"Name": "Paolo Banchero", "Team": "ORL", "MIN": 37, "PTS": 22, "REB": 11, "AST": 8, "STL": 1, "BLK": 0, "TOV": 3, "3PM": 0},
    {"Name": "Franz Wagner", "Team": "ORL", "MIN": 34, "PTS": 18, "REB": 5, "AST": 2, "STL": 0, "BLK": 0, "TOV": 4, "3PM": 0},
    {"Name": "Wendell Carter Jr.", "Team": "ORL", "MIN": 30, "PTS": 11, "REB": 5, "AST": 1, "STL": 0, "BLK": 0, "TOV": 0, "3PM": 1},
    {"Name": "Desmond Bane", "Team": "ORL", "MIN": 23, "PTS": 9, "REB": 3, "AST": 2, "STL": 1, "BLK": 0, "TOV": 3, "3PM": 2},
    {"Name": "Jalen Suggs", "Team": "ORL", "MIN": 20, "PTS": 12, "REB": 2, "AST": 4, "STL": 3, "BLK": 0, "TOV": 2, "3PM": 2},
    {"Name": "Anthony Black", "Team": "ORL", "MIN": 26, "PTS": 7, "REB": 4, "AST": 3, "STL": 1, "BLK": 1, "TOV": 1, "3PM": 1},
    {"Name": "Tristan da Silva", "Team": "ORL", "MIN": 25, "PTS": 20, "REB": 4, "AST": 2, "STL": 2, "BLK": 0, "TOV": 3, "3PM": 4},
    {"Name": "Goga Bitadze", "Team": "ORL", "MIN": 13, "PTS": 4, "REB": 2, "AST": 2, "STL": 0, "BLK": 0, "TOV": 0, "3PM": 0},
    {"Name": "Jonathan Isaac", "Team": "ORL", "MIN": 9, "PTS": 1, "REB": 3, "AST": 0, "STL": 0, "BLK": 0, "TOV": 0, "3PM": 0},
    {"Name": "Tyus Jones", "Team": "ORL", "MIN": 8, "PTS": 0, "REB": 0, "AST": 1, "STL": 0, "BLK": 0, "TOV": 0, "3PM": 0},
    {"Name": "Jase Richardson", "Team": "ORL", "MIN": 7, "PTS": 0, "REB": 1, "AST": 0, "STL": 0, "BLK": 1, "TOV": 0, "3PM": 0},
    {"Name": "Jett Howard", "Team": "ORL", "MIN": 6, "PTS": 2, "REB": 0, "AST": 0, "STL": 0, "BLK": 0, "TOV": 2, "3PM": 1},
    {"Name": "Noah Penda", "Team": "ORL", "MIN": 2, "PTS": 4, "REB": 2, "AST": 0, "STL": 0, "BLK": 0, "TOV": 0, "3PM": 0},
]

atl_players = [
    {"Name": "Dyson Daniels", "Team": "ATL", "MIN": 35, "PTS": 18, "REB": 6, "AST": 6, "STL": 2, "BLK": 0, "TOV": 3, "3PM": 1},
    {"Name": "Jalen Johnson", "Team": "ATL", "MIN": 33, "PTS": 17, "REB": 6, "AST": 5, "STL": 1, "BLK": 0, "TOV": 3, "3PM": 1},
    {"Name": "Nickeil Alexander-Walker", "Team": "ATL", "MIN": 32, "PTS": 20, "REB": 1, "AST": 2, "STL": 0, "BLK": 1, "TOV": 7, "3PM": 3},
    {"Name": "Kristaps Porzingis", "Team": "ATL", "MIN": 26, "PTS": 15, "REB": 2, "AST": 3, "STL": 3, "BLK": 0, "TOV": 0, "3PM": 0},
    {"Name": "Zaccharie Risacher", "Team": "ATL", "MIN": 26, "PTS": 21, "REB": 4, "AST": 1, "STL": 1, "BLK": 0, "TOV": 0, "3PM": 3},
    {"Name": "Onyeka Okongwu", "Team": "ATL", "MIN": 27, "PTS": 14, "REB": 7, "AST": 3, "STL": 1, "BLK": 1, "TOV": 1, "3PM": 2},
    {"Name": "Luke Kennard", "Team": "ATL", "MIN": 20, "PTS": 6, "REB": 1, "AST": 0, "STL": 3, "BLK": 0, "TOV": 3, "3PM": 0},
    {"Name": "Mouhamed Gueye", "Team": "ATL", "MIN": 18, "PTS": 5, "REB": 3, "AST": 0, "STL": 0, "BLK": 3, "TOV": 0, "3PM": 1},
    {"Name": "Keaton Wallace", "Team": "ATL", "MIN": 11, "PTS": 4, "REB": 3, "AST": 2, "STL": 0, "BLK": 1, "TOV": 1, "3PM": 1},
    {"Name": "Vit Krejci", "Team": "ATL", "MIN": 11, "PTS": 6, "REB": 1, "AST": 0, "STL": 0, "BLK": 0, "TOV": 1, "3PM": 1},
    {"Name": "Asa Newell", "Team": "ATL", "MIN": 1, "PTS": 0, "REB": 0, "AST": 0, "STL": 1, "BLK": 0, "TOV": 0, "3PM": 0},
    {"Name": "Caleb Houstan", "Team": "ATL", "MIN": 1, "PTS": 0, "REB": 0, "AST": 0, "STL": 0, "BLK": 0, "TOV": 0, "3PM": 0},
    {"Name": "Trae Young", "Team": "ATL", "MIN": 0, "PTS": 0, "REB": 0, "AST": 0, "STL": 0, "BLK": 0, "TOV": 0, "3PM": 0},
]

# Game 2: TOR 128, MIL 100
mil_players = [
    {"Name": "Giannis Antetokounmpo", "Team": "MIL", "MIN": 24, "PTS": 22, "REB": 8, "AST": 3, "STL": 0, "BLK": 1, "TOV": 1, "3PM": 0},
    {"Name": "Kyle Kuzma", "Team": "MIL", "MIN": 27, "PTS": 18, "REB": 3, "AST": 0, "STL": 0, "BLK": 1, "TOV": 4, "3PM": 2},
    {"Name": "Ryan Rollins", "Team": "MIL", "MIN": 28, "PTS": 5, "REB": 5, "AST": 0, "STL": 0, "BLK": 2, "TOV": 3, "3PM": 1},
    {"Name": "Myles Turner", "Team": "MIL", "MIN": 23, "PTS": 10, "REB": 8, "AST": 3, "STL": 2, "BLK": 1, "TOV": 1, "3PM": 2},
    {"Name": "Cole Anthony", "Team": "MIL", "MIN": 24, "PTS": 12, "REB": 4, "AST": 4, "STL": 1, "BLK": 1, "TOV": 0, "3PM": 0},
    {"Name": "Gary Trent Jr.", "Team": "MIL", "MIN": 21, "PTS": 6, "REB": 4, "AST": 0, "STL": 0, "BLK": 1, "TOV": 2, "3PM": 2},
    {"Name": "AJ Green", "Team": "MIL", "MIN": 17, "PTS": 9, "REB": 1, "AST": 0, "STL": 0, "BLK": 0, "TOV": 4, "3PM": 3},
    {"Name": "Bobby Portis", "Team": "MIL", "MIN": 20, "PTS": 0, "REB": 2, "AST": 0, "STL": 0, "BLK": 1, "TOV": 1, "3PM": 0},
    {"Name": "Taurean Prince", "Team": "MIL", "MIN": 21, "PTS": 0, "REB": 1, "AST": 0, "STL": 0, "BLK": 1, "TOV": 4, "3PM": 0},
    {"Name": "Mark Sears", "Team": "MIL", "MIN": 5, "PTS": 8, "REB": 0, "AST": 0, "STL": 0, "BLK": 0, "TOV": 0, "3PM": 1},
    {"Name": "Gary Harris", "Team": "MIL", "MIN": 8, "PTS": 1, "REB": 0, "AST": 1, "STL": 0, "BLK": 0, "TOV": 1, "3PM": 0},
    {"Name": "Andre Jackson Jr.", "Team": "MIL", "MIN": 6, "PTS": 0, "REB": 1, "AST": 0, "STL": 0, "BLK": 0, "TOV": 0, "3PM": 0},
    {"Name": "Amir Coffey", "Team": "MIL", "MIN": 10, "PTS": 0, "REB": 1, "AST": 0, "STL": 0, "BLK": 0, "TOV": 1, "3PM": 0},
    {"Name": "Jericho Sims", "Team": "MIL", "MIN": 7, "PTS": 0, "REB": 0, "AST": 1, "STL": 1, "BLK": 0, "TOV": 0, "3PM": 0},
]

tor_players = [
    {"Name": "Scottie Barnes", "Team": "TOR", "MIN": 28, "PTS": 23, "REB": 3, "AST": 6, "STL": 4, "BLK": 2, "TOV": 1, "3PM": 3},
    {"Name": "RJ Barrett", "Team": "TOR", "MIN": 31, "PTS": 23, "REB": 8, "AST": 4, "STL": 1, "BLK": 2, "TOV": 1, "3PM": 4},
    {"Name": "Sandro Mamukelashvili", "Team": "TOR", "MIN": 23, "PTS": 15, "REB": 7, "AST": 1, "STL": 0, "BLK": 0, "TOV": 1, "3PM": 3},
    {"Name": "Gradey Dick", "Team": "TOR", "MIN": 22, "PTS": 14, "REB": 1, "AST": 0, "STL": 0, "BLK": 0, "TOV": 0, "3PM": 3},
    {"Name": "Immanuel Quickley", "Team": "TOR", "MIN": 26, "PTS": 15, "REB": 6, "AST": 7, "STL": 1, "BLK": 0, "TOV": 1, "3PM": 2},
    {"Name": "Brandon Ingram", "Team": "TOR", "MIN": 31, "PTS": 13, "REB": 8, "AST": 4, "STL": 0, "BLK": 2, "TOV": 2, "3PM": 0},
    {"Name": "Jakob Poeltl", "Team": "TOR", "MIN": 20, "PTS": 8, "REB": 9, "AST": 0, "STL": 0, "BLK": 2, "TOV": 0, "3PM": 0},
    {"Name": "Jamal Shead", "Team": "TOR", "MIN": 17, "PTS": 5, "REB": 2, "AST": 8, "STL": 2, "BLK": 0, "TOV": 1, "3PM": 1},
    {"Name": "Collin Murray-Boyles", "Team": "TOR", "MIN": 15, "PTS": 1, "REB": 2, "AST": 0, "STL": 0, "BLK": 0, "TOV": 1, "3PM": 0},
    {"Name": "Ochai Agbaji", "Team": "TOR", "MIN": 0, "PTS": 0, "REB": 0, "AST": 0, "STL": 0, "BLK": 0, "TOV": 0, "3PM": 0},
    {"Name": "Ja'Kobe Walter", "Team": "TOR", "MIN": 0, "PTS": 0, "REB": 0, "AST": 0, "STL": 0, "BLK": 0, "TOV": 0, "3PM": 0},
]

# Game 3: NO 116, CHA 112
cha_players = [
    {"Name": "Miles Bridges", "Team": "CHA", "MIN": 37, "PTS": 22, "REB": 8, "AST": 3, "STL": 1, "BLK": 0, "TOV": 1, "3PM": 3},
    {"Name": "Kon Knueppel", "Team": "CHA", "MIN": 32, "PTS": 20, "REB": 12, "AST": 2, "STL": 0, "BLK": 0, "TOV": 2, "3PM": 3},
    {"Name": "Ryan Kalkbrenner", "Team": "CHA", "MIN": 30, "PTS": 10, "REB": 11, "AST": 2, "STL": 1, "BLK": 4, "TOV": 2, "3PM": 0},
    {"Name": "Collin Sexton", "Team": "CHA", "MIN": 30, "PTS": 17, "REB": 0, "AST": 5, "STL": 1, "BLK": 0, "TOV": 5, "3PM": 2},
    {"Name": "Sion James", "Team": "CHA", "MIN": 31, "PTS": 6, "REB": 5, "AST": 2, "STL": 1, "BLK": 1, "TOV": 2, "3PM": 1},
    {"Name": "Tre Mann", "Team": "CHA", "MIN": 24, "PTS": 18, "REB": 2, "AST": 2, "STL": 0, "BLK": 4, "TOV": 2, "3PM": 2},
    {"Name": "Moussa Diabate", "Team": "CHA", "MIN": 18, "PTS": 6, "REB": 5, "AST": 2, "STL": 2, "BLK": 1, "TOV": 1, "3PM": 0},
    {"Name": "KJ Simpson", "Team": "CHA", "MIN": 13, "PTS": 8, "REB": 3, "AST": 0, "STL": 0, "BLK": 0, "TOV": 1, "3PM": 1},
    {"Name": "Liam McNeeley", "Team": "CHA", "MIN": 13, "PTS": 2, "REB": 2, "AST": 1, "STL": 0, "BLK": 0, "TOV": 1, "3PM": 0},
    {"Name": "Pat Connaughton", "Team": "CHA", "MIN": 11, "PTS": 3, "REB": 1, "AST": 0, "STL": 0, "BLK": 0, "TOV": 0, "3PM": 1},
]

no_players = [
    {"Name": "Trey Murphy III", "Team": "NO", "MIN": 37, "PTS": 21, "REB": 3, "AST": 3, "STL": 1, "BLK": 1, "TOV": 1, "3PM": 5},
    {"Name": "Herbert Jones", "Team": "NO", "MIN": 36, "PTS": 11, "REB": 5, "AST": 1, "STL": 1, "BLK": 0, "TOV": 2, "3PM": 3},
    {"Name": "Jose Alvarado", "Team": "NO", "MIN": 31, "PTS": 18, "REB": 3, "AST": 6, "STL": 2, "BLK": 0, "TOV": 1, "3PM": 4},
    {"Name": "Saddiq Bey", "Team": "NO", "MIN": 27, "PTS": 17, "REB": 4, "AST": 0, "STL": 0, "BLK": 0, "TOV": 0, "3PM": 3},
    {"Name": "Derik Queen", "Team": "NO", "MIN": 18, "PTS": 12, "REB": 8, "AST": 7, "STL": 4, "BLK": 0, "TOV": 3, "3PM": 0},
    {"Name": "Karlo Matkovic", "Team": "NO", "MIN": 15, "PTS": 13, "REB": 6, "AST": 2, "STL": 0, "BLK": 1, "TOV": 1, "3PM": 0},
    {"Name": "Jordan Poole", "Team": "NO", "MIN": 27, "PTS": 11, "REB": 2, "AST": 6, "STL": 0, "BLK": 0, "TOV": 3, "3PM": 1},
    {"Name": "Jeremiah Fears", "Team": "NO", "MIN": 17, "PTS": 11, "REB": 3, "AST": 0, "STL": 1, "BLK": 0, "TOV": 0, "3PM": 1},
    {"Name": "Kevon Looney", "Team": "NO", "MIN": 15, "PTS": 0, "REB": 8, "AST": 0, "STL": 1, "BLK": 1, "TOV": 2, "3PM": 0},
    {"Name": "Jordan Hawkins", "Team": "NO", "MIN": 17, "PTS": 2, "REB": 1, "AST": 0, "STL": 0, "BLK": 0, "TOV": 1, "3PM": 0},
    {"Name": "DeAndre Jordan", "Team": "NO", "MIN": 0, "PTS": 0, "REB": 0, "AST": 0, "STL": 0, "BLK": 0, "TOV": 0, "3PM": 0},
]

# Game 4: CHI 113, PHI 111
phi_players = [
    {"Name": "Tyrese Maxey", "Team": "PHI", "MIN": 39, "PTS": 39, "REB": 5, "AST": 5, "STL": 1, "BLK": 3, "TOV": 4, "3PM": 6},
    {"Name": "VJ Edgecombe", "Team": "PHI", "MIN": 38, "PTS": 12, "REB": 11, "AST": 4, "STL": 2, "BLK": 0, "TOV": 2, "3PM": 2},
    {"Name": "Kelly Oubre Jr.", "Team": "PHI", "MIN": 33, "PTS": 18, "REB": 3, "AST": 2, "STL": 2, "BLK": 2, "TOV": 1, "3PM": 2},
    {"Name": "Joel Embiid", "Team": "PHI", "MIN": 26, "PTS": 20, "REB": 6, "AST": 2, "STL": 3, "BLK": 4, "TOV": 3, "3PM": 1},
    {"Name": "Jabari Walker", "Team": "PHI", "MIN": 19, "PTS": 2, "REB": 3, "AST": 1, "STL": 0, "BLK": 0, "TOV": 0, "3PM": 0},
    {"Name": "Quentin Grimes", "Team": "PHI", "MIN": 30, "PTS": 10, "REB": 3, "AST": 4, "STL": 4, "BLK": 1, "TOV": 4, "3PM": 0},
    {"Name": "Adem Bona", "Team": "PHI", "MIN": 22, "PTS": 2, "REB": 4, "AST": 0, "STL": 0, "BLK": 1, "TOV": 2, "3PM": 0},
    {"Name": "Trendon Watford", "Team": "PHI", "MIN": 19, "PTS": 8, "REB": 4, "AST": 2, "STL": 0, "BLK": 0, "TOV": 2, "3PM": 0},
    {"Name": "Jared McCain", "Team": "PHI", "MIN": 15, "PTS": 0, "REB": 2, "AST": 0, "STL": 0, "BLK": 0, "TOV": 2, "3PM": 0},
]

chi_players = [
    {"Name": "Josh Giddey", "Team": "CHI", "MIN": 37, "PTS": 29, "REB": 15, "AST": 12, "STL": 1, "BLK": 0, "TOV": 5, "3PM": 2},
    {"Name": "Isaac Okoro", "Team": "CHI", "MIN": 33, "PTS": 16, "REB": 2, "AST": 5, "STL": 2, "BLK": 1, "TOV": 1, "3PM": 3},
    {"Name": "Nikola Vucevic", "Team": "CHI", "MIN": 31, "PTS": 19, "REB": 10, "AST": 0, "STL": 1, "BLK": 1, "TOV": 1, "3PM": 1},
    {"Name": "Matas Buzelis", "Team": "CHI", "MIN": 30, "PTS": 10, "REB": 10, "AST": 1, "STL": 2, "BLK": 2, "TOV": 3, "3PM": 2},
    {"Name": "Tre Jones", "Team": "CHI", "MIN": 27, "PTS": 8, "REB": 6, "AST": 3, "STL": 0, "BLK": 1, "TOV": 0, "3PM": 0},
    {"Name": "Kevin Huerter", "Team": "CHI", "MIN": 25, "PTS": 12, "REB": 5, "AST": 2, "STL": 1, "BLK": 2, "TOV": 4, "3PM": 2},
    {"Name": "Patrick Williams", "Team": "CHI", "MIN": 21, "PTS": 3, "REB": 2, "AST": 2, "STL": 1, "BLK": 1, "TOV": 0, "3PM": 0},
    {"Name": "Jalen Smith", "Team": "CHI", "MIN": 17, "PTS": 14, "REB": 4, "AST": 0, "STL": 0, "BLK": 0, "TOV": 4, "3PM": 3},
    {"Name": "Dalen Terry", "Team": "CHI", "MIN": 14, "PTS": 2, "REB": 2, "AST": 2, "STL": 0, "BLK": 0, "TOV": 4, "3PM": 0},
    {"Name": "Julian Phillips", "Team": "CHI", "MIN": 5, "PTS": 0, "REB": 1, "AST": 0, "STL": 1, "BLK": 0, "TOV": 0, "3PM": 0},
]

# ============================================================
# CALCULATE DK POINTS FOR ALL PLAYERS
# ============================================================
all_players = (orl_players + atl_players + mil_players + tor_players +
               cha_players + no_players + phi_players + chi_players)

for p in all_players:
    p["DK_Fantasy_Points"] = calc_dk_points(
        p["PTS"], p["REB"], p["AST"], p["STL"], p["BLK"], p["TOV"], p["3PM"]
    )

# Build lookup by player name
player_dk_lookup = {}
for p in all_players:
    player_dk_lookup[p["Name"]] = p["DK_Fantasy_Points"]

# ============================================================
# SAVE BOX SCORE CSVs
# ============================================================
output_dir = "/Users/sineshawmesfintesfaye/mlb-draftkings-system"

def save_game_csv(filename, home_team, away_team, home_players, away_players, home_score, away_score):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Team", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "3PM", "DK_Fantasy_Points"])
        for p in away_players + home_players:
            writer.writerow([p["Name"], p["Team"], p["MIN"], p["PTS"], p["REB"], p["AST"],
                           p["STL"], p["BLK"], p["TOV"], p["3PM"], p["DK_Fantasy_Points"]])
    print(f"  Saved: {filepath}")
    print(f"  {away_team} {away_score} @ {home_team} {home_score}")

print("=" * 70)
print("SAVING BOX SCORE CSVs — Nov 4, 2025")
print("=" * 70)
save_game_csv("NOV4_boxscore_ORL_at_ATL.csv", "ATL", "ORL", atl_players, orl_players, 127, 112)
save_game_csv("NOV4_boxscore_MIL_at_TOR.csv", "TOR", "MIL", tor_players, mil_players, 128, 100)
save_game_csv("NOV4_boxscore_CHA_at_NO.csv", "NO", "CHA", no_players, cha_players, 116, 112)
save_game_csv("NOV4_boxscore_PHI_at_CHI.csv", "CHI", "PHI", chi_players, phi_players, 113, 111)

# ============================================================
# PRINT INDIVIDUAL PLAYER DK SCORES (SORTED)
# ============================================================
print("\n" + "=" * 70)
print("ALL PLAYER DK FANTASY POINTS — Nov 4, 2025 (Sorted by DK Points)")
print("=" * 70)

# Check for DD/TD
for p in all_players:
    doubles = sum(1 for x in [p["PTS"], p["REB"], p["AST"]] if x >= 10)
    bonus = ""
    if doubles >= 3:
        bonus = " [TRIPLE-DOUBLE +3]"
    elif doubles >= 2:
        bonus = " [DOUBLE-DOUBLE +1.5]"
    p["bonus_tag"] = bonus

sorted_players = sorted(all_players, key=lambda x: x["DK_Fantasy_Points"], reverse=True)
for i, p in enumerate(sorted_players, 1):
    if p["DK_Fantasy_Points"] > 0:
        print(f"  {i:3d}. {p['Name']:30s} ({p['Team']:3s})  {p['DK_Fantasy_Points']:6.1f} DK pts  "
              f"[{p['PTS']}p/{p['REB']}r/{p['AST']}a/{p['STL']}s/{p['BLK']}b/{p['TOV']}to/{p['3PM']}tpm]{p['bonus_tag']}")

# ============================================================
# SCORE LINEUPS FROM DK ENTRIES CSV
# ============================================================
print("\n" + "=" * 70)
print("SCORING LINEUPS FROM dk_entries_1771316097246.csv")
print("=" * 70)

entries_file = "/Users/sineshawmesfintesfaye/Downloads/dk_entries_1771316097246.csv"
lineups = []
missing_players = set()

with open(entries_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        positions = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
        players_in_lineup = []
        total_dk = 0
        for pos in positions:
            name = row[pos].strip().strip('"')
            dk_pts = player_dk_lookup.get(name, None)
            if dk_pts is None:
                missing_players.add(name)
                dk_pts = 0  # Unknown player
            players_in_lineup.append((pos, name, dk_pts))
            total_dk += dk_pts
        lineups.append({
            "entry_id": row["Entry ID"].strip('"'),
            "players": players_in_lineup,
            "total": total_dk
        })

# Sort by total DK points
lineups.sort(key=lambda x: x["total"], reverse=True)

if missing_players:
    print(f"\n  WARNING: Players not found in box scores: {missing_players}")

# Print top 20 and bottom 5
print(f"\n  Total lineups: {len(lineups)}")
print(f"  Best lineup:   {lineups[0]['total']:.1f} DK pts")
print(f"  Worst lineup:  {lineups[-1]['total']:.1f} DK pts")
print(f"  Average:       {sum(l['total'] for l in lineups)/len(lineups):.1f} DK pts")
print(f"  Median:        {lineups[len(lineups)//2]['total']:.1f} DK pts")

print(f"\n{'─' * 70}")
print(f"  TOP 20 LINEUPS")
print(f"{'─' * 70}")
for i, lineup in enumerate(lineups[:20], 1):
    print(f"\n  #{i:2d}  Entry {lineup['entry_id']}  —  TOTAL: {lineup['total']:.1f} DK pts")
    for pos, name, dk_pts in lineup["players"]:
        print(f"       {pos:4s}: {name:30s}  {dk_pts:6.1f}")

print(f"\n{'─' * 70}")
print(f"  BOTTOM 5 LINEUPS")
print(f"{'─' * 70}")
for i, lineup in enumerate(lineups[-5:], len(lineups) - 4):
    print(f"\n  #{i:2d}  Entry {lineup['entry_id']}  —  TOTAL: {lineup['total']:.1f} DK pts")
    for pos, name, dk_pts in lineup["players"]:
        print(f"       {pos:4s}: {name:30s}  {dk_pts:6.1f}")

# ============================================================
# DISTRIBUTION SUMMARY
# ============================================================
print(f"\n{'=' * 70}")
print(f"LINEUP SCORE DISTRIBUTION")
print(f"{'=' * 70}")

brackets = [(300, 999), (280, 300), (260, 280), (240, 260), (220, 240), (200, 220), (0, 200)]
for low, high in brackets:
    count = sum(1 for l in lineups if low <= l["total"] < high)
    if count > 0:
        pct = count / len(lineups) * 100
        bar = "#" * int(pct / 2)
        print(f"  {low:3d}-{high:3d}: {count:3d} lineups ({pct:5.1f}%) {bar}")

# ============================================================
# PLAYER EXPOSURE ANALYSIS
# ============================================================
print(f"\n{'=' * 70}")
print(f"PLAYER EXPOSURE & ACTUAL DK POINTS")
print(f"{'=' * 70}")

player_counts = {}
for lineup in lineups:
    for pos, name, dk_pts in lineup["players"]:
        if name not in player_counts:
            player_counts[name] = {"count": 0, "dk_pts": dk_pts}
        player_counts[name]["count"] += 1

sorted_exposure = sorted(player_counts.items(), key=lambda x: x[1]["count"], reverse=True)
print(f"\n  {'Player':30s}  {'Exposure':>10s}  {'DK Pts':>8s}  {'Value':>8s}")
print(f"  {'─' * 30}  {'─' * 10}  {'─' * 8}  {'─' * 8}")
for name, data in sorted_exposure:
    exposure_pct = data["count"] / len(lineups) * 100
    value = "GOOD" if data["dk_pts"] >= 25 else ("OK" if data["dk_pts"] >= 15 else "BAD")
    print(f"  {name:30s}  {data['count']:3d} ({exposure_pct:5.1f}%)  {data['dk_pts']:6.1f}    {value}")

# Save scored lineups to CSV
scored_file = os.path.join(output_dir, "NOV4_scored_lineups.csv")
with open(scored_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Rank", "Entry_ID", "Total_DK_Points", "PG", "PG_DK", "SG", "SG_DK",
                     "SF", "SF_DK", "PF", "PF_DK", "C", "C_DK", "G", "G_DK", "F", "F_DK", "UTIL", "UTIL_DK"])
    for i, lineup in enumerate(lineups, 1):
        row = [i, lineup["entry_id"], lineup["total"]]
        for pos, name, dk_pts in lineup["players"]:
            row.extend([name, dk_pts])
        writer.writerow(row)
print(f"\n  Saved scored lineups: {scored_file}")
