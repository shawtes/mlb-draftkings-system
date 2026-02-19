#!/usr/bin/env python3
"""
Create DK entries file from the DK player pool data
"""

import pandas as pd
import re

# The DK player pool data (from user's paste)
DK_PLAYER_POOL = """Nikola Jokic	40556309	C/UTIL	11500	DEN@MIN 10/27/2025 09:30PM ET
Victor Wembanyama	40556311	C/UTIL	11000	TOR@SAS 10/27/2025 08:00PM ET
Shai Gilgeous-Alexander	40556313	PG/G/UTIL	10500	OKC@DAL 10/27/2025 08:30PM ET
Anthony Davis	40556316	PF/C/F/UTIL	10000	OKC@DAL 10/27/2025 08:30PM ET
Cade Cunningham	40556320	PG/G/UTIL	9800	CLE@DET 10/27/2025 07:00PM ET
Zion Williamson	40556323	PF/F/UTIL	9600	BOS@NOP 10/27/2025 08:00PM ET
Alperen Sengun	40556326	PF/C/F/UTIL	9500	BKN@HOU 10/27/2025 08:00PM ET
Donovan Mitchell	40556330	PG/SG/G/UTIL	9400	CLE@DET 10/27/2025 07:00PM ET
Tyrese Maxey	40556334	PG/G/UTIL	9300	ORL@PHI 10/27/2025 07:00PM ET
Trae Young	40556337	PG/G/UTIL	9200	ATL@CHI 10/27/2025 08:00PM ET
Anthony Edwards	40556340	SG/G/UTIL	9100	DEN@MIN 10/27/2025 09:30PM ET
Kevin Durant	40556343	SG/SF/F/G/UTIL	9000	BKN@HOU 10/27/2025 08:00PM ET
Devin Booker	40556348	PG/G/UTIL	8900	PHX@UTA 10/27/2025 09:00PM ET
Josh Giddey	40556351	PG/SG/G/UTIL	8700	ATL@CHI 10/27/2025 08:00PM ET
Evan Mobley	40556355	PF/C/F/UTIL	8600	CLE@DET 10/27/2025 07:00PM ET
Jaylen Brown	40556359	SG/SF/F/G/UTIL	8500	BOS@NOP 10/27/2025 08:00PM ET
Jalen Johnson	40556364	PF/F/UTIL	8400	ATL@CHI 10/27/2025 08:00PM ET
Paolo Banchero	40556367	PF/F/UTIL	8300	ORL@PHI 10/27/2025 07:00PM ET
Scottie Barnes	40556370	PG/SF/F/G/UTIL	8200	TOR@SAS 10/27/2025 08:00PM ET
Nikola Vucevic	40556375	C/UTIL	8100	ATL@CHI 10/27/2025 08:00PM ET
Jalen Williams	40556380	SF/PF/F/UTIL	8000	OKC@DAL 10/27/2025 08:30PM ET
Derrick White	40556377	PG/G/UTIL	8000	BOS@NOP 10/27/2025 08:00PM ET
Darius Garland	40556384	PG/G/UTIL	7900	CLE@DET 10/27/2025 07:00PM ET
Chet Holmgren	40556387	PF/C/F/UTIL	7900	OKC@DAL 10/27/2025 08:30PM ET
Franz Wagner	40556391	SF/PF/F/UTIL	7800	ORL@PHI 10/27/2025 07:00PM ET
Joel Embiid	40556395	C/UTIL	7700	ORL@PHI 10/27/2025 07:00PM ET
Coby White	40556397	PG/SG/G/UTIL	7600	ATL@CHI 10/27/2025 08:00PM ET
Julius Randle	40556401	PF/F/UTIL	7500	DEN@MIN 10/27/2025 09:30PM ET
Jamal Murray	40556404	PG/G/UTIL	7500	DEN@MIN 10/27/2025 09:30PM ET
Mark Williams	40556410	C/UTIL	7500	PHX@UTA 10/27/2025 09:00PM ET
RJ Barrett	40556407	SG/G/UTIL	7500	TOR@SAS 10/27/2025 08:00PM ET
Amen Thompson	40556412	PG/G/UTIL	7400	BKN@HOU 10/27/2025 08:00PM ET
Paul George	40556415	SF/PF/F/UTIL	7400	ORL@PHI 10/27/2025 07:00PM ET
Cam Thomas	40556419	PG/SG/G/UTIL	7300	BKN@HOU 10/27/2025 08:00PM ET
Jalen Duren	40556438	C/UTIL	7200	CLE@DET 10/27/2025 07:00PM ET
Kristaps Porzingis	40556431	PF/C/F/UTIL	7200	ATL@CHI 10/27/2025 08:00PM ET
Brandon Ingram	40556423	SF/F/UTIL	7200	TOR@SAS 10/27/2025 08:00PM ET
De'Aaron Fox	40556435	PG/G/UTIL	7200	TOR@SAS 10/27/2025 08:00PM ET
Trey Murphy III	40556426	SG/SF/F/G/UTIL	7200	BOS@NOP 10/27/2025 08:00PM ET
Stephon Castle	40556440	PG/SG/G/UTIL	7100	TOR@SAS 10/27/2025 08:00PM ET
Aaron Gordon	40556453	PF/F/UTIL	7000	DEN@MIN 10/27/2025 09:30PM ET
Onyeka Okongwu	40556451	C/UTIL	7000	ATL@CHI 10/27/2025 08:00PM ET
Lauri Markkanen	40556447	SF/PF/F/UTIL	7000	PHX@UTA 10/27/2025 09:00PM ET
Immanuel Quickley	40556444	PG/G/UTIL	7000	TOR@SAS 10/27/2025 08:00PM ET
Desmond Bane	40556456	SG/SF/F/G/UTIL	7000	ORL@PHI 10/27/2025 07:00PM ET
Cooper Flagg	40556461	PG/G/UTIL	6900	OKC@DAL 10/27/2025 08:30PM ET
Michael Porter Jr.	40556468	PF/F/UTIL	6800	BKN@HOU 10/27/2025 08:00PM ET
VJ Edgecombe	40556471	SG/SF/F/G/UTIL	6800	ORL@PHI 10/27/2025 07:00PM ET
Jordan Poole	40556464	PG/SG/G/UTIL	6800	BOS@NOP 10/27/2025 08:00PM ET"""

def parse_dk_pool():
    """Parse the DK player pool text"""
    
    players = []
    
    for line in DK_PLAYER_POOL.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) >= 5:
            name = parts[0].strip()
            dk_id = parts[1].strip()
            roster_position = parts[2].strip()
            salary = parts[3].strip().replace(',', '')
            
            # Extract team from game info
            game_info = parts[4].strip()
            # Format: "TOR@SAS 10/27/2025 08:00PM ET"
            game_match = re.match(r'([A-Z]+)[@]([A-Z]+)', game_info)
            if game_match:
                team1, team2 = game_match.groups()
                # Determine which team the player is on (need to guess)
                # For now, just use first team
                team = team1
            else:
                team = "UNK"
            
            players.append({
                'Name': name,
                'DK_ID': dk_id,
                'Roster_Position': roster_position,
                'Salary': int(salary),
                'Team': team
            })
    
    return pd.DataFrame(players)


def main():
    """Create DK entries file"""
    
    print("\n📋 Creating DK Entries File")
    print("="*70)
    
    # Parse DK pool
    df = parse_dk_pool()
    
    print(f"\n✅ Parsed {len(df)} players from DK pool")
    print(f"📊 Salary range: ${df['Salary'].min()} - ${df['Salary'].max()}")
    
    # Save
    df.to_csv('dk_player_pool_oct27.csv', index=False)
    print(f"\n💾 Saved to: dk_player_pool_oct27.csv")
    
    print(f"\n📋 Sample:")
    print(df[['Name', 'DK_ID', 'Roster_Position', 'Salary']].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

