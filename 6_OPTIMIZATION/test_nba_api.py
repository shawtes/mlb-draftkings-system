"""
Quick test of NBA SportsData.io API connection
"""

from nba_sportsdata_fetcher import NBADataFetcher
from datetime import datetime

# Your API key
API_KEY = "d62d0ae315504e53a232ff7d1c3bea33"

print("🏀 Testing NBA SportsData.io API Connection")
print("=" * 60)

# Initialize fetcher
fetcher = NBADataFetcher(API_KEY)

# Test 1: Get current season
print("\n1. Testing API connection...")
try:
    endpoint = "/fantasy/json/CurrentSeason"
    data = fetcher._make_request(endpoint)
    if data:
        print(f"✅ API Connected! Current Season: {data}")
    else:
        print("❌ API connection failed")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Get active players
print("\n2. Fetching active NBA players...")
try:
    players = fetcher.get_active_players()
    if not players.empty:
        print(f"✅ Retrieved {len(players)} active players")
        print(f"   Sample players: {players['Name'].head(5).tolist()}")
    else:
        print("❌ No players retrieved")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Get today's projections
print("\n3. Fetching today's projections...")
try:
    today = datetime.now().strftime('%Y-%b-%d').upper()
    print(f"   Date: {today}")
    projections = fetcher.get_daily_projections(today)
    
    if not projections.empty:
        print(f"✅ Retrieved {len(projections)} player projections")
        print(f"\n   Top 5 projected players:")
        top5 = projections.nlargest(5, 'Predicted_DK_Points')[
            ['Name', 'Position', 'Team', 'Predicted_DK_Points', 'Salary']
        ]
        for _, p in top5.iterrows():
            print(f"      {p['Name']:25s} | {p['Position']:3s} | {p['Team']:3s} | "
                  f"{p['Predicted_DK_Points']:5.1f} pts | ${p['Salary']:,}")
    else:
        print(f"⚠️  No games today ({today})")
        print("   Try a game day date like '2025-JAN-15'")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 4: Get game odds
print("\n4. Fetching Vegas odds...")
try:
    today = datetime.now().strftime('%Y-%b-%d').upper()
    odds = fetcher.get_games_with_odds(today)
    
    if not odds.empty:
        print(f"✅ Retrieved odds for {len(odds)} games")
        print(f"\n   Games with highest totals:")
        for _, game in odds.nlargest(3, 'GameTotal').iterrows():
            print(f"      {game['AwayTeam']} @ {game['HomeTeam']} | O/U: {game['GameTotal']}")
    else:
        print(f"⚠️  No games with odds today")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("✅ API Test Complete!")
print("\nNext steps:")
print("  1. Run: python nba_optimizer_example.py")
print("  2. Or use: pipeline = NBAResearchPipeline(API_KEY)")
print("=" * 60)

