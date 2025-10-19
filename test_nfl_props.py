#!/usr/bin/env python3
"""
Test NFL Prop Generation - Command Line Version
Tests the prop generation logic without GUI
"""

import pandas as pd
import os

def test_nfl_prop_generation():
    """Test NFL prop generation from sportsdata.io data"""
    
    # Load NFL data
    data_file = '6_OPTIMIZATION/nfl_week7_CASH_SPORTSDATA.csv'
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    print(f"📊 Loading NFL data from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df)} players")
    print(f"📋 Teams: {sorted(df['Team'].unique())}")
    print(f"📊 Positions: {df['Position'].value_counts().to_dict()}")
    
    # Check available columns
    print(f"\n📋 Available columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col}")
    
    # Generate props
    prop_bets = []
    
    for _, player in df.iterrows():
        name = player['Name']
        team = player['Team']
        position = player['Position']
        opponent = player.get('Opponent', 'N/A')
        
        # Skip if missing essential data
        if pd.isna(name) or pd.isna(team) or pd.isna(position):
            continue
        
        print(f"\nProcessing {name} ({position}, {team})...")
        
        # QB Props
        if position == 'QB':
            if 'PassingYards' in player and not pd.isna(player['PassingYards']):
                yards = player['PassingYards']
                if yards > 0:
                    prop_bets.append({
                        'player': name, 'team': team, 'opponent': opponent,
                        'prop': 'Passing Yards', 'line': round(yards),
                        'projection': yards, 'position': position
                    })
                    print(f"  ✅ Added Passing Yards prop: {yards}")
            
            if 'PassingTouchdowns' in player and not pd.isna(player['PassingTouchdowns']):
                tds = player['PassingTouchdowns']
                if tds > 0:
                    prop_bets.append({
                        'player': name, 'team': team, 'opponent': opponent,
                        'prop': 'Passing TDs', 'line': round(tds, 1),
                        'projection': tds, 'position': position
                    })
                    print(f"  ✅ Added Passing TDs prop: {tds}")
            
            if 'RushingYards' in player and not pd.isna(player['RushingYards']):
                rush_yards = player['RushingYards']
                if rush_yards > 5:
                    prop_bets.append({
                        'player': name, 'team': team, 'opponent': opponent,
                        'prop': 'Rushing Yards', 'line': round(rush_yards),
                        'projection': rush_yards, 'position': position
                    })
                    print(f"  ✅ Added Rushing Yards prop: {rush_yards}")
        
        # RB Props
        elif position == 'RB':
            if 'RushingYards' in player and not pd.isna(player['RushingYards']):
                yards = player['RushingYards']
                if yards > 5:
                    prop_bets.append({
                        'player': name, 'team': team, 'opponent': opponent,
                        'prop': 'Rushing Yards', 'line': round(yards),
                        'projection': yards, 'position': position
                    })
                    print(f"  ✅ Added Rushing Yards prop: {yards}")
            
            if 'RushingTouchdowns' in player and not pd.isna(player['RushingTouchdowns']):
                tds = player['RushingTouchdowns']
                if tds > 0:
                    prop_bets.append({
                        'player': name, 'team': team, 'opponent': opponent,
                        'prop': 'Rushing TDs', 'line': round(tds, 1),
                        'projection': tds, 'position': position
                    })
                    print(f"  ✅ Added Rushing TDs prop: {tds}")
            
            if 'ReceivingYards' in player and not pd.isna(player['ReceivingYards']):
                rec_yards = player['ReceivingYards']
                if rec_yards > 0:
                    prop_bets.append({
                        'player': name, 'team': team, 'opponent': opponent,
                        'prop': 'Receiving Yards', 'line': round(rec_yards),
                        'projection': rec_yards, 'position': position
                    })
                    print(f"  ✅ Added Receiving Yards prop: {rec_yards}")
        
        # WR Props
        elif position == 'WR':
            if 'ReceivingYards' in player and not pd.isna(player['ReceivingYards']):
                yards = player['ReceivingYards']
                if yards > 5:
                    prop_bets.append({
                        'player': name, 'team': team, 'opponent': opponent,
                        'prop': 'Receiving Yards', 'line': round(yards),
                        'projection': yards, 'position': position
                    })
                    print(f"  ✅ Added Receiving Yards prop: {yards}")
            
            if 'ReceivingTouchdowns' in player and not pd.isna(player['ReceivingTouchdowns']):
                tds = player['ReceivingTouchdowns']
                if tds > 0:
                    prop_bets.append({
                        'player': name, 'team': team, 'opponent': opponent,
                        'prop': 'Receiving TDs', 'line': round(tds, 1),
                        'projection': tds, 'position': position
                    })
                    print(f"  ✅ Added Receiving TDs prop: {tds}")
            
            if 'Receptions' in player and not pd.isna(player['Receptions']):
                rec = player['Receptions']
                if rec > 0:
                    prop_bets.append({
                        'player': name, 'team': team, 'opponent': opponent,
                        'prop': 'Receptions', 'line': round(rec, 1),
                        'projection': rec, 'position': position
                    })
                    print(f"  ✅ Added Receptions prop: {rec}")
        
        # TE Props
        elif position == 'TE':
            if 'ReceivingYards' in player and not pd.isna(player['ReceivingYards']):
                yards = player['ReceivingYards']
                if yards > 0:
                    prop_bets.append({
                        'player': name, 'team': team, 'opponent': opponent,
                        'prop': 'Receiving Yards', 'line': round(yards),
                        'projection': yards, 'position': position
                    })
                    print(f"  ✅ Added Receiving Yards prop: {yards}")
            
            if 'Receptions' in player and not pd.isna(player['Receptions']):
                rec = player['Receptions']
                if rec > 0:
                    prop_bets.append({
                        'player': name, 'team': team, 'opponent': opponent,
                        'prop': 'Receptions', 'line': round(rec, 1),
                        'projection': rec, 'position': position
                    })
                    print(f"  ✅ Added Receptions prop: {rec}")
    
    print(f"\n🎯 SUMMARY:")
    print(f"Total Props Generated: {len(prop_bets)}")
    
    if prop_bets:
        # Group by prop type
        prop_types = {}
        for prop in prop_bets:
            prop_type = prop['prop']
            if prop_type not in prop_types:
                prop_types[prop_type] = 0
            prop_types[prop_type] += 1
        
        print(f"\nProp Types:")
        for prop_type, count in prop_types.items():
            print(f"  • {prop_type}: {count}")
        
        # Show top 10 props
        print(f"\nTop 10 Props by Projection:")
        sorted_props = sorted(prop_bets, key=lambda x: x['projection'], reverse=True)
        for i, prop in enumerate(sorted_props[:10], 1):
            print(f"  {i:2d}. {prop['player']} ({prop['team']}) - {prop['prop']} {prop['line']} "
                  f"(Proj: {prop['projection']:.1f})")
        
        # Save to CSV
        props_df = pd.DataFrame(prop_bets)
        output_file = 'nfl_props_test.csv'
        props_df.to_csv(output_file, index=False)
        print(f"\n💾 Props saved to: {output_file}")
    
    else:
        print("❌ No props generated! Check the data columns.")

if __name__ == "__main__":
    test_nfl_prop_generation()


