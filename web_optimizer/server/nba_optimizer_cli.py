#!/usr/bin/env python3
"""
NBA Optimizer CLI Wrapper
Calls the Markov chain optimizer from the command line for web integration
"""

import sys
import json
import os
import pandas as pd
import numpy as np

# Add the 6_OPTIMIZATION directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '6_OPTIMIZATION'))

# Import Markov probabilities module
try:
    from nba_markov_probabilities import apply_markov_adjustments
    MARKOV_AVAILABLE = True
    print("✅ Markov probabilities module loaded", file=sys.stderr)
except Exception as e:
    MARKOV_AVAILABLE = False
    print(f"⚠️ Markov module not available: {e}", file=sys.stderr)

def optimize_nba_lineups(players_data, num_lineups=10, min_salary=48000, max_salary=50000, 
                         unique_players=3, max_exposure=100, stack_settings=None):
    """
    Optimize NBA lineups using a simplified algorithm
    
    Args:
        players_data: List of player dictionaries with id, name, position, salary, projection
        num_lineups: Number of lineups to generate
        min_salary: Minimum salary requirement
        max_salary: Maximum salary (salary cap)
        unique_players: Minimum unique players between lineups
        max_exposure: Maximum exposure percentage per player
        stack_settings: Dictionary with stacking preferences
    
    Returns:
        List of optimized lineups
    """
    
    # Convert to DataFrame
    df = pd.DataFrame(players_data)
    
    # Validate required columns
    required_cols = ['id', 'name', 'position', 'salary', 'projection']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Filter out players with no projection
    df = df[df['projection'] > 0].copy()
    
    if len(df) < 8:
        raise ValueError("Not enough players with projections to create lineups")
    
    # Apply Markov Chain adjustments if available
    if MARKOV_AVAILABLE:
        try:
            cache_dir = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/nba_historical_cache"
            print(f"🔮 Applying Markov Chain probability adjustments...", file=sys.stderr)
            
            # Rename columns to match what Markov module expects
            df_for_markov = df.copy()
            df_for_markov['Name'] = df_for_markov['name']
            df_for_markov['Position'] = df_for_markov['position']
            df_for_markov['Salary'] = df_for_markov['salary']
            df_for_markov['Predicted_DK_Points'] = df_for_markov['projection']
            
            # Apply Markov adjustments
            df_adjusted = apply_markov_adjustments(
                df_players=df_for_markov,
                history_df=None,
                cache_dir=cache_dir,
                blend_alpha=0.25,  # 25% Markov, 75% original projection
                min_games=30,
                player_thresholds=(20.0, 25.0, 30.0),
            )
            
            # Check if Markov adjustments were added
            if 'Predicted_DK_Points_MarkovBlend' in df_adjusted.columns:
                # Use Markov-adjusted projections
                df['projection'] = df_adjusted['Predicted_DK_Points_MarkovBlend'].values
                print(f"✅ Markov adjustments applied successfully!", file=sys.stderr)
                print(f"   Markov columns added: {[c for c in df_adjusted.columns if 'MC_' in c]}", file=sys.stderr)
            elif 'MC_Expected' in df_adjusted.columns:
                # Fallback to MC_Expected if available
                df['projection'] = df_adjusted['MC_Expected'].values
                print(f"✅ Markov MC_Expected projections applied!", file=sys.stderr)
            else:
                print(f"ℹ️ No historical data found, using base projections", file=sys.stderr)
                
        except Exception as e:
            print(f"⚠️ Markov adjustment failed: {e}, using base projections", file=sys.stderr)
    
    # Position requirements for DraftKings NBA
    position_reqs = {
        'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1,
        'G': 1,   # Guard (PG or SG)
        'F': 1,   # Forward (SF or PF)
        'UTIL': 1  # Utility (any position)
    }
    
    # Group players by position
    players_by_pos = {
        'PG': df[df['position'].str.contains('PG', na=False)].to_dict('records'),
        'SG': df[df['position'].str.contains('SG', na=False)].to_dict('records'),
        'SF': df[df['position'].str.contains('SF', na=False)].to_dict('records'),
        'PF': df[df['position'].str.contains('PF', na=False)].to_dict('records'),
        'C': df[df['position'].str.contains('C', na=False)].to_dict('records'),
    }
    
    # Sort each position by projection
    for pos in players_by_pos:
        players_by_pos[pos] = sorted(players_by_pos[pos], key=lambda x: x['projection'], reverse=True)
    
    lineups = []
    exposure_tracker = {}
    used_lineup_keys = set()
    
    strategies = ['high_projection', 'balanced', 'value', 'contrarian']
    
    for i in range(num_lineups):
        strategy = strategies[i % len(strategies)]
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts:
            lineup = generate_lineup(
                players_by_pos, 
                strategy, 
                exposure_tracker, 
                max_exposure,
                num_lineups,
                min_salary,
                max_salary
            )
            
            if lineup:
                lineup_key = '|'.join(sorted([p['id'] for p in lineup['players']]))
                
                # Check uniqueness
                if lineup_key not in used_lineup_keys:
                    # Check if lineup is unique enough
                    is_unique = True
                    for existing_key in used_lineup_keys:
                        existing_ids = set(existing_key.split('|'))
                        new_ids = set(lineup_key.split('|'))
                        common = len(existing_ids & new_ids)
                        if common > (8 - unique_players):
                            is_unique = False
                            break
                    
                    if is_unique:
                        used_lineup_keys.add(lineup_key)
                        
                        # Update exposure
                        for player in lineup['players']:
                            exposure_tracker[player['id']] = exposure_tracker.get(player['id'], 0) + 1
                        
                        lineups.append(lineup)
                        break
            
            attempts += 1
    
    return lineups


def generate_lineup(players_by_pos, strategy, exposure_tracker, max_exposure, 
                    total_lineups, min_salary, max_salary):
    """Generate a single lineup using the specified strategy"""
    
    lineup_players = []
    used_ids = set()
    total_salary = 0
    total_projection = 0
    
    # Fill core positions first (PG, SG, SF, PF, C)
    core_positions = ['PG', 'SG', 'SF', 'PF', 'C']
    
    for pos in core_positions:
        player = select_player(
            players_by_pos[pos],
            strategy,
            used_ids,
            exposure_tracker,
            max_exposure,
            total_lineups
        )
        
        if not player:
            return None
        
        player_copy = player.copy()
        player_copy['rosterPosition'] = pos
        lineup_players.append(player_copy)
        used_ids.add(player['id'])
        total_salary += player['salary']
        total_projection += player['projection']
    
    # Fill G position (any guard not used)
    guard_pool = [p for p in players_by_pos['PG'] + players_by_pos['SG'] 
                  if p['id'] not in used_ids]
    guard_player = select_player(
        guard_pool, strategy, used_ids, exposure_tracker, max_exposure, total_lineups
    )
    
    if not guard_player:
        return None
    
    guard_copy = guard_player.copy()
    guard_copy['rosterPosition'] = 'G'
    lineup_players.append(guard_copy)
    used_ids.add(guard_player['id'])
    total_salary += guard_player['salary']
    total_projection += guard_player['projection']
    
    # Fill F position (any forward not used)
    forward_pool = [p for p in players_by_pos['SF'] + players_by_pos['PF'] 
                    if p['id'] not in used_ids]
    forward_player = select_player(
        forward_pool, strategy, used_ids, exposure_tracker, max_exposure, total_lineups
    )
    
    if not forward_player:
        return None
    
    forward_copy = forward_player.copy()
    forward_copy['rosterPosition'] = 'F'
    lineup_players.append(forward_copy)
    used_ids.add(forward_player['id'])
    total_salary += forward_player['salary']
    total_projection += forward_player['projection']
    
    # Fill UTIL position (any player not used)
    all_remaining = []
    for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
        all_remaining.extend([p for p in players_by_pos[pos] if p['id'] not in used_ids])
    
    util_player = select_player(
        all_remaining, strategy, used_ids, exposure_tracker, max_exposure, total_lineups
    )
    
    if not util_player:
        return None
    
    util_copy = util_player.copy()
    util_copy['rosterPosition'] = 'UTIL'
    lineup_players.append(util_copy)
    used_ids.add(util_player['id'])
    total_salary += util_player['salary']
    total_projection += util_player['projection']
    
    # Validate salary constraints
    if total_salary < min_salary or total_salary > max_salary:
        return None
    
    return {
        'players': lineup_players,
        'totalSalary': total_salary,
        'totalProjection': round(total_projection, 2),
        'strategy': strategy
    }


def select_player(player_pool, strategy, used_ids, exposure_tracker, 
                  max_exposure, total_lineups):
    """Select a player from the pool based on strategy"""
    
    # Filter eligible players
    eligible = [
        p for p in player_pool
        if p['id'] not in used_ids
        and exposure_tracker.get(p['id'], 0) < (max_exposure / 100 * total_lineups)
    ]
    
    if not eligible:
        return None
    
    # Sort by strategy
    if strategy == 'high_projection':
        eligible.sort(key=lambda x: x['projection'], reverse=True)
        # 80% from top 3, 20% from top 6
        pool_size = min(3 if np.random.random() < 0.8 else 6, len(eligible))
    elif strategy == 'value':
        eligible.sort(key=lambda x: x['projection'] / x['salary'], reverse=True)
        pool_size = min(5, len(eligible))
    elif strategy == 'balanced':
        # Balance of projection and value
        eligible.sort(key=lambda x: (x['projection'] * 0.7 + (x['projection'] / x['salary'] * 1000) * 0.3), reverse=True)
        pool_size = min(4, len(eligible))
    else:  # contrarian
        # Use randomness for contrarian plays
        pool_size = min(8, len(eligible))
    
    # Select from top pool
    selected_idx = np.random.randint(0, pool_size)
    return eligible[selected_idx]


def main():
    """Main CLI entry point"""
    
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input data provided"}))
        sys.exit(1)
    
    try:
        # Read input from command line argument (JSON string)
        input_data = json.loads(sys.argv[1])
        
        players = input_data.get('players', [])
        num_lineups = input_data.get('numLineups', 10)
        min_salary = input_data.get('minSalary', 48000)
        max_salary = input_data.get('maxSalary', 50000)
        unique_players = input_data.get('uniquePlayers', 3)
        max_exposure = input_data.get('maxExposure', 100)
        stack_settings = input_data.get('stackSettings', {})
        
        # Run optimization
        lineups = optimize_nba_lineups(
            players,
            num_lineups,
            min_salary,
            max_salary,
            unique_players,
            max_exposure,
            stack_settings
        )
        
        # Output result as JSON
        output = {
            "success": True,
            "lineups": lineups,
            "summary": {
                "totalLineups": len(lineups),
                "avgProjection": round(np.mean([l['totalProjection'] for l in lineups]), 2) if lineups else 0,
                "avgSalary": round(np.mean([l['totalSalary'] for l in lineups]), 2) if lineups else 0,
                "topProjection": round(max([l['totalProjection'] for l in lineups]), 2) if lineups else 0,
            }
        }
        
        print(json.dumps(output))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()

