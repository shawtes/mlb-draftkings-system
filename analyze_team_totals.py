#!/usr/bin/env python3
import pandas as pd
import numpy as np

def analyze_team_totals():
    # Load the data
    df = pd.read_csv('6_OPTIMIZATION/nfl_week7_CASH_SPORTSDATA.csv')
    
    # Group by team and sum the projected points
    team_totals = df.groupby('Team')['Predicted_DK_Points'].sum().sort_values(ascending=False)
    
    print('🏈 NFL WEEK 7 - TEAMS BY PROJECTED FANTASY POINTS')
    print('=' * 60)
    print()
    
    for i, (team, total) in enumerate(team_totals.head(10).items(), 1):
        print(f'{i:2d}. {team:3s} - {total:6.1f} total fantasy points')
    
    print()
    print('📊 DETAILED BREAKDOWN:')
    print('-' * 60)
    
    # Show top 3 teams with player breakdown
    for team in team_totals.head(3).index:
        team_players = df[df['Team'] == team].sort_values('Predicted_DK_Points', ascending=False)
        print(f'\n{team} ({team_totals[team]:.1f} total points):')
        for _, player in team_players.head(5).iterrows():
            pos = player['Position']
            name = player['Name']
            points = player['Predicted_DK_Points']
            salary = player['Salary']
            print(f'  {pos:2s} {name:20s} - {points:5.1f} pts (${salary:4.0f})')
    
    print()
    print('🎯 TOP STACKING OPPORTUNITIES:')
    print('-' * 60)
    
    # Find teams with multiple high-scoring players
    for team in team_totals.head(5).index:
        team_players = df[df['Team'] == team].sort_values('Predicted_DK_Points', ascending=False)
        high_scorers = team_players[team_players['Predicted_DK_Points'] >= 12.0]
        if len(high_scorers) >= 3:
            print(f'\n{team} - {len(high_scorers)} players with 12+ projected points:')
            for _, player in high_scorers.iterrows():
                pos = player['Position']
                name = player['Name']
                points = player['Predicted_DK_Points']
                print(f'  {pos:2s} {name:20s} - {points:5.1f} pts')

if __name__ == "__main__":
    analyze_team_totals()



