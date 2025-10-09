import pandas as pd
import numpy as np

def analyze_probability_projections():
    """
    Analyze and display probability projections in an easy-to-read format
    """
    try:
        df = pd.read_csv('c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/dfs_projections_with_probabilities.csv')
        print("DFS Projections with Probability Analysis")
        print("="*80)
        print(f"Total Players: {len(df)}")
        print(f"Positions: {df['Pos'].value_counts().to_dict()}")
        print("="*80)
        
        # Top players by position
        positions = ['P', 'C', '1B', '2B', '3B', 'SS', 'OF']
        
        for pos in positions:
            pos_df = df[df['Pos'] == pos].head(5)
            if len(pos_df) > 0:
                print(f"\nTop {pos} Players:")
                print("-" * 50)
                for _, row in pos_df.iterrows():
                    print(f"{row['Name']:<20} {row['Predicted_Points']:>6.1f} pts | "
                          f"P(>10): {row['Prob_Over_10']:>6} | "
                          f"P(>20): {row['Prob_Over_20']:>6} | "
                          f"P(>30): {row['Prob_Over_30']:>6}")
        
        # High probability players for different thresholds
        print("\n" + "="*80)
        print("HIGH PROBABILITY PLAYS")
        print("="*80)
        
        # Players with high probability of 10+ points
        high_prob_10 = df[df['Prob_Over_10'].str.replace('%', '').astype(float) > 80]
        if len(high_prob_10) > 0:
            print(f"\nPlayers with >80% chance of 10+ points ({len(high_prob_10)} players):")
            print("-" * 60)
            for _, row in high_prob_10.head(10).iterrows():
                print(f"{row['Name']:<20} ({row['Pos']:<3}) {row['Predicted_Points']:>6.1f} pts | P(>10): {row['Prob_Over_10']}")
        
        # Players with decent probability of 20+ points
        high_prob_20 = df[df['Prob_Over_20'].str.replace('%', '').astype(float) > 10]
        if len(high_prob_20) > 0:
            print(f"\nPlayers with >10% chance of 20+ points ({len(high_prob_20)} players):")
            print("-" * 60)
            for _, row in high_prob_20.head(10).iterrows():
                print(f"{row['Name']:<20} ({row['Pos']:<3}) {row['Predicted_Points']:>6.1f} pts | P(>20): {row['Prob_Over_20']}")
        
        # Value plays (low salary, decent upside)
        if 'Salary' in df.columns:
            value_plays = df[(df['Salary'] < df['Salary'].median()) & 
                           (df['Prob_Over_10'].str.replace('%', '').astype(float) > 50)]
            if len(value_plays) > 0:
                print(f"\nValue Plays (Below median salary, >50% chance of 10+ points):")
                print("-" * 70)
                for _, row in value_plays.head(10).iterrows():
                    print(f"{row['Name']:<20} ({row['Pos']:<3}) ${row['Salary']:<6} | "
                          f"{row['Predicted_Points']:>6.1f} pts | P(>10): {row['Prob_Over_10']}")
        
        print("\n" + "="*80)
        
    except FileNotFoundError:
        print("Error: dfs_projections_with_probabilities.csv not found.")
        print("Please run add_probability_predictions.py first.")
    except Exception as e:
        print(f"Error analyzing projections: {e}")

def create_lineup_optimizer_input():
    """
    Create a simplified file for lineup optimizers
    """
    try:
        df = pd.read_csv('c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/dfs_projections_with_probabilities.csv')
        
        # Create optimizer-friendly format
        optimizer_df = df.copy()
        
        # Add ceiling and floor estimates based on probabilities
        optimizer_df['Ceiling'] = optimizer_df['Predicted_Points'] + (
            optimizer_df['Prob_Over_30'].str.replace('%', '').astype(float) / 100 * 10
        )
        optimizer_df['Floor'] = optimizer_df['Predicted_Points'] * (
            optimizer_df['Prob_Over_10'].str.replace('%', '').astype(float) / 100
        )
        
        # Save optimizer input
        optimizer_df.to_csv('c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/optimizer_input_with_probabilities.csv', index=False)
        print("Created optimizer_input_with_probabilities.csv")
        
    except Exception as e:
        print(f"Error creating optimizer input: {e}")

if __name__ == "__main__":
    analyze_probability_projections()
    create_lineup_optimizer_input()
