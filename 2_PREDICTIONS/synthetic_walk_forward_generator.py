#!/usr/bin/env python3
"""
Synthetic Walk-Forward Data Generator for Team Selection Classifier
Creates realistic synthetic data for training the team selection classifier
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class SyntheticDataGenerator:
    """Generate synthetic walk-forward prediction data for team classifier training"""
    
    def __init__(self, num_players=200, prediction_days=200):
        self.num_players = num_players
        self.prediction_days = prediction_days
        self.player_names = self.generate_player_names()
        
    def generate_player_names(self):
        """Generate realistic player names"""
        first_names = [
            'Mike', 'John', 'David', 'Chris', 'Matt', 'Ryan', 'Alex', 'Justin', 'Kyle', 'Tyler',
            'Kevin', 'Brian', 'Jason', 'Josh', 'Nick', 'Jake', 'Adam', 'Mark', 'Steve', 'Dan',
            'Carlos', 'Jose', 'Luis', 'Juan', 'Miguel', 'Antonio', 'Francisco', 'Ramon', 'Pedro',
            'Mookie', 'Ronald', 'Fernando', 'Yordan', 'Vladimir', 'Gleyber', 'Rafael', 'Salvador'
        ]
        
        last_names = [
            'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
            'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
            'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson',
            'Betts', 'Acuna', 'Tatis', 'Judge', 'Trout', 'Harper', 'Soto', 'Freeman',
            'Guerrero', 'Torres', 'Devers', 'Perez', 'Alvarez', 'Machado', 'Turner'
        ]
        
        names = []
        for i in range(self.num_players):
            first = random.choice(first_names)
            last = random.choice(last_names)
            names.append(f"{first} {last}")
        
        return list(set(names))[:self.num_players]  # Remove duplicates
    
    def generate_player_skill_levels(self):
        """Generate skill levels for players (affects their performance consistency)"""
        skill_levels = {}
        
        for player in self.player_names:
            # Create different player archetypes
            archetype = random.choice(['superstar', 'star', 'regular', 'bench', 'rookie'])
            
            if archetype == 'superstar':
                base_points = random.uniform(12, 18)
                volatility = random.uniform(0.15, 0.25)
                consistency = random.uniform(0.8, 0.95)
            elif archetype == 'star':
                base_points = random.uniform(9, 14)
                volatility = random.uniform(0.2, 0.35)
                consistency = random.uniform(0.7, 0.85)
            elif archetype == 'regular':
                base_points = random.uniform(6, 10)
                volatility = random.uniform(0.25, 0.4)
                consistency = random.uniform(0.6, 0.75)
            elif archetype == 'bench':
                base_points = random.uniform(3, 7)
                volatility = random.uniform(0.3, 0.5)
                consistency = random.uniform(0.4, 0.65)
            else:  # rookie
                base_points = random.uniform(2, 8)
                volatility = random.uniform(0.4, 0.6)
                consistency = random.uniform(0.3, 0.6)
            
            skill_levels[player] = {
                'archetype': archetype,
                'base_points': base_points,
                'volatility': volatility,
                'consistency': consistency
            }
        
        return skill_levels
    
    def simulate_daily_performance(self, player, skill_level, date, recent_performance):
        """Simulate a player's performance for a given day"""
        base = skill_level['base_points']
        volatility = skill_level['volatility']
        consistency = skill_level['consistency']
        
        # Add some randomness
        random_factor = np.random.normal(0, volatility)
        
        # Add momentum based on recent performance
        if len(recent_performance) > 0:
            recent_avg = np.mean(recent_performance[-5:])  # Last 5 games
            momentum = (recent_avg - base) * 0.3  # 30% momentum carry-over
        else:
            momentum = 0
        
        # Add consistency factor
        consistency_factor = np.random.normal(consistency, 0.1)
        
        # Calculate predicted points
        predicted_points = base + random_factor + momentum
        predicted_points *= consistency_factor
        
        # Ensure realistic range
        predicted_points = max(0, min(predicted_points, 50))
        
        return predicted_points
    
    def generate_walk_forward_data(self, start_date='2024-04-01', end_date='2024-10-31'):
        """Generate complete walk-forward prediction dataset"""
        
        print("Generating synthetic walk-forward prediction data...")
        
        # Generate skill levels for all players
        skill_levels = self.generate_player_skill_levels()
        
        # Generate date range
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        # Filter to baseball season (exclude some off days)
        mlb_dates = [d for d in date_range if d.month >= 4 and d.month <= 10 and d.weekday() < 6]
        
        print(f"Generating data for {len(mlb_dates)} dates and {len(self.player_names)} players")
        
        all_predictions = []
        player_histories = {player: [] for player in self.player_names}
        
        for i, current_date in enumerate(mlb_dates):
            if i % 20 == 0:
                print(f"Processing date {i+1}/{len(mlb_dates)}: {current_date.date()}")
            
            daily_predictions = []
            
            for player in self.player_names:
                skill_level = skill_levels[player]
                recent_performance = player_histories[player]
                
                # Simulate the prediction
                predicted_points = self.simulate_daily_performance(
                    player, skill_level, current_date, recent_performance
                )
                
                # Add to history
                player_histories[player].append(predicted_points)
                
                # Keep only recent history (last 30 games)
                if len(player_histories[player]) > 30:
                    player_histories[player] = player_histories[player][-30:]
                
                # Create prediction record
                prediction = {
                    'Name': player,
                    'Date': current_date.date(),
                    'Predicted_DK_Points': predicted_points,
                    'Player_Archetype': skill_level['archetype'],
                    'Base_Skill_Level': skill_level['base_points'],
                    'Consistency_Score': skill_level['consistency'],
                    'Recent_Avg_5_Games': np.mean(recent_performance[-5:]) if len(recent_performance) >= 5 else predicted_points,
                    'Recent_Avg_10_Games': np.mean(recent_performance[-10:]) if len(recent_performance) >= 10 else predicted_points,
                    'Games_Played': len(recent_performance),
                    'Volatility': skill_level['volatility']
                }
                
                daily_predictions.append(prediction)
            
            all_predictions.extend(daily_predictions)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_predictions)
        
        print(f"Generated {len(df)} total predictions")
        print(f"Players: {df['Name'].nunique()}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Average predicted points: {df['Predicted_DK_Points'].mean():.2f}")
        
        return df
    
    def add_team_construction_features(self, df):
        """Add features relevant for team construction"""
        
        print("Adding team construction features...")
        
        # Sort by date to ensure proper ordering
        df = df.sort_values(['Date', 'Predicted_DK_Points'], ascending=[True, False])
        
        # Add daily rankings
        df['Daily_Rank'] = df.groupby('Date')['Predicted_DK_Points'].rank(method='dense', ascending=False)
        df['Daily_Percentile'] = df.groupby('Date')['Predicted_DK_Points'].rank(pct=True)
        
        # Add salary simulation (correlate with skill level but add noise)
        df['Simulated_Salary'] = (
            df['Base_Skill_Level'] * 800 + 
            np.random.normal(0, 1000, len(df)) + 
            4000
        ).clip(4000, 12000).round(100)  # DraftKings salary range
        
        # Add position simulation
        positions = ['C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF', 'DH']  # DK positions
        df['Position'] = df['Name'].apply(lambda x: random.choice(positions))
        
        # Add value calculations
        df['Points_Per_Dollar'] = df['Predicted_DK_Points'] / (df['Simulated_Salary'] / 1000)
        
        # Add team construction metrics
        df['Is_Top_10_Day'] = df['Daily_Rank'] <= 10
        df['Is_Top_25_Day'] = df['Daily_Rank'] <= 25
        df['Is_Value_Play'] = df['Points_Per_Dollar'] > df['Points_Per_Dollar'].quantile(0.75)
        
        print("Team construction features added")
        return df

def create_team_selection_training_data():
    """Create synthetic training data for team selection classifier"""
    
    print("="*60)
    print("SYNTHETIC TEAM SELECTION TRAINING DATA GENERATOR")
    print("="*60)
    
    # Generate synthetic data
    generator = SyntheticDataGenerator(num_players=250, prediction_days=200)
    df = generator.generate_walk_forward_data()
    
    # Add team construction features
    df = generator.add_team_construction_features(df)
    
    # Save the data
    output_file = '2_PREDICTIONS/synthetic_walk_forward_predictions_2024.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\nSynthetic data saved to: {output_file}")
    
    # Display summary
    print(f"\nSynthetic Dataset Summary:")
    print(f"Total records: {len(df):,}")
    print(f"Unique players: {df['Name'].nunique()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Unique dates: {df['Date'].nunique()}")
    
    print(f"\nPlayer Archetypes:")
    archetype_counts = df['Player_Archetype'].value_counts()
    for archetype, count in archetype_counts.items():
        print(f"  {archetype}: {count:,} predictions")
    
    print(f"\nPredicted Points Statistics:")
    print(f"  Mean: {df['Predicted_DK_Points'].mean():.2f}")
    print(f"  Std: {df['Predicted_DK_Points'].std():.2f}")
    print(f"  Min: {df['Predicted_DK_Points'].min():.2f}")
    print(f"  Max: {df['Predicted_DK_Points'].max():.2f}")
    
    print(f"\nTop 10 Predicted Performances:")
    top_performances = df.nlargest(10, 'Predicted_DK_Points')
    for _, row in top_performances.iterrows():
        print(f"  {row['Name']} ({row['Player_Archetype']}): {row['Predicted_DK_Points']:.2f} points on {row['Date']}")
    
    print("\nSynthetic walk-forward data generation complete!")
    print("This data can now be used to train the team selection classifier.")
    
    return df

if __name__ == "__main__":
    df = create_team_selection_training_data()
