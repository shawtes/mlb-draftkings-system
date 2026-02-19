#!/usr/bin/env python3
"""
Test script for PuLP-integrated RL system
"""

import pandas as pd
import numpy as np
import logging
from realistic_rl_system import RealisticMLBRLSystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pulp_lineup_generation():
    """Test PuLP lineup generation with sample data"""
    
    logger.info("=== Testing PuLP Lineup Generation ===")
    
    # Configuration
    DATA_PATH = '5_ENTRIES/data_with_dk_entries_salaries.csv'
    PREDICTION_MODEL_PATH = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/batters_final_ensemble_model_pipeline01.pkl'
    
    # Initialize system
    rl_system = RealisticMLBRLSystem(DATA_PATH, PREDICTION_MODEL_PATH)
    
    # Load sample data
    logger.info("Loading sample data...")
    try:
        sample_data = pd.read_csv(DATA_PATH, nrows=100)
        logger.info(f"Loaded {len(sample_data)} sample records")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Create sample players data with proper DK positions
    players_data = []
    position_counts = {'C': 0, '1B': 0, '2B': 0, '3B': 0, 'SS': 0, 'OF': 0}
    
    for _, row in sample_data.iterrows():
        # Get position, ensuring we have enough of each
        original_pos = row.get('position', 'OF')
        
        # Map to standard DK positions
        if original_pos in ['SP', 'RP']:
            continue  # Skip pitchers for now
        elif original_pos in ['1B/C', '1B/3B', '1B/OF']:
            dk_position = '1B'
        elif original_pos in ['2B/3B', '2B/SS', '2B/OF', '2B/C']:
            dk_position = '2B'
        elif original_pos in ['3B/OF', '3B/SS']:
            dk_position = '3B'
        elif original_pos == 'SS':
            dk_position = 'SS'
        elif original_pos == 'C':
            dk_position = 'C'
        else:
            dk_position = 'OF'
        
        # Ensure position distribution
        if dk_position != 'OF' and position_counts[dk_position] >= 3:
            dk_position = 'OF'
        
        # Count positions
        position_counts[dk_position] += 1
        
        players_data.append({
            'name': row.get('Name', f'Player_{len(players_data)}'),
            'position': dk_position,
            'salary': int(row.get('salary', 5000)),
            'projected_points': float(row.get('rolling_30_ppg', 10)),
            'actual_points': float(row.get('calculated_dk_fpts', 8))
        })
        
        if len(players_data) >= 50:  # Enough for testing
            break
    
    logger.info(f"Created {len(players_data)} player entries")
    logger.info(f"Position distribution: {position_counts}")
    
    # Test PuLP lineup generation
    logger.info("\nGenerating lineup with PuLP...")
    try:
        lineup = rl_system.generate_lineup_with_pulp(players_data)
        
        if lineup is not None and len(lineup) > 0:
            logger.info("✅ PuLP Lineup Generated Successfully!")
            logger.info(f"Players: {len(lineup)}")
            logger.info(f"Total Salary: ${lineup['salary'].sum():,.0f}")
            logger.info(f"Total Projected Points: {lineup['projected_points'].sum():.1f}")
            
            # Validate DK constraints
            logger.info("\nValidating DraftKings constraints...")
            
            # Check lineup size
            if len(lineup) == 8:
                logger.info("✅ Lineup size: 8 players")
            else:
                logger.warning(f"❌ Lineup size: {len(lineup)} players (should be 8)")
            
            # Check salary cap
            total_salary = lineup['salary'].sum()
            if total_salary <= 50000:
                logger.info(f"✅ Salary: ${total_salary:,} (within cap)")
            else:
                logger.warning(f"❌ Salary: ${total_salary:,} (exceeds cap)")
            
            # Check positions
            position_counts = lineup['position'].value_counts()
            logger.info(f"Position distribution: {position_counts.to_dict()}")
            
            # Show lineup
            logger.info("\n📋 Lineup Details:")
            for i, (_, player) in enumerate(lineup.iterrows(), 1):
                logger.info(f"  {i}. {player['name']} ({player['position']}) - "
                           f"${player['salary']:,.0f} - {player['projected_points']:.1f} pts")
            
            # Calculate efficiency
            efficiency = lineup['projected_points'].sum() / (lineup['salary'].sum() / 1000)
            logger.info(f"\n📊 Efficiency: {efficiency:.2f} points per $1,000")
            
        else:
            logger.error("❌ PuLP failed to generate lineup")
            
    except Exception as e:
        logger.error(f"❌ Error in PuLP lineup generation: {e}")
        import traceback
        traceback.print_exc()

def test_method_comparison():
    """Test comparison between PuLP and RL methods"""
    
    logger.info("\n=== Testing Method Comparison ===")
    
    # This would require trained RL agent, so we'll skip for now
    logger.info("Method comparison requires trained RL agent")
    logger.info("Use the full main() function to train and compare methods")

def main():
    """Main test function"""
    
    logger.info("🚀 Starting PuLP-RL Integration Tests")
    
    # Test PuLP lineup generation
    test_pulp_lineup_generation()
    
    # Test method comparison
    test_method_comparison()
    
    logger.info("\n✅ Tests completed!")

if __name__ == "__main__":
    main()
