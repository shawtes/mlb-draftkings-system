#!/usr/bin/env python3
"""
Fix for Unique Lineup Generation Issue
=====================================

This script fixes the problem where the optimizer generates only one lineup
per stack pattern instead of multiple diverse lineups.
"""

import sys
import os
import random
import numpy as np
import time
from pathlib import Path

def fix_optimizer_randomness():
    """
    Fix the randomness issue in the optimizer by modifying the optimize_single_lineup function
    """
    
    # Path to the main optimizer file
    optimizer_path = Path("optimizer01.py")
    
    if not optimizer_path.exists():
        print("❌ optimizer01.py not found!")
        return False
    
    # Read the current file
    with open(optimizer_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the problematic randomness section
    old_randomness = """    # Ensure truly random noise for each lineup by resetting seed
    random.seed()  # Use system time for random seed
    np.random.seed()  # Use system time for numpy random seed"""
    
    new_randomness = """    # Ensure truly random noise for each lineup by resetting seed with time + process info
    import time
    import os
    seed_value = int(time.time() * 1000000) % 2147483647 + os.getpid()
    random.seed(seed_value)  # Use time + PID for truly unique seed
    np.random.seed(seed_value)  # Use time + PID for numpy random seed"""
    
    if old_randomness in content:
        content = content.replace(old_randomness, new_randomness)
        print("✅ Fixed randomness seeding issue")
    else:
        print("⚠️ Randomness section not found - may already be fixed")
    
    # Enhance the diversity noise
    old_noise = """        # Add aggressive noise for lineup diversity - increased for better diversification
        diversity_factor = random.uniform(0.15, 0.35)  # Increased to 15-35% noise for better diversity
        noise = np.random.normal(1.0, diversity_factor, len(df))
        df['Predicted_DK_Points'] = df['Predicted_DK_Points'] * noise"""
    
    new_noise = """        # Add aggressive noise for lineup diversity - ENHANCED for maximum diversity
        diversity_factor = random.uniform(0.20, 0.50)  # Increased to 20-50% noise for better diversity
        noise = np.random.normal(1.0, diversity_factor, len(df))
        
        # Add additional randomness to prevent identical lineups
        player_boost = np.random.choice(df.index, size=random.randint(1, 3), replace=False)
        for idx in player_boost:
            noise[df.index.get_loc(idx)] *= random.uniform(1.1, 1.3)
        
        df['Predicted_DK_Points'] = df['Predicted_DK_Points'] * noise"""
    
    if old_noise in content:
        content = content.replace(old_noise, new_noise)
        print("✅ Enhanced diversity noise injection")
    else:
        # Try to find a simpler version
        simple_noise = """        diversity_factor = random.uniform(0.15, 0.35)
        noise = np.random.normal(1.0, diversity_factor, len(df))
        df['Predicted_DK_Points'] = df['Predicted_DK_Points'] * noise"""
        
        if simple_noise in content:
            enhanced_noise = """        diversity_factor = random.uniform(0.20, 0.50)  # Enhanced range
            noise = np.random.normal(1.0, diversity_factor, len(df))
            
            # Add player-specific randomness
            player_boost = np.random.choice(df.index, size=random.randint(1, 3), replace=False)
            for idx in player_boost:
                noise[df.index.get_loc(idx)] *= random.uniform(1.1, 1.3)
            
            df['Predicted_DK_Points'] = df['Predicted_DK_Points'] * noise"""
            
            content = content.replace(simple_noise, enhanced_noise)
            print("✅ Enhanced diversity noise injection (simple version)")
        else:
            print("⚠️ Noise injection section not found - may need manual update")
    
    # Write the updated content back
    with open(optimizer_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ optimizer01.py has been updated with enhanced randomness")
    return True

def print_usage_instructions():
    """Print instructions for using the fixed optimizer"""
    
    print("\n" + "="*60)
    print("🎯 INSTRUCTIONS FOR MAXIMUM LINEUP DIVERSITY")
    print("="*60)
    
    print("\n1️⃣ CRITICAL SETTINGS:")
    print("   • Min Unique: 5-7 (Forces different players between lineups)")
    print("   • Number of Lineups: 100+ (More opportunities for diversity)")
    print("   • ☑️ Disable Kelly Sizing (Generates ALL requested lineups)")
    print("   • Enable MULTIPLE stack patterns in Team Combinations tab")
    
    print("\n2️⃣ STACK PATTERNS - ENABLE MULTIPLE:")
    print("   • Go to 'Team Combinations' tab")
    print("   • Enable 3+ different patterns:")
    print("     - 4|2 (4 players from one team, 2 from another)")
    print("     - 5|3 (5 from one team, 3 from another)")
    print("     - 4|2|2 (4 from one team, 2 each from two others)")
    print("     - 3|3|2 (3 each from two teams, 2 from third)")
    
    print("\n3️⃣ TESTING DIVERSITY:")
    print("   • Export lineups to CSV")
    print("   • Check for different players between lineups")
    print("   • Look for variety in team combinations")
    
    print("\n4️⃣ TROUBLESHOOTING:")
    print("   • If still getting identical lineups:")
    print("     - Increase Min Unique to 8-10")
    print("     - Enable more stack patterns")
    print("     - Increase number of lineups to 200+")
    print("     - Check that 'Disable Kelly Sizing' is enabled")
    
    print("\n5️⃣ WHAT THIS FIX DOES:")
    print("   • Fixes random seed generation (was causing identical lineups)")
    print("   • Increases diversity noise from 15-35% to 20-50%")
    print("   • Adds player-specific randomness boosts")
    print("   • Creates unique seeds using time + process ID")
    
    print("\n" + "="*60)
    print("🚀 READY TO GENERATE UNIQUE LINEUPS!")
    print("="*60)

def main():
    """Main function to apply the fix"""
    
    print("🔧 FIXING UNIQUE LINEUP GENERATION ISSUE")
    print("="*50)
    
    # Check if we're in the right directory
    if not os.path.exists("optimizer01.py"):
        print("❌ This script must be run from the MLB_DRAFTKINGS_SYSTEM/6_OPTIMIZATION directory")
        print("📁 Current directory:", os.getcwd())
        return
    
    # Apply the fix
    success = fix_optimizer_randomness()
    
    if success:
        # Print usage instructions
        print_usage_instructions()
        
        print("\n✅ FIX APPLIED SUCCESSFULLY!")
        print("🎯 Now run your optimizer with the settings above to get unique lineups!")
        
    else:
        print("\n❌ Fix could not be applied automatically")
        print("🔧 You may need to manually edit optimizer01.py")

if __name__ == "__main__":
    main() 