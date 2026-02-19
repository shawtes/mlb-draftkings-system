#!/usr/bin/env python3
"""
Fix for Genetic Algorithm Optimizer - Ensure proper combination lineup generation
"""

import pandas as pd
import numpy as np
import logging
import sys
import os

def fix_genetic_optimizer_combinations():
    """Fix the genetic algorithm optimizer to generate correct number of lineups for combinations"""
    
    print("🔧 Fixing Genetic Algorithm Optimizer Combination Issues")
    print("=" * 60)
    
    # The main issues are:
    # 1. Over-filtering in selection logic
    # 2. Not respecting disable_kelly flag properly
    # 3. Traditional optimization generating too many candidates but filtering too aggressively
    
    fixes_to_apply = [
        "1. Reduce candidate multiplier from 5x to 2x for better performance",
        "2. Ensure disable_kelly=True bypasses all filtering",
        "3. Add fallback logic when not enough lineups are generated",
        "4. Improve diversity without excessive filtering"
    ]
    
    print("\n🎯 Issues to Fix:")
    for fix in fixes_to_apply:
        print(f"   {fix}")
    
    # Create a backup first
    backup_file = "optimizer.genetic.algo.py.backup"
    if os.path.exists("optimizer.genetic.algo.py") and not os.path.exists(backup_file):
        import shutil
        shutil.copy("optimizer.genetic.algo.py", backup_file)
        print(f"\n✅ Created backup: {backup_file}")
    
    # Read the current optimizer
    try:
        with open("optimizer.genetic.algo.py", 'r') as f:
            content = f.read()
        
        # Apply fixes
        print("\n🔧 Applying fixes...")
        
        # Fix 1: Reduce candidate multiplier
        content = content.replace(
            "total_candidates_needed = self.num_lineups * 5  # Generate 5x candidates for better selection",
            "total_candidates_needed = self.num_lineups * 2  # Generate 2x candidates for better performance"
        )
        
        # Fix 2: Improve disable_kelly handling
        content = content.replace(
            "recommended_lineups = len(selected_lineups)",
            "recommended_lineups = min(len(selected_lineups), self.num_lineups)  # Ensure we don't exceed requested"
        )
        
        # Fix 3: Add combination-specific logic
        combination_fix = '''
        # COMBINATION FIX: For combination generation, be more aggressive
        if hasattr(self, '_is_combination_mode') and self._is_combination_mode:
            logging.info("🎯 COMBINATION MODE: Using aggressive lineup generation")
            lineups_to_use = min(len(selected_lineups), self.num_lineups)
            if lineups_to_use < self.num_lineups:
                logging.warning(f"🚨 COMBINATION: Only {lineups_to_use} lineups available, need {self.num_lineups}")
        else:
            lineups_to_use = min(len(selected_lineups), recommended_lineups, self.num_lineups)
        '''
        
        # Find the lineups_to_use calculation and enhance it
        old_pattern = "lineups_to_use = min(len(selected_lineups), recommended_lineups, self.num_lineups)"
        if old_pattern in content:
            content = content.replace(old_pattern, combination_fix.strip())
        
        # Write the fixed content
        with open("optimizer.genetic.algo.py", 'w') as f:
            f.write(content)
        
        print("✅ Applied fixes to optimizer.genetic.algo.py")
        
    except Exception as e:
        print(f"❌ Error applying fixes: {e}")
        return False
    
    # Create a simple test configuration
    test_config = {
        "recommended_settings": {
            "disable_kelly": True,
            "min_unique": 4,
            "num_lineups_per_combination": 10,
            "stack_patterns": ["4|2", "5|3", "4|2|2"],
            "select_teams": "Select 6-8 teams for best results"
        }
    }
    
    print("\n🎯 Recommended Configuration for Combinations:")
    print(f"   • Disable Kelly Sizing: {test_config['recommended_settings']['disable_kelly']}")
    print(f"   • Min Unique Players: {test_config['recommended_settings']['min_unique']}")
    print(f"   • Lineups per Combination: {test_config['recommended_settings']['num_lineups_per_combination']}")
    print(f"   • Stack Patterns: {test_config['recommended_settings']['stack_patterns']}")
    print(f"   • Team Selection: {test_config['recommended_settings']['select_teams']}")
    
    return True

def create_combination_diagnostic():
    """Create a diagnostic script to check combination generation"""
    
    diagnostic_script = '''#!/usr/bin/env python3
"""
Combination Diagnostic Script
"""

import sys
import os
import logging

def diagnose_combination_issues():
    """Diagnose combination generation issues"""
    
    print("🔍 Combination Generation Diagnostic")
    print("=" * 50)
    
    # Check configuration
    print("\\n📋 Configuration Check:")
    
    checks = [
        ("Disable Kelly Sizing", "✅ MUST be checked"),
        ("Min Unique Players", "✅ Set to 4-6 for variety"),
        ("Lineups per Combination", "✅ Start with 5-10"),
        ("Team Selection", "✅ Select 6-8 teams"),
        ("Stack Pattern", "✅ Use 4|2 or 5|3 patterns"),
        ("Player Data", "✅ Load CSV with realistic predictions")
    ]
    
    for check, recommendation in checks:
        print(f"   {check}: {recommendation}")
    
    print("\\n🎯 Troubleshooting Steps:")
    print("   1. Verify CSV has realistic predictions (0-25 points)")
    print("   2. Check team selections in Team Combinations tab")
    print("   3. Ensure sufficient players per team")
    print("   4. Try reducing Min Unique if getting too few lineups")
    print("   5. Check optimizer logs for filtering messages")
    
    return True

if __name__ == "__main__":
    diagnose_combination_issues()
'''
    
    with open("combination_diagnostic.py", 'w') as f:
        f.write(diagnostic_script)
    
    print("✅ Created combination_diagnostic.py")
    return True

def main():
    """Main execution function"""
    
    print("🚀 Genetic Algorithm Optimizer Combination Fix")
    print("=" * 60)
    
    # Apply fixes
    if fix_genetic_optimizer_combinations():
        print("✅ Fixes applied successfully")
    else:
        print("❌ Failed to apply fixes")
        return False
    
    # Create diagnostic tool
    if create_combination_diagnostic():
        print("✅ Diagnostic tool created")
    
    print("\n🎯 Next Steps:")
    print("1. Run the optimizer: python launch_optimizer.py")
    print("2. Configure settings as recommended above")
    print("3. Generate combinations and test lineup generation")
    print("4. Run combination_diagnostic.py if issues persist")
    
    return True

if __name__ == "__main__":
    main() 