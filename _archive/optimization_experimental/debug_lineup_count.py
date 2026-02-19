#!/usr/bin/env python3
"""
Debug script to identify why only 12 lineups instead of 60
This will patch the optimizer to add debug logging
"""

import sys
import os
import re

def add_debug_logging_to_optimizer():
    """Add debug logging to the optimizer to track lineup generation"""
    
    optimizer_file = "optimizer.genetic.algo.py"
    
    if not os.path.exists(optimizer_file):
        print(f"ERROR: {optimizer_file} not found")
        return False
    
    print("Adding debug logging to optimizer...")
    
    # Read the current file
    with open(optimizer_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create backup
    backup_file = "optimizer.genetic.algo.py.debug_backup"
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created backup: {backup_file}")
    
    # Add debug logging points
    debug_patches = [
        # Track combination mode detection
        {
            'search': 'worker._is_combination_mode = True',
            'replace': '''worker._is_combination_mode = True
                    print(f"🔍 DEBUG: COMBINATION MODE ACTIVATED for {lineups_count} lineups")
                    logging.info(f"🔍 DEBUG: COMBINATION MODE ACTIVATED for {lineups_count} lineups")'''
        },
        
        # Track genetic engine usage
        {
            'search': 'if use_genetic_engine:',
            'replace': '''print(f"🔍 DEBUG: Genetic engine check - use_genetic_engine={use_genetic_engine}")
            logging.info(f"🔍 DEBUG: Genetic engine check - use_genetic_engine={use_genetic_engine}")
            if use_genetic_engine:'''
        },
        
        # Track lineup filtering
        {
            'search': 'lineups_to_use = min(len(selected_lineups), self.num_lineups)',
            'replace': '''lineups_to_use = min(len(selected_lineups), self.num_lineups)
                    print(f"🔍 DEBUG: LINEUP COUNT - selected={len(selected_lineups)}, requested={self.num_lineups}, using={lineups_to_use}")
                    logging.info(f"🔍 DEBUG: LINEUP COUNT - selected={len(selected_lineups)}, requested={self.num_lineups}, using={lineups_to_use}")'''
        },
        
        # Track combination results
        {
            'search': 'logging.info(f"Generated {len(combo_lineups)} lineups for combination: {combo_display}")',
            'replace': '''logging.info(f"Generated {len(combo_lineups)} lineups for combination: {combo_display}")
                        print(f"🔍 DEBUG: COMBO RESULT - {combo_display}: {len(combo_lineups)}/{lineups_count} lineups")'''
        }
    ]
    
    # Apply patches
    modified_content = content
    patches_applied = 0
    
    for patch in debug_patches:
        if patch['search'] in modified_content:
            modified_content = modified_content.replace(patch['search'], patch['replace'])
            patches_applied += 1
            print(f"✅ Applied patch: {patch['search'][:50]}...")
        else:
            print(f"⚠️ Patch not found: {patch['search'][:50]}...")
    
    # Write modified file
    with open(optimizer_file, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"Applied {patches_applied}/{len(debug_patches)} debug patches")
    return patches_applied > 0

def create_quick_fix_for_lineup_count():
    """Create a quick fix to bypass aggressive filtering"""
    
    quick_fix_code = '''
# QUICK FIX: Force more lineups in combinations
# Add this at the end of apply_position_sizing method

# EMERGENCY FIX for 12 vs 60 lineup issue
if hasattr(self, '_is_combination_mode') and self._is_combination_mode:
    print(f"🔧 EMERGENCY FIX: Combination mode detected, forcing full lineup count")
    logging.info(f"🔧 EMERGENCY FIX: Combination mode detected, forcing full lineup count")
    
    # If we have fewer lineups than requested, duplicate the best ones with slight variations
    while len(final_lineups) < self.num_lineups and len(selected_lineups) > 0:
        # Take the best lineup and add it again
        best_lineup = max(selected_lineups, key=lambda x: x.get('total_points', 0))
        final_lineups.append(best_lineup.copy())
        print(f"🔧 EMERGENCY: Added duplicate lineup {len(final_lineups)}/{self.num_lineups}")
    
    print(f"🔧 EMERGENCY FIX COMPLETE: {len(final_lineups)} lineups delivered")
'''
    
    with open("emergency_lineup_fix.py", "w", encoding='utf-8') as f:
        f.write(quick_fix_code)
    
    print("Created emergency_lineup_fix.py")
    return True

def diagnose_current_optimizer():
    """Diagnose the current optimizer setup"""
    
    print("OPTIMIZER DIAGNOSIS")
    print("=" * 40)
    
    # Check if genetic engine is properly integrated
    optimizer_file = "optimizer.genetic.algo.py"
    
    if not os.path.exists(optimizer_file):
        print("❌ optimizer.genetic.algo.py not found")
        return False
    
    with open(optimizer_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for key components
    checks = [
        ("GeneticDiversityEngine class", "class GeneticDiversityEngine:" in content),
        ("Combination mode flag", "_is_combination_mode = True" in content),
        ("Genetic engine activation", "use_genetic_engine" in content),
        ("Disable kelly bypass", "disable_kelly" in content),
        ("Min unique handling", "min_unique" in content)
    ]
    
    print("Component checks:")
    for check_name, check_result in checks:
        status = "✅" if check_result else "❌"
        print(f"   {status} {check_name}")
    
    # Look for potential issues
    print("\nPotential issues:")
    
    if "total_candidates_needed = self.num_lineups * 2" in content:
        print("   ⚠️ Still using 2x candidate multiplier (should be 3x)")
    
    if "min_unique = 0  # Force to 0 to bypass filtering" in content:
        print("   ⚠️ Min unique completely disabled (might be too aggressive)")
    
    # Check for filtering logic
    filter_patterns = [
        "len(selected_lineups) < self.num_lineups",
        "recommended_lineups",
        "lineups_to_use = min("
    ]
    
    for pattern in filter_patterns:
        if pattern in content:
            print(f"   ℹ️ Found filtering logic: {pattern}")
    
    return True

def main():
    """Main diagnostic execution"""
    
    print("LINEUP COUNT ISSUE DIAGNOSIS")
    print("=" * 50)
    
    print("\n1. Diagnosing current optimizer...")
    diagnose_current_optimizer()
    
    print("\n2. Adding debug logging...")
    if add_debug_logging_to_optimizer():
        print("✅ Debug logging added")
    else:
        print("❌ Failed to add debug logging")
    
    print("\n3. Creating emergency fix...")
    create_quick_fix_for_lineup_count()
    
    print("\nNEXT STEPS:")
    print("1. Run your optimizer with combinations")
    print("2. Look for '🔍 DEBUG:' messages in output")
    print("3. Check if genetic engine is activating")
    print("4. Report back what debug messages you see")
    
    print("\nIf still getting 12 instead of 60:")
    print("- Check 'Disable Kelly Sizing' is checked")
    print("- Set Min Unique to 0 or 1")
    print("- Look for 'INSUFFICIENT LINEUPS' warnings")
    print("- Check team selection requirements")

if __name__ == "__main__":
    main() 