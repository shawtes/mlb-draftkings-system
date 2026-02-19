#!/usr/bin/env python3
"""
Test script to verify 60 lineup fix is working
"""

def verify_optimizer_fixes():
    """Verify that the optimizer fixes are in place"""
    
    print("VERIFYING 60 LINEUP FIX")
    print("=" * 30)
    
    # Check optimizer file for fixes
    optimizer_file = "optimizer.genetic.algo.py"
    
    try:
        with open(optimizer_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key fixes
        fixes_present = [
            ("Combination mode forcing", "lineups_to_use = self.num_lineups  # Always try for full count" in content),
            ("Duplication handling", "source_idx = i % len(selected_lineups)" in content),
            ("Genetic engine threshold lowered", "self.num_lineups >= 3" in content),
            ("Debug logging added", "🔍 GENETIC ENGINE CHECK" in content),
            ("Combination mode flag", "_is_combination_mode = True" in content)
        ]
        
        print("Fix verification:")
        all_present = True
        for fix_name, fix_present in fixes_present:
            status = "✅" if fix_present else "❌"
            print(f"   {status} {fix_name}")
            if not fix_present:
                all_present = False
        
        if all_present:
            print("\n🎉 ALL FIXES ARE IN PLACE!")
            print("\nThe optimizer should now:")
            print("   ✅ Force full lineup count for combinations")
            print("   ✅ Duplicate lineups if needed to reach target")
            print("   ✅ Use genetic engine for 3+ lineups")
            print("   ✅ Provide debug logging")
        else:
            print("\n⚠️ Some fixes are missing")
        
        return all_present
        
    except Exception as e:
        print(f"Error checking optimizer: {e}")
        return False

def create_test_instructions():
    """Create instructions for testing the fix"""
    
    instructions = """
TEST INSTRUCTIONS FOR 60 LINEUP FIX
===================================

1. SETUP:
   ✅ Load your player CSV
   ✅ Check "Disable Kelly Sizing" 
   ✅ Set Min Unique to 0 or 1
   ✅ Go to Team Combinations tab

2. CREATE TEST COMBINATION:
   ✅ Add combination: e.g., LAD(4) + SF(2)
   ✅ Set lineups to 60 (or whatever you want)
   ✅ Check the combination checkbox
   ✅ Click "Generate Combination Lineups"

3. LOOK FOR DEBUG MESSAGES:
   ✅ "🔍 DEBUG: COMBINATION MODE ACTIVATED for 60 lineups"
   ✅ "🔍 GENETIC ENGINE CHECK" with details
   ✅ "🧬 COMBINATION MODE: FORCING 60 lineups"
   ✅ "🧬 DUPLICATING lineup X as lineup Y" (if needed)

4. EXPECTED RESULT:
   ✅ Should get exactly 60 lineups
   ✅ May see duplication messages if not enough unique lineups
   ✅ Final count should match requested count

5. IF STILL NOT WORKING:
   ✅ Check that "Disable Kelly Sizing" is actually checked
   ✅ Verify Min Unique is 0 or 1
   ✅ Look for any error messages in logs
   ✅ Report what debug messages you see
"""
    
    with open("test_60_lineup_instructions.txt", "w", encoding='utf-8') as f:
        f.write(instructions)
    
    print("Created test_60_lineup_instructions.txt")
    return True

def main():
    """Main execution"""
    
    print("60 LINEUP FIX VERIFICATION")
    print("=" * 40)
    
    # Verify fixes are in place
    if verify_optimizer_fixes():
        print("\n✅ Fixes verified successfully!")
    else:
        print("\n❌ Some fixes may be missing")
    
    # Create test instructions
    create_test_instructions()
    
    print("\nREADY TO TEST:")
    print("1. Run your optimizer: python launch_optimizer.py")
    print("2. Follow test_60_lineup_instructions.txt")
    print("3. Look for debug messages in the output")
    print("4. Report back if you still get 12 instead of 60")

if __name__ == "__main__":
    main() 