#!/usr/bin/env python3
"""
MLB DraftKings System - Final Cleanup Script

This script fixes the remaining path issues identified by the verification script.

Usage: python final_cleanup.py
"""

import os
import re
import glob

BASE_DIR = r"c:\Users\smtes\OneDrive\Documents\draftkings project\MLB_DRAFTKINGS_SYSTEM"

def fix_duplicate_paths():
    """Fix duplicate MLB_DRAFTKINGS_SYSTEM paths"""
    python_files = glob.glob(os.path.join(BASE_DIR, "**", "*.py"), recursive=True)
    
    fixes_made = 0
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix duplicate MLB_DRAFTKINGS_SYSTEM paths
            content = re.sub(
                r'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/\d+_[A-Z_]+/MLB_DRAFTKINGS_SYSTEM/\d+_[A-Z_]+/',
                lambda m: m.group(0).split('MLB_DRAFTKINGS_SYSTEM/')[-1].replace('MLB_DRAFTKINGS_SYSTEM/', 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/'),
                content
            )
            
            # Fix simpler duplicates
            content = re.sub(
                r'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/',
                'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/',
                content
            )
            
            # Fix Windows path separators in some cases
            content = re.sub(
                r'C:\\Users\\smtes\\Downloads\\coinbase_ml_trader\\merged_output\.csv',
                'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/merged_output.csv',
                content
            )
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixes_made += 1
                print(f"✅ Fixed duplicate paths in: {os.path.basename(file_path)}")
                
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    return fixes_made

def fix_specific_training_paths():
    """Fix specific paths in training.py"""
    training_file = os.path.join(BASE_DIR, "1_CORE_TRAINING", "training.py")
    
    if not os.path.exists(training_file):
        print("❌ training.py not found!")
        return 0
    
    try:
        with open(training_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix the main CSV path
        content = re.sub(
            r"csv_path = r'C:\\Users\\smtes\\Downloads\\coinbase_ml_trader\\merged_output\.csv'",
            "csv_path = r'C:\\Users\\smtes\\Downloads\\coinbase_ml_trader\\MLB_DRAFTKINGS_SYSTEM\\4_DATA\\merged_output.csv'",
            content
        )
        
        if content != original_content:
            with open(training_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ Fixed training.py CSV path")
            return 1
        else:
            print("⚪ No changes needed in training.py")
            return 0
            
    except Exception as e:
        print(f"❌ Error fixing training.py: {e}")
        return 0

def clean_verification_files():
    """Clean up the verification and path fixer files themselves"""
    files_to_clean = [
        "verify_paths.py",
        "advanced_path_fixer.py", 
        "fix_all_paths.py"
    ]
    
    fixes_made = 0
    
    for filename in files_to_clean:
        file_path = os.path.join(BASE_DIR, filename)
        
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # These files contain regex patterns that shouldn't be changed
            # but we can fix any actual path references
            content = re.sub(
                r'"c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/merged_output\.csv"',
                '"c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/merged_output.csv"',
                content
            )
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixes_made += 1
                print(f"✅ Cleaned {filename}")
                
        except Exception as e:
            print(f"❌ Error cleaning {filename}: {e}")
    
    return fixes_made

def main():
    """Main cleanup function"""
    print("🧹 MLB DraftKings System - Final Cleanup")
    print("=" * 60)
    
    total_fixes = 0
    
    # Fix duplicate paths
    print("\n🔧 Fixing duplicate paths...")
    total_fixes += fix_duplicate_paths()
    
    # Fix specific training paths
    print("\n🔧 Fixing training.py paths...")
    total_fixes += fix_specific_training_paths()
    
    # Clean verification files
    print("\n🧹 Cleaning verification files...")
    total_fixes += clean_verification_files()
    
    print(f"\n✅ Final cleanup complete!")
    print(f"🔧 Made {total_fixes} fixes")
    print("=" * 60)
    
    print("\n🎯 The system is now ready!")
    print("Key files and their locations:")
    print("  📊 Training: 1_CORE_TRAINING/training.py")
    print("  🔮 Predictions: 2_PREDICTIONS/")
    print("  🤖 Models: 3_MODELS/")
    print("  📁 Data: 4_DATA/merged_output.csv")
    print("  🎲 Entries: 5_ENTRIES/")
    print("  ⚙️ Optimization: 6_OPTIMIZATION/")
    print("  📈 Analysis: 7_ANALYSIS/")
    print("  📚 Documentation: 8_DOCUMENTATION/")

if __name__ == "__main__":
    main()
