#!/usr/bin/env python3
"""
Final Path Cleanup Script for MLB DraftKings System
This script handles remaining path issues and mixed paths
"""

import os
import glob

# Define the current base directory
BASE_DIR = r"c:\Users\smtes\OneDrive\Documents\draftkings project\MLB_DRAFTKINGS_SYSTEM"

# Define old and new base paths
OLD_BASE_PATH = "C:/Users/smtes/Downloads/coinbase_ml_trader"
NEW_BASE_PATH = "c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM"

# File extensions to process
FILE_EXTENSIONS = ['.py', '.md', '.txt', '.json']

def get_all_files():
    """Get all files in the MLB_DRAFTKINGS_SYSTEM directory"""
    files = []
    for ext in FILE_EXTENSIONS:
        pattern = os.path.join(BASE_DIR, "**", f"*{ext}")
        files.extend(glob.glob(pattern, recursive=True))
    return files

def fix_mixed_paths(file_path):
    """Fix mixed and malformed paths"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        original_content = content
        
        # Fix specific issues found
        fixes = [
            # Fix double paths
            (f"c:/Users/smtes/Downloads/coinbase_ml_trader/{NEW_BASE_PATH}/4_DATA/", 
             f"{NEW_BASE_PATH}/4_DATA/"),
            
            # Fix remaining old paths
            (f"{OLD_BASE_PATH}/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/", 
             f"{NEW_BASE_PATH}/2_PREDICTIONS/"),
            
            # Fix sys.path.append lines
            (f"sys.path.append('{OLD_BASE_PATH}/MLB_DRAFTKINGS_SYSTEM/2_PREDICTIONS')", 
             f"sys.path.append('{NEW_BASE_PATH}/2_PREDICTIONS')"),
            
            # Fix specific file references
            ("batters_predictions_20250713_fixed.csv", 
             "batters_predictions_20250713.csv"),
            
            # Fix any remaining old base paths
            (f"{OLD_BASE_PATH}/", 
             f"{NEW_BASE_PATH}/4_DATA/"),
            
        ]
        
        # Apply fixes
        changes_made = 0
        for old_str, new_str in fixes:
            if old_str in content:
                content = content.replace(old_str, new_str)
                changes_made += 1
        
        # Write back if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return changes_made
        else:
            return 0
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return 0

def main():
    """Main function to process all files"""
    print("Starting final path cleanup...")
    print(f"Base directory: {BASE_DIR}")
    print("=" * 60)
    
    # Get all files to process
    all_files = get_all_files()
    print(f"Found {len(all_files)} files to process")
    
    total_changes = 0
    processed_files = 0
    
    for file_path in all_files:
        try:
            # Skip this script itself
            if file_path.endswith('final_path_cleanup.py'):
                continue
                
            changes = fix_mixed_paths(file_path)
            if changes > 0:
                processed_files += 1
                total_changes += changes
                print(f"✓ Fixed {changes} paths in: {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"✗ Error processing {file_path}: {str(e)}")
    
    print("=" * 60)
    print(f"Final cleanup complete!")
    print(f"Files processed: {processed_files}")
    print(f"Total path fixes: {total_changes}")
    
    if total_changes > 0:
        print("\nFinal path cleanup completed successfully!")
        print("All remaining path issues have been resolved.")
    else:
        print("\nNo additional path fixes were needed.")

if __name__ == "__main__":
    main()
