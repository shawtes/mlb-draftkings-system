#!/usr/bin/env python3
"""
Simple Path Refactor Script for MLB DraftKings System

This script will update all file paths in the MLB_DRAFTKINGS_SYSTEM to use the correct
current directory structure using simple string replacements.
"""

import os
import glob
from pathlib import Path

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

def update_file_paths(file_path):
    """Update file paths in a given file using simple string replacements"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        original_content = content
        
        # Simple string replacements
        replacements = [
            # 1. Complete old base path replacements
            (f"{OLD_BASE_PATH}/MLB_DRAFTKINGS_SYSTEM/", f"{NEW_BASE_PATH}/"),
            (f"{OLD_BASE_PATH}\\MLB_DRAFTKINGS_SYSTEM\\", f"{NEW_BASE_PATH}\\"),
            
            # 2. Old app folder paths to appropriate new folders
            (f"{OLD_BASE_PATH}/app/", f"{NEW_BASE_PATH}/4_DATA/"),
            (f"{OLD_BASE_PATH}\\app\\", f"{NEW_BASE_PATH}\\4_DATA\\"),
            
            # 3. Simple app/ references
            ("app/", f"{NEW_BASE_PATH}/4_DATA/"),
            ("app\\", f"{NEW_BASE_PATH}\\4_DATA\\"),
            
            # 4. Root coinbase_ml_trader references
            (f"{OLD_BASE_PATH}/", f"{NEW_BASE_PATH}/4_DATA/"),
            (f"{OLD_BASE_PATH}\\", f"{NEW_BASE_PATH}\\4_DATA\\"),
            
            # 5. Update BASE_DIR references in other scripts
            ('BASE_DIR = r"C:\\Users\\smtes\\Downloads\\coinbase_ml_trader\\MLB_DRAFTKINGS_SYSTEM"', 
             f'BASE_DIR = r"{BASE_DIR}"'),
            
        ]
        
        # Apply replacements
        changes_made = 0
        for old_str, new_str in replacements:
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
    print("Starting simple path refactoring...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Old base path: {OLD_BASE_PATH}")
    print(f"New base path: {NEW_BASE_PATH}")
    print("=" * 60)
    
    # Get all files to process
    all_files = get_all_files()
    print(f"Found {len(all_files)} files to process")
    
    total_changes = 0
    processed_files = 0
    
    for file_path in all_files:
        try:
            # Skip this script itself
            if file_path.endswith('simple_path_refactor.py'):
                continue
                
            changes = update_file_paths(file_path)
            if changes > 0:
                processed_files += 1
                total_changes += changes
                print(f"✓ Updated {changes} paths in: {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"✗ Error processing {file_path}: {str(e)}")
    
    print("=" * 60)
    print(f"Refactoring complete!")
    print(f"Files processed: {processed_files}")
    print(f"Total path changes: {total_changes}")
    
    if total_changes > 0:
        print("\nPath refactoring completed successfully!")
        print("All file paths have been updated to use the new directory structure.")
    else:
        print("\nNo path changes were needed.")

if __name__ == "__main__":
    main()
