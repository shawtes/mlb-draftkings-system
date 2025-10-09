#!/usr/bin/env python3
"""
MLB DraftKings System - Path Verification Script

This script verifies that all file paths have been correctly updated
in the MLB_DRAFTKINGS_SYSTEM and identifies any remaining issues.

Usage: python verify_paths.py
"""

import os
import re
import glob
from pathlib import Path

BASE_DIR = r"c:\Users\smtes\OneDrive\Documents\draftkings project\MLB_DRAFTKINGS_SYSTEM"

def find_old_paths():
    """Find any remaining old paths that need to be fixed"""
    old_path_patterns = [
        r"c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/",
        r"c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/merged_output\.csv",
        r"c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/[^/]*\.csv",
        r"c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/[^/]*\.pkl",
        r"c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/[^/]*\.joblib",
    ]
    
    issues = []
    
    # Get all Python files
    python_files = glob.glob(os.path.join(BASE_DIR, "**", "*.py"), recursive=True)
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for line_num, line in enumerate(content.split('\n'), 1):
                for pattern in old_path_patterns:
                    if re.search(pattern, line):
                        # Skip if it's already in MLB_DRAFTKINGS_SYSTEM
                        if "MLB_DRAFTKINGS_SYSTEM" in line:
                            continue
                        
                        issues.append({
                            'file': os.path.relpath(file_path, BASE_DIR),
                            'line': line_num,
                            'content': line.strip(),
                            'pattern': pattern
                        })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return issues

def verify_file_existence():
    """Verify that referenced files actually exist"""
    missing_files = []
    
    python_files = glob.glob(os.path.join(BASE_DIR, "**", "*.py"), recursive=True)
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find file path references
            path_matches = re.findall(r"['\"]([^'\"]*\.(?:csv|pkl|joblib|png|jpg))['\"]", content)
            
            for path_match in path_matches:
                if os.path.isabs(path_match):
                    if not os.path.exists(path_match):
                        missing_files.append({
                            'file': os.path.relpath(file_path, BASE_DIR),
                            'missing_path': path_match
                        })
        except Exception as e:
            print(f"Error checking {file_path}: {e}")
    
    return missing_files

def check_directory_structure():
    """Check if all expected directories exist"""
    expected_dirs = [
        "1_CORE_TRAINING",
        "2_PREDICTIONS", 
        "3_MODELS",
        "4_DATA",
        "5_ENTRIES",
        "6_OPTIMIZATION",
        "7_ANALYSIS",
        "8_DOCUMENTATION",
        "9_BACKUP"
    ]
    
    missing_dirs = []
    for dir_name in expected_dirs:
        dir_path = os.path.join(BASE_DIR, dir_name)
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_name)
    
    return missing_dirs

def main():
    """Main verification function"""
    print("🔍 MLB DraftKings System - Path Verification")
    print("=" * 60)
    
    # Check for old paths
    print("\n🔍 Checking for remaining old paths...")
    old_paths = find_old_paths()
    
    if old_paths:
        print(f"❌ Found {len(old_paths)} remaining old paths:")
        for issue in old_paths[:10]:  # Show first 10
            print(f"  📁 {issue['file']} (line {issue['line']})")
            print(f"     {issue['content']}")
        if len(old_paths) > 10:
            print(f"     ... and {len(old_paths) - 10} more issues")
    else:
        print("✅ No old paths found!")
    
    # Check for missing files
    print("\n🔍 Checking for missing files...")
    missing_files = verify_file_existence()
    
    if missing_files:
        print(f"❌ Found {len(missing_files)} missing file references:")
        for issue in missing_files[:10]:
            print(f"  📁 {issue['file']}")
            print(f"     Missing: {issue['missing_path']}")
        if len(missing_files) > 10:
            print(f"     ... and {len(missing_files) - 10} more missing files")
    else:
        print("✅ All referenced files exist!")
    
    # Check directory structure
    print("\n🔍 Checking directory structure...")
    missing_dirs = check_directory_structure()
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
    else:
        print("✅ All directories exist!")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Verification Summary:")
    print(f"  Old paths found: {len(old_paths)}")
    print(f"  Missing files: {len(missing_files)}")
    print(f"  Missing directories: {len(missing_dirs)}")
    
    if old_paths or missing_files or missing_dirs:
        print("⚠️  Some issues found - review above for details")
    else:
        print("✅ All paths verified successfully!")
    
    print("=" * 60)
    
    # Show corrected structure
    print("\n📂 Current MLB_DRAFTKINGS_SYSTEM Structure:")
    for root, dirs, files in os.walk(BASE_DIR):
        level = root.replace(BASE_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")

if __name__ == "__main__":
    main()
