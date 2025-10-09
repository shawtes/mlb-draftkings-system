#!/usr/bin/env python3
"""
MLB DraftKings System - Path Fixer Script

This script updates all file paths in the MLB_DRAFTKINGS_SYSTEM to use the correct
organized directory structure. It will scan through all Python files and update
any hardcoded paths to match the new folder organization.

Directory Structure:
- MLB_DRAFTKINGS_SYSTEM/
  - 1_CORE_TRAINING/
  - 2_PREDICTIONS/
  - 3_MODELS/
  - 4_DATA/
  - 5_ENTRIES/
  - 6_OPTIMIZATION/
  - 7_ANALYSIS/
  - 8_DOCUMENTATION/
  - 9_BACKUP/

Usage: python fix_all_paths.py
"""

import os
import re
import glob
from pathlib import Path

# Define the base directory
BASE_DIR = r"c:\Users\smtes\OneDrive\Documents\draftkings project\MLB_DRAFTKINGS_SYSTEM"

# Define path mappings from old to new structure
PATH_MAPPINGS = {
    # Old app folder paths -> New organized paths
    r'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/': {
        "models": "MLB_DRAFTKINGS_SYSTEM/3_MODELS/",
        "data": "MLB_DRAFTKINGS_SYSTEM/4_DATA/",
        "predictions": "MLB_DRAFTKINGS_SYSTEM/2_PREDICTIONS/",
        "entries": "MLB_DRAFTKINGS_SYSTEM/5_ENTRIES/",
        "analysis": "MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/",
        "default": "MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/"
    },
    
    # Root coinbase_ml_trader paths
    r"c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/": {
        "merged_output.csv": "MLB_DRAFTKINGS_SYSTEM/4_DATA/merged_output.csv",
        "default": "MLB_DRAFTKINGS_SYSTEM/4_DATA/"
    }
}

# File extensions to process
FILE_EXTENSIONS = ['.py', '.md', '.txt', '.json', '.csv']

def get_all_files():
    """Get all files in the MLB_DRAFTKINGS_SYSTEM directory"""
    files = []
    for ext in FILE_EXTENSIONS:
        pattern = os.path.join(BASE_DIR, "**", f"*{ext}")
        files.extend(glob.glob(pattern, recursive=True))
    return files

def update_file_paths(file_path):
    """Update file paths in a given file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = 0
        
        # Common path patterns to fix
        path_replacements = [
            # Model files
            (r'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/([^'\"]*\.pkl)", 
             r"c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/\1"),
            
            (r'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/([^'\"]*\.joblib)", 
             r"c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/\1"),
            
            # Data files
            (r'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/([^'\"]*\.csv)", 
             r"c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/\1"),
            
            # Analysis files
            (r'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/([^'\"]*\.png)", 
             r"c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/\1"),
            
            # Main data file
            (r'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/merged_output\.csv', 
             r'4_DATA/merged_output.csv'),
            
            # Prediction files
            (r'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/(.*prediction.*\.csv)', 
             r"c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/2_PREDICTIONS/\1"),
            
            # Entry files
            (r'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/(.*entries.*\.csv)', 
             r"c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/5_ENTRIES/\1"),
            
            # Feature importance files
            (r'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/(.*feature.*\.csv)', 
             r"c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/\1"),
            
            # Generic app folder references
            (r'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/([^'\"]*)", 
             r"c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/\1"),
        ]
        
        # Apply replacements
        for old_pattern, new_pattern in path_replacements:
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_pattern, content)
                changes_made += 1
        
        # Specific file name mappings
        specific_mappings = {
            # Model files
            "batters_final_ensemble_model_pipeline01.pkl": "3_MODELS/batters_final_ensemble_model_pipeline01.pkl",
            "probability_predictor01.pkl": "3_MODELS/probability_predictor01.pkl",
            "trained_model.joblib": "3_MODELS/trained_model.joblib",
            "label_encoder_name_sep2.pkl": "3_MODELS/label_encoder_name_sep2.pkl",
            "label_encoder_team_sep2.pkl": "3_MODELS/label_encoder_team_sep2.pkl",
            "scaler_sep2.pkl": "3_MODELS/scaler_sep2.pkl",
            
            # Data files
            "battersfinal_dataset_with_features.csv": "4_DATA/battersfinal_dataset_with_features.csv",
            "merged_output.csv": "4_DATA/merged_output.csv",
            
            # Prediction files
            "final_predictions.csv": "2_PREDICTIONS/final_predictions.csv",
            "probability_predictions.csv": "2_PREDICTIONS/probability_predictions.csv",
            
            # Analysis files
            "feature_importances.csv": "7_ANALYSIS/feature_importances.csv",
            "feature_importances_plot.png": "7_ANALYSIS/feature_importances_plot.png",
        }
        
        # Apply specific mappings
        for old_file, new_path in specific_mappings.items():
            if old_file in content:
                # Replace the full path
                full_new_path = f"c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/{new_path}"
                content = content.replace(old_file, full_new_path)
                changes_made += 1
        
        # Save the updated content if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated {changes_made} paths in: {os.path.basename(file_path)}")
            return True
        else:
            print(f"⚪ No changes needed in: {os.path.basename(file_path)}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        return False

def create_missing_directories():
    """Create any missing directories in the structure"""
    directories = [
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
    
    for directory in directories:
        dir_path = os.path.join(BASE_DIR, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"📁 Created directory: {directory}")

def move_misplaced_files():
    """Move any files that are in the wrong location"""
    # Check for common misplaced files
    misplaced_files = {
        "merged_output.csv": "4_DATA/merged_output.csv"
    }
    
    root_dir = r"C:\Users\smtes\Downloads\coinbase_ml_trader"
    
    for file_name, target_path in misplaced_files.items():
        source_path = os.path.join(root_dir, file_name)
        target_full_path = os.path.join(BASE_DIR, target_path)
        
        if os.path.exists(source_path) and not os.path.exists(target_full_path):
            # Create target directory if it doesn't exist
            os.makedirs(os.path.dirname(target_full_path), exist_ok=True)
            
            # Move the file
            import shutil
            shutil.move(source_path, target_full_path)
            print(f"📦 Moved {file_name} to {target_path}")

def main():
    """Main function to fix all paths"""
    print("🚀 MLB DraftKings System - Path Fixer")
    print("=" * 60)
    
    # Create missing directories
    print("\n📁 Creating missing directories...")
    create_missing_directories()
    
    # Move misplaced files
    print("\n📦 Moving misplaced files...")
    move_misplaced_files()
    
    # Get all files to process
    print("\n🔍 Scanning for files to update...")
    files = get_all_files()
    print(f"Found {len(files)} files to process")
    
    # Update file paths
    print("\n🔧 Updating file paths...")
    updated_files = 0
    
    for file_path in files:
        if update_file_paths(file_path):
            updated_files += 1
    
    print(f"\n✅ Path fixing complete!")
    print(f"📊 Updated {updated_files} out of {len(files)} files")
    print("=" * 60)
    
    # Show final directory structure
    print("\n📂 Final Directory Structure:")
    for root, dirs, files in os.walk(BASE_DIR):
        level = root.replace(BASE_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files per directory
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")

if __name__ == "__main__":
    main()
