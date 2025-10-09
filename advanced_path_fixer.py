#!/usr/bin/env python3
"""
Advanced MLB DraftKings System - Path Fixer Script

This script performs comprehensive path fixing for all MLB-related files,
including Python scripts, configuration files, and documentation.

Features:
- Fixes hardcoded file paths
- Updates import statements
- Corrects relative path references
- Handles CSV data file paths
- Updates model save/load paths
- Fixes documentation references

Usage: python advanced_path_fixer.py
"""

import os
import re
import glob
import shutil
from pathlib import Path
import json

# Define the base directory
BASE_DIR = r"c:\Users\smtes\OneDrive\Documents\draftkings project\MLB_DRAFTKINGS_SYSTEM"
ROOT_DIR = r"C:\Users\smtes\Downloads\coinbase_ml_trader"

class MLBPathFixer:
    def __init__(self):
        self.changes_log = []
        self.error_log = []
        
    def log_change(self, file_path, old_path, new_path):
        """Log a path change"""
        self.changes_log.append({
            'file': file_path,
            'old_path': old_path,
            'new_path': new_path
        })
        
    def log_error(self, file_path, error):
        """Log an error"""
        self.error_log.append({
            'file': file_path,
            'error': str(error)
        })
    
    def get_new_path_for_file_type(self, filename, current_path=""):
        """Determine the correct new path based on file type and name"""
        filename_lower = filename.lower()
        
        # Model files
        if any(ext in filename_lower for ext in ['.pkl', '.joblib', '.h5', '.pt']):
            return "3_MODELS"
        
        # Data files  
        if filename_lower.endswith('.csv'):
            if 'prediction' in filename_lower:
                return "2_PREDICTIONS"
            elif 'entries' in filename_lower or 'lineup' in filename_lower:
                return "5_ENTRIES"
            elif 'feature' in filename_lower or 'importance' in filename_lower:
                return "7_ANALYSIS"
            else:
                return "4_DATA"
        
        # Image files
        if any(ext in filename_lower for ext in ['.png', '.jpg', '.jpeg', '.svg']):
            return "7_ANALYSIS"
        
        # Documentation
        if any(ext in filename_lower for ext in ['.md', '.txt', '.rst']):
            return "8_DOCUMENTATION"
        
        # Python scripts
        if filename_lower.endswith('.py'):
            if 'train' in filename_lower:
                return "1_CORE_TRAINING"
            elif 'predict' in filename_lower:
                return "2_PREDICTIONS"
            elif 'optim' in filename_lower:
                return "6_OPTIMIZATION"
            elif 'analysis' in filename_lower:
                return "7_ANALYSIS"
            else:
                return "1_CORE_TRAINING"  # Default for Python files
        
        # Default
        return "7_ANALYSIS"
    
    def fix_python_file_paths(self, file_path):
        """Fix paths in Python files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            changes_made = 0
            
            # Common path patterns in Python files
            path_patterns = [
                # Absolute paths to app folder
                (r"['\"]c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/([^'\"]*)['\"]",
                 lambda m: f"'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/{self.get_new_path_for_file_type(m.group(1))}/{m.group(1)}'"),
                
                # Merged output CSV
                (r"['\"]c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/merged_output\.csv['\"]",
                 "'4_DATA/merged_output.csv'"),
                
                # Generic coinbase_ml_trader paths
                (r"['\"]c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/([^'\"]*\.csv)['\"]",
                 lambda m: f"'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/{m.group(1)}'"),
                
                # Joblib.dump and joblib.load paths
                (r"joblib\.(dump|load)\([^,]+,\s*['\"]([^'\"]*)['\"]",
                 lambda m: f"joblib.{m.group(1)}({m.group(0).split(',')[0]}, '{self.fix_single_path(m.group(2))}'"),
                
                # pd.read_csv paths
                (r"pd\.read_csv\(['\"]([^'\"]*)['\"]",
                 lambda m: f"pd.read_csv('c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/{self.fix_single_path(m.group(1))}'"),
                
                # to_csv paths
                (r"\.to_csv\(['\"]([^'\"]*)['\"]",
                 lambda m: f".to_csv('c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/{self.fix_single_path(m.group(1))}'"),
            ]
            
            # Apply pattern replacements
            for pattern, replacement in path_patterns:
                if callable(replacement):
                    def replace_func(match):
                        try:
                            return replacement(match)
                        except Exception as e:
                            self.log_error(file_path, f"Pattern replacement error: {e}")
                            return match.group(0)
                    
                    new_content = re.sub(pattern, replace_func, content)
                else:
                    new_content = re.sub(pattern, replacement, content)
                
                if new_content != content:
                    changes_made += 1
                    content = new_content
            
            # Save if changes were made
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return changes_made
            
            return 0
            
        except Exception as e:
            self.log_error(file_path, e)
            return 0
    
    def fix_single_path(self, path):
        """Fix a single path string"""
        # Remove any existing MLB_DRAFTKINGS_SYSTEM duplication
        if "MLB_DRAFTKINGS_SYSTEM" in path:
            return path
        
        # Extract filename
        filename = os.path.basename(path)
        
        # Determine new location
        new_folder = self.get_new_path_for_file_type(filename)
        
        # Return new path
        return f"c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/{new_folder}/{filename}"
    
    def create_directory_structure(self):
        """Create the complete directory structure"""
        directories = {
            "1_CORE_TRAINING": "Core ML training scripts",
            "2_PREDICTIONS": "Prediction outputs and scripts", 
            "3_MODELS": "Trained models and encoders",
            "4_DATA": "Raw and processed data files",
            "5_ENTRIES": "DraftKings entry files",
            "6_OPTIMIZATION": "Optimization and tuning scripts",
            "7_ANALYSIS": "Analysis results and visualizations",
            "8_DOCUMENTATION": "Documentation and guides",
            "9_BACKUP": "Backup files and archives"
        }
        
        for directory, description in directories.items():
            dir_path = os.path.join(BASE_DIR, directory)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"📁 Created {directory}: {description}")
    
    def move_root_files(self):
        """Move files from root coinbase_ml_trader to appropriate folders"""
        files_to_move = [
            ("merged_output.csv", "4_DATA/merged_output.csv"),
            ("batters_probability_predictions_20250705.csv", "2_PREDICTIONS/batters_probability_predictions_20250705.csv"),
        ]
        
        for source_file, target_path in files_to_move:
            source_path = os.path.join(ROOT_DIR, source_file)
            target_full_path = os.path.join(BASE_DIR, target_path)
            
            if os.path.exists(source_path):
                # Create target directory if needed
                os.makedirs(os.path.dirname(target_full_path), exist_ok=True)
                
                # Move file
                try:
                    shutil.move(source_path, target_full_path)
                    print(f"📦 Moved {source_file} to {target_path}")
                except Exception as e:
                    self.log_error(source_path, f"Failed to move file: {e}")
    
    def process_all_files(self):
        """Process all files in the system"""
        print("\n🔧 Processing all files...")
        
        # Get all Python files
        python_files = glob.glob(os.path.join(BASE_DIR, "**", "*.py"), recursive=True)
        
        total_changes = 0
        processed_files = 0
        
        for file_path in python_files:
            try:
                changes = self.fix_python_file_paths(file_path)
                if changes > 0:
                    total_changes += changes
                    print(f"✅ Updated {changes} paths in: {os.path.basename(file_path)}")
                else:
                    print(f"⚪ No changes needed: {os.path.basename(file_path)}")
                processed_files += 1
                
            except Exception as e:
                self.log_error(file_path, e)
                print(f"❌ Error processing: {os.path.basename(file_path)}")
        
        return total_changes, processed_files
    
    def generate_report(self):
        """Generate a report of all changes made"""
        report = {
            "summary": {
                "total_changes": len(self.changes_log),
                "total_errors": len(self.error_log),
                "processed_files": len(set(change['file'] for change in self.changes_log))
            },
            "changes": self.changes_log,
            "errors": self.error_log
        }
        
        report_path = os.path.join(BASE_DIR, "8_DOCUMENTATION", "path_fixing_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Report saved to: {report_path}")
        return report
    
    def run(self):
        """Run the complete path fixing process"""
        print("🚀 Advanced MLB DraftKings System - Path Fixer")
        print("=" * 60)
        
        # Create directory structure
        print("\n📁 Creating directory structure...")
        self.create_directory_structure()
        
        # Move root files
        print("\n📦 Moving misplaced files...")
        self.move_root_files()
        
        # Process all files
        total_changes, processed_files = self.process_all_files()
        
        # Generate report
        report = self.generate_report()
        
        # Final summary
        print("\n" + "=" * 60)
        print("✅ Path fixing complete!")
        print(f"📊 Processed {processed_files} files")
        print(f"🔧 Made {total_changes} total changes")
        print(f"❌ Encountered {len(self.error_log)} errors")
        print("=" * 60)
        
        # Show directory structure
        print("\n📂 Final Directory Structure:")
        for root, dirs, files in os.walk(BASE_DIR):
            level = root.replace(BASE_DIR, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:3]:  # Show first 3 files per directory
                print(f"{subindent}{file}")
            if len(files) > 3:
                print(f"{subindent}... and {len(files) - 3} more files")

def main():
    """Main function"""
    fixer = MLBPathFixer()
    fixer.run()

if __name__ == "__main__":
    main()
