#!/usr/bin/env python3
"""
Fix CSV Column Names for DFS Optimizer
=====================================

This script fixes the column name mismatches that prevent the optimizer from working.
"""

import pandas as pd
import os
import shutil

def fix_csv_columns():
    """Fix column name mismatches in CSV files"""
    
    print("🔧 FIXING CSV COLUMN NAMES")
    print("=" * 40)
    
    # Find CSV files
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ No CSV files found!")
        return False
    
    # Column mapping
    column_mapping = {
        'Pos': 'Position',
        'Predicted_Points': 'Predicted_DK_Points'
    }
    
    for csv_file in csv_files:
        print(f"\n📊 Processing: {csv_file}")
        
        try:
            # Read CSV
            df = pd.read_csv(csv_file)
            original_columns = df.columns.tolist()
            
            # Check if we need to rename columns
            needs_fix = any(old_col in df.columns for old_col in column_mapping.keys())
            
            if needs_fix:
                # Create backup
                backup_file = csv_file.replace('.csv', '_backup.csv')
                shutil.copy2(csv_file, backup_file)
                print(f"   💾 Created backup: {backup_file}")
                
                # Rename columns
                df = df.rename(columns=column_mapping)
                new_columns = df.columns.tolist()
                
                # Save fixed CSV
                df.to_csv(csv_file, index=False)
                
                print(f"   ✅ Fixed columns:")
                for old_col, new_col in column_mapping.items():
                    if old_col in original_columns:
                        print(f"      {old_col} → {new_col}")
                
                print(f"   📋 New columns: {new_columns}")
                
            else:
                print(f"   ✅ No column fixes needed")
                
        except Exception as e:
            print(f"   ❌ Error processing {csv_file}: {e}")
    
    print(f"\n🎯 COLUMN FIXING COMPLETE!")
    print("Now try running the optimizer again.")
    
    return True

if __name__ == "__main__":
    fix_csv_columns() 