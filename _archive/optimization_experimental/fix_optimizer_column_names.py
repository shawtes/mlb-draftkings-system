#!/usr/bin/env python3
"""
Fix Optimizer Column Name References
===================================

This script fixes all references to old column names in the optimizer code.
"""

import re
import shutil

def fix_optimizer_column_names():
    """Fix all column name references in optimizer01.py"""
    
    print("🔧 FIXING OPTIMIZER COLUMN REFERENCES")
    print("=" * 50)
    
    # Create backup
    optimizer_file = "optimizer01.py"
    backup_file = "optimizer01_backup_column_fix.py"
    shutil.copy2(optimizer_file, backup_file)
    print(f"💾 Created backup: {backup_file}")
    
    # Read the file
    with open(optimizer_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count original occurrences
    original_pos_count = content.count("'Pos'")
    original_predicted_points_count = content.count("'Predicted_Points'")
    
    print(f"📊 Found {original_pos_count} references to 'Pos'")
    print(f"📊 Found {original_predicted_points_count} references to 'Predicted_Points'")
    
    # Replace column references
    replacements = [
        ("'Pos'", "'Position'"),
        ('"Pos"', '"Position"'),
        ("'Predicted_Points'", "'Predicted_DK_Points'"),
        ('"Predicted_Points"', '"Predicted_DK_Points"'),
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    # Count new occurrences
    new_pos_count = content.count("'Position'")
    new_predicted_points_count = content.count("'Predicted_DK_Points'")
    
    print(f"✅ Fixed {original_pos_count} 'Pos' → 'Position' references")
    print(f"✅ Fixed {original_predicted_points_count} 'Predicted_Points' → 'Predicted_DK_Points' references")
    
    # Write the fixed file
    with open(optimizer_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"💾 Updated {optimizer_file}")
    print(f"📊 Now has {new_pos_count} 'Position' references")
    print(f"📊 Now has {new_predicted_points_count} 'Predicted_DK_Points' references")
    
    print(f"\n🎯 OPTIMIZER COLUMN FIXING COMPLETE!")
    print("Now the optimizer should work with the fixed CSV files.")
    
    return True

if __name__ == "__main__":
    fix_optimizer_column_names() 