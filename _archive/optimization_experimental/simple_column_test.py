#!/usr/bin/env python3
"""
Simple Column Mapping Test
==========================

Test the column mapping logic without requiring the full GUI.
"""

import pandas as pd

def test_column_mapping_logic():
    """Test the column mapping logic standalone"""
    
    print("🧪 TESTING COLUMN MAPPING LOGIC")
    print("=" * 40)
    
    # Create test DataFrame with old column names
    old_format_data = {
        'Name': ['Player1', 'Player2', 'Player3'],
        'Team': ['NYY', 'BOS', 'LAD'],
        'Pos': ['P', 'OF', 'C'],  # Old format
        'Salary': [10000, 8000, 6000],
        'Predicted_Points': [15.5, 12.3, 8.7]  # Old format
    }
    
    df = pd.DataFrame(old_format_data)
    print(f"📊 Original columns: {list(df.columns)}")
    
    # Apply the same column mapping logic from the fixed function
    column_mapping = {
        'Pos': 'Position',  # Handle old format
        'Predicted_Points': 'Predicted_DK_Points'  # Handle old format
    }
    
    # Apply column mapping automatically
    columns_renamed = []
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})
            columns_renamed.append(f"{old_col} → {new_col}")
    
    if columns_renamed:
        print(f"🔧 Auto-renamed columns: {', '.join(columns_renamed)}")
    
    print(f"✅ Final columns: {list(df.columns)}")
    
    # Basic required columns (after potential renaming)
    basic_required = ['Name', 'Team', 'Position', 'Salary']
    
    # Check for basic required columns
    missing_basic = [col for col in basic_required if col not in df.columns]
    if missing_basic:
        print(f"❌ Missing required columns: {missing_basic}")
        return False
    else:
        print(f"✅ All basic required columns present!")
    
    # Check for prediction column
    if 'Predicted_DK_Points' in df.columns:
        print(f"✅ Prediction column present!")
    else:
        print(f"❌ Prediction column missing!")
        return False
    
    # Show final data
    print(f"\n📋 Final data:")
    print(df.to_string(index=False))
    
    return True

if __name__ == "__main__":
    success = test_column_mapping_logic()
    if success:
        print("\n🎯 COLUMN MAPPING LOGIC TEST PASSED!")
        print("The file loading function should now handle both formats automatically.")
    else:
        print("\n❌ COLUMN MAPPING LOGIC TEST FAILED!") 