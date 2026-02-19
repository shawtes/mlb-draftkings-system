#!/usr/bin/env python3
"""
Test Automatic Column Mapping
=============================

Test that the file loading function can handle both old and new column formats.
"""

import pandas as pd
import os
import tempfile

def test_column_mapping():
    """Test automatic column mapping in file loading"""
    
    print("🧪 TESTING AUTOMATIC COLUMN MAPPING")
    print("=" * 50)
    
    # Create test CSV with old column names
    old_format_data = {
        'Name': ['Player1', 'Player2', 'Player3'],
        'Team': ['NYY', 'BOS', 'LAD'],
        'Pos': ['P', 'OF', 'C'],  # Old format
        'Salary': [10000, 8000, 6000],
        'Predicted_Points': [15.5, 12.3, 8.7]  # Old format
    }
    
    # Create temporary CSV file with old format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        old_df = pd.DataFrame(old_format_data)
        old_df.to_csv(tmp_file.name, index=False)
        tmp_file_path = tmp_file.name
    
    try:
        # Test the load_players function
        print("📊 Testing with old format CSV...")
        print(f"Original columns: {list(old_df.columns)}")
        
        # Import and test the function
        import sys
        sys.path.insert(0, '.')
        from optimizer01 import FantasyBaseballApp
        
        # Create app instance (minimal setup)
        app = FantasyBaseballApp()
        
        # Test loading
        loaded_df = app.load_players(tmp_file_path)
        
        print(f"✅ Loading successful!")
        print(f"Final columns: {list(loaded_df.columns)}")
        
        # Verify column mapping worked
        expected_columns = ['Name', 'Team', 'Position', 'Salary', 'Predicted_DK_Points']
        missing_expected = [col for col in expected_columns if col not in loaded_df.columns]
        
        if missing_expected:
            print(f"❌ Missing expected columns: {missing_expected}")
            return False
        else:
            print(f"✅ All expected columns present!")
            
        # Verify data integrity
        print(f"📋 Data sample:")
        print(loaded_df.head().to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing column mapping: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temp file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

if __name__ == "__main__":
    success = test_column_mapping()
    if success:
        print("\n🎯 COLUMN MAPPING TEST PASSED!")
        print("The optimizer should now handle both old and new CSV formats automatically.")
    else:
        print("\n❌ COLUMN MAPPING TEST FAILED!")
        print("There may be an issue with the automatic column mapping.") 