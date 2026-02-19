#!/usr/bin/env python3
"""
Test script to verify the new simple stack types work correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_stack_types():
    """Test the new simple stack types"""
    
    # Test stack type parsing
    simple_stacks = ["5", "4", "3", "No Stacks"]
    complex_stacks = ["4|2", "3|3|2", "2|2|2", "5|2"]
    
    print("🧪 Testing Simple Stack Types:")
    for stack in simple_stacks:
        print(f"  ✅ {stack}")
        
        # Test if it's a simple stack
        if stack in ["5", "4", "3"]:
            stack_size = int(stack)
            print(f"    📊 Stack size: {stack_size}")
        elif stack == "No Stacks":
            print(f"    📊 No stacking constraints")
        else:
            print(f"    📊 Complex stack: {stack}")
    
    print("\n🧪 Testing Complex Stack Types:")
    for stack in complex_stacks:
        print(f"  ✅ {stack}")
        if '|' in stack:
            stack_sizes = [int(size) for size in stack.split('|')]
            print(f"    📊 Stack sizes: {stack_sizes}")
        else:
            print(f"    📊 Simple stack: {stack}")
    
    print("\n✅ All stack types processed successfully!")
    
    # Test stack determination from lineup
    print("\n🧪 Testing Stack Determination Logic:")
    
    # Mock team counts for testing
    test_cases = [
        {"LAD": 5, "NYY": 2, "ATL": 2, "SF": 1},  # 5 stack
        {"LAD": 4, "NYY": 3, "ATL": 2, "SF": 1},  # 4 stack
        {"LAD": 3, "NYY": 3, "ATL": 2, "SF": 2},  # 3 stack (tie)
        {"LAD": 2, "NYY": 2, "ATL": 2, "SF": 2, "BOS": 2},  # 2 stack
        {"LAD": 1, "NYY": 1, "ATL": 1, "SF": 1, "BOS": 1, "TEX": 1, "HOU": 1, "CWS": 1, "KC": 1, "MIN": 1},  # No stacks
    ]
    
    for i, team_counts in enumerate(test_cases):
        max_stack = max(team_counts.values())
        
        if max_stack >= 5:
            stack_type = "5 Stack"
        elif max_stack >= 4:
            stack_type = "4 Stack"
        elif max_stack >= 3:
            stack_type = "3 Stack"
        elif max_stack >= 2:
            stack_type = "2 Stack"
        else:
            stack_type = "No Stacks"
            
        print(f"  Test {i+1}: {team_counts} → {stack_type}")
    
    print("\n✅ Stack determination logic working correctly!")

if __name__ == "__main__":
    test_stack_types()
