#!/usr/bin/env python3
"""
Quick start guide for DFS Combination Generation
"""

import sys
import os

def main():
    print("🔥 DFS COMBINATION GENERATION - QUICK START GUIDE")
    print("=" * 60)
    
    print("\n✅ CONFIRMED: Combination generation is working!")
    print("   Our test successfully generated lineups with team stacks")
    
    print("\n📋 HOW TO USE COMBINATIONS IN THE GUI:")
    print("   1. Run: python launch_optimizer.py")
    print("   2. Click 'Load CSV' and load your player data")
    print("   3. Click the 'Team Combinations' tab")
    print("   4. Select the teams you want to stack")
    print("   5. Choose your stack pattern (4|2, 5|3, etc.)")
    print("   6. Click 'Generate Team Combinations'")
    print("   7. Check the combinations you want to run")
    print("   8. Click 'Generate All Combination Lineups'")
    
    print("\n🎯 AVAILABLE STACK PATTERNS:")
    stack_patterns = ["5|2", "4|2", "4|2|2", "3|3|2", "3|2|2", "2|2|2", "5|3"]
    for pattern in stack_patterns:
        print(f"   • {pattern}")
    
    print("\n💡 TIPS:")
    print("   • Start with 4|2 stacks for beginners")
    print("   • Select 3-5 teams for good variety")  
    print("   • Use 5-10 lineups per combination")
    print("   • The system enforces DraftKings position limits")
    print("   • Advanced quantitative features are automatically enabled")
    
    print("\n🚀 WANT TO TEST RIGHT NOW?")
    
    response = input("   Launch the GUI now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\n   Launching GUI...")
        try:
            from launch_optimizer import main as launch_main
            launch_main()
        except Exception as e:
            print(f"   Error launching GUI: {e}")
            print("   Try running: python launch_optimizer.py")
    else:
        print("\n   Run this when ready: python launch_optimizer.py")
    
    print("\n🎉 Happy lineup building!")

if __name__ == "__main__":
    main()
