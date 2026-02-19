#!/usr/bin/env python3
"""
Quick launcher for RL Parlay System
"""

import sys
import os
import subprocess

def main():
    """Launch RL Parlay System"""
    rl_dir = os.path.join(os.path.dirname(__file__), "rl_parlay_system")
    
    if not os.path.exists(rl_dir):
        print("❌ RL Parlay System not found!")
        print("   Make sure you're in the correct directory")
        return
    
    # Change to RL directory
    os.chdir(rl_dir)
    
    # Run main script
    subprocess.run([sys.executable, "main.py"] + sys.argv[1:])

if __name__ == "__main__":
    main()










