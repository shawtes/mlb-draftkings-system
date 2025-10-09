#!/usr/bin/env python3
"""
Simple wrapper to run the duplicate diagnosis
"""

import os
import sys
import subprocess

# Change to optimizer directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Run the diagnosis
if __name__ == "__main__":
    from diagnose_duplicates import diagnose_duplicates
    diagnose_duplicates() 