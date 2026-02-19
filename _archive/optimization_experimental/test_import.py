#!/usr/bin/env python3
"""Test imports for the optimizer"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    from safe_logging import safe_log_info, safe_log_debug, safe_log_warning, safe_log_error
    print("✅ Safe logging imported successfully")
except ImportError as e:
    print(f"⚠️ Safe logging not available: {e}")

try:
    from advanced_quant_optimizer import AdvancedQuantitativeOptimizer
    print("✅ Advanced Quantitative Optimizer imported successfully")
except ImportError as e:
    print(f"⚠️ Advanced Quantitative Optimizer not available: {e}")

try:
    from checkbox_fix import CheckboxManager
    print("✅ Enhanced Checkbox Manager imported successfully")
except ImportError as e:
    print(f"⚠️ Enhanced Checkbox Manager not available: {e}")

try:
    from dfs_risk_engine import DFSRiskEngine, DFSBankrollManager, RiskMetrics
    print("✅ Advanced Risk Engine imported successfully")
except ImportError as e:
    print(f"⚠️ Advanced Risk Engine not available: {e}")

try:
    from optimizer01 import FantasyBaseballApp
    print("✅ Main optimizer imported successfully")
except ImportError as e:
    print(f"❌ Main optimizer failed: {e}")

print("Import testing complete!")
