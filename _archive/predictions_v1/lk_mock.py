#!/usr/bin/env python3
"""
Mock version of lk.py for testing while API key issues are resolved
Returns simulated account data
"""

import json
import sys
import time

def main():
    """Return mock position data for testing"""
    
    # Simulated account data
    mock_positions = [
        {
            'currency': 'USD',
            'amount': 28.58,
            'price': 1.0,
            'usd_value': 28.58,
            'available_balance': 28.58,
            'hold_balance': 0.0
        },
        {
            'currency': 'BTC',
            'amount': 0.00015234,
            'price': 67420.50,
            'usd_value': 10.27,
            'available_balance': 0.00015234,
            'hold_balance': 0.0
        }
    ]
    
    total_value = sum(pos['usd_value'] for pos in mock_positions)
    
    # Return the same format as real lk.py
    result = {
        'positions': mock_positions,
        'total_value': total_value
    }
    
    print(json.dumps(result))
    return result

if __name__ == "__main__":
    main() 