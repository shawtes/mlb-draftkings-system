#!/usr/bin/env python3
"""
Improved Coinbase Account Balance Fetcher
Incorporates user feedback and production-ready features
"""

import os
import sys
import logging
import json
from typing import Dict, List, Optional, Union
from coinbase.rest import RESTClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def load_credentials():
    """Load API credentials with proper fallback and validation"""
    try:
        # Try to import from config.py first
        from config import (
            COINBASE_API_KEY as KEY_NAME,
            COINBASE_API_SECRET as PRIVATE_KEY_PEM,
            COINBASE_BASE_URL as BASE_URL,
            COINBASE_ORG_ID as ORG_ID
        )
        logger.info("✅ Loaded API credentials from config.py")
        return KEY_NAME, PRIVATE_KEY_PEM, BASE_URL, ORG_ID
        
    except ImportError:
        # Fallback to environment variables
        logger.info("🔄 config.py not found, checking environment variables...")
        
        key_name = os.getenv('COINBASE_API_KEY')
        private_key = os.getenv('COINBASE_API_SECRET') 
        base_url = os.getenv('COINBASE_BASE_URL', 'api.coinbase.com')
        org_id = os.getenv('COINBASE_ORG_ID', 'b98ec8e1-610f-451a-9324-40ae8e705d00')
        
        if key_name and private_key:
            logger.info("✅ Loaded API credentials from environment variables")
            return key_name, private_key, base_url, org_id
        
        # Final fallback to hardcoded (not recommended for production)
        logger.warning("⚠️ Using hardcoded credentials (not recommended for production)")
        KEY_NAME = 'organizations/b98ec8e1-610f-451a-9324-40ae8e705d00/apiKeys/0ea5fdb5-1c84-44b6-bd17-c61ae1bc90a8'
        PRIVATE_KEY_PEM = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIDF+Jmog53vB054/TKJq/ortq8Pi1WuxNXTvaENh/d63oAoGCCqGSM49
AwEHoUQDQgAEqqFBmtotFjIjYgtydVoBm8UjRPoY6gv54fKIIKTL3zs5OvlKLMdU
2NfCAihyGfi2OZG4/g1kfPnMj4EWvLjB1w==
-----END EC PRIVATE KEY-----"""
        BASE_URL = 'api.coinbase.com'
        ORG_ID = 'b98ec8e1-610f-451a-9324-40ae8e705d00'
        
        return KEY_NAME, PRIVATE_KEY_PEM, BASE_URL, ORG_ID

def parse_balance_value(balance_obj: Union[Dict, object]) -> float:
    """Parse balance value from either dict or object format"""
    if balance_obj is None:
        return 0.0
    
    try:
        if isinstance(balance_obj, dict):
            return float(balance_obj.get('value', '0'))
        else:
            # Object format
            return float(getattr(balance_obj, 'value', '0'))
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning(f"Could not parse balance value from {balance_obj}: {e}")
        return 0.0

def parse_account(account: Union[Dict, object]) -> Optional[Dict]:
    """Parse account data into standardized format"""
    try:
        # Handle both dict and object formats
        if isinstance(account, dict):
            currency = account.get('currency')
            available_balance = account.get('available_balance', {})
            hold = account.get('hold', {})
        else:
            # Object format
            currency = getattr(account, 'currency', None)
            available_balance = getattr(account, 'available_balance', None)
            hold = getattr(account, 'hold', None)
        
        if not currency:
            return None
        
        # Parse balance values
        available_value = parse_balance_value(available_balance)
        hold_value = parse_balance_value(hold)
        total_value = available_value + hold_value
        
        return {
            'currency': currency,
            'available_balance': available_value,
            'hold_balance': hold_value,
            'total_balance': total_value
        }
        
    except Exception as e:
        logger.error(f"Error parsing account: {e}")
        return None

def check_currency_pair_exists(client: RESTClient, currency: str) -> bool:
    """Check if a currency pair exists before trying to get its price"""
    try:
        # First, try a quick product check
        client.get_product(f"{currency}-USD")
        return True
    except Exception as e:
        error_str = str(e).lower()
        if '404' in error_str or 'not found' in error_str:
            logger.warning(f"Currency pair {currency}-USD not found")
            return False
        else:
            # Other error - might be temporary, so we'll try anyway
            logger.warning(f"Could not verify {currency}-USD existence: {e}")
            return True

def get_currency_price(client: RESTClient, currency: str) -> Optional[float]:
    """Get current price for a currency with proper error handling"""
    if currency == 'USD':
        return 1.0
    
    try:
        # Check if pair exists first
        if not check_currency_pair_exists(client, currency):
            return None
        
        product = client.get_product(f"{currency}-USD")
        
        # Handle both dict and object responses
        if isinstance(product, dict):
            price = product.get('price')
        else:
            price = getattr(product, 'price', None)
        
        if price is not None and float(price) > 0:
            return float(price)
        else:
            logger.warning(f"Invalid price for {currency}: {price}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting price for {currency}-USD: {e}")
        return None

def get_all_accounts(client: RESTClient) -> List[Dict]:
    """Get all accounts with improved pagination and error handling"""
    next_cursor = None
    all_accounts = []
    page = 1
    max_pages = 100  # Safety limit
    
    while page <= max_pages:
        logger.info(f"Fetching page {page} of accounts...")
        try:
            # Get accounts with pagination
            if next_cursor:
                response = client.get_accounts(cursor=next_cursor, limit=250)
            else:
                response = client.get_accounts(limit=250)
            
            # Handle response format (dict vs object)
            if isinstance(response, dict):
                accounts = response.get('accounts', [])
                has_next = response.get('has_next', False)
                next_cursor = response.get('cursor', None)
            else:
                # Object format
                accounts = getattr(response, 'accounts', [])
                has_next = getattr(response, 'has_next', False)
                next_cursor = getattr(response, 'cursor', None)
            
            logger.info(f"Found {len(accounts)} accounts on page {page}")
            
            # Parse each account
            for account in accounts:
                parsed = parse_account(account)
                if parsed and parsed['total_balance'] > 0:
                    all_accounts.append(parsed)
                    logger.debug(f"Account {parsed['currency']}: {parsed['total_balance']:.8f}")
            
            if not has_next or not next_cursor:
                break
                
            page += 1
            
        except Exception as e:
            logger.error(f"Error fetching accounts page {page}: {e}")
            break
    
    logger.info(f"Retrieved total of {len(all_accounts)} non-zero accounts across {page} pages")
    return all_accounts

def test_api_connection(client: RESTClient) -> bool:
    """Test API connection and authentication"""
    logger.info("🔍 Testing API connection and authentication...")
    
    try:
        # Try to get a single account to test auth
        response = client.get_accounts(limit=1)
        logger.info("✅ API authentication successful")
        return True
    except Exception as e:
        error_str = str(e)
        logger.error(f"❌ API authentication failed: {error_str}")
        
        # Provide specific guidance based on error
        if '401' in error_str:
            logger.error("🔑 Authentication Error - Possible causes:")
            logger.error("   1. API key not activated (wait 30 minutes)")
            logger.error("   2. Wrong Coinbase product (should be Advanced Trade)")
            logger.error("   3. Insufficient permissions (need View + Trade)")
            logger.error("   4. Organization/portfolio mismatch")
        elif '403' in error_str:
            logger.error("🚫 Forbidden - API key lacks required permissions")
        elif '429' in error_str:
            logger.error("⏰ Rate limited - too many requests")
        else:
            logger.error("🌐 Network or server error")
        
        return False

def main():
    """Main function with improved error handling and structure"""
    try:
        # Load credentials
        key_name, private_key, base_url, org_id = load_credentials()
        
        # Validate credentials
        if not key_name or not private_key:
            logger.error("❌ Missing API credentials")
            return {"error": "Missing API credentials", "positions": [], "total_value": 0}
        
        # Create client
        logger.info(f"🔌 Connecting to Coinbase API at {base_url}")
        client = RESTClient(
            api_key=key_name,
            api_secret=private_key,
            base_url=base_url
        )
        
        # Test connection first
        if not test_api_connection(client):
            logger.error("❌ Cannot proceed without valid API connection")
            return {"error": "API authentication failed", "positions": [], "total_value": 0}
        
        # Get all accounts
        accounts = get_all_accounts(client)
        
        if not accounts:
            logger.warning("⚠️ No accounts with balances found")
            return {"positions": [], "total_value": 0}
        
        # Process accounts into positions
        positions = []
        total_value = 0
        
        logger.info(f"💰 Processing {len(accounts)} accounts with balances...")
        
        for account in accounts:
            try:
                currency = account['currency']
                amount = account['total_balance']
                
                # Get current price
                price = get_currency_price(client, currency)
                
                if price is None:
                    logger.warning(f"⏭️ Skipping {currency} - could not get price")
                    continue
                
                usd_value = amount * price
                
                position = {
                    'currency': currency,
                    'amount': amount,
                    'price': price,
                    'usd_value': usd_value,
                    'available_balance': account['available_balance'],
                    'hold_balance': account['hold_balance']
                }
                
                positions.append(position)
                total_value += usd_value
                
                logger.info(f"💎 {currency}: {amount:.8f} @ ${price:.4f} = ${usd_value:.2f}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {currency}: {e}")
                continue
        
        # Final result
        result = {
            'positions': positions,
            'total_value': total_value
        }
        
        logger.info(f"🎉 Portfolio Total: ${total_value:.2f}")
        return result
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        return {"error": str(e), "positions": [], "total_value": 0}

if __name__ == "__main__":
    try:
        result = main()
        print(json.dumps(result, indent=2))
        
        if "error" in result:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1) 