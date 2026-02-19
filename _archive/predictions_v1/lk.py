#!/usr/bin/env python3
import os
import sys
import logging
from coinbase.rest import RESTClient
import json

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

try:
    # Import credentials from config.py (same as maybe.py)
    from config import (
        COINBASE_API_KEY as KEY_NAME,
        COINBASE_API_SECRET as PRIVATE_KEY_PEM,
        COINBASE_BASE_URL as BASE_URL,
        COINBASE_ORG_ID as ORG_ID
    )
    logger.info("[OK] Loaded API credentials from config.py")
except ImportError:
    # Fallback to hardcoded credentials if config.py not available - USER SPECIFIED KEY (Updated)
    KEY_NAME = 'organizations/b98ec8e1-610f-451a-9324-40ae8e705d00/apiKeys/87f4e417-95de-420f-96bc-d7235b740ebe'
    PRIVATE_KEY_PEM = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIMpf2xIaq8lZ4F8/Be59qLMy1FlqncvJ5tImgztiTgcEoAoGCCqGSM49
AwEHoUQDQgAEkEu0TA1WDbUETNVHgiw9NzzcVuIrOe3F8sX7HjNWs0RbSTCcXV3L
BcZfzmmTTaqN6bOyVCwIMEyvd3AvRtNv1A==
-----END EC PRIVATE KEY-----"""
    BASE_URL = 'api.coinbase.com'
    ORG_ID = 'b98ec8e1-610f-451a-9324-40ae8e705d00'
    logger.info("[OK] Loaded API credentials from hardcoded values")

# Create client instance for import
client = RESTClient(
    api_key=KEY_NAME,
    api_secret=PRIVATE_KEY_PEM,
    base_url=BASE_URL
)

def get_all_accounts(client):
    """Get all accounts with proper pagination handling"""
    next_cursor = None
    all_accounts = []
    page = 1
    
    while True:
        logger.info(f"Fetching page {page} of accounts...")
        try:
            # Get accounts with pagination
            response = client.get_accounts(cursor=next_cursor, limit=250) if next_cursor else client.get_accounts(limit=250)
            
            # Handle response based on type
            if hasattr(response, 'accounts'):
                # It's a ListAccountsResponse object
                accounts = response.accounts
                has_next = getattr(response, 'has_next', False)
                next_cursor = getattr(response, 'cursor', None)
                
                # Log account details
                for account in accounts:
                    try:
                        currency = getattr(account, 'currency', None)
                        available_balance = getattr(account, 'available_balance', None)
                        hold = getattr(account, 'hold', None)
                        
                        # Debug log raw balance objects
                        logger.info(f"Raw available balance: {available_balance}")
                        logger.info(f"Raw hold balance: {hold}")
                        
                        # Handle balance objects that are dictionaries
                        if isinstance(available_balance, dict):
                            available_value = float(available_balance.get('value', '0'))
                        else:
                            available_value = float(getattr(available_balance, 'value', '0')) if available_balance else 0
                            
                        if isinstance(hold, dict):
                            hold_value = float(hold.get('value', '0'))
                        else:
                            hold_value = float(getattr(hold, 'value', '0')) if hold else 0
                            
                        total_value = available_value + hold_value
                        
                        if currency:  # Log all accounts, even zero balances for debugging
                            logger.info(f"Account {currency}: total={total_value:.8f} (available={available_value:.8f}, hold={hold_value:.8f})")
                    except Exception as e:
                        logger.error(f"Error processing account details: {str(e)}")
                        logger.error(f"Account raw data: {str(account)}")
            else:
                # It's a dict response
                accounts = response.get('accounts', [])
                has_next = response.get('has_next', False)
                next_cursor = response.get('cursor', None)
                
                # Log account details
                for account in accounts:
                    try:
                        currency = account.get('currency')
                        available_balance = account.get('available_balance', {})
                        hold = account.get('hold', {})
                        
                        # Debug log raw balance objects
                        logger.info(f"Raw available balance: {available_balance}")
                        logger.info(f"Raw hold balance: {hold}")
                        
                        available_value = float(available_balance.get('value', '0'))
                        hold_value = float(hold.get('value', '0'))
                        total_value = available_value + hold_value
                        
                        if currency:  # Log all accounts, even zero balances for debugging
                            logger.info(f"Account {currency}: total={total_value:.8f} (available={available_value:.8f}, hold={hold_value:.8f})")
                    except Exception as e:
                        logger.error(f"Error processing account details: {str(e)}")
                        logger.error(f"Account raw data: {account}")
            
            logger.info(f"Found {len(accounts)} accounts on page {page}")
            all_accounts.extend(accounts)
            
            if not has_next or not next_cursor:
                break
                
            page += 1
            
        except Exception as e:
            logger.error(f"Error fetching accounts page {page}: {str(e)}")
            break
    
    logger.info(f"Retrieved total of {len(all_accounts)} accounts across {page} pages")
    return all_accounts

if __name__ == "__main__":
    try:
        accounts = get_all_accounts(client)
        
        # Initialize lists to store position information
        positions = []
        total_value = 0
        
        # Process all accounts
        for account in accounts:
            try:
                # Handle account object
                if hasattr(account, 'currency'):
                    currency = account.currency
                    available_balance = getattr(account, 'available_balance', None)
                    hold = getattr(account, 'hold', None)
                    
                    # Handle balance objects that are dictionaries
                    if isinstance(available_balance, dict):
                        available_value = float(available_balance.get('value', '0'))
                    else:
                        available_value = float(getattr(available_balance, 'value', '0')) if available_balance else 0
                        
                    if isinstance(hold, dict):
                        hold_value = float(hold.get('value', '0'))
                    else:
                        hold_value = float(getattr(hold, 'value', '0')) if hold else 0
                        
                    value = available_value + hold_value
                else:
                    currency = account.get('currency')
                    available_balance = account.get('available_balance', {})
                    hold = account.get('hold', {})
                    
                    available_value = float(available_balance.get('value', '0'))
                    hold_value = float(hold.get('value', '0'))
                    value = available_value + hold_value
                
                if currency and value > 0:
                    usd_value = value
                    price = None
                    
                    # If it's not USD, get the current price
                    if currency != 'USD':
                        try:
                            product = client.get_product(f"{currency}-USD")
                            if hasattr(product, 'price'):
                                price = float(product.price)
                            elif isinstance(product, dict):
                                price = float(product.get('price', 0))
                                
                            if price > 0:
                                usd_value = value * price
                                logger.info(f"Found {currency} position: {value:.8f} @ ${price:.4f} = ${usd_value:.2f}")
                                positions.append({
                                    'currency': currency,
                                    'amount': value,
                                    'price': price,
                                    'usd_value': usd_value
                                })
                        except Exception as e:
                            logger.error(f"Error getting price for {currency}-USD: {str(e)}")
                    else:
                        logger.info(f"Found USD balance: ${value:.2f}")
                        positions.append({
                            'currency': currency,
                            'amount': value,
                            'price': 1.0,
                            'usd_value': value
                        })
                    
                    total_value += usd_value
                    logger.info(f"Running total: ${total_value:.2f}")
            
            except Exception as e:
                logger.error(f"Error processing account {str(account)}: {str(e)}")
                continue
        
        # Print final results
        print(json.dumps({
            'positions': positions,
            'total_value': total_value
        }))
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1) 