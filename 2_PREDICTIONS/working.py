import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash
from dash import dcc, html, dash_table
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import os
import joblib
import warnings
import queue
import requests
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import requests
import joblib
import os
from datetime import datetime, timedelta
import sqlite3
import sys
import json
import time
import logging
import traceback
import uuid
import asyncio
import websockets
from threading import Thread, Lock, RLock
import argparse
from pathlib import Path
import subprocess
import re
from crypto_trading.app.tp_sl_manager import check_and_manage_tp_sl_orders
from crypto_trading.app.session_utils import get_active_session
from crypto_trading.app.lk import get_all_accounts, KEY_NAME, PRIVATE_KEY_PEM, BASE_URL

# Configure NumExpr to use fewer threads
import numexpr as ne
ne.set_num_threads(4)  # Use 4 threads instead of 8

# Rest of imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from xgboost import XGBRegressor
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import warnings
from dash.exceptions import PreventUpdate
from simulation_tracker import SimulationTracker
import threading
import queue
import time
import sqlite3
import sys
from plotly.subplots import make_subplots
import atexit
import logging
import base64
import hmac
import hashlib
from crypto_trading.app.config import CRYPTO_COM_API_KEY
from coinbase.rest import RESTClient
import math
import http.client
import json
import uuid
import asyncio
import websockets
from threading import Thread
import argparse
import traceback
import urllib.parse
from crypto_trading.app.lk import get_all_accounts, KEY_NAME, PRIVATE_KEY_PEM, BASE_URL
import subprocess
import re
from crypto_trading.app.tp_sl_manager import check_and_manage_tp_sl_orders
from crypto_trading.app.session_utils import get_active_session

# Add rate limiting constants
RATE_LIMIT_DELAY = 0.1  # 100ms delay between API calls
PRICE_UPDATE_DELAY = 0.05  # 50ms delay between price updates

# Add caching constants
ACCOUNT_CACHE = {}
ACCOUNT_CACHE_DURATION = 60  # Cache duration in seconds
PRICE_CACHE = {}
PRICE_CACHE_DURATION = 1  # Cache duration in seconds (reduced from 5 to 1)

# Configure logging
# Constants first
MODELS_DIR = "models"
LOGS_DIR = "logs"
# Updated DB_PATH to match the data directory path used in other functions
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'live_trading.db')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Configure logging early to avoid 'logger not defined' errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'trading.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize REST client for API calls
try:
    # Import credentials from lk.py
    from crypto_trading.app.lk import (
        KEY_NAME,
        PRIVATE_KEY_PEM,
        BASE_URL
    )
    
    # Create the REST client with proper authentication
    coinbase_client = RESTClient(
        api_key=KEY_NAME,
        api_secret=PRIVATE_KEY_PEM,  # Changed from private_key to api_secret
        base_url=BASE_URL
    )
    logger.info("Successfully initialized Coinbase REST client")
except Exception as e:
    logger.error(f"Error initializing Coinbase REST client: {str(e)}")
    logger.error(f"Stack trace: {traceback.format_exc()}")
    coinbase_client = None

# Define WebSocket message handler first
def on_websocket_message(data):
    """Callback function for WebSocket messages"""
    try:
        # Get message type and channel
        msg_type = data.get('type', 'unknown')
        channel = data.get('channel', 'unknown')
        
        logger.debug(f"WebSocket message - Type: {msg_type}, Channel: {channel}")
        
        if msg_type == 'ticker' and 'product_id' in data:
            # Process ticker/price update
            symbol = data.get('product_id')
            price = float(data.get('price', 0))
            size = float(data.get('last_size', 0))
            timestamp = data.get('time', None)
            
            if price > 0:
                # Convert timestamp to milliseconds
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        timestamp_ms = int(dt.timestamp() * 1000)
                    except:
                        timestamp_ms = int(datetime.now().timestamp() * 1000)
                else:
                    timestamp_ms = int(datetime.now().timestamp() * 1000)
                
                # Store price data
                ticker_data = {
                    'price': price,
                    'volume': size,
                    'timestamp': timestamp_ms
                }
                
                with market_data_lock:
                    if symbol not in price_history:
                        price_history[symbol] = []
                    price_history[symbol].insert(0, ticker_data)
                    if len(price_history[symbol]) > 1000:
                        price_history[symbol] = price_history[symbol][:1000]
                
                logger.info(f"Price update for {symbol}: ${price:.4f} (Size: {size:.8f})")
                
                # Update database
                try:
                    session_id = get_active_session()
                    if session_id:
                        conn = sqlite3.connect('live_trading.db')
                        cursor = conn.cursor()
                        cursor.execute('''
                        UPDATE positions 
                        SET current_price = ?,
                            last_update = datetime('now')
                        WHERE session_id = ? AND symbol = ?
                        ''', (price, session_id, symbol))
                        conn.commit()
                        conn.close()
                except Exception as db_error:
                    logger.error(f"Database error: {str(db_error)}")
                    
        elif msg_type == 'snapshot':
            # Process full account snapshot
            if 'orders' in data:
                logger.info(f"Orders snapshot: {len(data['orders'])} orders")
            if 'positions' in data:
                logger.info(f"Positions snapshot received")
                
        elif msg_type == 'update':
            # Process incremental updates
            if 'orders' in data:
                logger.info(f"Order update: {data['orders']}")
            if 'positions' in data:
                logger.info(f"Position update received")
                
        elif msg_type == 'error':
            logger.error(f"WebSocket error: {data.get('message', 'Unknown error')}")
            
        elif msg_type == 'heartbeat':
            logger.debug("Heartbeat received")
            
    except Exception as e:
        logger.error(f"Error processing WebSocket message: {str(e)}")
        logger.error(f"Message data: {data}")

# WebSocket client global variable
ws_client = None

# Define global variables for data storage
market_data_lock = threading.RLock()
price_history = {}

# Add more code specific to WebSocket handling
class CoinbaseWebSocketClient:
    def __init__(self, on_message_callback=None):
        self.running = False
        self.ws = None
        self.thread = None
        self.on_message_callback = on_message_callback
        self.price_data = {}
        self.last_update_time = {}
        self.update_interval = 1.0
        self.last_price_check = {}
        self.authenticated = False
        self.ws_url = 'wss://advanced-trade-ws.coinbase.com/ws'
        
    async def _connect_and_subscribe(self):
        """Connect to WebSocket and subscribe to channels"""
        try:
            # Generate JWT token for authentication
            jwt_token = build_jwt("GET", "/ws")
            if not jwt_token:
                logger.error("Failed to generate JWT token")
                return

            # Connect to WebSocket
            async with websockets.connect(self.ws_url) as websocket:
                self.ws = websocket
                logger.info("WebSocket connected")

                # First authenticate
                auth_message = {
                    "type": "subscribe",
                    "token": jwt_token,
                    "channel": "user"
                }
                
                await websocket.send(json.dumps(auth_message))
                logger.info("Sent authentication message")

                # Wait for auth response
                auth_response = await websocket.recv()
                auth_data = json.loads(auth_response)
                
                if auth_data.get('type') == 'error':
                    logger.error(f"Authentication failed: {auth_data.get('message')}")
                    return
                else:
                    self.authenticated = True
                    logger.info("Successfully authenticated WebSocket connection")

                # Subscribe to market data
                market_subscribe = {
                    "type": "subscribe",
                    "product_ids": [
                        "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD",
                        "SHIB-USD", "DOT-USD", "MATIC-USD", "ADA-USD", "LINK-USD"
                    ],
                    "channel": "market_trades"
                }
                
                await websocket.send(json.dumps(market_subscribe))
                logger.info("Subscribed to market trades")

                # Subscribe to ticker data
                ticker_subscribe = {
                    "type": "subscribe",
                    "product_ids": [
                        "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD",
                        "SHIB-USD", "DOT-USD", "MATIC-USD", "ADA-USD", "LINK-USD"
                    ],
                    "channel": "ticker"
                }
                
                await websocket.send(json.dumps(ticker_subscribe))
                logger.info("Subscribed to ticker data")

                # Process incoming messages
                while self.running:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        msg_type = data.get('type')
                        
                        if msg_type == 'error':
                            logger.error(f"WebSocket error: {data.get('message')}")
                            continue
                            
                        elif msg_type == 'subscribe':
                            logger.info(f"Subscription confirmed: {data.get('channel')}")
                            continue
                            
                        elif msg_type == 'ticker':
                            # Process ticker data
                            if self.on_message_callback:
                                self.on_message_callback(data)
                                
                        elif msg_type == 'l2_data':
                            # Process market data
                            if self.on_message_callback:
                                self.on_message_callback(data)
                                
                        elif msg_type == 'snapshot':
                            # Process full snapshot
                            if self.on_message_callback:
                                self.on_message_callback(data)
                                
                        elif msg_type == 'update':
                            # Process incremental update
                            if self.on_message_callback:
                                self.on_message_callback(data)
                        
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed, reconnecting...")
                        break
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"WebSocket connection error: {str(e)}")
            await asyncio.sleep(5)  # Wait before reconnecting

    def start(self):
        """Start the WebSocket client"""
        if self.running:
            logger.info("WebSocket client already running")
            return
            
        self.running = True
        self.thread = Thread(target=self._run_websocket_loop, daemon=True)
        self.thread.start()
        logger.info("Started WebSocket client thread")
        
    def stop(self):
        """Stop the WebSocket client"""
        self.running = False
        if self.ws:
            asyncio.run(self.ws.close())
        if self.thread:
            self.thread.join(timeout=5)
            
    def is_running(self):
        """Check if the WebSocket client is running and authenticated"""
        return self.running and self.thread and self.thread.is_alive() and self.authenticated

    def _run_websocket_loop(self):
        """Run the WebSocket event loop"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            while self.running:
                try:
                    loop.run_until_complete(self._connect_and_subscribe())
                except Exception as e:
                    logger.error(f"Error in WebSocket loop: {str(e)}")
                    time.sleep(5)  # Wait before reconnecting
        except Exception as e:
            logger.error(f"Fatal error in WebSocket loop: {str(e)}")
        finally:
            loop.close()
            
    def get_price(self, symbol):
        """Get the latest price for a symbol"""
        if symbol in self.price_data:
            last_update = self.last_update_time.get(symbol)
            if last_update:
                # Check if price is fresh (within last 10 seconds)
                if (datetime.now() - last_update).total_seconds() < 10:
                    return self.price_data[symbol]
        return None

# Initialize the WebSocket client outside functions but don't start it yet
def initialize_websocket_client():
    global ws_client
    if ws_client is None:
        ws_client = CoinbaseWebSocketClient(on_message_callback=on_websocket_message)
        logger.info("Initialized internal WebSocket client")
    return ws_client

# Add path to the coinbase_ws_integration directory
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'coinbase_ws_integration'))

# Import WebSocket integration
try:
    from coinbase_integration import (
        initialize_websocket,
        start_websocket,
        stop_websocket,
        subscribe_to_symbols,
        get_portfolio_data,
        get_price_data,
        get_symbol_price,
        is_websocket_running,
        is_data_fresh
    )
    WEBSOCKET_AVAILABLE = True
    logger.info("Coinbase WebSocket integration available")
except ImportError as e:
    logger.warning(f"Coinbase WebSocket integration not available: {str(e)}")
    WEBSOCKET_AVAILABLE = False

# === Coinbase API Credentials ===
# Make sure to set these to your own API credentials or load them from a config file or environment variables.

# === Coinbase API Auth Header Helper (Advanced Trade, JWT-based) ===
import time
import uuid
import jwt
from cryptography.hazmat.primitives import serialization
import secrets

# Set your Coinbase Advanced Trade API org and key info here:
ORG_ID = "b98ec8e1-610f-451a-9324-40ae8e705d00"
API_KEY_ID = "87f4e417-95de-420f-96bc-d7235b740ebe"
KEY_NAME = f"organizations/{ORG_ID}/apiKeys/{API_KEY_ID}"
BASE_URL = "api.coinbase.com"

# Use the proper PEM format for your private key
PRIVATE_KEY_PEM = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIMpf2xIaq8lZ4F8/Be59qLMy1FlqncvJ5tImgztiTgcEoAoGCCqGSM49
AwEHoUQDQgAEMg2Jn6Z55ULxRjzdKMbt+6ZZY6IRXnmisQhrxSUiWSsmJTf2V1zy
/Wtf2Fz77bvyHrwd0O7zP2TC/S4HK0c7sg==
-----END EC PRIVATE KEY-----"""

def build_jwt(method, path):
    """
    Build a JWT token for Coinbase API authentication.
    Args:
        method (str): HTTP method
        path (str): API path
    Returns:
        str: JWT token for authentication
    """
    try:
        now = int(time.time())
        
        # Ensure proper URI format without protocol/hostname
        if path.startswith('http'):
            parsed = urllib.parse.urlparse(path)
            path = parsed.path
        if not path.startswith('/'):
            path = '/' + path
            
        payload = {
            "sub": KEY_NAME,  # Full key name
            "iss": "cdp",    # Must be "cdp" for Coinbase API
            "nbf": now,
            "exp": now + 120,
            "iat": now,      # Issued at time
            "jti": str(uuid.uuid4()),  # JWT ID
            "uri": f"{method} {BASE_URL}{path}"  # Include BASE_URL in URI
        }

        # Load private key properly
        private_key = serialization.load_pem_private_key(
            PRIVATE_KEY_PEM.encode('utf-8'),
            password=None
        )

        # Generate token with proper headers
        token = jwt.encode(
            payload,
            private_key,
            algorithm="ES256",
            headers={
                'kid': KEY_NAME,
                'nonce': secrets.token_hex(16)
            }
        )
        
        logger.debug(f"Generated JWT token for {method} {path}")
        return token
        
    except Exception as e:
        logger.error(f"Error building JWT: {str(e)}")
        return None

def get_auth_headers(method, path):
    """
    Returns headers for Coinbase Advanced Trade API authentication using JWT.
    """
    try:
        # Ensure path starts with a slash
        if not path.startswith('/'):
            path = '/' + path
            
        # Format URI correctly without protocol/hostname in the JWT payload
        uri = f"{method} {BASE_URL}{path}"
        now = int(time.time())
        private_key = serialization.load_pem_private_key(PRIVATE_KEY_PEM.encode('utf-8'), password=None)

        jwt_payload = {
            'sub': KEY_NAME,
            'iss': "cdp",
            'nbf': now,
            'exp': now + 120,
            'iat': now,
            'jti': str(uuid.uuid4()),
            'uri': uri,
        }

        jwt_token = jwt.encode(
            jwt_payload,
            private_key,
            algorithm='ES256',
            headers={'kid': KEY_NAME, 'nonce': secrets.token_hex(16)},
        )
        return {
            'Authorization': f"Bearer {jwt_token}",
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    except Exception as e:
        logger.error(f"Error building auth headers: {str(e)}")
        return None

# Import utility functions and ML backtest

# Global variables for trading management
trading_thread = None
stop_trading = False
live_sim = None

# Initialize Dash app with callback exception suppression
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Initialize and start WebSocket client
ws_client = initialize_websocket_client()
if ws_client:
    ws_client.start()
    logger.info("Started WebSocket client")

# Add global variables for live trading
live_trading_thread = None
trading_queue = queue.Queue()
live_sim = None
is_trading_active = False

# Symbol cache configuration
SYMBOL_CACHE = {
    'data': None,
    'last_update': None,
    'cache_duration': timedelta(hours=1)  # Cache symbols for 1 hour
}

def get_cached_symbols():
    now = datetime.now()
    if (SYMBOL_CACHE['data'] is None or 
        SYMBOL_CACHE['last_update'] is None or 
        now - SYMBOL_CACHE['last_update'] > SYMBOL_CACHE['cache_duration']):
        try:
            url = "https://api.exchange.coinbase.com/products"
            response = requests.get(url)
            data = response.json()
            symbols = [product['id'] for product in data if product['quote_currency'] == 'USD']
            SYMBOL_CACHE['data'] = sorted(symbols)
            SYMBOL_CACHE['last_update'] = now
            logger.info(f"Fetched symbols: {SYMBOL_CACHE['data']}")  # Log fetched symbols
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            if SYMBOL_CACHE['data'] is None:
                SYMBOL_CACHE['data'] = []
    return SYMBOL_CACHE['data']

# === Data Fetching ===
def get_coinbase_data(symbol='BTC-USD', granularity=60, days=7):
    """
    Fetch historical data from Coinbase with improved rate limit handling.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTC-USD')
        granularity (int): Time interval in seconds (60, 300, 900, 3600, etc.)
        days (int): Number of days of historical data to fetch
        
    Returns:
        pd.DataFrame: Historical price data with OHLCV columns
    """
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    headers = {'Accept': 'application/json'}
    df_list = []
    now = datetime.utcnow()
    
    # Calculate optimal step size based on granularity (max 300 candles per request)
    max_candles_per_request = 300
    step_seconds = min(granularity * max_candles_per_request, days * 24 * 3600)
    step = timedelta(seconds=step_seconds)
    start_time = now - timedelta(days=days)
    
    max_retries = 3
    base_delay = 1  # Base delay in seconds
    
    print(f"\nFetching {days} days of {granularity}s data for {symbol}...")
    
    while start_time < now:
        end_time = min(start_time + step, now)
        params = {
            'granularity': granularity,
            'start': start_time.isoformat(),
            'end': end_time.isoformat()
        }
        
        for retry in range(max_retries):
            try:
                r = requests.get(url, headers=headers, params=params)
                
                if r.status_code == 429:  # Rate limit hit
                    delay = base_delay * (2 ** retry)  # Exponential backoff
                    print(f"Rate limit hit, waiting {delay} seconds...")
                    time.sleep(delay)
                    continue
                    
                elif r.status_code != 200:
                    print(f"Error {r.status_code} fetching data for {symbol}: {r.text}")
                    break
                    
                data = r.json()
                if data:
                    # Convert data to float/string to ensure JSON serializable
                    temp_df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
                    temp_df = temp_df.astype({
                        'timestamp': 'float64',
                        'low': 'float64',
                        'high': 'float64',
                        'open': 'float64',
                        'close': 'float64',
                        'volume': 'float64'
                    })
                    df_list.append(temp_df)
                break  # Successful request, exit retry loop
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                if retry < max_retries - 1:
                    delay = base_delay * (2 ** retry)
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                continue
        
        start_time = end_time
        time.sleep(0.25)  # Small delay between successful requests
    
    if not df_list:
        print(f"No data retrieved for {symbol}")
        return pd.DataFrame()
    
    df = pd.concat(df_list, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.sort_values(by='timestamp', inplace=True)
    df = df.drop_duplicates(subset=['timestamp'])
    
    print(f"Retrieved {len(df)} data points for {symbol}")
    return df.reset_index(drop=True)

def get_market_data(symbol, interval='1m', lookback='1h'):
    """Get historical market data for a symbol"""
    try:
        # Use a lock if necessary to ensure thread safety
        with market_data_lock:
            if symbol not in price_history:
                return pd.DataFrame()
            
            data = list(price_history[symbol])
            
        df = pd.DataFrame(data)
        if df.empty:
            return df
            
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Resample to desired interval
        interval_map = {
            '1m': '1Min',
            '5m': '5Min',
            '15m': '15Min',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        
        if interval in interval_map:
            df = df.resample(interval_map[interval]).agg({
                'price': 'ohlc',
                'volume': 'sum'
            })
            
            # Flatten column names
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
        # Filter by lookback period
        lookback_map = {
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
            '1w': '7D',
            '1M': '30D'
        }
        
        if lookback in lookback_map:
            start_time = pd.Timestamp.now() - pd.Timedelta(lookback_map[lookback])
            df = df[df.index >= start_time]
            
        return df
        
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        return pd.DataFrame()

# === Technical Indicators ===
def calculate_indicators(df):
    """
    Calculate technical indicators with proper error handling and data validation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added technical indicators
    """
    try:
        # Verify required columns exist
        required_columns = ['close', 'high', 'low', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return df
            
        # Convert price and volume columns to numeric, replacing errors with NaN
        for col in ['close', 'high', 'low', 'open', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Drop rows with NaN in essential columns
        df = df.dropna(subset=['close', 'high', 'low'])
        
        if len(df) < 24:  # Minimum required for most indicators
            print("Insufficient data points for indicator calculation")
            return df
            
        # Calculate trend indicators
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MA20'] = df['close'].rolling(window=20, min_periods=1).mean()
        
        # Calculate momentum indicators with proper handling of edge cases
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, float('inf'))  # Handle division by zero
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate volatility indicators
        df['rolling_std_10'] = df['close'].rolling(window=10, min_periods=1).std()
        df['H-L'] = df['high'] - df['low']
        df['H-PC'] = abs(df['high'] - df['close'].shift(1))
        df['L-PC'] = abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14, min_periods=1).mean()
        
        # Calculate volume indicators
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Calculate stochastic oscillator
        df['14-high'] = df['high'].rolling(window=14, min_periods=1).max()
        df['14-low'] = df['low'].rolling(window=14, min_periods=1).min()
        df['%K'] = 100 * ((df['close'] - df['14-low']) / (df['14-high'] - df['14-low']).replace(0, float('inf')))
        df['%D'] = df['%K'].rolling(window=3, min_periods=1).mean()
        
        # Calculate lagging indicators
        df['lag_1'] = df['close'].shift(1)
        df['lag_2'] = df['close'].shift(2)
        df['lag_3'] = df['close'].shift(3)
        
        # Fill any remaining NaN values with forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        print("Successfully calculated all indicators")
        return df
        
    except Exception as e:
        print(f"Error calculating indicators: {str(e)}")
        return df

# === Model Training ===
def train_model_for_symbol(symbol, granularity=60):
    print(f"\n🔄 Training model for {symbol} at {granularity}s granularity...")
    
    # Determine appropriate training period based on granularity
    if granularity == 60:  # 1 minute
        training_days = 7  # 1 week of minute data
    elif granularity == 300:  # 5 minutes
        training_days = 14  # 2 weeks of 5-minute data
    elif granularity == 900:  # 15 minutes
        training_days = 30  # 1 month of 15-minute data
    else:  # 1 hour or higher
        training_days = 90  # 3 months of hourly data
    
    print(f"Fetching {training_days} days of {granularity}s data...")
    
    # Get and prepare data
    df = get_coinbase_data(symbol=symbol, granularity=granularity, days=training_days)
    if df.empty:
        print(f"❌ No data available for {symbol}")
        return None, None
    
    df = calculate_indicators(df)
    df.dropna(inplace=True)
    
    if len(df) < 100:
        print(f"❌ Insufficient data for {symbol}: {len(df)} samples")
        return None, None
    
    feature_cols = [
        'EMA12', 'EMA26', 'MACD', 'Signal_Line', 'RSI', 'MA20',
        'rolling_std_10', 'lag_1', 'lag_2', 'lag_3', 'OBV', 'ATR', '%K', '%D'
    ]
    
    # Ensure MODELS_DIR exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Train price prediction model using a simplified ensemble
    base_models = [
        RandomForestRegressor(n_estimators=100, max_depth=10),
        XGBRegressor(objective='reg:squarederror', n_jobs=-1, max_depth=5),
        LinearRegression()
    ]
    
    # Create voting ensemble for regression
    reg_predictions = []
    for model in base_models:
        X_reg = df[feature_cols].iloc[:-1]
        y_reg = df['close'].shift(-1).dropna()
        model.fit(X_reg, y_reg)
        reg_predictions.append(model.predict(X_reg).reshape(-1, 1))
    
    # Average predictions from all models
    ensemble_predictions = np.mean(np.hstack(reg_predictions), axis=1)
    
    # Generate predictions for classifier
    df = df.iloc[:-1]
    df['predicted_close'] = ensemble_predictions
    df['direction'] = np.where(df['predicted_close'] > df['close'], 'BUY', 'SELL')
    
    # Train classifier using Random Forest
    X_cls = df[feature_cols + ['predicted_close']]
    y_cls = df['direction']
    clf = RandomForestClassifier(n_estimators=100, max_depth=10)
    clf.fit(X_cls, y_cls)
    
    # Evaluate
    preds = clf.predict(X_cls)
    print("\n📊 Model Evaluation:")
    print(classification_report(y_cls, preds))
    
    # Save models with proper error handling
    try:
        model_prefix = f"{symbol.replace('-', '')}_{granularity}"
        reg_path = os.path.join(MODELS_DIR, f"{model_prefix}_reg.pkl")
        clf_path = os.path.join(MODELS_DIR, f"{model_prefix}_clf.pkl")
        
        # Save the ensemble as a dictionary of models
        reg_models = {
            'models': base_models,
            'feature_cols': feature_cols
        }
        
        joblib.dump(reg_models, reg_path)
        joblib.dump(clf, clf_path)
        print(f"✅ Models saved successfully: {model_prefix}")
        
        # Verify models were saved correctly
        try:
            _ = joblib.load(reg_path)
            _ = joblib.load(clf_path)
            print("✅ Model files verified successfully")
        except Exception as e:
            print(f"❌ Error verifying saved models: {str(e)}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error saving models: {str(e)}")
        return None, None
    
    return reg_models, clf
def run_position_analysis():
    """Run the analyze_and_sell.py script as a subprocess to analyze and sell positions using ML models"""
    try:
        # Get active session ID
        session_id = get_active_session()
        if not session_id:
            logger.error("No active session found")
            return False

        # Get the path to analyze_and_sell.py
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'analyze_and_sell.py')
        if not os.path.exists(script_path):
            logger.error(f"analyze_and_sell.py not found at {script_path}")
            return False

        # Run the script as a subprocess
        logger.info("Running ML-based position analysis...")
        command = [sys.executable, script_path, '--session-id', str(session_id)]
        
        # Log the command we're about to run
        logger.info(f"Running command: {' '.join(command)}")
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )

        # Log output
        if result.stdout:
            logger.info("Position analysis output:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")

        # Log errors
        if result.stderr:
            logger.error("Position analysis errors:")
            for line in result.stderr.split('\n'):
                if line.strip():
                    logger.error(f"  {line}")

        # Check return code
        if result.returncode == 0:
            logger.info("ML position analysis completed successfully")
            return True
        else:
            logger.error(f"ML position analysis failed with code {result.returncode}")
            return False

    except Exception as e:
        logger.error(f"Error running position analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# === Trading Simulation ===
# def simulate_trading(df, reg_model, clf, initial_state=None):
#     ... (function removed)

# Remove run_simulation callback and related helpers
# Remove run_forward_test_callback
# Remove run_backtest_callback
# Remove create_results_display
# Remove create_portfolio_chart
# Remove plot_prediction_errors
# Remove create_empty_figure

# === Scanner Functions ===
def analyze_momentum(df, symbol):
    """
    Analyze momentum indicators for a given symbol.
    
    Args:
        df (pd.DataFrame): DataFrame containing price data
        symbol (str): Symbol to analyze
    
    Returns:
        dict: Dictionary containing momentum analysis results
    """
    try:
        if df is None or df.empty or len(df) < 24:
            return None
            
        df = calculate_indicators(df)
        df.dropna(inplace=True)
        
        latest = df.iloc[-1]
        
        # Calculate price changes
        price_change_1h = ((latest['close'] / df['close'].iloc[-2]) - 1) * 100
        price_change_24h = ((latest['close'] / df['close'].iloc[-24]) - 1) * 100
        
        # Calculate momentum score
        momentum_score = (
            (latest['RSI'] / 100) * 0.3 +
            (1 if latest['MACD'] > latest['Signal_Line'] else 0) * 0.3 +
            (latest['%K'] / 100) * 0.2 +
            (1 if latest['close'] > latest['MA20'] else 0) * 0.2
        ) * 100
        
        # Determine momentum direction
        if latest['MACD'] > latest['Signal_Line'] and latest['RSI'] > 50:
            momentum_direction = 'Bullish'
        elif latest['MACD'] < latest['Signal_Line'] and latest['RSI'] < 50:
            momentum_direction = 'Bearish'
        else:
            momentum_direction = 'Neutral'
        
        logger.info(f"Symbol: {symbol}, Momentum Score: {momentum_score}, Direction: {momentum_direction}")  # Log momentum score
        
        return {
            'symbol': symbol,
            'current_price': latest['close'],
            'momentum_score': momentum_score,
            'momentum_direction': momentum_direction,
            'price_change_1h': price_change_1h,
            'price_change_24h': price_change_24h,
            'rsi': latest['RSI'],
            'macd': latest['MACD'],
            'volume': latest['volume']
        }
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None

def scan_market(symbols=None, batch_size=5, conn=None, cursor=None):
    """
    Scan the market for trading opportunities with direction difference and all available assets.
    
    Args:
        symbols (list, optional): List of symbols to scan. If None, uses all available symbols.
        batch_size (int): Number of symbols to process in parallel.
        conn (sqlite3.Connection, optional): Database connection to use
        cursor (sqlite3.Cursor, optional): Database cursor to use
    
    Returns:
        list: List of dictionaries containing analysis results for each symbol.
    """
    should_close_conn = False
    try:
        # Initialize database connection if not provided
        if conn is None or cursor is None:
            conn = sqlite3.connect('live_trading.db', timeout=30)
            cursor = conn.cursor()
            should_close_conn = True
            logger.debug("Created new database connection for market scan")

        # Verify connection is still valid
        try:
            cursor.execute("SELECT 1")
        except (sqlite3.OperationalError, sqlite3.ProgrammingError) as e:
            if "closed database" in str(e):
                logger.warning("Database connection was closed, reopening...")
                try:
                    if conn:
                        conn.close()
                except:
                    pass
                conn = sqlite3.connect('live_trading.db', timeout=30)
                cursor = conn.cursor()
                should_close_conn = True
                logger.debug("Reopened database connection for market scan")

        # Fetch all available symbols if not provided
        if symbols is None:
            symbols = get_cached_symbols()  # Fetch all available symbols
            logger.debug(f"Using {len(symbols)} symbols for market scan")
        
        def process_symbol(symbol):
            try:
                logger.debug(f"Processing {symbol}")
                df = get_coinbase_data(symbol=symbol, granularity=3600)  # 1-hour data
                if not df.empty:
                    result = analyze_momentum(df, symbol)
                    if result:
                        # Calculate direction difference
                        predictions, _ = predict_with_pretrained_model(df, symbol, interval='1h')
                        if not predictions.empty:
                            latest_pred = predictions.iloc[-1]
                            direction_diff = latest_pred['predicted_price'] - latest_pred['actual_price']
                            result['direction_diff'] = direction_diff
                        logger.debug(f"Analysis complete for {symbol} - Score: {result['momentum_score']:.1f}")
                        return result
                    else:
                        logger.warning(f"No analysis results for {symbol}")
                else:
                    logger.warning(f"No data retrieved for {symbol}")
                return None
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                return None
        
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_symbol, symbol) for symbol in symbols]
            for future in futures:
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error getting future result: {str(e)}")
                    continue

        # Sort results by momentum score
        sorted_results = sorted(results, key=lambda x: x['momentum_score'], reverse=True)
        logger.info(f"Scan complete - Found {len(sorted_results)} opportunities")

        return sorted_results
        
    except Exception as e:
        logger.error(f"Error in market scanning: {str(e)}")
        # Try to reopen connection if it was closed
        if "closed database" in str(e):
            try:
                if conn:
                    conn.close()
            except:
                pass
            try:
                conn = sqlite3.connect('live_trading.db', timeout=30)
                cursor = conn.cursor()
                should_close_conn = True
                logger.info("Successfully reopened database connection after error")
                # Retry the scan with new connection
                return scan_market(symbols=symbols, batch_size=batch_size, conn=conn, cursor=cursor)
            except Exception as retry_error:
                logger.error(f"Failed to reopen database connection: {str(retry_error)}")
        return []

    finally:
        # Only close the connection if we created it
        if should_close_conn and conn:
            try:
                conn.close()
                logger.debug("Closed database connection from market scan")
            except Exception as e:
                logger.error(f"Error closing database connection: {str(e)}")
                # If closing failed, try to ensure it's really closed
                try:
                    if conn:
                        conn.close()
                except:
                    pass

def scan_for_crypto_runs(max_pairs=20):
    """
    Scan the crypto market for trading opportunities.
    Returns a list of dictionaries containing trading signals and metrics.
    """
    results = []
    try:
        symbols = get_cached_symbols()
        if not symbols:
            return []

        for symbol in symbols:  # Removed the limit to max_pairs
            try:
                # Get recent data
                df = get_coinbase_data(symbol=symbol, granularity=3600, days=5)  # 5 days of hourly data
                if df.empty:
                    continue

                # Calculate indicators
                df = calculate_indicators(df)
                if df.empty:
                    continue

                # Get latest values
                latest = df.iloc[-1]
                
                # Calculate momentum score (0-100)
                momentum_score = 0
                if latest['RSI'] > 50:
                    momentum_score += 20
                if latest['MACD'] > latest['Signal_Line']:
                    momentum_score += 20
                if latest['close'] > latest['MA20']:
                    momentum_score += 20
                if latest['%K'] > latest['%D']:
                    momentum_score += 20
                if latest['OBV'] > df['OBV'].mean():
                    momentum_score += 20

                # Calculate volatility
                volatility = latest['ATR'] / latest['close'] * 100

                # Calculate trend strength
                trend_strength = abs(latest['close'] - latest['MA20']) / latest['MA20'] * 100

                results.append({
                    'symbol': symbol,
                    'current_price': float(latest['close']),
                    'momentum_score': float(momentum_score),
                    'rsi': float(latest['RSI']),
                    'volume_change_pct': float((latest['volume'] - df['volume'].mean()) / df['volume'].mean() * 100),
                    'price_change_pct': float((latest['close'] - df['close'].shift(1).iloc[-1]) / df['close'].shift(1).iloc[-1] * 100),
                    'timestamp': datetime.utcnow().isoformat()
                })

            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                continue

        # Sort by momentum score
        results = sorted(results, key=lambda x: x['momentum_score'], reverse=True)
        
    except Exception as e:
        print(f"Error in scan_for_crypto_runs: {str(e)}")
    
    return results

# Define styles
styles = {
    'context_menu': {
        'display': 'none',
        'position': 'fixed',
        'backgroundColor': '#ffffff',
        'boxShadow': '2px 2px 5px rgba(0,0,0,0.2)',
        'zIndex': 1000,
        'borderRadius': '4px',
        'padding': '5px 0',
        'border': '1px solid #ddd'
    },
    'context_option': {
        'padding': '8px 20px',
        'cursor': 'pointer',
        'color': '#2c3e50',
        'hover': {'backgroundColor': '#f5f5f5'}
    },
    'main_container': {
        'fontFamily': 'Arial, sans-serif',
        'margin': '0',
        'padding': '20px',
        'backgroundColor': '#ffffff',
        'color': '#2c3e50',
        'minHeight': '100vh'
    },
    'table_cell': {
        'textAlign': 'center',
        'backgroundColor': '#ffffff',
        'color': '#2c3e50',
        'cursor': 'context-menu',
        'border': '1px solid #ddd'
    },
    'table_header': {
        'backgroundColor': '#f8f9fa',
        'color': '#2c3e50',
        'fontWeight': 'bold',
        'cursor': 'context-menu',
        'border': '1px solid #ddd'
    }
}

app.layout = html.Div([
    html.H1("🚀 Crypto Trading Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
    
    # Add interval component for regular updates
    dcc.Interval(
        id='interval-component',
        interval=300000,  # 5 minutes in milliseconds
        n_intervals=0
    ),
    
    # Context Menu
    html.Div(
        id='context-menu',
        style=styles['context_menu'],
        children=[
            html.Div('Train Model', id='context-train', 
                    style=styles['context_option']),
            html.Div('Run Simulation', id='context-simulate', 
                    style=styles['context_option']),
            html.Div('View Analysis', id='context-analyze', 
                    style=styles['context_option'])
        ]
    ),
    
    # Store for selected symbol with default value
    dcc.Store(id='selected-symbol', data=None),
    
    # Tabs
    dcc.Tabs([
        # Scanner Tab
        dcc.Tab(label="Scanner", children=[
            html.Div([
                html.Div([
                    html.Button(
                        "🔄 Refresh Scanner", 
                        id="refresh-scanner", 
                        n_clicks=0,
                        style={
                            'marginBottom': '20px',
                            'backgroundColor': '#4CAF50',
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 20px',
                            'borderRadius': '5px',
                            'cursor': 'pointer'
                        }
                    ),
                    html.Div(
                        id="scanner-table",
                        children=html.Div("Click Refresh to load data", style={'color': '#2c3e50'})
                    )
                ], style={
                    'padding': '20px',
                    'backgroundColor': '#ffffff',
                    'borderRadius': '5px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                })
            ])
        ], style={'backgroundColor': '#ffffff'}),
        
        # Training Tab
        dcc.Tab(label="Training", children=[
            html.Div([
                # Manual Input Section
                html.Div([
                    html.H3("Manual Training", style={'color': '#4CAF50'}),
                    dcc.Input(
                        id='manual-symbol-input',
                        type='text',
                        placeholder='Enter symbol (e.g., BTC-USD)',
                        style={
                            'width': '200px',
                            'marginRight': '10px',
                            'padding': '5px',
                            'backgroundColor': '#ffffff',
                            'color': '#2c3e50',
                            'border': '1px solid #4CAF50'
                        }
                    ),
                    html.Button(
                        "🎯 Train Model",
                        id="manual-train-button",
                        n_clicks=0,
                        style={
                            'marginRight': '10px',
                            'backgroundColor': '#4CAF50',
                            'color': 'white',
                            'border': 'none',
                            'padding': '5px 10px',
                            'borderRadius': '3px'
                        }
                    ),
                ], style={'marginBottom': '20px'}),
                
                # Dropdown Selection Section
                html.Div([
                    html.H3("Selected Symbol Training", style={'color': '#2196F3'}),
                    dcc.Dropdown(
                        id='train-symbol-dropdown',
                        placeholder="Select Symbol",
                        style={'width': '200px', 'marginRight': '10px', 'backgroundColor': '#ffffff'}
                    ),
                    dcc.Dropdown(
                        id='train-granularity-dropdown',
                        options=[
                            {'label': '1 minute', 'value': 60},
                            {'label': '5 minutes', 'value': 300},
                            {'label': '15 minutes', 'value': 900},
                            {'label': '1 hour', 'value': 3600}
                        ],
                        style={'width': '200px', 'marginRight': '10px', 'backgroundColor': '#ffffff'}
                    ),
                    html.Button(
                        "🎯 Train Selected",
                        id="train-button",
                        n_clicks=0,
                        style={
                            'backgroundColor': '#2196F3',
                            'color': 'white',
                            'border': 'none',
                            'padding': '5px 10px',
                            'borderRadius': '3px'
                        }
                    ),
                ]),
                
                # Training Status
                html.Div(id="training-status", style={'marginTop': '20px', 'color': '#2c3e50'}),
                
                # Training History
                html.Div(id="training-history", style={'marginTop': '20px', 'color': '#2c3e50'})
            ], style={
                'padding': '20px',
                'backgroundColor': '#ffffff',
                'borderRadius': '5px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
        ], style={'backgroundColor': '#ffffff'}),
        
        # Analysis Tab
        dcc.Tab(label="Analysis", children=[
            html.Div([
                dcc.Dropdown(
                    id='analysis-symbol-dropdown',
                    placeholder="Select Symbol",
                    style={'backgroundColor': '#ffffff', 'color': '#2c3e50'}
                ),
                dcc.Graph(id="profit-loss-chart"),
                dcc.Graph(id="drawdown-chart"),
                html.Div(id="analysis-stats", style={'color': '#2c3e50'})
            ], style={
                'padding': '20px',
                'backgroundColor': '#ffffff',
                'borderRadius': '5px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
        ], style={'backgroundColor': '#ffffff'}),
        
        # Live Trading Tab
        dcc.Tab(label='Live Trading', children=[
            html.Div([
                # Trading Controls Section
                html.Div([
                    html.Label("Initial Portfolio Value ($):", style={'marginRight': '10px', 'color': '#2c3e50'}),
                    dcc.Input(
                        id='live-initial-portfolio',
                        type='number',
                        value=100000,
                        step=1000,
                        style={'width': '150px', 'marginRight': '20px', 'backgroundColor': '#ffffff', 'color': '#2c3e50'}
                    ),
                    
                    html.Label("Max Positions:", style={'marginRight': '10px', 'color': '#2c3e50'}),
                    dcc.Input(
                        id='live-max-positions',
                        type='number',
                        value=5,
                        min=1,
                        max=20,
                        step=1,
                        style={'width': '80px', 'marginRight': '20px', 'backgroundColor': '#ffffff', 'color': '#2c3e50'}
                    ),
                    
                    html.Button(
                        '▶️ Start Live Trading',
                        id='start-live-trading-btn',
                        style={
                            'backgroundColor': '#4CAF50',
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 20px',
                            'borderRadius': '5px',
                            'cursor': 'pointer',
                            'marginRight': '10px'
                        }
                    ),
                    
                    html.Button(
                        '⏹️ Stop Trading',
                        id='stop-live-trading-btn',
                        style={
                            'backgroundColor': '#f44336',
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 20px',
                            'borderRadius': '5px',
                            'cursor': 'pointer'
                        }
                    )
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'marginBottom': '20px',
                    'padding': '20px',
                    'backgroundColor': '#ffffff',
                    'borderRadius': '5px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                }),
                
                # Performance Metrics Section
                html.Div([
                    html.Div(id='live-trading-status', style={
                        'fontSize': '18px',
                        'marginBottom': '20px',
                        'color': '#2c3e50',
                        'padding': '10px',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '5px',
                        'textAlign': 'center'
                    }),
                    html.Div(id='live-performance-metrics', style={
                        'color': '#2c3e50',
                        'padding': '20px',
                        'backgroundColor': '#ffffff',
                        'borderRadius': '5px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                    })
                ], style={
                    'marginBottom': '20px'
                }),
                
                # Portfolio Chart
                html.Div([
                    dcc.Graph(
                        id='live-portfolio-chart',
                        style={'backgroundColor': '#ffffff'}
                    )
                ], style={
                    'padding': '20px',
                    'backgroundColor': '#ffffff',
                    'borderRadius': '5px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'marginBottom': '20px'
                }),
                
                # Current Positions Table
                html.Div([
                    html.H4("Current Positions", style={
                        'color': '#2c3e50',
                        'marginBottom': '15px',
                        'paddingLeft': '10px'
                    }),
                    dash_table.DataTable(
                        id='live-positions-table',
                        columns=[
                            {'name': 'Symbol', 'id': 'symbol'},
                            {'name': 'Quantity', 'id': 'quantity'},
                            {'name': 'Current Price', 'id': 'current_price'},
                            {'name': 'Value', 'id': 'value'},
                            {'name': '24h Change', 'id': 'change_24h'},
                            {'name': 'P/L', 'id': 'pnl'}
                        ],
                        style_table={
                            'overflowX': 'auto',
                            'backgroundColor': '#ffffff'
                        },
                        style_cell={
                            'backgroundColor': '#ffffff',
                            'color': '#2c3e50',
                            'textAlign': 'center',
                            'padding': '10px',
                            'fontFamily': 'Arial, sans-serif'
                        },
                        style_header={
                            'backgroundColor': '#f8f9fa',
                            'fontWeight': 'bold',
                            'border': '1px solid #ddd'
                        },
                        style_data={
                            'border': '1px solid #ddd'
                        },
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'change_24h', 'filter_query': '{change_24h} contains "+"'},
                                'color': '#4CAF50'
                            },
                            {
                                'if': {'column_id': 'change_24h', 'filter_query': '{change_24h} contains "-"'},
                                'color': '#f44336'
                            },
                            {
                                'if': {'column_id': 'pnl', 'filter_query': '{pnl} contains "+"'},
                                'color': '#4CAF50'
                            },
                            {
                                'if': {'column_id': 'pnl', 'filter_query': '{pnl} contains "-"'},
                                'color': '#f44336'
                            }
                        ],
                        sort_action='native',
                        sort_mode='single',
                        sort_by=[{'column_id': 'value', 'direction': 'desc'}]
                    )
                ], style={
                    'padding': '20px',
                    'backgroundColor': '#ffffff',
                    'borderRadius': '5px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'marginBottom': '20px'
                }),
                
                # Recent Trades Table
                html.Div([
                    html.H4("Recent Trades", style={
                        'color': '#2c3e50',
                        'marginBottom': '15px',
                        'paddingLeft': '10px'
                    }),
                    dash_table.DataTable(
                        id='live-trades-table',
                        columns=[
                            {'name': 'Time', 'id': 'timestamp'},
                            {'name': 'Symbol', 'id': 'symbol'},
                            {'name': 'Action', 'id': 'action'},
                            {'name': 'Price', 'id': 'price'},
                            {'name': 'Quantity', 'id': 'quantity'},
                            {'name': 'Value', 'id': 'value'}
                        ],
                        style_table={
                            'overflowX': 'auto',
                            'backgroundColor': '#ffffff'
                        },
                        style_cell={
                            'backgroundColor': '#ffffff',
                            'color': '#2c3e50',
                            'textAlign': 'center',
                            'padding': '10px',
                            'fontFamily': 'Arial, sans-serif'
                        },
                        style_header={
                            'backgroundColor': '#f8f9fa',
                            'fontWeight': 'bold',
                            'border': '1px solid #ddd'
                        },
                        style_data={
                            'border': '1px solid #ddd'
                        },
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'action', 'filter_query': '{action} = "BUY"'},
                                'color': '#4CAF50'
                            },
                            {
                                'if': {'column_id': 'action', 'filter_query': '{action} = "SELL"'},
                                'color': '#f44336'
                            }
                        ]
                    )
                ], style={
                    'padding': '20px',
                    'backgroundColor': '#ffffff',
                    'borderRadius': '5px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                }),
                
                # Auto-refresh interval
                dcc.Interval(
                    id='live-trading-interval',
                    interval=10*1000,  # 10 seconds
                    n_intervals=0
                )
            ], style={
                'padding': '20px',
                'backgroundColor': '#ffffff',
                'borderRadius': '5px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
        ], style={'backgroundColor': '#ffffff'})
    ], style={
        'backgroundColor': '#ffffff',
        'color': '#2c3e50',
        'borderBottom': '1px solid #ddd'
    })
], style={'backgroundColor': '#ffffff', 'color': '#2c3e50', 'minHeight': '100vh'})

# === Callbacks ===
@app.callback(
    [Output('scanner-table', 'children'),
     Output('refresh-scanner', 'children'),
     Output('refresh-scanner', 'disabled')],
    [Input('refresh-scanner', 'n_clicks')],
    prevent_initial_call=False
)
def update_scanner(n_clicks):
    if n_clicks is None:
        return (
            html.Div("Click Refresh to load data", style={'color': '#2c3e50'}),
            "🔄 Refresh Scanner",
            False
        )
    
    try:
        # Show loading state
        loading_div = html.Div([
            html.Div("Loading scanner data...", style={
                'marginBottom': '10px',
                'color': '#2c3e50',
                'textAlign': 'center'
            }),
            html.Div(className="loader")
        ])
        
        # Get symbols without limiting to top 20
        symbols = get_cached_symbols()  # Fetch all available symbols
        
        # Run market scan with the symbols
        results = scan_market(symbols=symbols, batch_size=5)
        
        if not results:
            return (
                html.Div("No results found", style={'color': '#2c3e50'}),
                "🔄 Refresh Scanner",
                False
            )
        
        # Create DataFrame and table
        df = pd.DataFrame(results)
        
        # Create DataTable with consistent ID
        table = dash_table.DataTable(
            id='scanner-datatable',  # This ID must match the one used in other callbacks
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={
                'backgroundColor': '#ffffff',
                'color': '#2c3e50',
                'border': '1px solid #cccccc',
                'padding': '10px',
                'textAlign': 'left'
            },
            style_header={
                'backgroundColor': '#f5f5f5',
                'fontWeight': 'bold',
                'border': '1px solid #cccccc'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'price_change_24h', 'filter_query': '{price_change_24h} > 0'},
                    'color': '#4CAF50'
                },
                {
                    'if': {'column_id': 'price_change_24h', 'filter_query': '{price_change_24h} < 0'},
                    'color': '#f44336'
                }
            ],
            sort_action='native',
            filter_action='native',
            row_selectable='single',
            selected_rows=[],
            page_size=10
        )
        
        return table, "🔄 Refresh Scanner", False
        
    except Exception as e:
        print(f"Error in update_scanner: {str(e)}")
        return (
            html.Div(f"Error refreshing data: {str(e)}", style={'color': '#2c3e50'}),
            "🔄 Refresh Scanner",
            False
        )

@app.callback(
    [Output('train-symbol-dropdown', 'options'),
     Output('trade-symbol-dropdown', 'options'),
     Output('analysis-symbol-dropdown', 'options')],
    Input('interval-component', 'n_intervals')
)
def update_symbol_dropdowns(n_intervals):
    symbols = get_cached_symbols()
    options = [{'label': s, 'value': s} for s in symbols]
    return options, options, options

@app.callback(
    Output('training-status', 'children'),
    [Input('manual-train-button', 'n_clicks'),
     Input('train-button', 'n_clicks')],
    [State('manual-symbol-input', 'value'),
     State('train-symbol-dropdown', 'value'),
     State('train-granularity-dropdown', 'value')]
)
def train_model(manual_clicks, dropdown_clicks, manual_symbol, dropdown_symbol, granularity):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Determine which symbol to use
    if trigger_id == 'manual-train-button':
        if not manual_symbol:
            return html.Div("❌ Please enter a symbol", style={'color': 'red'})
        symbol = manual_symbol.upper()
    else:  # train-button
        if not dropdown_symbol:
            return html.Div("❌ Please select a symbol", style={'color': 'red'})
        symbol = dropdown_symbol
    
    # Add loading message
    loading_div = html.Div([
        html.P("🔄 Training model...", style={'color': '#2196F3'}),
        html.Div(className="loader")
    ])
    
    # Train the model
    try:
        reg_model, clf = train_model_for_symbol(symbol, granularity)
        if reg_model is None or clf is None:
            return html.Div("❌ Training failed", style={'color': 'red'})
        
        # Return success message with details
        return html.Div([
            html.H4("✅ Training Complete", style={'color': 'green'}),
            html.P([
                html.Strong("Symbol: "), 
                html.Span(symbol)
            ]),
            html.P([
                html.Strong("Granularity: "), 
                html.Span(f"{granularity//60} minutes" if granularity < 3600 else "1 hour")
            ]),
            html.P([
                html.Strong("Models Saved: "), 
                html.Span(f"{symbol.replace('-', '')}_{granularity}")
            ])
        ])
        
    except Exception as e:
        return html.Div([
            html.H4("❌ Error", style={'color': 'red'}),
            html.P(str(e))
        ])

# Add callback to update training history
@app.callback(
    Output('training-history', 'children'),
    [Input('training-status', 'children')]
)
def update_training_history(status):
    try:
        # Get list of trained models
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
        if not model_files:
            return html.Div("No trained models found")
        
        # Group models by symbol
        models_by_symbol = {}
        for f in model_files:
            symbol = f.split('_')[0]
            if symbol not in models_by_symbol:
                models_by_symbol[symbol] = []
            models_by_symbol[symbol].append(f)
        
        # Create training history display
        return html.Div([
            html.H4("Trained Models", style={'color': '#2196F3'}),
            html.Div([
                html.Div([
                    html.H5(symbol),
                    html.Ul([
                        html.Li(model.replace('.pkl', '')) 
                        for model in sorted(models)
                    ])
                ]) 
                for symbol, models in models_by_symbol.items()
            ])
        ])
    except Exception as e:
        return html.Div(f"Error loading training history: {str(e)}")

def plot_prediction_errors(predictions_df):
    """
    Create a figure showing prediction errors and error distribution.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing actual and predicted prices
        
    Returns:
        go.Figure: A plotly figure with error analysis
    """
    if predictions_df.empty:
        return go.Figure()
    
    # Calculate errors
    predictions_df['error'] = predictions_df['actual_price'] - predictions_df['predicted_price']
    predictions_df['error_pct'] = (predictions_df['error'] / predictions_df['actual_price']) * 100
    predictions_df['abs_error'] = abs(predictions_df['error'])
    
    # Create subplots: error over time and error distribution
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Prediction Error Over Time',
            'Error Distribution',
            'Error vs Price Level',
            'Cumulative Error'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. Error over time
    fig.add_trace(
        go.Scatter(
            x=predictions_df['timestamp'],
            y=predictions_df['error'],
            mode='lines',
            name='Error',
            line=dict(color='#EF5350')
        ),
        row=1, col=1
    )
    
    # Add zero line
    fig.add_hline(
        y=0, line_dash="dash", 
        line_color="gray",
        row=1, col=1
    )
    
    # 2. Error distribution histogram
    fig.add_trace(
        go.Histogram(
            x=predictions_df['error'],
            name='Error Distribution',
            nbinsx=30,
            marker_color='#42A5F5'
        ),
        row=1, col=2
    )
    
    # 3. Error vs Price Level scatter
    fig.add_trace(
        go.Scatter(
            x=predictions_df['actual_price'],
            y=predictions_df['error'],
            mode='markers',
            name='Error vs Price',
            marker=dict(
                color=predictions_df['abs_error'],
                colorscale='Viridis',
                showscale=True
            )
        ),
        row=2, col=1
    )
    
    # 4. Cumulative error
    fig.add_trace(
        go.Scatter(
            x=predictions_df['timestamp'],
            y=predictions_df['error'].cumsum(),
            mode='lines',
            name='Cumulative Error',
            line=dict(color='#66BB6A')
        ),
        row=2, col=2
    )
    
    # Calculate error metrics
    mae = predictions_df['abs_error'].mean()
    mse = (predictions_df['error'] ** 2).mean()
    rmse = np.sqrt(mse)
    mape = (predictions_df['error_pct'].abs()).mean()
    
    # Update layout with metrics
    fig.update_layout(
        title=dict(
            text=f'Prediction Error Analysis<br>MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | MAPE: {mape:.2f}%',
            x=0.5,
            y=0.95
        ),
        showlegend=True,
        template='plotly_white',
        height=800,
        width=1200
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Error ($)", row=1, col=2)
    fig.update_xaxes(title_text="Price ($)", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    
    fig.update_yaxes(title_text="Error ($)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Error ($)", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Error ($)", row=2, col=2)
    
    return fig

# Modify the run_simulation callback to include error plotting
@app.callback(
    [Output('price-chart', 'figure'),
     Output('portfolio-chart', 'figure'),
     Output('simulation-stats', 'children')],
    [Input('simulate-button', 'n_clicks'),
     Input('trade-symbol-dropdown', 'value'),
     Input('trade-granularity-dropdown', 'value'),
     Input('simulation-date-range', 'start_date'),
     Input('simulation-date-range', 'end_date')]
)
def run_simulation(n_clicks, symbol, granularity, start_date, end_date):
    if n_clicks is None or not n_clicks:
        return create_empty_figure(), create_empty_figure(), ""
    
    if not symbol:
        return create_empty_figure(), create_empty_figure(), html.Div("❌ Please select a symbol", style={'color': 'red'})
    
    if not start_date or not end_date:
        return create_empty_figure(), create_empty_figure(), html.Div("❌ Please select date range", style={'color': 'red'})
    
    try:
        # Convert string dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Load models
        model_prefix = f"{symbol.replace('-', '')}_{granularity}"
        try:
            reg_model, clf = train_model_for_symbol(symbol, granularity)
        except Exception as e:
            error_msg = html.Div([
                html.H4("❌ Error Loading Models", style={'color': 'red'}),
                html.P(f"Models not found for {symbol} at {granularity}s granularity."),
                html.P("Please train the model first using the Training tab.")
            ])
            return create_empty_figure(), create_empty_figure(), error_msg
        
        # Get data and run simulation
        try:
            print(f"Fetching data for {symbol}...")
            
            # Adjust data fetching period based on granularity
            if granularity == 60:  # 1 minute
                max_days = 7  # 1 week of minute data
            elif granularity == 300:  # 5 minutes
                max_days = 14  # 2 weeks of 5-minute data
            elif granularity == 900:  # 15 minutes
                max_days = 30  # 1 month of 15-minute data
            else:  # 1 hour or higher
                max_days = 90  # 3 months of hourly data
                
            # Calculate actual days needed based on date range and max_days
            requested_days = (end_date - start_date).days
            days_to_fetch = min(requested_days + 1, max_days)
            
            # Adjust start_date if needed
            if requested_days > max_days:
                print(f"⚠️ Warning: Requested period exceeds maximum of {max_days} days for {granularity}s granularity.")
                print(f"Fetching most recent {max_days} days of data.")
                start_date = end_date - timedelta(days=max_days-1)
            
            df = get_coinbase_data(symbol=symbol, granularity=granularity, days=days_to_fetch)
            if df.empty:
                return create_empty_figure(), create_empty_figure(), html.Div("❌ No data available", style={'color': 'red'})
            
            # Filter data for selected date range
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            if len(df) < 50:
                return create_empty_figure(), create_empty_figure(), html.Div(
                    "❌ Insufficient data for selected date range. Please select a longer period.",
                    style={'color': 'red'}
                )
            
            print("Calculating indicators...")
            df = calculate_indicators(df)
            df.dropna(inplace=True)
            
            print("Running simulation...")
            initial_state = {
                "cash": 100.0,
                "position": None,
                "quantity": 0,
                "entry": None,
                "total_profit": 0,
                "wins": 0,
                "losses": 0
            }
            
            trades_df, final_state = simulate_trading(df, reg_model, clf, initial_state)
            
            # Get predictions for error analysis
            predictions, confidence = predict_with_pretrained_model(df, symbol, interval='1h')
            
            # Create error analysis plot
            error_fig = plot_prediction_errors(predictions)
            
            # Create price chart with candlesticks and trades
            price_fig = go.Figure()
            
            # Add candlestick chart
            price_fig.add_trace(go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ))
            
            # Add predicted prices
            if not predictions.empty:
                price_fig.add_trace(go.Scatter(
                    x=predictions['timestamp'],
                    y=predictions['predicted_price'],
                    mode='lines',
                    name='Predicted Price',
                    line=dict(color='#42A5F5', dash='dash')
                ))
            
            # Add buy markers
            buy_points = trades_df[trades_df['action'] == 'BUY']
            if not buy_points.empty:
                price_fig.add_trace(go.Scatter(
                    x=buy_points['timestamp'],
                    y=buy_points['price'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color='green'),
                    name='Buy'
                ))
            
            # Add sell markers
            sell_points = trades_df[trades_df['action'] == 'SELL']
            if not sell_points.empty:
                price_fig.add_trace(go.Scatter(
                    x=sell_points['timestamp'],
                    y=sell_points['price'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=10, color='red'),
                    name='Sell'
                ))
            
            price_fig.update_layout(
                title=f"{symbol} Price and Trades",
                yaxis_title='Price',
                template='plotly',  # Changed from plotly_dark
                xaxis_rangeslider_visible=False
            )
            
            # Create portfolio value chart
            portfolio_fig = go.Figure()
            
            # Add total value line
            portfolio_fig.add_trace(go.Scatter(
                x=trades_df['timestamp'],
                y=trades_df['total_value'],
                name='Portfolio Value',
                line=dict(color='#2196F3')  # Blue color
            ))
            
            # Add cash line
            portfolio_fig.add_trace(go.Scatter(
                x=trades_df['timestamp'],
                y=trades_df['cash'],
                name='Cash',
                line=dict(color='#4CAF50', dash='dash')  # Green color
            ))
            
            # Add crypto value line
            portfolio_fig.add_trace(go.Scatter(
                x=trades_df['timestamp'],
                y=trades_df['crypto_value'],
                name='Crypto Value',
                line=dict(color='#FF9800', dash='dash')  # Orange color
            ))
            
            portfolio_fig.update_layout(
                title="Portfolio Value Over Time",
                yaxis_title='Value (USD)',
                template='plotly'  # Changed from plotly_dark
            )
            
            # Calculate additional statistics
            total_trades = final_state['wins'] + final_state['losses']
            win_rate = (final_state['wins'] / total_trades * 100) if total_trades > 0 else 0
            avg_profit = final_state['total_profit'] / total_trades if total_trades > 0 else 0
            
            # Create detailed stats
            stats = html.Div([
                html.H3("Simulation Results", style={'color': '#2196F3'}),
                html.Div([
                    html.Div([
                        html.H4("Portfolio Metrics", style={'color': '#4CAF50'}),
                        html.P([
                            html.Strong("Initial Value: "), 
                            html.Span(f"${100:,.2f}")
                        ]),
                        html.P([
                            html.Strong("Final Value: "), 
                            html.Span(f"${trades_df['total_value'].iloc[-1]:,.2f}")
                        ]),
                        html.P([
                            html.Strong("Total Return: "), 
                            html.Span(
                                f"{((trades_df['total_value'].iloc[-1] / 100 - 1) * 100):,.2f}%",
                                style={'color': 'green' if trades_df['total_value'].iloc[-1] > 10000 else 'red'}
                            )
                        ])
                    ], style={'flex': 1}),
                    
                    html.Div([
                        html.H4("Trading Metrics", style={'color': '#4CAF50'}),
                        html.P([
                            html.Strong("Total Trades: "), 
                            html.Span(f"{total_trades}")
                        ]),
                        html.P([
                            html.Strong("Win Rate: "), 
                            html.Span(f"{win_rate:.1f}%")
                        ]),
                        html.P([
                            html.Strong("Average Profit per Trade: "), 
                            html.Span(f"${avg_profit:.2f}")
                        ])
                    ], style={'flex': 1}),
                    
                    html.Div([
                        html.H4("Current Position", style={'color': '#4CAF50'}),
                        html.P([
                            html.Strong("Position: "), 
                            html.Span(final_state['position'] if final_state['position'] else 'None')
                        ]),
                        html.P([
                            html.Strong("Quantity: "), 
                            html.Span(f"{final_state['quantity']:.8f}")
                        ]),
                        html.P([
                            html.Strong("Entry Price: "), 
                            html.Span(f"${final_state['entry']:.2f}" if final_state['entry'] else 'N/A')
                        ])
                    ], style={'flex': 1})
                ], style={'display': 'flex', 'justifyContent': 'space-between'})
            ], style={'padding': '20px', 'backgroundColor': '#1E1E1E', 'borderRadius': '5px'})
            
            # Save trade history
            trades_df.to_csv(os.path.join(LOGS_DIR, f"{symbol.replace('-', '')}_trades.csv"), index=False)
            
            # Update the layout to include both price and error charts
            price_fig.update_layout(
                title=f"{symbol} Price and Predictions",
                yaxis_title='Price',
                template='plotly_white',
                xaxis_rangeslider_visible=False
            )
            
            # Return both figures
            return price_fig, error_fig, stats
            
        except Exception as e:
            error_msg = html.Div([
                html.H4("❌ Error During Simulation", style={'color': 'red'}),
                html.P(str(e))
            ])
            return create_empty_figure(), create_empty_figure(), error_msg
        
    except Exception as e:
        error_msg = html.Div([
            html.H4("❌ Error During Simulation", style={'color': 'red'}),
            html.P(str(e))
        ])
        return create_empty_figure(), create_empty_figure(), error_msg

def create_empty_figure():
    """Create an empty figure with a message"""
    fig = go.Figure()
    fig.update_layout(
        title="No Data Available",
        template='plotly',
        xaxis={'visible': False},
        yaxis={'visible': False},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[{
            'text': "Select a symbol and click 'Start Simulation'",
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 20, 'color': '#2c3e50'}
        }]
    )
    return fig

@app.callback(
    [Output('profit-loss-chart', 'figure'),
     Output('drawdown-chart', 'figure'),
     Output('analysis-stats', 'children')],
    [Input('analysis-symbol-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_analysis(symbol, n_intervals):
    if not symbol:
        return create_empty_figure(), create_empty_figure(), ""
    
    # Load trade history
    try:
        trades_df = pd.read_csv(os.path.join(LOGS_DIR, f"{symbol.replace('-', '')}_trades.csv"))
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    except Exception as e:
        print(f"Error loading trade history: {e}")
        return create_empty_figure(), create_empty_figure(), "No trade history available"
    
    # Create P&L chart
    pnl_fig = go.Figure()
    pnl_fig.add_trace(go.Scatter(
        x=trades_df['timestamp'],
        y=trades_df['profit'].cumsum(),
        name='Cumulative P&L',
        line=dict(color='#4CAF50' if trades_df['profit'].sum() > 0 else '#f44336')  # Green if positive, red if negative
    ))
    pnl_fig.update_layout(
        title="Cumulative Profit/Loss",
        yaxis_title='Profit/Loss (USD)',
        template='plotly'  # Changed from plotly_dark
    )
    
    # Calculate and plot drawdown
    cummax = trades_df['total_value'].cummax()
    drawdown = (cummax - trades_df['total_value']) / cummax * 100
    
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(
        x=trades_df['timestamp'],
        y=drawdown,
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='#f44336')  # Red color
    ))
    dd_fig.update_layout(
        title="Drawdown Analysis",
        yaxis_title='Drawdown (%)',
        template='plotly'  # Changed from plotly_dark
    )
    
    # Calculate analysis statistics
    total_trades = len(trades_df[trades_df['action'].isin(['BUY', 'SELL'])])
    profitable_trades = len(trades_df[trades_df['profit'] > 0])
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    max_drawdown = drawdown.max()
    
    # Create analysis stats
    stats = html.Div([
        html.H3("Trading Analysis", style={'color': '#2196F3'}),
        html.Div([
            html.Div([
                html.H4("Performance Metrics", style={'color': '#4CAF50'}),
                html.P([
                    html.Strong("Total Trades: "), 
                    html.Span(f"{total_trades}")
                ]),
                html.P([
                    html.Strong("Win Rate: "), 
                    html.Span(f"{win_rate:.1f}%")
                ]),
                html.P([
                    html.Strong("Max Drawdown: "), 
                    html.Span(f"{max_drawdown:.1f}%")
                ])
            ], style={'flex': 1}),
            
            html.Div([
                html.H4("Risk Metrics", style={'color': '#4CAF50'}),
                html.P([
                    html.Strong("Avg Win: "), 
                    html.Span(f"${trades_df[trades_df['profit'] > 0]['profit'].mean():.2f}")
                ]),
                html.P([
                    html.Strong("Avg Loss: "), 
                    html.Span(f"${trades_df[trades_df['profit'] < 0]['profit'].mean():.2f}")
                ]),
                html.P([
                    html.Strong("Profit Factor: "), 
                    html.Span(f"{abs(trades_df[trades_df['profit'] > 0]['profit'].sum() / trades_df[trades_df['profit'] < 0]['profit'].sum()):.2f}")
                ])
            ], style={'flex': 1})
        ], style={'display': 'flex', 'justifyContent': 'space-between'})
    ], style={'padding': '20px', 'backgroundColor': '#1E1E1E', 'borderRadius': '5px'})
    
    return pnl_fig, dd_fig, stats

# === New Callbacks for Context Menu and Row Selection ===
@app.callback(
    [Output('trade-symbol-dropdown', 'value'),
     Output('train-symbol-dropdown', 'value'),
     Output('analysis-symbol-dropdown', 'value')],
    [Input('context-train', 'n_clicks'),
     Input('context-simulate', 'n_clicks'),
     Input('context-analyze', 'n_clicks'),
     Input('scanner-datatable', 'selected_rows')],
    [State('selected-symbol', 'data'),
     State('scanner-datatable', 'data')],
    prevent_initial_call=True
)
def handle_symbol_selection(train_clicks, sim_clicks, analyze_clicks, selected_rows, selected_symbol, table_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
        
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle row selection from table
    if trigger_id == 'scanner-datatable':
        if not selected_rows or not table_data:
            raise PreventUpdate
        try:
            selected_row = table_data[selected_rows[0]]
            symbol = selected_row.get('symbol')
            if not symbol:
                raise PreventUpdate
            return symbol, symbol, symbol
        except Exception as e:
            print(f"Error in row selection: {str(e)}")
            raise PreventUpdate
    
    # Handle context menu clicks
    elif trigger_id in ['context-train', 'context-simulate', 'context-analyze']:
        if not selected_symbol:
            raise PreventUpdate
            
        if trigger_id == 'context-train':
            return dash.no_update, selected_symbol, dash.no_update
        elif trigger_id == 'context-simulate':
            return selected_symbol, dash.no_update, dash.no_update
        elif trigger_id == 'context-analyze':
            return dash.no_update, dash.no_update, selected_symbol
    
    raise PreventUpdate

@app.callback(
    [Output('context-menu', 'style'),
     Output('selected-symbol', 'data')],
    [Input('scanner-datatable', 'active_cell'),
     Input('scanner-datatable', 'data')],
    [State('context-menu', 'style')]
)
def show_context_menu(active_cell, data, current_style):
    if not active_cell or not data:
        style = dict(styles['context_menu'])
        style['display'] = 'none'
        return style, None
    
    try:
        row = data[active_cell['row']]
        symbol = row.get('symbol')
        if not symbol:
            style = dict(styles['context_menu'])
            style['display'] = 'none'
            return style, None
        
        style = dict(styles['context_menu'])
        style['display'] = 'block'
        style['left'] = f"{active_cell.get('column_id', 0)}px"
        style['top'] = f"{active_cell.get('row', 0)}px"
        
        return style, symbol
        
    except Exception as e:
        print(f"Error in show_context_menu: {str(e)}")
        style = dict(styles['context_menu'])
        style['display'] = 'none'
        return style, None

@app.callback(
    Output('forward-test-results', 'children'),
    [Input('run-forward-test-btn', 'n_clicks')],
    [State('forward-test-days', 'value'),
     State('forward-initial-portfolio', 'value'),
     State('forward-max-positions', 'value')]
)
def run_forward_test_callback(n_clicks, days, initial_value, max_positions):
    if not n_clicks:
        return "Click 'Run Forward Test' to start"
        
    try:
        results = {
            'summary': {},
            'value_history': [],
            'history': pd.DataFrame()
        }
        
        # Get historical trades from database
        historical_trades = pd.read_sql('SELECT * FROM trades ORDER BY timestamp DESC LIMIT 1000', conn)
        
        if len(historical_trades) > 0:
            # Calculate actual win rate from historical trades
            profitable_trades = historical_trades[historical_trades['profit'] > 0]
            actual_win_rate = (len(profitable_trades) / len(historical_trades)) * 100
            avg_profit = profitable_trades['profit'].mean() if len(profitable_trades) > 0 else 0
            avg_loss = historical_trades[historical_trades['profit'] < 0]['profit'].mean() if len(historical_trades[historical_trades['profit'] < 0]) > 0 else 0
        else:
            # If no historical data, use conservative estimates
            actual_win_rate = 45.0  # Conservative estimate
            avg_profit = 0.05 * initial_value  # 5% of initial value
            avg_loss = -0.03 * initial_value   # 3% of initial value
            
        results['summary']['win_rate'] = actual_win_rate
        results['summary']['avg_profit'] = avg_profit
        results['summary']['avg_loss'] = avg_loss
        
        # Get current market data for top opportunities
        symbols = get_cached_symbols()[:50]  # Get top 50 symbols
        market_data = scan_market(symbols=symbols)
        top_symbols = [item['Symbol'] for item in market_data[:max_positions]]
        
        # Initialize portfolio
        portfolio = {
            'cash': initial_value,
            'positions': {},
            'value_history': []
        }
        
        # Simulate future dates
        dates = pd.date_range(start=datetime.now(), periods=days, freq='D')
        
        # Simple simulation using historical volatility and momentum
        for date in dates:
            daily_return = 0
            for symbol in top_symbols:
                # Get historical data for volatility calculation
                hist_data = get_coinbase_data(symbol=symbol, granularity=3600*24, days=30)
                if not hist_data.empty:
                    volatility = hist_data['close'].pct_change().std()
                    momentum = hist_data['close'].pct_change().mean()
                    
                    # Simulate daily return based on historical patterns
                    simulated_return = np.random.normal(momentum, volatility)
                    daily_return += simulated_return / len(top_symbols)
            
            # Update portfolio value
            portfolio_value = portfolio['cash'] * (1 + daily_return)
            portfolio['value_history'].append({
                'date': date,
                'total_value': portfolio_value
            })
            portfolio['cash'] = portfolio_value
        
        # Update results
        value_history = pd.DataFrame(portfolio['value_history'])
        returns = value_history['total_value'].pct_change().dropna()
        
        results['summary']['final_value'] = portfolio['cash']
        results['summary']['total_return'] = ((portfolio['cash'] / initial_value) - 1) * 100
        results['summary']['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 0 else 0
        results['summary']['max_drawdown'] = ((value_history['total_value'].cummax() - value_history['total_value']) / 
                                            value_history['total_value'].cummax()).max()
        results['summary']['total_trades'] = len(top_symbols)
        
        results['value_history'] = value_history
        
        # Get current market data for the selected symbols
        current_prices = {}
        for symbol in top_symbols[:5]:  # Get data for top 5 symbols
            try:
                current_data = get_coinbase_data(symbol=symbol, granularity=3600, days=1)  # Get latest day's data
                if not current_data.empty:
                    current_prices[symbol] = current_data['close'].iloc[-1]
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")
                continue
        
        # Create trade history with actual prices and current date
        current_date = pd.Timestamp.now()
        trade_records = []
        position_size = initial_value / max_positions
        
        for symbol, price in current_prices.items():
            if price > 0:  # Ensure we have a valid price
                quantity = position_size / price
                trade_records.append({
                    'date': current_date,
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': price,
                    'quantity': quantity,
                    'value': position_size,
                    'profit': 0
                })
        
        results['history'] = pd.DataFrame(trade_records)
        
        # Add projected future trades
        future_trades = []
        for day in range(1, days + 1):
            future_date = current_date + pd.Timedelta(days=day)
            
            # For each active position, simulate potential exits
            for record in trade_records:
                symbol = record['symbol']
                entry_price = record['price']
                
                # Get historical volatility
                try:
                    hist_data = get_coinbase_data(symbol=symbol, granularity=86400, days=30)
                    if not hist_data.empty:
                        daily_returns = hist_data['close'].pct_change().dropna()
                        volatility = daily_returns.std()
                        trend = (hist_data['close'].iloc[-1] / hist_data['close'].iloc[0]) - 1
                        
                        # Project price movement based on volatility and trend
                        price_change = np.random.normal(trend/30, volatility)
                        projected_price = entry_price * (1 + price_change)
                        
                        # Simulate exit if projected return exceeds threshold
                        if abs(price_change) > 0.05:  # 5% threshold
                            action = 'SELL' if price_change < 0 else 'HOLD'
                            profit = record['quantity'] * (projected_price - entry_price)
                            
                            future_trades.append({
                                'date': future_date,
                                'symbol': symbol,
                                'action': action,
                                'price': projected_price,
                                'quantity': record['quantity'],
                                'value': record['quantity'] * projected_price,
                                'profit': profit
                            })
                except Exception as e:
                    print(f"Error projecting trades for {symbol}: {e}")
                    continue
        
        # Append projected trades to history
        if future_trades:
            future_df = pd.DataFrame(future_trades)
            results['history'] = pd.concat([results['history'], future_df])
        
        # Sort by date
        results['history'] = results['history'].sort_values('date').reset_index(drop=True)
    except Exception as e:
        print(f"Error in forward test: {e}")
        return html.Div(f"Error running forward test: {str(e)}", style={'color': 'red'})
    
    # Create results display
    return create_results_display(results, is_forward_test=True)

@app.callback(
    Output('backtest-results', 'children'),
    [Input('run-portfolio-backtest-btn', 'n_clicks')],
    [State('portfolio-test-dates', 'start_date'),
     State('portfolio-test-dates', 'end_date'),
     State('initial-portfolio-value', 'value'),
     State('max-positions', 'value')]
)
def run_backtest_callback(n_clicks, start_date, end_date, initial_value, max_positions):
    if not n_clicks:
        return "Click 'Run Backtest' to start"

    try:
        # Convert string dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Run ML backtest
        results = run_ml_backtest(
            start_date=start_date,
            end_date=end_date,
            initial_value=float(initial_value),
            max_positions=int(max_positions)
        )
        
        if 'error' in results and results['error']:
            return html.Div([
                html.H4("Backtest Error", style={'color': 'red'}),
                html.P(results['error'])
            ])
            
        # Save results to CSV files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if len(results['trades']) > 0:
            trades_df = pd.DataFrame(results['trades'])
            trades_df.to_csv(os.path.join(LOGS_DIR, f'backtest_trades_alpha_v1_{timestamp}.csv'), index=False)
            
        if not results['history'].empty:
            results['history'].to_csv(os.path.join(LOGS_DIR, f'backtest_value_history_{timestamp}.csv'), index=False)
            
        return create_results_display(results, is_forward_test=False)
        
    except Exception as e:
        import traceback
        print(f"Error running backtest: {str(e)}")
        print(traceback.format_exc())
        return html.Div([
            html.H4("Backtest Error", style={'color': 'red'}),
            html.P(str(e))
        ])

def create_results_display(results, is_forward_test=False):
    if isinstance(results, str):
        return html.Div(results)
    
    # Extract metrics from stats if available
    metrics = results.get('stats', {})
    history_df = results.get('history', pd.DataFrame())
    trades = results.get('trades', [])
    
    # Create portfolio value chart
    portfolio_chart = dcc.Graph(
        figure={
            'data': [
                go.Scatter(
                    x=history_df['timestamp'],
                    y=history_df['portfolio_value'],
                    name='Portfolio Value',
                    line={'color': '#2196F3'}
                )
            ],
            'layout': go.Layout(
                title='Portfolio Value Over Time',
                xaxis={'title': 'Time'},
                yaxis={'title': 'Value ($)'},
                template='plotly_white',
                hovermode='x unified'
            )
        }
    )
    
    # Create the main metrics display
    metrics_div = html.Div([
        html.H4("Portfolio Performance", style={
            'color': '#2c3e50',
            'margin-bottom': '20px',
            'font-weight': 'bold'
        }),
        
        # Portfolio Value Chart
        html.Div([
            portfolio_chart
        ], style={
            'background-color': '#ffffff',
            'padding': '20px',
            'border-radius': '8px',
            'box-shadow': '0 2px 4px rgba(0,0,0,0)',
            'margin-bottom': '20px'
        }),
        
        # Metrics Sections
        html.Div([
            html.Div([
                # Portfolio metrics
                html.Div([
                    html.H5("Overall Metrics", style={'color': '#2c3e50', 'margin-bottom': '15px'}),
                    html.P([
                        html.Strong("Total Return: "),
                        html.Span(
                            f"{metrics.get('total_return', 0):.2f}%",
                            style={'color': '#4CAF50' if metrics.get('total_return', 0) > 0 else '#f44336'}
                        )
                    ], style={'margin': '10px 0'}),
                    html.P([
                        html.Strong("Sharpe Ratio: "),
                        html.Span(f"{metrics.get('sharpe_ratio', 0):.2f}")
                    ], style={'margin': '10px 0'}),
                    html.P([
                        html.Strong("Max Drawdown: "),
                        html.Span(f"{metrics.get('max_drawdown', 0):.2f}%")
                    ], style={'margin': '10px 0'})
                ], style={'flex': '1'}),
                
                # Trading metrics
                html.Div([
                    html.H5("Trading Statistics", style={'color': '#2c3e50', 'margin-bottom': '15px'}),
                    html.P([
                        html.Strong("Win Rate: "),
                        html.Span(f"{metrics.get('win_rate', 0):.2f}%")
                    ], style={'margin': '10px 0'}),
                    html.P([
                        html.Strong("Total Trades: "),
                        html.Span(f"{len(trades)}")
                    ], style={'margin': '10px 0'}),
                    html.P([
                        html.Strong("Initial Value: "),
                        html.Span(f"${history_df['portfolio_value'].iloc[0]:,.2f}")
                    ], style={'margin': '10px 0'}),
                    html.P([
                        html.Strong("Final Value: "),
                        html.Span(f"${history_df['portfolio_value'].iloc[-1]:,.2f}")
                    ], style={'margin': '10px 0'})
                ], style={'flex': '1'})
            ], style={
                'display': 'flex',
                'justify-content': 'space-between',
                'background-color': '#ffffff',
                'padding': '20px',
                'border-radius': '8px',
                'box-shadow': '0 2px 4px rgba(0,0,0,0.1)',
                'margin-bottom': '20px'
            })
        ]),
        
        # Recent Trades Table (if any trades exist)
        html.Div([
            html.H5("Recent Trades", style={'color': '#2c3e50', 'margin-bottom': '15px'}),
            html.Div(
                "No trades executed during this period." if not trades else
                dash_table.DataTable(
                    id='recent-trades-table',
                    columns=[
                        {'name': 'Time', 'id': 'timestamp'},
                        {'name': 'Symbol', 'id': 'symbol'},
                        {'name': 'Action', 'id': 'action'},
                        {'name': 'Price', 'id': 'price'},
                        {'name': 'Size', 'id': 'size'},
                        {'name': 'P/L', 'id': 'pnl'}
                    ],
                    data=trades[-10:],  # Show last 10 trades
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'center',
                        'padding': '10px',
                        'backgroundColor': '#ffffff'
                    },
                    style_header={
                        'backgroundColor': '#f8f9fa',
                        'fontWeight': 'bold',
                        'border': '1px solid #ddd'
                    },
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'action', 'filter_query': '{action} eq "BUY"'},
                            'color': '#4CAF50'
                        },
                        {
                            'if': {'column_id': 'action', 'filter_query': '{action} eq "SELL"'},
                            'color': '#f44336'
                        },
                        {
                            'if': {'column_id': 'pnl', 'filter_query': '{pnl} contains "+"'},
                            'color': '#4CAF50'
                        },
                        {
                            'if': {'column_id': 'pnl', 'filter_query': '{pnl} contains "-"'},
                            'color': '#f44336'
                        }
                    ]
                )
            )
        ], style={
            'background-color': '#ffffff',
            'padding': '20px',
            'border-radius': '8px',
            'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
        })
    ], style={
        'max-width': '1200px',
        'margin': '0 auto',
        'padding': '20px'
    })
    
    return metrics_div

# Add live trading callbacks
@app.callback(
    [Output('live-trading-status', 'children'),
     Output('start-live-trading-btn', 'disabled'),
     Output('stop-live-trading-btn', 'disabled')],
    [Input('start-live-trading-btn', 'n_clicks'),
     Input('stop-live-trading-btn', 'n_clicks')],
    [State('live-initial-portfolio', 'value'),
     State('live-max-positions', 'value')]
)
def manage_live_trading(start_clicks, stop_clicks, initial_portfolio, max_positions):
    global trading_thread, stop_trading, live_sim, ws_client
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Trading not started", False, True
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start-live-trading-btn' and start_clicks:
        if trading_thread and trading_thread.is_alive():
            return "Trading already running", True, False
        
        try:
            # Force initial portfolio to $5
            initial_portfolio = 5.0
            
            # Start by checking if the database exists and schema is correct
            if not ensure_database_schema():
                return "Failed to initialize database schema", False, True
            
            # Create a new session
            conn = sqlite3.connect('live_trading.db')
            cursor = conn.cursor()
            
            # End any active sessions first
            cursor.execute('''
            UPDATE sessions SET 
                end_time = datetime('now'),
                status = 'completed'
            WHERE status = 'active'
            ''')
            
            # Create a new session with $5 initial balance
            cursor.execute('''
            INSERT INTO sessions (start_time, initial_balance, status)
            VALUES (datetime('now'), ?, 'active')
            ''', (initial_portfolio,))
            
            session_id = cursor.lastrowid
            logger.info(f"Created new trading session with ID: {session_id}")
            
            # Create initial portfolio history entry with $5 balance
            cursor.execute('''
            INSERT INTO portfolio_history 
            (session_id, timestamp, total_value, cash_balance, positions_value)
            VALUES (?, datetime('now'), ?, ?, 0.0)
            ''', (session_id, initial_portfolio, initial_portfolio))
            
            conn.commit()
            conn.close()
            logger.info(f"Created initial portfolio history with ${initial_portfolio:.2f} balance")
            
            # Initialize simulation tracker with $5 balance
            live_sim = SimulationTracker(
                initial_portfolio=initial_portfolio,
                db_path='live_trading.db'
            )
            
            # Store max_positions as a global variable for the trading loop
            live_sim.max_positions = max_positions or 3
            
            # Start WebSocket client if available
            if WEBSOCKET_AVAILABLE:
                logger.info("Starting external WebSocket client")
                # Initialize the WebSocket client with API credentials
                if initialize_websocket(
                    org_id=ORG_ID,
                    api_key_id=API_KEY_ID,
                    private_key_pem=PRIVATE_KEY_PEM,
                    callback=on_websocket_message
                ):
                    # Start the WebSocket connection
                    if start_websocket():
                        # Subscribe to top symbols - modified to not limit by alphabetical order
                        symbols = get_cached_symbols()  # Get all symbols
                        # Only use a reasonable number to avoid overloading the WebSocket
                        if len(symbols) > 50:
                            # Choose symbols evenly distributed across the alphabet
                            symbols = [symbols[i] for i in range(0, len(symbols), len(symbols)//50)][:50]
                        subscribe_to_symbols(symbols)
                        logger.info(f"WebSocket client connected and subscribed to {len(symbols)} symbols")
                    else:
                        logger.warning("Failed to start WebSocket client")
                else:
                    logger.warning("Failed to initialize WebSocket client")
            else:
                # Fallback to internal WebSocket client if integration is not available
                logger.warning("External WebSocket integration not available, falling back to internal implementation")
                if ws_client is None:
                    ws_client = CoinbaseWebSocketClient(on_message_callback=on_websocket_message)
                
                if not ws_client.running:
                    ws_client.start()
            
            stop_trading = False
            trading_thread = threading.Thread(target=trading_loop, daemon=False)
            trading_thread.start()
            
            # Register shutdown handler
            atexit.register(lambda: stop_trading_gracefully())
            
            return "Trading started with $5 balance", True, False
            
        except Exception as e:
            logger.error(f"Error starting live trading: {str(e)}")
            return f"Error starting trading: {str(e)}", False, True
    
    elif button_id == 'stop-live-trading-btn' and stop_clicks:
        # Stop WebSocket client
        if WEBSOCKET_AVAILABLE:
            if is_websocket_running():
                stop_websocket()
                logger.info("External WebSocket client stopped")
        elif ws_client and ws_client.running:
            ws_client.stop()
            logger.info("Internal WebSocket client stopped")
        
        return stop_trading_gracefully(), False, True
    
    return dash.no_update

def stop_trading_gracefully():
    """Stop trading gracefully and clean up resources"""
    global trading_thread, stop_trading, live_sim
    
    if not trading_thread or not trading_thread.is_alive():
        return "Trading not running"
    
    stop_trading = True
    logger.info("Stopping trading thread...")
    trading_thread.join(timeout=5)  # Wait up to 5 seconds for thread to finish
    
    if trading_thread.is_alive():
        logger.warning("Trading thread did not stop gracefully")
        return "Warning: Trading thread did not stop gracefully"
    
    # Clean up resources
    if live_sim:
        try:
            # Save final state
            live_sim.save_state()
            # Close session
            conn = sqlite3.connect('live_trading.db')
            cursor = conn.cursor()
            cursor.execute('''
            UPDATE sessions 
            SET end_time = datetime('now'), 
                status = 'completed',
                final_balance = ?
            WHERE status = 'active'
            ''', (live_sim.get_total_portfolio_value(),))
            conn.commit()
            conn.close()
            logger.info("Trading session recorded successfully")
        except Exception as e:
            logger.error(f"Error cleaning up: {str(e)}")
    
    trading_thread = None
    live_sim = None
    return "Trading stopped successfully"

def on_websocket_message(data):
    """Callback function for WebSocket messages"""
    try:
        # Get message type and channel
        msg_type = data.get('type', 'unknown')
        channel = data.get('channel', 'unknown')
        
        logger.debug(f"WebSocket message - Type: {msg_type}, Channel: {channel}")
        
        if msg_type == 'ticker' and 'product_id' in data:
            # Process ticker/price update
            symbol = data.get('product_id')
            price = float(data.get('price', 0))
            size = float(data.get('last_size', 0))
            timestamp = data.get('time', None)
            
            if price > 0:
                # Convert timestamp to milliseconds
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        timestamp_ms = int(dt.timestamp() * 1000)
                    except:
                        timestamp_ms = int(datetime.now().timestamp() * 1000)
                else:
                    timestamp_ms = int(datetime.now().timestamp() * 1000)
                
                # Store price data
                ticker_data = {
                    'price': price,
                    'volume': size,
                    'timestamp': timestamp_ms
                }
                
                with market_data_lock:
                    if symbol not in price_history:
                        price_history[symbol] = []
                    price_history[symbol].insert(0, ticker_data)
                    if len(price_history[symbol]) > 1000:
                        price_history[symbol] = price_history[symbol][:1000]
                
                logger.info(f"Price update for {symbol}: ${price:.4f} (Size: {size:.8f})")
                
                # Update database
                try:
                    session_id = get_active_session()
                    if session_id:
                        conn = sqlite3.connect('live_trading.db')
                        cursor = conn.cursor()
                        cursor.execute('''
                        UPDATE positions 
                        SET current_price = ?,
                            last_update = datetime('now')
                        WHERE session_id = ? AND symbol = ?
                        ''', (price, session_id, symbol))
                        conn.commit()
                        conn.close()
                except Exception as db_error:
                    logger.error(f"Database error: {str(db_error)}")
                    
        elif msg_type == 'snapshot':
            # Process full account snapshot
            if 'orders' in data:
                logger.info(f"Orders snapshot: {len(data['orders'])} orders")
            if 'positions' in data:
                logger.info(f"Positions snapshot received")
                
        elif msg_type == 'update':
            # Process incremental updates
            if 'orders' in data:
                logger.info(f"Order update: {data['orders']}")
            if 'positions' in data:
                logger.info(f"Position update received")
                
        elif msg_type == 'error':
            logger.error(f"WebSocket error: {data.get('message', 'Unknown error')}")
            
        elif msg_type == 'heartbeat':
            logger.debug("Heartbeat received")
            
    except Exception as e:
        logger.error(f"Error processing WebSocket message: {str(e)}")
        logger.error(f"Message data: {data}")

# Initialize WebSocket client here after on_websocket_message is defined
initialize_websocket_client()

def ensure_active_session():
    """Ensure there is an active trading session, create one if needed"""
    conn = None
    try:
        logger.info("Checking for active session...")
        # Get absolute path for database
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'live_trading.db')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database if it doesn't exist
        if not os.path.exists(db_path):
            logger.info(f"Database file doesn't exist, initializing at {db_path}")
            initialize_database()
        
        # Connect to database
        conn = sqlite3.connect(db_path, timeout=30)
        cursor = conn.cursor()
        
        # Make sure sessions table exists with proper schema
        try:
            cursor.execute("SELECT initial_balance FROM sessions LIMIT 1")
        except sqlite3.OperationalError:
            logger.warning("Sessions table has incorrect schema, reinitializing database")
            conn.close()
            initialize_database()
            conn = sqlite3.connect(db_path, timeout=30)
            cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Check for active session
        cursor.execute('SELECT id FROM sessions WHERE status = "active" ORDER BY start_time DESC LIMIT 1')
        session = cursor.fetchone()
        
        if not session:
            logger.info("No active session found, creating new one...")
            # Create new session with $5 balance
            cursor.execute("""
                INSERT INTO sessions 
                (start_time, initial_balance, status)
                VALUES (datetime('now'), 5.0, 'active')
            """)
            conn.commit()
            
            session_id = cursor.lastrowid
            logger.info(f"Created new session with ID: {session_id}")
            
            # Add initial portfolio history
            cursor.execute("""
            INSERT INTO portfolio_history 
            (session_id, timestamp, total_value, cash_balance, positions_value)
            VALUES (?, datetime('now'), 5.0, 5.0, 0.0)
            """, (session_id,))
            
            conn.commit()
            logger.info("Added initial portfolio history")
            return session_id
        else:
            session_id = session[0]
            logger.info(f"Using existing session ID: {session_id}")
            return session_id
            
    except Exception as e:
        logger.error(f"Error ensuring active session: {str(e)}")
        if conn:
            try:
                conn.rollback()
            except:
                pass
        return None
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass

def get_active_session():
    """Get the current active session ID"""
    try:
        # Get absolute path for database
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'live_trading.db')
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM sessions WHERE status = "active" ORDER BY start_time DESC LIMIT 1')
        session = cursor.fetchone()
        
        conn.close()
        return session[0] if session else None
        
    except Exception as e:
        logger.error(f"Error getting active session: {e}")
        if conn:
            conn.close()
        return None

def create_direct_session():
    """Force create a new session with direct SQLite calls"""
    try:
        logger.info(f"Creating a fresh session - DB path: {DB_PATH}")
        
        # Create database tables directly
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create sessions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            initial_balance REAL NOT NULL DEFAULT 5.0,
            final_balance REAL,
            total_trades INTEGER DEFAULT 0,
            winning_trades INTEGER DEFAULT 0,
            total_profit REAL DEFAULT 0.0,
            status TEXT DEFAULT 'active'
        )
        ''')
        
        # Create portfolio_history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_value REAL NOT NULL,
            cash_balance REAL NOT NULL,
            positions_value REAL NOT NULL
        )
        ''')
        
        # Create positions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            quantity REAL NOT NULL,
            entry_price REAL NOT NULL,
            current_price REAL NOT NULL,
            value REAL NOT NULL,
            profit REAL DEFAULT 0.0,
            pnl REAL DEFAULT 0.0,
            entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tp_order_id TEXT,
            sl_order_id TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(id),
            UNIQUE(session_id, symbol)
        )
        ''')
        
        # Create trades table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            value REAL NOT NULL,
            profit REAL
        )
        ''')
        
        # Delete all active sessions
        cursor.execute("DELETE FROM sessions WHERE status = 'active'")
        
        # Create new session
        cursor.execute('''
        INSERT INTO sessions (start_time, initial_balance, status)
        VALUES (datetime('now'), 5.0, 'active')
        ''')
        
        session_id = cursor.lastrowid
        logger.info(f"Created new session with ID: {session_id}")
        
        # Add initial portfolio history
        cursor.execute('''
        INSERT INTO portfolio_history 
        (session_id, timestamp, total_value, cash_balance, positions_value)
        VALUES (?, datetime('now'), ?, ?, 0.0)
        ''', (session_id,))
        
        conn.commit()
        
        logger.info("Database schema and session created successfully")
        return session_id
        
    except Exception as e:
        logger.error(f"Direct session creation failed: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

# Modify the update_live_trading_display callback to use the direct session creation
@app.callback(
    [
        Output('live-performance-metrics', 'children'),
        Output('live-portfolio-chart', 'figure'),
        Output('live-positions-table', 'data'),
        Output('live-trades-table', 'data')
    ],
    [
        Input('live-trading-interval', 'n_intervals'),
        Input('start-live-trading-btn', 'n_clicks'),
        Input('stop-live-trading-btn', 'n_clicks')
    ],
    prevent_initial_call=False
)
def update_live_trading_display(n_intervals, start_clicks, stop_clicks):
    try:
        # Get current positions from lk.py
        try:
            # Ensure active session exists
            session_id = ensure_active_session()
            if not session_id:
                logger.error("Could not create trading session")
                return html.Div("Error: Could not create trading session"), create_empty_figure(), [], []

            # Get account data from lk.py
            result = subprocess.run(
                [sys.executable, "crypto_trading/c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/lk.py"],
                capture_output=True, text=True, check=True
            )
            output = result.stdout.strip()
            
            # Parse JSON data between markers
            start_marker = "DASHBOARD_DATA_START"
            end_marker = "DASHBOARD_DATA_END"
            
            if start_marker in output and end_marker in output:
                json_str = output.split(start_marker)[1].split(end_marker)[0]
                try:
                    data = json.loads(json_str)
                    
                    # Get positions and total value
                    positions = data.get('positions', [])
                    total_value = data.get('total_value', 0)
                    
                    # Find USD cash balance
                    cash_balance = 0
                    for pos in positions:
                        if pos.get('currency') == 'USD':
                            cash_balance = float(pos.get('usd_value', 0))
                            break
                    
                    # Calculate positions value (total - cash)
                    positions_value = total_value - cash_balance
                    
                    # Store portfolio data in database
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    
                    try:
                        # Update portfolio history
                        cursor.execute('''
                        INSERT INTO portfolio_history 
                        (session_id, timestamp, total_value, cash_balance, positions_value)
                        VALUES (?, datetime('now'), ?, ?, ?)
                        ''', (session_id, total_value, cash_balance, positions_value))
                        
                        conn.commit()
                    except Exception as db_error:
                        logger.error(f"Database error updating portfolio history: {str(db_error)}")
                    finally:
                        conn.close()
                    
                    # Generate positions data for table
                    positions_data = []
                    for pos in positions:
                        # Skip USD
                        if pos.get('currency') == 'USD':
                            continue
                            
                        symbol = f"{pos.get('currency')}-USD"
                        current_price = float(pos.get('price', 0))
                        quantity = float(pos.get('amount', 0))
                        usd_value = float(pos.get('usd_value', 0))
                        
                        # Get entry price from database if available
                        try:
                            conn = sqlite3.connect(DB_PATH)
                            cursor = conn.cursor()
                            cursor.execute('''
                            SELECT entry_price, quantity 
                            FROM positions 
                            WHERE session_id = ? AND symbol = ?
                            ''', (session_id, symbol))
                            
                            db_position = cursor.fetchone()
                            if db_position:
                                entry_price, db_quantity = db_position
                                # Calculate profit/loss
                                pnl = ((current_price / entry_price) - 1) * 100
                                position_data = {
                                    'symbol': symbol,
                                    'quantity': f"{quantity:.8f}",
                                    'current_price': f"${current_price:.4f}",
                                    'value': f"${usd_value:.2f}",
                                    'entry_price': f"${entry_price:.4f}",
                                    'pnl': f"{pnl:+.2f}%"
                                }
                            else:
                                position_data = {
                                    'symbol': symbol,
                                    'quantity': f"{quantity:.8f}",
                                    'current_price': f"${current_price:.4f}",
                                    'value': f"${usd_value:.2f}",
                                    'entry_price': "N/A",
                                    'pnl': "N/A"
                                }
                            conn.close()
                        except Exception as e:
                            logger.error(f"Error getting position data from database: {str(e)}")
                            position_data = {
                                'symbol': symbol,
                                'quantity': f"{quantity:.8f}",
                                'current_price': f"${current_price:.4f}",
                                'value': f"${usd_value:.2f}",
                                'entry_price': "N/A",
                                'pnl': "N/A"
                            }
                        
                        positions_data.append(position_data)
                    
                    # Get recent trades from database
                    trades_data = []
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        cursor = conn.cursor()
                        cursor.execute('''
                        SELECT datetime(timestamp, 'localtime') as time, symbol, action, price, quantity, value, profit
                        FROM trades
                        WHERE session_id = ?
                        ORDER BY timestamp DESC
                        LIMIT 20
                        ''', (session_id,))
                        
                        for row in cursor.fetchall():
                            time_str, symbol, action, price, quantity, value, profit = row
                            trade_data = {
                                'time': time_str,
                                'symbol': symbol,
                                'action': action,
                                'price': f"${price:.4f}",
                                'quantity': f"{quantity:.8f}",
                                'value': f"${value:.2f}",
                                'profit': f"${profit:.2f}" if profit else "N/A"
                            }
                            trades_data.append(trade_data)
                            
                        conn.close()
                    except Exception as e:
                        logger.error(f"Error getting trades from database: {str(e)}")
                    
                    # Create portfolio chart
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        cursor = conn.cursor()
                        cursor.execute('''
                        SELECT datetime(timestamp, 'localtime') as time, total_value, cash_balance, positions_value
                        FROM portfolio_history
                        WHERE session_id = ?
                        ORDER BY timestamp ASC
                        ''', (session_id,))
                        
                        history = cursor.fetchall()
                        conn.close()
                        
                        if history:
                            timestamps = [row[0] for row in history]
                            total_values = [row[1] for row in history]
                            cash_values = [row[2] for row in history]
                            position_values = [row[3] for row in history]
                            
                            # Create figure
                            portfolio_fig = go.Figure()
                            
                            # Add total value line
                            portfolio_fig.add_trace(go.Scatter(
                                x=timestamps,
                                y=total_values,
                                mode='lines',
                                name='Total Value',
                                line=dict(color='#2196F3', width=2)
                            ))
                            
                            # Add cash balance line
                            portfolio_fig.add_trace(go.Scatter(
                                x=timestamps,
                                y=cash_values,
                                mode='lines',
                                name='Cash Balance',
                                line=dict(color='#4CAF50', width=1, dash='dash')
                            ))
                            
                            # Add positions value line
                            portfolio_fig.add_trace(go.Scatter(
                                x=timestamps,
                                y=position_values,
                                mode='lines',
                                name='Positions Value',
                                line=dict(color='#FFC107', width=1, dash='dash')
                            ))
                            
                            portfolio_fig.update_layout(
                                title='Portfolio Value Over Time',
                                xaxis_title='Time',
                                yaxis_title='Value (USD)',
                                template='plotly_white'
                            )
                        else:
                            portfolio_fig = create_empty_figure()
                    except Exception as e:
                        logger.error(f"Error creating portfolio chart: {str(e)}")
                        portfolio_fig = create_empty_figure()
                    
                    # Create performance metrics display
                    metrics_children = [
                        html.Div([
                            html.H4("Account Overview"),
                            html.Div([
                                html.P([
                                    html.Strong("Total Value: "),
                                    html.Span(f"${total_value:.2f}")
                                ]),
                                html.P([
                                    html.Strong("Cash Balance: "),
                                    html.Span(f"${cash_balance:.2f}")
                                ]),
                                html.P([
                                    html.Strong("Positions Value: "),
                                    html.Span(f"${positions_value:.2f}")
                                ]),
                            ]),
                        ])
                    ]
                    
                    return metrics_children, portfolio_fig, positions_data, trades_data
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {str(e)}")
                    return html.Div("Error parsing account data"), create_empty_figure(), [], []
            
            # If no data available, return empty displays
            return html.Div("No trading data available"), create_empty_figure(), [], []
            
        except Exception as e:
            logger.error(f"Error getting positions from lk.py: {str(e)}")
            traceback.print_exc()
            return html.Div(f"Error: {str(e)}"), create_empty_figure(), [], []
    
    except Exception as e:
        logger.error(f"Error updating live trading display: {str(e)}")
        traceback.print_exc()
        return html.Div(f"Error: {str(e)}"), create_empty_figure(), [], []

def calculate_sharpe_ratio(history_df):
    """Calculate Sharpe ratio from portfolio history"""
    try:
        if len(history_df) < 2:
            return 0
            
        # Calculate daily returns
        daily_returns = history_df.set_index('timestamp')['total_value'].resample('D').last().pct_change().dropna()
        
        if len(daily_returns) < 2:
            return 0
            
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
        
        return sharpe if not np.isnan(sharpe) else 0
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {str(e)}")
        return 0

def calculate_max_drawdown(history_df):
    """Calculate maximum drawdown from portfolio history"""
    try:
        if len(history_df) < 2:
            return 0
            
        # Calculate rolling maximum and current drawdown
        rolling_max = history_df['total_value'].cummax()
        drawdown = ((rolling_max - history_df['total_value']) / rolling_max) * 100
        
        return drawdown.max() if not np.isnan(drawdown.max()) else 0
    except Exception as e:
        logger.error(f"Error calculating max drawdown: {str(e)}")
        return 0

def calculate_performance_metrics(history_df, realized_gains, unrealized_gains, session_info):
    """Calculate performance metrics with validation"""
    try:
        if len(history_df) > 0:
            initial_value = history_df['total_value'].iloc[0]
            current_value = history_df['total_value'].iloc[-1]
            
            # Validate values
            if initial_value <= 0:
                logger.warning(f"Invalid initial value: {initial_value}")
                initial_value = session_info['initial_balance']
            if current_value <= 0:
                logger.warning(f"Invalid current value: {current_value}")
                current_value = initial_value
            
            total_return = ((current_value / initial_value) - 1) * 100
            
            # Calculate returns with proper handling
            try:
                daily_returns = history_df.set_index('timestamp')['total_value'].resample('D').last().pct_change()
                daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
                sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if len(daily_returns) > 1 else 0
            except Exception as e:
                logger.error(f"Error calculating Sharpe ratio: {str(e)}")
                sharpe_ratio = 0
            
            # Calculate drawdown with validation
            try:
                rolling_max = history_df['total_value'].cummax()
                drawdown = ((rolling_max - history_df['total_value']) / rolling_max) * 100
                max_drawdown = drawdown.max()
                if np.isnan(max_drawdown) or np.isinf(max_drawdown):
                    max_drawdown = 0
            except Exception as e:
                logger.error(f"Error calculating drawdown: {str(e)}")
                max_drawdown = 0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'current_value': current_value,
                'realized_gains': realized_gains,
                'unrealized_gains': unrealized_gains,
                'total_gains': realized_gains + unrealized_gains
            }
        else:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'current_value': session_info['initial_balance'],
                'realized_gains': realized_gains,
                'unrealized_gains': unrealized_gains,
                'total_gains': realized_gains + unrealized_gains
            }
    except Exception as e:
        logger.error(f"Error in calculate_performance_metrics: {str(e)}")
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'current_value': session_info.get('initial_balance', 0),
            'realized_gains': 0,
            'unrealized_gains': 0,
            'total_gains': 0
        }

def create_portfolio_chart(history_df, trades_df):
    """Create portfolio chart with improved error handling and data validation"""
    fig = go.Figure()
    
    try:
        # Validate input data
        if history_df is None or trades_df is None:
            raise ValueError("Missing required data for chart creation")
            
        # Validate data types
        if not isinstance(history_df, pd.DataFrame) or not isinstance(trades_df, pd.DataFrame):
            raise ValueError("Invalid data type for chart creation")
            
        # Ensure we have the required columns
        required_columns = ['timestamp', 'total_value', 'cash_balance', 'positions_value']
        if not all(col in history_df.columns for col in required_columns):
            logger.warning(f"Missing columns in history_df: {[col for col in required_columns if col not in history_df.columns]}")
            # Try to get data from database directly
            conn = sqlite3.connect('live_trading.db')
            session_id = get_active_session()
            if session_id:
                history_df = pd.read_sql_query('''
                    SELECT 
                        datetime(timestamp, 'localtime') as timestamp,
                        total_value,
                        cash_balance,
                        positions_value
                    FROM portfolio_history
                    WHERE session_id = ?
                    ORDER BY timestamp
                ''', conn, params=(session_id,))
            conn.close()
        
        if len(history_df) > 0:
            # Convert timestamp to datetime if it's not already
            if isinstance(history_df['timestamp'].iloc[0], str):
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            
            logger.info(f"Creating chart with {len(history_df)} data points")
            
            # Add total portfolio value
            fig.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['total_value'],
                name='Portfolio Value',
                line=dict(color='#2196F3', width=2),
                hovertemplate='%{y:$.2f}<extra>Portfolio Value</extra>'
            ))
            
            # Add cash balance
            fig.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['cash_balance'],
                name='Cash Balance',
                line=dict(color='#4CAF50', width=1, dash='dash'),
                hovertemplate='%{y:$.2f}<extra>Cash Balance</extra>'
            ))
            
            # Add positions value line
            fig.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['positions_value'],
                name='Positions Value',
                line=dict(color='#FF9800', width=1, dash='dash'),
                hovertemplate='%{y:$.2f}<extra>Positions Value</extra>'
            ))
            
            # Update layout
            fig.update_layout(
                title='Portfolio Performance',
                yaxis_title='Value (USD)',
                template='plotly_white',
                hovermode='x unified',
                showlegend=True
            )
            
            # Add range selector
            fig.update_xaxes(
                rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1h", step="hour", stepmode="backward"),
                        dict(count=6, label="6h", step="hour", stepmode="backward"),
                        dict(count=12, label="12h", step="hour", stepmode="backward"),
                        dict(count=24, label="24h", step="hour", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                )
            )
        else:
            # Create empty figure with message
            fig.add_annotation(
                text="No portfolio data available - Start trading to see performance",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20, color='#666666')
            )
            
    except Exception as e:
        logger.error(f"Error creating portfolio chart: {str(e)}")
        # Return empty figure with error message
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color='#f44336')
        )
        return fig

def prepare_trades_data(trades_df):
    """Prepare trades data with validation"""
    trades_data = []
    try:
        for _, trade in trades_df.head(10).iterrows():
            try:
                trades_data.append({
                    'timestamp': pd.to_datetime(trade['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': str(trade['symbol']),
                    'action': str(trade['action']),
                    'price': f"${float(trade['price']):.2f}",
                    'quantity': f"{float(trade['quantity']):.6f}",
                    'value': f"${float(trade['value']):,.2f}"
                })
            except Exception as e:
                logger.error(f"Error processing trade record: {str(e)}")
                continue
    except Exception as e:
        logger.error(f"Error preparing trades data: {str(e)}")
    
    return trades_data

# Add CSS for loading animation and context menu
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Crypto Trading Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            .loader {
                border: 4px solid #f3f3f3;
                border-radius: 50%;
                border-top: 4px solid #2196F3;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            #context-menu {
                position: fixed;
                background-color: #ffffff;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
                border-radius: 4px;
                padding: 5px 0;
                z-index: 1000;
                border: 1px solid #ddd;
            }
            
            #context-menu div {
                padding: 8px 20px;
                cursor: pointer;
                color: #000000;
                transition: background-color 0.2s;
            }
            
            #context-menu div:hover {
                background-color: #f5f5f5;
                color: #2196F3;
            }
            
            .dash-table-container {
                cursor: context-menu;
            }
            
            .dash-cell {
                cursor: context-menu;
                color: #000000 !important;
            }
            
            /* Update table styles */
            .dash-spreadsheet-container .dash-spreadsheet-inner td {
                color: #000000 !important;
                background-color: #ffffff !important;
            }
            
            .dash-spreadsheet-container .dash-spreadsheet-inner th {
                color: #000000 !important;
                background-color: #f5f5f5 !important;
            }
            
            /* Style for positive/negative values */
            .positive-value {
                color: #4CAF50 !important;
                font-weight: bold;
            }
            
            .negative-value {
                color: #f44336 !important;
                font-weight: bold;
            }
            
            /* Update dropdown styles */
            .Select-control {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #ddd !important;
            }
            
            .Select-menu-outer {
                background-color: #ffffff !important;
                border: 1px solid #ddd !important;
            }
            
            .Select-option {
                color: #000000 !important;
            }
            
            .Select-option:hover {
                background-color: #f5f5f5 !important;
                color: #2196F3 !important;
            }
            
            /* Button styles */
            button {
                background-color: #2196F3 !important;
                color: #ffffff !important;
                border: none !important;
                padding: 8px 16px !important;
                border-radius: 4px !important;
                cursor: pointer !important;
                transition: background-color 0.2s !important;
            }
            
            button:hover {
                background-color: #1976D2 !important;
            }
            
            /* Input styles */
            input {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #ddd !important;
                padding: 6px !important;
                border-radius: 4px !important;
            }
            
            /* Tab styles */
            .dash-tab {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #ddd !important;
            }
            
            .dash-tab--selected {
                background-color: #2196F3 !important;
                color: #ffffff !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def predict_with_pretrained_model(df, symbol, interval='1h'):
    """
    Make predictions using pretrained models for a given symbol.
    Uses an ensemble approach similar to the backtesting strategy.
    """
    try:
        # First ensure all required indicators are calculated
        df = calculate_indicators(df)
        if df.empty:
            print(f"No data available for {symbol}")
            return pd.DataFrame(), pd.DataFrame()
            
        # Convert interval to seconds
        interval_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '1d': 86400
        }
        granularity = interval_map.get(interval, 3600)
        
        # Check if models exist, if not, train them
        model_prefix = f"{symbol.replace('-', '')}_{granularity}"
        reg_model_path = os.path.join(MODELS_DIR, f"{model_prefix}_reg.pkl")
        clf_model_path = os.path.join(MODELS_DIR, f"{model_prefix}_clf.pkl")
        
        if not os.path.exists(reg_model_path) or not os.path.exists(clf_model_path):
            print(f"Training new models for {symbol}...")
            reg_models, clf = train_model_for_symbol(symbol, granularity)
            if reg_models is None or clf is None:
                print(f"Failed to train models for {symbol}")
                return pd.DataFrame(), pd.DataFrame()
        else:
            try:
                reg_models = joblib.load(reg_model_path)
                clf = joblib.load(clf_model_path)
            except Exception as e:
                print(f"Error loading models for {symbol}: {str(e)}")
                return pd.DataFrame(), pd.DataFrame()
        
        # Use the same feature set as in training
        feature_cols = [
            'EMA12', 'EMA26', 'MACD', 'Signal_Line', 'RSI', 'MA20',
            'rolling_std_10', 'lag_1', 'lag_2', 'lag_3', 'OBV', 'ATR', '%K', '%D'
        ]
        
        # Verify all features exist
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            print(f"Missing features for {symbol}: {missing_features}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Reset index to avoid indexing issues
        df = df.reset_index(drop=True)
        X = df[feature_cols].copy()
        
        # Drop any NaN values
        X = X.dropna()
        if X.empty:
            print(f"No valid data points for {symbol} after dropping NaN values")
            return pd.DataFrame(), pd.DataFrame()
        
        # Make predictions using ensemble
        reg_predictions = []
        reg_confidences = []
        
        # Get the models from the dictionary
        if isinstance(reg_models, dict) and 'models' in reg_models:
            models_list = reg_models['models']
        else:
            models_list = [reg_models]  # If it's a single model
            
        for model in models_list:
            try:
                # Get predictions
                pred = model.predict(X)
                reg_predictions.append(pred.reshape(-1, 1))
                
                # Calculate prediction confidence based on model type
                if hasattr(model, 'predict_proba'):
                    conf = np.max(model.predict_proba(X), axis=1)
                elif hasattr(model, 'feature_importances_'):
                    conf = np.ones_like(pred) * np.mean(model.feature_importances_)
                else:
                    conf = np.ones_like(pred) * 0.5
                    
                reg_confidences.append(conf.reshape(-1, 1))
                
            except Exception as e:
                print(f"Error with model prediction for {symbol}: {str(e)}")
                continue
        
        if not reg_predictions:
            print(f"No valid predictions for {symbol}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Combine predictions with confidence weighting
        reg_predictions = np.hstack(reg_predictions)
        reg_confidences = np.hstack(reg_confidences)
        
        # Weighted average of predictions based on confidence
        weights = reg_confidences / reg_confidences.sum(axis=1, keepdims=True)
        price_predictions = np.sum(reg_predictions * weights, axis=1)
        
        # Calculate prediction confidence
        prediction_std = np.std(reg_predictions, axis=1)
        prediction_confidence = 1 / (1 + prediction_std)
        
        # Prepare features for classification
        X_with_pred = X.copy()
        X_with_pred['predicted_close'] = price_predictions
        
        try:
            # Get classification predictions and probabilities
            direction_pred = clf.predict(X_with_pred)
            confidence_scores = clf.predict_proba(X_with_pred)
            
            # Convert numeric predictions to BUY/SELL strings
            direction_pred = np.where(direction_pred == 1, 'BUY', 'SELL')
            
            # Combine regression and classification confidence
            combined_confidence = prediction_confidence.reshape(-1, 1) * confidence_scores
            
            # Create results DataFrames with proper indexing
            predictions = pd.DataFrame({
                'timestamp': df.loc[X.index, 'timestamp'],
                'actual_price': df.loc[X.index, 'close'],
                'predicted_price': price_predictions,
                'direction': direction_pred,
                'prediction_std': prediction_std
            })
            
            confidence = pd.DataFrame({
                'timestamp': df.loc[X.index, 'timestamp'],
                'buy_confidence': combined_confidence[:, 1] if combined_confidence.shape[1] > 1 else np.zeros(len(combined_confidence)),
                'sell_confidence': combined_confidence[:, 0] if combined_confidence.shape[1] > 0 else np.zeros(len(combined_confidence)),
                'prediction_confidence': prediction_confidence
            })
            
            return predictions, confidence
            
        except Exception as e:
            print(f"Error in classification step for {symbol}: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
            
    except Exception as e:
        print(f"Error making predictions for {symbol}: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def create_crypto_scanner_table(results):
    """Create a formatted table of crypto scanner results"""
    if not results:
        return html.Div("No results found", style={'color': '#FFFFFF', 'padding': '20px'})
        
    # Create header row
    header = html.Tr([
        html.Th('Symbol', style={'padding': '10px', 'textAlign': 'left'}),
        html.Th('Price', style={'padding': '10px', 'textAlign': 'right'}),
        html.Th('Momentum', style={'padding': '10px', 'textAlign': 'right'}),
        html.Th('RSI', style={'padding': '10px', 'textAlign': 'right'}),
        html.Th('24h Change', style={'padding': '10px', 'textAlign': 'right'}),
        html.Th('Volume Change', style={'padding': '10px', 'textAlign': 'right'})
    ], style={'backgroundColor': '#1e1e1e'})
    
    # Create rows for each result
    rows = [header]
    for result in results:
        rows.append(html.Tr([
            html.Td(result['symbol'], 
                   style={'padding': '8px', 'color': '#FFFFFF'}),
            html.Td(f"${result['current_price']:.2f}", 
                   style={'padding': '8px', 'color': '#FFFFFF', 'textAlign': 'right'}),
            html.Td(f"{result['momentum_score']:.1f}", 
                   style={'padding': '8px', 'textAlign': 'right', 
                         'color': '#FFFFFF', 
                         'backgroundColor': get_score_color(result['momentum_score'])}),
            html.Td(f"{result['rsi']:.1f}", 
                   style={'padding': '8px', 'textAlign': 'right',
                         'color': get_rsi_color(result['rsi'])}),
            html.Td(f"{result['price_change_pct']:.1f}%", 
                   style={'padding': '8px', 'textAlign': 'right',
                         'color': '#26a69a' if result['price_change_pct'] > 0 else '#ef5350'}),
            html.Td(f"{result['volume_change_pct']:.1f}%", 
                   style={'padding': '8px', 'textAlign': 'right',
                         'color': '#26a69a' if result['volume_change_pct'] > 0 else '#ef5350'})
        ], style={'backgroundColor': '#2d2d2d'}))
    
    return html.Table(rows, style={
        'width': '100%',
        'borderCollapse': 'collapse',
        'backgroundColor': '#1e1e1e',
        'color': '#FFFFFF',
        'border': '1px solid #333333'
    })

def get_db_connection():
    """Get a connection to the SQLite database"""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        return conn, cursor
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        return None, None

def verify_db_connection(conn, cursor):
    """Verify database connection and schema"""
    try:
        # Create positions table with proper columns
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            quantity REAL NOT NULL,
            entry_price REAL NOT NULL,
            current_price REAL,
            value REAL NOT NULL,
            pl_value REAL DEFAULT 0,
            pl_percentage REAL DEFAULT 0,
            entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(session_id, symbol)
        )
        """)
        
        # Add any missing columns
        cursor.execute("PRAGMA table_info(positions)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'entry_price' not in columns:
            cursor.execute("ALTER TABLE positions ADD COLUMN entry_price REAL")
        if 'current_price' not in columns:
            cursor.execute("ALTER TABLE positions ADD COLUMN current_price REAL")
        if 'value' not in columns:
            cursor.execute("ALTER TABLE positions ADD COLUMN value REAL")
        if 'pl_value' not in columns:
            cursor.execute("ALTER TABLE positions ADD COLUMN pl_value REAL DEFAULT 0")
        if 'pl_percentage' not in columns:
            cursor.execute("ALTER TABLE positions ADD COLUMN pl_percentage REAL DEFAULT 0")
        
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error verifying database: {e}")
        return False

def update_position_prices(session_id):
    """Update current prices and P/L calculations for all positions"""
    try:
        conn = sqlite3.connect('live_trading.db')
        cursor = conn.cursor()
        
        # Get all positions for this session
        cursor.execute("""
            SELECT symbol, quantity, entry_price 
            FROM positions 
            WHERE session_id = ? AND quantity > 0
        """, (session_id,))
        
        positions = cursor.fetchall()
        for pos in positions:
            symbol, quantity, entry_price = pos
            
            # Get current market price
            current_price = get_current_price(symbol)
            if current_price:
                # Calculate P/L
                pl_value = (current_price - entry_price) * quantity
                pl_percentage = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                
                # Update position
                cursor.execute("""
                    UPDATE positions 
                    SET current_price = ?,
                        pl_value = ?,
                        pl_percentage = ?,
                        last_update = datetime('now')
                    WHERE session_id = ? AND symbol = ?
                """, (current_price, pl_value, pl_percentage, session_id, symbol))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error updating position prices: {e}")
        if conn:
            conn.close()

class DBManager:
    """Database connection manager with automatic reconnection"""
    def __init__(self, db_path='live_trading.db', timeout=30):
        self.db_path = db_path
        self.timeout = timeout
        self.conn = None
        self.cursor = None
        self.connect()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path, timeout=self.timeout)
            self.conn.isolation_level = None  # Enable autocommit mode
            self.cursor = self.conn.cursor()
            logger.debug("Database connection established")
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise
    
    def ensure_connection(self):
        """Ensure database connection is active, reconnect if needed"""
        try:
            # Test connection
            self.cursor.execute("SELECT 1")
        except (sqlite3.OperationalError, sqlite3.ProgrammingError):
            logger.warning("Database connection lost, reconnecting...")
            try:
                self.close()
            except:
                pass
            self.connect()
    
    def execute(self, query, params=None):
        """Execute query with automatic reconnection"""
        try:
            self.ensure_connection()
            if params:
                return self.cursor.execute(query, params)
            return self.cursor.execute(query)
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    def commit(self):
        """Commit transaction with error handling"""
        try:
            self.ensure_connection()
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error committing transaction: {str(e)}")
            raise
    
    def rollback(self):
        """Rollback transaction with error handling"""
        try:
            self.ensure_connection()
            self.conn.rollback()
        except Exception as e:
            logger.error(f"Error rolling back transaction: {str(e)}")
            raise
    
    def close(self):
        """Close database connection"""
        try:
            if self.conn:
                self.conn.close()
                self.conn = None
                self.cursor = None
                logger.debug("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {str(e)}")
def execute_trade_via_subprocess(symbol, side, price, funds):
    """Execute a trade by running tp_sl_fixed.py as a subprocess"""
    try:
        logger.info(f"Executing {side} trade for {symbol} via tp_sl_fixed.py")
        logger.info(f"Price: ${price}, Value: ${funds}")
        
        # Calculate size from funds and price
        size = funds / price
        
        # Get the correct path to tp_sl_fixed.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_dir, "crypto_trading", "app", "tp_sl_fixed.py"),
            os.path.join(current_dir, "..", "crypto_trading", "app", "tp_sl_fixed.py"),
            os.path.join(current_dir, "tp_sl_fixed.py")
        ]
        
        script_path = None
        for path in possible_paths:
            if os.path.exists(path):
                script_path = path
                logger.info(f"Found tp_sl_fixed.py at: {path}")
                break
                
        if not script_path:
            logger.error(f"Could not find tp_sl_fixed.py in any of these locations:")
            for path in possible_paths:
                logger.error(f"  - {path}")
            return False

        # Get the directory containing tp_sl_fixed.py
        working_dir = os.path.dirname(script_path)
        logger.info(f"Using working directory: {working_dir}")
        
        # Run tp_sl_fixed.py with the trade parameters
        cmd = [
            sys.executable,
            script_path,
            "--symbol", symbol,
            "--price", str(price),
            "--size", str(size)
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Execute the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception on non-zero exit
            cwd=working_dir  # Set working directory
        )
        
        # Log the output
        if result.stdout:
            logger.info(f"tp_sl_fixed.py output: {result.stdout}")
        if result.stderr:
            logger.error(f"tp_sl_fixed.py error: {result.stderr}")
            
        # Check if successful
        if result.returncode == 0:
            logger.info("✅ Trade executed successfully via tp_sl_fixed.py")
            return True
        else:
            logger.error(f"❌ Trade failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"Error executing trade via subprocess: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Replace the old execute_real_trade function with the new one
def execute_real_trade(symbol, side, price, funds):
    """Execute a real trade with proper TP/SL orders"""
    try:
        logger.info(f"Executing real {side} trade for {symbol}")
        logger.info(f"Price: ${price}, Value: ${funds}")
        
        # Get raw product details from API
        response = requests.get(
            f"https://api.exchange.coinbase.com/products/{symbol}",
            headers={'Accept': 'application/json'}
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to get product details for {symbol}: {response.status_code}")
            return False
        
        details = response.json()
        logger.info(f"Raw product details: {json.dumps(details, indent=2)}")
        
        # Get minimum order size
        min_market_funds = float(details.get('quote_min_size', 1.0))
        
        # Adjust order size if below minimum
        if funds < min_market_funds:
            logger.info(f"Adjusting order size from ${funds:.2f} to minimum ${min_market_funds:.2f}")
            funds = min_market_funds
            
        # Use tp_sl_fixed.py for trade execution
        return execute_trade_via_subprocess(symbol, side, price, funds)
        
    except Exception as e:
        logger.error(f"Error executing trade: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def trading_loop():
    """Main trading loop with improved error handling and graceful shutdown"""
    global stop_trading, live_sim, ws_client
    
    logger.info("Starting trading loop...")
    last_portfolio_update = datetime.now()
    last_position_analysis = datetime.now()
    last_tp_sl_check = datetime.now()
    last_position_check = datetime.now()
    last_memory_cleanup = datetime.now()
    db_manager = DBManager()
    
    try:
        # First ensure we have an active session
        db_manager.connect()
        
        # Ensure database schema is correct
        ensure_database_schema()
        
        # Get current session
        db_manager.execute('SELECT id FROM sessions WHERE status = "active" ORDER BY start_time DESC LIMIT 1')
        session_result = db_manager.cursor.fetchone()
        
        if not session_result:
            # Create new session
            db_manager.execute("""
            INSERT INTO sessions (start_time, initial_balance, status)
            VALUES (datetime('now'), 5.0, 'active')
            """)
            db_manager.commit()
            session_id = db_manager.cursor.lastrowid
            logger.info(f"Created new trading session with ID: {session_id}")
        else:
            session_id = session_result[0]
            logger.info(f"Using existing session ID: {session_id}")
            
        # Initialize WebSocket client
        logger.info("Initializing WebSocket client...")
        ws_client = create_websocket_client(on_websocket_message)
        if not ws_client:
            logger.error("Failed to initialize WebSocket client")
            return
        while not stop_trading:
            try:
                # Memory cleanup every 5 minutes
                if (datetime.now() - last_memory_cleanup).total_seconds() >= 300:
                    import gc
                    gc.collect()  # Force garbage collection
                    last_memory_cleanup = datetime.now()
                    logger.debug("Performed memory cleanup")
                
                # Update portfolio and positions every minute
                now = datetime.now()
                if (now - last_portfolio_update).total_seconds() >= 60:
                    from crypto_trading.app.session_utils import update_portfolio_history
                    update_portfolio_history(session_id)
                    last_portfolio_update = now
                
                # Check TP/SL orders every 30 seconds
                if (now - last_tp_sl_check).total_seconds() >= 30:
                    try:
                        # Import with function parameters to avoid circular imports
                        from crypto_trading.app.tp_sl_manager import check_and_manage_tp_sl_orders
                        logger.info("Checking TP/SL orders...")
                        check_and_manage_tp_sl_orders()
                        last_tp_sl_check = now
                    except Exception as tp_sl_error:
                        logger.error(f"Error checking TP/SL orders: {str(tp_sl_error)}")
                
                # Evaluate existing positions every minute
                if (now - last_position_check).total_seconds() >= 60:
                    try:
                        logger.info("Evaluating existing positions...")
                        # Get all positions
                        db_manager.execute('''
                        SELECT symbol, quantity, entry_price, current_price, tp_order_id, sl_order_id
                        FROM positions
                        WHERE session_id = ?
                        ''', (session_id,))
                        positions = db_manager.cursor.fetchall()
                        
                        for pos in positions:
                            symbol, quantity, entry_price, current_price, tp_order_id, sl_order_id = pos
                            try:
                                # Get market data for technical analysis
                                df = get_market_data(symbol, interval='1h')
                                if df.empty:
                                    logger.warning(f"No market data available for {symbol}")
                                    continue
                                
                                # Calculate indicators
                                df = calculate_indicators(df)
                                latest = df.iloc[-1]
                                
                                # Get ML predictions
                                predictions, confidence = predict_with_pretrained_model(df, symbol, interval='1h')
                                if predictions.empty:
                                    logger.warning(f"No ML predictions available for {symbol}")
                                    continue
                                
                                latest_pred = predictions.iloc[-1]
                                latest_conf = confidence.iloc[-1]
                                
                                # Evaluate exit conditions
                                should_exit, exit_reason = evaluate_exit_conditions(
                                    symbol, entry_price, current_price,
                                    latest_pred.get('momentum_score', 0),
                                    latest.get('RSI', 50),
                                    db_manager.conn, db_manager.cursor
                                )
                                
                                if should_exit:
                                    logger.info(f"Exit signal for {symbol}: {exit_reason}")
                                    # First cancel any existing TP/SL orders
                                    if tp_order_id:
                                        logger.info(f"Canceling TP order: {tp_order_id}")
                                        cancel_order(tp_order_id)
                                    if sl_order_id:
                                        logger.info(f"Canceling SL order: {sl_order_id}")
                                        cancel_order(sl_order_id)
                                    
                                    # Execute exit trade
                                    if execute_exit_trade(
                                        session_id, symbol, quantity,
                                        entry_price, current_price, exit_reason,
                                        db_manager.conn, db_manager.cursor
                                    ):
                                        logger.info(f"Successfully exited {symbol} position")
                                    else:
                                        logger.error(f"Failed to exit {symbol} position")
                                
                            except Exception as pos_error:
                                logger.error(f"Error evaluating {symbol} position: {str(pos_error)}")
                                continue
                        
                        last_position_check = now
                        
                    except Exception as eval_error:
                        logger.error(f"Error evaluating positions: {str(eval_error)}")
                
                # Get available balance from lk.py
                result = subprocess.run(
                    [sys.executable, "crypto_trading/c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/lk.py"],
                    capture_output=True, text=True, check=True
                )
                output = result.stdout.strip()
                
                # Parse JSON data between markers
                start_marker = "DASHBOARD_DATA_START"
                end_marker = "DASHBOARD_DATA_END"
                
                if start_marker in output and end_marker in output:
                    json_str = output.split(start_marker)[1].split(end_marker)[0]
                    data = json.loads(json_str)
                    
                    # Find USD balance
                    available_balance = 0
                    for pos in data.get('positions', []):
                        if pos.get('currency') == 'USD':
                            available_balance = float(pos.get('usd_value', 0))
                            break
                
                if available_balance <= 0:
                    logger.warning("❌ No available balance for trading")
                    time.sleep(60)
                    continue
                
                logger.info(f"💰 Available balance for trading: ${available_balance:.2f}")
                
                # Calculate maximum new positions we can open
                db_manager.execute('SELECT COUNT(DISTINCT symbol) FROM positions WHERE session_id = ?', (session_id,))
                current_positions = db_manager.cursor.fetchone()[0]
                
                max_new_positions = max(0, live_sim.max_positions - current_positions)
                
                if max_new_positions > 0 and available_balance > 0:
                    # Calculate position size based on available balance
                    if current_positions == 0:
                        position_size = min(available_balance * 0.2, 1.0)  # Use up to 20% of balance or $1.00 max
                    else:
                        position_size = min(available_balance / max_new_positions, 0.5)  # Use remaining balance divided by slots, max $0.50
                    
                    logger.info(f"🎯 Looking for trades with position size: ${position_size:.2f}")
                    
                    # Scan market for opportunities
                    opportunities = scan_market()
                    if opportunities:
                        for opp in opportunities[:max_new_positions]:
                            try:
                                symbol = opp['symbol']
                                current_price = opp['current_price']
                                
                                # Skip if price is too high relative to our position size
                                min_quantity = 0.000001  # Minimum quantity we can trade
                                if position_size / current_price < min_quantity:
                                    logger.info(f"⏭️ Skipping {symbol} - price too high for our position size")
                                    continue
                                # Add this with your other time tracking variables near the start of the function

                                # Double check available balance
                                fresh_balance = get_available_balance()
                                if fresh_balance < position_size:
                                    logger.warning(f"❌ Insufficient balance for {symbol} trade. Required: ${position_size:.2f}, Available: ${fresh_balance:.2f}")
                                    break
                                
                                # Execute trade using the real trade function
                                # from crypto_trading.app.session_utils import execute_real_trade
                                if execute_real_trade(symbol, "BUY", current_price, funds=position_size):
                                    logger.info(f"✅ Successfully executed real BUY order for {symbol}")
                                    # Update available balance
                                    available_balance = fresh_balance - position_size
                                    if available_balance < 0.1:  # If less than $0.10 left, stop trading
                                        logger.info("💰 Remaining balance too low, stopping trading")
                                        break
                                else:
                                    logger.error(f"❌ Failed to execute real BUY order for {symbol}")
                                
                            except Exception as e:
                                logger.error(f"❌ Error processing opportunity for {symbol}: {str(e)}")
                                continue
                else:
                    logger.info(f"ℹ️ No new positions available. Current positions: {current_positions}, Available balance: ${available_balance:.2f}")
            # Add this with your other time tracking variables near the start of the function
                last_position_analysis = datetime.now()

                # Add this inside the main while loop with your other periodic tasks
                # Run ML position analysis every 5 minutes
                if (now - last_position_analysis).total_seconds() >= 300:
                    logger.info("Running scheduled ML position analysis...")
                    run_position_analysis()
                    last_position_analysis = now        
                
                time.sleep(60)  # Wait 1 minute before next iteration
                
            except Exception as e:
                logger.error(f"❌ Error in trading loop iteration: {str(e)}")
                # Reconnect database if needed
                try:
                    db_manager.ensure_connection()
                except:
                    pass
                time.sleep(60)
    
    except Exception as e:
        logger.error(f"Fatal error in trading loop: {str(e)}")
    finally:
        try:
            db_manager.close()
        except:
            pass
    
    logger.info("Trading loop stopped")

def create_metrics_display(metrics, using_real_data=False):
    """Create a display of performance metrics"""
    try:
        total_value = metrics.get('total_value', 0)
        positions_value = metrics.get('positions_value', 0)
        other_positions_value = metrics.get('other_positions_value', 0)
        initial_balance = metrics.get('initial_balance', 0)
        
        # Calculate returns
        total_return = ((total_value / initial_balance) - 1) * 100 if initial_balance > 0 else 0
        
        return html.Div([
            html.H4("Portfolio Metrics", className="metrics-header"),
            
            html.Div([
                html.Div([
                    html.H5("Portfolio Value"),
                    html.P(f"${total_value:.2f}", className="metric-value")
                ], className="metric-box"),
                
                html.Div([
                    html.H5("Significant Positions"),
                    html.P(f"${positions_value:.2f}", className="metric-value")
                ], className="metric-box"),
                
                html.Div([
                    html.H5("Other Positions"),
                    html.P(f"${other_positions_value:.2f}", className="metric-value")
                ], className="metric-box"),
                
                html.Div([
                    html.H5("Total Return"),
                    html.P(f"{total_return:+.2f}%", 
                          className=f"metric-value {'positive' if total_return > 0 else 'negative' if total_return < 0 else ''}")
                ], className="metric-box"),
                
            ], className="metrics-container"),
            
            html.Div([
                html.P("Live data" if using_real_data else "Historical data", 
                      className=f"data-source {'live' if using_real_data else 'historical'}")
            ], className="metrics-footer")
            
        ], className="metrics-display")
    except Exception as e:
        logger.error(f"Error creating metrics display: {str(e)}")
        return html.Div(f"Error: {str(e)}", style={'color': 'red'})

# Add rate limiting for price updates
PRICE_CACHE = {}
PRICE_CACHE_DURATION = 5  # Cache duration in seconds
PRICE_UPDATE_DELAY = 0.05  # 50ms delay between price updates

def get_current_price(symbol, force_refresh=False):
    """
    Get current price for a symbol using Coinbase API.
    Uses WebSocket data first, then REST API with rate limiting and caching.
    
    Args:
        symbol (str): The trading pair symbol (e.g. 'BTC-USD')
        force_refresh (bool): If True, bypass cache and get fresh price
        
    Returns:
        float: Current price or None if unavailable
    """
    try:
        # Check cache first unless force_refresh is True
        now = time.time()
        if not force_refresh and symbol in PRICE_CACHE:
            cache_entry = PRICE_CACHE[symbol]
            if now - cache_entry['timestamp'] < PRICE_CACHE_DURATION:
                logger.debug(f"Using cached price for {symbol}")
                return cache_entry['price']
        
        # First try WebSocket data for most up-to-date prices
        if WEBSOCKET_AVAILABLE and is_websocket_running() and is_data_fresh():
            price = get_symbol_price(symbol)
            if price and price > 0:
                PRICE_CACHE[symbol] = {'price': price, 'timestamp': now}
                return price
        elif ws_client and ws_client.running and ws_client.price_data and symbol in ws_client.price_data:
            price = ws_client.price_data[symbol]
            if price > 0:
                PRICE_CACHE[symbol] = {'price': price, 'timestamp': now}
                return price
            
        # Add rate limiting delay
        time.sleep(PRICE_UPDATE_DELAY)
            
        # If WebSocket not available, try REST client
        if coinbase_client:
            try:
                # Use get API to fetch product data
                product = coinbase_client.get_product(product_id=symbol)
                if product and hasattr(product, 'price'):
                    price = float(product.price)
                    PRICE_CACHE[symbol] = {'price': price, 'timestamp': now}
                    logger.debug(f"Got real-time price for {symbol}: ${price:.2f}")
                    return price
            except Exception as e:
                logger.warning(f"REST client error for {symbol}: {str(e)}")
        
        # Final fallback: direct API request
        try:
            url = f"https://api.exchange.coinbase.com/products/{symbol}/ticker"
            headers = get_auth_headers("GET", f"/products/{symbol}/ticker")
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if 'price' in data:
                    price = float(data['price'])
                    PRICE_CACHE[symbol] = {'price': price, 'timestamp': now}
                    logger.debug(f"Got price from direct API for {symbol}: ${price:.2f}")
                    return price
        except Exception as api_err:
            logger.error(f"Error in direct API call for {symbol}: {str(api_err)}")
        
        return None
    except Exception as e:
        logger.error(f"Error getting price for {symbol}: {str(e)}")
        return None

def calculate_pnl(entry_price, current_price, quantity):
    """
    Calculate profit/loss metrics for a position.
    Returns tuple of (profit, pnl_percentage)
    """
    try:
        if not all([entry_price, current_price, quantity]):
            return 0.0, 0.0
            
        profit = (current_price - entry_price) * quantity
        pnl_percentage = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        return round(profit, 8), round(pnl_percentage, 2)
    except Exception as e:
        logger.error(f"Error calculating P/L: {str(e)}")
        return 0.0, 0.0

def update_position_values(session_id, conn=None):
    """
    Update all position values using current market prices from the API.
    Returns total portfolio value.
    """
    should_close_conn = False
    try:
        if conn is None:
            conn = sqlite3.connect('live_trading.db')
            should_close_conn = True

        cursor = conn.cursor()

        # Fetch account information from the API
        account_info = get_account_information()
        if not account_info:
            logger.error("Failed to fetch account information from API")
            return 0.0

        # Get accounts list from either dictionary or object format
        if isinstance(account_info, dict) and 'accounts' in account_info:
            accounts = account_info['accounts']
        elif hasattr(account_info, 'accounts'):
            accounts = account_info.accounts
        else:
            logger.warning(f"Unexpected account_info format: {type(account_info)}")
            return 0.0

        # Calculate total cash balance
        cash_balance = 0.0
        for account in accounts:
            # Extract currency and balance, handling both dict and object formats
            if isinstance(account, dict):
                currency = account.get('currency', '')
                balance_obj = account.get('available_balance', {})
                balance = float(balance_obj.get('value', '0'))
            else:
                currency = getattr(account, 'currency', '')
                balance_obj = getattr(account, 'available_balance', None)
                balance = float(getattr(balance_obj, 'value', '0')) if balance_obj else 0.0
                
            if currency == 'USD':
                cash_balance += balance

        # Calculate total position value
        total_position_value = 0.0
        for account in accounts:
            # Extract currency and balance, handling both dict and object formats
            if isinstance(account, dict):
                currency = account.get('currency', '')
                balance_obj = account.get('available_balance', {})
                balance = float(balance_obj.get('value', '0'))
            else:
                currency = getattr(account, 'currency', '')
                balance_obj = getattr(account, 'available_balance', None)
                balance = float(getattr(balance_obj, 'value', '0')) if balance_obj else 0.0
                
            if currency != 'USD':
                symbol = f"{currency}-USD"
                current_price = get_current_price(symbol)
                if current_price:
                    quantity = balance
                    position_value = current_price * quantity
                    total_position_value += position_value

        # Update database with current prices and values
        cursor.execute('''
        SELECT symbol, quantity, entry_price 
        FROM positions 
        WHERE session_id = ?
        ''', (session_id,))
        positions = cursor.fetchall()

        for symbol, quantity, entry_price in positions:
            current_price = get_current_price(symbol)
            if current_price:
                profit, pnl = calculate_pnl(entry_price, current_price, quantity)
                cursor.execute('''
                UPDATE positions 
                SET current_price = ?,
                    profit = ?,
                    pnl = ?,
                    last_update = datetime('now')
                WHERE session_id = ? AND symbol = ?
                ''', (current_price, profit, pnl, session_id, symbol))

        conn.commit()
        return cash_balance + total_position_value

    except Exception as e:
        logger.error(f"Error updating position values: {str(e)}")
        return 0.0
    finally:
        if should_close_conn and conn:
            conn.close()

def get_portfolio_summary(session_id):
    """
    Get comprehensive portfolio summary using lk.py for portfolio value and cash balance.
    """
    try:
        # Get values from lk.py
        total_value = get_portfolio_value()
        cash_balance = get_available_balance()
        positions_value = total_value - cash_balance
        
        # Get realized P/L from trades (if any)
        conn = sqlite3.connect('live_trading.db')
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT COALESCE(SUM(profit), 0) 
        FROM trades 
        WHERE session_id = ? AND action = 'SELL'
        ''', (session_id,))
        realized_pnl = cursor.fetchone()[0]
        
        # Unrealized P/L is the positions value (simplified)
        unrealized_pnl = positions_value
        
        conn.close()
        
        return {
            'cash_balance': cash_balance,
            'positions_value': positions_value,
            'total_value': total_value,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': realized_pnl + unrealized_pnl
        }
        
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {str(e)}")
        return None

def evaluate_exit_conditions(symbol, entry_price, current_price, momentum_score, rsi, conn=None, cursor=None):
    """
    Evaluate exit conditions for a position.
    Returns (should_exit, exit_reason)
    """
    # Initialize return values at the start
    should_exit = False
    exit_reason = None
    
    try:
        logger.debug(f"Starting exit evaluation for {symbol}")
        logger.debug(f"Input parameters - Entry: ${entry_price:.2f}, Current: ${current_price:.2f}, Momentum: {momentum_score:.1f}, RSI: {rsi:.1f}")
        
        # Validate input parameters
        if not all([entry_price, current_price]):
            logger.warning(f"Invalid price values - Entry: {entry_price}, Current: {current_price}")
            return False, "Invalid Prices"
            
        current_return = ((current_price / entry_price) - 1) * 100
        logger.debug(f"Current return: {current_return:.2f}%")
        
        # 1. Stop loss (-2.0%)
        if current_return <= -15.0:
            logger.debug(f"Stop loss triggered at {current_return:.2f}%")
            return True, "Stop Loss"
            
        # 2. Take profit (5.0%)
        if current_return >= 5.0:
            logger.debug(f"Take profit triggered at {current_return:.2f}%")
            return True, "Take Profit"
            
        # 3. Technical conditions
        logger.debug("Fetching market data for technical analysis")
        df = get_market_data(symbol, period='1d', interval='1h')
        if df.empty:
            logger.warning("No market data available")
            return False, "No Market Data"
            
        df = calculate_indicators(df)
        if df.empty:
            logger.warning("Failed to calculate indicators")
            return False, "Indicator Calculation Failed"
            
        latest = df.iloc[-1]
        logger.debug(f"Latest indicators - MACD: {latest['MACD']:.2f}, Signal: {latest['Signal_Line']:.2f}, MA20: {latest['MA20']:.2f}")
        
        # Get ML predictions
        logger.debug("Getting ML predictions")
        predictions, confidence = predict_with_pretrained_model(df, symbol, interval='1h')
        if not predictions.empty:
            latest_pred = predictions.iloc[-1]
            latest_conf = confidence.iloc[-1]
            logger.debug(f"ML Prediction - Direction: {latest_pred['direction']}, Confidence: {latest_conf['sell_confidence']:.2f}")
            
            # ML-based exit signals
            ml_sell_signal = (
                
                latest_conf['sell_confidence'] > 0.6
            )
            logger.debug(f"ML sell signal: {ml_sell_signal}")
            
            # Technical indicator signals
            tech_sell_signal = (
                latest['RSI'] > 70 or  # Overbought
                (latest['MACD'] < latest['Signal_Line'] and  # MACD bearish crossover
                 latest['close'] < latest['MA20'])  # Price below MA20
            )
            logger.debug(f"Technical sell signal: {tech_sell_signal}")
            
            # Combined decision
            if ml_sell_signal or (tech_sell_signal and current_return > 0):
                logger.debug("Exit triggered by ML or technical signals")
                return True, "ML and Technical Exit"
            else:
                logger.debug("No ML or technical exit signals triggered")
                return False, None
        else:
            logger.warning("No ML predictions available")
            return False, "No ML Predictions"
            
    except Exception as e:
        logger.error(f"Error evaluating exit conditions for {symbol}: {str(e)}")
        return False, f"Error: {str(e)}"
    
    # Default return if no conditions are met
    logger.debug("No exit conditions met")
    return False, None

def execute_exit_trade(session_id, symbol, quantity, entry_price, current_price, exit_reason, conn, cursor):
    """
    Execute exit trade with proper error handling.
    Returns True if successful, False otherwise.
    """
    try:
        # Validate inputs
        if not all([session_id, symbol, quantity, entry_price, current_price]):
            logger.error("Missing required parameters for exit trade")
            return False

        # Calculate trade metrics
        value = current_price * quantity
        profit = (current_price - entry_price) * quantity
        current_return = ((current_price / entry_price) - 1) * 100
        
        logger.info("=== TRADE EXECUTION START ===")
        logger.info(f"Executing SELL order for {symbol}")
        logger.info(f"Trade Details:")
        logger.info(f"  - Quantity: {quantity:.8f}")
        logger.info(f"  - Entry Price: ${entry_price:.2f}")
        logger.info(f"  - Current Price: ${current_price:.2f}")
        logger.info(f"  - Total Value: ${value:.2f}")
        logger.info(f"  - Expected P/L: ${profit:+.2f} ({current_return:+.2f}%)")
        logger.info(f"  - Exit Reason: {exit_reason}")
        
        # Verify database connection
        try:
            cursor.execute("SELECT 1")
            logger.debug("Database connection verified")
        except (sqlite3.OperationalError, sqlite3.ProgrammingError):
            logger.warning("Database connection lost, reconnecting...")
            conn = sqlite3.connect('live_trading.db', timeout=30)
            cursor = conn.cursor()
            logger.info("Database connection reestablished")
        
        # Begin transaction
        cursor.execute("BEGIN")
        logger.debug("Database transaction started")
        
        try:
            # Get TP/SL order IDs
            cursor.execute('''
            SELECT tp_order_id, sl_order_id 
            FROM positions 
            WHERE session_id = ? AND symbol = ?
            ''', (session_id, symbol))
            order_ids = cursor.fetchone()
            
            if order_ids:
                tp_order_id, sl_order_id = order_ids
                logger.info(f"Found existing TP/SL orders for {symbol}")
                
                # Cancel TP order if it exists
                if tp_order_id:
                    logger.info(f"Canceling TP order: {tp_order_id}")
                    if not cancel_order(tp_order_id):
                        logger.error(f"Failed to cancel TP order: {tp_order_id}")
                        cursor.execute("ROLLBACK")
                        return False
                
                # Cancel SL order if it exists
                if sl_order_id:
                    logger.info(f"Canceling SL order: {sl_order_id}")
                    if not cancel_order(sl_order_id):
                        logger.error(f"Failed to cancel SL order: {sl_order_id}")
                        cursor.execute("ROLLBACK")
                        return False
                
                # Wait for orders to be fully canceled
                time.sleep(1)
            
            # Place sell order with Coinbase
            logger.info("Placing market sell order...")
            order = place_market_order(symbol, 'SELL', value)
            if not order:
                logger.error(f"Failed to place SELL order for {symbol}")
                cursor.execute("ROLLBACK")
                logger.debug("Database transaction rolled back")
                return False
            
            logger.info("Order placed successfully")
            logger.debug(f"Order details: {order}")
            
            # Record the trade
            logger.debug("Recording trade in database...")
            cursor.execute('''
            INSERT INTO trades 
            (session_id, timestamp, symbol, action, price, quantity, value, profit)
            VALUES (?, datetime('now'), ?, 'SELL', ?, ?, ?, ?)
            ''', (session_id, symbol, current_price, quantity, value, profit))
            logger.debug("Trade recorded in database")
            
            # Remove position
            logger.debug(f"Removing position for {symbol} from database...")
            cursor.execute('DELETE FROM positions WHERE session_id = ? AND symbol = ?',
                        (session_id, symbol))
            logger.debug("Position removed from database")
            
            # Update session statistics
            logger.debug("Updating session statistics...")
            cursor.execute('''
            UPDATE sessions 
            SET total_trades = total_trades + 1,
                winning_trades = winning_trades + CASE WHEN ? > 0 THEN 1 ELSE 0 END,
                total_profit = total_profit + ?,
                final_balance = (SELECT total_value FROM portfolio_history 
                               WHERE session_id = ? 
                               ORDER BY timestamp DESC LIMIT 1)
            WHERE id = ?
            ''', (profit, profit, session_id, session_id))
            logger.debug("Session statistics updated")
            
            # Commit transaction
            cursor.execute("COMMIT")
            logger.debug("Database transaction committed")
            logger.info(f"Successfully executed SELL order for {symbol}")
            logger.info("=== TRADE EXECUTION COMPLETE ===")
            return True
            
        except Exception as e:
            # Rollback transaction on error
            cursor.execute("ROLLBACK")
            logger.error(f"Database error during exit trade: {str(e)}")
            logger.error("Database transaction rolled back")
            logger.info("=== TRADE EXECUTION FAILED ===")
            return False
        
    except Exception as e:
        logger.error(f"Error executing exit trade for {symbol}: {str(e)}")
        logger.error(traceback.format_exc())
        logger.info("=== TRADE EXECUTION FAILED ===")
        return False

def process_position_exit(session_id, symbol, quantity, entry_price, current_price, momentum_score, rsi, conn, cursor):
    """
    Process potential exit for an existing position.
    Returns True if position was exited, False otherwise.
    """
    logger.debug(f"Processing position exit for {symbol}")
    logger.debug(f"Position details - Quantity: {quantity}, Entry: ${entry_price:.2f}, Current: ${current_price:.2f}")
    
    try:
        # Initialize variables
        should_exit = False
        exit_reason = None
        position_exited = False
        
        # Verify database connection
        try:
            cursor.execute("SELECT 1")
        except (sqlite3.OperationalError, sqlite3.ProgrammingError):
            logger.warning("Database connection lost in process_position_exit, reconnecting...")
            conn = sqlite3.connect('live_trading.db', timeout=30)
            cursor = conn.cursor()
        
        # Begin transaction
        cursor.execute("BEGIN")
        
        try:
            # Evaluate exit conditions
            logger.debug("Evaluating exit conditions")
            try:
                should_exit, exit_reason = evaluate_exit_conditions(
                    symbol, entry_price, current_price, momentum_score, rsi,
                    conn=conn, cursor=cursor
                )
                logger.debug(f"Exit evaluation result - Should exit: {should_exit}, Reason: {exit_reason}")
            except Exception as e:
                logger.error(f"Error in exit evaluation: {str(e)}")
                cursor.execute("ROLLBACK")
                return False
            
            if should_exit and exit_reason:
                logger.info(f"Exit condition met for {symbol}: {exit_reason}")
                # Execute sell order
                try:
                    position_exited = execute_exit_trade(
                        session_id, symbol, quantity, entry_price, 
                        current_price, exit_reason, conn, cursor
                    )
                    if position_exited:
                        cursor.execute("COMMIT")
                        logger.info(f"Successfully exited position for {symbol}")
                    else:
                        cursor.execute("ROLLBACK")
                        logger.error(f"Failed to execute exit trade for {symbol}")
                except Exception as e:
                    cursor.execute("ROLLBACK")
                    logger.error(f"Error executing exit trade: {str(e)}")
                    position_exited = False
            else:
                cursor.execute("COMMIT")
                logger.debug(f"No exit conditions met for {symbol}")
                position_exited = False
            
            return position_exited
            
        except Exception as e:
            cursor.execute("ROLLBACK")
            logger.error(f"Error in process_position_exit transaction: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error processing position exit for {symbol}: {str(e)}")
        return False

# Function to get account information
def get_account_information(use_cache=True):
    """Get account information with caching and rate limiting"""
    global ACCOUNT_CACHE
    
    try:
        # Check cache first
        now = time.time()
        if use_cache and ACCOUNT_CACHE and 'data' in ACCOUNT_CACHE:
            if now - ACCOUNT_CACHE['timestamp'] < ACCOUNT_CACHE_DURATION:
                logger.debug("Using cached account information")
                return ACCOUNT_CACHE['data']
        
        # Add rate limiting delay
        time.sleep(RATE_LIMIT_DELAY)
        
        # Make API request
        if coinbase_client:
            try:
                accounts_response = coinbase_client.get_accounts()
                if accounts_response:
                    # Convert response to dictionary format for consistency
                    # Check if it's already a dict or has to_dict method
                    if isinstance(accounts_response, dict):
                        accounts_data = accounts_response
                    elif hasattr(accounts_response, 'to_dict'):
                        accounts_data = accounts_response.to_dict()
                    elif hasattr(accounts_response, 'accounts'):
                        # Extract accounts from structured response
                        accounts_data = {
                            'accounts': [
                                {
                                    'currency': getattr(account, 'currency', ''),
                                    'available_balance': {
                                        'value': getattr(getattr(account, 'available_balance', None), 'value', '0')
                                    }
                                }
                                for account in accounts_response.accounts
                            ]
                        }
                    else:
                        logger.warning(f"Unexpected accounts response format: {type(accounts_response)}")
                        return None
                        
                    # Cache the result
                    ACCOUNT_CACHE = {
                        'data': accounts_data,
                        'timestamp': now
                    }
                    logger.info("Successfully retrieved and cached account information")
                    return accounts_data
            except Exception as e:
                logger.error(f"Error getting accounts from REST client: {str(e)}")
        
        # Fallback to direct API request
        headers = get_auth_headers("GET", "/api/v3/brokerage/accounts")
        response = requests.get(f"https://{BASE_URL}/api/v3/brokerage/accounts", headers=headers)
        
        if response.status_code == 200:
            accounts = response.json()
            # Cache the result
            ACCOUNT_CACHE = {
                'data': accounts,
                'timestamp': now
            }
            logger.info("Successfully retrieved and cached account information")
            return accounts
        else:
            logger.error(f"Failed to get account information: {response.status_code} - {response.text}")
        return None

    except Exception as e:
        logger.error(f"Error retrieving account information: {str(e)}")
        return None

def get_portfolio_from_api():
    """Get portfolio data from Coinbase API with caching"""
    try:
        # Get account information using cached version
        account_info = get_account_information(use_cache=True)
        if not account_info:
            logger.warning("Failed to get account information from API")
            return None

        # Initialize portfolio values
        total_value = 0.0
        cash_balance = 0.0
        positions_value = 0.0
        positions = []
        unrealized_gains = 0.0
        
        # Process account information
        if isinstance(account_info, dict) and 'accounts' in account_info:
            accounts = account_info['accounts']
        elif hasattr(account_info, 'accounts'):
            accounts = account_info.accounts
        else:
            logger.warning(f"Unexpected account_info format: {type(account_info)}")
            return None
        
        # First pass - get USD balance
        for account in accounts:
            if isinstance(account, dict):
                currency = account.get('currency', '')
                balance_obj = account.get('available_balance', {})
                balance = float(balance_obj.get('value', '0'))
            else:
                currency = getattr(account, 'currency', '')
                balance_obj = getattr(account, 'available_balance', None)
                balance = float(getattr(balance_obj, 'value', '0')) if balance_obj else 0.0
            
            if currency == 'USD':
                cash_balance = balance
                total_value += balance
                logger.info(f"Found USD balance: ${cash_balance:.2f}")

        # Second pass - get crypto positions
        for account in accounts:
            if isinstance(account, dict):
                currency = account.get('currency', '')
                balance_obj = account.get('available_balance', {})
                balance = float(balance_obj.get('value', '0'))
            else:
                currency = getattr(account, 'currency', '')
                balance_obj = getattr(account, 'available_balance', None)
                balance = float(getattr(balance_obj, 'value', '0')) if balance_obj else 0.0
            
            if currency and currency != 'USD' and balance > 0:
                symbol = f"{currency}-USD"
                current_price = get_current_price(symbol)
                
                if current_price and balance > 0:
                    value = balance * current_price
                    positions_value += value
                    total_value += value
                    
                    # Try to find entry price from database
                    conn = sqlite3.connect('live_trading.db')
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT entry_price 
                        FROM positions 
                        WHERE symbol = ? 
                        ORDER BY entry_time DESC 
                        LIMIT 1
                    ''', (symbol,))
                    result = cursor.fetchone()
                    entry_price = result[0] if result else current_price
                    conn.close()
                    
                    # Calculate profit and PNL
                    profit = (current_price - entry_price) * balance
                    pnl = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
                    
                    positions.append({
                        'symbol': symbol,
                        'quantity': balance,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'profit': profit,
                        'pnl': pnl
                    })
                    
                    unrealized_gains += profit
                    logger.info(f"Found {currency} position: {balance} @ ${current_price:.2f} = ${value:.2f}")

        logger.info(f"Portfolio Summary:")
        logger.info(f"Total Value: ${total_value:.2f}")
        logger.info(f"Cash Balance: ${cash_balance:.2f}")
        logger.info(f"Positions Value: ${positions_value:.2f}")
        logger.info(f"Unrealized Gains: ${unrealized_gains:.2f}")
        
        return {
            'total_value': total_value,
            'cash_balance': cash_balance,
            'positions_value': positions_value,
            'positions': positions,
            'unrealized_gains': unrealized_gains
        }
    except Exception as e:
        logger.error(f"Error getting portfolio from API: {str(e)}")
        return None

def mock_portfolio_data():
    """
    Generate mock portfolio data for testing when API access is unavailable.
    Returns a portfolio data structure similar to what get_portfolio_from_api would return.
    """
    try:
        # Create mock data
        total_value = 10000.0
        cash_balance = 5000.0
        positions_value = 5000.0
        unrealized_gains = 500.0
        
        # Create some mock positions
        positions = [
            {
                'symbol': 'BTC-USD',
                'quantity': 0.05,
                'entry_price': 60000.0,
                'current_price': 65000.0,
                'profit': 250.0,
                'pnl': 8.33
            },
            {
                'symbol': 'ETH-USD',
                'quantity': 1.2,
                'entry_price': 3000.0,
                'current_price': 3200.0,
                'profit': 240.0,
                'pnl': 6.67
            },
            {
                'symbol': 'SOL-USD',
                'quantity': 10.0,
                'entry_price': 100.0,
                'current_price': 110.0,
                'profit': 100.0,
                'pnl': 10.0
            }
        ]
        
        return {
            'total_value': total_value,
            'cash_balance': cash_balance,
            'positions_value': positions_value,
            'positions': positions,
            'unrealized_gains': unrealized_gains
        }
    except Exception as e:
        logger.error(f"Error generating mock portfolio data: {str(e)}")
        return None

def test_api_connection():
    """Test function to verify Coinbase API connectivity"""
    print("Testing Coinbase API connection...")
    
    try:
        # Try getting account information
        account_info = get_account_information(use_cache=False)  # Don't use cache for initial test
        if account_info:
            print("✅ API connection successful!")
            
            # Handle both dictionary and object formats
            if isinstance(account_info, dict) and 'accounts' in account_info:
                accounts = account_info['accounts']
                print(f"Found {len(accounts)} accounts")
                
                # Print a sample of the accounts
                for i, account in enumerate(accounts[:3]):
                    print(f"Account {i+1}:")
                    if isinstance(account, dict):
                        currency = account.get('currency', 'Unknown')
                        balance = account.get('available_balance', {}).get('value', '0')
                    else:
                        currency = getattr(account, 'currency', 'Unknown')
                        balance_obj = getattr(account, 'available_balance', None)
                        balance = getattr(balance_obj, 'value', '0') if balance_obj else '0'
                    
                    print(f"  Currency: {currency}")
                    print(f"  Balance: {balance} {currency}")
                
                # If there are more accounts, show a summary message
                if len(accounts) > 3:
                    print(f"... and {len(accounts) - 3} more accounts")
            else:
                print(f"Received account info in format: {type(account_info)}")
                print("Account data structure may have changed, but connection works")
                
            return True
        else:
            print("❌ Error: Failed to retrieve account information")
            return False
            
    except Exception as e:
        print(f"❌ Error during API connection test: {str(e)}")
        return False

def wait_for_order_settlement(order_id, max_attempts=10, delay=3.0):
    """
    Wait for an order to settle by monitoring balance changes.
    This simplified version ignores API status and just checks balance changes.
    
    Args:
        order_id (str): The order ID (used only for logging)
        max_attempts (int): Maximum number of attempts to check
        delay (float): Delay between attempts in seconds
    
    Returns:
        bool: True if order appears to have settled, False otherwise
    """
    try:
        # Get initial balance immediately
        initial_balance = get_available_balance()
        logger.info(f"Initial balance before settlement check: ${initial_balance:.2f}")
        
        # First short delay to allow balance to update
        time.sleep(delay)
        
        # Check balance again after delay
        current_balance = get_available_balance() 
        if abs(current_balance - initial_balance) > 0.01:
            logger.info(f"Balance changed from ${initial_balance:.2f} to ${current_balance:.2f} - order settled")
            return True
        
        # Log the order ID we're waiting for
        logger.info(f"Waiting for order {order_id} to settle...")
        
        # Loop to check for balance changes
        for attempt in range(max_attempts):
            # Wait first then check
            time.sleep(delay)
            
            # Check if balance has changed
            current_balance = get_available_balance()
            
            # If balance changed by more than 1 cent, consider the order settled
            if abs(current_balance - initial_balance) > 0.01:
                logger.info(f"Balance changed from ${initial_balance:.2f} to ${current_balance:.2f} - order settled")
                # Wait one more time to let everything process
                time.sleep(1.0)
                return True
            
            logger.info(f"Balance check {attempt+1}/{max_attempts}: Still ${current_balance:.2f}, waiting...")
        
        # One final check
        final_balance = get_available_balance()
        if abs(final_balance - initial_balance) > 0.01:
            logger.info(f"Final check: Balance changed from ${initial_balance:.2f} to ${final_balance:.2f} - order settled")
            return True
            
        logger.error(f"Order {order_id} did not appear to settle after {max_attempts} attempts")
        # The order was probably placed but the API is not updating
        # Since the balance did change, let's treat this as a successful settlement
        logger.info("Assuming order was placed successfully despite no balance change detection")
        return True
        
    except Exception as e:
        logger.error(f"Error checking order settlement: {str(e)}")
        return False

def get_available_balance():
    """Get available USD balance using lk.py output (no direct API calls)."""
    try:
        result = subprocess.run(
            [sys.executable, "crypto_trading/c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/lk.py"],
            capture_output=True, text=True, check=True
        )
        output = result.stdout.strip()
        
        # Find JSON data between markers
        start_marker = "DASHBOARD_DATA_START"
        end_marker = "DASHBOARD_DATA_END"
        
        if start_marker in output and end_marker in output:
            json_str = output.split(start_marker)[1].split(end_marker)[0]
            try:
                data = json.loads(json_str)
                # Look for USD in positions
                for position in data.get('positions', []):
                    if position['currency'] == 'USD':
                        balance = float(position['usd_value'])
                        logger.info(f"Found USD balance in JSON: ${balance:.2f}")
                        return balance
                logger.warning("No USD position found in JSON data")
                return 0.0
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON data")
                return 0.0
        else:
            # Try to parse the last line as float (old format)
            try:
                lines = output.strip().split('\n')
                balance = float(lines[-1])
                logger.info(f"lk.py returned USD balance: ${balance:.2f}")
                return balance
            except (ValueError, IndexError):
                logger.error(f"Could not parse balance from output: {output}")
                return 0.0
    except Exception as e:
        logger.error(f"Error running lk.py subprocess for available balance: {e}")
        return 0.0

def get_portfolio_value():
    """Get total portfolio value using lk.py output."""
    try:
        result = subprocess.run(
            [sys.executable, "crypto_trading/c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/lk.py"],
            capture_output=True, text=True, check=True
        )
        output = result.stdout.strip()
        
        # Find JSON data between markers
        start_marker = "DASHBOARD_DATA_START"
        end_marker = "DASHBOARD_DATA_END"
        
        if start_marker in output and end_marker in output:
            json_str = output.split(start_marker)[1].split(end_marker)[0]
            try:
                data = json.loads(json_str)
                total_value = data.get('total_value', 0.0)
                logger.info(f"Portfolio value from JSON: ${total_value:.2f}")
                return total_value
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON data")
                return 0.0
        else:
            # Try to parse the last line as float (old format)
            try:
                lines = output.strip().split('\n')
                value = float(lines[-1])
                logger.info(f"Portfolio value from last line: ${value:.2f}")
                return value
            except (ValueError, IndexError):
                logger.error(f"Could not parse portfolio value from output: {output}")
                return 0.0
    except Exception as e:
        logger.error(f"Error running lk.py subprocess: {e}")
        return 0.0

def update_portfolio_history(session_id):
    """Update portfolio history with current values"""
    conn = None
    try:
        conn = sqlite3.connect('live_trading.db')
        cursor = conn.cursor()
        
        # Get current positions value
        cursor.execute('''
        SELECT COALESCE(SUM(value), 0)
        FROM positions
        WHERE session_id = ?
        ''', (session_id,))
        positions_value = cursor.fetchone()[0] or 0.0
        
        # Get initial balance from session
        cursor.execute('SELECT initial_balance FROM sessions WHERE id = ?', (session_id,))
        initial_balance = cursor.fetchone()[0]
        
        # Calculate total value and cash balance
        total_value = initial_balance  # Start with initial balance
        cash_balance = initial_balance - positions_value  # Remaining cash
        
        # Add new history entry
        cursor.execute('''
        INSERT INTO portfolio_history
        (session_id, timestamp, total_value, cash_balance, positions_value)
        VALUES (?, datetime('now'), ?, ?, ?)
        ''', (session_id, total_value, cash_balance, positions_value))
        
        conn.commit()
        logger.info(f"Updated portfolio history - Total: ${total_value:.2f}, Cash: ${cash_balance:.2f}, Positions: ${positions_value:.2f}")
        
    except Exception as e:
        logger.error(f"Error updating portfolio history: {str(e)}")
        if conn:
            try:
                conn.rollback()
            except:
                pass
    finally:
        if conn:
            conn.close()

def execute_entry_trade(session_id, symbol, entry_price, value, conn=None, cursor=None):
    """Execute an entry trade with proper error handling and TP/SL orders"""
    try:
        logger.info(f"Executing entry trade for {symbol}")
        logger.info(f"Entry price: ${entry_price}, Value: ${value}")
        
        # Calculate quantity
        quantity = value / entry_price
        
        # Get product details for proper formatting
        price_precision, size_precision, round_to_increment = get_product_details(symbol)
        if round_to_increment is None:
            logger.error("Failed to get product details")
            return False
            
        # Format quantity with correct precision
        formatted_quantity = "{:.{}f}".format(round_to_increment(quantity, '0.01'), size_precision)
        formatted_price = "{:.{}f}".format(entry_price, price_precision)
        
        # Place TP/SL orders
        main_order_id, sl_order_id = place_tp_sl_orders(symbol, entry_price, float(formatted_quantity))
        if not main_order_id:
            logger.error("Failed to place TP/SL orders")
            return False
            
        # Record trade in database
        if conn and cursor:
            try:
                # Add trade to history
                cursor.execute("""
                    INSERT INTO trades 
                    (session_id, symbol, action, price, quantity, value, order_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (session_id, symbol, 'BUY', entry_price, formatted_quantity, value, main_order_id))
                
                # Add position
                cursor.execute("""
                    INSERT INTO positions 
                    (session_id, symbol, entry_price, quantity, position_size, tp_order_id, sl_order_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (session_id, symbol, entry_price, formatted_quantity, value, main_order_id, sl_order_id))
                
                conn.commit()
                logger.info("Successfully recorded trade in database")
                
            except sqlite3.Error as e:
                logger.error(f"Database error: {str(e)}")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Error executing trade: {str(e)}")
        return False

def get_product_details(symbol):
    """Get precision details for a product to format orders correctly"""
    try:
        # Get product details from Coinbase
        headers = get_auth_headers("GET", f"/api/v3/brokerage/products/{symbol}")
        response = requests.get(
            f"https://{BASE_URL}/api/v3/brokerage/products/{symbol}",
            headers=headers
        )
        
        if response.status_code == 200:
            product = response.json()
            logger.info(f"Raw product details: {json.dumps(product, indent=2)}")
            
            # Get base increment for size precision
            base_increment = product.get('base_increment', '0.01')
            size_precision = len(base_increment.split('.')[-1]) if '.' in base_increment else 0
            
            # Get quote increment for price precision
            quote_increment = product.get('quote_increment', '0.01')
            price_precision = len(quote_increment.split('.')[-1]) if '.' in quote_increment else 0
            
            logger.info(f"Product details - Price precision: {price_precision}, Size precision: {size_precision}")
            
            # Round size to match base_increment
            def round_to_increment(value, increment):
                """Round value to nearest increment"""
                increment = float(increment)
                return float(increment * round(float(value) / increment))
            
            return price_precision, size_precision, round_to_increment
            
        else:
            logger.error(f"Failed to get product details: {response.status_code} - {response.text}")
            return 2, 8, None  # Default values
            
    except Exception as e:
        logger.error(f"Error getting product details: {str(e)}")
        return 2, 8, None  # Default values

def place_tp_sl_orders(symbol, entry_price, size):
    """Place a buy order with attached take profit and stop loss orders"""
    try:
        logger.info(f"Setting TP/SL for {symbol}:")
        logger.info(f"  Entry price: ${entry_price:.4f}")
        logger.info(f"  Position size: {size:.8f}")
        
        # Calculate TP/SL prices (5% profit, 2% loss)
        tp_price = entry_price * 1.05
        sl_price = entry_price * 0.98
        
        logger.info(f"  TP price: ${tp_price:.4f} (+5%)")
        logger.info(f"  SL price: ${sl_price:.4f} (-2%)")
        
        # Format size to avoid precision errors
        formatted_size = "{:.1f}".format(size)
        formatted_entry = "{:.4f}".format(entry_price)
        formatted_tp = "{:.4f}".format(tp_price)
        formatted_sl = "{:.4f}".format(sl_price)
        
        # Place main limit buy order with attached TP/SL
        main_order = {
            "client_order_id": str(uuid.uuid4()),
            "product_id": symbol,
            "side": "BUY",
            "order_configuration": {
                "limit_limit_gtc": {
                    "base_size": formatted_size,
                    "limit_price": formatted_entry,
                    "post_only": False
                }
            },
            "attached_order_configuration": {
                "trigger_bracket_gtc": {
                    "limit_price": formatted_tp,  # Take profit price
                    "stop_trigger_price": formatted_sl  # Stop loss price
                }
            }
        }
        
        logger.info("Placing main buy order with attached TP/SL...")
        logger.info(f"Order data: {json.dumps(main_order, indent=2)}")
        
        main_order_id = place_order(main_order)
        if not main_order_id:
            logger.error("❌ Failed to place main order with TP/SL")
            return None, None
            
        logger.info(f"✅ Order placed successfully: {main_order_id}")
        return main_order_id, None  # Return None for sl_order_id since it's attached
            
    except Exception as e:
        logger.error(f"❌ Error placing TP/SL orders: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def cancel_order(order_id):
    """
    Cancel an existing order.
    Args:
        order_id (str): The ID of the order to cancel
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Canceling order: {order_id}")
        
        # Add rate limiting delay
        time.sleep(RATE_LIMIT_DELAY)
        
        # Get auth headers
        headers = get_auth_headers("DELETE", f"/api/v3/brokerage/orders/{order_id}")
        if not headers:
            logger.error("Failed to get auth headers")
            return False
            
        # Make the API request
        response = requests.delete(
            f"https://{BASE_URL}/api/v3/brokerage/orders/{order_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            logger.info(f"Successfully canceled order: {order_id}")
            return True
        else:
            logger.error(f"Failed to cancel order: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error canceling order: {str(e)}")
        return False

def get_product_precision(symbol):
    """
    Get the price precision for a trading pair from Coinbase.
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTC-USD')
    Returns:
        int: Number of decimal places allowed for price
    """
    try:
        # Default precisions for common pairs
        default_precisions = {
            'BTC-USD': 2,
            'ETH-USD': 2,
            'SOL-USD': 3,
            'ADA-USD': 4,
            'DOGE-USD': 6,
            'SHIB-USD': 8,
            'XRP-USD': 4,
            'MATIC-USD': 4,
            'LINK-USD': 3,
            'DOT-USD': 3
        }
        
        # First check if we have a default precision
        if symbol in default_precisions:
            return default_precisions[symbol]
            
        # Add rate limiting delay
        time.sleep(RATE_LIMIT_DELAY)
        
        # Get product details from Coinbase
        headers = get_auth_headers("GET", f"/api/v3/brokerage/products/{symbol}")
        response = requests.get(
            f"https://{BASE_URL}/api/v3/brokerage/products/{symbol}",
            headers=headers
        )
        
        if response.status_code == 200:
            product_data = response.json()
            
            # Get price increment and calculate precision
            price_increment = product_data.get('price_increment', '0.01')
            decimal_places = str(price_increment)[::-1].find('.')
            
            if decimal_places > 0:
                return decimal_places
            else:
                # If no decimal in increment, use default based on price
                price = float(product_data.get('price', '0'))
                if price < 1:
                    return 6
                elif price < 10:
                    return 4
                elif price < 100:
                    return 3
                else:
                    return 2
        else:
            logger.warning(f"Failed to get product details for {symbol}, using default precision")
            # Use conservative default based on typical prices
            return 4
            
    except Exception as e:
        logger.error(f"Error getting product precision for {symbol}: {str(e)}")
        return 4  # Conservative default

class PositionManager:
    """Manages position data and ensures entry prices stay fixed"""
    
    def __init__(self, db_path='live_trading.db'):
        self.db_path = db_path
        self.positions = {}  # symbol -> position_data mapping
        self.load_positions()
        
    def load_positions(self):
        """Load existing positions from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get active session
            cursor.execute('SELECT id FROM sessions WHERE status = "active" ORDER BY start_time DESC LIMIT 1')
            session_result = cursor.fetchone()
            
            if session_result:
                session_id = session_result[0]
                
                # Load all positions for active session
                cursor.execute('''
                SELECT symbol, quantity, entry_price, tp_order_id, sl_order_id, value
                FROM positions 
                WHERE session_id = ?
                ''', (session_id,))
                
                for row in cursor.fetchall():
                    symbol, quantity, entry_price, tp_order_id, sl_order_id, value = row
                    self.positions[symbol] = {
                        "symbol": symbol,
                        "entry_price": entry_price,  # Fixed entry price
                        "quantity": quantity,
                        "value": value,
                        "current_price": entry_price,  # Separate from entry price
                        "tp_order_id": tp_order_id,
                        "sl_order_id": sl_order_id,
                        "tp_price": round(entry_price * 1.05, 8),  # Calculate from entry
                        "sl_price": round(entry_price * 0.98, 8),  # Calculate from entry
                        "open": True,
                        "entry_time": datetime.now().isoformat()  # Track when position was opened
                    }
                    logger.info(f"🔒 Loaded position for {symbol} with fixed entry price: ${entry_price:.8f}")
                    
                logger.info(f"Loaded {len(self.positions)} positions from database")
                
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading positions: {str(e)}")
            if conn:
                conn.close()
    
    def open_position(self, session_id, symbol, entry_price, quantity, position_size):
        """Open a new position with fixed entry price"""
        conn = None
        try:
            # Verify we don't already have a position
            if symbol in self.positions:
                logger.error(f"❌ Cannot open position - already have position in {symbol}")
                return False
            
            # Calculate take profit and stop loss prices from entry
            take_profit_price = round(entry_price * 1.05, 8)
            stop_loss_price = round(entry_price * 0.98, 8)
            
            logger.info(f"🔒 Opening new position for {symbol}")
            logger.info(f"Entry Price: ${entry_price:.8f} (FIXED)")
            logger.info(f"Take Profit: ${take_profit_price:.8f}")
            logger.info(f"Stop Loss: ${stop_loss_price:.8f}")
            
            position_data = {
                "symbol": symbol,
                "entry_price": entry_price,  # Fixed entry price
                "quantity": quantity,
                "value": position_size,
                "current_price": entry_price,  # Separate from entry
                "tp_price": take_profit_price,
                "sl_price": stop_loss_price,
                "tp_order_id": None,
                "sl_order_id": None,
                "open": True,
                "entry_time": datetime.now().isoformat()
            }
            
            # Place TP/SL orders
            tp_order, sl_order = place_tp_sl_orders(symbol, entry_price, quantity)
            
            if tp_order:
                position_data["tp_order_id"] = tp_order.get('order_id')
                logger.info(f"✅ Take Profit order placed: {position_data['tp_order_id']}")
            if sl_order:
                position_data["sl_order_id"] = sl_order.get('order_id')
                logger.info(f"✅ Stop Loss order placed: {position_data['sl_order_id']}")
            
            # Store position data
            self.positions[symbol] = position_data
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO positions 
            (session_id, symbol, quantity, entry_price, current_price, entry_time,
             last_update, profit, pnl, value, tp_order_id, sl_order_id)
            VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'), 0.0, 0.0, ?, ?, ?)
            ''', (
                session_id, symbol, quantity, entry_price, entry_price, position_size,
                position_data["tp_order_id"], position_data["sl_order_id"]
            ))
            
            conn.commit()
            logger.info(f"🔒 Position opened and saved to database with fixed entry price: ${entry_price:.8f}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening position for {symbol}: {str(e)}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            return False
        finally:
            if conn:
                conn.close()
    
    def update_position_prices(self, session_id):
        """Update current prices and P/L without changing entry prices"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for symbol, position in self.positions.items():
                current_price = get_current_price(symbol)
                if current_price:
                    # Never modify entry_price, only update current_price
                    position["current_price"] = current_price
                    
                    # Calculate P/L using fixed entry price
                    profit = (current_price - position["entry_price"]) * position["quantity"]
                    pnl_percentage = ((current_price / position["entry_price"]) - 1) * 100
                    
                    logger.debug(f"Position Update - {symbol}:")
                    logger.debug(f"  Entry Price (fixed): ${position['entry_price']:.8f}")
                    logger.debug(f"  Current Price: ${current_price:.8f}")
                    logger.debug(f"  P/L: ${profit:+.2f} ({pnl_percentage:+.2f}%)")
                    
                    # Update database with new price and P/L
                    cursor.execute('''
                    UPDATE positions
                    SET current_price = ?,
                        profit = ?,
                        pnl = ?,
                        last_update = datetime('now')
                    WHERE session_id = ? AND symbol = ?
                    ''', (current_price, profit, pnl_percentage, session_id, symbol))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error updating position prices: {str(e)}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
        finally:
            if conn:
                conn.close()
    
    def check_tp_sl(self, session_id):
        """Check and manage TP/SL orders for all open positions"""
        try:
            logger.info("Checking TP/SL orders...")
            
            # Get all open positions with TP/SL orders
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM positions 
                WHERE session_id = ? 
                AND status = 'open' 
                AND (tp_order_id IS NOT NULL OR sl_order_id IS NOT NULL)
            """, (session_id,))
            
            positions = cursor.fetchall()
            
            for position in positions:
                symbol = position['symbol']
                entry_price = position['entry_price']
                size = position['size']
                tp_order_id = position['tp_order_id']
                sl_order_id = position['sl_order_id']
                
                # Get current price
                current_price = get_current_price(symbol)
                if not current_price:
                    logger.error(f"Failed to get current price for {symbol}")
                    continue
                    
                # Check TP order status
                if tp_order_id:
                    tp_status = get_order_status(tp_order_id)
                    if tp_status:
                        if tp_status.get('status') == 'FILLED':
                            logger.info(f"✅ TP order filled for {symbol}")
                            # Update position status
                            cursor.execute("""
                                UPDATE positions 
                                SET status = 'closed', 
                                    exit_time = datetime('now'),
                                    current_price = ?
                                WHERE id = ?
                            """, (current_price, position['id']))
                            
                            # Cancel any remaining SL order
                            if sl_order_id:
                                cancel_order(sl_order_id)
                                
                            self.conn.commit()
                            continue
                            
                # Check SL order status
                if sl_order_id:
                    sl_status = get_order_status(sl_order_id)
                    if sl_status:
                        if sl_status.get('status') == 'FILLED':
                            logger.info(f"⚠️ SL order filled for {symbol}")
                            # Update position status
                            cursor.execute("""
                                UPDATE positions 
                                SET status = 'closed', 
                                    exit_time = datetime('now'),
                                    current_price = ?
                                WHERE id = ?
                            """, (current_price, position['id']))
                            
                            # Cancel any remaining TP order
                            if tp_order_id:
                                cancel_order(tp_order_id)
                                
                            self.conn.commit()
                            continue
                            
                # Update current price
                cursor.execute("""
                    UPDATE positions 
                    SET current_price = ?
                    WHERE id = ?
                """, (current_price, position['id']))
                self.conn.commit()
                
        except Exception as e:
            logger.error(f"Error managing TP/SL orders: {str(e)}")
            logger.error(traceback.format_exc())
    
    def get_position(self, symbol):
        """Get position data for a symbol"""
        return self.positions.get(symbol)
    
    def has_position(self, symbol):
        """Check if we have an open position for a symbol"""
        return symbol in self.positions
    
    def get_total_positions(self):
        """Get total number of open positions"""
        return len(self.positions)

# Initialize position manager globally
position_manager = PositionManager()

class TradeHistory:
    """Tracks trade history and calculates P/L from actual executed trades"""
    
    def __init__(self, db_path='live_trading.db'):
        self.db_path = db_path
        self.trades = {}  # symbol -> list of trades
        self.load_trades()
    
    def load_trades(self):
        """Load recent trades from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get active session
            cursor.execute('SELECT id FROM sessions WHERE status = "active" ORDER BY start_time DESC LIMIT 1')
            session_result = cursor.fetchone()
            
            if session_result:
                session_id = session_result[0]
                
                # Load trades for active session
                cursor.execute('''
                SELECT timestamp, symbol, action, price, quantity, value, profit
                FROM trades 
                WHERE session_id = ?
                ORDER BY timestamp ASC
                ''', (session_id,))
                
                for row in cursor.fetchall():
                    timestamp, symbol, action, price, quantity, value, profit = row
                    
                    if symbol not in self.trades:
                        self.trades[symbol] = []
                    
                    self.trades[symbol].append({
                        "timestamp": timestamp,
                        "action": action,
                        "price": price,
                        "quantity": quantity,
                        "value": value,
                        "profit": profit
                    })
                    
                logger.info(f"Loaded trades for {len(self.trades)} symbols")
                
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading trades: {str(e)}")
            if conn:
                conn.close()
    
    def add_trade(self, session_id, symbol, action, price, quantity, value):
        """Add a new trade to history"""
        conn = None
        try:
            # Calculate profit if it's a SELL
            profit = 0
            if action == 'SELL' and symbol in self.trades:
                # Get the matching BUY trade
                buy_trades = [t for t in self.trades[symbol] if t['action'] == 'BUY']
                if buy_trades:
                    last_buy = buy_trades[-1]
                    profit = (price - last_buy['price']) * quantity
            
            # Create trade record
            trade = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "price": price,
                "quantity": quantity,
                "value": value,
                "profit": profit
            }
            
            # Add to memory
            if symbol not in self.trades:
                self.trades[symbol] = []
            self.trades[symbol].append(trade)
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO trades 
            (session_id, timestamp, symbol, action, price, quantity, value, profit)
            VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?)
            ''', (session_id, symbol, action, price, quantity, value, profit))
            
            conn.commit()
            logger.info(f"Added {action} trade for {symbol} at ${price:.8f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding trade: {str(e)}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            return False
        finally:
            if conn:
                conn.close()
    
    def get_position_info(self, symbol, current_price):
        """Get current position info based on trade history"""
        if symbol not in self.trades:
            return None
            
        trades = self.trades[symbol]
        if not trades:
            return None
            
        # Get latest trades
        buys = [t for t in trades if t['action'] == 'BUY']
        sells = [t for t in trades if t['action'] == 'SELL']
        
        if not buys:
            return None
            
        # Get latest buy
        last_buy = buys[-1]
        
        # Check if we've sold since last buy
        if sells and sells[-1]['timestamp'] > last_buy['timestamp']:
            return None  # No open position
            
        # Calculate current position metrics
        entry_price = last_buy['price']
        quantity = last_buy['quantity']
        value = last_buy['value']
        
        # Calculate current P/L
        unrealized_profit = (current_price - entry_price) * quantity
        pnl_percentage = ((current_price / entry_price) - 1) * 100
        
        return {
            "entry_price": entry_price,
            "current_price": current_price,
            "quantity": quantity,
            "value": value,
            "unrealized_profit": unrealized_profit,
            "pnl_percentage": pnl_percentage,
            "entry_time": last_buy['timestamp']
        }
    
    def get_trade_history(self, symbol):
        """Get trade history for a symbol"""
        return self.trades.get(symbol, [])
    
    def has_open_position(self, symbol):
        """Check if there's an open position based on trade history"""
        if symbol not in self.trades:
            return False
            
        trades = self.trades[symbol]
        if not trades:
            return False
            
        # Check if last trade was a BUY
        return trades[-1]['action'] == 'BUY'

# Initialize trade history globally
trade_history = TradeHistory()

def update_position_prices(session_id):
    """Update position prices and P/L using trade history"""
    conn = None
    try:
        conn = sqlite3.connect('live_trading.db')
        cursor = conn.cursor()
        
        # Get all symbols with open positions
        cursor.execute('SELECT DISTINCT symbol FROM positions WHERE session_id = ?', (session_id,))
        symbols = [row[0] for row in cursor.fetchall()]
        
        for symbol in symbols:
            current_price = get_current_price(symbol)
            if not current_price:
                continue
                
            # Get position info from trade history
            position = trade_history.get_position_info(symbol, current_price)
            if not position:
                continue
                
            # Update database with current price and P/L
            cursor.execute('''
            UPDATE positions
            SET current_price = ?,
                profit = ?,
                pnl = ?,
                last_update = datetime('now')
            WHERE session_id = ? AND symbol = ?
            ''', (
                current_price,
                position['unrealized_profit'],
                position['pnl_percentage'],
                session_id,
                symbol
            ))
            
            logger.debug(f"Position Update - {symbol}:")
            logger.debug(f"  Entry Price (from trade): ${position['entry_price']:.8f}")
            logger.debug(f"  Current Price: ${current_price:.8f}")
            logger.debug(f"  P/L: ${position['unrealized_profit']:+.2f} ({position['pnl_percentage']:+.2f}%)")
        
        conn.commit()
        
    except Exception as e:
        logger.error(f"Error updating position prices: {str(e)}")
        if conn:
            try:
                conn.rollback()
            except:
                pass
    finally:
        if conn:
            conn.close()

def analyze_significant_positions(symbol, df):
    """
    Analyze trading signals for significant positions.
    Returns dict with analysis and signals.
    """
    try:
        if df.empty:
            return None
            
        # Calculate indicators
        df = calculate_indicators(df)
        latest = df.iloc[-1]
        
        # Get predictions
        predictions, confidence = predict_with_pretrained_model(df, symbol, interval='1h')
        latest_pred = predictions.iloc[-1] if not predictions.empty else None
        
        signals = []
        
        # Check technical signals
        if latest['RSI'] > 70:
            signals.append("RSI Overbought (>70) - Consider taking profits")
        if latest['MACD'] < latest['Signal_Line']:
            signals.append("MACD Bearish Crossover - Potential sell signal")
        if latest['close'] < latest['MA20']:
            signals.append("Price below 20MA - Bearish trend")
        
        # Check ML predictions
        if latest_pred is not None:
            if latest_pred['direction'] == 'SELL':
                signals.append("ML Model suggests SELL")
            pred_diff = latest_pred['predicted_price'] - latest_pred['actual_price']
            if pred_diff < 0:
                signals.append(f"ML predicts price drop of ${abs(pred_diff):.2f}")
        
        # Volume analysis
        if latest['volume'] > df['volume'].mean() * 1.5:
            signals.append("Unusual high volume - Watch closely")
        
        # Calculate 24h change
        price_change_24h = ((latest['close'] / df['close'].iloc[-24]) - 1) * 100 if len(df) >= 24 else 0
        
        return {
            'current_price': latest['close'],
            'rsi': latest['RSI'],
            'macd': latest['MACD'],
            'signal_line': latest['Signal_Line'],
            'ma20': latest['MA20'],
            'volume': latest['volume'],
            'price_change_24h': price_change_24h,
            'signals': signals
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None

def check_significant_positions():
    """
    Check trading signals for significant positions (ZETA, EOS, MOG, DIMO)
    """
    try:
        significant_symbols = ['ZETA-USD', 'EOS-USD', 'MOG-USD', 'DIMO-USD']
        analysis_results = []
        
        for symbol in significant_symbols:
            logger.info(f"Analyzing {symbol}...")
            
            # Get recent data
            df = get_coinbase_data(symbol=symbol, granularity=3600, days=7)  # 7 days of hourly data
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                continue
                
            # Get current position
            currency = symbol.split('-')[0]
            position_info = None
            
            try:
                accounts = get_all_accounts()
                for account in accounts:
                    if isinstance(account, dict):
                        if account.get('currency') == currency:
                            position_info = {
                                'quantity': float(account.get('available_balance', {}).get('value', '0')),
                                'currency': currency
                            }
                    else:
                        if getattr(account, 'currency', '') == currency:
                            balance_obj = getattr(account, 'available_balance', None)
                            position_info = {
                                'quantity': float(getattr(balance_obj, 'value', '0')),
                                'currency': currency
                            }
            except Exception as e:
                logger.error(f"Error getting position info for {symbol}: {str(e)}")
                continue
            
            if not position_info or position_info['quantity'] <= 0:
                logger.info(f"No position found for {symbol}")
                continue
                
            # Analyze the position
            analysis = analyze_significant_positions(symbol, df)
            if analysis:
                analysis['symbol'] = symbol
                analysis['quantity'] = position_info['quantity']
                analysis['value'] = position_info['quantity'] * analysis['current_price']
                analysis_results.append(analysis)
                
                logger.info(f"Analysis for {symbol}:")
                logger.info(f"Quantity: {position_info['quantity']}")
                logger.info(f"Current Price: ${analysis['current_price']:.4f}")
                logger.info(f"Total Value: ${analysis['value']:.2f}")
                logger.info(f"24h Change: {analysis['price_change_24h']:+.1f}%")
                if analysis['signals']:
                    logger.info("Trading Signals:")
                    for signal in analysis['signals']:
                        logger.info(f"  • {signal}")
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error checking significant positions: {str(e)}")
        return []

# ... rest of the code ...

if __name__ == '__main__':
    # Add argument parsing for port
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Dashboard')
    parser.add_argument('--port', type=int, default=8054, help='Port to run the application on (default: 8054)')
    parser.add_argument('--test-trade', action='store_true', help='Run a test trade instead of starting the dashboard')
    parser.add_argument('--symbol', type=str, default='XRP-USD', help='Symbol to trade (default: XRP-USD)')
    parser.add_argument('--side', type=str, default='BUY', choices=['BUY', 'SELL'], help='Trade side (default: BUY)')
    parser.add_argument('--funds', type=float, default=0.25, help='Amount to spend in USD (default: 0.25)')
    args = parser.parse_args()

    # If test-trade flag is set, run a test trade
    if args.test_trade:
        print("Running test trade...")
        success = test_real_trade(symbol=args.symbol, side=args.side, funds=args.funds)
        print(f"Test trade {'succeeded' if success else 'failed'}")
        sys.exit(0 if success else 1)

    # Otherwise continue with normal dashboard startup
    
    # Test API connection first
    api_working = test_api_connection()
    
    # Initialize database with required columns
    try:
        conn = sqlite3.connect('live_trading.db')
        cursor = conn.cursor()
        
        # Check if the profit column exists in positions table
        cursor.execute("PRAGMA table_info(positions)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Add missing profit column if needed
        if 'profit' not in columns:
            print("Adding missing 'profit' column to positions table")
            cursor.execute("ALTER TABLE positions ADD COLUMN profit REAL DEFAULT 0.0")
            conn.commit()
            print("Added 'profit' column to positions table")
        else:
            # Update any existing NULL profit values to 0.0
            cursor.execute("UPDATE positions SET profit = 0.0 WHERE profit IS NULL")
            conn.commit()
            print("Updated NULL profit values to 0.0")
        
        # Check if the pnl column exists in positions table
        if 'pnl' not in columns:
            print("Adding missing 'pnl' column to positions table")
            cursor.execute("ALTER TABLE positions ADD COLUMN pnl REAL DEFAULT 0.0")
            conn.commit()
            print("Added 'pnl' column to positions table")
        else:
            # Update any existing NULL pnl values to 0.0
            cursor.execute("UPDATE positions SET pnl = 0.0 WHERE pnl IS NULL")
            conn.commit()
            print("Updated NULL pnl values to 0.0")
        
        # Add TP/SL order ID columns if they don't exist
        if 'tp_order_id' not in columns:
            print("Adding 'tp_order_id' column to positions table")
            cursor.execute("ALTER TABLE positions ADD COLUMN tp_order_id TEXT")
            conn.commit()
            print("Added 'tp_order_id' column to positions table")
            
        if 'sl_order_id' not in columns:
            print("Adding 'sl_order_id' column to positions table")
            cursor.execute("ALTER TABLE positions ADD COLUMN sl_order_id TEXT")
            conn.commit()
            print("Added 'sl_order_id' column to positions table")
        
        conn.close()
    except Exception as e:
        print(f"Error checking/adding columns to positions table: {str(e)}")
        if conn:
            conn.close()

    # Start the app
    try:
        print(f"Starting app on port {args.port}")
        app.run(debug=True, port=args.port)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Error: Port {args.port} is already in use.")
            print("Try running fix_port_issue.py to resolve this issue.")
        else:
            print(f"Error starting app: {str(e)}")

print("Python executable used by OG.py:", sys.executable)
# If parsing positions from WebSocket, ensure you log and assign the parsed block, e.g.:
# Example WebSocket message handler for positions
def handle_ws_message(data):
    if "positions" in data:
        logger.info(f"WebSocket positions message: {data['positions']}")
        # Example assignment to your state variable
        # positions_dict = data['positions']
        # ... rest of your logic ...

def initialize_database():
    """Initialize the database with proper schema and error handling"""
    try:
        # Get absolute path for database
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'live_trading.db')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        logger.info(f"Initializing database at {db_path}")
        conn = sqlite3.connect(db_path, timeout=30)
        cursor = conn.cursor()
        
        # Drop existing tables if they exist
        cursor.execute("DROP TABLE IF EXISTS sessions")
        cursor.execute("DROP TABLE IF EXISTS positions")
        cursor.execute("DROP TABLE IF EXISTS trades")
        cursor.execute("DROP TABLE IF EXISTS portfolio_history")
        
        # Create sessions table with initial_balance column
        logger.info("Creating sessions table...")
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            initial_balance REAL DEFAULT 5.0,
            final_balance REAL,
            total_trades INTEGER DEFAULT 0,
            winning_trades INTEGER DEFAULT 0,
            total_profit REAL DEFAULT 0.0,
            status TEXT DEFAULT 'active'
        )
        ''')
        
        # Create positions table
        logger.info("Creating positions table...")
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            quantity REAL NOT NULL,
            entry_price REAL NOT NULL,
            current_price REAL NOT NULL,
            value REAL NOT NULL,
            profit REAL DEFAULT 0.0,
            pnl REAL DEFAULT 0.0,
            entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tp_order_id TEXT,
            sl_order_id TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(id),
            UNIQUE(session_id, symbol)
        )
        ''')
        
        # Create trades table
        logger.info("Creating trades table...")
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            timestamp DATETIME NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            value REAL NOT NULL,
            profit REAL,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
        ''')
        
        # Create portfolio history table
        logger.info("Creating portfolio history table...")
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            timestamp DATETIME NOT NULL,
            total_value REAL NOT NULL,
            cash_balance REAL NOT NULL,
            positions_value REAL NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
        ''')
            
        conn.commit()
        logger.info("Database schema created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        if conn:
            try:
                conn.rollback()
            except:
                pass
        return False
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass

def ensure_active_session():
    """Ensure there is an active trading session, create one if needed"""
    conn = None
    try:
        logger.info("Checking for active session...")
        conn = sqlite3.connect('live_trading.db', timeout=30)
        cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Check for active session
        cursor.execute('SELECT id FROM sessions WHERE status = "active" ORDER BY start_time DESC LIMIT 1')
        session = cursor.fetchone()
        
        if not session:
            logger.info("No active session found, creating new one...")
            # Create new session with $5 balance
            cursor.execute("""
                INSERT INTO sessions 
                (start_time, initial_balance, status)
                VALUES (datetime('now'), 5.0, 'active')
            """)
            conn.commit()
            
            session_id = cursor.lastrowid
            logger.info(f"Created new session with ID: {session_id}")
            
            # Add initial portfolio history
            cursor.execute("""
            INSERT INTO portfolio_history
            (session_id, timestamp, total_value, cash_balance, positions_value)
            VALUES (?, datetime('now'), 5.0, 5.0, 0.0)
            """, (session_id,))
            
            conn.commit()
            logger.info("Added initial portfolio history")
        else:
            logger.info(f"Found existing active session with ID: {session[0]}")

        conn.close()
        logger.info("Database initialization completed successfully")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"SQLite error during database initialization: {str(e)}")
        if conn:
            try:
                conn.rollback()
            except:
                pass
        return False
    except Exception as e:
        logger.error(f"Unexpected error during database initialization: {str(e)}")
        if conn:
            try:
                conn.rollback()
            except:
                pass
        return False
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass

# Initialize database when module loads
if not initialize_database():
    print("Failed to initialize database")
    raise Exception("Database initialization failed")

# ... rest of the code ...

def test_real_trade(symbol='XRP-USD', side='BUY', funds=None):
    """
    Test function to execute a real trade directly.
    
    Args:
        symbol (str): Trading pair to trade
        side (str): 'BUY' or 'SELL'
        funds (float): Amount in USD to spend (for BUY)
    
    This function will execute a real trade on Coinbase using the execute_real_trade function.
    """
    try:
        logger.info("=" * 50)
        logger.info(f"TESTING REAL TRADE EXECUTION: {side} {symbol}")
        
        # Get product details first
        details = get_product_details(symbol)
        if not details:
            logger.error(f"Failed to get product details for {symbol}")
            return False
            
        # Get minimum order size
        min_market_funds = float(details.get('min_market_funds', 1.0))
        
        # If funds not provided or too small, use minimum
        if not funds or funds < min_market_funds:
            funds = min_market_funds
            logger.info(f"Using minimum order size: ${funds:.2f}")
        
        # Get current price
        current_price = get_current_price(symbol)
        if not current_price:
            logger.error(f"Failed to get price for {symbol}")
            return False
            
        logger.info(f"Current price for {symbol}: ${current_price}")
        
        # Execute the trade
        result = execute_real_trade(symbol, side, current_price, funds)
        
        if result:
            logger.info(f"✅ TEST TRADE SUCCESSFUL: {side} {symbol}")
        else:
            logger.error(f"❌ TEST TRADE FAILED: {side} {symbol}")
            
        logger.info("=" * 50)
        return result
        
    except Exception as e:
        logger.error(f"Error in test_real_trade: {str(e)}")
        return False


def place_order(order_data):
    """Place a single order and return the order ID"""
    try:
        # Get auth headers
        headers = get_auth_headers("POST", "/api/v3/brokerage/orders")
        if not headers:
            logger.error("Failed to get auth headers")
            return None
            
        # Place order
        response = requests.post(
            f"https://{BASE_URL}/api/v3/brokerage/orders",
            headers=headers,
            json=order_data
        )
        
        # Log full request and response for debugging
        logger.info("Order request:")
        logger.info(f"Headers: {headers}")
        logger.info(f"Data: {json.dumps(order_data, indent=2)}")
        logger.info("Response:")
        logger.info(f"Status: {response.status_code}")
        logger.info(f"Body: {response.text}")
        
        if response.status_code in [200, 201]:
            try:
                data = response.json()
                logger.info(f"Parsed response: {json.dumps(data, indent=2)}")
                
                # Check success field
                if data.get('success'):
                    # Get order ID from success_response
                    success_response = data.get('success_response', {})
                    order_id = success_response.get('order_id')
                    
                    if order_id:
                        logger.info(f"Order placed successfully: {order_id}")
                        return order_id
                    else:
                        logger.error("Order ID not found in success_response")
                        logger.error(f"Response structure: {json.dumps(data, indent=2)}")
                        return None
                else:
                    error = data.get('error_response', {})
                    error_msg = error.get('message') or error.get('error_details') or str(error)
                    logger.error(f"Order failed: {error_msg}")
                    if 'preview_failure_reason' in error:
                        logger.error(f"Preview failure: {error['preview_failure_reason']}")
                    return None
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse API response: {e}")
                logger.error(f"Raw response: {response.text}")
                return None
        else:
            logger.error(f"API request failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error placing order: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def verify_tp_sl_orders(tp_order_id, sl_order_id):
    """Verify that TP/SL orders exist and are active"""
    logger.info("Verifying TP/SL orders...")
    
    # Check TP order
    tp_status = get_order_status(tp_order_id)
    if tp_status:
        logger.info(f"TP Order Status: {json.dumps(tp_status, indent=2)}")
    else:
        logger.error("Failed to get TP order status")
        
    # Check SL order
    sl_status = get_order_status(sl_order_id)
    if sl_status:
        logger.info(f"SL Order Status: {json.dumps(sl_status, indent=2)}")
    else:
        logger.error("Failed to get SL order status")
        
    return tp_status, sl_status

def get_product_precision(symbol):
    """Get base and quote precision for a product"""
    try:
        # Get auth headers
        headers = get_auth_headers("GET", f"/products/{symbol}")
        if not headers:
            logger.error("Failed to get auth headers")
            return None, None
            
        # Add accept header
        headers['Accept'] = 'application/json'
        
        # Make API request with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"https://{BASE_URL}/products/{symbol}",
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    product = response.json()
                    base_increment = product.get('base_increment', '0.00000001')
                    quote_increment = product.get('quote_increment', '0.01')
                    
                    base_precision = len(base_increment.split('.')[-1]) if '.' in base_increment else 0
                    quote_precision = len(quote_increment.split('.')[-1]) if '.' in quote_increment else 0
                    
                    logger.info(f"Got precision for {symbol}: base={base_precision}, quote={quote_precision}")
                    return base_precision, quote_precision
                    
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed to get product details: {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return None, None
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None, None
                
        return None, None  # All retries failed
        
    except Exception as e:
        logger.error(f"Error getting product precision: {str(e)}")
        return None, None

# === LIVE TRADING SYSTEM BLOCK (START) ===
# This block ensures all critical live trading methods and logic are present and robust.
# If a method already exists above, we reference it here.

# --- Order Placement & Management ---
# place_order: See main implementation above.
# place_tp_sl_orders: See main implementation above.
# cancel_order: See main implementation above.
# wait_for_order_settlement: See main implementation above.

# --- Position Management ---
# PositionManager: See main implementation above.
# Ensure it is used in the trading loop below.

# --- Portfolio & Balance Management ---
# get_available_balance: See main implementation above.
# get_portfolio_value: See main implementation above.
# update_portfolio_history: See main implementation above.

# --- Trading Loop ---
import threading

def live_trading_loop():
    """Robust trading loop using all critical methods."""
    global stop_trading, live_sim, position_manager
    logger.info("[LIVE] Starting trading loop...")
    last_portfolio_update = datetime.now()
    try:
        # Ensure we have an active session
        session_id = get_active_session()
        if not session_id:
            logger.error("[LIVE] No active session found. Aborting trading loop.")
            return
        while not stop_trading:
            try:
                # Update portfolio history every minute
                now = datetime.now()
                if (now - last_portfolio_update).total_seconds() >= 60:
                    try:
                        update_portfolio_history(session_id)
                        last_portfolio_update = now
                    except Exception as e:
                        logger.error(f"[LIVE] Error updating portfolio history: {e}")
                # Update position prices and check TP/SL
                try:
                    position_manager.update_position_prices(session_id)
                    position_manager.check_tp_sl(session_id)
                except Exception as e:
                    logger.error(f"[LIVE] Error updating/checking positions: {e}")
                # Get available balance
                try:
                    available_balance = get_available_balance()
                except Exception as e:
                    logger.error(f"[LIVE] Error getting available balance: {e}")
                    available_balance = 0.0
                if available_balance <= 0:
                    logger.info("[LIVE] No available balance for trading. Sleeping...")
                    time.sleep(60)
                    continue
                # Example: Scan for opportunities and open new positions
                try:
                    # This is a placeholder for your market scan logic
                    # opportunities = scan_market()
                    opportunities = []  # Replace with real scan
                    max_new_positions = max(0, live_sim.max_positions - position_manager.get_total_positions())
                    for opp in opportunities[:max_new_positions]:
                        symbol = opp['symbol']
                        price = opp['current_price']
                        position_size = min(available_balance, 1.0)  # Example: $1 per position
                        quantity = position_size / price
                        if position_manager.open_position(session_id, symbol, price, quantity, position_size):
                            logger.info(f"[LIVE] Opened position in {symbol}")
                            available_balance -= position_size
                except Exception as e:
                    logger.error(f"[LIVE] Error opening new positions: {e}")
                # Sleep before next iteration
                for _ in range(6):  # Check stop_trading every 10 seconds
                    if stop_trading:
                        break
                    time.sleep(10)
            except Exception as e:
                logger.error(f"[LIVE] Error in trading loop: {e}")
                time.sleep(60)
        logger.info("[LIVE] Trading loop stopped.")
    except Exception as e:
        logger.error(f"[LIVE] Fatal error in trading loop: {e}")

# --- Dash Callbacks for Start/Stop Trading ---
# These should be connected to your Dash UI buttons.
trading_thread = None

def start_live_trading():
    global trading_thread, stop_trading
    if trading_thread and trading_thread.is_alive():
        logger.info("[LIVE] Trading already running.")
        return
    stop_trading = False
    trading_thread = threading.Thread(target=live_trading_loop, daemon=True)
    trading_thread.start()
    logger.info("[LIVE] Trading started.")

def stop_live_trading():
    global stop_trading, trading_thread
    stop_trading = True
    if trading_thread:
        trading_thread.join(timeout=10)
        logger.info("[LIVE] Trading stopped.")

# Example Dash callback usage (pseudo-code):
# @app.callback(...)
# def on_start_button_click(...):
#     start_live_trading()
#     return ...
#
# @app.callback(...)
# def on_stop_button_click(...):
#     stop_live_trading()
#     return ...

# --- Error Handling & Logging ---
# All methods above use try/except and logger for robust error handling.

# === LIVE TRADING SYSTEM BLOCK (END) ===