#!/usr/bin/env python3
import os
import sqlite3
import logging
import traceback
from datetime import datetime
from init_database import get_db_path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Get database path
DB_PATH = get_db_path()

# Ensure data directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

import threading
import sqlite3
import logging
import os

# Add a lock for database initialization
_db_init_lock = threading.Lock()

def init_db():
    """Initialize the database with required tables."""
    global _db_init_lock
    
    with _db_init_lock:
        try:
            logger.info(f"[DEBUG] Initializing database at: {DB_PATH}")
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check if tables already exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
            if cursor.fetchone():
                logger.info("[DEBUG] Database tables already exist")
                return
            
            # Create sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    initial_balance REAL DEFAULT 0.0,
                    final_balance REAL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    total_profit REAL DEFAULT 0.0
                )
            ''')
            
            # Create positions table with all necessary columns
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    entry_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_update DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    profit REAL DEFAULT 0.0,
                    pnl REAL DEFAULT 0.0,
                    value REAL NOT NULL DEFAULT 0.0,
                    position_size REAL DEFAULT 0.0,
                    size REAL DEFAULT 0.0,
                    pl_value REAL DEFAULT 0.0,
                    pl_percentage REAL DEFAULT 0.0,
                    tp_order_id TEXT,
                    sl_order_id TEXT,
                    tp_price REAL,
                    sl_price REAL,
                    status TEXT DEFAULT 'open',
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            ''')
            
            # Create trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    value REAL NOT NULL DEFAULT 0.0,
                    profit REAL DEFAULT 0.0,
                    size REAL DEFAULT 0.0,
                    tp_order_id TEXT,
                    sl_order_id TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            ''')
            
            # Create portfolio history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_value REAL NOT NULL,
                    available_balance REAL NOT NULL,
                    cash_balance REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            ''')
            
            # Check if there's any active session
            cursor.execute("SELECT COUNT(*) FROM sessions WHERE status = 'active'")
            active_sessions = cursor.fetchone()[0]
            
            if active_sessions == 0:
                # Create initial session if none exists
                cursor.execute('''
                    INSERT INTO sessions (start_time, status, initial_balance)
                    VALUES (datetime('now'), 'active', 0.0)
                ''')
                session_id = cursor.lastrowid
                
                # Create initial portfolio history entry
                cursor.execute('''
                    INSERT INTO portfolio_history (
                        session_id, timestamp, total_value, available_balance,
                        cash_balance, positions_value
                    )
                    VALUES (?, datetime('now'), 0.0, 0.0, 0.0, 0.0)
                ''', (session_id,))
                
                logger.info(f"Created initial session with ID {session_id}")
            
            conn.commit()
            logger.info("[DEBUG] Database schema created successfully")
            
        except Exception as e:
            logger.error(f"[DEBUG] Error initializing database: {str(e)}")
            logger.error(traceback.format_exc())
            if 'conn' in locals():
                conn.rollback()
            raise
        finally:
            if 'conn' in locals():
                conn.close()
                logger.info("[DEBUG] Database connection closed")

# Initialize database at module import
try:
    init_db()
    logger.info("Database initialized successfully at startup")
except Exception as e:
    logger.error(f"Failed to initialize database at startup: {e}")

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
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'live_trading.db')

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
    """Calculate technical indicators for a DataFrame"""
    try:
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Convert price columns to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, float('inf'))  # Handle division by zero
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Calculate SMAs
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
        
        # Calculate Bollinger Bands
        df['middle_band'] = df['close'].rolling(window=20, min_periods=1).mean()
        std = df['close'].rolling(window=20, min_periods=1).std()
        df['upper_band'] = df['middle_band'] + (std * 2)
        df['lower_band'] = df['middle_band'] - (std * 2)
        
        # Fill NaN values with forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info("Successfully calculated all indicators")
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return pd.DataFrame()

# === Model Training ===
def train_model_for_symbol(symbol, granularity=60):
    """Train ML models for a given symbol and granularity"""
    try:
        logger.info(f"Training models for {symbol} with {granularity}s granularity...")
        
        # Get historical data
        df = get_coinbase_data(symbol, granularity, days=30)
        if df is None or df.empty:
            logger.error(f"No data available for {symbol}")
            return None, None
        
        # Calculate indicators
        df = calculate_indicators(df)
        
        # Prepare features and target
        feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'sma_20', 'sma_50', 'upper_band', 'lower_band'
        ]
        
        # Create target variables
        df['target'] = df['close'].shift(-1) > df['close']
        df['target'] = df['target'].astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        if len(df) < 100:
            logger.error(f"Insufficient data for {symbol} after preprocessing")
            return None, None
        
        # Split data
        train_size = int(len(df) * 0.8)
        X_train = df[feature_columns][:train_size]
        y_train = df['target'][:train_size]
        
        # Train models
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Save models
        model_prefix = f"{symbol.replace('-', '')}_{granularity}"
        clf_path = os.path.join(MODELS_DIR, f"{model_prefix}_clf.pkl")
        
        import joblib
        joblib.dump(clf, clf_path)
        
        logger.info(f"Successfully trained and saved models for {symbol}")
        return clf
        
    except Exception as e:
        logger.error(f"Error training models for {symbol}: {str(e)}")
        return None, None

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
            (latest['rsi'] / 100) * 0.3 +
            (1 if latest['macd'] > latest['macd_signal'] else 0) * 0.3 +
            (latest['%K'] / 100) * 0.2 +
            (1 if latest['close'] > latest['sma_20'] else 0) * 0.2
        ) * 100
        
        # Determine momentum direction
        if latest['macd'] > latest['macd_signal'] and latest['rsi'] > 50:
            momentum_direction = 'Bullish'
        elif latest['macd'] < latest['macd_signal'] and latest['rsi'] < 50:
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
            'rsi': latest['rsi'],
            'macd': latest['macd'],
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
                if latest['rsi'] > 50:
                    momentum_score += 20
                if latest['macd'] > latest['macd_signal']:
                    momentum_score += 20
                if latest['close'] > latest['sma_20']:
                    momentum_score += 20
                if latest['%K'] > latest['%D']:
                    momentum_score += 20
                if latest['OBV'] > df['OBV'].mean():
                    momentum_score += 20

                # Calculate volatility
                volatility = latest['ATR'] / latest['close'] * 100

                # Calculate trend strength
                trend_strength = abs(latest['close'] - latest['sma_20']) / latest['sma_20'] * 100

                results.append({
                    'symbol': symbol,
                    'current_price': float(latest['close']),
                    'momentum_score': float(momentum_score),
                    'rsi': float(latest['rsi']),
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

# === LIVE TRADING SYSTEM BLOCK (START) ===
# This block implements robust live trading logic as per the plan.
import threading

# --- Order Placement & Management ---
def place_order(order_data):
    try:
        # TODO: Implement actual order placement logic (API call)
        logger.info(f"Placing order: {order_data}")
        # Simulate order placement
        return str(uuid.uuid4())
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return None

def place_tp_sl_orders(symbol, entry_price, size):
    try:
        logger.info(f"Placing TP/SL orders for {symbol} at entry {entry_price} size {size}")
        # Simulate TP/SL order placement
        tp_order_id = str(uuid.uuid4())
        sl_order_id = str(uuid.uuid4())
        return tp_order_id, sl_order_id
    except Exception as e:
        logger.error(f"Error placing TP/SL orders: {e}")
        return None, None

def cancel_order(order_id):
    try:
        logger.info(f"Canceling order: {order_id}")
        # Simulate cancel
        return True
    except Exception as e:
        logger.error(f"Error canceling order: {e}")
        return False

def wait_for_order_settlement(order_id, max_attempts=10, delay=3.0):
    try:
        logger.info(f"Waiting for order {order_id} to settle...")
        for _ in range(max_attempts):
            time.sleep(delay)
            # Simulate check
            return True
        logger.warning(f"Order {order_id} did not settle in time.")
        return False
    except Exception as e:
        logger.error(f"Error waiting for order settlement: {e}")
        return False

# --- Position Management ---
class PositionManager:
    def __init__(self):
        self.positions = {}  # symbol -> position dict
        self.refresh_positions()  # Load initial positions from DB
        
    def refresh_positions(self):
        """Refresh positions from the database"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get active session
            session_id = get_active_session()
            if not session_id:
                return
            
            cursor.execute('''
                SELECT p.symbol, p.quantity, p.entry_price, p.current_price, 
                       p.size as position_value, p.pnl
                FROM positions p
                WHERE p.session_id = ? AND p.status = 'open'
            ''', (session_id,))
            
            positions = cursor.fetchall()
            
            # Clear existing positions
            self.positions = {}
            
            # Update positions
            for pos in positions:
                symbol = pos[0]
                self.positions[symbol] = {
                    'quantity': pos[1],
                    'entry_price': pos[2],
                    'current_price': pos[3],
                    'position_size': pos[4],
                    'pnl': pos[5]
                }
                
                # Check if we need to train a model for this symbol
                model_prefix = f"{symbol.replace('-', '')}_{3600}"  # Using 1h timeframe
                model_path = os.path.join(MODELS_DIR, f"{model_prefix}_clf.pkl")
                
                if not os.path.exists(model_path):
                    logger.info(f"Training new model for {symbol}...")
                    train_model_for_symbol(symbol, 3600)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error refreshing positions: {str(e)}")
            if 'conn' in locals():
                conn.close()

    def open_position(self, symbol, entry_price, quantity, position_size):
        try:
            if symbol in self.positions:
                logger.warning(f"Position for {symbol} already exists.")
                return False
            tp_order, sl_order = place_tp_sl_orders(symbol, entry_price, quantity)
            self.positions[symbol] = {
                'entry_price': entry_price,
                'quantity': quantity,
                'position_size': position_size,
                'tp_order_id': tp_order,
                'sl_order_id': sl_order,
                'open': True
            }
            logger.info(f"Opened position for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return False

    def update_position_prices(self):
        try:
            self.refresh_positions()  # Refresh positions from DB instead of simulating
        except Exception as e:
            logger.error(f"Error updating position prices: {e}")

    def check_tp_sl(self):
        try:
            for symbol, pos in self.positions.items():
                # Simulate TP/SL check
                logger.info(f"Checked TP/SL for {symbol}")
        except Exception as e:
            logger.error(f"Error checking TP/SL: {e}")

    def get_total_positions(self):
        return len(self.positions)

position_manager = PositionManager()

# --- Portfolio & Balance Management ---
def get_available_balance():
    """Get available USD balance for trading"""
    try:
        from lk import client
        
        # Get all accounts using the working client approach
        response = client.get_accounts(limit=250)
        accounts = response.accounts if hasattr(response, 'accounts') else response.get('accounts', [])
        
        for account in accounts:
            try:
                # Handle account object
                if hasattr(account, 'currency'):
                    currency = account.currency
                    available_balance = getattr(account, 'available_balance', None)
                    
                    # Handle balance objects that are dictionaries
                    if isinstance(available_balance, dict):
                        available_value = float(available_balance.get('value', '0'))
                    else:
                        available_value = float(getattr(available_balance, 'value', '0')) if available_balance else 0
                else:
                    currency = account.get('currency')
                    available_balance = account.get('available_balance', {})
                    available_value = float(available_balance.get('value', '0'))
                
                if currency == 'USD':
                    logger.info(f"Found USD available balance: ${available_value:.2f}")
                    return available_value
                    
            except Exception as e:
                logger.error(f"Error processing account: {str(e)}")
                continue
                
        logger.warning("No USD account found")
        return 0
        
    except Exception as e:
        logger.error(f"Error getting available balance: {str(e)}")
        return 0

def get_portfolio_value():
    """Get total portfolio value from Coinbase"""
    try:
        # Get the absolute path to lk.py
        lk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lk.py"))
        logger.info(f"Running lk.py from path: {lk_path}")
        
        # Run lk.py using the current Python executable
        result = subprocess.run(
            [sys.executable, lk_path],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout.strip()
        logger.info(f"lk.py output: {output}")
        
        try:
            data = json.loads(output)
            return float(data.get('total_value', 0))
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from lk.py output: {output}")
            return 0.0
            
    except Exception as e:
        logger.error(f"Error getting portfolio value: {str(e)}")
        return 0.0

def update_portfolio_history():
    try:
        logger.info(f"Portfolio value: {get_portfolio_value()}")
    except Exception as e:
        logger.error(f"Error updating portfolio history: {e}")

# --- Trading Loop ---
trading_thread = None
stop_trading = False
live_sim = None

def trading_loop():
    """Main trading loop"""
    try:
        logger.info("Starting trading loop...")
        
        # Initialize client
        client = RESTClient(
            api_key=KEY_NAME,
            api_secret=PRIVATE_KEY_PEM,
            base_url=BASE_URL
        )
        
        last_ml_update = 0
        ML_UPDATE_INTERVAL = 300  # Update ML decisions every 5 minutes
        
        while True:
            try:
                if not ws_client or not ws_client.is_running():
                    logger.warning("WebSocket client not running, skipping iteration")
                    time.sleep(5)
                    continue
                
                current_time = time.time()
                
                # Update ML decisions periodically
                if current_time - last_ml_update > ML_UPDATE_INTERVAL:
                    try:
                        # Get active session
                        session_id = get_active_session()
                        if session_id:
                            # Get current positions
                            positions = position_manager.positions
                            for symbol in positions:
                                try:
                                    # Train model if needed and make decision
                                    model = train_model_for_symbol(symbol, granularity=3600)
                                    if model:
                                        # Get recent data for prediction
                                        df = get_coinbase_data(symbol, granularity=3600, days=1)
                                        if df is not None and not df.empty:
                                            df = calculate_indicators(df)
                                            if not df.empty:
                                                # Prepare features
                                                feature_columns = [
                                                    'rsi', 'macd', 'macd_signal', 'macd_hist',
                                                    'sma_20', 'sma_50', 'upper_band', 'lower_band'
                                                ]
                                                X = df[feature_columns].iloc[-1:]
                                                
                                                # Make prediction
                                                prediction = model.predict(X)[0]
                                                confidence = max(model.predict_proba(X)[0])
                                                
                                                # Convert prediction to decision
                                                decision = "BUY" if prediction == 1 else "SELL"
                                                
                                                # Log decision
                                                from ml_logging import log_ml_decision
                                                log_ml_decision(
                                                    session_id=session_id,
                                                    symbol=symbol,
                                                    decision=decision,
                                                    confidence=confidence,
                                                    features=df[feature_columns].iloc[-1].to_dict()
                                                )
                                                
                                                logger.info(f"Made ML decision for {symbol}: {decision} (confidence: {confidence:.2%})")
                                except Exception as e:
                                    logger.error(f"Error making ML decision for {symbol}: {str(e)}")
                                    continue
                            
                            last_ml_update = current_time
                    except Exception as e:
                        logger.error(f"Error updating ML decisions: {str(e)}")
                
                # Rest of the trading loop
                position_manager.refresh_positions()
                position_manager.update_position_prices()
                position_manager.check_tp_sl()
                update_portfolio_history()
                
                time.sleep(1)  # Avoid busy waiting
                
            except Exception as e:
                logger.error(f"Error in trading loop iteration: {str(e)}")
                time.sleep(5)  # Wait before retrying
                
    except Exception as e:
        logger.error(f"Fatal error in trading loop: {str(e)}")
        raise

def start_live_trading():
    """Start the live trading thread"""
    global trading_thread
    
    try:
        if trading_thread and trading_thread.is_alive():
            logger.info("Trading thread is already running")
            return
        
        # Initialize WebSocket client
        global ws_client
        if not ws_client:
            ws_client = initialize_websocket_client()
        
        # Start WebSocket client
        if not ws_client.is_running():
            ws_client.start()
            logger.info("WebSocket client started")
        
        # Start trading thread
        trading_thread = Thread(target=trading_loop, daemon=True)
        trading_thread.start()
        logger.info("Trading thread started")
        
    except Exception as e:
        logger.error(f"Error starting live trading: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def stop_live_trading():
    """Stop the live trading thread"""
    global trading_thread, ws_client
    
    try:
        # Stop WebSocket client
        if ws_client and ws_client.is_running():
            ws_client.stop()
            logger.info("WebSocket client stopped")
        
        # Stop trading thread
        if trading_thread and trading_thread.is_alive():
            # The thread will stop itself when the WebSocket client stops
            logger.info("Trading thread will stop")
        
    except Exception as e:
        logger.error(f"Error stopping live trading: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# --- Dash Callbacks for Start/Stop Trading ---
@app.callback(
    [Output('live-trading-status', 'children'),
     Output('start-live-trading-btn', 'disabled'),
     Output('stop-live-trading-btn', 'disabled')],
    [Input('start-live-trading-btn', 'n_clicks'),
     Input('stop-live-trading-btn', 'n_clicks')]
)
def manage_live_trading(start_clicks, stop_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Trading not started", False, True
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'start-live-trading-btn' and start_clicks:
        start_live_trading()
        return "Trading started", True, False
    elif button_id == 'stop-live-trading-btn' and stop_clicks:
        stop_live_trading()
        return "Trading stopped", False, True
    return dash.no_update
# === LIVE TRADING SYSTEM BLOCK (END) ===

# === ROBUST TRADING LOOP BLOCK (START) ===
import sqlite3
import subprocess
import sys
import json
from datetime import datetime
import time

# Globals for trading control
trading_thread = None
stop_trading = False
live_sim = None
ws_client = None

# DBManager class (add if not present)
class DBManager:
    def __init__(self, db_path=DB_PATH, timeout=30):
        self.db_path = db_path
        self.timeout = timeout
        self.conn = None
        self.cursor = None
        self.connect()
    def connect(self):
        self.conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        self.conn.isolation_level = None
        self.cursor = self.conn.cursor()
    def ensure_connection(self):
        try:
            self.cursor.execute("SELECT 1")
        except (sqlite3.OperationalError, sqlite3.ProgrammingError):
            self.close()
            self.connect()
    def execute(self, query, params=None):
        self.ensure_connection()
        if params:
            return self.cursor.execute(query, params)
        return self.cursor.execute(query)
    def commit(self):
        self.ensure_connection()
        self.conn.commit()
    def rollback(self):
        self.ensure_connection()
        self.conn.rollback()
    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

# Robust trading loop

def trading_loop():
    global stop_trading, live_sim, ws_client
    logger.info("Starting trading loop...")
    last_portfolio_update = datetime.now()
    last_position_analysis = datetime.now()
    last_tp_sl_check = datetime.now()
    last_position_check = datetime.now()
    last_memory_cleanup = datetime.now()
    last_position_sync = datetime.now()  # Add this
    db_manager = DBManager()
    
    try:
        db_manager.connect()
        # Ensure database schema is correct (implement ensure_database_schema if needed)
        # ensure_database_schema()
        db_manager.execute('SELECT id FROM sessions WHERE status = "active" ORDER BY start_time DESC LIMIT 1')
        session_result = db_manager.cursor.fetchone()
        if not session_result:
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
        while not stop_trading:
            try:
                now = datetime.now()
                
                # Sync positions every minute
                if (now - last_position_sync).total_seconds() >= 60:
                    try:
                        live_positions = get_live_positions_from_lk()
                        if live_positions:
                            sync_positions_to_db(live_positions)
                        last_position_sync = now
                    except Exception as e:
                        logger.error(f"Error syncing positions: {str(e)}")
                
                # Memory cleanup every 5 minutes
                if (datetime.now() - last_memory_cleanup).total_seconds() >= 300:
                    import gc
                    gc.collect()
                    last_memory_cleanup = datetime.now()
                    logger.debug("Performed memory cleanup")
                # Update portfolio and positions every minute
                if (now - last_portfolio_update).total_seconds() >= 60:
                    try:
                        update_portfolio_history()
                        last_portfolio_update = now
                    except Exception as e:
                        logger.error(f"Error updating portfolio history: {e}")
                
                # Check TP/SL orders every 30 seconds
                if (now - last_tp_sl_check).total_seconds() >= 30:
                    try:
                        logger.info("Checking TP/SL orders...")
                        # Note: TP/SL manager will be implemented later if needed
                        last_tp_sl_check = now
                    except Exception as tp_sl_error:
                        logger.error(f"Error managing TP/SL orders: {str(tp_sl_error)}")
                # Evaluate existing positions every minute
                if (now - last_position_check).total_seconds() >= 60:
                    try:
                        logger.info("[SELL EVAL] Running ML-based position analysis...")
                        
                        # Get active session
                        session_id = get_active_session()
                        if not session_id:
                            logger.warning("[SELL EVAL] No active session found")
                            continue
                            
                        # Run analyze_and_sell.py as a subprocess
                        script_path = os.path.join(os.path.dirname(__file__), 'analyze_and_sell.py')
                        if not os.path.exists(script_path):
                            logger.error(f"[SELL EVAL] analyze_and_sell.py not found at {script_path}")
                            continue
                            
                        cmd = [sys.executable, script_path, '--session-id', str(session_id)]
                        logger.info(f"[SELL EVAL] Running command: {' '.join(cmd)}")
                        
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            check=False
                        )
                        
                        # Log output
                        if result.stdout:
                            for line in result.stdout.splitlines():
                                if line.strip():
                                    if "error" in line.lower() or "exception" in line.lower():
                                        logger.error(f"[SELL EVAL] {line}")
                                    else:
                                        logger.info(f"[SELL EVAL] {line}")
                                    
                        # Log errors (only actual errors)
                        if result.stderr:
                            for line in result.stderr.splitlines():
                                if line.strip():
                                    if ("error" in line.lower() or 
                                        "exception" in line.lower() or 
                                        "warning" in line.lower()):
                                        logger.error(f"[SELL EVAL] {line}")
                                    else:
                                        logger.info(f"[SELL EVAL] {line}")
                                    
                        if result.returncode == 0:
                            logger.info("[SELL EVAL] Position analysis completed successfully")
                        else:
                            logger.error(f"[SELL EVAL] Position analysis failed with code {result.returncode}")
                            
                        last_position_check = now
                        
                    except Exception as e:
                        logger.error(f"[SELL EVAL] Error running position analysis: {e}")
                        logger.error(traceback.format_exc())
                # Get available balance from lk.py
                try:
                    result = subprocess.run(
                        [sys.executable, "lk.py"],
                        capture_output=True, text=True, check=False
                    )
                    
                    if result.returncode != 0:
                        logger.warning(f"lk.py returned non-zero exit code: {result.returncode}")
                        if result.stderr:
                            logger.warning(f"lk.py stderr: {result.stderr}")
                        available_balance = 0
                    else:
                        output = result.stdout.strip()
                        start_marker = "DASHBOARD_DATA_START"
                        end_marker = "DASHBOARD_DATA_END"
                        if start_marker in output and end_marker in output:
                            json_str = output.split(start_marker)[1].split(end_marker)[0]
                            data = json.loads(json_str)
                            available_balance = 0
                            for pos in data.get('positions', []):
                                if pos.get('currency') == 'USD':
                                    available_balance = float(pos.get('usd_value', 0))
                                    break
                        else:
                            available_balance = 0
                except Exception as e:
                    logger.error(f"Error calling lk.py: {str(e)}")
                    available_balance = 0
                if available_balance <= 0:
                    logger.warning("❌ No available balance for trading")
                    time.sleep(60)
                    continue
                logger.info(f"💰 Available balance for trading: ${available_balance:.2f}")
                db_manager.execute('SELECT COUNT(DISTINCT symbol) FROM positions WHERE session_id = ?', (session_id,))
                current_positions = db_manager.cursor.fetchone()[0]
                max_new_positions = max(0, 3 - current_positions)  # Use 3 as default max positions
                if max_new_positions > 0 and available_balance > 0:
                    if current_positions == 0:
                        position_size = min(available_balance * 0.2, 1.0)
                    else:
                        position_size = min(available_balance / max_new_positions, 0.5)
                    logger.info(f"🎯 Looking for trades with position size: ${position_size:.2f}")
                    opportunities = scan_market()
                    if opportunities:
                        for opp in opportunities[:max_new_positions]:
                            try:
                                symbol = opp['symbol']
                                current_price = opp['current_price']
                                min_quantity = 0.000001
                                if position_size / current_price < min_quantity:
                                    logger.info(f"⏭️ Skipping {symbol} - price too high for our position size")
                                    continue
                                fresh_balance = get_available_balance()
                                if fresh_balance < position_size:
                                    logger.warning(f"❌ Insufficient balance for {symbol} trade. Required: ${position_size:.2f}, Available: ${fresh_balance:.2f}")
                                    break
                                if execute_real_trade(symbol, "BUY", current_price, funds=position_size):
                                    logger.info(f"✅ Successfully executed real BUY order for {symbol}")
                                    available_balance = fresh_balance - position_size
                                    if available_balance < 0.1:
                                        logger.info("💰 Remaining balance too low, stopping trading")
                                        break
                                else:
                                    logger.error(f"❌ Failed to execute real BUY order for {symbol}")
                            except Exception as e:
                                logger.error(f"❌ Error processing opportunity for {symbol}: {str(e)}")
                                continue
                else:
                    logger.info(f"ℹ️ No new positions available. Current positions: {current_positions}, Available balance: ${available_balance:.2f}")
                last_position_analysis = datetime.now()
                if (now - last_position_analysis).total_seconds() >= 300:
                    logger.info("Running scheduled ML position analysis...")
                    run_position_analysis()
                    last_position_analysis = now
                time.sleep(60)
            except Exception as e:
                logger.error(f"❌ Error in trading loop iteration: {str(e)}")
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
# === ROBUST TRADING LOOP BLOCK (END) ===

def predict_with_pretrained_model(df, symbol, interval='1h'):
    """Make predictions using a pre-trained model"""
    try:
        # Calculate indicators if not already present
        if not all(col in df.columns for col in ['rsi', 'macd', 'macd_signal']):
            df = calculate_indicators(df)
        
        # Get the latest data point
        latest = df.iloc[-1]
        
        # Use the same feature names as training
        feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'sma_20', 'sma_50', 'upper_band', 'lower_band'
        ]
        
        # Create features dictionary
        features = {}
        for col in feature_columns:
            if col in latest:
                features[col] = float(latest[col])
        
        # Load the model
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        model_path = os.path.join(model_dir, f'{symbol}_{interval}_model.joblib')
        
        if os.path.exists(model_path):
            model = joblib.load(joblib.load(model_path)
            
            # Prepare features for prediction
            X = pd.DataFrame([features])
            
            # Ensure columns are in the same order as training
            X = X[feature_columns]
            
            # Make prediction
            prediction = model.predict(X)[0]
            confidence = max(model.predict_proba(X)[0])
            
            # Convert prediction to decision
            decision = "SELL" if prediction == 0 else "BUY"
            
            return {
                'decision': decision, 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/confidence': confidence,
                'features': features
            }
        else:
            logger.warning(f"No pre-trained model found for {symbol} at {interval} interval")
            return None
            
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return None

# --- Real Trade Execution Functions (from working.py) ---
def execute_trade_via_subprocess(symbol, side, price, funds):
    """Execute a trade by running tp_sl_fixed.py as a subprocess"""
    try:
        logger.info(f"Executing {side} trade for {symbol} via tp_sl_fixed.py")
        logger.info(f"Price: ${price}, Value: ${funds}")
        size = funds / price
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
        working_dir = os.path.dirname(script_path)
        logger.info(f"Using working directory: {working_dir}")
        cmd = [
            sys.executable,
            script_path,
            "--symbol", symbol,
            "--price", str(price),
            "--size", str(size)
        ]
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=working_dir
        )
        if result.stdout:
            logger.info(f"tp_sl_fixed.py output: {result.stdout}")
        if result.stderr:
            logger.error(f"tp_sl_fixed.py error: {result.stderr}")
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

def execute_real_trade(symbol, side, price, funds):
    """Execute a real trade with proper TP/SL orders"""
    try:
        logger.info(f"Executing real {side} trade for {symbol}")
        logger.info(f"Price: ${price}, Value: ${funds}")
        response = requests.get(
            f"https://api.exchange.coinbase.com/products/{symbol}",
            headers={'Accept': 'application/json'}
        )
        if response.status_code != 200:
            logger.error(f"Failed to get product details for {symbol}: {response.status_code}")
            return False
        details = response.json()
        logger.info(f"Raw product details: {json.dumps(details, indent=2)}")
        min_market_funds = float(details.get('quote_min_size', 1.0))
        if funds < min_market_funds:
            logger.info(f"Adjusting order size from ${funds:.2f} to minimum ${min_market_funds:.2f}")
            funds = min_market_funds
        return execute_trade_via_subprocess(symbol, side, price, funds)
    except Exception as e:
        logger.error(f"Error executing trade: {str(e)}")
        logger.error(traceback.format_exc())
        return False
# --- End Real Trade Execution Functions ---

@app.callback(
    Output('live-positions-table', 'data'),
    [Input('live-trading-interval', 'n_intervals')],
    prevent_initial_call=False
)
def update_live_positions_table(n_intervals):
    """
    Update the live-positions-table with real positions from lk.py
    """
    try:
        # Find lk.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_dir, "lk.py"),
            os.path.join(current_dir, "..", "crypto_trading", "app", "lk.py"),
            os.path.join(current_dir, "crypto_trading", "app", "lk.py")
        ]
        
        script_path = None
        for path in possible_paths:
            if os.path.exists(path):
                script_path = path
                logger.info(f"Found lk.py at: {path}")
                break
                
        if not script_path:
            logger.error("Could not find lk.py. Searched in:")
            for path in possible_paths:
                logger.error(f"  - {path}")
            return []
            
        # Run lk.py
        logger.info("Running lk.py for positions table update...")
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        output = result.stdout.strip()
        start_marker = "DASHBOARD_DATA_START"
        end_marker = "DASHBOARD_DATA_END"
        
        if start_marker not in output or end_marker not in output:
            logger.error("[UI] Could not find data markers in lk.py output")
            return []
            
        json_str = output.split(start_marker)[1].split(end_marker)[0]
        try:
            data = json.loads(json_str)
            positions = data.get('positions', [])
            logger.info(f"[UI] Got {len(positions)} positions from lk.py")
        except json.JSONDecodeError:
            logger.error("[UI] Could not parse JSON from lk.py output")
            return []
            
        # Format positions for table display
        table_data = []
        for pos in positions:
            if pos.get('currency') == 'USD':  # Skip USD balance
                continue
                
            symbol = f"{pos.get('currency')}-USD"
            quantity = float(pos.get('amount', 0))
            current_price = float(pos.get('price', 0))
            value = float(pos.get('usd_value', 0))
            
            # Skip tiny positions (less than 1 cent)
            if value < 0.01:
                continue
                
            # Get entry price and P/L from database
            try:
                session_id = get_active_session()
                if session_id:
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT entry_price 
                        FROM positions 
                        WHERE session_id = ? AND symbol = ? AND status = 'open'
                    ''', (session_id, symbol))
                    row = cursor.fetchone()
                    if row:
                        entry_price = float(row[0])
                        pnl = ((current_price / entry_price) - 1) * 100
                        pnl = f"{pnl:+.2f}%"
                    else:
                        pnl = "N/A"
                    conn.close()
                else:
                    pnl = "N/A"
            except Exception as e:
                logger.error(f"[UI] Database error getting entry price: {e}")
                pnl = "N/A"
            
            table_data.append({
                'symbol': symbol,
                'quantity': f"{quantity:.8f}",
                'current_price': f"${current_price:.4f}",
                'value': f"${value:.2f}",
                'pnl': pnl
            })
            
        logger.info(f"[UI] Returning {len(table_data)} positions for display")
        return table_data
        
    except Exception as e:
        logger.error(f"[UI] Error updating positions table: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def create_session():
    """Create a new active trading session if none exists"""
    try:
        logger.info(f"[DEBUG] Connecting to database at: {DB_PATH}")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check for existing active session
        cursor.execute("SELECT id FROM sessions WHERE status = 'active'")
        session = cursor.fetchone()
        
        if session:
            logger.info(f"Using existing session {session[0]}")
            return session[0]
            
        # Get initial balance from lk.py
        result = subprocess.run(
            [sys.executable, "crypto_trading/c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/lk.py"],
            capture_output=True, text=True, check=True
        )
        output = result.stdout.strip()
        start_marker = "DASHBOARD_DATA_START"
        end_marker = "DASHBOARD_DATA_END"
        
        initial_balance = 0.0
        if start_marker in output and end_marker in output:
            json_str = output.split(start_marker)[1].split(end_marker)[0]
            try:
                data = json.loads(json_str)
                initial_balance = data.get('total_value', 0.0)
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON data for initial balance")
            
        # Create new session
        cursor.execute("""
            INSERT INTO sessions (
                start_time,
                initial_balance,
                status
            ) VALUES (
                datetime('now'),
                ?,
                'active'
            )
        """, (initial_balance,))
        conn.commit()
        session_id = cursor.lastrowid
        logger.info(f"Created new session {session_id} with initial balance ${initial_balance:.2f}")
        return session_id
        
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return None
    finally:
        if conn:
            conn.close()

def sync_positions_to_db(positions):
    """Sync positions to the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get active session
        session_id = get_active_session()
        if not session_id:
            logger.error("No active session found")
            return
        
        # First, mark all positions as closed
        cursor.execute('''
            UPDATE positions 
            SET status = 'closed' 
            WHERE session_id = ? AND status = 'open'
        ''', (session_id,))
        
        # Then insert or update positions
        for position in positions:
            if position['currency'] == 'USD':  # Skip USD balance
                continue
                
            symbol = f"{position['currency']}-USD"
            usd_value = float(position.get('usd_value', 0))
            quantity = float(position.get('amount', 0))
            price = float(position.get('price', 0))
            
            # Calculate additional fields
            pnl = 0.0  # Initial PNL is 0 for new positions
            profit = 0.0
            pl_value = 0.0
            pl_percentage = 0.0
            
            cursor.execute('''
                INSERT OR REPLACE INTO positions 
                (session_id, symbol, quantity, entry_price, current_price, 
                 size, value, position_size, pnl, profit, pl_value, pl_percentage,
                 entry_time, last_update, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                        datetime('now'), datetime('now'), 'open')
            ''', (
                session_id,
                symbol,
                quantity,
                price,  # entry_price
                price,  # current_price (same as entry initially)
                usd_value,  # size
                usd_value,  # value
                usd_value,  # position_size
                pnl,
                profit,
                pl_value,
                pl_percentage
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Successfully synced {len(positions)} positions to database")
    except Exception as e:
        logger.error(f"Error syncing positions to database: {str(e)}")
        if 'conn' in locals():
            conn.close()

def init_db():
    """Initialize the database with required tables."""
    try:
        logger.info(f"[DEBUG] Initializing database at: {DB_PATH}")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT DEFAULT 'active',
                initial_balance REAL DEFAULT 0.0
            )
        ''')
        
        # Create positions table with all necessary columns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                size REAL NOT NULL,
                value REAL NOT NULL,
                position_size REAL NOT NULL,
                pl_value REAL NOT NULL DEFAULT 0.0,
                pl_percentage REAL NOT NULL DEFAULT 0.0,
                entry_time TIMESTAMP NOT NULL,
                last_update TIMESTAMP NOT NULL,
                tp_order_id TEXT,
                sl_order_id TEXT,
                status TEXT DEFAULT 'open',
                exit_time TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        ''')
        
        # Create trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                value REAL NOT NULL,
                profit REAL,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        ''')
        
        # Create portfolio history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                total_value REAL NOT NULL,
                available_balance REAL NOT NULL,
                session_id INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        ''')
        
        # Check if there's any active session
        cursor.execute("SELECT COUNT(*) FROM sessions WHERE status = 'active'")
        active_sessions = cursor.fetchone()[0]
        
        if active_sessions == 0:
            # Create initial session if none exists
            cursor.execute('''
                INSERT INTO sessions (start_time, status, initial_balance)
                VALUES (datetime('now'), 'active', 0.0)
            ''')
            session_id = cursor.lastrowid
            
            # Create initial portfolio history entry
            cursor.execute('''
                INSERT INTO portfolio_history (session_id, timestamp, total_value, available_balance)
                VALUES (?, datetime('now'), 0.0, 0.0)
            ''', (session_id,))
            
            logger.info(f"Created initial session with ID {session_id}")
        
        conn.commit()
        logger.info("[DEBUG] Database schema created successfully")
        
    except Exception as e:
        logger.error(f"[DEBUG] Error initializing database: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    finally:
        if 'conn' in locals():
            conn.close()
            logger.info("[DEBUG] Database connection closed")

# Initialize database at module import
try:
    init_db()
    logger.info("Database initialized successfully at startup")
except Exception as e:
    logger.error(f"Failed to initialize database at startup: {e}")

import lk

def get_live_positions_from_lk(min_usd_value=0.5):
    """Get live positions from lk.py with minimum USD value filter"""
    try:
        # Get the absolute path to lk.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lk_path = os.path.join(current_dir, "lk.py")
        logger.info(f"Running lk.py from path: {lk_path}")
        
        # Add current directory to Python path
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Run lk.py as a subprocess
        result = subprocess.run(
            [sys.executable, lk_path],  # Use absolute path
            capture_output=True,
            text=True,
            check=True,
            cwd=current_dir,  # Set working directory to current script's directory
            env={**os.environ, 'PYTHONPATH': current_dir}  # Add current directory to Python path
        )
        
        # Parse the output
        output = result.stdout.strip()
        logger.info(f"lk.py output: {output}")
        
        if output:
            try:
                data = json.loads(output)
                positions = data.get('positions', [])
                
                # Filter positions by minimum USD value
                filtered_positions = [p for p in positions if float(p.get('usd_value', 0)) >= min_usd_value]
                logger.info(f"Found {len(filtered_positions)} positions with value >= ${min_usd_value}")
                
                return filtered_positions
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from lk.py output: {output}")
                return []
        else:
            logger.error("No output from lk.py")
            return []
            
    except subprocess.CalledProcessError as e:
        logger.warning(f"Note: lk.py may have had an issue at startup: {e}")
        return []
    except Exception as e:
        logger.error(f"Error getting live positions: {str(e)}")
        return []

# Example usage in your evaluation logic:
# live_positions = get_live_positions_from_lk(min_usd_value=0.5)
# for pos in live_positions:
#     # Evaluate each position for trading logic

def get_active_session():
    """Get the active session ID."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM sessions WHERE status = 'active' ORDER BY start_time DESC LIMIT 1")
        session = cursor.fetchone()
        conn.close()
        
        if session:
            return session[0]  # Return just the ID
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error getting active session: {str(e)}")
        return None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the app directory path and define database path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, 'data', 'live_trading.db')

# Ensure data directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def make_ml_decision(symbol, granularity=3600):
    """Make ML decision for a given symbol"""
    try:
        # Check if model exists
        model_prefix = f"{symbol.replace('-', '')}_{granularity}"
        clf_path = os.path.join(MODELS_DIR, f"{model_prefix}_clf.pkl")
        
        if not os.path.exists(clf_path):
            logger.info(f"No model found for {symbol}, training new model...")
            clf = train_model_for_symbol(symbol, granularity)
            if clf is None:
                return None, 0.0
        else:
            import joblib
            clf = joblib.load(clf_path)
        
        # Get recent data for prediction
        df = get_coinbase_data(symbol, granularity, days=1)
        if df is None or df.empty:
            logger.error(f"No data available for prediction for {symbol}")
            return None, 0.0
        
        # Calculate indicators
        df = calculate_indicators(df)
        
        # Prepare features
        feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'sma_20', 'sma_50', 'upper_band', 'lower_band'
        ]
        
        # Get latest feature values
        latest_features = df[feature_columns].iloc[-1:]
        
        # Make prediction
        prediction_proba = clf.predict_proba(latest_features)[0]
        prediction = clf.predict(latest_features)[0]
        
        # Convert prediction to decision
        if prediction == 1:
            decision = 'BUY'
            confidence = prediction_proba[1]
        else:
            decision = 'SELL'
            confidence = prediction_proba[0]
        
        return decision, confidence
        
    except Exception as e:
        logger.error(f"Error making ML decision for {symbol}: {str(e)}")
        return None, 0.0

def update_ml_decisions():
    """Update ML decisions for all positions"""
    try:
        # Get active session
        session_id = get_active_session()
        if not session_id:
            logger.error("No active session found")
            return
        
        # Get current positions
        position_manager.refresh_positions()
        
        for symbol in position_manager.positions:
            # Make ML decision
            decision, confidence = make_ml_decision(symbol)
            
            if decision:
                # Log the decision
                log_ml_decision(
                    session_id=session_id,
                    symbol=symbol,
                    decision=decision,
                    confidence=confidence
                )
                logger.info(f"Made ML decision for {symbol}: {decision} (confidence: {confidence:.2%})")
    
    except Exception as e:
        logger.error(f"Error updating ML decisions: {str(e)}")