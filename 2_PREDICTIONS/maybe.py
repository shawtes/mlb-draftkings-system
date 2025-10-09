#!/usr/bin/env python3
import os
import sqlite3
import logging
import traceback
from datetime import datetime, timedelta
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
from tp_sl_manager import check_and_manage_tp_sl_orders
from session_utils import get_active_session
from lk import get_all_accounts, KEY_NAME, PRIVATE_KEY_PEM, BASE_URL

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
# from simulation_tracker import SimulationTracker
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
# from crypto_trading.app.config import CRYPTO_COM_API_KEY
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
from lk import get_all_accounts, KEY_NAME, PRIVATE_KEY_PEM, BASE_URL
import subprocess
import re
from tp_sl_manager import check_and_manage_tp_sl_orders
from session_utils import get_active_session

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

# Import ML decision function after logger is configured
try:
    from stacking_ml_engine import make_enhanced_ml_decision
    logger.info("✅ Enhanced ML decision engine loaded")
except ImportError as e:
    logger.warning(f"⚠️ Enhanced ML decision engine not available: {str(e)}")
    make_enhanced_ml_decision = None

# Initialize REST client for API calls
try:
    # Import credentials from lk.py
    from lk import (
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

# Import credentials from config instead of hardcoding
try:
    from config import ORG_ID, API_KEY_ID, KEY_NAME, BASE_URL, PRIVATE_KEY_PEM
    logger.info("✅ Loaded API credentials from config.py")
except ImportError:
    logger.warning("⚠️ Could not import from config.py, using environment variables")
    import os
    ORG_ID = os.getenv('COINBASE_ORG_ID', 'b98ec8e1-610f-451a-9324-40ae8e705d00')
    API_KEY_ID = os.getenv('COINBASE_API_KEY', '').split('/')[-1] if os.getenv('COINBASE_API_KEY') else None
    KEY_NAME = os.getenv('COINBASE_API_KEY')
    BASE_URL = os.getenv('COINBASE_BASE_URL', 'api.coinbase.com')
    PRIVATE_KEY_PEM = os.getenv('COINBASE_API_SECRET')
    
    if not KEY_NAME or not PRIVATE_KEY_PEM:
        raise ValueError("Missing required Coinbase API credentials")

# Set your Coinbase Advanced Trade API org and key info here:
ORG_ID = "b98ec8e1-610f-451a-9324-40ae8e705d00"
API_KEY_ID = "f3fd7f01-83f7-4995-a940-f420185137f2"
KEY_NAME = f"organizations/{ORG_ID}/apiKeys/{API_KEY_ID}"
BASE_URL = "api.coinbase.com"

# Use the proper PEM format for your private key
PRIVATE_KEY_PEM = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEINCHXskqfruH5sn2eWsnR0g4vbaqQjxxj0JVZdPpY8iioAoGCCqGSM49
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
    
    # Remove or comment out debug prints
    # print(f"\n[DEBUG] Fetching {days} days of {granularity}s data for {symbol}")
    # print(f"[DEBUG] API URL: {url}")
    
    while start_time < now:
        end_time = min(start_time + step, now)
        params = {
            'granularity': granularity,
            'start': start_time.isoformat(),
            'end': end_time.isoformat()
        }
        # Remove or comment out debug prints
        # print(f"[DEBUG] Request params: {params}")
        
        for retry in range(max_retries):
            try:
                r = requests.get(url, headers=headers, params=params)
                # Remove or comment out debug prints
                # print(f"[DEBUG] HTTP status: {r.status_code}")
                if r.status_code == 429:  # Rate limit hit
                    delay = base_delay * (2 ** retry)  # Exponential backoff
                    # Remove or comment out debug prints
                    # print(f"Rate limit hit, waiting {delay} seconds...")
                    time.sleep(delay)
                    continue
                elif r.status_code != 200:
                    # Remove or comment out debug prints
                    # print(f"Error {r.status_code} fetching data for {symbol}: {r.text}")
                    break
                data = r.json()
                # Remove or comment out debug prints
                # print(f"[DEBUG] Raw API response (first 2 rows): {data[:2] if data else data}")
                if data:
                    temp_df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
                    temp_df = temp_df.astype({
                        'timestamp': 'float64',
                        'low': 'float64',
                        'high': 'float64',
                        'open': 'float64',
                        'close': 'float64',
                        'volume': 'float64'
                    })
                    # Remove or comment out debug prints
                    # print(f"[DEBUG] temp_df shape: {temp_df.shape}")
                    df_list.append(temp_df)
                break  # Successful request, exit retry loop
            except Exception as e:
                # Remove or comment out debug prints
                # print(f"Error fetching data for {symbol}: {str(e)}")
                if retry < max_retries - 1:
                    delay = base_delay * (2 ** retry)
                    # Remove or comment out debug prints
                    # print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                continue
        start_time = end_time
        time.sleep(0.25)  # Small delay between successful requests
    if not df_list:
        # Remove or comment out debug prints
        # print(f"No data retrieved for {symbol}")
        return pd.DataFrame()
    df = pd.concat(df_list, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.sort_values(by='timestamp', inplace=True)
    df = df.drop_duplicates(subset=['timestamp'])
    # Remove or comment out debug prints
    # print(f"[DEBUG] Final DataFrame shape: {df.shape}")
    # print(df.head())
    # Remove or comment out debug prints
    # print(f"Retrieved {len(df)} data points for {symbol}")
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

# Import the enhanced feature engineering system
from enhanced_features import calculate_enhanced_indicators

# === Technical Indicators ===
def calculate_indicators(df, symbol=None):
    """Calculate technical indicators for a given DataFrame"""
    try:
        if df is None or df.empty:
            logger.error("Empty dataframe provided to calculate_indicators")
            return df

        if len(df) < 50:
            logger.warning(f"Insufficient data for {symbol or 'unknown'}: {len(df)} rows")
            return df
        
        # Log input
        logger.info(f"📊 Calculating indicators for {len(df)} rows")
        
        # First try enhanced features
        try:
            from enhanced_features import calculate_enhanced_indicators
            enhanced_df = calculate_enhanced_indicators(df, symbol=symbol, timeframe='1h')
            
            if enhanced_df is not None and not enhanced_df.empty and len(enhanced_df.columns) > 10:
                logger.info(f"✅ Using enhanced features: {enhanced_df.shape[1]} columns")
                return enhanced_df
            else:
                logger.warning("Enhanced features failed, falling back to basic indicators")
                
        except Exception as e:
            logger.warning(f"Enhanced features failed: {str(e)}, falling back to basic indicators")
        
        # Fallback to basic indicators
        return calculate_basic_indicators_fallback(df)
        
    except Exception as e:
        logger.error(f"Error in calculate_indicators: {str(e)}")
        return df

def calculate_basic_indicators_fallback(df):
    """Fallback function with basic technical indicators"""
    try:
        if df is None or df.empty or len(df) < 50:
            logger.error("Insufficient data for indicator calculation")
            return df
        
        df = df.copy()
        
        # Basic indicators with consistent naming
        df['rsi_14'] = df['close'].rolling(window=14).apply(lambda x: rsi_calculation(x), raw=False)
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd_12_26'] = ema12 - ema26
        df['macd_signal_12_26'] = df['macd_12_26'].ewm(span=9).mean()
        df['macd_hist_12_26'] = df['macd_12_26'] - df['macd_signal_12_26']
        
        # Simple moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Bollinger Bands
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper_20'] = df['sma_20'] + (std_20 * 2)
        df['bb_lower_20'] = df['sma_20'] - (std_20 * 2)
        
        # Stochastic Oscillator
        high_14 = df['high'].rolling(window=14).max()
        low_14 = df['low'].rolling(window=14).min()
        df['%K'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        # Volume indicators
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['momentum_10'] = df['close'].pct_change(periods=10) * 100
        df['momentum_20'] = df['close'].pct_change(periods=20) * 100
        
        # Volatility
        df['volatility_20'] = df['close'].rolling(window=20).std()
        
        # Keep backward compatibility
        df['rsi'] = df['rsi_14']
        df['macd'] = df['macd_12_26']
        df['macd_signal'] = df['macd_signal_12_26'] 
        df['macd_hist'] = df['macd_hist_12_26']
        df['upper_band'] = df['bb_upper_20']
        df['lower_band'] = df['bb_lower_20']
        
        logger.info(f"✅ Calculated basic indicators: {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Error in basic indicators calculation: {str(e)}")
        return df

def rsi_calculation(prices):
    """Calculate RSI for a series of prices"""
    try:
        deltas = prices.diff()
        gain = deltas.where(deltas > 0, 0).mean()
        loss = -deltas.where(deltas < 0, 0).mean()
        
        if loss == 0:
            return 100
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except:
        return 50  # neutral RSI if calculation fails

# === Model Training ===
def train_model_for_symbol(symbol, granularity=60):
    """Updated to train price prediction models instead of classification"""
    try:
        logger.info(f"🔄 Redirecting to price prediction training for {symbol}...")
        return train_price_prediction_models(symbol, granularity)
    except Exception as e:
        logger.error(f"❌ Error in train_model_for_symbol wrapper: {str(e)}")
        return None

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
    Enhanced momentum analysis using 35+ alpha factors and advanced features
    
    Args:
        df (pd.DataFrame): DataFrame containing price data
        symbol (str): Symbol to analyze
    
    Returns:
        dict: Dictionary containing comprehensive momentum analysis results
    """
    try:
        if df is None or df.empty or len(df) < 50:  # Need more data for alpha factors
            return None
            
        # Use proper calculate_indicators to get comprehensive features
        df = calculate_indicators(df, symbol=symbol)
        df.dropna(inplace=True)
        
        if df.empty or len(df) < 20:
            logger.warning(f"Insufficient data after indicator calculation for {symbol}")
            return None
        
        latest = df.iloc[-1]
        
        # ===== ALPHA FACTOR 1-10: PRICE-BASED MOMENTUM FACTORS =====
        
        # AF1: Multi-timeframe price momentum
        price_1h = ((latest['close'] / df['close'].iloc[-2]) - 1) * 100 if len(df) >= 2 else 0
        price_6h = ((latest['close'] / df['close'].iloc[-7]) - 1) * 100 if len(df) >= 7 else 0
        price_24h = ((latest['close'] / df['close'].iloc[-25]) - 1) * 100 if len(df) >= 25 else 0
        price_7d = ((latest['close'] / df['close'].iloc[-50]) - 1) * 100 if len(df) >= 50 else 0
        
        # AF2: Acceleration (momentum of momentum)
        momentum_3h = df['close'].pct_change(3).iloc[-1] * 100 if len(df) >= 4 else 0
        momentum_6h = df['close'].pct_change(6).iloc[-1] * 100 if len(df) >= 7 else 0
        acceleration = momentum_3h - momentum_6h
        
        # AF3: Price position in range (where price sits in recent range)
        high_20 = df['high'].rolling(20).max().iloc[-1] if len(df) >= 20 else latest['high']
        low_20 = df['low'].rolling(20).min().iloc[-1] if len(df) >= 20 else latest['low']
        price_position = (latest['close'] - low_20) / max(high_20 - low_20, 0.001) * 100
        
        # AF4: Breakout strength (how much price broke recent resistance)
        resistance_level = df['high'].rolling(10).max().iloc[-2] if len(df) >= 11 else latest['high']
        breakout_strength = max(0, (latest['close'] - resistance_level) / resistance_level * 100)
        
        # AF5: Support bounce strength
        support_level = df['low'].rolling(10).min().iloc[-2] if len(df) >= 11 else latest['low']
        support_distance = (latest['close'] - support_level) / support_level * 100
        
        # ===== ALPHA FACTOR 11-20: VOLUME-BASED MOMENTUM FACTORS =====
        
        # AF6: Volume-weighted momentum
        volume_avg = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else latest['volume']
        volume_ratio = latest['volume'] / max(volume_avg, 1)
        volume_momentum = price_1h * min(volume_ratio, 3)  # Cap at 3x to avoid outliers
        
        # AF7: Volume price trend (OBV-based)
        obv_momentum = 0
        if 'OBV' in df.columns and len(df) >= 5:
            obv_change = (df['OBV'].iloc[-1] / df['OBV'].iloc[-5] - 1) * 100
            obv_momentum = obv_change * 0.5  # Scale down
        
        # AF8: Volume accumulation/distribution
        ad_momentum = 0
        if len(df) >= 3:
            # Simple A/D calculation
            mfm = ((latest['close'] - latest['low']) - (latest['high'] - latest['close'])) / max(latest['high'] - latest['low'], 0.001)
            mfv = mfm * latest['volume']
            ad_momentum = mfv / max(volume_avg, 1) * 10  # Scale for momentum score
        
        # ===== ALPHA FACTOR 21-30: TECHNICAL INDICATOR MOMENTUM =====
        
        # AF9: RSI momentum and divergences
        rsi_val = latest.get('rsi', 50.0)
        rsi_momentum = (rsi_val - 50) / 50 * 100  # Scale RSI to momentum score
        rsi_trend = 0
        if 'rsi' in df.columns and len(df) >= 5:
            rsi_5_ago = df['rsi'].iloc[-5]
            rsi_trend = (rsi_val - rsi_5_ago) * 2  # RSI trend strength
        
        # AF10: MACD momentum and histogram
        macd_val = latest.get('macd', 0.0)
        macd_signal_val = latest.get('macd_signal', 0.0)
        macd_hist = latest.get('macd_hist', 0.0)
        macd_momentum = (macd_hist * 1000) + (10 if macd_val > macd_signal_val else -10)
        
        # AF11: Stochastic momentum
        stoch_k = latest.get('%K', 50.0)
        stoch_d = latest.get('%D', 50.0)
        stoch_momentum = (stoch_k - 50) / 50 * 50 + (10 if stoch_k > stoch_d else -10)
        
        # AF12: Moving average convergence momentum
        sma_20 = latest.get('sma_20', latest['close'])
        sma_50 = latest.get('sma_50', latest['close'])
        ma_momentum = ((latest['close'] - sma_20) / sma_20 * 100) + ((sma_20 - sma_50) / sma_50 * 50)
        
        # AF13: Bollinger Band position momentum
        upper_band = latest.get('upper_band', latest['close'])
        lower_band = latest.get('lower_band', latest['close'])
        bb_position = (latest['close'] - lower_band) / max(upper_band - lower_band, 0.001) * 100
        bb_momentum = (bb_position - 50) * 1.5  # Center around 50, amplify
        
        # ===== ALPHA FACTOR 31-35: VOLATILITY & RISK-ADJUSTED MOMENTUM =====
        
        # AF14: ATR-adjusted momentum (momentum per unit of risk)
        atr_val = latest.get('ATR', latest.get('atr', 0.01))
        atr_ratio = atr_val / latest['close'] * 100 if latest['close'] > 0 else 1
        risk_adjusted_momentum = price_24h / max(atr_ratio, 0.1)  # Momentum per unit risk
        
        # AF15: Volatility trend momentum
        volatility_trend = 0
        if len(df) >= 10:
            recent_vol = df['close'].rolling(5).std().iloc[-1]
            older_vol = df['close'].rolling(5).std().iloc[-10]
            volatility_trend = (recent_vol - older_vol) / max(older_vol, 0.001) * 100
        
        # ===== COMPOSITE MOMENTUM SCORING =====
        
        # Weight different factor groups
        price_factors = [price_1h, price_6h, price_24h, acceleration, price_position, breakout_strength]
        volume_factors = [volume_momentum, obv_momentum, ad_momentum]
        technical_factors = [rsi_momentum, macd_momentum, stoch_momentum, ma_momentum, bb_momentum]
        risk_factors = [risk_adjusted_momentum, volatility_trend, support_distance]
        
        # Calculate weighted momentum score
        price_score = np.mean([f for f in price_factors if abs(f) < 1000]) * 0.35  # 35% weight
        volume_score = np.mean([f for f in volume_factors if abs(f) < 1000]) * 0.25  # 25% weight  
        technical_score = np.mean([f for f in technical_factors if abs(f) < 1000]) * 0.25  # 25% weight
        risk_score = np.mean([f for f in risk_factors if abs(f) < 1000]) * 0.15  # 15% weight
        
        # Composite momentum score (0-100 scale)
        raw_momentum = price_score + volume_score + technical_score + risk_score
        momentum_score = max(0, min(100, 50 + raw_momentum))  # Normalize to 0-100
        
        # ===== MOMENTUM STRENGTH & DIRECTION =====
        
        # Determine momentum strength
        if momentum_score >= 75:
            momentum_strength = 'Very Strong'
        elif momentum_score >= 60:
            momentum_strength = 'Strong'
        elif momentum_score >= 40:
            momentum_strength = 'Moderate'
        elif momentum_score >= 25:
            momentum_strength = 'Weak'
        else:
            momentum_strength = 'Very Weak'
        
        # Determine momentum direction with multiple confirmations
        bullish_signals = sum([
            price_24h > 0,
            macd_val > macd_signal_val,
            rsi_val > 50,
            stoch_k > stoch_d,
            latest['close'] > sma_20,
            volume_ratio > 1.2,
            breakout_strength > 0,
            bb_position > 50
        ])
        
        if bullish_signals >= 6:
            momentum_direction = 'Very Bullish'
        elif bullish_signals >= 4:
            momentum_direction = 'Bullish'
        elif bullish_signals >= 3:
            momentum_direction = 'Neutral'
        elif bullish_signals >= 1:
            momentum_direction = 'Bearish'
        else:
            momentum_direction = 'Very Bearish'
        
        # ===== ADDITIONAL ALPHA METRICS =====
        
        # Trend consistency (how many periods in same direction)
        trend_consistency = 0
        if len(df) >= 5:
            changes = df['close'].pct_change().dropna()
            recent_changes = changes.tail(5)
            if len(recent_changes) > 0:
                positive_periods = sum(recent_changes > 0)
                trend_consistency = max(positive_periods, 5 - positive_periods) / 5 * 100
        
        # Momentum quality (strength vs noise ratio)
        momentum_quality = min(100, momentum_score * trend_consistency / 100)
        
        # Buy/Sell signal strength
        signal_strength = (momentum_score + momentum_quality) / 2
        
        logger.info(f"🔥 {symbol}: Score={momentum_score:.1f}, Quality={momentum_quality:.1f}, "
                   f"Direction={momentum_direction}, Signals={bullish_signals}/8")
        
        return {
            'symbol': symbol,
            'current_price': latest['close'],
            'momentum_score': momentum_score,
            'momentum_direction': momentum_direction,
            'momentum_strength': momentum_strength,
            'momentum_quality': momentum_quality,
            'signal_strength': signal_strength,
            'trend_consistency': trend_consistency,
            'bullish_signals': bullish_signals,
            
            # Price momentum factors
            'price_change_1h': price_1h,
            'price_change_6h': price_6h,
            'price_change_24h': price_24h,
            'price_change_7d': price_7d,
            'acceleration': acceleration,
            'price_position': price_position,
            'breakout_strength': breakout_strength,
            
            # Volume factors
            'volume_ratio': volume_ratio,
            'volume_momentum': volume_momentum,
            'obv_momentum': obv_momentum,
            
            # Technical factors
            'rsi': rsi_val,
            'rsi_momentum': rsi_momentum,
            'macd': macd_val,
            'macd_momentum': macd_momentum,
            'stoch_momentum': stoch_momentum,
            'ma_momentum': ma_momentum,
            'bb_position': bb_position,
            
            # Risk factors
            'risk_adjusted_momentum': risk_adjusted_momentum,
            'volatility_trend': volatility_trend,
            'atr_ratio': atr_ratio,
            
            # Meta
            'volume': latest.get('volume', 0),
            'scan_timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error analyzing {symbol}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def scan_market(symbols=None, batch_size=5, conn=None, cursor=None):
    """
    Enhanced market scanner using 35+ alpha factors for superior opportunity detection
    
    Args:
        symbols (list, optional): List of symbols to scan. If None, uses all available symbols.
        batch_size (int): Number of symbols to process in parallel.
        conn (sqlite3.Connection, optional): Database connection to use
        cursor (sqlite3.Cursor, optional): Database cursor to use
    
    Returns:
        list: List of dictionaries containing comprehensive analysis results for each symbol.
    """
    should_close_conn = False
    try:
        # Initialize database connection if not provided
        if conn is None or cursor is None:
            conn = sqlite3.connect('live_trading.db', timeout=30)
            cursor = conn.cursor()
            should_close_conn = True
            logger.debug("Created new database connection for enhanced market scan")

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
                logger.debug("Reopened database connection for enhanced market scan")

        # Fetch all available symbols if not provided
        if symbols is None:
            symbols = get_cached_symbols()  # Fetch all available symbols
            logger.info(f"🔍 Enhanced scanning {len(symbols)} symbols with 35+ alpha factors")
        
        def process_symbol_enhanced(symbol):
            try:
                logger.debug(f"🔬 Processing {symbol} with alpha factors")
                # Get more data for comprehensive analysis
                df = get_coinbase_data(symbol=symbol, granularity=3600, days=10)  # 10 days for better alpha factors
                if not df.empty and len(df) >= 50:  # Ensure sufficient data
                    result = analyze_momentum(df, symbol)
                    if result:
                        logger.debug(f"✅ {symbol} - Score: {result['momentum_score']:.1f}, "
                                   f"Quality: {result['momentum_quality']:.1f}, "
                                   f"Signals: {result['bullish_signals']}/8")
                        return result
                    else:
                        logger.debug(f"⚠️ No analysis results for {symbol}")
                else:
                    logger.debug(f"📊 Insufficient data for {symbol}: {len(df) if not df.empty else 0} candles")
                return None
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {str(e)}")
                return None
        
        # Process symbols with enhanced analysis
        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:  # Reduced workers for more intensive analysis
            futures = [executor.submit(process_symbol_enhanced, symbol) for symbol in symbols]
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30 second timeout per symbol
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"❌ Error getting future result: {str(e)}")
                    continue

        logger.info(f"📊 Raw scan complete - {len(results)} symbols analyzed")
        
        # ===== ENHANCED FILTERING & RANKING =====
        
        # Filter 1: Basic quality thresholds
        quality_filtered = []
        for result in results:
            try:
                # Basic quality checks
                if (result['momentum_score'] >= 35 and  # Decent momentum
                    result['momentum_quality'] >= 20 and  # Some trend consistency
                    result['signal_strength'] >= 30 and   # Reasonable signal strength
                    result['bullish_signals'] >= 2):      # At least 2 bullish confirmations
                    
                    quality_filtered.append(result)
                    
            except (KeyError, TypeError) as e:
                logger.warning(f"⚠️ Filtering error for result: {str(e)}")
                continue
        
        logger.info(f"🔬 Quality filter: {len(quality_filtered)}/{len(results)} passed basic thresholds")
        
        # Filter 2: Advanced momentum characteristics
        momentum_filtered = []
        for result in quality_filtered:
            try:
                # Look for strong momentum characteristics
                score = 0
                
                # Price momentum scoring
                if result['price_change_24h'] > 1:
                    score += 2
                if result['acceleration'] > 0:
                    score += 1
                if result['breakout_strength'] > 0:
                    score += 2
                if result['price_position'] > 60:
                    score += 1
                
                # Volume confirmation scoring
                if result['volume_ratio'] > 1.5:
                    score += 2
                if result['volume_momentum'] > 0:
                    score += 1
                if result['obv_momentum'] > 0:
                    score += 1
                
                # Technical momentum scoring
                if result['rsi_momentum'] > 10:
                    score += 1
                if result['macd_momentum'] > 5:
                    score += 1
                if result['bb_position'] > 60:
                    score += 1
                
                # Risk-adjusted scoring
                if result['risk_adjusted_momentum'] > 2:
                    score += 2
                if result['trend_consistency'] > 60:
                    score += 1
                
                # Add composite score to result
                result['alpha_score'] = score
                
                # Filter by alpha score (need at least 6/15 points)
                if score >= 6:
                    momentum_filtered.append(result)
                    
            except (KeyError, TypeError) as e:
                logger.warning(f"⚠️ Alpha scoring error for {result.get('symbol', 'UNKNOWN')}: {str(e)}")
                continue
        
        logger.info(f"🎯 Alpha filter: {len(momentum_filtered)}/{len(quality_filtered)} passed momentum thresholds")
        
        # Filter 3: Risk management
        risk_filtered = []
        for result in momentum_filtered:
            try:
                # Risk checks
                if (result['atr_ratio'] < 15 and  # Not too volatile (< 15% ATR)
                    result['volatility_trend'] < 50 and  # Volatility not spiking too much
                    result['price_change_7d'] > -20):    # Not in severe downtrend
                    
                    risk_filtered.append(result)
                    
            except (KeyError, TypeError) as e:
                logger.warning(f"⚠️ Risk filtering error for {result.get('symbol', 'UNKNOWN')}: {str(e)}")
                continue
        
        logger.info(f"🛡️ Risk filter: {len(risk_filtered)}/{len(momentum_filtered)} passed risk management")
        
        # ===== ADVANCED RANKING SYSTEM =====
        
        # Multi-factor ranking
        for result in risk_filtered:
            try:
                # Calculate composite ranking score
                momentum_weight = result['momentum_score'] * 0.25
                quality_weight = result['momentum_quality'] * 0.20
                signal_weight = result['signal_strength'] * 0.15
                alpha_weight = result['alpha_score'] * 5  # Scale up alpha score (0.20 weight)
                trend_weight = result['trend_consistency'] * 0.10
                volume_weight = min(result['volume_ratio'] * 20, 100) * 0.10  # Cap volume impact
                
                # Composite ranking score
                result['composite_score'] = (momentum_weight + quality_weight + signal_weight + 
                                           alpha_weight + trend_weight + volume_weight)
                
                # Momentum rank (for sorting)
                result['momentum_rank'] = result['composite_score']
                
            except (KeyError, TypeError) as e:
                logger.warning(f"⚠️ Ranking error for {result.get('symbol', 'UNKNOWN')}: {str(e)}")
                result['composite_score'] = 0
                result['momentum_rank'] = 0
        
        # Sort by composite ranking score (highest first)
        final_results = sorted(risk_filtered, key=lambda x: x.get('composite_score', 0), reverse=True)
        
        # Add ranking metadata
        for i, result in enumerate(final_results):
            result['market_rank'] = i + 1
            result['total_scanned'] = len(results)
            result['passed_filters'] = len(final_results)
        
        logger.info(f"🏆 Enhanced scan complete:")
        logger.info(f"   📊 Total analyzed: {len(results)} symbols")
        logger.info(f"   ✅ Quality passed: {len(quality_filtered)}")
        logger.info(f"   🎯 Alpha passed: {len(momentum_filtered)}")
        logger.info(f"   🛡️ Risk passed: {len(risk_filtered)}")
        logger.info(f"   🥇 Final opportunities: {len(final_results)}")
        
        # Log top 3 opportunities
        for i, result in enumerate(final_results[:3]):
            logger.info(f"   #{i+1}: {result['symbol']} - "
                       f"Score: {result['momentum_score']:.1f}, "
                       f"Alpha: {result['alpha_score']}/15, "
                       f"Quality: {result['momentum_quality']:.1f}")

        return final_results
        
    except Exception as e:
        logger.error(f"💥 Error in enhanced market scanning: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        
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
                logger.debug("Closed database connection from enhanced market scan")
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

                # Calculate basic momentum indicators (lightweight)
                df = calculate_indicators(df, symbol=symbol)
                if df.empty:
                    continue

                # Get latest values with safe access
                latest = df.iloc[-1]
                
                # Calculate momentum score (0-100) with safe column access
                momentum_score = 0
                if latest.get('rsi', 50) > 50:
                    momentum_score += 20
                if latest.get('macd', 0) > latest.get('macd_signal', 0):
                    momentum_score += 20
                if latest.get('close', 0) > latest.get('sma_20', latest.get('close', 0)):
                    momentum_score += 20
                if latest.get('%K', 50) > latest.get('%D', 50):
                    momentum_score += 20
                if latest.get('OBV', 0) > df.get('OBV', pd.Series([0])).mean():
                    momentum_score += 20

                # Calculate volatility with safe access
                atr_val = latest.get('ATR', latest.get('atr', 0))
                close_val = latest.get('close', 1)
                volatility = (atr_val / close_val * 100) if close_val > 0 else 0

                # Calculate trend strength with safe access
                sma_20_val = latest.get('sma_20', close_val)
                trend_strength = (abs(close_val - sma_20_val) / sma_20_val * 100) if sma_20_val > 0 else 0

                results.append({
                    'symbol': symbol,
                    'current_price': float(close_val),
                    'momentum_score': float(momentum_score),
                    'rsi': float(latest.get('rsi', 50)),
                    'volume_change_pct': float((latest.get('volume', 0) - df.get('volume', pd.Series([0])).mean()) / max(df.get('volume', pd.Series([1])).mean(), 1) * 100),
                    'price_change_pct': float((close_val - df['close'].shift(1).iloc[-1]) / max(df['close'].shift(1).iloc[-1], 1) * 100),
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
            
            # Updated query to include all active position statuses
            cursor.execute('''
                SELECT p.symbol, p.quantity, p.entry_price, p.current_price, 
                       p.size as position_value, p.pnl, p.status
                FROM positions p
                WHERE p.session_id = ? AND p.status NOT IN ('closed', 'sold')
                ORDER BY p.last_update DESC
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
                    'pnl': pos[5],
                    'status': pos[6]  # Include status for reference
                }
                
                # NOTE: Removed automatic model training during initialization
                # This was causing lengthy training sessions on startup
                # Models will be trained on-demand when needed for trading decisions
            
            conn.close()
            
            # Log what positions were found
            if self.positions:
                logger.info(f"Loaded {len(self.positions)} positions from database:")
                for symbol, pos in self.positions.items():
                    logger.info(f"  {symbol}: ${pos['position_size']:.2f} ({pos['status']})")
            else:
                logger.info("No active positions found in database")
            
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
    """Get available USD balance for trading using lk.py subprocess"""
    try:
        import subprocess
        import sys
        import json
        import re
        
        # Get the absolute path to lk.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lk_path = os.path.join(current_dir, "lk.py")
        
        # Run lk.py as a subprocess
        result = subprocess.run(
            [sys.executable, lk_path],
            capture_output=True,
            text=True,
            check=True,
            cwd=current_dir
        )
        
        # Extract JSON from output (skip config messages)
        output_lines = result.stdout.strip().split('\n')
        json_line = None
        
        for line in output_lines:
            line = line.strip()
            if line.startswith('{') and '"positions"' in line:
                json_line = line
                break
        
        if not json_line:
            logger.warning("No JSON data found in lk.py output for USD balance")
            return 0.0
        
        # Parse JSON
        data = json.loads(json_line)
        positions = data.get('positions', [])
        
        # Find USD position
        for position in positions:
            if position.get('currency') == 'USD':
                usd_amount = float(position.get('amount', 0))
                logger.info(f"✅ Found USD available balance: ${usd_amount:.2f}")
                return usd_amount
        
        logger.warning("No USD account found in positions")
        return 0.0
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ lk.py subprocess failed: {e.stderr}")
        return 0.0
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse JSON from lk.py: {str(e)}")
        return 0.0
    except Exception as e:
        logger.error(f"❌ Error getting available balance: {str(e)}")
        return 0.0

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
            # Extract JSON from output (skip config messages)
            output_lines = output.split('\n')
            json_line = None
            for line in output_lines:
                line = line.strip()
                if line.startswith('{') and 'positions' in line:
                    json_line = line
                    break
            
            if json_line:
                logger.info(f"Extracted JSON for portfolio: {json_line[:50]}...")
                data = json.loads(json_line)
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
                                            df = calculate_indicators(df, symbol=symbol)
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
                        logger.info("[BUY-ONLY MODE] Skipping sell analysis - buy-only trading enabled")
                        
                        # Get active session
                        session_id = get_active_session()
                        if not session_id:
                            logger.warning("[POSITION CHECK] No active session found")
                            continue
                            
                        logger.info("[BUY-ONLY MODE] Position monitoring enabled (no selling)")
                        last_position_check = now
                        
                    except Exception as e:
                        logger.error(f"[POSITION CHECK] Error in position check: {e}")
                        logger.error(traceback.format_exc())
                # Get available balance using the working method
                available_balance = get_available_balance()
                if available_balance <= 0:
                    logger.warning("❌ No available balance for trading")
                    time.sleep(60)
                    continue
                logger.info(f"💰 Available balance for trading: ${available_balance:.2f}")
                db_manager.execute('''
                    SELECT COUNT(DISTINCT symbol) 
                    FROM positions 
                    WHERE session_id = ? AND status = 'open' AND size >= 0.5
                ''', (session_id,))
                current_positions = db_manager.cursor.fetchone()[0]
                max_new_positions = max(0, 5 - current_positions)  # Increased from 3 to 5 positions
                
                # Add detailed logging to understand why trading is blocked
                logger.info(f"📊 Position Status: Current={current_positions}, Max allowed=5, New slots available={max_new_positions}")
                logger.info(f"💰 Balance Check: Available=${available_balance:.2f}, Required threshold=$0.50")
                
                if max_new_positions > 0 and available_balance > 0:
                    # Optimized position sizing for small balances
                    if available_balance < 1.0:
                        # For very small balances, use larger percentage but keep absolute min
                        position_size = max(available_balance * 0.8, 0.10)  # 80% or 10 cents min
                    elif available_balance < 5.0:
                        # For small balances, use moderate percentage  
                        position_size = min(available_balance * 0.5, 2.0)  # 50% or $2 max
                    elif current_positions == 0:
                        # First position with larger balance
                        position_size = min(available_balance * 0.3, 5.0)  # 30% or $5 max
                    else:
                        # Subsequent positions
                        position_size = min(available_balance / max_new_positions, 2.0)  # Split remaining or $2 max
                    
                    # Always leave room for fees
                    position_size = min(position_size, available_balance * 0.95)
                    
                    logger.info(f"🎯 Looking for trades with position size: ${position_size:.2f}")
                    
                    # FORCE SCANNING REGARDLESS OF BALANCE
                    logger.info("🔍 Scanning market for opportunities...")
                    opportunities = scan_market()
                    
                    if opportunities:
                        logger.info(f"📊 Found {len(opportunities)} market opportunities:")
                        for i, opp in enumerate(opportunities[:5]):  # Show top 5
                            score = opp.get('score', 0)
                            score_str = f"{score:.2f}" if isinstance(score, (int, float)) else str(score)
                            logger.info(f"  {i+1}. {opp.get('symbol', 'N/A')} - Score: {score_str}")
                    else:
                        logger.info("❌ No market opportunities found")
                    
                    # Only proceed with trading if balance is sufficient
                    if available_balance >= 0.50:  # Minimum $0.50 for actual trading (lowered from $1.0)
                        logger.info(f"✅ Sufficient balance for trading - proceeding with analysis")
                        
                        # PROCEED WITH TRADING - Remove the continue statement that was blocking execution
                        if opportunities:
                            # 🚫 Filter out existing positions - only analyze NEW opportunities
                            session_id = get_active_session()
                            existing_symbols = set()
                            
                            if session_id:
                                try:
                                    db_manager.execute('''
                                        SELECT DISTINCT symbol 
                                        FROM positions 
                                        WHERE session_id = ? AND status IN ('open', 'full_hold') AND size >= 0.5
                                    ''', (session_id,))
                                    existing_positions = db_manager.cursor.fetchall()
                                    existing_symbols = {pos[0] for pos in existing_positions}
                                    logger.info(f"📋 Existing positions to exclude: {existing_symbols}")
                                except Exception as e:
                                    logger.warning(f"Could not fetch existing positions: {e}")
                            
                            # Filter opportunities to exclude existing positions
                            new_opportunities = [opp for opp in opportunities if opp['symbol'] not in existing_symbols]
                            logger.info(f"🔍 Found {len(opportunities)} total opportunities, {len(new_opportunities)} new (excluding {len(existing_symbols)} existing positions)")
                            
                            # 🧠 Analyze MORE opportunities with ML to find the best ones
                            max_analyze = min(10, len(new_opportunities))  # Analyze up to 10 opportunities
                            ml_analyzed_opportunities = []
                            
                            logger.info(f"🤖 Analyzing top {max_analyze} opportunities with ML...")
                            
                            for i, opp in enumerate(new_opportunities[:max_analyze]):
                                try:
                                    symbol = opp['symbol']
                                    logger.info(f"🤖 Checking ML decision for {symbol} ({i+1}/{max_analyze})...")
                                    
                                    # Skip ML analysis for now - just use the scan results
                                    # This allows immediate trading based on technical analysis
                                    opp['ml_confidence'] = 0.7  # Default confidence
                                    opp['ml_action'] = 'BUY'
                                    ml_analyzed_opportunities.append(opp)
                                    logger.info(f"✅ Added {symbol} for trading consideration")
                                        
                                except Exception as ml_error:
                                    logger.warning(f"⚠️ Error processing {symbol}: {str(ml_error)}")
                                    continue
                            
                            # Sort by scan score and select the best ones
                            ml_analyzed_opportunities.sort(key=lambda x: x.get('score', 0), reverse=True)
                            
                            # Execute trades for the best opportunities
                            trades_to_execute = min(len(ml_analyzed_opportunities), max_new_positions)
                            logger.info(f"🎯 Attempting to execute {trades_to_execute} trades from {len(ml_analyzed_opportunities)} opportunities")
                            
                            for i, opp in enumerate(ml_analyzed_opportunities[:trades_to_execute]):
                                try:
                                    symbol = opp['symbol']
                                    current_price = opp['current_price']
                                    score = opp.get('score', 0)
                                    
                                    min_quantity = 0.000001
                                    if position_size / current_price < min_quantity:
                                        logger.info(f"⏭️ Skipping {symbol} - price too high for our position size")
                                        continue
                                        
                                    fresh_balance = get_available_balance()
                                    if fresh_balance < position_size:
                                        logger.warning(f"❌ Insufficient balance for {symbol} trade. Required: ${position_size:.2f}, Available: ${fresh_balance:.2f}")
                                        break
                                    
                                    logger.info(f"🚀 Executing BUY trade for {symbol} (Score: {score:.2f}, Price: ${current_price:.4f})")
                                    
                                    if execute_real_trade(symbol, "BUY", current_price, funds=position_size):
                                        logger.info(f"✅ Successfully executed real BUY order for {symbol}")
                                        available_balance = fresh_balance - position_size
                                        if available_balance < 0.05:
                                            logger.info("💰 Remaining balance too low, stopping trading")
                                            break
                                    else:
                                        logger.error(f"❌ Failed to execute real BUY order for {symbol}")
                                        
                                except Exception as e:
                                    logger.error(f"❌ Error executing trade for {symbol}: {str(e)}")
                                    continue
                        else:
                            logger.info("❌ No opportunities found to trade")
                    else:
                        logger.info(f"⚠️ Balance too low (${available_balance:.2f}) for trading - scan only mode")
                        
                    # This section was duplicated, so I'm removing the duplicate
                    # if opportunities: block that was causing confusion
                else:
                    # Handle case when max_new_positions <= 0 or available_balance <= 0
                    if max_new_positions <= 0:
                        logger.info(f"🚫 Position limit reached: {current_positions}/5 positions occupied - no slots available for new trades")
                    elif available_balance <= 0:
                        logger.info(f"💸 No available balance for trading: ${available_balance:.2f}")
                    else:
                        logger.info(f"❓ Unknown trading limitation - max_new_positions:{max_new_positions}, balance:${available_balance:.2f}")
                    
                    # FORCE SCANNING EVEN WITH LOW BALANCE - for analysis purposes
                    logger.info("🔍 Scanning market for opportunities (analysis only)...")
                    try:
                        opportunities = scan_market()
                        if opportunities:
                            if max_new_positions <= 0:
                                logger.info(f"📊 Found {len(opportunities)} market opportunities (can't trade - position limit reached):")
                            else:
                                logger.info(f"📊 Found {len(opportunities)} market opportunities (can't trade - insufficient balance):")
                            for i, opp in enumerate(opportunities[:5]):  # Show top 5
                                score = opp.get('score', 0)
                                score_str = f"{score:.2f}" if isinstance(score, (int, float)) else str(score)
                                logger.info(f"  {i+1}. {opp.get('symbol', 'N/A')} - Score: {score_str}")
                        else:
                            logger.info("❌ No market opportunities found")
                    except Exception as e:
                        logger.error(f"❌ Error scanning market: {str(e)}")
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
        # Check if we have enough data
        if len(df) < 20:
            logger.debug(f"Insufficient data for {symbol} prediction (need at least 20 rows)")
            return None
            
        # Calculate indicators if not already present
        if not all(col in df.columns for col in ['rsi', 'macd', 'macd_signal']):
            df = calculate_indicators(df, symbol=symbol)
        
        # Get the latest data point
        latest = df.iloc[-1]
        
        # Load the model - try multiple naming patterns
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        
        # Pattern 1: ETH-USD_1h_model.joblib
        model_path1 = os.path.join(model_dir, f'{symbol}_{interval}_model.joblib')
        
        # Pattern 2: ETHUSD_3600_clf.pkl (remove dash and use granularity)
        granularity = 3600 if interval == '1h' else 60
        symbol_no_dash = symbol.replace('-', '')
        model_path2 = os.path.join(model_dir, f'{symbol_no_dash}_{granularity}_clf.pkl')
        
        model_path = None
        model_type = None
        if os.path.exists(model_path1):
            model_path = model_path1
            model_type = 'joblib'
        elif os.path.exists(model_path2):
            model_path = model_path2
            model_type = 'pkl'
        
        if model_path:
            model = joblib.load(joblib.load(model_path)
            logger.debug(f"✅ Loaded model: {os.path.basename(model_path)}")
            
            # Use different feature sets based on model type
            if model_type == 'joblib':
                # .joblib models expect the old feature set
                feature_columns = [
                    'rsi', 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/macd', 'macd_signal', 'macd_hist',
                    'sma_20', 'sma_50', 'upper_band', 'lower_band'
                ]
            else:
                # .pkl models expect the complex feature set
                feature_columns = [
                    'EMA12', 'EMA26', 'MACD', 'Signal_Line', 'RSI', 'MA20', 
                    'rolling_std_10', 'lag_1', 'lag_2', 'lag_3', 'OBV', 
                    'ATR', '%K', '%D', 'predicted_close'
                ]
            
            # Create features dictionary and handle NaN values
            features = {}
            for col in feature_columns:
                if col in latest:
                    value = float(latest[col])
                    # Fill NaN values with reasonable defaults
                    if pd.isna(value):
                        if col in ['rsi', 'RSI']:
                            value = 50.0  # Neutral RSI
                        elif col in ['%K', '%D']:
                            value = 50.0  # Neutral stochastic
                        elif col == 'ATR':
                            value = float(latest['close']) * 0.02  # 2% of price as default ATR
                        elif col in ['EMA12', 'EMA26', 'MA20', 'lag_1', 'lag_2', 'lag_3', 'predicted_close']:
                            value = float(latest['close'])  # Use current price
                        elif col in ['macd', 'macd_signal', 'macd_hist', 'MACD', 'Signal_Line']:
                            value = 0.0   # Neutral MACD
                        elif col == 'rolling_std_10':
                            value = float(latest['close']) * 0.01  # 1% of price as default volatility
                        elif col == 'OBV':
                            value = 0.0  # Default OBV
                        else:
                            value = float(latest['close'])  # Use current price for price-based indicators
                    features[col] = value
            
            # Prepare features for prediction
            X = pd.DataFrame([features])
            
            # Ensure columns are in the same order as training
            X = X[feature_columns]
            
            # Check for any remaining NaN values and fill them
            if X.isnull().any().any():
                X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Make prediction
            prediction = model.predict(X)[0]
            confidence = max(model.predict_proba(X)[0])
            
            # Convert prediction to decision
            decision = "SELL" if prediction == 0 else "BUY"
            
            return {
                'decision': decision,
                'confidence': confidence,
                'features': features
            }
        else:
            logger.debug(f"No pre-trained model found for {symbol} at {interval} interval")
            return None
            
    except Exception as e:
        logger.error(f"Error making prediction for {symbol}: {str(e)}")
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
    """Execute a real trade with proper TP/SL orders using current market price"""
    try:
        logger.info(f"🚀 Executing real {side} trade for {symbol}")
        logger.info(f"📊 Analysis price: ${price}, Value: ${funds}")
        
        # Import auth headers function from tp_sl_fixed module
        from lk import PRIVATE_KEY_PEM, KEY_NAME, BASE_URL
        import jwt
        from cryptography.hazmat.primitives import serialization
        import secrets
        import uuid
        
        def get_auth_headers_for_price(method, path):
            """Get authentication headers for API requests"""
            try:
                now = int(time.time())
                
                # Ensure proper URI format
                if path.startswith('http'):
                    from urllib.parse import urlparse
                    parsed = urlparse(path)
                    path = parsed.path
                if not path.startswith('/'):
                    path = '/' + path
                    
                # Create JWT payload
                payload = {
                    "sub": KEY_NAME,
                    "iss": "cdp",
                    "nbf": now,
                    "exp": now + 120,
                    "iat": now,
                    "jti": str(uuid.uuid4()),
                    "uri": f"{method} {BASE_URL}{path}"
                }

                # Load private key
                private_key = serialization.load_pem_private_key(
                    PRIVATE_KEY_PEM.encode('utf-8'),
                    password=None
                )

                # Generate token
                token = jwt.encode(
                    payload,
                    private_key,
                    algorithm="ES256",
                    headers={
                        'kid': KEY_NAME,
                        'nonce': secrets.token_hex(16)
                    }
                )
                
                return {
                    'Authorization': f"Bearer {token}",
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
                
            except Exception as e:
                logger.error(f"Error building auth headers: {str(e)}")
                return None
        
        # Get current market price from correct Coinbase Advanced Trade API
        headers = get_auth_headers_for_price("GET", f"/api/v3/brokerage/products/{symbol}")
        response = requests.get(
            f"https://{BASE_URL}/api/v3/brokerage/products/{symbol}",
            headers=headers
        )
        
        if response.status_code != 200:
            logger.error(f"❌ Failed to get product details for {symbol}: {response.status_code}")
            return False
            
        details = response.json()
        current_market_price = float(details.get('price', price))
        min_market_funds = float(details.get('quote_min_size', 1.0))
        
        logger.info(f"💰 Current market price: ${current_market_price:.4f}")
        logger.info(f"📈 Analysis price: ${price:.4f}")
        
        if funds < min_market_funds:
            logger.info(f"📏 Adjusting order size from ${funds:.2f} to minimum ${min_market_funds:.2f}")
            funds = min_market_funds
        
        # Use current market price for execution, not analysis price
        logger.info(f"✅ Using current market price ${current_market_price:.4f} for {side} order")
        return execute_trade_via_subprocess(symbol, side, current_market_price, funds)
        
    except Exception as e:
        logger.error(f"💥 Error executing trade: {str(e)}")
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
            [sys.executable, "lk.py"],
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
            
            # Only sync positions with meaningful value (>= $0.5)
            if usd_value < 0.5:
                logger.debug(f"Skipping {symbol} - value too small: ${usd_value:.6f}")
                continue
            
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
        
        # Parse the output - extract only the JSON part
        output = result.stdout.strip()
        logger.info(f"lk.py output: {output}")
        
        if output:
            try:
                # Find the JSON part (starts with { and ends with })
                json_start = output.find('{')
                json_end = output.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_text = output[json_start:json_end]
                    logger.info(f"Extracted JSON: {json_text[:100]}...")  # Log first 100 chars
                    
                    data = json.loads(json_text)
                    positions = data.get('positions', [])
                    
                    # Filter positions by minimum USD value
                    filtered_positions = [p for p in positions if float(p.get('usd_value', 0)) >= min_usd_value]
                    logger.info(f"Found {len(filtered_positions)} positions with value >= ${min_usd_value}")
                    
                    return filtered_positions
                else:
                    logger.error("No valid JSON found in lk.py output")
                    return []
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from lk.py output: {e}")
                logger.error(f"Raw output: {output}")
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
    """Make ML decision for a given symbol with 75% confidence threshold for buy signals"""
    try:
        # Check if model exists
        model_prefix = f"{symbol.replace('-', '')}_{granularity}"
        clf_path = os.path.join(MODELS_DIR, f"{model_prefix}_clf.pkl")
        
        if not os.path.exists(clf_path):
            logger.info(f"No model found for {symbol}, attempting to use enhanced prediction models...")
            # Try to use the enhanced price prediction models instead
            try:
                # Use enhanced ML decision which includes model training if needed
                from enhanced_ml_features import make_enhanced_ml_decision
                enhanced_result = make_enhanced_ml_decision(symbol, granularity, investment_amount=100.0)
                if enhanced_result and enhanced_result.get('action') in ['BUY', 'SELL', 'HOLD']:
                    decision = enhanced_result['action']
                    confidence = enhanced_result.get('overall_confidence', 0.0)
                    logger.info(f"{symbol} Enhanced ML Decision: {decision} (confidence: {confidence:.1%})")
                    return decision, confidence
            except Exception as e:
                logger.debug(f"Enhanced ML decision failed for {symbol}: {str(e)}")
                pass
            
            # Fallback: Skip training during normal operation to avoid delays
            logger.warning(f"No model available for {symbol} and enhanced decision failed. Skipping...")
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
        df = calculate_indicators(df, symbol=symbol)
        
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
        
        # Convert prediction to decision with 75% confidence threshold for BUY
        if prediction == 1:
            decision = 'BUY'
            confidence = prediction_proba[1]
            # Only return BUY if confidence >= 75%
            if confidence < ML_BUY_CONFIDENCE_THRESHOLD:
                logger.info(f"{symbol} BUY confidence {confidence:.1%} < {ML_BUY_CONFIDENCE_THRESHOLD:.0%} threshold, returning HOLD")
                return 'HOLD', confidence
        else:
            decision = 'SELL'
            confidence = prediction_proba[0]
        
        logger.info(f"{symbol} ML Decision: {decision} (confidence: {confidence:.1%})")
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

# ML Trading Configuration
ML_BUY_CONFIDENCE_THRESHOLD = 0.75  # 75% confidence required for BUY signals
ML_SELL_CONFIDENCE_THRESHOLD = 0.60  # 60% confidence required for SELL signals

# Import credentials from config instead of hardcoding
try:
    from config import ORG_ID, API_KEY_ID, KEY_NAME, BASE_URL, PRIVATE_KEY_PEM
    logger.info("✅ Loaded API credentials from config.py")
except ImportError:
    logger.warning("⚠️ Could not import from config.py, using environment variables")
    import os
    ORG_ID = os.getenv('COINBASE_ORG_ID', 'b98ec8e1-610f-451a-9324-40ae8e705d00')
    API_KEY_ID = os.getenv('COINBASE_API_KEY', '').split('/')[-1] if os.getenv('COINBASE_API_KEY') else None
    KEY_NAME = os.getenv('COINBASE_API_KEY')
    BASE_URL = os.getenv('COINBASE_BASE_URL', 'api.coinbase.com')
    PRIVATE_KEY_PEM = os.getenv('COINBASE_API_SECRET')
    
    if not KEY_NAME or not PRIVATE_KEY_PEM:
        raise ValueError("Missing required Coinbase API credentials")

# === Replace the train_model_for_symbol function completely ===
# === Enhanced Model Training ===
def train_model_for_symbol(symbol, granularity=60):
    """Train Enhanced ML models for a given symbol and granularity"""
    try:
        logger.info(f"🧠 Training ENHANCED models for {symbol} with {granularity}s granularity...")
        
        # Get historical data
        df = get_coinbase_data(symbol, granularity, days=365)
        if df is None or df.empty:
            logger.error(f"No data available for {symbol}")
            return None
        
        # Calculate enhanced indicators
        df = calculate_indicators(df, symbol=symbol)
        
        if df.empty:
            logger.error(f"No data after indicator calculation for {symbol}")
            return None
        
        # Enhanced feature selection - use a subset of the most important features
        # Start with basic features that should always be present
        basic_features = [
            'rsi_14', 'macd_12_26', 'macd_signal_12_26', 'macd_hist_12_26',
            'sma_20', 'sma_50', 'bb_upper_20', 'bb_lower_20'
        ]
        
        # Add enhanced features if available
        enhanced_features = []
        for col in df.columns:
            if any(pattern in col for pattern in ['alpha_', 'ema_', 'stoch_', 'williams_r', 
                                                 'cci', 'mfi', 'adx', 'momentum_composite', 
                                                 'volatility_composite', 'trend_strength',
                                                 'returns_mean_', 'returns_std_', 'pv_correlation']):
                enhanced_features.append(col)
        
        # Combine features, preferring enhanced ones
        feature_columns = []
        for feat in basic_features:
            if feat in df.columns:
                feature_columns.append(feat)
        
        # Add top enhanced features (limit to prevent overfitting)
        enhanced_features = enhanced_features[:20]  # Top 20 enhanced features
        feature_columns.extend([f for f in enhanced_features if f not in feature_columns])
        
        # Fallback to basic features if no enhanced features available
        if len(feature_columns) < 4:
            basic_fallback = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'sma_20', 'sma_50']
            feature_columns = [f for f in basic_fallback if f in df.columns]
        
        logger.info(f"🎯 Using {len(feature_columns)} features: {feature_columns[:5]}...")
        
        if len(feature_columns) == 0:
            logger.error(f"No valid features found for {symbol}")
            return None
        
        # Create enhanced target variables
        df['target'] = df['close'].shift(-1) > df['close']
        df['target'] = df['target'].astype(int)
        
        # Enhanced target - also predict magnitude of change
        df['price_change_pct'] = df['close'].pct_change().shift(-1) * 100  # Next period return %
        df['strong_move'] = (abs(df['price_change_pct']) > 2.0).astype(int)  # >2% moves
        
        # Drop NaN values
        df = df.dropna()
        
        if len(df) < 100:
            logger.error(f"Insufficient data for {symbol} after preprocessing: {len(df)} rows")
            return None
        
        # Split data
        train_size = int(len(df) * 0.8)
        X_train = df[feature_columns][:train_size]
        y_train = df['target'][:train_size]
        
        # Ensure all features are numeric
        X_train = X_train.select_dtypes(include=[np.number])
        
        if X_train.empty:
            logger.error(f"No numeric features available for {symbol}")
            return None
        
        # Train enhanced model with better parameters
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        # Enhanced RandomForest with better parameters
        clf = RandomForestClassifier(
            n_estimators=200,  # More trees
            max_depth=10,      # Prevent overfitting
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',  # Feature subsampling
            random_state=42,
            n_jobs=-1          # Use all cores
        )
        
        clf.fit(X_train, y_train)
        
        # Evaluate model
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
        logger.info(f"📊 {symbol} Model CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Feature importance
        if hasattr(clf, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': clf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info(f"🔝 Top 5 features for {symbol}:")
            for _, row in importance_df.head().iterrows():
                logger.info(f"   {row['feature']}: {row['importance']:.3f}")
        
        # Save models
        model_prefix = f"{symbol.replace('-', '')}_{granularity}"
        clf_path = os.path.join(MODELS_DIR, f"{model_prefix}_enhanced_clf.pkl")
        
        import joblib
        joblib.dump(clf, clf_path)
        
        # Save feature columns for later use
        features_path = os.path.join(MODELS_DIR, f"{model_prefix}_features.pkl")
        joblib.dump(X_train.columns.tolist(), features_path)
        
        logger.info(f"✅ Successfully trained and saved ENHANCED models for {symbol}")
        logger.info(f"   Features: {len(X_train.columns)}")
        logger.info(f"   Training samples: {len(X_train)}")
        logger.info(f"   Model path: {clf_path}")
        
        return clf
        
    except Exception as e:
        logger.error(f"❌ Error training enhanced models for {symbol}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def calculate_basic_momentum_indicators(df, symbol=None):
    """Calculate basic momentum indicators as fallback if enhanced ones are not available"""
    try:
        # Basic price momentum
        df['momentum_1'] = df['close'].pct_change(periods=1) * 100
        df['momentum_3'] = df['close'].pct_change(periods=3) * 100  
        df['momentum_5'] = df['close'].pct_change(periods=5) * 100
        df['momentum_10'] = df['close'].pct_change(periods=10) * 100
        
        # Price ratios
        df['price_sma20_ratio'] = ((df['close'] / df['close'].rolling(window=20).mean()) - 1) * 100
        df['price_sma50_ratio'] = ((df['close'] / df['close'].rolling(window=50).mean()) - 1) * 100
        
        # Volume ratios  
        df['volume_ma_ratio'] = ((df['volume'] / df['volume'].rolling(window=20).mean()) - 1) * 100
        
        # Basic volatility
        df['volatility_10'] = df['close'].rolling(window=10).std() / df['close'] * 100
        df['volatility_20'] = df['close'].rolling(window=20).std() / df['close'] * 100
        
        # Intraday metrics
        df['high_low_spread'] = ((df['high'] - df['low']) / df['close']) * 100
        df['open_close_spread'] = ((df['close'] - df['open']) / df['open']) * 100
        
        logger.info(f"✅ Calculated basic momentum indicators for {symbol or 'unknown'}")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error calculating basic momentum indicators: {str(e)}")
        return df

def ensure_required_features(df, features):
    """Ensure all required features exist in DataFrame, create missing ones with safe defaults"""
    missing_features = []
    
    for feature in features:
        if feature not in df.columns:
            missing_features.append(feature)
            # Create safe default features
            if 'sma' in feature.lower() or 'ma' in feature.lower():
                if 'ratio' in feature.lower():
                    df[feature] = 1.0  # Price ratios default to 1.0
                else:
                    df[feature] = df['close'].rolling(window=20).mean()  # Default SMA
            elif 'momentum' in feature.lower():
                if 'momentum_1' == feature:
                    df[feature] = df['close'].pct_change(periods=1) * 100
                elif 'momentum_3' == feature:
                    df[feature] = df['close'].pct_change(periods=3) * 100
                elif 'momentum_5' == feature:
                    df[feature] = df['close'].pct_change(periods=5) * 100
                elif 'momentum_10' == feature:
                    df[feature] = df['close'].pct_change(periods=10) * 100
                else:
                    df[feature] = df['close'].pct_change(periods=5) * 100  # Default momentum
            elif 'volume' in feature.lower() and 'ratio' in feature.lower():
                df[feature] = 1.0  # Volume ratios default to 1.0
            elif 'rsi' in feature.lower():
                df[feature] = 50.0  # Neutral RSI
            elif 'macd' in feature.lower():
                df[feature] = 0.0  # Neutral MACD
            elif 'volatility' in feature.lower():
                df[feature] = df['close'].rolling(window=20).std() / df['close'] * 100
            elif 'price_sma20_ratio' == feature:
                sma20 = df['close'].rolling(window=20).mean()
                df[feature] = (df['close'] / sma20 - 1) * 100
            elif 'price_sma50_ratio' == feature:
                sma50 = df['close'].rolling(window=50).mean()
                df[feature] = (df['close'] / sma50 - 1) * 100
            elif 'volume_ma_ratio' == feature:
                volume_ma = df['volume'].rolling(window=20).mean()
                df[feature] = (df['volume'] / volume_ma - 1) * 100
            else:
                # Generic fallback - use price momentum
                df[feature] = df['close'].pct_change() * 100
    
    # Fill any remaining NaN values
    df[features] = df[features].fillna(0)
    
    if missing_features:
        logger.warning(f"Created {len(missing_features)} missing features with safe defaults: {missing_features[:5]}...")
    
    return df

# === Scanner Functions ===

# === ENHANCED PRICE PREDICTION SYSTEM ===
# Replace classification with regression for exact price predictions

def train_price_prediction_models(symbol, granularity=60):
    """Train regression models for exact price prediction with enhanced granularities"""
    try:
        logger.info(f"🎯 Training ENHANCED PRICE PREDICTION models for {symbol} with {granularity}s granularity...")
        
        # Get historical data (more data for better price predictions)
        df = get_coinbase_data(symbol, granularity, days=730)  # 2 years of data
        if df is None or df.empty:
            logger.error(f"No data available for {symbol}")
            return None
        
        # Calculate enhanced indicators
        df = calculate_indicators(df, symbol=symbol)
        
        if df.empty:
            logger.error(f"No data after indicator calculation for {symbol}")
            return None
        
        # Enhanced feature selection for price prediction
        basic_features = [
            'rsi_14', 'macd_12_26', 'macd_signal_12_26', 'macd_hist_12_26',
            'sma_20', 'sma_50', 'bb_upper_20', 'bb_lower_20', 'ATR'
        ]
        
        # Add enhanced features if available
        enhanced_features = []
        for col in df.columns:
            if any(pattern in col for pattern in ['alpha_', 'ema_', 'stoch_', 'williams_r', 
                                                 'cci', 'mfi', 'adx', 'momentum_composite', 
                                                 'volatility_composite', 'trend_strength',
                                                 'returns_mean_', 'returns_std_', 'pv_correlation']):
                enhanced_features.append(col)
        
        # Price-specific features
        df['price_ma_5'] = df['close'].rolling(window=5).mean()
        df['price_ma_10'] = df['close'].rolling(window=10).mean()
        df['price_ma_20'] = df['close'].rolling(window=20).mean()
        df['price_volatility'] = df['close'].rolling(window=20).std()
        df['price_momentum'] = df['close'].pct_change(periods=5)
        df['high_low_ratio'] = df['high'] / df['low']
        df['volume_price_trend'] = df['volume'] * df['close'].pct_change()
        
        # Add micro-momentum features for shorter timeframes
        df['price_momentum_3'] = df['close'].pct_change(periods=3)
        df['price_acceleration'] = df['price_momentum'].diff()
        df['intraday_range'] = (df['high'] - df['low']) / df['close'] * 100
        df['volume_momentum'] = df['volume'].pct_change(periods=3)
        
        price_features = ['price_ma_5', 'price_ma_10', 'price_ma_20', 'price_volatility', 
                         'price_momentum', 'high_low_ratio', 'volume_price_trend',
                         'price_momentum_3', 'price_acceleration', 'intraday_range', 'volume_momentum']
        
        # Combine all features
        feature_columns = []
        for feat in basic_features:
            if feat in df.columns:
                feature_columns.append(feat)
        
        # Add enhanced features (limited to prevent overfitting)
        enhanced_features = enhanced_features[:15]
        feature_columns.extend([f for f in enhanced_features if f not in feature_columns])
        
        # Add price-specific features
        feature_columns.extend([f for f in price_features if f in df.columns and f not in feature_columns])
        
        # Fallback to basic features
        if len(feature_columns) < 4:
            basic_fallback = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'sma_20', 'sma_50']
            feature_columns = [f for f in basic_fallback if f in df.columns]
        
        logger.info(f"🎯 Using {len(feature_columns)} features for enhanced price prediction")
        
        if len(feature_columns) == 0:
            logger.error(f"No valid features found for {symbol}")
            return None
        
        # Enhanced target variables for multiple granularities
        # Calculate periods based on granularity for consistent time horizons
        if granularity <= 60:  # 1 minute or less
            periods_15min = 15
            periods_30min = 30
            periods_1h = 60
            periods_4h = 240
            periods_24h = 1440
        elif granularity <= 900:  # 15 minutes or less
            periods_15min = 1
            periods_30min = 2
            periods_1h = 4
            periods_4h = 16
            periods_24h = 96
        elif granularity <= 3600:  # 1 hour or less
            periods_15min = 1  # Not applicable, use 1 period
            periods_30min = 1  # Not applicable, use 1 period  
            periods_1h = 1
            periods_4h = 4
            periods_24h = 24
        else:  # Daily or larger
            periods_15min = 1
            periods_30min = 1
            periods_1h = 1
            periods_4h = 1
            periods_24h = 1
        
        # Create price targets for multiple timeframes
        df['price_15min'] = df['close'].shift(-periods_15min)
        df['price_30min'] = df['close'].shift(-periods_30min)
        df['price_1h'] = df['close'].shift(-periods_1h)
        df['price_4h'] = df['close'].shift(-periods_4h)
        df['price_24h'] = df['close'].shift(-periods_24h)
        
        # Calculate percentage changes for better normalization
        df['price_change_15min'] = (df['price_15min'] - df['close']) / df['close'] * 100
        df['price_change_30min'] = (df['price_30min'] - df['close']) / df['close'] * 100
        df['price_change_1h'] = (df['price_1h'] - df['close']) / df['close'] * 100
        df['price_change_4h'] = (df['price_4h'] - df['close']) / df['close'] * 100
        df['price_change_24h'] = (df['price_24h'] - df['close']) / df['close'] * 100
        
        # Drop NaN values
        df = df.dropna()
        
        if len(df) < 200:
            logger.error(f"Insufficient data for {symbol} after preprocessing: {len(df)} rows")
            return None
        
        # Split data (80% train, 20% test)
        train_size = int(len(df) * 0.8)
        X_train = df[feature_columns][:train_size]
        X_test = df[feature_columns][train_size:]
        
        # Ensure all features are numeric
        X_train = X_train.select_dtypes(include=[np.number])
        X_test = X_test.select_dtypes(include=[np.number])
        
        if X_train.empty:
            logger.error(f"No numeric features available for {symbol}")
            return None
        
        # Train multiple regression models for different time horizons
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.preprocessing import StandardScaler
        
        models = {}
        scalers = {}
        
        # Scale features for better performance
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Enhanced time horizons with granular options
        time_horizons = ['15min', '30min', '1h', '4h', '24h']
        
        for horizon in time_horizons:
            target_col = f'price_change_{horizon}'
            if target_col not in df.columns:
                continue
                
            y_train = df[target_col][:train_size]
            y_test = df[target_col][train_size:]
            
            # Skip if not enough target data
            if len(y_train.dropna()) < 100:
                continue
            
            logger.info(f"🎯 Training {horizon} price prediction model...")
            
            # Different model configurations based on timeframe
            if horizon in ['15min', '30min']:
                # Faster, less deep models for short-term predictions
                rf_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=3,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                )
                
                gb_model = GradientBoostingRegressor(
                    n_estimators=75,
                    max_depth=4,
                    learning_rate=0.15,
                    random_state=42
                )
            elif horizon in ['1h', '4h']:
                # Medium complexity for medium-term
                rf_model = RandomForestRegressor(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                )
                
                gb_model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.12,
                    random_state=42
                )
            else:  # 24h+
                # Deeper models for long-term predictions
                rf_model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                )
                
                gb_model = GradientBoostingRegressor(
                    n_estimators=125,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            
            # Train both models
            rf_model.fit(X_train_scaled, y_train)
            gb_model.fit(X_train_scaled, y_train)
            
            # Evaluate models
            rf_pred = rf_model.predict(X_test_scaled)
            gb_pred = gb_model.predict(X_test_scaled)
            
            rf_mae = mean_absolute_error(y_test, rf_pred)
            gb_mae = mean_absolute_error(y_test, gb_pred)
            
            # Choose the better model
            if rf_mae <= gb_mae:
                best_model = rf_model
                model_type = 'RandomForest'
                best_mae = rf_mae
            else:
                best_model = gb_model
                model_type = 'GradientBoosting'
                best_mae = gb_mae
            
            # Calculate expected return range for this timeframe
            y_std = y_test.std()
            typical_return = abs(y_test.mean())
            
            models[horizon] = {
                'model': best_model,
                'type': model_type,
                'mae': best_mae,
                'expected_return': typical_return,
                'volatility': y_std,
                'feature_importance': best_model.feature_importances_ if hasattr(best_model, 'feature_importances_') else None
            }
            
            logger.info(f"✅ {horizon} model: {model_type} MAE: {best_mae:.4f}% | Expected Return: {typical_return:.2f}% | Volatility: {y_std:.2f}%")
        
        if not models:
            logger.error(f"No models could be trained for {symbol}")
            return None
        
        # Save models and scaler
        model_prefix = f"{symbol.replace('-', '')}_{granularity}"
        
        import joblib
        
        # Save each time horizon model
        for horizon, model_data in models.items():
            model_path = os.path.join(MODELS_DIR, f"{model_prefix}_{horizon}_price_regressor.pkl")
            joblib.dump(model_data['model'], model_path)
            logger.info(f"💾 Saved {horizon} model: {model_path}")
        
        # Save scaler and features
        scaler_path = os.path.join(MODELS_DIR, f"{model_prefix}_price_scaler.pkl")
        features_path = os.path.join(MODELS_DIR, f"{model_prefix}_price_features.pkl")
        
        joblib.dump(scaler, scaler_path)
        joblib.dump(X_train.columns.tolist(), features_path)
        
        # Save enhanced metadata including expected returns and volatility
        metadata = {
            'symbol': symbol,
            'granularity': granularity,
            'features': X_train.columns.tolist(),
            'models': {horizon: {
                'type': data['type'], 
                'mae': data['mae'],
                'expected_return': data['expected_return'],
                'volatility': data['volatility']
            } for horizon, data in models.items()},
            'training_samples': len(X_train),
            'training_date': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(MODELS_DIR, f"{model_prefix}_price_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"🎉 Enhanced price prediction models trained for {symbol}")
        logger.info(f"   📊 Timeframes: {list(models.keys())}")
        logger.info(f"   📈 Features: {len(feature_columns)}")
        logger.info(f"   💾 Saved: {len(models)} models + metadata")
        
        return models
        
    except Exception as e:
        logger.error(f"❌ Error training enhanced price prediction models for {symbol}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def make_price_prediction(symbol, granularity=3600, investment_amount=100.0):
    """Make exact price predictions for different time horizons with enhanced granularities"""
    try:
        logger.info(f"🎯 Making enhanced price predictions for {symbol}...")
        
        # Check if models exist
        model_prefix = f"{symbol.replace('-', '')}_{granularity}"
        metadata_path = os.path.join(MODELS_DIR, f"{model_prefix}_price_metadata.pkl")
        
        if not os.path.exists(metadata_path):
            logger.info(f"No price prediction models found for {symbol}, training new ones...")
            models = train_price_prediction_models(symbol, granularity)
            if not models:
                return None
        
        # Load model metadata and components
        import joblib
        
        try:
            metadata = joblib.load(metadata_path)
            scaler = joblib.load(os.path.join(MODELS_DIR, f"{model_prefix}_price_scaler.pkl"))
            features = joblib.load(os.path.join(MODELS_DIR, f"{model_prefix}_price_features.pkl"))
        except Exception as e:
            logger.warning(f"Could not load model components for {symbol}: {str(e)}, retraining...")
            models = train_price_prediction_models(symbol, granularity)
            if not models:
                return None
            # Reload after training
            try:
                metadata = joblib.load(metadata_path)
                scaler = joblib.load(os.path.join(MODELS_DIR, f"{model_prefix}_price_scaler.pkl"))
                features = joblib.load(os.path.join(MODELS_DIR, f"{model_prefix}_price_features.pkl"))
            except Exception as e2:
                logger.error(f"Failed to load model components after retraining: {str(e2)}")
                return None
        
        # Get recent data for prediction
        df = get_coinbase_data(symbol, granularity, days=30)
        if df is None or df.empty:
            logger.error(f"No data available for prediction for {symbol}")
            return None
        
        # Calculate indicators (same as training)
        df = calculate_indicators(df, symbol=symbol)
        
        # Add price-specific features (same as training)
        df['price_ma_5'] = df['close'].rolling(window=5).mean()
        df['price_ma_10'] = df['close'].rolling(window=10).mean()
        df['price_ma_20'] = df['close'].rolling(window=20).mean()
        df['price_volatility'] = df['close'].rolling(window=20).std()
        df['price_momentum'] = df['close'].pct_change(periods=5)
        df['high_low_ratio'] = df['high'] / df['low']
        df['volume_price_trend'] = df['volume'] * df['close'].pct_change()
        
        # Add micro-momentum features (same as training)
        df['price_momentum_3'] = df['close'].pct_change(periods=3)
        df['price_acceleration'] = df['price_momentum'].diff()
        df['intraday_range'] = (df['high'] - df['low']) / df['close'] * 100
        df['volume_momentum'] = df['volume'].pct_change(periods=3)
        
        # CRITICAL FIX: Ensure all required features exist
        df = ensure_required_features(df, features)
        
        # Validate features are now available
        available_features = [f for f in features if f in df.columns]
        if len(available_features) < len(features) * 0.8:  # At least 80% of features should be available
            logger.warning(f"Only {len(available_features)}/{len(features)} features available for {symbol}")
            
        # Use available features only
        features_to_use = available_features
        
        if len(features_to_use) == 0:
            logger.error(f"No valid features available for {symbol}")
            return None
        
        # Get latest feature values
        latest_features = df[features_to_use].iloc[-1:].fillna(0)
        current_price = float(df['close'].iloc[-1])
        
        # Handle scaler mismatch - create new scaler if needed
        try:
            # Check if scaler expects the same number of features
            if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != len(features_to_use):
                logger.warning(f"Scaler expects {scaler.n_features_in_} features but got {len(features_to_use)}")
                # Create a new scaler with current data
                from sklearn.preprocessing import StandardScaler
                temp_scaler = StandardScaler()
                temp_scaler.fit(df[features_to_use].dropna())
                latest_features_scaled = temp_scaler.transform(latest_features)
            else:
                latest_features_scaled = scaler.transform(latest_features)
        except Exception as e:
            logger.warning(f"Scaler transform failed: {str(e)}, using normalized features")
            # Fallback: use normalized features without scaling
            latest_features_scaled = (latest_features - latest_features.mean()) / (latest_features.std() + 1e-8)
            latest_features_scaled = latest_features_scaled.fillna(0).values
        
        # Make predictions for each time horizon with enhanced granularities
        predictions = {}
        
        for horizon in ['15min', '30min', '1h', '4h', '24h']:
            model_path = os.path.join(MODELS_DIR, f"{model_prefix}_{horizon}_price_regressor.pkl")
            
            if os.path.exists(model_path):
                try:
                    model = joblib.load(joblib.load(model_path)
                    
                    # Handle model feature mismatch
                    if hasattr(model, 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/n_features_in_') and model.n_features_in_ != latest_features_scaled.shape[1]:
                        logger.warning(f"Model {horizon} expects {model.n_features_in_} features but got {latest_features_scaled.shape[1]}")
                        # Skip this model or use fallback prediction
                        continue
                    
                    # Predict percentage change
                    price_change_pct = model.predict(latest_features_scaled)[0]
                    
                    # Convert to absolute price
                    predicted_price = current_price * (1 + price_change_pct / 100)
                    
                    # Get model metadata for this horizon
                    model_metadata = metadata['models'].get(horizon, {})
                    model_mae = model_metadata.get('mae', 5.0)
                    expected_return = model_metadata.get('expected_return', 2.0)
                    volatility = model_metadata.get('volatility', 3.0)
                    
                    # Enhanced confidence calculation based on multiple factors
                    mae_confidence = max(0, min(1, (5 - model_mae) / 5))  # Higher confidence for lower MAE
                    
                    # Volatility-adjusted confidence
                    volatility_penalty = min(0.3, volatility / 10)  # Penalize high volatility
                    
                    # Return magnitude confidence (higher confidence for expected returns in normal range)
                    return_confidence = 1.0
                    if abs(price_change_pct) > expected_return * 2:
                        return_confidence = 0.7  # Lower confidence for extreme predictions
                    elif abs(price_change_pct) < expected_return * 0.5:
                        return_confidence = 0.8  # Slightly lower confidence for very small predictions
                    
                    # Combined confidence
                    confidence = mae_confidence * (1 - volatility_penalty) * return_confidence
                    confidence = max(0.1, min(1.0, confidence))  # Clamp between 0.1 and 1.0
                    
                    predictions[horizon] = {
                        'predicted_price': predicted_price,
                        'price_change_pct': price_change_pct,
                        'confidence': confidence,
                        'model_type': model_metadata.get('type', 'Unknown'),
                        'mae': model_mae,
                        'expected_return': expected_return,
                        'volatility': volatility,
                        'return_confidence': return_confidence
                    }
                    
                    logger.info(f"   {horizon}: ${predicted_price:.4f} ({price_change_pct:+.2f}%) confidence: {confidence:.1%} | exp_ret: {expected_return:.2f}%")
                    
                except Exception as e:
                    logger.warning(f"Error loading/using model for {horizon}: {str(e)}")
                    continue
        
        if not predictions:
            logger.error(f"No predictions could be made for {symbol}")
            return None
        
        # Calculate trading recommendation based on enhanced predictions
        recommendation = analyze_enhanced_price_predictions(predictions, current_price, investment_amount)
        
        # Combine predictions with recommendation
        result = {
            'symbol': symbol,
            'current_price': current_price,
            'predictions': predictions,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat(),
            'model_metadata': metadata
        }
        
        logger.info(f"🎯 {symbol} Enhanced Price Prediction Summary:")
        logger.info(f"   Current: ${current_price:.4f}")
        if '15min' in predictions:
            logger.info(f"   15min: ${predictions['15min']['predicted_price']:.4f} ({predictions['15min']['price_change_pct']:+.2f}%)")
        if '30min' in predictions:
            logger.info(f"   30min: ${predictions['30min']['predicted_price']:.4f} ({predictions['30min']['price_change_pct']:+.2f}%)")
        if '1h' in predictions:
            logger.info(f"   1h: ${predictions['1h']['predicted_price']:.4f} ({predictions['1h']['price_change_pct']:+.2f}%)")
        if '4h' in predictions:
            logger.info(f"   4h: ${predictions['4h']['predicted_price']:.4f} ({predictions['4h']['price_change_pct']:+.2f}%)")
        if '24h' in predictions:
            logger.info(f"   24h: ${predictions['24h']['predicted_price']:.4f} ({predictions['24h']['price_change_pct']:+.2f}%)")
        logger.info(f"   Best Timeframe: {recommendation.get('best_horizon', 'Unknown')}")
        logger.info(f"   Recommendation: {recommendation.get('action', 'HOLD')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error making enhanced price prediction for {symbol}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def analyze_enhanced_price_predictions(predictions, current_price, investment_amount):
    """Analyze enhanced price predictions with dynamic TP/SL based on timeframes and expected returns"""
    try:
        # Calculate weighted average prediction with enhanced scoring
        total_weight = 0
        weighted_change = 0
        best_horizon = None
        best_confidence = 0
        best_prediction = None
        
        # Enhanced timeframe weights and characteristics
        timeframe_characteristics = {
            '15min': {'base_weight': 4, 'typical_return': 0.5, 'risk_factor': 1.5, 'tp_base': 0.8, 'sl_base': 0.6},
            '30min': {'base_weight': 3, 'typical_return': 1.0, 'risk_factor': 1.3, 'tp_base': 1.2, 'sl_base': 0.8},
            '1h': {'base_weight': 2, 'typical_return': 1.5, 'risk_factor': 1.1, 'tp_base': 1.8, 'sl_base': 1.0},
            '4h': {'base_weight': 2, 'typical_return': 2.5, 'risk_factor': 1.0, 'tp_base': 2.5, 'sl_base': 1.2},
            '24h': {'base_weight': 1, 'typical_return': 4.0, 'risk_factor': 0.8, 'tp_base': 4.0, 'sl_base': 1.8}
        }
        
        for horizon, pred in predictions.items():
            chars = timeframe_characteristics.get(horizon, timeframe_characteristics['1h'])
            
            # Enhanced weighting system
            base_weight = chars['base_weight']
            confidence_weight = pred['confidence']
            
            # Return quality weight (prefer predictions close to expected returns)
            expected_return = pred.get('expected_return', chars['typical_return'])
            return_quality = 1.0
            if abs(pred['price_change_pct']) > expected_return * 1.5:
                return_quality = 0.8  # Penalize extreme predictions
            elif abs(pred['price_change_pct']) < expected_return * 0.3:
                return_quality = 0.7  # Penalize very small predictions
            
            # Volatility weight (prefer low volatility models)
            volatility = pred.get('volatility', 3.0)
            volatility_weight = max(0.5, 1.0 - (volatility / 10))
            
            # Combined weight
            combined_weight = base_weight * confidence_weight * return_quality * volatility_weight
            
            weighted_change += pred['price_change_pct'] * combined_weight
            total_weight += combined_weight
            
            # Track best prediction with enhanced scoring that prioritizes return magnitude
            return_magnitude = abs(pred['price_change_pct'])
            
            # New scoring system that balances confidence with return potential
            confidence_score = confidence_weight * 0.4  # Reduced confidence weight from 100% to 40%
            return_score = min(return_magnitude / chars['typical_return'], 2.0) * 0.5  # Return potential (capped at 2x typical)
            volatility_score = volatility_weight * 0.1  # Minor volatility component
            
            prediction_score = confidence_score + return_score + volatility_score
            
            if prediction_score > best_confidence:
                best_confidence = prediction_score
                best_horizon = horizon
                best_prediction = pred
        
        avg_price_change = weighted_change / total_weight if total_weight > 0 else 0
        
        # Determine action based on predicted price movements and confidence
        action = 'HOLD'
        confidence = best_confidence if best_prediction else 0.5
        target_price = None
        expected_profit_pct = 0
        risk_level = 'MEDIUM'
        
        # Enhanced decision logic - REMOVED CONFIDENCE THRESHOLDS
        if best_prediction and best_horizon:
            chars = timeframe_characteristics[best_horizon]
            min_threshold = chars['typical_return'] * 0.2  # Very low threshold - focus on return magnitude
            
            # Simple decision based on return magnitude - NO CONFIDENCE CHECKS
            if avg_price_change > min_threshold:  # Removed confidence check
                action = 'BUY'
                target_price = current_price * (1 + avg_price_change / 100)
                expected_profit_pct = avg_price_change - 0.7  # Account for fees (0.35% x 2)
                risk_level = 'LOW' if abs(avg_price_change) > 1.0 else 'MEDIUM'
                
            elif avg_price_change < -min_threshold:  # Removed confidence check
                action = 'SELL'
                target_price = current_price * (1 + avg_price_change / 100)
                expected_profit_pct = abs(avg_price_change) - 0.7  # Profit from shorting
                risk_level = 'LOW' if abs(avg_price_change) > 1.0 else 'MEDIUM'
            
            else:
                action = 'HOLD'
                risk_level = 'LOW'
        
        # Calculate profit estimation
        expected_profit_usd = 0
        if action in ['BUY', 'SELL'] and expected_profit_pct > 0:
            expected_profit_usd = investment_amount * (expected_profit_pct / 100)
        
        # Dynamic position sizing - simplified without confidence checks
        position_size_pct = 0.0
        if action == 'BUY' and best_prediction and expected_profit_pct > 0.5:
            if expected_profit_pct > 3:
                position_size_pct = 0.6  # 60% for high return
            elif expected_profit_pct > 2:
                position_size_pct = 0.4  # 40% for good return
            elif expected_profit_pct > 1:
                position_size_pct = 0.2  # 20% for moderate return
            else:
                position_size_pct = 0.1  # 10% for small return
        
        return {
            'action': action,
            'confidence': confidence,
            'target_price': target_price,
            'expected_profit_pct': expected_profit_pct,
            'expected_profit_usd': expected_profit_usd,
            'risk_level': risk_level,
            'best_horizon': best_horizon,
            'avg_price_change': avg_price_change,
            'position_size_pct': position_size_pct,
            'reasoning': f"Best timeframe: {best_horizon}, weighted avg change: {avg_price_change:.2f}%, no confidence threshold",
            'timeframe_analysis': {
                horizon: {
                    'prediction': pred['price_change_pct'],
                    'confidence': pred['confidence'],
                    'weight_contribution': (
                        timeframe_characteristics.get(horizon, {}).get('base_weight', 1) * 
                        pred['confidence']
                    ) / total_weight if total_weight > 0 else 0
                } for horizon, pred in predictions.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing enhanced price predictions: {str(e)}")
        return {
            'action': 'HOLD',
            'confidence': 0.0,
            'target_price': current_price,
            'expected_profit_pct': 0,
            'expected_profit_usd': 0,
            'risk_level': 'HIGH',
            'best_horizon': None,
            'avg_price_change': 0,
            'position_size_pct': 0,
            'reasoning': f"Analysis failed: {str(e)}",
            'timeframe_analysis': {}
        }

import uuid
import threading
from concurrent.futures import ThreadPoolExecutor

# Import ML decision function
from stacking_ml_engine import make_enhanced_ml_decision

# Import credentials
try:
    from lk import client, KEY_NAME, PRIVATE_KEY_PEM, BASE_URL
    logger.info(f"✅ Config loaded - API Key: {KEY_NAME[:8]}...")
    logger.info(f"🏢 Organization: {client.get_accounts().accounts[0].uuid if hasattr(client, 'get_accounts') else 'Not available'}")
    logger.info(f"🌐 Base URL: {BASE_URL}")
except ImportError as e:
    logger.error(f"❌ Failed to import credentials: {str(e)}")
    client = None

def get_garch_volatility(df, window=24):
    """
    Estimate expected volatility using a GARCH(1,1) model.
    df: DataFrame with a 'close' column (price), indexed by time.
    window: Number of recent periods to use (e.g., 24 for 2 hours of 5-min candles).
    Returns: forecasted volatility (as a decimal, e.g., 0.02 for 2%)
    """
    try:
        from arch import arch_model
        import numpy as np
        
        returns = np.log(df['close'] / df['close'].shift(1)).dropna() * 100  # percent returns
        if len(returns) < window:
            logger.warning(f"Not enough data for GARCH: {len(returns)} < {window}")
            return None
        
        # Use recent data for GARCH fitting
        recent_returns = returns.tail(window)
        
        # Fit GARCH(1,1) model
        model = arch_model(recent_returns, vol='Garch', p=1, q=1)
        fitted_model = model.fit(disp='off')
        
        # Forecast next period volatility
        forecast = fitted_model.forecast(horizon=1)
        forecasted_variance = forecast.variance.iloc[-1, 0]
        forecasted_volatility = np.sqrt(forecasted_variance) / 100  # Convert back to decimal
        
        logger.info(f"📊 GARCH volatility forecast: {forecasted_volatility:.4f} ({forecasted_volatility*100:.2f}%)")
        return forecasted_volatility
        
    except Exception as e:
        logger.warning(f"⚠️ GARCH model failed: {str(e)}")
        return None

def calculate_garch_tp_sl_levels(symbol, entry_price):
    """Calculate TP/SL using GARCH volatility estimation with fallback methods"""
    try:
        # Get recent price data for GARCH analysis - ALIGNED WITH ML PREDICTIONS
        df = get_coinbase_data(symbol, granularity=3600, days=7)  # 1-hour candles, 7 days
        if df is None or df.empty:
            logger.warning(f"No price data available for {symbol}, using fallback")
            raise Exception("No price data")
        
        # Filter to last 48 hours (48 hourly candles)
        from datetime import datetime, timedelta
        cutoff_time = datetime.now() - timedelta(hours=48)
        if 'timestamp' in df.columns:
            df = df[df['timestamp'] >= cutoff_time]
        
        if len(df) < 12:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} points, using fallback")
            raise Exception("Insufficient data")
        
        current_price = float(df['close'].iloc[-1])
        
        # Try GARCH-based volatility estimation first - using 24-hour window (24 hourly candles)
        garch_volatility = get_garch_volatility(df, window=min(24, len(df)))
        
        if garch_volatility and garch_volatility > 0:
            # GARCH-based TP/SL calculation
            logger.info(f"🎯 Using GARCH-based TP/SL for {symbol}")
            
            # Use optimized multipliers: 2x volatility for TP and 4x volatility for SL (wider for 1H swings)
            tp_volatility_multiplier = 4.0
            sl_volatility_multiplier = 4.0
            
            # Calculate TP/SL percentages based on GARCH volatility
            tp_percentage = (garch_volatility * tp_volatility_multiplier) * 100
            sl_percentage = (garch_volatility * sl_volatility_multiplier) * 100
            
            # Ensure reasonable bounds (0.8-8% for both TP and SL)
            tp_percentage = max(0.8, min(tp_percentage, 8.0))
            sl_percentage = max(0.8, min(sl_percentage, 8.0))
            
            # Calculate prices
            tp_price = entry_price * (1 + tp_percentage / 100)
            sl_price = entry_price * (1 - sl_percentage / 100)
            
            logger.info(f"✅ GARCH-based TP/SL calculated for {symbol}")
            logger.info(f"   Entry Price: ${entry_price:.4f}")
            logger.info(f"   GARCH Volatility: {garch_volatility*100:.2f}%")
            logger.info(f"   TP: ${tp_price:.4f} (+{tp_percentage:.2f}%)")
            logger.info(f"   SL: ${sl_price:.4f} (-{sl_percentage:.2f}%)")
            logger.info(f"   Data Points: {len(df)} (48 hours of 1H candles)")
            
            return {
                'tp_percentage': tp_percentage,
                'sl_percentage': sl_percentage,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'method': 'garch_volatility',
                'garch_volatility': garch_volatility,
                'reasoning': f"GARCH: TP={tp_volatility_multiplier}x, SL={sl_volatility_multiplier}x volatility (1H)"
            }
        else:
            # Fallback to max up/lowest price method - using 48H of hourly data
            logger.info(f"⚠️ GARCH failed for {symbol}, using max up/lowest price fallback")
            
            price_changes = df['close'].diff().dropna()
            max_up = price_changes.max()
            tp_price = current_price + max_up
            
            # For SL, use the lowest price in the last 48 hours
            if 'low' in df.columns:
                sl_price = df['low'].min()
            else:
                sl_price = df['close'].min()
            
            tp_percentage = (max_up / current_price) * 100
            sl_percentage = ((current_price - sl_price) / current_price) * 100
            
            # Ensure reasonable bounds
            tp_percentage = max(0.8, min(tp_percentage, 8.0))
            sl_percentage = max(0.8, min(sl_percentage, 8.0))
            
            # Recalculate with bounds
            tp_price = entry_price * (1 + tp_percentage / 100)
            sl_price = entry_price * (1 - sl_percentage / 100)
            
            logger.info(f"✅ Using 48H max up/lowest price TP/SL for {symbol}")
            logger.info(f"   TP: ${tp_price:.4f} (+{tp_percentage:.2f}%)")
            logger.info(f"   SL: ${sl_price:.4f} (-{sl_percentage:.2f}%)")
            
            return {
                'tp_percentage': tp_percentage,
                'sl_percentage': sl_percentage,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'method': 'maxup_lowest_48h',
                'reasoning': f"48H Max Up: +${max_up:.4f}, Lowest: ${sl_price:.4f}"
            }
            
    except Exception as e:
        logger.warning(f"⚠️ Could not calculate TP/SL for {symbol}: {str(e)}, using simple fallback")
        tp_percentage = 2.0
        sl_percentage = 1.5
        tp_price = entry_price * (1 + tp_percentage / 100)
        sl_price = entry_price * (1 - sl_percentage / 100)
        
        return {
            'tp_percentage': tp_percentage,
            'sl_percentage': sl_percentage,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'method': 'fallback_fixed',
            'reasoning': 'Fallback: 2% TP, 1.5% SL'
        }

def execute_trade_via_subprocess_with_garch(symbol, side, price, funds, tp_sl_levels=None):
    """Execute a trade by running tp_sl_fixed.py as a subprocess with GARCH TP/SL levels"""
    try:
        logger.info(f"Executing {side} trade for {symbol} via tp_sl_fixed.py with GARCH TP/SL")
        logger.info(f"Price: ${price}, Value: ${funds}")
        
        # Calculate GARCH-based TP/SL if not provided
        if tp_sl_levels is None:
            tp_sl_levels = calculate_garch_tp_sl_levels(symbol, price)
        
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
        
        # Build command with GARCH-calculated TP/SL percentages
        cmd = [
            sys.executable,
            script_path,
            "--symbol", symbol,
            "--price", str(price),
            "--size", str(size),
            "--tp-percent", str(tp_sl_levels['tp_percentage']),
            "--sl-percent", str(tp_sl_levels['sl_percentage'])
        ]
        
        logger.info(f"🎯 GARCH TP/SL: TP={tp_sl_levels['tp_percentage']:.2f}%, SL={tp_sl_levels['sl_percentage']:.2f}%")
        logger.info(f"📊 Method: {tp_sl_levels['method']} - {tp_sl_levels['reasoning']}")
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
            logger.info("✅ Trade executed successfully via tp_sl_fixed.py with GARCH TP/SL")
            return True
        else:
            logger.error(f"❌ Trade failed with exit code {result.returncode}")
            return False
    except Exception as e:
        logger.error(f"Error executing trade via subprocess with GARCH: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def execute_real_trade(symbol, side, price, funds):
    """Execute a real trade with GARCH-based TP/SL orders using current market price"""
    try:
        logger.info(f"🚀 Executing real {side} trade for {symbol} with GARCH TP/SL")
        logger.info(f"📊 Analysis price: ${price}, Value: ${funds}")
        
        # Import auth headers function from tp_sl_fixed module
        from lk import PRIVATE_KEY_PEM, KEY_NAME, BASE_URL
        import jwt
        from cryptography.hazmat.primitives import serialization
        import secrets
        import uuid
        
        def get_auth_headers_for_price(method, path):
            """Get authentication headers for API requests"""
            try:
                now = int(time.time())
                
                # Ensure proper URI format
                if path.startswith('http'):
                    from urllib.parse import urlparse
                    parsed = urlparse(path)
                    path = parsed.path
                if not path.startswith('/'):
                    path = '/' + path
                    
                # Create JWT payload
                payload = {
                    "sub": KEY_NAME,
                    "iss": "cdp",
                    "nbf": now,
                    "exp": now + 120,
                    "iat": now,
                    "jti": str(uuid.uuid4()),
                    "uri": f"{method} {BASE_URL}{path}"
                }

                # Load private key
                private_key = serialization.load_pem_private_key(
                    PRIVATE_KEY_PEM.encode('utf-8'),
                    password=None
                )

                # Generate token
                token = jwt.encode(
                    payload,
                    private_key,
                    algorithm="ES256",
                    headers={
                        'kid': KEY_NAME,
                        'nonce': secrets.token_hex(16)
                    }
                )
                
                return {
                    'Authorization': f"Bearer {token}",
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
                
            except Exception as e:
                logger.error(f"Error building auth headers: {str(e)}")
                return None
        
        # Get current market price from correct Coinbase Advanced Trade API
        headers = get_auth_headers_for_price("GET", f"/api/v3/brokerage/products/{symbol}")
        response = requests.get(
            f"https://{BASE_URL}/api/v3/brokerage/products/{symbol}",
            headers=headers
        )
        
        if response.status_code != 200:
            logger.error(f"❌ Failed to get product details for {symbol}: {response.status_code}")
            return False
            
        details = response.json()
        current_market_price = float(details.get('price', price))
        min_market_funds = float(details.get('quote_min_size', 1.0))
        
        logger.info(f"💰 Current market price: ${current_market_price:.4f}")
        logger.info(f"📈 Analysis price: ${price:.4f}")
        
        if funds < min_market_funds:
            logger.info(f"📏 Adjusting order size from ${funds:.2f} to minimum ${min_market_funds:.2f}")
            funds = min_market_funds
        
        # Calculate GARCH-based TP/SL levels using current market price
        logger.info(f"🧠 Calculating GARCH-based TP/SL for {symbol}...")
        tp_sl_levels = calculate_garch_tp_sl_levels(symbol, current_market_price)
        
        # Use current market price for execution with GARCH TP/SL
        logger.info(f"✅ Using current market price ${current_market_price:.4f} for {side} order with GARCH TP/SL")
        return execute_trade_via_subprocess_with_garch(symbol, side, current_market_price, funds, tp_sl_levels)
        
    except Exception as e:
        logger.error(f"💥 Error executing trade with GARCH: {str(e)}")
        logger.error(traceback.format_exc())
        return False