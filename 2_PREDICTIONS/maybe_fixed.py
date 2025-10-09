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
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, Input, Output, State
import json
import time
import threading
import queue
import requests
import sys
import subprocess
import re
import uuid
import asyncio
import websockets
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
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
            
            # Enhanced ML decisions tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS enhanced_ml_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    overall_confidence REAL DEFAULT 0,
                    expected_profit_pct REAL DEFAULT 0,
                    expected_profit_usd REAL DEFAULT 0,
                    profit_probability REAL DEFAULT 0,
                    risk_reward_ratio REAL DEFAULT 0,
                    recommended_tp_price REAL,
                    recommended_sl_price REAL,
                    timeframe_count INTEGER DEFAULT 0,
                    signal_strength REAL DEFAULT 0,
                    investment_amount REAL DEFAULT 0,
                    enhanced_data TEXT,
                    timestamp TEXT NOT NULL,
                    executed BOOLEAN DEFAULT 0,
                    execution_result TEXT
                )
            """)
            logger.info("Creating enhanced ML decisions table...")

            # Add enhanced_data column to existing ml_decisions if it doesn't exist
            cursor.execute("PRAGMA table_info(ml_decisions)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'enhanced_data' not in columns:
                cursor.execute("ALTER TABLE ml_decisions ADD COLUMN enhanced_data TEXT")
                logger.info("Added enhanced_data column to ml_decisions table")
            
            if 'profit_probability' not in columns:
                cursor.execute("ALTER TABLE ml_decisions ADD COLUMN profit_probability REAL DEFAULT 0")
                logger.info("Added profit_probability column to ml_decisions table")
            
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
from lk import get_all_accounts, KEY_NAME, PRIVATE_KEY_PEM, BASE_URL

# Import bull run detection configuration
try:
    from bull_run_config import *
    logger.info("✅ Bull run detection configuration loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Could not import bull run config: {e}")
    # Fallback default values - TEMPORARILY LOWERED FOR MARKET CONDITIONS
    BULL_RUN_CONFIDENCE_THRESHOLD = 0.65
    VOLUME_SURGE_MULTIPLIER = 1.5
    MIN_MOMENTUM_SCORE = 5  # Lowered from 30 to 5 for current market
    HIGH_MOMENTUM_THRESHOLD = 50  # Lowered from 70 to 50
    HIGH_MOMENTUM_VOLUME_RISK = 80  # Added missing constant
    ML_BUY_CONFIDENCE_THRESHOLD = 0.60  # LOWERED from 0.75 to 0.60 for better trading

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
# from session_utils import get_active_session  # REMOVED: Causes DB path conflicts - using local version

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
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
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
                    # TEMPORARILY COMMENTED OUT - will fix session handling separately
                    # session_id = get_active_session()
                    # if session_id:
                    #     conn = sqlite3.connect('live_trading.db')
                    #     cursor = conn.cursor()
                    #     cursor.execute('''
                    #     UPDATE positions 
                    #     SET current_price = ?,
                    #         last_update = datetime('now')
                    #     WHERE session_id = ? AND symbol = ?
                    #     ''', (price, session_id, symbol))
                    #     conn.commit()
                    #     conn.close()
                    pass  # Database update temporarily disabled
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
try:
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.config.suppress_callback_exceptions = True
    logger.info("✅ Dash app initialized successfully")
except Exception as e:
    logger.error(f"❌ Error initializing Dash app: {str(e)}")
    raise

# Add error handling for WebSocket client initialization
try:
    # Initialize and start WebSocket client
    ws_client = initialize_websocket_client()
    if ws_client:
        ws_client.start()
        logger.info("✅ WebSocket client started successfully")
    else:
        logger.warning("⚠️ WebSocket client not initialized")
except Exception as e:
    logger.error(f"❌ Error starting WebSocket client: {str(e)}")
    ws_client = None

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

# === Technical Indicators and Feature Engineering ===
def calculate_indicators(df):
    """Calculate comprehensive features for ML model including technical indicators, returns, rankings, and time features"""
    try:
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Convert price columns to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure we have minimum required data
        if len(df) < 100:
            logger.warning(f"Limited data for feature calculation: {len(df)} rows")
        
        # === BASIC OHLCV DATA (already present) ===
        # open, close, low, high, volume
        
        # === DOLLAR VOLUME METRICS ===
        df['dollar_vol'] = df['close'] * df['volume']
        
        # Calculate dollar volume rank (percentile within rolling window)
        df['dollar_vol_rank'] = df['dollar_vol'].rolling(window=min(252, len(df)), min_periods=20).rank(pct=True)
        
        # === TECHNICAL INDICATORS ===
        
        # RSI (both 'rsi' and 'RSI' for different models)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, float('inf'))
        df['rsi'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['rsi']  # Alias
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_middle = df['close'].rolling(window=bb_period, min_periods=1).mean()
        bb_std_dev = df['close'].rolling(window=bb_period, min_periods=1).std()
        df['bb_high'] = bb_middle + (bb_std_dev * bb_std)
        df['bb_low'] = bb_middle - (bb_std_dev * bb_std)
        df['bb_middle'] = bb_middle
        
        # Normalized Average True Range (NATR)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=14, min_periods=1).mean()
        df['ATR'] = atr
        df['NATR'] = (atr / df['close']) * 100  # Normalized ATR as percentage
        
        # PPO (Percentage Price Oscillator)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['PPO'] = ((ema_12 - ema_26) / ema_26) * 100
        
        # MACD (both 'macd' and 'MACD' for different models)
        df['macd'] = ema_12 - ema_26
        df['MACD'] = df['macd']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['Signal_Line'] = df['macd_signal']
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Additional legacy indicators for backward compatibility
        df['EMA12'] = ema_12
        df['EMA26'] = ema_26
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['MA20'] = df['sma_20']
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
        df['upper_band'] = df['bb_high']
        df['lower_band'] = df['bb_low']
        df['middle_band'] = df['bb_middle']
        df['ema_12'] = ema_12  # Alias for breakout patterns function
        df['ema_26'] = ema_26  # Alias for breakout patterns function
        df['bb_upper'] = df['upper_band']  # Alias for breakout patterns function
        df['bb_lower'] = df['lower_band']  # Alias for breakout patterns function
        
        # Stochastic Oscillator
        low_min = df['low'].rolling(window=14, min_periods=1).min()
        high_max = df['high'].rolling(window=14, min_periods=1).max()
        df['%K'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['%D'] = df['%K'].rolling(window=3, min_periods=1).mean()
        
        # On-Balance Volume (OBV)
        price_change = df['close'].diff()
        obv_change = df['volume'].where(price_change > 0, 
                                       -df['volume'].where(price_change < 0, 0))
        df['OBV'] = obv_change.cumsum()
        
        # Rolling standard deviation
        df['rolling_std_10'] = df['close'].rolling(window=10, min_periods=1).std()
        
        # Lag features
        df['lag_1'] = df['close'].shift(1)
        df['lag_2'] = df['close'].shift(2)
        df['lag_3'] = df['close'].shift(3)
        df['predicted_close'] = df['close']  # Placeholder
        
        # === SECTOR INFORMATION ===
        # For crypto, we'll use a simple mapping based on symbol type
        df['sector'] = 1  # All crypto assets in same sector for now
        
        # === RETURNS AT DIFFERENT PERIODS ===
        # Calculate returns for different time periods
        periods = [1, 5, 10, 21, 42, 63]
        
        for period in periods:
            period_str = f"{period:02d}"
            # Forward-looking returns (shifted backward to avoid look-ahead bias in training)
            df[f'r{period_str}'] = df['close'].pct_change(period)
            
        # === DECILE RANKINGS ===
        # Calculate rolling decile rankings for returns
        for period in periods:
            period_str = f"{period:02d}"
            col_name = f'r{period_str}'
            if col_name in df.columns:
                # Use rolling window for ranking (252 trading days ≈ 1 year)
                rolling_window = min(252, len(df))
                df[f'r{period_str}dec'] = df[col_name].rolling(window=rolling_window, min_periods=20).rank(pct=True) * 10
                df[f'r{period_str}dec'] = df[f'r{period_str}dec'].round().clip(1, 10)
        
        # === SECTOR-RELATIVE QUANTILE RANKINGS ===
        # Since we're using a single sector for crypto, these will be the same as regular rankings
        for period in periods:
            period_str = f"{period:02d}"
            col_name = f'r{period_str}'
            if col_name in df.columns:
                rolling_window = min(252, len(df))
                df[f'r{period_str}q_sector'] = df[col_name].rolling(window=rolling_window, min_periods=20).rank(pct=True)
        
        # === FORWARD RETURNS ===
        # These are typically used as targets, but included as features with proper shifting
        forward_periods = [1, 5, 21]
        for period in forward_periods:
            period_str = f"{period:02d}"
            # Shift forward returns to avoid look-ahead bias
            df[f'r{period_str}_fwd'] = df['close'].pct_change(period).shift(-period)
        
        # === TIME FEATURES ===
        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            # If timestamp column exists, use it
            if 'timestamp' in df.columns:
                df.index = pd.to_datetime(df['timestamp'])
            else:
                # Create a dummy datetime index
                df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
        
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['weekday'] = df.index.dayofweek  # Monday=0, Sunday=6
        
        # === CLEAN UP AND FINAL PROCESSING ===
        
        # Fill NaN values with appropriate methods
        # Forward fill first, then backward fill, then fill remaining with 0
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Replace any infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        # Ensure all numeric columns are float64
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].astype('float64')
        
        logger.info(f"Successfully calculated {len(df.columns)} features including comprehensive feature set")
        return df
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        logger.error(traceback.format_exc())
        return df if 'df' in locals() else pd.DataFrame()

# === Model Training ===
def train_model_for_symbol(symbol, granularity=60):
    """Train ML models for a given symbol and granularity"""
    try:
        logger.info(f"Training models for {symbol} with {granularity}s granularity...")
        
        # Get historical data - Updated to 6 months (180 days)
        df = get_coinbase_data(symbol, granularity, days=365)
        if df is None or df.empty:
            logger.error(f"No data available for {symbol}")
            return None
        
        # Calculate indicators
        df = calculate_indicators(df)
        
        # Use simplified feature set for more reliable training
        feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'sma_20', 'sma_50', 'upper_band', 'lower_band',
            'volume', '%K', '%D', 'OBV', 'ATR'
        ]
        
        # Ensure all features exist, fill missing with defaults
        for col in feature_columns:
            if col not in df.columns:
                if col in ['rsi', '%K', '%D']:
                    df[col] = 50  # Neutral value for oscillators
                elif col in ['macd', 'macd_signal', 'macd_hist']:
                    df[col] = 0   # Neutral value for MACD
                elif col in ['sma_20', 'sma_50', 'upper_band', 'lower_band']:
                    df[col] = df['close']  # Use close price as fallback
                else:
                    df[col] = 0   # Default to 0 for other features
        
        # Check if all required features are available
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            logger.error(f"Missing features for {symbol}: {missing_features}")
            return None
        
        # Create target variables
        df['target'] = df['close'].shift(-1) > df['close']
        df['target'] = df['target'].astype(int)
        
        # Fill NaN values with appropriate defaults
        for col in feature_columns:
            if col in ['rsi', '%K', '%D']:
                df[col] = df[col].fillna(50.0)  # Neutral RSI and Stochastic values
            elif col == 'ATR':
                df[col] = df[col].fillna(df['close'] * 0.02)  # 2% of price as default ATR
            elif col in ['sma_20', 'sma_50', 'upper_band', 'lower_band']:
                df[col] = df[col].fillna(df['close'])  # Use current price for moving averages
            elif col in ['macd', 'macd_signal', 'macd_hist']:
                df[col] = df[col].fillna(0.0)  # Neutral MACD values
            elif col == 'volume':
                df[col] = df[col].fillna(df['volume'].median())  # Use median volume
        
        # Drop NaN values
        df = df.dropna()
        
        if len(df) < 100:
            logger.error(f"Insufficient data for {symbol} after preprocessing (need 100, got {len(df)})")
            return None
        
        # Split data
        train_size = int(len(df) * 0.8)
        X_train = df[feature_columns][:train_size]
        y_train = df['target'][:train_size]
        
        # Train models
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
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
        logger.error(traceback.format_exc())
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
    Analyze momentum indicators for a given symbol.
    Simple and effective approach based on the working version.
    
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
        df = df.dropna(subset=['close', 'volume'])  # Only drop rows missing essential data
        
        if df.empty or len(df) < 2:
            return None
        
        latest = df.iloc[-1]
        
        # Calculate price changes safely
        price_change_1h = 0
        price_change_24h = 0
        
        if len(df) >= 2:
            price_change_1h = ((latest['close'] / df['close'].iloc[-2]) - 1) * 100
        
        if len(df) >= 24:
            price_change_24h = ((latest['close'] / df['close'].iloc[-24]) - 1) * 100
        
        # Calculate momentum score (0-100) - SIMPLIFIED AND EFFECTIVE
        momentum_score = (
            (latest.get('rsi', 50) / 100) * 0.3 +
            (1 if latest.get('macd', 0) > latest.get('macd_signal', 0) else 0) * 0.3 +
            (latest.get('%K', 50) / 100) * 0.2 +
            (1 if latest['close'] > latest.get('sma_20', latest['close']) else 0) * 0.2
        ) * 100
        
        # Determine momentum direction
        if latest.get('macd', 0) > latest.get('macd_signal', 0) and latest.get('rsi', 50) > 50:
            momentum_direction = 'Bullish'
        elif latest.get('macd', 0) < latest.get('macd_signal', 0) and latest.get('rsi', 50) < 50:
            momentum_direction = 'Bearish'
        else:
            momentum_direction = 'Neutral'
        
        # Simple bull run detection
        bull_run_detected = (
            momentum_score > 60 and
            latest.get('rsi', 50) > 50 and
            latest.get('macd', 0) > latest.get('macd_signal', 0) and
            price_change_24h > 2
        )
        
        # Risk assessment
        risk_level = "Low"
        if latest.get('rsi', 50) > 70:
            risk_level = "High"
        elif latest.get('rsi', 50) > 60:
            risk_level = "Medium"
        
        logger.debug(f"Symbol: {symbol}, Momentum Score: {momentum_score:.1f}, Direction: {momentum_direction}")
        
        return {
            'symbol': symbol,
            'current_price': float(latest['close']),
            'momentum_score': float(momentum_score),
            'momentum_direction': momentum_direction,
            'price_change_1h': float(price_change_1h),
            'price_change_24h': float(price_change_24h),
            'rsi': float(latest.get('rsi', 50)),
            'macd': float(latest.get('macd', 0)),
            'volume': float(latest.get('volume', 0)),
            'bull_run_detected': bull_run_detected,
            'risk_level': risk_level,
            'breakout_detected': momentum_score > 70
        }
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None

def scan_market(symbols=None, batch_size=5, conn=None, cursor=None):
    """
    Scan the market for trading opportunities.
    SIMPLIFIED VERSION BASED ON WORKING COPY - No complex filtering that blocks opportunities.
    
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
            conn = sqlite3.connect(DB_PATH, timeout=30)
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
                conn = sqlite3.connect(DB_PATH, timeout=30)
                cursor = conn.cursor()
                should_close_conn = True
                logger.debug("Reopened database connection for market scan")

        # Fetch all available symbols if not provided
        if symbols is None:
            symbols = get_cached_symbols()  # Fetch all available symbols
            logger.info(f"📊 Scanning {len(symbols)} symbols for opportunities...")
        else:
            logger.info(f"📊 Scanning {len(symbols)} specified symbols...")
        
        def process_symbol(symbol):
            try:
                logger.debug(f"Processing {symbol}")
                df = get_coinbase_data(symbol=symbol, granularity=3600, days=7)  # 1-hour data, 7 days
                if not df.empty:
                    result = analyze_momentum(df, symbol)
                    if result:
                        logger.debug(f"✅ Analysis complete for {symbol} - Score: {result['momentum_score']:.1f}")
                        return result
                    else:
                        logger.debug(f"⚠️ No analysis results for {symbol}")
                else:
                    logger.debug(f"⚠️ No data retrieved for {symbol}")
                return None
            except Exception as e:
                logger.debug(f"❌ Error processing {symbol}: {str(e)}")
                return None
        
        results = []
        from concurrent.futures import ThreadPoolExecutor
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

        # Sort results by momentum score (highest first)
        sorted_results = sorted(results, key=lambda x: x['momentum_score'], reverse=True)
        
        # SIMPLE FILTERING - Only exclude very poor opportunities
        filtered_results = []
        for result in sorted_results:
            # Only basic filtering - much more permissive
            if result['momentum_score'] > 10:  # Very low threshold
                filtered_results.append(result)
        
        logger.info(f"🎯 SCAN COMPLETE: Found {len(filtered_results)} opportunities from {len(results)} symbols analyzed")
        
        # Log top opportunities
        for i, result in enumerate(filtered_results[:10]):
            logger.info(f"   #{i+1}: {result['symbol']} - Score: {result['momentum_score']:.1f}, Direction: {result['momentum_direction']}")

        return filtered_results
        
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
                conn = sqlite3.connect(DB_PATH, timeout=30)
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
                html.Div(id="training-history", style={'marginTop': '20px', 'color': '#2c3e50'}),
                
                # Auto-training Status
                html.Div(id="auto-training-status", style={'marginTop': '20px', 'color': '#2c3e50'})
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
     Output('analysis-symbol-dropdown', 'options')],
    Input('interval-component', 'n_intervals')
)
def update_symbol_dropdowns(n_intervals):
    """Update symbol dropdown options"""
    try:
        symbols = get_cached_symbols()
        if not symbols:
            logger.warning("No symbols returned from get_cached_symbols()")
            # Return default symbols as fallback
            default_symbols = [
                'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD',
                'MATIC-USD', 'AVAX-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD'
            ]
            options = [{'label': s, 'value': s} for s in default_symbols]
            return options, options
            
        # Limit to first 50 symbols to prevent UI overload
        limited_symbols = symbols[:50] if len(symbols) > 50 else symbols
        options = [{'label': s, 'value': s} for s in limited_symbols]
        
        logger.info(f"Updated symbol dropdowns with {len(options)} symbols")
        return options, options
        
    except Exception as e:
        logger.error(f"Error updating symbol dropdowns: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return safe default options
        default_options = [
            {'label': 'BTC-USD', 'value': 'BTC-USD'},
            {'label': 'ETH-USD', 'value': 'ETH-USD'},
            {'label': 'SOL-USD', 'value': 'SOL-USD'},
            {'label': 'ADA-USD', 'value': 'ADA-USD'},
            {'label': 'Error loading symbols', 'value': 'ERROR', 'disabled': True}
        ]
        return default_options, default_options

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
    """Update training history display"""
    try:
        # Get all models
        models_by_symbol = {}
        for model_file in glob.glob(os.path.join('models', '*.pkl')):
            symbol = os.path.basename(model_file).replace('_rf_model.pkl', '').replace('_lr_model.pkl', '')
            if symbol not in models_by_symbol:
                models_by_symbol[symbol] = []
            models_by_symbol[symbol].append(model_file)
        
        if not models_by_symbol:
            return html.Div("No trained models found", style={'color': '#ff5722'})
        
        return html.Div([
            html.H4("Trained Models", style={'color': '#4CAF50'}),
            html.Div([
                html.Div([
                    html.Strong(f"{symbol}: "),
                    html.Span(f"{len(models)} models")
                ]) 
                for symbol, models in models_by_symbol.items()
            ])
        ])
    except Exception as e:
        return html.Div(f"Error loading training history: {str(e)}")

# Auto-training callback - retrains models every 12 intervals (1 hour with 5-min intervals)
@app.callback(
    Output('auto-training-status', 'children', allow_duplicate=True),
    [Input('interval-component', 'n_intervals')],
    prevent_initial_call=True
)
def auto_retrain_models(n_intervals):
    """Automatically retrain models for active positions every hour"""
    try:
        # Only retrain every 12 intervals (1 hour if interval is 5 minutes)
        if n_intervals % 12 != 0:
            return dash.no_update
        
        logger.info(f"🤖 Starting automatic model retraining (interval {n_intervals})")
        
        # Get active positions from database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT symbol 
            FROM positions 
            WHERE status IN ('active', 'full_hold')
            ORDER BY symbol
        """)
        active_symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not active_symbols:
            logger.info("🤖 No active positions found for auto-retraining")
            return html.Div("🤖 Auto-training: No active positions", 
                          style={'color': '#ff9800', 'fontSize': '12px'})
        
        # Retrain models for active positions
        retrained_count = 0
        for symbol in active_symbols:
            try:
                logger.info(f"🤖 Auto-retraining model for {symbol}...")
                model_rf, model_lr = train_model_for_symbol(symbol, granularity=3600)
                if model_rf is not None and model_lr is not None:
                    retrained_count += 1
                    logger.info(f"✅ Auto-retrained models for {symbol}")
                else:
                    logger.warning(f"❌ Failed to auto-retrain models for {symbol}")
            except Exception as e:
                logger.error(f"❌ Error auto-retraining {symbol}: {str(e)}")
        
        success_message = f"🤖 Auto-retrained {retrained_count}/{len(active_symbols)} models at {datetime.now().strftime('%H:%M:%S')}"
        logger.info(success_message)
        
        return html.Div(success_message, 
                       style={'color': '#4CAF50', 'fontSize': '12px', 'fontStyle': 'italic'})
        
    except Exception as e:
        error_msg = f"🤖 Auto-training error: {str(e)}"
        logger.error(error_msg)
        return html.Div(error_msg, 
                       style={'color': '#f44336', 'fontSize': '12px'})

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
                
                # Check if we need to train a model for this symbol
                model_prefix = f"{symbol.replace('-', '')}_{3600}"  # Using 1h timeframe
                model_path = os.path.join(MODELS_DIR, f"{model_prefix}_clf.pkl")
                
                if not os.path.exists(model_path):
                    logger.info(f"Training new model for {symbol}...")
                    train_model_for_symbol(symbol, 3600)
            
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
    """Get available USD balance for trading using the working lk.py approach"""
    try:
        # Get the absolute path to lk.py
        lk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lk.py"))
        logger.info(f"🔍 Running lk.py from path: {lk_path}")
        
        # Run lk.py using the current Python executable
        result = subprocess.run(
            [sys.executable, lk_path],
            capture_output=True,
            text=True,
            timeout=30  # Add timeout to prevent hanging
        )
        
        logger.info(f"🔍 lk.py return code: {result.returncode}")
        
        if result.returncode != 0:
            logger.warning(f"lk.py returned non-zero exit code: {result.returncode}")
            if result.stderr:
                logger.error(f"lk.py stderr: {result.stderr}")
        
        # Check both stdout and stderr for USD balance
        output_sources = [
            ("stdout", result.stdout),
            ("stderr", result.stderr)
        ]
        
        for source_name, output in output_sources:
            if not output:
                logger.info(f"🔍 {source_name} is empty")
                continue
                
            logger.info(f"🔍 lk.py {source_name} length: {len(output)} characters")
            
            # Method 1: Look for USD account line in the output
            usd_lines_found = 0
            for line in output.split('\n'):
                if 'Account USD:' in line:
                    usd_lines_found += 1
                    logger.info(f"🔍 Found USD line {usd_lines_found} in {source_name}: {line}")
                    if 'available=' in line:
                        logger.info(f"🔍 USD line has 'available=' - processing...")
                        # Extract available balance using regex
                        available_match = re.search(r'available=([0-9]+\.?[0-9]*)', line)
                        if available_match:
                            balance = float(available_match.group(1))
                            logger.info(f"✅ Method 1 ({source_name}): Extracted USD balance: ${balance:.2f}")
                            return balance
                        else:
                            logger.warning(f"🔍 No regex match found in line: {line}")
                    else:
                        logger.info(f"🔍 USD line missing 'available=': {line}")
            
            if usd_lines_found == 0:
                logger.info(f"🔍 No 'Account USD:' lines found in {source_name}")
            
            # Method 2: Try JSON parsing as fallback (mainly for stdout)
            if source_name == "stdout":
                try:
                    lines = output.strip().split('\n')
                    json_lines = [line for line in lines if line.startswith('{') and line.endswith('}')]
                    logger.info(f"🔍 Found {len(json_lines)} JSON lines in stdout")
                    
                    for json_line in json_lines:
                        try:
                            data = json.loads(json_line)
                            if isinstance(data, dict) and 'currency' in data and data.get('currency') == 'USD':
                                available_balance = data.get('available_balance', {}).get('value', '0')
                                balance = float(available_balance)
                                logger.info(f"✅ Method 2: Found USD balance from JSON: ${balance:.2f}")
                                return balance
                        except (json.JSONDecodeError, ValueError, KeyError):
                            continue
                            
                except Exception as e:
                    logger.debug(f"Method 2 (JSON) failed: {e}")
            
            # Method 3: Try regex across all output
            usd_matches = re.findall(r'USD.*?available[=:]?\s*([0-9]+\.?[0-9]*)', output, re.IGNORECASE)
            logger.info(f"🔍 Method 3 found {len(usd_matches)} regex matches in {source_name}")
            if usd_matches:
                balance = float(usd_matches[0])
                logger.info(f"✅ Method 3 ({source_name}): Found USD balance via regex: ${balance:.2f}")
                return balance
        
        logger.warning("❌ No USD balance found in lk.py output")
        logger.debug(f"Full lk.py stdout for debugging:\n{result.stdout}")
        logger.debug(f"Full lk.py stderr for debugging:\n{result.stderr}")
        return 0.0
        
    except subprocess.TimeoutExpired:
        logger.error("lk.py execution timed out")
        return 0.0
    except Exception as e:
        logger.error(f"Error getting available balance: {e}")
        return 0.0

def get_portfolio_value():
    """Get total portfolio value from Coinbase"""
    try:
        # Get the absolute path to lk.py
        lk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lk.py"))
        logger.debug(f"Running lk.py from path: {lk_path}")
        
        # Run lk.py using the current Python executable
        result = subprocess.run(
            [sys.executable, lk_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            logger.error(f"lk.py failed with return code {result.returncode}")
            return 0.0
        
        output = result.stdout.strip()
        logger.debug(f"lk.py output: {output[:200]}...")
        
        # Find the JSON part - look for the line that starts with {
        json_line = None
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('{') and 'total_value' in line:
                json_line = line
                break
        
        if not json_line:
            logger.error("No JSON data found in lk.py output for portfolio value")
            return 0.0
        
        try:
            data = json.loads(json_line)
            total_value = float(data.get('total_value', 0))
            logger.info(f"✅ Portfolio value: ${total_value:.2f}")
            return total_value
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from lk.py output: {json_line}")
            logger.error(f"JSON error: {e}")
            return 0.0
            
    except subprocess.TimeoutExpired:
        logger.error("lk.py execution timed out for portfolio value")
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
                                                    # Basic OHLCV data
                                                    'open', 'close', 'low', 'high', 'volume',
                                                    
                                                    # Dollar volume metrics
                                                    'dollar_vol', 'dollar_vol_rank',
                                                    
                                                    # Technical indicators
                                                    'rsi', 'bb_high', 'bb_low', 'NATR', 'ATR', 'PPO', 'MACD',
                                                    
                                                    # Sector information
                                                    'sector',
                                                    
                                                    # Returns at different periods
                                                    'r01', 'r05', 'r10', 'r21', 'r42', 'r63',
                                                    
                                                    # Decile rankings
                                                    'r01dec', 'r05dec', 'r10dec', 'r21dec', 'r42dec', 'r63dec',
                                                    
                                                    # Sector-relative quantile rankings
                                                    'r01q_sector', 'r05q_sector', 'r10q_sector', 'r21q_sector', 'r42q_sector', 'r63q_sector',
                                                    
                                                    # Forward returns
                                                    'r01_fwd', 'r05_fwd', 'r21_fwd',
                                                    
                                                    # Time features
                                                    'year', 'month', 'weekday'
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
    
    # Trading loop configuration
    trading_delay = 30  # 30 seconds between trading iterations
    
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
                
                # First, scan for market opportunities to know which symbols we might need
                logger.info("🔍 Scanning for market opportunities...")
                opportunities = scan_market()
                
                if not opportunities:
                    logger.info("🔍 No opportunities found from market scan")
                    opportunities = [
                        {'symbol': 'BTC-USD', 'current_price': 50000, 'momentum_score': 30},
                        {'symbol': 'ETH-USD', 'current_price': 3000, 'momentum_score': 25}
                    ]
                else:
                    logger.info(f"🎯 Found {len(opportunities)} opportunities: {', '.join([o['symbol'] for o in opportunities[:10]])}")
                
                # Check for auto-training needs
                opportunity_symbols = [opp['symbol'] for opp in opportunities[:10]]
                
                # Check if any ML models exist for the opportunities
                model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
                os.makedirs(model_dir, exist_ok=True)
                
                models_needed = []
                for symbol in opportunity_symbols:
                    # Check for price prediction models across granularities
                    granularities = [900, 3600, 14400]  # 15m, 1h, 4h
                    models_exist = False
                    for gran in granularities:
                        model_prefix = f"{symbol.replace('-', '')}_{gran}"
                        reg_path = os.path.join(model_dir, f"{model_prefix}_regressor.pkl")
                        if os.path.exists(reg_path):
                            models_exist = True
                            break
                
                    if not models_exist:
                        models_needed.append(symbol)
                
                if models_needed:
                    logger.info(f"🤖 Auto-training price prediction models for {len(models_needed)} symbols: {', '.join(models_needed)}")
                    
                    for symbol in models_needed[:5]:  # Limit to 5 symbols at a time
                        try:
                            logger.info(f"🤖 Auto-training price models for opportunity: {symbol}...")
                            # Train models for multiple granularities
                            for gran in [900, 3600, 14400]:
                                train_price_prediction_model(symbol, gran)
                                time.sleep(2)  # Small delay between model training
                        except Exception as e:
                            logger.error(f"❌ Failed to auto-train price models for {symbol}: {str(e)}")
                    
                    logger.info(f"🎉 Auto-training completed for price prediction models!")
                else:
                    logger.info("✅ All opportunity symbols have trained price prediction models")
                
                # Check available balance for trading
                available_balance = get_available_balance()
                logger.info(f"💰 Available balance for trading: ${available_balance:.2f}")
                
                if available_balance <= 0:
                    logger.warning("❌ No available balance for trading")
                    time.sleep(trading_delay)
                    continue
                
                # Calculate position limits
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(DISTINCT symbol) 
                    FROM positions 
                    WHERE session_id = ? AND status = 'open' AND size >= 0.5
                ''', (session_id,))
                current_positions = cursor.fetchone()[0]
                conn.close()
                
                max_new_positions = max(0, 5 - current_positions)
                
                if max_new_positions > 0 and available_balance > 0:
                    # Dynamic position sizing based on balance
                    if available_balance < 1.0:
                        position_size = max(available_balance * 0.8, 0.10)
                    elif available_balance < 5.0:
                        position_size = min(available_balance * 0.5, 2.0)
                    elif current_positions == 0:
                        position_size = min(available_balance * 0.3, 5.0)
                    else:
                        position_size = min(available_balance / max_new_positions, 2.0)
                    
                    position_size = min(position_size, available_balance * 0.95)  # Leave room for fees
                    
                    logger.info(f"🎯 Looking for trades with position size: ${position_size:.2f}")
                    
                    # Process opportunities with multi-granularity analysis
                    if opportunities:
                        logger.info(f"🎯 Processing {len(opportunities)} opportunities with multi-granularity price prediction...")
                        
                        trade_opportunities = []
                        
                        # Analyze each opportunity with price prediction
                        for opp in opportunities[:max_new_positions * 3]:  # Check more symbols
                            try:
                                symbol = opp['symbol']
                                current_price = opp['current_price']
                                
                                logger.info(f"📊 Analyzing {symbol} with profit-based price prediction...")
                                
                                # Calculate expected profit from price predictions
                                expected_profit_pct, profit_details = calculate_expected_profit(symbol)
                                
                                # Minimum expected profit threshold (after fees)
                                MIN_EXPECTED_PROFIT = 0.05  # 0.05% minimum expected profit after fees (very achievable)
                                
                                if expected_profit_pct >= MIN_EXPECTED_PROFIT:
                                    trade_opportunities.append({
                                        'symbol': symbol,
                                        'current_price': current_price,
                                        'expected_profit': expected_profit_pct,
                                        'profit_details': profit_details,
                                        'opportunity_score': opp.get('momentum_score', 0)
                                    })
                                    # Enhanced logging for profitable opportunities
                                    logger.info(f"🎯 ===============================================")
                                    logger.info(f"✅ {symbol}: PROFITABLE OPPORTUNITY - {expected_profit_pct:.2f}% expected profit after fees")
                                    logger.info(f"   📊 {profit_details}")
                                    logger.info(f"   💰 Current Price: ${current_price:.4f}")
                                    logger.info(f"   🎯 Will trade if balance sufficient!")
                                    logger.info(f"🎯 ===============================================")
                                    # Also print to console to ensure visibility
                                    print(f"🎯 PROFIT OPPORTUNITY: {symbol} - {expected_profit_pct:.2f}% expected profit after fees")
                                else:
                                    logger.info(f"⏭️ {symbol}: Expected profit {expected_profit_pct:.2f}% < {MIN_EXPECTED_PROFIT:.1f}% minimum threshold")
                                    logger.info(f"   📊 {profit_details}")
                                    # Log negative results too for debugging
                                    print(f"⏭️ REJECTED: {symbol} - {expected_profit_pct:.2f}% profit < {MIN_EXPECTED_PROFIT:.1f}% threshold")
                                    
                            except Exception as e:
                                logger.error(f"❌ Error analyzing {symbol}: {str(e)}")
                                continue
                        
                        # Sort by expected profit (highest first)
                        trade_opportunities.sort(key=lambda x: x['expected_profit'], reverse=True)
                        
                        logger.info(f"🎯 Found {len(trade_opportunities)} profitable trade opportunities")
                        
                        # Execute trades for the best opportunities
                        for opportunity in trade_opportunities[:max_new_positions]:
                            try:
                                symbol = opportunity['symbol']
                                current_price = opportunity['current_price']
                                expected_profit = opportunity['expected_profit']
                                
                                # Minimum quantity check
                                min_quantity = 0.000001
                                if position_size / current_price < min_quantity:
                                    logger.info(f"⏭️ Skipping {symbol} - price too high for position size")
                                    continue
                                
                                # Final balance check
                                fresh_balance = get_available_balance()
                                if fresh_balance < position_size:
                                    logger.warning(f"❌ Insufficient balance for {symbol}. Required: ${position_size:.2f}, Available: ${fresh_balance:.2f}")
                                    break
                                
                                logger.info(f"🚀 EXECUTING PROFITABLE TRADE: {symbol}")
                                logger.info(f"   💰 Expected Profit: {expected_profit:.2f}%")
                                logger.info(f"   📊 Price: ${current_price:.4f}")
                                logger.info(f"   💵 Position Size: ${position_size:.2f}")
                                
                                if execute_real_trade(symbol, "BUY", current_price, funds=position_size):
                                    logger.info(f"✅ SUCCESS: Executed profitable trade for {symbol} with {expected_profit:.2f}% expected profit")
                                    break  # Only execute one trade per cycle
                                else:
                                    logger.error(f"❌ FAILED: Could not execute trade for {symbol}")
                                    
                            except Exception as e:
                                logger.error(f"❌ Error executing trade for {symbol}: {str(e)}")
                                continue
                        
                        if not trade_opportunities:
                            logger.info(f"ℹ️ No opportunities met the {MIN_EXPECTED_PROFIT:.1f}% minimum expected profit threshold")
                        else:
                        logger.info("ℹ️ No trading opportunities found from scanner")
                    else:
                    logger.info(f"ℹ️ No new positions available. Current: {current_positions}/3, Balance: ${available_balance:.2f}")
                
                # Sleep before next iteration
                time.sleep(trading_delay)
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
            df = calculate_indicators(df)
        
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
                # .pkl models expect the comprehensive feature set
                feature_columns = [
                    # Basic OHLCV data
                    'open', 'close', 'low', 'high', 'volume',
                    
                    # Dollar volume metrics
                    'dollar_vol', 'dollar_vol_rank',
                    
                    # Technical indicators
                    'rsi', 'bb_high', 'bb_low', 'NATR', 'ATR', 'PPO', 'MACD',
                    
                    # Sector information
                    'sector',
                    
                    # Returns at different periods
                    'r01', 'r05', 'r10', 'r21', 'r42', 'r63',
                    
                    # Decile rankings
                    'r01dec', 'r05dec', 'r10dec', 'r21dec', 'r42dec', 'r63dec',
                    
                    # Sector-relative quantile rankings
                    'r01q_sector', 'r05q_sector', 'r10q_sector', 'r21q_sector', 'r42q_sector', 'r63q_sector',
                    
                    # Forward returns
                    'r01_fwd', 'r05_fwd', 'r21_fwd',
                    
                    # Time features
                    'year', 'month', 'weekday'
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
# --- End Real Trade Execution Functions ---

@app.callback(
    Output('live-positions-table', 'data'),
    [Input('live-trading-interval', 'n_intervals')],
    prevent_initial_call=False
)
def update_live_positions_table(n_intervals):
    """Update live positions table data"""
    try:
        session_id = get_active_session()
        if not session_id:
            logger.warning("No active session found for positions table")
            return []
        
        conn = sqlite3.connect(DB_PATH)
        
        # Get current positions for the active session
        positions_df = pd.read_sql_query('''
            SELECT symbol, quantity, entry_price, current_price, 
                   profit, pnl, value, pl_percentage, status, entry_time
            FROM positions 
            WHERE session_id = ? AND status = 'open'
            ORDER BY entry_time DESC
        ''', conn, params=(session_id,))
        
        conn.close()
        
        if positions_df.empty:
            logger.info("No open positions found")
            return []
        
        # Format the data for the table
        table_data = []
        for _, row in positions_df.iterrows():
            try:
                # Safely format values with error handling
                profit = float(row['profit']) if pd.notna(row['profit']) else 0.0
                pl_percentage = float(row['pl_percentage']) if pd.notna(row['pl_percentage']) else 0.0
                current_price = float(row['current_price']) if pd.notna(row['current_price']) else 0.0
                entry_price = float(row['entry_price']) if pd.notna(row['entry_price']) else 0.0
                quantity = float(row['quantity']) if pd.notna(row['quantity']) else 0.0
                value = float(row['value']) if pd.notna(row['value']) else 0.0
                
                table_data.append({
                    'Symbol': str(row['symbol']),
                    'Quantity': f"{quantity:.6f}",
                    'Entry Price': f"${entry_price:.4f}",
                    'Current Price': f"${current_price:.4f}",
                    'Value': f"${value:.2f}",
                    'P&L': f"${profit:.2f}",
                    'P&L %': f"{pl_percentage:.2f}%",
                    'Status': str(row['status']).title()
                })
            except Exception as row_error:
                logger.error(f"Error formatting position row: {row_error}")
                # Add a safe fallback row
                table_data.append({
                    'Symbol': str(row.get('symbol', 'ERROR')),
                    'Quantity': 'Error',
                    'Entry Price': 'Error',
                    'Current Price': 'Error',
                    'Value': 'Error',
                    'P&L': 'Error',
                    'P&L %': 'Error',
                    'Status': 'Error'
                })
        
        logger.info(f"Updated positions table with {len(table_data)} positions")
        return table_data
        
    except sqlite3.Error as db_error:
        logger.error(f"Database error in positions table: {str(db_error)}")
        return [{'Symbol': 'Database Error', 'Quantity': '-', 'Entry Price': '-', 
                'Current Price': '-', 'Value': '-', 'P&L': '-', 'P&L %': '-', 'Status': '-'}]
        
    except Exception as e:
        logger.error(f"Error updating live positions table: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return [{'Symbol': 'Error', 'Quantity': '-', 'Entry Price': '-', 
                'Current Price': '-', 'Value': '-', 'P&L': '-', 'P&L %': '-', 'Status': '-'}]

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
    """Get live positions from lk.py script with minimum USD value filter"""
    try:
        # Get the absolute path to lk.py in the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lk_path = os.path.join(current_dir, "lk.py")
        
        if not os.path.exists(lk_path):
            logger.error(f"lk.py not found at {lk_path}")
            return []
        
        logger.info(f"Running lk.py from path: {lk_path}")
        
        # Run lk.py
        result = subprocess.run(
            [sys.executable, lk_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            logger.error(f"lk.py failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"lk.py stderr: {result.stderr}")
            return []
        
        output = result.stdout.strip()
        logger.info(f"lk.py output: {output[:500]}...")
        
        # Find the JSON part - look for the line that starts with {
        json_line = None
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('{') and 'positions' in line:
                json_line = line
                break
        
        if not json_line:
            logger.error("No JSON data found in lk.py output")
            return []
        
        try:
            data = json.loads(json_line)
            positions = data.get('positions', [])
            logger.info(f"✅ Successfully parsed {len(positions)} positions from lk.py")
            
            # Filter positions by minimum USD value and exclude USD currency
            filtered_positions = []
            for pos in positions:
                if pos.get('currency') == 'USD':
                    continue  # Skip USD balance
                
                usd_value = float(pos.get('usd_value', 0))
                if usd_value >= min_usd_value:
                    # Convert to symbol format
                    symbol = f"{pos['currency']}-USD"
                    filtered_positions.append({
                        'symbol': symbol,
                        'currency': pos['currency'],
                        'amount': float(pos['amount']),
                        'price': float(pos['price']),
                        'usd_value': usd_value
                    })
                    logger.info(f"📊 Position: {symbol} - {pos['amount']} @ ${pos['price']:.4f} = ${usd_value:.2f}")
            
            logger.info(f"Returning {len(filtered_positions)} filtered positions (min ${min_usd_value})")
            return filtered_positions
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from lk.py output: {json_line}")
            logger.error(f"JSON error: {e}")
            return []
            
    except subprocess.TimeoutExpired:
        logger.error("lk.py execution timed out")
        return []
    except Exception as e:
        logger.error(f"Error getting positions from lk.py: {str(e)}")
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

def calculate_expected_profit(symbol):
    """Calculate expected profit percentage from price predictions across ALL timeframes including fees"""
    try:
        # COINBASE-SUPPORTED GRANULARITIES MAPPED TO REQUESTED TIMEFRAMES
        # We'll use supported granularities and create composite predictions
        coinbase_granularities = {
            '15m': 900,      # 15 minutes - SUPPORTED
            '1h': 3600,      # 1 hour - SUPPORTED  
            '6h': 21600,     # 6 hours - SUPPORTED
            '1d': 86400      # 1 day (24 hours) - SUPPORTED
        }
        
        # Create composite predictions for non-supported timeframes
        composite_timeframes = {
            '30m': ['15m'],           # 30m = 2x 15m analysis
            '4h': ['1h', '6h'],       # 4h = interpolate between 1h and 6h
            '12h': ['6h', '1d'],      # 12h = interpolate between 6h and 1d  
            '48h': ['1d']             # 48h = 2x 1d analysis
        }
        
        predictions = []
        timeframe_details = []
        
        logger.info(f"🎯 Multi-timeframe profit analysis for {symbol} using Coinbase-supported granularities...")
        
        # Get predictions from supported granularities
        base_predictions = {}
        for timeframe_name, gran in coinbase_granularities.items():
            try:
                prediction, confidence = get_price_prediction_for_granularity(symbol, gran)
                if prediction is not None:
                    base_predictions[timeframe_name] = {
                        'prediction': prediction,
                        'confidence': confidence
                    }
                    logger.info(f"   📊 {timeframe_name}: {prediction['direction']} | Change: {prediction['price_change_pct']*100:.2f}% | Confidence: {confidence:.1%}")
                else:
                    logger.warning(f"   ❌ {timeframe_name}: No prediction available")
            except Exception as e:
                logger.warning(f"   ❌ {timeframe_name} ({gran}s) failed: {str(e)}")
                continue
        
        # Process base predictions
        for timeframe_name, pred_data in base_predictions.items():
            prediction = pred_data['prediction']
            confidence = pred_data['confidence']
            
            if prediction['direction'] == 'BUY':
                predictions.append(prediction)
                timeframe_details.append({
                    'timeframe': timeframe_name,
                    'granularity': coinbase_granularities[timeframe_name],
                    'direction': prediction['direction'],
                    'price_change_pct': prediction['price_change_pct'],
                    'confidence': confidence,
                    'predicted_price': prediction['predicted_price'],
                    'horizon_hours': prediction.get('prediction_horizon_hours', coinbase_granularities[timeframe_name]/3600),
                    'type': 'DIRECT'
                })
        
        # Create composite predictions for unsupported timeframes
        for composite_tf, source_timeframes in composite_timeframes.items():
            try:
                composite_prediction = create_composite_prediction(symbol, composite_tf, source_timeframes, base_predictions)
                if composite_prediction is not None and composite_prediction['direction'] == 'BUY':
                    predictions.append(composite_prediction)
                    timeframe_details.append({
                        'timeframe': composite_tf,
                        'granularity': get_composite_granularity(composite_tf),
                        'direction': composite_prediction['direction'],
                        'price_change_pct': composite_prediction['price_change_pct'],
                        'confidence': composite_prediction['confidence'],
                        'predicted_price': composite_prediction['predicted_price'],
                        'horizon_hours': composite_prediction.get('prediction_horizon_hours', get_composite_granularity(composite_tf)/3600),
                        'type': 'COMPOSITE'
                    })
                    logger.info(f"   📊 {composite_tf} (composite): {composite_prediction['direction']} | Change: {composite_prediction['price_change_pct']*100:.2f}% | Confidence: {composite_prediction['confidence']:.1%}")
            except Exception as e:
                logger.warning(f"   ❌ {composite_tf} (composite) failed: {str(e)}")
                continue
        
        if not predictions:
            logger.info(f"   ❌ No valid BUY predictions found for {symbol} across any timeframe")
            return 0.0, "No valid BUY predictions across all timeframes"
        
        # INTELLIGENT TIMEFRAME WEIGHTING
        timeframe_weights = {
            '15m': 0.5,   # Short-term noise, lower weight
            '30m': 0.7,   # Better than 15m, still somewhat noisy
            '1h': 1.0,    # Sweet spot for crypto trading
            '4h': 1.2,    # Very reliable for trend detection
            '6h': 1.1,    # Good for longer trends
            '12h': 0.9,   # Long-term but less actionable
            '1d': 0.8,    # Trend confirmation but slow
            '48h': 0.6    # Very long-term, less weight
        }
        
        # Calculate weighted average expected price change
        total_weighted_change = 0
        total_weight = 0
        consensus_details = []
        
        for detail in timeframe_details:
            timeframe = detail['timeframe']
            weight = timeframe_weights.get(timeframe, 0.5)
            
            # Boost weight for composite predictions if they're based on multiple sources
            if detail['type'] == 'COMPOSITE':
                weight *= 0.8  # Slightly reduce composite prediction weight
            
            weighted_change = detail['price_change_pct'] * weight * detail['confidence']
            
            total_weighted_change += weighted_change
            total_weight += weight * detail['confidence']
            
            consensus_details.append(f"{timeframe}({detail['price_change_pct']*100:.1f}%@{detail['confidence']:.0%})")
        
        if total_weight == 0:
            return 0.0, "No weighted predictions"
        
        avg_expected_change = total_weighted_change / total_weight
        
        # INTELLIGENT FEE CALCULATION
        trading_fees = 0.0035  # 0.35% maker fees (realistic for limit orders)
        
        # Adjust fees based on prediction strength and number of timeframes
        if len(predictions) >= 4:  # Strong multi-timeframe consensus
            trading_fees *= 0.9  # 10% fee discount for strong signals
        
        # Net expected profit after fees
        expected_profit_pct = (avg_expected_change - trading_fees) * 100
        
        # Enhanced prediction summary
        prediction_summary = (
            f"Multi-TF Analysis: {len(predictions)}/{len(granularities)} BUY signals | "
            f"Consensus: {consensus_details[:3]} | "  # Show top 3 timeframes
            f"Gross: {avg_expected_change*100:.2f}% | "
            f"Fees: {trading_fees*100:.2f}%"
        )
        
        logger.info(f"   💰 {symbol}: Expected profit {expected_profit_pct:.2f}% after fees ({len(predictions)} timeframes)")
        logger.info(f"   📋 Details: {prediction_summary}")
        
        return expected_profit_pct, prediction_summary
        
    except Exception as e:
        logger.error(f"Error calculating multi-timeframe expected profit for {symbol}: {str(e)}")
        return 0.0, f"Error: {str(e)}"

def get_composite_granularity(timeframe):
    """Get granularity in seconds for composite timeframes"""
    granularity_map = {
        '30m': 1800,    # 30 minutes
        '4h': 14400,    # 4 hours
        '12h': 43200,   # 12 hours
        '48h': 172800   # 48 hours
    }
    return granularity_map.get(timeframe, 3600)

def create_composite_prediction(symbol, target_timeframe, source_timeframes, base_predictions):
    """Create composite prediction by combining multiple timeframe predictions"""
    try:
        available_sources = []
        
        # Get available source predictions
        for source_tf in source_timeframes:
            if source_tf in base_predictions:
                available_sources.append(base_predictions[source_tf])
        
        if not available_sources:
            return None
        
        # Get current price from any available prediction
        current_price = available_sources[0]['prediction']['current_price']
        
        # Composite prediction strategies based on target timeframe
        if target_timeframe == '30m':
            # 30m = Enhanced 15m prediction (extend time horizon)
            source = available_sources[0]
            pred = source['prediction']
            
            # Scale prediction for longer horizon
            scaled_change = pred['price_change_pct'] * 1.5  # 30m = 1.5x 15m effect
            predicted_price = current_price * (1 + scaled_change)
            confidence = source['confidence'] * 0.9  # Slight confidence reduction for extrapolation
            
        elif target_timeframe == '4h':
            # 4h = Interpolate between 1h and 6h
            if len(available_sources) >= 2:
                # Weighted average of 1h and 6h predictions
                pred1, pred6 = available_sources[0]['prediction'], available_sources[1]['prediction']
                conf1, conf6 = available_sources[0]['confidence'], available_sources[1]['confidence']
                
                # Weight toward 1h (more reliable for 4h prediction)
                w1, w6 = 0.7, 0.3
                scaled_change = (pred1['price_change_pct'] * w1) + (pred6['price_change_pct'] * w6)
                predicted_price = current_price * (1 + scaled_change)
                confidence = (conf1 * w1) + (conf6 * w6)
            else:
                # Fall back to available prediction
                source = available_sources[0]
                pred = source['prediction']
                scaled_change = pred['price_change_pct'] * 2.0  # Scale for 4h
                predicted_price = current_price * (1 + scaled_change)
                confidence = source['confidence'] * 0.8
        
        elif target_timeframe == '12h':
            # 12h = Interpolate between 6h and 1d
            if len(available_sources) >= 2:
                pred6h, pred1d = available_sources[0]['prediction'], available_sources[1]['prediction']
                conf6h, conf1d = available_sources[0]['confidence'], available_sources[1]['confidence']
                
                # Weight toward 6h for 12h prediction
                w6h, w1d = 0.6, 0.4
                scaled_change = (pred6h['price_change_pct'] * w6h) + (pred1d['price_change_pct'] * w1d)
                predicted_price = current_price * (1 + scaled_change)
                confidence = (conf6h * w6h) + (conf1d * w1d)
            else:
                source = available_sources[0]
                pred = source['prediction']
                scaled_change = pred['price_change_pct'] * 1.8  # Scale for 12h
                predicted_price = current_price * (1 + scaled_change)
                confidence = source['confidence'] * 0.75
        
        elif target_timeframe == '48h':
            # 48h = Extended 1d prediction
            source = available_sources[0]
            pred = source['prediction']
            
            scaled_change = pred['price_change_pct'] * 1.8  # 48h = ~2x 1d effect, but diminished
            predicted_price = current_price * (1 + scaled_change)
            confidence = source['confidence'] * 0.7  # Lower confidence for long-term extrapolation
        
        else:
            return None
        
        # Determine direction
        if abs(scaled_change) > 0.003:  # 0.3% minimum threshold
            direction = 'BUY' if scaled_change > 0 else 'SELL'
            else:
            direction = 'HOLD'
        
        return {
            'direction': direction,
            'confidence': confidence,
            'predicted_price': predicted_price,
            'current_price': current_price,
            'price_change_pct': scaled_change,
            'prediction_horizon_hours': get_composite_granularity(target_timeframe) / 3600,
            'composite_source': source_timeframes
        }
        
    except Exception as e:
        logger.error(f"Error creating composite prediction for {target_timeframe}: {str(e)}")
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
    """Make ML decision with multi-timeframe price prediction and profit probability analysis"""
    try:
        # COINBASE-SUPPORTED GRANULARITIES 
        coinbase_granularities = {
            '15m': 900,      # 15 minutes - SUPPORTED
            '1h': 3600,      # 1 hour - SUPPORTED  
            '6h': 21600,     # 6 hours - SUPPORTED
            '1d': 86400      # 1 day (24 hours) - SUPPORTED
        }
        
        predictions = {}
        confidences = {}
        
        logger.info(f"🎯 Multi-timeframe ML decision for {symbol} using Coinbase-supported granularities")
        
        for timeframe_name, gran in coinbase_granularities.items():
            try:
                prediction, confidence = get_price_prediction_for_granularity(symbol, gran)
                if prediction is not None:
                    predictions[gran] = prediction
                    confidences[gran] = confidence
                    logger.info(f"   📊 {timeframe_name}: {prediction['direction']} | Price: ${prediction['predicted_price']:.4f} | Change: {prediction['price_change_pct']*100:.2f}% | Confidence: {confidence:.1%}")
                else:
                    logger.warning(f"   ❌ {timeframe_name}: No prediction available")
            except Exception as e:
                logger.warning(f"   ❌ {timeframe_name} ({gran}s) failed: {str(e)}")
                continue
        
        if not predictions:
            logger.warning(f"❌ No valid predictions for {symbol} across any supported timeframe")
            return None, 0.0
        
        # Analyze consensus across timeframes
        consensus = analyze_multi_timeframe_consensus(predictions, confidences)
        
        if consensus['action'] == 'BUY':
            # Calculate profit probability with fees
            profit_analysis = calculate_profit_probability(symbol, consensus)
            
            if profit_analysis['profit_probability'] >= 0.55:  # 55% chance of profit
                logger.info(f"✅ {symbol}: {consensus['action']} | Multi-TF Consensus: {len(predictions)} timeframes | Profit Prob: {profit_analysis['profit_probability']:.1%} | Expected Return: {profit_analysis['expected_return']:.2%}")
                return consensus['action'], profit_analysis['profit_probability']
            else:
                logger.info(f"⏭️ {symbol}: Skipping - Low profit probability {profit_analysis['profit_probability']:.1%} (need ≥55%)")
                return 'HOLD', profit_analysis['profit_probability']
        
        logger.info(f"📊 {symbol}: {consensus['action']} | Multi-TF Consensus: {len(predictions)} timeframes | Confidence: {consensus['confidence']:.1%}")
        return consensus['action'], consensus['confidence']
        
    except Exception as e:
        logger.error(f"❌ Multi-timeframe ML decision error for {symbol}: {str(e)}")
        return None, 0.0

def get_price_prediction_for_granularity(symbol, granularity):
    """Get intelligent price prediction with dynamic timeframes and price targets"""
    try:
        # Check if regression model exists
        model_prefix = f"{symbol.replace('-', '')}_{granularity}"
        reg_path = os.path.join(MODELS_DIR, f"{model_prefix}_regressor.pkl")
        
        logger.info(f"🔍 Looking for multi-timeframe model: {reg_path}")
        
        if not os.path.exists(reg_path):
            logger.info(f"Training multi-timeframe price prediction model for {symbol} ({granularity}s)...")
            train_price_prediction_model(symbol, granularity)
            
            if not os.path.exists(reg_path):
                logger.warning(f"❌ Model still doesn't exist after training: {reg_path}")
                return None, 0.0
        
        logger.info(f"✅ Found existing multi-timeframe model: {reg_path}")
        
        # Load model with metadata
        import joblib
        model_data = joblib.load(reg_path)
        
        # Handle both old and new model formats
        if isinstance(model_data, dict):
            regressor = model_data['model']
            prediction_horizon_hours = model_data.get('prediction_horizon_hours', 4.0)
            direction_accuracy = model_data.get('direction_accuracy', 0.5)
            mae = model_data.get('mae', 0.01)
            features = model_data.get('features', [])
        else:
            # Old format - just the model
            regressor = model_data
            prediction_horizon_hours = 4.0  # Default
            direction_accuracy = 0.5
            mae = 0.01
            features = []
        
        logger.info(f"✅ Loaded multi-timeframe model for {symbol} ({granularity}s)")
        logger.info(f"📊 Model predicts {prediction_horizon_hours:.1f}h ahead with {direction_accuracy:.1%} accuracy")
        
        # Get recent data for prediction  
        df = get_coinbase_data(symbol, granularity, days=90)
        if df is None or df.empty:
            logger.error(f"No recent data for {symbol}")
            return None, 0.0
        
        logger.info(f"📊 Got {len(df)} data points for {symbol} ({granularity}s)")
        
        # Calculate indicators
        df = calculate_indicators(df)
        df = df.dropna(subset=['close', 'volume'])  # Only drop rows missing essential data
        
        if len(df) < 10:  # Reduced minimum requirement due to longer data window
            logger.error(f"Insufficient data for {symbol}: {len(df)} rows")
            return None, 0.0
        
        # Get current price
        current_price = df['close'].iloc[-1]
        logger.info(f"💰 Current price: ${current_price:.4f}")
        
        # Prepare features for prediction
        feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'sma_20', 'sma_50', 'upper_band', 'lower_band',
            'volume', '%K', '%D', 'OBV', 'ATR'
        ]
        
        # Add crypto-specific features
        df['price_momentum_3'] = df['close'].pct_change(3)
        df['price_momentum_5'] = df['close'].pct_change(5)
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['price_volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        df['volume_volatility'] = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()
        df['trend_strength'] = abs(df['close'].rolling(10).mean() - df['close'].rolling(30).mean()) / df['close']
        
        extended_features = feature_columns + [
            'price_momentum_3', 'price_momentum_5', 'volume_ratio', 
            'price_volatility', 'volume_volatility', 'trend_strength'
        ]
        
        # Get the latest feature values
        try:
            # Try extended features first (for newer models)
            latest_features = df[extended_features].iloc[-1:].fillna(0)
            predicted_return = regressor.predict(latest_features)[0]
            feature_set_used = "extended"
            logger.info(f"✅ Using extended feature set ({len(extended_features)} features)")
        except Exception as feature_error:
            # Fall back to basic features if extended features fail
            logger.warning(f"Extended features failed ({str(feature_error)[:50]}), falling back to basic features")
            try:
                latest_features = df[feature_columns].iloc[-1:].fillna(0)
                predicted_return = regressor.predict(latest_features)[0]
                feature_set_used = "basic" 
                logger.info(f"✅ Using basic feature set ({len(feature_columns)} features)")
            except Exception as basic_error:
                logger.error(f"Even basic features failed: {str(basic_error)}")
                return None, 0.0
        
        predicted_price = current_price * (1 + predicted_return)
        predicted_price_change_pct = predicted_return
        
        logger.info(f"🔮 Predicted return: {predicted_return*100:.3f}% over {prediction_horizon_hours:.1f}h ({feature_set_used} features)")
        logger.info(f"🔮 Predicted price: ${predicted_price:.4f}")
        
        # Determine direction and confidence with optimized thresholds for crypto
        min_threshold = 0.001  # 0.1% minimum for any direction (more sensitive)
        confidence_threshold = max(min_threshold, mae * 1.0)  # Use 1x model MAE as threshold (more aggressive)
        
        if abs(predicted_price_change_pct) > confidence_threshold:
            direction = 'BUY' if predicted_price_change_pct > 0 else 'SELL'
            
            # Calculate confidence based on prediction strength and model accuracy (optimized for crypto)
            prediction_strength = abs(predicted_price_change_pct) / 0.02  # Normalize to 2% max (more sensitive)
            confidence = min(0.95, direction_accuracy * prediction_strength * 1.5)  # Boost confidence more
            
            logger.info(f"✅ Direction: {direction} | Confidence: {confidence:.1%}")
        else:
            direction = 'HOLD'
            confidence = 0.2
            logger.info(f"⏸️ Direction: {direction} (change too small: {predicted_price_change_pct*100:.3f}%)")
        
        # Calculate intelligent TP/SL levels based on prediction
        prediction_data = {
            'direction': direction,
            'confidence': confidence,
            'predicted_price': predicted_price,
            'current_price': current_price,
            'price_change_pct': predicted_price_change_pct,
            'prediction_horizon_hours': prediction_horizon_hours,
            'model_accuracy': direction_accuracy,
            'model_mae': mae,
            'granularity': granularity
        }
        
        return prediction_data, confidence
        
    except Exception as e:
        logger.error(f"Error getting price prediction for {symbol} ({granularity}s): {str(e)}")
        return None, 0.0

def train_price_prediction_model(symbol, granularity):
    """Train a more realistic model for crypto price direction prediction with multiple timeframe targets"""
    try:
        logger.info(f"🤖 Training multi-timeframe price model for {symbol} ({granularity}s)...")
        
        # Get historical data
        df = get_coinbase_data(symbol, granularity, days=120)
        if df is None or df.empty:
            logger.error(f"No data for {symbol}")
            return None
            
        # Calculate indicators
        df = calculate_indicators(df)
        df.dropna(inplace=True)
        
        if len(df) < 50:
            logger.error(f"Insufficient data for {symbol}: {len(df)} rows")
            return None

        # Better feature engineering for crypto
        feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'sma_20', 'sma_50', 'upper_band', 'lower_band',
            'volume', '%K', '%D', 'OBV', 'ATR'
        ]
        
        # Add crypto-specific momentum features
        df['price_momentum_3'] = df['close'].pct_change(3)
        df['price_momentum_5'] = df['close'].pct_change(5)
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Add volatility features (crucial for crypto)
        df['price_volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        df['volume_volatility'] = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()
        
        # Add trend strength features
        df['trend_strength'] = abs(df['close'].rolling(10).mean() - df['close'].rolling(30).mean()) / df['close']
        
        extended_features = feature_columns + [
            'price_momentum_3', 'price_momentum_5', 'volume_ratio', 
            'price_volatility', 'volume_volatility', 'trend_strength'
        ]
        
        # MULTI-TIMEFRAME PREDICTION TARGETS
        # Predict multiple horizons: 1x, 4x, 12x, 24x the granularity
        prediction_horizons = {
            'short': max(1, int(granularity // 900)),      # ~1 period ahead
            'medium': max(4, int(granularity // 225)),     # ~4 periods ahead  
            'long': max(12, int(granularity // 75)),       # ~12 periods ahead
            'extended': max(24, int(granularity // 37.5))  # ~24 periods ahead
        }
        
        # Create multiple targets
        for horizon_name, steps in prediction_horizons.items():
            df[f'future_return_{horizon_name}'] = df['close'].shift(-steps) / df['close'] - 1
            df[f'future_price_{horizon_name}'] = df['close'].shift(-steps)
            
        # For this model, focus on medium-term (gives best balance of predictability vs utility)
        primary_target = 'future_return_medium'
        df['capped_return'] = df[primary_target].clip(-0.1, 0.1)  # Cap at ±10%
        
        # Clean data - be more lenient about missing data
        essential_columns = ['close', 'volume']
        df = df.dropna(subset=essential_columns)  # Only require basic price/volume data
        
        # Fill NaN values in feature columns more intelligently
        for col in extended_features:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Handle target variable more gracefully
        df[primary_target] = df[primary_target].fillna(0)
        df['capped_return'] = df[primary_target].clip(-0.1, 0.1)  # Cap at ±10%
        
        if len(df) < 10:  # Very lenient minimum requirement (was 20)
            logger.error(f"Insufficient clean data for {symbol}: {len(df)} rows")
            return None
        
        # Prepare features and target
        X = df[extended_features].fillna(0)
        y = df['capped_return'].fillna(0)
        
        # Split data (use more recent data for testing - important for time series)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Use a simpler, more robust model for crypto
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        regressor = RandomForestRegressor(
            n_estimators=50,      # Reduced to prevent overfitting
            max_depth=5,          # Much shallower trees
            min_samples_split=10, # More conservative splitting
            min_samples_leaf=5,   # Larger leaf sizes
            max_features=0.7,     # Don't use all features
            random_state=42
        )
        
        regressor.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate more realistic metrics for crypto
        correct_direction = sum((y_test > 0) == (y_pred > 0)) / len(y_test)
        
        # Calculate prediction horizon in hours for logging
        horizon_hours = prediction_horizons['medium'] * granularity / 3600
        
        logger.info(f"📊 {symbol} ({granularity}s) - Predicting {horizon_hours:.1f}h ahead")
        logger.info(f"📊 MSE: {mse:.6f}, R²: {r2:.3f}, MAE: {mae:.4f}")
        logger.info(f"📊 Direction Accuracy: {correct_direction:.1%} (more important than R² for crypto)")
        
        # For crypto, direction accuracy > 55% is actually quite good!
        if correct_direction > 0.52:  # 52% is better than random
            logger.info(f"✅ Model shows predictive power - Direction accuracy: {correct_direction:.1%}")
        else:
            logger.warning(f"⚠️ Model may not be predictive - Direction accuracy: {correct_direction:.1%}")
        
        # Save model with metadata
        import joblib
        model_prefix = f"{symbol.replace('-', '')}_{granularity}"
        reg_path = os.path.join(MODELS_DIR, f"{model_prefix}_regressor.pkl")
        
        # Save model with metadata
        model_data = {
            'model': regressor,
            'prediction_horizon_steps': prediction_horizons['medium'],
            'prediction_horizon_hours': horizon_hours,
            'direction_accuracy': correct_direction,
            'mae': mae,
            'features': extended_features
        }
        
        joblib.dump(model_data, reg_path)
        
        logger.info(f"✅ Multi-timeframe price model saved for {symbol} ({granularity}s)")
        return regressor
        
    except Exception as e:
        logger.error(f"Error training improved price model for {symbol}: {str(e)}")
        return None

def analyze_multi_timeframe_consensus(predictions, confidences):
    """Analyze consensus across multiple timeframes"""
    try:
        if not predictions:
            return {'action': 'HOLD', 'confidence': 0.0}
        
        # Weight timeframes (only supported granularities)
        weights = {900: 0.4, 3600: 0.6}  # 15m, 1h (removed 14400s)
        
        buy_score = 0
        sell_score = 0
        total_weight = 0
        
        price_predictions = []
        
        for gran, prediction in predictions.items():
            weight = weights.get(gran, 0.3)
            confidence = confidences.get(gran, 0.5)
            
            # Weighted scoring
            if prediction['direction'] == 'BUY':
                buy_score += weight * confidence
            elif prediction['direction'] == 'SELL':
                sell_score += weight * confidence
            
            total_weight += weight
            price_predictions.append(prediction['predicted_price'])
        
        # Normalize scores
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
        
        # Determine consensus (lower thresholds)
        if buy_score > sell_score and buy_score > 0.4:  # Lowered from 0.6 to 0.4
            action = 'BUY'
            confidence = buy_score
        elif sell_score > buy_score and sell_score > 0.4:  # Lowered from 0.6 to 0.4
            action = 'SELL'  
            confidence = sell_score
        else:
            action = 'HOLD'
            confidence = max(buy_score, sell_score, 0.2)
        
        # Average predicted price
        avg_predicted_price = sum(price_predictions) / len(price_predictions) if price_predictions else 0
        
        return {
            'action': action,
            'confidence': confidence,
            'predicted_price': avg_predicted_price,
            'buy_score': buy_score,
            'sell_score': sell_score
        }
        
    except Exception as e:
        logger.error(f"Error analyzing consensus: {str(e)}")
        return {'action': 'HOLD', 'confidence': 0.0}

def calculate_profit_probability(symbol, consensus):
    """Calculate profit probability considering fees and predicted price movement"""
    try:
        # Coinbase Advanced Trade fees (approximate)
        MAKER_FEE = 0.006  # 0.6%
        TAKER_FEE = 0.008  # 0.8%
        
        # Use taker fee as worst case (market orders)
        TOTAL_FEES = TAKER_FEE * 2  # Buy + Sell
        
        # Get current price
        current_price = consensus.get('avg_predicted_price', 0)
        if current_price <= 0:
            return {'profit_probability': 0.0, 'expected_return': 0.0}
        
        # Calculate minimum price movement needed to break even
        breakeven_threshold = TOTAL_FEES + 0.005  # Fees + 0.5% profit margin
        
        # Expected price movement from consensus
        expected_return = 0
        for gran, prediction in predictions.items():
            if prediction and 'price_change_pct' in prediction:
                expected_return += prediction['price_change_pct']
        
        expected_return /= len(predictions) if predictions else 1
        
        # Calculate profit probability
        if expected_return > breakeven_threshold:
            # Simple probability model based on expected return vs required return
            profit_probability = min(0.95, (expected_return / breakeven_threshold) * 0.7)
        else:
            profit_probability = max(0.1, expected_return / breakeven_threshold * 0.3)
        
        # Adjust for confidence
        confidence_factor = consensus.get('confidence', 0.5)
        profit_probability *= confidence_factor
        
        # Net expected return after fees
        net_expected_return = expected_return - TOTAL_FEES
        
        return {
            'profit_probability': profit_probability,
            'expected_return': net_expected_return,
            'breakeven_threshold': breakeven_threshold,
            'total_fees': TOTAL_FEES,
            'confidence_factor': confidence_factor
        }
        
    except Exception as e:
        logger.error(f"Error calculating profit probability: {str(e)}")
        return {'profit_probability': 0.0, 'expected_return': 0.0}

# === Enhanced Scanner Functions ===

# Enhanced position sizing logic from the advanced trading system
def calculate_position_size(available_balance, current_positions=0, max_positions=5):
    """
    Calculate optimal position size based on available balance and current positions.
    
    Multi-tier strategy:
    - Very small balances (<$1): Use 80% or 10 cents minimum
    - Small balances (<$5): Use 50% or $2 maximum  
    - First position with larger balance: Use 30% or $5 maximum
    - Subsequent positions: Split remaining or $2 maximum
    - Always leave 5% for fees
    """
    try:
        max_new_positions = max(0, max_positions - current_positions)
        
        if max_new_positions <= 0:
            return 0.0
            
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
        
        logger.info(f"📊 Position sizing: Balance=${available_balance:.2f}, Positions={current_positions}/{max_positions}, Size=${position_size:.2f}")
        return position_size
        
    except Exception as e:
        logger.error(f"Error calculating position size: {str(e)}")
        return 0.0

def calculate_intelligent_tp_sl(predictions, current_price, symbol):
    """Calculate intelligent TP/SL based on prediction timeframes and price targets"""
    try:
        if not predictions:
            logger.warning(f"No predictions available for {symbol}, using default TP/SL")
            return {
                'tp_price': current_price * 1.02,  # Default +2%
                'sl_price': current_price * 0.98,  # Default -2%
                'tp_pct': 2.0,
                'sl_pct': -2.0,
                'timeframe_hours': 4.0,
                'strategy': 'default'
            }
        
        # Analyze predictions from multiple timeframes
        buy_predictions = []
        total_confidence = 0
        
        for gran, prediction in predictions.items():
            if prediction and prediction['direction'] == 'BUY':
                buy_predictions.append(prediction)
                total_confidence += prediction.get('confidence', 0.5)
        
        if not buy_predictions:
            logger.info(f"No BUY predictions for {symbol}, using conservative levels")
            return {
                'tp_price': current_price * 1.01,  # Conservative +1%
                'sl_price': current_price * 0.995, # Conservative -0.5%
                'tp_pct': 1.0,
                'sl_pct': -0.5,
                'timeframe_hours': 2.0,
                'strategy': 'conservative'
            }
        
        # Calculate weighted average prediction
        total_weight = 0
        weighted_return = 0
        avg_horizon_hours = 0
        avg_accuracy = 0
        avg_mae = 0
        
        for pred in buy_predictions:
            confidence = pred.get('confidence', 0.5)
            price_change = pred.get('price_change_pct', 0.02)
            horizon_hours = pred.get('prediction_horizon_hours', 4.0)
            accuracy = pred.get('model_accuracy', 0.5)
            mae = pred.get('model_mae', 0.01)
            
            # Weight by confidence
            weight = confidence
            weighted_return += price_change * weight
            avg_horizon_hours += horizon_hours * weight
            avg_accuracy += accuracy * weight
            avg_mae += mae * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_return /= total_weight
            avg_horizon_hours /= total_weight
            avg_accuracy /= total_weight
            avg_mae /= total_weight
        
        # Calculate intelligent TP/SL based on prediction
        predicted_gain_pct = weighted_return * 100
        
        # Dynamic TP calculation
        if predicted_gain_pct > 5.0:
            # Large prediction: Take 70% of predicted gain
            tp_pct = predicted_gain_pct * 0.7
            strategy = 'aggressive'
        elif predicted_gain_pct > 2.0:
            # Medium prediction: Take 80% of predicted gain  
            tp_pct = predicted_gain_pct * 0.8
            strategy = 'moderate'
        else:
            # Small prediction: Take 90% of predicted gain
            tp_pct = max(1.0, predicted_gain_pct * 0.9)  # Minimum 1%
            strategy = 'conservative'
        
        # Dynamic SL calculation based on model accuracy and MAE
        if avg_accuracy > 0.6 and avg_mae < 0.01:
            # High accuracy model: Tighter stop loss
            sl_pct = -max(0.5, tp_pct * 0.4)  # 40% of TP, minimum 0.5%
        elif avg_accuracy > 0.55:
            # Good accuracy: Balanced stop loss
            sl_pct = -max(1.0, tp_pct * 0.5)  # 50% of TP, minimum 1%
        else:
            # Lower accuracy: Wider stop loss
            sl_pct = -max(1.5, tp_pct * 0.6)  # 60% of TP, minimum 1.5%
        
        # Calculate actual prices
        tp_price = current_price * (1 + tp_pct/100)
        sl_price = current_price * (1 + sl_pct/100)
        
        # Ensure minimum profit after fees (0.35% maker fee)
        min_profit_after_fees = 0.8  # 0.8% minimum profit after 0.35% fees
        if tp_pct < min_profit_after_fees:
            tp_pct = min_profit_after_fees
            tp_price = current_price * (1 + tp_pct/100)
            strategy = 'min_profit'
        
        logger.info(f"🎯 {symbol} Intelligent TP/SL Strategy: {strategy}")
        logger.info(f"   💰 Current: ${current_price:.4f}")
        logger.info(f"   📈 TP: ${tp_price:.4f} (+{tp_pct:.1f}%)")
        logger.info(f"   📉 SL: ${sl_price:.4f} ({sl_pct:.1f}%)")
        logger.info(f"   ⏰ Timeframe: {avg_horizon_hours:.1f}h")
        logger.info(f"   🎯 Predicted: +{predicted_gain_pct:.1f}%")
        logger.info(f"   📊 Model Accuracy: {avg_accuracy:.1%}")
        
        return {
            'tp_price': tp_price,
            'sl_price': sl_price,
            'tp_pct': tp_pct,
            'sl_pct': sl_pct,
            'timeframe_hours': avg_horizon_hours,
            'predicted_gain_pct': predicted_gain_pct,
            'model_accuracy': avg_accuracy,
            'strategy': strategy,
            'total_confidence': total_confidence
        }
        
    except Exception as e:
        logger.error(f"Error calculating intelligent TP/SL for {symbol}: {str(e)}")
        # Return safe defaults
        return {
            'tp_price': current_price * 1.015,  # +1.5%
            'sl_price': current_price * 0.99,   # -1%
            'tp_pct': 1.5,
            'sl_pct': -1.0,
            'timeframe_hours': 4.0,
            'strategy': 'error_fallback'
        }

# Add missing callbacks after the existing callbacks

# Add callback for live portfolio chart
@app.callback(
    Output('live-portfolio-chart', 'figure'),
    [Input('live-trading-interval', 'n_intervals')],
    prevent_initial_call=False
)
def update_live_portfolio_chart(n_intervals):
    """Update the live portfolio chart"""
    try:
        # Get portfolio history from database
        session_id = get_active_session()
        if not session_id:
            # Return empty chart if no session
            fig = go.Figure()
            fig.add_annotation(
                text="No active trading session",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Time",
                yaxis_title="Portfolio Value ($)",
                template="plotly_dark"
            )
            return fig
        
        conn = sqlite3.connect(DB_PATH)
        
        # Get portfolio history
        portfolio_df = pd.read_sql_query('''
            SELECT timestamp, total_value, available_balance, positions_value
            FROM portfolio_history 
            WHERE session_id = ?
            ORDER BY timestamp DESC LIMIT 100
        ''', conn, params=(session_id,))
        
        conn.close()
        
        if portfolio_df.empty:
            # Return chart with no data message
            fig = go.Figure()
            fig.add_annotation(
                text="No portfolio data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Time",
                yaxis_title="Portfolio Value ($)",
                template="plotly_dark"
            )
            return fig
        
        # Convert timestamp to datetime
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        
        # Create the figure
        fig = go.Figure()
        
        # Add portfolio value line
        fig.add_trace(go.Scatter(
            x=portfolio_df['timestamp'],
            y=portfolio_df['total_value'],
            mode='lines+markers',
            name='Total Portfolio Value',
            line=dict(color='#00ff88', width=2),
            marker=dict(size=4)
        ))
        
        # Add positions value
        fig.add_trace(go.Scatter(
            x=portfolio_df['timestamp'],
            y=portfolio_df['positions_value'],
            mode='lines',
            name='Positions Value',
            line=dict(color='#ffa500', width=1),
            fill='tonexty',
            fillcolor='rgba(255, 165, 0, 0.1)'
        ))
        
        # Add available balance
        fig.add_trace(go.Scatter(
            x=portfolio_df['timestamp'],
            y=portfolio_df['available_balance'],
            mode='lines',
            name='Available Balance',
            line=dict(color='#00aaff', width=1)
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Portfolio Performance - Session {session_id}",
            xaxis_title="Time",
            yaxis_title="Value ($)",
            template="plotly_dark",
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Format x-axis
        fig.update_xaxes(
            tickformat='%H:%M:%S',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
        
        # Format y-axis
        fig.update_yaxes(
            tickformat='$,.2f',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error updating live portfolio chart: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return error chart
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading chart: {str(e)[:100]}...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title="Portfolio Value Over Time (Error)",
            xaxis_title="Time",
            yaxis_title="Portfolio Value ($)",
            template="plotly_dark",
            height=400
        )
        return fig

# Add callback for profit-loss chart
@app.callback(
    Output('profit-loss-chart', 'figure'),
    [Input('analysis-symbol-dropdown', 'value')],
    prevent_initial_call=True
)
def update_profit_loss_chart(selected_symbol):
    """Update profit/loss chart for selected symbol"""
    try:
        if not selected_symbol:
            # Return empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="Please select a symbol to view P&L chart",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="Profit & Loss Analysis",
                xaxis_title="Time",
                yaxis_title="P&L ($)",
                template="plotly_dark",
                height=400
            )
            return fig
        
        # Get session data
        session_id = get_active_session()
        if not session_id:
            fig = go.Figure()
            fig.add_annotation(
                text="No active trading session",
                xref="paper", yref="paper", 
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="orange")
            )
            fig.update_layout(
                title="Profit & Loss Analysis",
                xaxis_title="Time",
                yaxis_title="P&L ($)",
                template="plotly_dark",
                height=400
            )
            return fig
        
        # Query database for P&L data
        conn = sqlite3.connect(DB_PATH)
        
        trades_df = pd.read_sql_query('''
            SELECT timestamp, symbol, profit, value
            FROM trades 
            WHERE session_id = ? AND symbol = ?
            ORDER BY timestamp
        ''', conn, params=(session_id, selected_symbol))
        
        conn.close()
        
        if trades_df.empty:
            # Return chart with no data message
            fig = go.Figure()
            fig.add_annotation(
                text=f"No trading data available for {selected_symbol}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title=f"Profit & Loss Analysis - {selected_symbol}",
                xaxis_title="Time",
                yaxis_title="P&L ($)",
                template="plotly_dark",
                height=400
            )
            return fig
        
        # Process data for chart
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df['cumulative_pnl'] = trades_df['profit'].cumsum()
        
        # Create figure
        fig = go.Figure()
        
        # Add cumulative P&L line
        fig.add_trace(go.Scatter(
            x=trades_df['timestamp'],
            y=trades_df['cumulative_pnl'],
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(color='#00ff88', width=2),
            marker=dict(size=6),
            fill='tonexty',
            fillcolor='rgba(0, 255, 136, 0.1)'
        ))
        
        # Add individual trade profits as bars
        colors = ['green' if profit >= 0 else 'red' for profit in trades_df['profit']]
        fig.add_trace(go.Bar(
            x=trades_df['timestamp'],
            y=trades_df['profit'],
            name='Trade P&L',
            marker_color=colors,
            opacity=0.7
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Profit & Loss Analysis - {selected_symbol}",
            xaxis_title="Time",
            yaxis_title="P&L ($)",
            template="plotly_dark",
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right", 
                x=1
            )
        )
        
        # Format axes
        fig.update_xaxes(
            tickformat='%m/%d %H:%M',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
        
        fig.update_yaxes(
            tickformat='$,.2f',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
        
        return fig
        
    except sqlite3.Error as db_error:
        logger.error(f"Database error in profit/loss chart: {str(db_error)}")
        fig = go.Figure()
        fig.add_annotation(
            text="Database error loading P&L data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title="Profit & Loss Analysis (Error)",
            xaxis_title="Time",
            yaxis_title="P&L ($)",
            template="plotly_dark",
            height=400
        )
        return fig
        
    except Exception as e:
        logger.error(f"Error updating profit/loss chart: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return error chart
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading chart: {str(e)[:50]}...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title="Profit & Loss Analysis (Error)",
            xaxis_title="Time",
            yaxis_title="P&L ($)",
            template="plotly_dark",
            height=400
        )
        return fig

# Add callback for drawdown chart
@app.callback(
    Output('drawdown-chart', 'figure'),
    [Input('analysis-symbol-dropdown', 'value')],
    prevent_initial_call=True
)
def update_drawdown_chart(selected_symbol):
    """Update drawdown chart for selected symbol"""
    try:
        if not selected_symbol:
            # Return empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="Please select a symbol to view drawdown chart",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="Drawdown Analysis",
                xaxis_title="Time",
                yaxis_title="Drawdown (%)",
                template="plotly_dark",
                height=400
            )
            return fig
        
        # Get session data
        session_id = get_active_session()
        if not session_id:
            fig = go.Figure()
            fig.add_annotation(
                text="No active trading session",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="orange")
            )
            fig.update_layout(
                title="Drawdown Analysis",
                xaxis_title="Time", 
                yaxis_title="Drawdown (%)",
                template="plotly_dark",
                height=400
            )
            return fig
        
        # Query database for portfolio history
        conn = sqlite3.connect(DB_PATH)
        
        portfolio_df = pd.read_sql_query('''
            SELECT timestamp, total_value
            FROM portfolio_history 
            WHERE session_id = ?
            ORDER BY timestamp
        ''', conn, params=(session_id,))
        
        conn.close()
        
        if portfolio_df.empty:
            # Return chart with no data message
            fig = go.Figure()
            fig.add_annotation(
                text="No portfolio data available for drawdown analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="Drawdown Analysis",
                xaxis_title="Time",
                yaxis_title="Drawdown (%)",
                template="plotly_dark",
                height=400
            )
            return fig
        
        # Calculate drawdown
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        portfolio_df['running_max'] = portfolio_df['total_value'].expanding().max()
        portfolio_df['drawdown'] = ((portfolio_df['total_value'] - portfolio_df['running_max']) / portfolio_df['running_max']) * 100
        
        # Create figure
        fig = go.Figure()
        
        # Add drawdown area chart
        fig.add_trace(go.Scatter(
            x=portfolio_df['timestamp'],
            y=portfolio_df['drawdown'],
            mode='lines',
            name='Drawdown',
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='red', width=2)
        ))
        
        # Add zero line
        fig.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="gray",
            annotation_text="Break-even"
        )
        
        # Update layout
        fig.update_layout(
            title=f"Drawdown Analysis - Portfolio",
            xaxis_title="Time",
            yaxis_title="Drawdown (%)",
            template="plotly_dark",
            height=400,
            showlegend=False
        )
        
        # Format axes
        fig.update_xaxes(
            tickformat='%m/%d %H:%M',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
        
        fig.update_yaxes(
            tickformat='.2f',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
        
        return fig
        
    except sqlite3.Error as db_error:
        logger.error(f"Database error in drawdown chart: {str(db_error)}")
        fig = go.Figure()
        fig.add_annotation(
            text="Database error loading drawdown data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title="Drawdown Analysis (Error)",
            xaxis_title="Time",
            yaxis_title="Drawdown (%)",
            template="plotly_dark",
            height=400
        )
        return fig
        
    except Exception as e:
        logger.error(f"Error updating drawdown chart: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return error chart
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading chart: {str(e)[:50]}...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title="Drawdown Analysis (Error)",
            xaxis_title="Time",
            yaxis_title="Drawdown (%)",
            template="plotly_dark",
            height=400
        )
        return fig

# Add callback for analysis stats
@app.callback(
    Output('analysis-stats', 'children'),
    [Input('analysis-symbol-dropdown', 'value')],
    prevent_initial_call=True
)
def update_analysis_stats(selected_symbol):
    """Update analysis statistics"""
    try:
        if not selected_symbol:
            return html.Div("Select a symbol to view statistics", 
                          style={'color': '#2c3e50', 'textAlign': 'center', 'padding': '20px'})
        
        # Get recent data for analysis
        df = get_coinbase_data(selected_symbol, granularity=3600, days=7)
        
        if df.empty:
            return html.Div(f"No data available for {selected_symbol}",
                          style={'color': '#f44336', 'textAlign': 'center', 'padding': '20px'})
        
        # Calculate statistics
        df = calculate_indicators(df)
        latest = df.iloc[-1] if not df.empty else None
        
        if latest is None:
            return html.Div("Error calculating statistics",
                          style={'color': '#f44336', 'textAlign': 'center', 'padding': '20px'})
        
        # Price statistics
        price_change_24h = ((latest['close'] / df['close'].iloc[0]) - 1) * 100 if len(df) > 1 else 0
        volatility = df['close'].pct_change().std() * 100 if len(df) > 1 else 0
        
        # Technical indicators
        rsi = latest.get('rsi', 0)
        macd = latest.get('macd', 0)
        volume = latest.get('volume', 0)
        
        # Create stats display
        stats_content = html.Div([
            html.H4(f"Analysis for {selected_symbol}", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            
            html.Div([
                html.Div([
                    html.H5("Price Statistics", style={'color': '#4CAF50'}),
                    html.P(f"Current Price: ${latest['close']:.4f}"),
                    html.P(f"24h Change: {price_change_24h:+.2f}%", 
                          style={'color': '#4CAF50' if price_change_24h > 0 else '#f44336'}),
                    html.P(f"Volatility: {volatility:.2f}%"),
                ], style={'flex': '1', 'padding': '10px'}),
                
                html.Div([
                    html.H5("Technical Indicators", style={'color': '#2196F3'}),
                    html.P(f"RSI: {rsi:.1f}"),
                    html.P(f"MACD: {macd:.4f}"),
                    html.P(f"Volume: {volume:,.0f}"),
                ], style={'flex': '1', 'padding': '10px'}),
                
            ], style={'display': 'flex', 'justifyContent': 'space-between'})
            
        ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '5px'})
        
        return stats_content
        
    except Exception as e:
        logger.error(f"Error updating analysis stats: {str(e)}")
        return html.Div(f"Error loading statistics: {str(e)[:100]}...",
                       style={'color': '#f44336', 'textAlign': 'center', 'padding': '20px'})

# Add callback for live performance metrics
@app.callback(
    Output('live-performance-metrics', 'children'),
    [Input('live-trading-interval', 'n_intervals')],
    prevent_initial_call=False
)
def update_live_performance_metrics(n_intervals):
    """Update live trading performance metrics"""
    try:
        # Get current portfolio value
        current_value = get_portfolio_value()
        available_balance = get_available_balance()
        
        # Get session information
        session_id = get_active_session()
        initial_balance = 0.0
        
        if session_id:
            try:
                conn = sqlite3.connect(DB_PATH, timeout=10)
                cursor = conn.cursor()
                cursor.execute('SELECT initial_balance FROM sessions WHERE id = ?', (session_id,))
                result = cursor.fetchone()
                if result:
                    initial_balance = float(result[0])
                conn.close()
            except Exception as db_error:
                logger.error(f"Database error getting session info: {str(db_error)}")
        
        # Calculate performance metrics
        if initial_balance > 0:
            total_return = ((current_value / initial_balance) - 1) * 100
        else:
            total_return = 0.0
        
        # Get position count
        positions = get_live_positions_from_lk(min_usd_value=0.50)
        position_count = len([p for p in positions if p.get('currency') != 'USD'])
        
        # Create metrics display
        metrics_content = html.Div([
            html.Div([
                html.Div([
                    html.H4(f"${current_value:.2f}", style={'color': '#4CAF50', 'margin': '0'}),
                    html.P("Total Portfolio Value", style={'margin': '5px 0', 'color': '#666'})
                ], style={'textAlign': 'center', 'flex': '1'}),
                
                html.Div([
                    html.H4(f"${available_balance:.2f}", style={'color': '#2196F3', 'margin': '0'}),
                    html.P("Available Balance", style={'margin': '5px 0', 'color': '#666'})
                ], style={'textAlign': 'center', 'flex': '1'}),
                
                html.Div([
                    html.H4(f"{total_return:+.2f}%", 
                           style={'color': '#4CAF50' if total_return >= 0 else '#f44336', 'margin': '0'}),
                    html.P("Total Return", style={'margin': '5px 0', 'color': '#666'})
                ], style={'textAlign': 'center', 'flex': '1'}),
                
                html.Div([
                    html.H4(f"{position_count}", style={'color': '#FF9800', 'margin': '0'}),
                    html.P("Active Positions", style={'margin': '5px 0', 'color': '#666'})
                ], style={'textAlign': 'center', 'flex': '1'}),
                
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),
            
            html.Hr(style={'margin': '10px 0'}),
            
            html.P(f"Session ID: {session_id if session_id else 'N/A'}", 
                  style={'color': '#666', 'fontSize': '12px', 'textAlign': 'center', 'margin': '0'})
        ])
        
        return metrics_content
        
    except Exception as e:
        logger.error(f"Error updating performance metrics: {str(e)}")
        return html.Div([
            html.P("Error loading performance metrics", style={'color': '#f44336', 'textAlign': 'center'}),
            html.P(f"Error: {str(e)[:100]}...", style={'color': '#666', 'fontSize': '12px', 'textAlign': 'center'})
        ])

# Add callback for recent trades table
@app.callback(
    Output('live-trades-table', 'data'),
    [Input('live-trading-interval', 'n_intervals')],
    prevent_initial_call=False
)
def update_live_trades_table(n_intervals):
    """Update the recent trades table"""
    try:
        session_id = get_active_session()
        if not session_id:
            return []
        
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()
        
        # Get recent trades
        cursor.execute('''
            SELECT timestamp, symbol, action, price, quantity, value
            FROM trades 
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT 20
        ''', (session_id,))
        
        trades_data = cursor.fetchall()
        conn.close()
        
        # Format trades for table display
        table_data = []
        for trade in trades_data:
            table_data.append({
                'timestamp': trade[0],
                'symbol': trade[1],
                'action': trade[2],
                'price': f"${float(trade[3]):.4f}",
                'quantity': f"{float(trade[4]):.8f}",
                'value': f"${float(trade[5]):.2f}"
            })
        
        return table_data
        
    except Exception as e:
        logger.error(f"Error updating trades table: {str(e)}")
        return []

# ... existing code ...

# Add main function and app runner at the end of the file

def main():
    """Main function to run the trading dashboard"""
    try:
        logger.info("🚀 Starting Crypto Trading Dashboard...")
        
        # Ensure all required directories exist
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        # Initialize database
        init_db()
        
        # Run the Dash app
        logger.info("🌐 Starting web server on http://127.0.0.1:8050")
        app.run(
            debug=False,  # Set to False for production
            host='127.0.0.1',
            port=8050,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"❌ Error starting application: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()

def get_multi_timeframe_predictions(symbol):
    """Get detailed predictions for all timeframes with comprehensive analysis"""
    try:
        # COINBASE-SUPPORTED GRANULARITIES
        coinbase_granularities = {
            '15m': 900,      # 15 minutes - SUPPORTED
            '1h': 3600,      # 1 hour - SUPPORTED  
            '6h': 21600,     # 6 hours - SUPPORTED
            '1d': 86400      # 1 day (24 hours) - SUPPORTED
        }
        
        # Create composite predictions for non-supported timeframes
        composite_timeframes = {
            '30m': ['15m'],           # 30m = Enhanced 15m analysis
            '4h': ['1h', '6h'],       # 4h = interpolate between 1h and 6h
            '12h': ['6h', '1d'],      # 12h = interpolate between 6h and 1d  
            '48h': ['1d']             # 48h = Extended 1d analysis
        }
        
        timeframe_predictions = {}
        current_price = None
        
        logger.info(f"🔍 Getting multi-timeframe predictions for {symbol}...")
        
        # Get base predictions from supported granularities
        base_predictions = {}
        for timeframe_name, gran in coinbase_granularities.items():
            try:
                prediction, confidence = get_price_prediction_for_granularity(symbol, gran)
                
                if prediction is not None:
                    if current_price is None:
                        current_price = prediction['current_price']
                    
                    base_predictions[timeframe_name] = {
                        'prediction': prediction,
                        'confidence': confidence
                    }
                    
                    timeframe_predictions[timeframe_name] = {
                        'timeframe': timeframe_name,
                        'granularity_seconds': gran,
                        'direction': prediction['direction'],
                        'confidence': confidence,
                        'current_price': prediction['current_price'],
                        'predicted_price': prediction['predicted_price'],
                        'price_change_pct': prediction['price_change_pct'],
                        'price_change_usd': prediction['predicted_price'] - prediction['current_price'],
                        'prediction_horizon_hours': prediction.get('prediction_horizon_hours', gran/3600),
                        'model_accuracy': prediction.get('model_accuracy', 0.5),
                        'model_mae': prediction.get('model_mae', 0.01),
                        'status': 'SUCCESS',
                        'type': 'DIRECT'
                    }
                    
                    logger.info(f"   ✅ {timeframe_name}: {prediction['direction']} | ${prediction['predicted_price']:.4f} ({prediction['price_change_pct']*100:+.2f}%) | Confidence: {confidence:.1%}")
                else:
                    timeframe_predictions[timeframe_name] = {
                        'timeframe': timeframe_name,
                        'granularity_seconds': gran,
                        'direction': 'NO_PREDICTION',
                        'confidence': 0.0,
                        'current_price': current_price or 0.0,
                        'predicted_price': current_price or 0.0,
                        'price_change_pct': 0.0,
                        'price_change_usd': 0.0,
                        'prediction_horizon_hours': gran/3600,
                        'model_accuracy': 0.0,
                        'model_mae': 0.0,
                        'status': 'NO_MODEL',
                        'type': 'DIRECT'
                    }
                    logger.warning(f"   ❌ {timeframe_name}: No prediction available")
                    
            except Exception as e:
                timeframe_predictions[timeframe_name] = {
                    'timeframe': timeframe_name,
                    'granularity_seconds': gran,
                    'direction': 'ERROR',
                    'confidence': 0.0,
                    'current_price': current_price or 0.0,
                    'predicted_price': current_price or 0.0,
                    'price_change_pct': 0.0,
                    'price_change_usd': 0.0,
                    'prediction_horizon_hours': gran/3600,
                    'model_accuracy': 0.0,
                    'model_mae': 0.0,
                    'status': f'ERROR: {str(e)[:50]}...',
                    'type': 'DIRECT'
                }
                logger.error(f"   💥 {timeframe_name}: {str(e)}")
        
        # Create composite predictions
        for composite_tf, source_timeframes in composite_timeframes.items():
            try:
                composite_prediction = create_composite_prediction(symbol, composite_tf, source_timeframes, base_predictions)
                
                if composite_prediction is not None:
                    timeframe_predictions[composite_tf] = {
                        'timeframe': composite_tf,
                        'granularity_seconds': get_composite_granularity(composite_tf),
                        'direction': composite_prediction['direction'],
                        'confidence': composite_prediction['confidence'],
                        'current_price': composite_prediction['current_price'],
                        'predicted_price': composite_prediction['predicted_price'],
                        'price_change_pct': composite_prediction['price_change_pct'],
                        'price_change_usd': composite_prediction['predicted_price'] - composite_prediction['current_price'],
                        'prediction_horizon_hours': composite_prediction.get('prediction_horizon_hours', get_composite_granularity(composite_tf)/3600),
                        'model_accuracy': 0.75,  # Composite predictions are generally good
                        'model_mae': 0.01,
                        'status': f'COMPOSITE (from {composite_prediction.get("composite_source", source_timeframes)})',
                        'type': 'COMPOSITE'
                    }
                    logger.info(f"   ✅ {composite_tf} (composite): {composite_prediction['direction']} | ${composite_prediction['predicted_price']:.4f} ({composite_prediction['price_change_pct']*100:+.2f}%) | Confidence: {composite_prediction['confidence']:.1%}")
                else:
                    timeframe_predictions[composite_tf] = {
                        'timeframe': composite_tf,
                        'granularity_seconds': get_composite_granularity(composite_tf),
                        'direction': 'NO_PREDICTION',
                        'confidence': 0.0,
                        'current_price': current_price or 0.0,
                        'predicted_price': current_price or 0.0,
                        'price_change_pct': 0.0,
                        'price_change_usd': 0.0,
                        'prediction_horizon_hours': get_composite_granularity(composite_tf)/3600,
                        'model_accuracy': 0.0,
                        'model_mae': 0.0,
                        'status': 'NO_COMPOSITE_SOURCES',
                        'type': 'COMPOSITE'
                    }
                    logger.warning(f"   ❌ {composite_tf} (composite): No sources available")
                    
            except Exception as e:
                timeframe_predictions[composite_tf] = {
                    'timeframe': composite_tf,
                    'granularity_seconds': get_composite_granularity(composite_tf),
                    'direction': 'ERROR',
                    'confidence': 0.0,
                    'current_price': current_price or 0.0,
                    'predicted_price': current_price or 0.0,
                    'price_change_pct': 0.0,
                    'price_change_usd': 0.0,
                    'prediction_horizon_hours': get_composite_granularity(composite_tf)/3600,
                    'model_accuracy': 0.0,
                    'model_mae': 0.0,
                    'status': f'COMPOSITE_ERROR: {str(e)[:50]}...',
                    'type': 'COMPOSITE'
                }
                logger.error(f"   💥 {composite_tf} (composite): {str(e)}")
        
        # Calculate summary statistics
        buy_signals = sum(1 for p in timeframe_predictions.values() if p['direction'] == 'BUY')
        sell_signals = sum(1 for p in timeframe_predictions.values() if p['direction'] == 'SELL')
        hold_signals = sum(1 for p in timeframe_predictions.values() if p['direction'] == 'HOLD')
        
        avg_confidence = sum(p['confidence'] for p in timeframe_predictions.values()) / len(timeframe_predictions)
        
        # Calculate weighted consensus
        timeframe_weights = {
            '15m': 0.5, '30m': 0.7, '1h': 1.0, '4h': 1.2,
            '6h': 1.1, '12h': 0.9, '1d': 0.8, '48h': 0.6
        }
        
        weighted_bullishness = 0
        total_weight = 0
        
        for tf_name, pred in timeframe_predictions.items():
            if pred['direction'] in ['BUY', 'SELL', 'HOLD']:
                weight = timeframe_weights.get(tf_name, 0.5) * pred['confidence']
                
                # Reduce weight for composite predictions
                if pred['type'] == 'COMPOSITE':
                    weight *= 0.8
                
                if pred['direction'] == 'BUY':
                    weighted_bullishness += weight
                elif pred['direction'] == 'SELL':
                    weighted_bullishness -= weight
                # HOLD contributes 0
                
                total_weight += weight
        
        consensus_score = weighted_bullishness / total_weight if total_weight > 0 else 0
        
        if consensus_score > 0.3:
            overall_consensus = 'BULLISH'
        elif consensus_score < -0.3:
            overall_consensus = 'BEARISH'
        else:
            overall_consensus = 'NEUTRAL'
        
        summary = {
            'symbol': symbol,
            'current_price': current_price or 0.0,
            'total_timeframes': len(timeframe_predictions),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'avg_confidence': avg_confidence,
            'consensus_score': consensus_score,
            'overall_consensus': overall_consensus,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"📊 {symbol} Multi-TF Summary: {overall_consensus} consensus ({consensus_score:.2f}) | BUY: {buy_signals}, SELL: {sell_signals}, HOLD: {hold_signals}")
        
        return {
            'predictions': timeframe_predictions,
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"💥 Error getting multi-timeframe predictions for {symbol}: {str(e)}")
        return {
            'predictions': {},
            'summary': {
                'symbol': symbol,
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }

def make_enhanced_ml_decision(symbol, granularity=3600, investment_amount=100.0):
    """
    Enhanced ML Decision System that provides:
    1. Confidence scores for direction
    2. Specific price predictions
    3. Profit/loss estimation with fees
    4. Risk/reward analysis
    """
    try:
        logger.info(f"🎯 Enhanced ML Analysis for {symbol} with ${investment_amount:.2f} investment")
        
        # Get multi-timeframe predictions
        coinbase_granularities = {
            '15m': 900,      # 15 minutes - SUPPORTED
            '1h': 3600,      # 1 hour - SUPPORTED  
            '6h': 21600,     # 6 hours - SUPPORTED
            '1d': 86400      # 1 day (24 hours) - SUPPORTED
        }
        
        timeframe_predictions = {}
        
        for timeframe_name, gran in coinbase_granularities.items():
            try:
                prediction_data, confidence = get_price_prediction_for_granularity(symbol, gran)
                if prediction_data and confidence > 0.5:  # Only use confident predictions
                    timeframe_predictions[timeframe_name] = {
                        'prediction': prediction_data,
                        'confidence': confidence,
                        'granularity': gran
                    }
                    logger.info(f"   📊 {timeframe_name}: {prediction_data['direction']} | "
                               f"Target: ${prediction_data['predicted_price']:.4f} | "
                               f"Change: {prediction_data['price_change_pct']*100:+.2f}% | "
                               f"Confidence: {confidence:.1%}")
            except Exception as e:
                logger.warning(f"   ❌ {timeframe_name} prediction failed: {str(e)}")
                continue
        
        if not timeframe_predictions:
            logger.warning(f"❌ No confident predictions available for {symbol}")
            return create_hold_decision(symbol, investment_amount)
        
        # Analyze consensus and create comprehensive decision
        enhanced_decision = analyze_enhanced_consensus(symbol, timeframe_predictions, investment_amount)
        
        logger.info(f"🎯 Enhanced Decision for {symbol}:")
        logger.info(f"   📊 Action: {enhanced_decision['action']}")
        logger.info(f"   🎯 Confidence: {enhanced_decision['overall_confidence']:.1%}")
        logger.info(f"   💰 Target Price: ${enhanced_decision['target_price']:.4f}")
        logger.info(f"   📈 Expected Profit: {enhanced_decision['expected_profit_pct']:+.2f}%")
        logger.info(f"   💵 Expected Profit $: ${enhanced_decision['expected_profit_usd']:+.2f}")
        logger.info(f"   ⚖️ Risk/Reward: 1:{enhanced_decision['risk_reward_ratio']:.1f}")
        
        return enhanced_decision
        
    except Exception as e:
        logger.error(f"❌ Enhanced ML decision error for {symbol}: {str(e)}")
        return create_hold_decision(symbol, investment_amount, error=str(e))

def analyze_enhanced_consensus(symbol, timeframe_predictions, investment_amount):
    """
    Analyze multi-timeframe predictions to create enhanced trading decision
    with confidence, price targets, and profit estimation
    """
    try:
        # Get current price
        current_price = None
        for tf_data in timeframe_predictions.values():
            if tf_data['prediction']['current_price']:
                current_price = tf_data['prediction']['current_price']
                break
        
        if not current_price:
            return create_hold_decision(symbol, investment_amount, error="No current price available")
        
        # Analyze buy/sell signals by timeframe
        buy_signals = []
        sell_signals = []
        
        for timeframe, tf_data in timeframe_predictions.items():
            pred = tf_data['prediction']
            conf = tf_data['confidence']
            
            # Check for BUY/SELL signals with lower confidence requirements
            if pred['direction'] == 'BUY' and conf >= 0.4:  # Lowered from 0.6 to 0.4
                buy_signals.append({
                    'timeframe': timeframe,
                    'confidence': conf,
                    'predicted_price': pred['predicted_price'],
                    'price_change_pct': pred['price_change_pct'],
                    'horizon_hours': pred['prediction_horizon_hours']
                })
            elif pred['direction'] == 'SELL' and conf >= 0.4:  # Lowered from 0.6 to 0.4
                sell_signals.append({
                    'timeframe': timeframe,
                    'confidence': conf,
                    'predicted_price': pred['predicted_price'],
                    'price_change_pct': pred['price_change_pct'],
                    'horizon_hours': pred['prediction_horizon_hours']
                })
        
        # Determine overall action with more sensitive requirements
        if len(buy_signals) >= 1 or (len(buy_signals) == 1 and buy_signals[0]['confidence'] >= 0.5):  # More lenient
            # Buy signal
            action = 'BUY'
            relevant_signals = buy_signals
        elif len(sell_signals) >= 1 or (len(sell_signals) == 1 and sell_signals[0]['confidence'] >= 0.5):  # More lenient
            # Sell signal  
            action = 'SELL'
            relevant_signals = sell_signals
        else:
            # No clear consensus
            return create_hold_decision(symbol, investment_amount, 
                                      reason=f"Mixed signals: {len(buy_signals)} BUY, {len(sell_signals)} SELL")
        
        # Calculate weighted consensus predictions
        total_weight = 0
        weighted_price_target = 0
        weighted_confidence = 0
        weighted_price_change = 0
        weighted_horizon = 0
        
        for signal in relevant_signals:
            # Weight by confidence and inverse time horizon (shorter term = higher weight)
            time_weight = 1.0 / (signal['horizon_hours'] / 24 + 0.1)  # Shorter timeframes get more weight
            weight = signal['confidence'] * time_weight
            
            weighted_price_target += signal['predicted_price'] * weight
            weighted_confidence += signal['confidence'] * weight
            weighted_price_change += signal['price_change_pct'] * weight
            weighted_horizon += signal['horizon_hours'] * weight
            total_weight += weight
        
        # Normalize weighted values
        target_price = weighted_price_target / total_weight
        overall_confidence = weighted_confidence / total_weight
        expected_price_change_pct = weighted_price_change / total_weight
        expected_horizon_hours = weighted_horizon / total_weight
        
        # Calculate profit estimation with fees
        profit_analysis = calculate_detailed_profit_estimation(
            symbol=symbol,
            action=action,
            current_price=current_price,
            target_price=target_price,
            investment_amount=investment_amount,
            expected_horizon_hours=expected_horizon_hours,
            confidence=overall_confidence
        )
        
        # Create enhanced decision
        enhanced_decision = {
            'symbol': symbol,
            'action': action,
            'overall_confidence': overall_confidence,
            'current_price': current_price,
            'target_price': target_price,
            'expected_price_change_pct': expected_price_change_pct * 100,  # Convert to percentage
            'expected_horizon_hours': expected_horizon_hours,
            'investment_amount': investment_amount,
            
            # Profit estimation
            'expected_profit_pct': profit_analysis['net_profit_pct'],
            'expected_profit_usd': profit_analysis['net_profit_usd'],
            'gross_profit_pct': profit_analysis['gross_profit_pct'],
            'gross_profit_usd': profit_analysis['gross_profit_usd'],
            'total_fees_usd': profit_analysis['total_fees'],
            
            # Risk analysis
            'risk_reward_ratio': profit_analysis['risk_reward_ratio'],
            'profit_probability': profit_analysis['profit_probability'],
            'breakeven_price': profit_analysis['breakeven_price'],
            
            # TP/SL recommendations
            'recommended_tp_price': profit_analysis['recommended_tp'],
            'recommended_sl_price': profit_analysis['recommended_sl'],
            'recommended_tp_pct': profit_analysis['recommended_tp_pct'],
            'recommended_sl_pct': profit_analysis['recommended_sl_pct'],
            
            # Supporting data
            'timeframe_count': len(relevant_signals),
            'timeframes': [s['timeframe'] for s in relevant_signals],
            'signal_strength': sum(s['confidence'] for s in relevant_signals) / len(relevant_signals),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return enhanced_decision
        
    except Exception as e:
        logger.error(f"❌ Error analyzing enhanced consensus: {str(e)}")
        return create_hold_decision(symbol, investment_amount, error=str(e))

def calculate_detailed_profit_estimation(symbol, action, current_price, target_price, 
                                       investment_amount, expected_horizon_hours, confidence):
    """
    Calculate detailed profit estimation including fees, risk analysis, and TP/SL recommendations
    """
    try:
        # Coinbase trading fees
        maker_fee_pct = 0.35  # 0.35% for maker orders
        taker_fee_pct = 0.6   # 0.6% for taker orders (assume worst case)
        
        # Calculate quantity
        quantity = investment_amount / current_price
        
        # Gross profit calculation
        if action == 'BUY':
            gross_profit_usd = (target_price - current_price) * quantity
            gross_profit_pct = ((target_price - current_price) / current_price) * 100
        else:  # SELL
            gross_profit_usd = (current_price - target_price) * quantity
            gross_profit_pct = ((current_price - target_price) / current_price) * 100
        
        # Fee calculation
        buy_fee = investment_amount * (taker_fee_pct / 100)  # Fee on buy
        sell_value = investment_amount + gross_profit_usd
        sell_fee = sell_value * (maker_fee_pct / 100)  # Fee on sell
        total_fees = buy_fee + sell_fee
        
        # Net profit after fees
        net_profit_usd = gross_profit_usd - total_fees
        net_profit_pct = (net_profit_usd / investment_amount) * 100
        
        # Breakeven price (price needed to break even after fees)
        breakeven_fee_pct = (maker_fee_pct + taker_fee_pct) / 100
        if action == 'BUY':
            breakeven_price = current_price * (1 + breakeven_fee_pct)
        else:
            breakeven_price = current_price * (1 - breakeven_fee_pct)
        
        # Risk/Reward ratio estimation
        # Risk: potential loss if price moves against us by same amount
        risk_price = current_price - (target_price - current_price) if action == 'BUY' else current_price + (current_price - target_price)
        potential_loss = abs((risk_price - current_price) * quantity) + total_fees
        risk_reward_ratio = abs(net_profit_usd) / potential_loss if potential_loss > 0 else 0
        
        # Profit probability based on confidence and market conditions
        base_probability = confidence * 0.8  # Conservative adjustment
        
        # Adjust based on expected return magnitude (smaller returns more likely)
        return_magnitude_factor = max(0.5, 1 - (abs(gross_profit_pct) / 20))  # Reduce probability for >20% expected returns
        profit_probability = base_probability * return_magnitude_factor
        
        # TP/SL recommendations based on timeframe and confidence
        if action == 'BUY':
            # Take Profit: Use predicted price or 80% of way there (conservative)
            recommended_tp = current_price + (target_price - current_price) * 0.8
            recommended_tp_pct = ((recommended_tp - current_price) / current_price) * 100
            
            # Stop Loss: Based on expected horizon and volatility
            if expected_horizon_hours <= 4:  # Short term
                stop_loss_pct = 2.0  # 2% stop loss
            elif expected_horizon_hours <= 24:  # Medium term
                stop_loss_pct = 3.5  # 3.5% stop loss
            else:  # Long term
                stop_loss_pct = 5.0  # 5% stop loss
            
            recommended_sl = current_price * (1 - stop_loss_pct / 100)
            recommended_sl_pct = -stop_loss_pct
            
        else:  # SELL
            recommended_tp = current_price - (current_price - target_price) * 0.8
            recommended_tp_pct = ((recommended_tp - current_price) / current_price) * 100
            
            if expected_horizon_hours <= 4:
                stop_loss_pct = 2.0
            elif expected_horizon_hours <= 24:
                stop_loss_pct = 3.5
            else:
                stop_loss_pct = 5.0
            
            recommended_sl = current_price * (1 + stop_loss_pct / 100)
            recommended_sl_pct = stop_loss_pct
        
        return {
            'gross_profit_usd': gross_profit_usd,
            'gross_profit_pct': gross_profit_pct,
            'net_profit_usd': net_profit_usd,
            'net_profit_pct': net_profit_pct,
            'total_fees': total_fees,
            'buy_fee': buy_fee,
            'sell_fee': sell_fee,
            'breakeven_price': breakeven_price,
            'risk_reward_ratio': risk_reward_ratio,
            'profit_probability': profit_probability,
            'recommended_tp': recommended_tp,
            'recommended_sl': recommended_sl,
            'recommended_tp_pct': recommended_tp_pct,
            'recommended_sl_pct': recommended_sl_pct,
            'quantity': quantity
        }
        
    except Exception as e:
        logger.error(f"❌ Error calculating profit estimation: {str(e)}")
        return {
            'gross_profit_usd': 0, 'gross_profit_pct': 0, 'net_profit_usd': 0, 'net_profit_pct': 0,
            'total_fees': 0, 'buy_fee': 0, 'sell_fee': 0, 'breakeven_price': current_price,
            'risk_reward_ratio': 0, 'profit_probability': 0, 'recommended_tp': current_price,
            'recommended_sl': current_price, 'recommended_tp_pct': 0, 'recommended_sl_pct': 0,
            'quantity': 0
        }

def create_hold_decision(symbol, investment_amount, reason="No clear signals", error=None):
    """Create a HOLD decision with basic structure"""
    try:
        # Try to get current price
        current_price = 0
        try:
            ticker = client.get_product(product_id=symbol)
            current_price = float(ticker['price'])
        except:
            current_price = 0
        
        return {
            'symbol': symbol,
            'action': 'HOLD',
            'overall_confidence': 0.0,
            'current_price': current_price,
            'target_price': current_price,
            'expected_price_change_pct': 0.0,
            'expected_horizon_hours': 0.0,
            'investment_amount': investment_amount,
            'expected_profit_pct': 0.0,
            'expected_profit_usd': 0.0,
            'gross_profit_pct': 0.0,
            'gross_profit_usd': 0.0,
            'total_fees_usd': 0.0,
            'risk_reward_ratio': 0.0,
            'profit_probability': 0.0,
            'breakeven_price': current_price,
            'recommended_tp_price': current_price,
            'recommended_sl_price': current_price,
            'recommended_tp_pct': 0.0,
            'recommended_sl_pct': 0.0,
            'timeframe_count': 0,
            'timeframes': [],
            'signal_strength': 0.0,
            'reason': reason,
            'error': error,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error creating hold decision: {str(e)}")
        return {'action': 'HOLD', 'error': str(e)}

# ... existing code ...

# === UNIFIED INTELLIGENT TRADING LOOP ===
def unified_trading_loop():
    """
    Unified intelligent trading loop combining:
    - Fast position monitoring (every iteration)
    - Enhanced ML analysis (periodic)
    - Smart entry signal detection
    - Consistent ML logic throughout
    """
    try:
        logger.info("🚀 Starting unified intelligent trading loop...")
        
        # Initialize client
        client = RESTClient(
            api_key=KEY_NAME,
            api_secret=PRIVATE_KEY_PEM,
            base_url=BASE_URL
        )
        
        # Load trading configuration
        try:
            from trading_config import get_trading_config
            config = get_trading_config()
            logger.info("✅ Loaded centralized trading configuration")
        except ImportError:
            # Fallback to inline configuration
            config = {
                'intervals': {
                    'enhanced_analysis': 300,  # 5 minutes
                    'position_check': 30,     # 30 seconds
                    'market_scan': 600,       # 10 minutes
                },
                'thresholds': {
                    'min_confidence': 0.6,
                    'min_profit_expectation': 2.0,
                    'min_profit_probability': 0.6,
                },
                'position_management': {
                    'max_position_size': 100.0,
                    'max_total_positions': 5,
                    'min_available_balance': 10.0,
                },
                'symbols': ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD", "MATIC-USD", "LINK-USD"]
            }
            logger.info("⚠️ Using fallback configuration")
        
        # Timing trackers
        last_enhanced_analysis = 0
        last_position_update = 0
        last_market_scan = 0
        
        logger.info(f"📊 Trading Config:")
        logger.info(f"   Enhanced Analysis: Every {config['intervals']['enhanced_analysis']}s")
        logger.info(f"   Position Updates: Every {config['intervals']['position_check']}s") 
        logger.info(f"   Market Scanning: Every {config['intervals']['market_scan']}s")
        logger.info(f"   Min Confidence: {config['thresholds']['min_confidence']:.1%}")
        logger.info(f"   Min Profit: {config['thresholds']['min_profit_expectation']:.1f}%")
        
        while True:
            try:
                if not ws_client or not ws_client.is_running():
                    logger.warning("WebSocket client not running, skipping iteration")
                    time.sleep(5)
                    continue
                
                current_time = time.time()
                session_id = get_active_session()
                
                if not session_id:
                    logger.warning("No active session found, skipping trading iteration")
                    time.sleep(30)
                    continue
                
                # ==========================================
                # 1. FAST POSITION MONITORING (Every 30s)
                # ==========================================
                if current_time - last_position_update > config['intervals']['position_check']:
                    try:
                        logger.debug("🔄 Updating positions and checking TP/SL...")
                        position_manager.refresh_positions()
                        position_manager.update_position_prices()
                        
                        # Enhanced TP/SL checking with ML insights
                        check_positions_with_enhanced_ml()
                        
                        update_portfolio_history()
                        last_position_update = current_time
                        
                    except Exception as e:
                        logger.error(f"❌ Error in position monitoring: {str(e)}")
                
                # =============================================
                # 2. ENHANCED ML ANALYSIS (Every 5 minutes)
                # =============================================
                if current_time - last_enhanced_analysis > config['intervals']['enhanced_analysis']:
                    try:
                        logger.info("🧠 Running enhanced ML analysis on existing positions...")
                        
                        positions = position_manager.positions
                        for symbol in positions:
                            analyze_position_with_enhanced_ml(symbol, config)
                        
                        last_enhanced_analysis = current_time
                        
                    except Exception as e:
                        logger.error(f"❌ Error in enhanced ML analysis: {str(e)}")
                
                # =============================================
                # 3. MARKET SCANNING (Every 10 minutes)
                # =============================================
                if current_time - last_market_scan > config['intervals']['market_scan']:
                    try:
                        logger.info("🔍 Scanning market for new opportunities...")
                        
                        # Check if we can take new positions
                        current_positions = len(position_manager.positions)
                        available_balance = get_available_balance()
                        
                        if (current_positions < config['position_management']['max_total_positions'] and 
                            available_balance >= config['position_management']['min_available_balance']):  # Need at least $10
                            
                            scan_for_trading_opportunities(config)
                        else:
                            logger.info(f"📊 Skipping market scan: {current_positions}/{config['position_management']['max_total_positions']} positions, ${available_balance:.2f} balance")
                        
                        last_market_scan = current_time
                        
                    except Exception as e:
                        logger.error(f"❌ Error in market scanning: {str(e)}")
                
                # Short sleep to avoid busy waiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error in unified trading loop iteration: {str(e)}")
                time.sleep(5)
                
    except Exception as e:
        logger.error(f"💥 Fatal error in unified trading loop: {str(e)}")
        raise

def check_positions_with_enhanced_ml():
    """Enhanced position monitoring using ML insights"""
    try:
        positions = position_manager.positions
        
        for symbol, position_data in positions.items():
            try:
                # Get enhanced ML decision for current position
                enhanced_decision = make_enhanced_ml_decision(
                    symbol=symbol,
                    granularity=3600,
                    investment_amount=float(position_data.get('position_size', 50))
                )
                
                if enhanced_decision and enhanced_decision['action'] == 'SELL':
                    confidence = enhanced_decision.get('overall_confidence', 0)
                    expected_profit = enhanced_decision.get('expected_profit_pct', 0)
                    
                    if confidence >= 0.7 and expected_profit >= 1.0:
                        logger.info(f"🚨 Enhanced ML suggests SELL for {symbol}")
                        logger.info(f"   Confidence: {confidence:.1%}")
                        logger.info(f"   Expected Profit: {expected_profit:.2f}%")
                        
                        # Could trigger automatic sell here
                        # execute_enhanced_sell(symbol, enhanced_decision)
                
            except Exception as e:
                logger.error(f"❌ Error analyzing position {symbol}: {str(e)}")
                
    except Exception as e:
        logger.error(f"❌ Error in enhanced position checking: {str(e)}")

def analyze_position_with_enhanced_ml(symbol, config):
    """Analyze existing position with enhanced ML"""
    try:
        logger.debug(f"🔍 Enhanced analysis for position: {symbol}")
        
        # Get enhanced decision
        enhanced_decision = make_enhanced_ml_decision(
            symbol=symbol,
            granularity=3600,
            investment_amount=50.0  # Standard analysis amount
        )
        
        if enhanced_decision:
            action = enhanced_decision['action']
            confidence = enhanced_decision.get('overall_confidence', 0)
            profit_prob = enhanced_decision.get('profit_probability', 0)
            
            logger.debug(f"📊 {symbol}: {action} | Confidence: {confidence:.1%} | Profit Prob: {profit_prob:.1%}")
            
            # Log decision to database for tracking
            log_enhanced_ml_decision(symbol, enhanced_decision)
            
    except Exception as e:
        logger.error(f"❌ Error analyzing position {symbol}: {str(e)}")

def scan_for_trading_opportunities(config):
    """Scan market for new trading opportunities using enhanced ML"""
    try:
        # Determine scanning strategy
        scanning_mode = config.get('scanning_strategy', {}).get('mode', 'TARGETED')
        
        if scanning_mode == "MOMENTUM":
            # MOMENTUM SCANNING: Find top momentum symbols first, then apply ML
            logger.info(f"🚀 MOMENTUM SCANNING: Finding top momentum symbols for ML analysis...")
            
            # Get momentum scan results
            momentum_results = scan_for_crypto_runs(max_pairs=200)  # Scan all available
            
            if not momentum_results:
                logger.warning("❌ No momentum results found")
                return
            
            # Filter by momentum criteria
            momentum_config = config.get('scanning_strategy', {})
            min_momentum_score = momentum_config.get('momentum_min_score', 60)
            min_price_change = momentum_config.get('momentum_price_change_min', 2.0)
            top_count = momentum_config.get('momentum_top_count', 20)
            
            # Filter and rank momentum symbols
            filtered_momentum = []
            for result in momentum_results:
                momentum_score = result.get('momentum_score', 0)
                price_change_pct = abs(result.get('price_change_pct', 0))
                
                if momentum_score >= min_momentum_score and price_change_pct >= min_price_change:
                    filtered_momentum.append(result)
            
            # Sort by momentum score and take top symbols
            filtered_momentum.sort(key=lambda x: x['momentum_score'], reverse=True)
            top_momentum_symbols = [r['symbol'] for r in filtered_momentum[:top_count]]
            
            logger.info(f"⚡ MOMENTUM FILTER RESULTS:")
            logger.info(f"   📊 Total symbols scanned: {len(momentum_results)}")
            logger.info(f"   🔥 High momentum symbols: {len(filtered_momentum)}")
            logger.info(f"   🎯 Top symbols for ML analysis: {len(top_momentum_symbols)}")
            
            if top_momentum_symbols:
                logger.info(f"🏆 TOP {len(top_momentum_symbols)} MOMENTUM SYMBOLS:")
                for i, symbol in enumerate(top_momentum_symbols[:10]):  # Show top 10
                    momentum_data = next((r for r in filtered_momentum if r['symbol'] == symbol), {})
                    logger.info(f"   #{i+1}: {symbol} - Score: {momentum_data.get('momentum_score', 0):.0f}, "
                              f"Change: {momentum_data.get('price_change_pct', 0):+.2f}%")
            
            symbols = top_momentum_symbols
            
        elif scanning_mode == "BROAD":
            # Get ALL available symbols for comprehensive analysis
            all_symbols = get_cached_symbols()
            # Limit for performance but still comprehensive
            symbols_limit = config.get('scanning_strategy', {}).get('broad_symbols_limit', 50)
            symbols = all_symbols[:symbols_limit] if all_symbols else []
            logger.info(f"🌐 BROAD ENHANCED ML SCAN: Deep analysis of {len(symbols)} symbols...")
            logger.info(f"   🎯 Using enhanced ML for ALL symbols (not just curated list)")
            
        elif scanning_mode == "HYBRID":
            # Combine both approaches
            symbols = config['symbols'] + get_cached_symbols()[:10]  # Targeted + top 10 others
            symbols = list(set(symbols))  # Remove duplicates
            logger.info(f"🔄 HYBRID SCAN: Analyzing {len(symbols)} symbols...")
        else:  # TARGETED (default)
            symbols = config['symbols']
            logger.info(f"🎯 TARGETED SCAN: Deep analysis of {len(symbols)} curated symbols...")
        
        opportunities = []
        analyzed_count = 0
        skipped_count = 0
        
        logger.info(f"🔍 Starting enhanced ML analysis across {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols):
            try:
                # Progress indicator for broad scans
                if (scanning_mode in ["BROAD", "MOMENTUM"]) and i % 5 == 0:
                    logger.info(f"📊 Progress: {i}/{len(symbols)} symbols analyzed...")
                
                # Skip if we already have a position
                if symbol in position_manager.positions:
                    skipped_count += 1
                    continue
                
                # Get enhanced ML decision with full analysis
                logger.debug(f"🔍 Enhanced ML analysis: {symbol}")
                enhanced_decision = make_enhanced_ml_decision(
                    symbol=symbol,
                    granularity=3600,
                    investment_amount=config['position_management']['max_position_size']
                )
                
                analyzed_count += 1
                
                if enhanced_decision and enhanced_decision['action'] == 'BUY':
                    confidence = enhanced_decision.get('overall_confidence', 0)
                    expected_profit = enhanced_decision.get('expected_profit_pct', 0)
                    profit_prob = enhanced_decision.get('profit_probability', 0)
                    
                    # Apply thresholds based on scanning mode
                    if scanning_mode in ["BROAD", "MOMENTUM"]:
                        # More lenient thresholds for broad/momentum scanning to catch more opportunities
                        min_confidence = config['thresholds']['min_confidence'] * 0.8  # 80% of normal
                        min_profit = config['thresholds']['min_profit_expectation'] * 0.7  # 70% of normal
                        min_prob = 0.5  # 50% probability threshold
                    else:
                        # Standard thresholds for targeted/hybrid
                        min_confidence = config['thresholds']['min_confidence']
                        min_profit = config['thresholds']['min_profit_expectation']
                        min_prob = 0.6
                    
                    if (confidence >= min_confidence and 
                        expected_profit >= min_profit and
                        profit_prob >= min_prob):
                        
                        # Calculate comprehensive opportunity score
                        score = confidence * expected_profit * profit_prob
                        
                        # Boost score for momentum symbols (they already passed momentum filter)
                        if scanning_mode == "MOMENTUM":
                            score *= 1.2  # 20% momentum bonus
                        
                        opportunities.append({
                            'symbol': symbol,
                            'decision': enhanced_decision,
                            'score': score,
                            'scanning_mode': scanning_mode,
                            'confidence': confidence,
                            'expected_profit': expected_profit,
                            'profit_probability': profit_prob
                        })
                        
                        logger.info(f"💡 {scanning_mode} Opportunity Found: {symbol}")
                        logger.info(f"   🎯 Confidence: {confidence:.1%}")
                        logger.info(f"   📈 Expected Profit: {expected_profit:.2f}%")
                        logger.info(f"   🎲 Success Probability: {profit_prob:.1%}")
                        logger.info(f"   ⭐ Composite Score: {score:.3f}")
                
                # Shorter delay for broad/momentum scanning to speed up analysis
                delay = 0.1 if scanning_mode in ["BROAD", "MOMENTUM"] else 0.2
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"❌ Error analyzing {symbol}: {str(e)}")
                continue
        
        # Comprehensive results summary
        logger.info(f"🏁 {scanning_mode} SCAN COMPLETE:")
        logger.info(f"   📊 Total symbols scanned: {len(symbols)}")
        logger.info(f"   🔍 Symbols analyzed: {analyzed_count}")
        logger.info(f"   ⏭️ Positions skipped: {skipped_count}")
        logger.info(f"   💡 Opportunities found: {len(opportunities)}")
        
        if opportunities:
            # Sort by comprehensive score and show top opportunities
            opportunities.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"🏆 TOP OPPORTUNITIES RANKED BY SCORE:")
            for i, opp in enumerate(opportunities[:5]):  # Show top 5
                logger.info(f"   #{i+1}: {opp['symbol']} - Score: {opp['score']:.3f}")
                logger.info(f"        Confidence: {opp['confidence']:.1%} | Profit: {opp['expected_profit']:.2f}% | Prob: {opp['profit_probability']:.1%}")
            
            # Execute the best opportunity
            best_opportunity = opportunities[0]
            logger.info(f"🚀 Executing BEST {scanning_mode.lower()} opportunity: {best_opportunity['symbol']}")
            logger.info(f"   🎯 Final Score: {best_opportunity['score']:.3f}")
            
            # Execute the trade
            execute_enhanced_trade(best_opportunity)
        else:
            logger.info(f"📊 No profitable {scanning_mode.lower()} opportunities found meeting criteria")
            logger.info(f"   🎯 Criteria: {min_confidence:.1%} confidence, {min_profit:.1f}% profit, {min_prob:.1%} probability")
            
    except Exception as e:
        logger.error(f"❌ Error in {scanning_mode} enhanced ML scanning: {str(e)}")

def execute_enhanced_trade(opportunity):
    """Execute a trade based on enhanced ML analysis"""
    try:
        symbol = opportunity['symbol']
        decision = opportunity['decision']
        
        logger.info(f"🎯 Executing enhanced trade: {symbol}")
        
        # Get current price
        current_price = decision['current_price']
        investment_amount = decision['investment_amount']
        
        # Execute buy order
        success = execute_real_trade(
            symbol=symbol,
            side="buy",
            price=current_price,
            funds=investment_amount
        )
        
        if success:
            logger.info(f"✅ Enhanced trade executed: {symbol}")
            
            # Set up intelligent TP/SL based on enhanced analysis
            tp_price = decision.get('recommended_tp_price')
            sl_price = decision.get('recommended_sl_price')
            
            if tp_price and sl_price:
                setup_intelligent_tp_sl(symbol, current_price, tp_price, sl_price, investment_amount)
            
            # Sync positions
            try:
                live_positions = get_live_positions_from_lk()
                sync_positions_to_db(live_positions)
            except Exception as e:
                logger.warning(f"Position sync failed: {str(e)}")
        else:
            logger.error(f"❌ Enhanced trade execution failed: {symbol}")
            
    except Exception as e:
        logger.error(f"❌ Error executing enhanced trade: {str(e)}")

def setup_intelligent_tp_sl(symbol, entry_price, tp_price, sl_price, position_size):
    """Set up intelligent TP/SL orders based on enhanced ML analysis"""
    try:
        logger.info(f"📋 Setting up intelligent TP/SL for {symbol}")
        logger.info(f"   Entry: ${entry_price:.4f}")
        logger.info(f"   TP: ${tp_price:.4f} (+{((tp_price/entry_price)-1)*100:.2f}%)")
        logger.info(f"   SL: ${sl_price:.4f} ({((sl_price/entry_price)-1)*100:.2f}%)")
        
        quantity = position_size / entry_price
        
        # Place Take Profit order
        try:
            tp_order = client.limit_order_gtc(
                client_order_id=f"TP_{symbol.replace('-', '')}_{int(time.time())}",
                product_id=symbol,
                side="SELL",
                base_size=str(quantity),
                limit_price=str(tp_price)
            )
            logger.info(f"✅ Take Profit order placed")
        except Exception as e:
            logger.error(f"❌ Failed to place TP order: {str(e)}")
        
        # Place Stop Loss order
        try:
            sl_order = client.stop_order_gtc(
                client_order_id=f"SL_{symbol.replace('-', '')}_{int(time.time())}",
                product_id=symbol,
                side="SELL",
                base_size=str(quantity),
                stop_price=str(sl_price)
            )
            logger.info(f"✅ Stop Loss order placed")
        except Exception as e:
            logger.error(f"❌ Failed to place SL order: {str(e)}")
            
    except Exception as e:
        logger.error(f"❌ Error setting up TP/SL: {str(e)}")

def log_enhanced_ml_decision(symbol, decision):
    """Log enhanced ML decision to database for tracking"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO enhanced_ml_decisions 
            (session_id, symbol, action, overall_confidence, expected_profit_pct, 
             expected_profit_usd, profit_probability, risk_reward_ratio, 
             recommended_tp_price, recommended_sl_price, timeframe_count, 
             signal_strength, investment_amount, enhanced_data, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            get_active_session(),
            symbol,
            decision['action'],
            decision.get('overall_confidence', 0),
            decision.get('expected_profit_pct', 0),
            decision.get('expected_profit_usd', 0),
            decision.get('profit_probability', 0),
            decision.get('risk_reward_ratio', 0),
            decision.get('recommended_tp_price'),
            decision.get('recommended_sl_price'),
            decision.get('timeframe_count', 0),
            decision.get('signal_strength', 0),
            decision.get('investment_amount', 0),
            json.dumps(decision),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Error logging enhanced ML decision: {str(e)}")

# Replace the old trading_loop function
trading_loop = unified_trading_loop