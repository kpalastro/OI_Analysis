# Web-based OI Tracker with Real-time Updates
# This script provides a web interface for tracking Option Open Interest changes

import logging
import sys
import time
import pandas as pd
import numpy as np
import math
import signal
import atexit
import os
from datetime import datetime, date, timedelta, timezone, time as dt_time
from threading import Thread, Lock, Event
from typing import Optional
from zoneinfo import ZoneInfo
from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from flask_socketio import SocketIO, emit
from functools import wraps
from urllib.parse import unquote
from kite_trade import *
from kiteconnect import KiteTicker
import database as db
from database import get_previous_close_price
from dotenv import load_dotenv
try:
    from analysis import OIBackAnalysisPipeline
except ImportError:
    OIBackAnalysisPipeline = None

# ML System imports
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ml_system.predictions.realtime_predictor import RealTimePredictor
    from ml_system.predictions.signal_generator import SignalGenerator
    from ml_system.features.feature_engineer import FeatureEngineer
    ML_SYSTEM_AVAILABLE = True
except ImportError as e:
    ML_SYSTEM_AVAILABLE = False
    logging.warning(f"ML System not available: {e}")

# Load environment variables from .env file
load_dotenv()


def _get_env_float(var_name: str, default: float) -> float:
    """Safely parse an environment variable as float with a fallback."""
    value = os.getenv(var_name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# --- Login Credentials (optional - can be entered via login UI) ---
# Note: Credentials can be loaded from .env for convenience, but are not required at startup
# Users can also enter them directly in the login UI
USER_ID = os.getenv('ZERODHA_USER_ID', '')
PASSWORD = os.getenv('ZERODHA_PASSWORD', '')

# TWOFA will be requested at runtime via user input

# --- Exchange Configuration Dictionaries ---
EXCHANGE_CONFIGS = {
    'NSE': {
        'underlying_symbol': 'NIFTY 50',
        'underlying_prefix': 'NIFTY',
        'strike_difference': 50,
        'options_count': 5,
        'options_exchange': 'NFO',
        'ltp_exchange': 'NSE'
    },
    'BSE': {
        'underlying_symbol': 'SENSEX',
        'underlying_prefix': 'SENSEX',
        'strike_difference': 100,
        'options_count': 5,
        'options_exchange': 'BFO',
        'ltp_exchange': 'BSE'
    }
}

# Current active exchange
current_exchange = 'NSE'  # Default to NSE

# --- Data Fetching Parameters ---
HISTORICAL_DATA_MINUTES = 40
OI_CHANGE_INTERVALS_MIN = (3, 5, 10, 15, 30)

# --- Financial Assumptions ---
RISK_FREE_RATE = 0.08  # 8% annual interest rate
MIN_TIME_TO_EXPIRY_SECONDS = 60 * 15  # Floor for time to expiry in seconds (15 minutes)

# --- Price Diff Highlight Thresholds ---
DEFAULT_DIFF_POS_THRESHOLD = 3.0
DEFAULT_DIFF_NEG_THRESHOLD = 2.0
DIFF_POS_THRESHOLD = _get_env_float('DIFF_POS_THRESHOLD', DEFAULT_DIFF_POS_THRESHOLD)
DIFF_NEG_THRESHOLD = _get_env_float('DIFF_NEG_THRESHOLD', DEFAULT_DIFF_NEG_THRESHOLD)
DIFF_THRESHOLDS = {
    'positive': DIFF_POS_THRESHOLD,
    'negative': DIFF_NEG_THRESHOLD
}

# --- Volatility Instruments ---
VIX_TOKEN = None
VIX_FALLBACK_TOKEN = 264969

# --- Display and Logging ---
UI_REFRESH_INTERVAL_SECONDS = 5   # Interval for UI updates
DB_SAVE_INTERVAL_SECONDS = 30     # Interval for persisting to the database
WEBSOCKET_THRESHOLD = 30          # Records per option before relying solely on websocket data
LOG_FILE_NAME = "oi_tracker_web.log"
FILE_LOG_LEVEL = "DEBUG"
PCT_CHANGE_THRESHOLDS = {
    3: 5.0,
    5: 8.0,
    10: 10.0,
    15: 15.0,
    30: 25.0
}

HEATMAP_VALUE_LABELS = {
    'pct_change_3m': 'OI % change (3 min)',
    'pct_change_5m': 'OI % change (5 min)',
    'pct_change_10m': 'OI % change (10 min)',
    'pct_change_15m': 'OI % change (15 min)',
    'pct_change_30m': 'OI % change (30 min)',
    'oi': 'Open Interest',
    'ltp': 'Last Traded Price'
}

# --- Auto Shutdown Configuration ---
AUTO_SHUTDOWN_ENABLED = True
AUTO_SHUTDOWN_IST_HOUR = 15
AUTO_SHUTDOWN_IST_MINUTE = 30
AUTO_SHUTDOWN_CHECK_INTERVAL_SECONDS = 30

# --- Trading Session Window (IST) ---
TRADING_START_IST = dt_time(9, 15)
TRADING_END_IST = dt_time(15, 30)

# ==============================================================================
# --- GLOBAL VARIABLES ---
# ==============================================================================

# Setup logging
logging.basicConfig(
    level=getattr(logging, FILE_LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    filename=LOG_FILE_NAME,
    filemode='a'
)

now = datetime.now()

# Helper function to get config for an exchange
def get_exchange_config(exchange):
    return EXCHANGE_CONFIGS[exchange]

def get_shared_feed_symbol(exchange: str) -> Optional[str]:
    return SHARED_FEED_SYMBOL_MAP.get(exchange)

def _record_tick_metadata(exchange: str, instrument_token: int, event_time: datetime) -> None:
    if exchange not in latest_tick_metadata:
        return
    latest_tick_metadata[exchange][instrument_token] = event_time

def _build_shared_feed_payload(exchange: str, tick: dict, event_time: Optional[datetime]) -> Optional[dict]:
    symbol_code = get_shared_feed_symbol(exchange)
    if not symbol_code:
        return None
    return {
        "symbol": symbol_code,
        "last_price": tick.get('last_price'),
        "oi": tick.get('oi'),
        "volume": tick.get('volume.traded'),
        "timestamp": event_time.isoformat() if event_time else None,
        "source": "oi_tracker_web"
    }

def get_latest_tick_for_symbol(symbol_code: str):
    if not symbol_code:
        return None, None, None

    normalized = symbol_code.strip().upper()
    if ':' not in normalized:
        return None, None, None

    exchange_prefix, symbol_name = normalized.split(':', 1)
    exchange_prefix = exchange_prefix.strip()
    symbol_name = symbol_name.strip()

    exchange_key = None
    for key, config in EXCHANGE_CONFIGS.items():
        if config['ltp_exchange'] == exchange_prefix and config['underlying_symbol'].upper() == symbol_name:
            exchange_key = key
            break

    if not exchange_key:
        return None, None, None

    underlying_token = exchange_instruments[exchange_key].get('underlying_token')
    if underlying_token is None:
        return None, None, None

    with data_lock:
        tick = latest_tick_data[exchange_key].get(underlying_token)
        event_time = latest_tick_metadata[exchange_key].get(underlying_token)
    return tick, event_time, exchange_key


# Global objects
kite = None
kws = None

# Multi-exchange data structures
latest_tick_data = {
    'NSE': {},
    'BSE': {}
}

latest_tick_metadata = {
    'NSE': {},
    'BSE': {}
}

latest_vix_data = {
    'value': None,
    'timestamp': None
}

market_close_processed = {
    'NSE': False,
    'BSE': False
}

shutdown_event = Event()

oi_history = {
    'NSE': {},  # Store OI history from WebSocket ticks: {token: [(timestamp, oi), ...]}
    'BSE': {}
}

ws_connected = False
data_lock = Lock()
cleanup_done = False  # Flag to prevent duplicate cleanup
auto_shutdown_triggered = False

# Currently active instruments and tokens per exchange
exchange_instruments = {
    'NSE': {
        'underlying_token': None,
        'nfo_instruments': [],
        'option_tokens': [],
        'expiry_date': None,
        'symbol_prefix': None
    },
    'BSE': {
        'underlying_token': None,
        'bfo_instruments': [],
        'option_tokens': [],
        'expiry_date': None,
        'symbol_prefix': None
    }
}

# Latest OI data for web display (per exchange)
latest_oi_data = {
    'NSE': {
        'call_options': [],
        'put_options': [],
        'atm_strike': None,
        'underlying_price': None,
        'last_update': None,
        'status': 'Initializing...',
        'pcr': None,
        'exchange': 'NSE',
        'underlying_name': 'NIFTY',
        'strike_difference': 50,
        'ml_prediction': None,
        'ml_signal': None,
        'ml_confidence': None,
        'ml_prediction_pct': None,
        'previous_close': None,
        'previous_close_change': None,
        'previous_close_change_pct': None,
        'vix': None,
        'diff_thresholds': DIFF_THRESHOLDS
    },
    'BSE': {
        'call_options': [],
        'put_options': [],
        'atm_strike': None,
        'underlying_price': None,
        'last_update': None,
        'status': 'Initializing...',
        'pcr': None,
        'exchange': 'BSE',
        'underlying_name': 'SENSEX',
        'strike_difference': 100,
        'ml_prediction': None,
        'ml_signal': None,
        'ml_confidence': None,
        'ml_prediction_pct': None,
        'previous_close': None,
        'previous_close_change': None,
        'previous_close_change_pct': None,
        'vix': None,
        'diff_thresholds': DIFF_THRESHOLDS
    }
}

previous_close_cache = {
    'NSE': {'date': None, 'price': None},
    'BSE': {'date': None, 'price': None}
}

# Previous close helper
def get_cached_previous_close(exchange):
    cache = previous_close_cache.setdefault(exchange, {'date': None, 'price': None})
    today = date.today()

    if cache['date'] != today or cache['price'] is None:
        price = get_previous_close_price(exchange)
        cache['price'] = price
        cache['date'] = today

    return cache['price']

# ML System objects (initialized later)
ml_predictor = None
ml_signal_generator = None
ml_feature_engineer = None
ml_models_loaded = False
analysis_pipeline = None


def get_analysis_pipeline():
    """Lazily initialise analysis pipeline with configured database path."""
    global analysis_pipeline

    if OIBackAnalysisPipeline is None:
        raise RuntimeError("Analysis tools are not available on this deployment.")

    if analysis_pipeline is None:
        db_path = os.getenv("OI_TRACKER_DB_PATH", os.path.join(os.path.dirname(__file__), "oi_tracker.db"))
        analysis_pipeline = OIBackAnalysisPipeline(db_path=db_path)
    return analysis_pipeline

# Paper Trading: Open Positions (per exchange)
# Structure: {exchange: {position_id: {symbol, type, side, entry_price, qty, entry_time, current_price, mtm}}}
open_positions = {
    'NSE': {},
    'BSE': {}
}

position_counter = {
    'NSE': 0,
    'BSE': 0
}

total_mtm = {
    'NSE': 0.0,
    'BSE': 0.0
}

closed_positions_pnl = {
    'NSE': 0.0,
    'BSE': 0.0
}

# Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-change-in-production')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Shared feed defaults
UI_STRINGS = {
    'survivor_feed_namespace': '/survivor_feed',
    'survivor_feed_event': 'tick',
    'survivor_feed_symbol_nifty': 'NSE:NIFTY 50',
    'survivor_feed_symbol_sensex': 'BSE:SENSEX',
    'survivor_feed_error_exchange': 'Unsupported symbol format. Use EXCHANGE:SYMBOL.',
    'survivor_feed_error_not_found': 'Latest price not available for the requested symbol.'
}

SHARED_FEED_NAMESPACE = UI_STRINGS['survivor_feed_namespace']
SHARED_FEED_EVENT = UI_STRINGS['survivor_feed_event']
SHARED_FEED_SYMBOL_MAP = {
    'NSE': UI_STRINGS['survivor_feed_symbol_nifty'],
    'BSE': UI_STRINGS['survivor_feed_symbol_sensex']
}

socketio_stop_lock = Lock()
socketio_stop_requested = False

# Authentication state
authenticated = False
current_enctoken = None

# ==============================================================================
# --- CORE FUNCTIONS (from original script) ---
# ==============================================================================

def get_instrument_token_for_symbol(instruments: list, symbol: str, exchange: str):
    """Gets the instrument token for a given symbol and exchange."""
    try:
        for inst in instruments:
            if inst['tradingsymbol'] == symbol and inst['exchange'] == exchange:
                return inst['instrument_token']
        logging.warning(f"Instrument token not found for {exchange}:{symbol}")
        return None
    except Exception as e:
        logging.error(f"Error finding instrument token for {symbol}: {e}", exc_info=True)
        return None


def _normalize_symbol(symbol: str) -> str:
    return ''.join(ch for ch in symbol.upper() if ch.isalnum()) if symbol else ''


def find_instrument_token_by_aliases(instruments: list, aliases: list, exchange: str):
    normalized_aliases = {_normalize_symbol(alias) for alias in aliases if alias}
    try:
        for inst in instruments:
            if inst.get('exchange') != exchange:
                continue
            tradingsymbol = inst.get('tradingsymbol')
            if _normalize_symbol(tradingsymbol) in normalized_aliases:
                return inst.get('instrument_token')
        return None
    except Exception as e:
        logging.error(f"Error searching aliases {aliases}: {e}", exc_info=True)
        return None

def get_atm_strike(underlying_token: int, strike_diff: int, exchange: str):
    """Fetches LTP from WebSocket data and calculates ATM strike."""
    global latest_tick_data
    
    try:
        if underlying_token not in latest_tick_data[exchange]:
            logging.error(f"{exchange}: No tick data available for token {underlying_token}")
            return None
        
        tick_info = latest_tick_data[exchange][underlying_token]
        ltp = tick_info.get('last_price')
        
        if ltp is None:
            logging.error(f"{exchange}: Last price not found in tick data for token {underlying_token}")
            return None
        
        atm_strike = round(ltp / strike_diff) * strike_diff
        logging.debug(f"{exchange}: LTP from WebSocket: {ltp}, Calculated ATM strike: {atm_strike}")
        return atm_strike
    except Exception as e:
        logging.error(f"{exchange}: Error in get_atm_strike: {e}", exc_info=True)
        return None

def get_nearest_weekly_expiry(instruments: list, underlying_prefix_str: str, options_exchange: str):
    """Finds the nearest future weekly expiry date."""
    today = date.today()
    possible_expiries = set()
    trading_symbol = {}

    for inst in instruments:
        if inst['name'] == underlying_prefix_str and inst['exchange'] == options_exchange:
            if isinstance(inst['expiry'], date) and inst['expiry'] >= today:
                possible_expiries.add(inst['expiry'])
                trading_symbol[inst['expiry']] = inst['tradingsymbol']

    if not possible_expiries:
        logging.error(f"No future expiries found for {underlying_prefix_str} on {options_exchange}.")
        return None

    nearest_expiry = sorted(list(possible_expiries))[0]
    trading_symbol_of_nearest_expiry = trading_symbol[nearest_expiry]
    symbol_prefix = trading_symbol_of_nearest_expiry[0:len(underlying_prefix_str)+5]
    
    logging.info(f"Nearest weekly expiry for {underlying_prefix_str}: {nearest_expiry}")
    return {"expiry": nearest_expiry, "symbol_prefix": symbol_prefix}

def get_relevant_option_details(instruments: list, atm_strike_val: float, expiry_dt: date, 
                                strike_diff_val: int, opt_count: int, underlying_prefix_str: str, 
                                symbol_prefix: str, options_exchange: str):
    """Identifies relevant option contracts around ATM strike.
    Fetches opt_count+1 strikes on each side to create a buffer for edge cases.
    """
    relevant_options = {}
    if not expiry_dt or atm_strike_val is None:
        logging.error("Expiry date or ATM strike is None, cannot fetch option details.")
        return relevant_options

    # Fetch one extra strike on each side as buffer
    fetch_count = opt_count + 1
    for i in range(-fetch_count, fetch_count + 1):
        current_strike = atm_strike_val + (i * strike_diff_val)
        
        ce_symbol_pattern_core = f"{symbol_prefix}{int(current_strike)}"
        pe_symbol_pattern_core = f"{symbol_prefix}{int(current_strike)}"
        
        found_ce, found_pe = None, None
        
        for inst in instruments:
            if inst['name'] == underlying_prefix_str and \
               inst['strike'] == current_strike and \
               inst['expiry'] == expiry_dt and \
               inst['exchange'] == options_exchange:
                
                if inst['instrument_type'] == 'CE' and ce_symbol_pattern_core in inst['tradingsymbol']:
                    found_ce = inst
                elif inst['instrument_type'] == 'PE' and pe_symbol_pattern_core in inst['tradingsymbol']:
                    found_pe = inst
            
            if found_ce and found_pe:
                break
        
        # Generate key suffix based on position relative to ATM
        if i == 0: 
            key_suffix = "atm"
        elif i < 0: 
            key_suffix = f"itm{-i}"
        else: 
            key_suffix = f"otm{i}"
        
        if found_ce:
            relevant_options[f"{key_suffix}_ce"] = {
                'tradingsymbol': found_ce['tradingsymbol'], 
                'instrument_token': found_ce['instrument_token'], 
                'strike': current_strike,
                'position': i  # Track position for filtering display
            }
            logging.debug(f"Found CE contract: {found_ce['tradingsymbol']} for strike {current_strike}")
        else:
            logging.warning(f"CE contract NOT found for strike {current_strike} (position: {i})")
        
        if found_pe:
            relevant_options[f"{key_suffix}_pe"] = {
                'tradingsymbol': found_pe['tradingsymbol'], 
                'instrument_token': found_pe['instrument_token'], 
                'strike': current_strike,
                'position': i  # Track position for filtering display
            }
            logging.debug(f"Found PE contract: {found_pe['tradingsymbol']} for strike {current_strike}")
        else:
            logging.warning(f"PE contract NOT found for strike {current_strike} (position: {i})")
    
    logging.info(f"Relevant option details identified: {len(relevant_options)} contracts (includes buffer strikes).")
    return relevant_options

def get_oi_data_hybrid(kite_obj, option_details_dict: dict, exchange: str,
                       minutes_of_data: int = HISTORICAL_DATA_MINUTES):
    """Gets OI data using intelligent hybrid approach.
    
    Strategy:
    1. First check WebSocket OI history - use if sufficient (>= 30 records)
    2. If WebSocket insufficient, try Historical API
    3. If API fails, use whatever WebSocket data is available
    4. Merge API + WebSocket data for best coverage
    
    This ensures immediate data on startup while building reliable WebSocket history.
    """
    global oi_history
    historical_oi_store = {}
    
    if not option_details_dict:
        logging.warning("No option details provided to get_oi_data_hybrid.")
        return historical_oi_store

    to_date = datetime.now()
    from_date = to_date - timedelta(minutes=minutes_of_data)

    for option_key, details in option_details_dict.items():
        instrument_token = details.get('instrument_token')
        tradingsymbol = details.get('tradingsymbol')

        if not instrument_token:
            logging.warning(f"Missing instrument_token for {option_key} ({tradingsymbol}).")
            historical_oi_store[option_key] = []
            continue
        
        ws_data = []
        api_data = []
        
        # STRATEGY 1: Check WebSocket OI history
        if instrument_token in oi_history[exchange] and len(oi_history[exchange][instrument_token]) > 0:
            ws_data = oi_history[exchange][instrument_token]
            
            # If we have sufficient WebSocket data (30+ records = ~30 minutes), use it exclusively
            if len(ws_data) >= 30:
                logging.info(f"{exchange}: ✓ Using WebSocket OI history for {tradingsymbol} ({len(ws_data)} records)")
                historical_oi_store[option_key] = ws_data
                continue
            else:
                logging.info(f"{exchange}: WebSocket history building for {tradingsymbol} ({len(ws_data)} records, need 30+)")
        
        # STRATEGY 2: Try Historical API (for immediate data or to supplement WebSocket)
        try:
            raw_data = kite_obj.historical_data(instrument_token, from_date, to_date, 'minute', 
                                               continuous=False, oi=True)
            
            if isinstance(raw_data, pd.DataFrame):
                api_data = raw_data.to_dict('records')
            else:
                api_data = raw_data if raw_data else []
            
            if len(api_data) > 0:
                logging.info(f"{exchange}: ✓ Fetched {len(api_data)} records for {tradingsymbol} from API")
                
                # STRATEGY 3: Merge API data with WebSocket data
                if len(ws_data) > 0:
                    # Combine both sources: API for history, WebSocket for latest
                    combined_data = api_data + ws_data
                    # Remove duplicates based on timestamp, keep latest
                    seen_times = set()
                    merged = []
                    for record in reversed(combined_data):
                        rec_time = record['date']
                        if rec_time not in seen_times:
                            seen_times.add(rec_time)
                            merged.insert(0, record)
                    
                    historical_oi_store[option_key] = merged
                    logging.info(f"{exchange}: Merged API ({len(api_data)}) + WebSocket ({len(ws_data)}) = {len(merged)} records for {tradingsymbol}")
                else:
                    historical_oi_store[option_key] = api_data
            else:
                # API returned empty - use WebSocket data (even if insufficient)
                logging.warning(f"{exchange}: ⚠️  No API data for {tradingsymbol}. Using WebSocket data ({len(ws_data)} records)")
                historical_oi_store[option_key] = ws_data
                
        except Exception as e:
            # API failed - use WebSocket data as fallback
            logging.warning(f"{exchange}: ⚠️ API fetch failed for {tradingsymbol}: {e}. Using WebSocket data ({len(ws_data)} records)")
            historical_oi_store[option_key] = ws_data

    return historical_oi_store

def find_oi_at_timestamp(historical_candles: list, target_time: datetime, 
                          latest_oi_and_time: tuple):
    """Finds OI at or just before a specific target_time."""
    if not historical_candles:
        return None

    for candle in reversed(historical_candles):
        candle_time = candle['date']
        
        # Ensure both datetimes are timezone-naive for comparison
        if hasattr(candle_time, 'tzinfo') and candle_time.tzinfo is not None:
            candle_time = candle_time.replace(tzinfo=None)
        if hasattr(target_time, 'tzinfo') and target_time.tzinfo is not None:
            target_time = target_time.replace(tzinfo=None)
        
        if candle_time <= target_time:
            if latest_oi_and_time:
                latest_time = latest_oi_and_time[1]
                if hasattr(latest_time, 'tzinfo') and latest_time.tzinfo is not None:
                    latest_time = latest_time.replace(tzinfo=None)
                if candle_time > latest_time:
                    continue
            return candle.get('oi')
    
    return None


def standard_normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def standard_normal_pdf(x: float) -> float:
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)


def black_scholes_option_price(option_type: str, spot: float, strike: float,
                               rate: float, volatility: float, time_years: float) -> Optional[float]:
    if spot <= 0 or strike <= 0 or volatility <= 0 or time_years <= 0:
        return None
    sqrt_time = math.sqrt(time_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility ** 2) * time_years) / (volatility * sqrt_time)
    d2 = d1 - volatility * sqrt_time
    if option_type.upper() == 'CE':
        return spot * standard_normal_cdf(d1) - strike * math.exp(-rate * time_years) * standard_normal_cdf(d2)
    if option_type.upper() == 'PE':
        return strike * math.exp(-rate * time_years) * standard_normal_cdf(-d2) - spot * standard_normal_cdf(-d1)
    return None


def black_scholes_vega(spot: float, strike: float, rate: float,
                       volatility: float, time_years: float) -> float:
    if spot <= 0 or strike <= 0 or volatility <= 0 or time_years <= 0:
        return 0.0
    sqrt_time = math.sqrt(time_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility ** 2) * time_years) / (volatility * sqrt_time)
    return spot * standard_normal_pdf(d1) * sqrt_time


def calculate_implied_volatility(option_type: str, option_price: float, spot: float,
                                 strike: float, time_years: float, rate: float = RISK_FREE_RATE,
                                 initial_vol: float = 0.2, tolerance: float = 1e-4, max_iterations: int = 100):
    if option_price is None or option_price <= 0:
        return None
    if spot is None or spot <= 0 or strike is None or strike <= 0:
        return None
    if time_years is None or time_years <= 0:
        return None

    sigma = max(initial_vol, 1e-4)
    for _ in range(max_iterations):
        price = black_scholes_option_price(option_type, spot, strike, rate, sigma, time_years)
        if price is None:
            return None
        price_diff = price - option_price
        if abs(price_diff) < tolerance:
            return round(sigma * 100, 2)
        vega = black_scholes_vega(spot, strike, rate, sigma, time_years)
        if vega == 0:
            break
        sigma -= price_diff / vega
        if sigma <= 0:
            sigma = 1e-4
    return None


def compute_theoretical_price(option_type: str, spot: float, strike: float,
                              time_years: float, option_iv_percent: Optional[float] = None,
                              vix_percentage: Optional[float] = None,
                              fallback_volatility: float = 0.2) -> Optional[float]:
    if spot is None or strike is None or time_years is None:
        return None
    if spot <= 0 or strike <= 0 or time_years <= 0:
        return None

    volatility = None
    for candidate in (vix_percentage, option_iv_percent):
        if candidate is None:
            continue
        try:
            candidate_value = float(candidate)
        except (TypeError, ValueError):
            continue
        if candidate_value > 0:
            volatility = candidate_value / 100.0
            break

    if volatility is None:
        volatility = max(fallback_volatility, 1e-4)

    price = black_scholes_option_price(option_type, spot, strike, RISK_FREE_RATE, volatility, time_years)
    if price is None:
        return None
    return round(price, 2)


def calculate_oi_differences(raw_historical_data_store: dict, intervals_min: tuple):
    """Calculates OI differences between latest and past intervals."""
    oi_differences_report = {}
    current_processing_time = datetime.now()  # Use timezone-naive for consistency

    for option_key, candles_list in raw_historical_data_store.items():
        oi_differences_report[option_key] = {}
        
        latest_oi, latest_oi_timestamp = None, None
        if candles_list:
            latest_candle = candles_list[-1]
            latest_oi = latest_candle.get('oi')
            latest_oi_timestamp = latest_candle.get('date')
            
            # Special check for zero OI (different from None)
            if latest_oi == 0:
                logging.warning(f"⚠️  Latest OI is ZERO for {option_key} - option has no open positions")
                latest_oi = 0  # Keep as 0, not None
        
        oi_differences_report[option_key]['latest_oi'] = latest_oi
        oi_differences_report[option_key]['latest_oi_timestamp'] = latest_oi_timestamp

        if latest_oi is None:
            logging.warning(f"⚠️  No latest OI for {option_key} (candles: {len(candles_list)}) - setting all intervals to None")
            for interval in intervals_min:
                oi_differences_report[option_key][f'abs_diff_{interval}m'] = None
                oi_differences_report[option_key][f'pct_diff_{interval}m'] = None
            continue
        
        if latest_oi == 0:
            # OI is zero but exists - calculate changes from zero
            logging.debug(f"Latest OI is 0 for {option_key} - will attempt to calculate changes from zero base")
            # Continue processing - don't skip like with None

        for interval in intervals_min:
            target_past_time = current_processing_time - timedelta(minutes=interval)
            
            past_oi = find_oi_at_timestamp(
                candles_list,
                target_past_time,
                latest_oi_and_time=(latest_oi, latest_oi_timestamp)
            )
            
            abs_oi_diff = None
            pct_oi_change = None
            if past_oi is not None:
                abs_oi_diff = latest_oi - past_oi
                if past_oi != 0:
                    pct_oi_change = (abs_oi_diff / past_oi) * 100
            
            oi_differences_report[option_key][f'abs_diff_{interval}m'] = abs_oi_diff
            oi_differences_report[option_key][f'pct_diff_{interval}m'] = pct_oi_change
    
    return oi_differences_report

def prepare_web_data(
    oi_report: dict,
    contract_details: dict,
    current_atm_strike: float,
    strike_step: int,
    num_strikes_each_side: int,
    exchange: str,
    underlying_price: float = None,
    time_to_expiry_years: float = None,
    vix_value: float = None,
    current_time: datetime = None
):
    """Prepare enriched option data for the UI."""
    if current_time is None:
        current_time = datetime.now()

    call_options = []
    put_options = []

    for i in range(-num_strikes_each_side, num_strikes_each_side + 1):
        strike_val = current_atm_strike + (i * strike_step)

        if i == 0:
            key_suffix = "atm"
        elif i < 0:
            key_suffix = f"itm{-i}"
        else:
            key_suffix = f"otm{i}"

        option_key_ce = f"{key_suffix}_ce"
        ce_data = oi_report.get(option_key_ce, {})
        ce_contract = contract_details.get(option_key_ce, {})

        ce_latest_oi = ce_data.get('latest_oi')
        ce_latest_oi_time = ce_data.get('latest_oi_timestamp')

        ce_token = ce_contract.get('instrument_token')
        ce_ltp = None
        ce_volume = None
        if ce_token and ce_token in latest_tick_data[exchange]:
            ce_tick = latest_tick_data[exchange][ce_token]
            ce_ltp = ce_tick.get('last_price')
            ce_volume = ce_tick.get('volume')

        call_row = {
            'strike': int(ce_contract.get('strike', strike_val)),
            'symbol': ce_contract.get('tradingsymbol', 'N/A'),
            'latest_oi': ce_latest_oi,
            'oi_time': ce_latest_oi_time.strftime("%H:%M:%S") if ce_latest_oi_time else None,
            'moneyness': 'ATM' if i == 0 else ('ITM' if i < 0 else 'OTM'),
            'strike_type': 'atm' if i == 0 else ('itm' if i < 0 else 'otm'),
            'ltp': ce_ltp,
            'token': ce_token,
            'pct_changes': {},
            'iv': None,
            'theoretical_price': None,
            'price_diff': None
        }
        if ce_volume is not None:
            call_row['volume'] = ce_volume

        for interval in OI_CHANGE_INTERVALS_MIN:
            pct_change = ce_data.get(f'pct_diff_{interval}m')
            call_row['pct_changes'][f'{interval}m'] = pct_change

        if ce_ltp is not None and underlying_price is not None and time_to_expiry_years:
            call_row['iv'] = calculate_implied_volatility(
                'CE',
                ce_ltp,
                underlying_price,
                call_row['strike'],
                time_to_expiry_years
            )

        if underlying_price is not None and time_to_expiry_years:
            theoretical_price = compute_theoretical_price(
                'CE',
                underlying_price,
                call_row['strike'],
                time_to_expiry_years,
                option_iv_percent=call_row['iv'],
                vix_percentage=vix_value
            )
            if theoretical_price is not None:
                call_row['theoretical_price'] = theoretical_price
                if ce_ltp is not None:
                    call_row['price_diff'] = round(ce_ltp - theoretical_price, 2)

        call_options.append(call_row)

        option_key_pe = f"{key_suffix}_pe"
        pe_data = oi_report.get(option_key_pe, {})
        pe_contract = contract_details.get(option_key_pe, {})

        pe_latest_oi = pe_data.get('latest_oi')
        pe_latest_oi_time = pe_data.get('latest_oi_timestamp')

        pe_token = pe_contract.get('instrument_token')
        pe_ltp = None
        pe_volume = None
        if pe_token and pe_token in latest_tick_data[exchange]:
            pe_tick = latest_tick_data[exchange][pe_token]
            pe_ltp = pe_tick.get('last_price')
            pe_volume = pe_tick.get('volume')

        put_row = {
            'strike': int(pe_contract.get('strike', strike_val)),
            'symbol': pe_contract.get('tradingsymbol', 'N/A'),
            'latest_oi': pe_latest_oi,
            'oi_time': pe_latest_oi_time.strftime("%H:%M:%S") if pe_latest_oi_time else None,
            'moneyness': 'ATM' if i == 0 else ('ITM' if i > 0 else 'OTM'),
            'strike_type': 'atm' if i == 0 else ('itm' if i > 0 else 'otm'),
            'ltp': pe_ltp,
            'token': pe_token,
            'pct_changes': {},
            'iv': None,
            'theoretical_price': None,
            'price_diff': None
        }
        if pe_volume is not None:
            put_row['volume'] = pe_volume

        for interval in OI_CHANGE_INTERVALS_MIN:
            pct_change = pe_data.get(f'pct_diff_{interval}m')
            put_row['pct_changes'][f'{interval}m'] = pct_change

        if pe_ltp is not None and underlying_price is not None and time_to_expiry_years:
            put_row['iv'] = calculate_implied_volatility(
                'PE',
                pe_ltp,
                underlying_price,
                put_row['strike'],
                time_to_expiry_years
            )

        if underlying_price is not None and time_to_expiry_years:
            theoretical_price = compute_theoretical_price(
                'PE',
                underlying_price,
                put_row['strike'],
                time_to_expiry_years,
                option_iv_percent=put_row['iv'],
                vix_percentage=vix_value
            )
            if theoretical_price is not None:
                put_row['theoretical_price'] = theoretical_price
                if pe_ltp is not None:
                    put_row['price_diff'] = round(pe_ltp - theoretical_price, 2)

        put_options.append(put_row)

    def annotate_neighbor_oi_changes(options_list):
        strike_to_change = {
            opt['strike']: opt.get('pct_changes', {}).get('5m')
            for opt in options_list
        }
        for opt in options_list:
            strike = opt['strike']
            opt['strike_minus_100_oi_change'] = strike_to_change.get(strike - 100)
            opt['strike_plus_100_oi_change'] = strike_to_change.get(strike + 100)

    annotate_neighbor_oi_changes(call_options)
    annotate_neighbor_oi_changes(put_options)

    return call_options, put_options

# ==============================================================================
# --- PAPER TRADING: POSITION MONITORING ---
# ==============================================================================

def monitor_positions(call_options, put_options, exchange):
    """Monitor open positions and auto-exit if price moves ±25."""
    global open_positions, total_mtm, closed_positions_pnl
    
    if not open_positions[exchange]:
        total_mtm[exchange] = 0.0
        return
    
    # Create price lookup dict
    price_map = {}
    for opt in call_options:
        if opt['ltp'] is not None:
            price_map[opt['symbol']] = opt['ltp']
    for opt in put_options:
        if opt['ltp'] is not None:
            price_map[opt['symbol']] = opt['ltp']
    
    positions_to_close = []
    cumulative_mtm = 0.0
    
    with data_lock:
        for pos_id, position in list(open_positions[exchange].items()):
            symbol = position['symbol']
            current_price = price_map.get(symbol)
            
            if current_price is None:
                continue
            
            # Update current price and MTM (Mark to Market)
            entry_price = position['entry_price']
            side = position['side']
            qty = position['qty']
            
            # Calculate MTM based on side
            if side == 'B':  # Long position
                price_diff = current_price - entry_price
                mtm = price_diff * qty
            else:  # Short position (side == 'S')
                price_diff = entry_price - current_price
                mtm = price_diff * qty
            
            position['current_price'] = current_price
            position['mtm'] = round(mtm, 2)
            cumulative_mtm += mtm
            
            # Check for auto-exit: ±25 points
            should_exit = False
            exit_reason = ""
            
            if side == 'B':  # Long position
                if current_price >= entry_price + 25:
                    should_exit = True
                    exit_reason = "Target Hit (+25)"
                elif current_price <= entry_price - 25:
                    should_exit = True
                    exit_reason = "Stop Loss (-25)"
            else:  # Short position
                if current_price <= entry_price - 25:
                    should_exit = True
                    exit_reason = "Target Hit (-25)"
                elif current_price >= entry_price + 25:
                    should_exit = True
                    exit_reason = "Stop Loss (+25)"
            
            if should_exit:
                positions_to_close.append((pos_id, position, exit_reason, current_price, mtm))
        
        total_mtm[exchange] = round(cumulative_mtm, 2)
    
    # Close positions outside the lock to avoid conflicts
    for pos_id, position, exit_reason, exit_price, realized_pnl in positions_to_close:
        close_position(pos_id, position, exit_reason, exit_price, realized_pnl, exchange)

def close_position(pos_id, position, exit_reason, exit_price, realized_pnl, exchange):
    """Close a position and log to file."""
    global open_positions, closed_positions_pnl
    
    with data_lock:
        if pos_id in open_positions[exchange]:
            del open_positions[exchange][pos_id]
            closed_positions_pnl[exchange] += realized_pnl
    
    # Log to file
    side_text = "BUY" if position['side'] == 'B' else "SELL"
    exit_side = "SELL" if position['side'] == 'B' else "BUY"
    
    logging.info(f"{exchange}: 🔔 AUTO EXIT: {exit_side} {position['qty']} x {position['symbol']} @ {exit_price}")
    logging.info(f"{exchange}:    Reason: {exit_reason} | Entry: {position['entry_price']} | Exit: {exit_price}")
    logging.info(f"{exchange}:    Realized P&L: ₹{realized_pnl:,.2f} | Total Closed P&L: ₹{closed_positions_pnl[exchange]:,.2f}")
    
    # Emit position update to clients
    socketio.emit('position_closed', {
        'exchange': exchange,
        'position_id': pos_id,
        'exit_reason': exit_reason,
        'exit_price': exit_price,
        'realized_pnl': realized_pnl,
        'closed_pnl': closed_positions_pnl[exchange]
    })

# ==============================================================================
# --- DYNAMIC TOKEN SUBSCRIPTION ---
# ==============================================================================

def update_subscribed_tokens(new_tokens, exchange):
    """
    Dynamically subscribe to new tokens and unsubscribe from removed tokens.
    This is called when ATM strike changes and new option contracts need to be tracked.
    
    Args:
        new_tokens: List of new option tokens to subscribe
        exchange: 'NSE' or 'BSE'
    """
    global kws, exchange_instruments
    
    current_tokens = exchange_instruments[exchange].get('option_tokens', [])
    
    tokens_to_add = set(new_tokens) - set(current_tokens)
    tokens_to_remove = set(current_tokens) - set(new_tokens)
    
    if tokens_to_add and kws:
        try:
            kws.subscribe(list(tokens_to_add))
            kws.set_mode(kws.MODE_FULL, list(tokens_to_add))
            logging.info(f"{exchange}: ✓ Subscribed to {len(tokens_to_add)} new tokens (ATM shifted)")
        except Exception as e:
            logging.warning(f"{exchange}: Failed to subscribe to new tokens: {e}")
    
    if tokens_to_remove and kws:
        try:
            kws.unsubscribe(list(tokens_to_remove))
            logging.info(f"{exchange}: ✓ Unsubscribed from {len(tokens_to_remove)} old tokens")
        except Exception as e:
            logging.warning(f"{exchange}: Failed to unsubscribe from old tokens: {e}")
    
    # Update stored tokens
    exchange_instruments[exchange]['option_tokens'] = new_tokens
    
    return new_tokens

# ==============================================================================
# --- DATA UPDATE THREAD (Exchange-Specific) ---
# ==============================================================================

def run_data_update_loop_exchange(exchange):
    """
    Background thread that continuously updates OI data for a specific exchange.
    
    Args:
        exchange: 'NSE' or 'BSE'
    """
    global kite, latest_oi_data, exchange_instruments, kws
    global ml_predictor, ml_signal_generator, ml_feature_engineer, ml_models_loaded
    
    # Get exchange-specific configuration
    config = EXCHANGE_CONFIGS[exchange]
    underlying_token = exchange_instruments[exchange]['underlying_token']
    instruments = exchange_instruments[exchange].get('nfo_instruments' if exchange == 'NSE' else 'bfo_instruments', [])
    expiry_date = exchange_instruments[exchange]['expiry_date']
    symbol_prefix = exchange_instruments[exchange]['symbol_prefix']
    
    last_db_save_time = datetime.now()
    db_save_counter = 0

    logging.info(f"{exchange}: Data update thread started")
    market_session_state = None
    
    while True:
        try:
            now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
            session_state = 'OPEN'
            if now_ist.time() < TRADING_START_IST:
                session_state = 'PRE_OPEN'
            elif now_ist.time() > TRADING_END_IST:
                session_state = 'CLOSED'

            if session_state != market_session_state:
                logging.info(f"{exchange}: Market session state -> {session_state}")
                market_session_state = session_state

            if session_state != 'OPEN':
                status_message = (
                    'Pre-open (market opens 09:15 IST)'
                    if session_state == 'PRE_OPEN'
                    else 'Market closed (trading resumes 09:15 IST)'
                )
                with data_lock:
                    latest_oi_data[exchange]['status'] = status_message
                    latest_oi_data[exchange]['last_update'] = now_ist.strftime('%H:%M:%S')
                time.sleep(UI_REFRESH_INTERVAL_SECONDS)
                continue

            logging.info(f"{exchange}: Starting data update iteration")

            current_iteration_time = datetime.now()
            
            # Get current ATM strike
            current_atm_strike = get_atm_strike(
                underlying_token, 
                config['strike_difference'],
                exchange
            )
            
            if not current_atm_strike:
                with data_lock:
                    latest_oi_data[exchange]['status'] = 'Error: Could not determine ATM strike'
                time.sleep(UI_REFRESH_INTERVAL_SECONDS)
                continue
            
            # Get option contracts around ATM
            option_contract_details = get_relevant_option_details(
                instruments, 
                current_atm_strike, 
                expiry_date,
                config['strike_difference'], 
                config['options_count'], 
                config['underlying_prefix'], 
                symbol_prefix,
                config['options_exchange']
            )
            
            if not option_contract_details:
                with data_lock:
                    latest_oi_data[exchange]['status'] = f'Warning: No contracts found for ATM {int(current_atm_strike)}'
                time.sleep(UI_REFRESH_INTERVAL_SECONDS)
                continue
            
            # Extract option tokens
            option_tokens = [details['instrument_token'] for details in option_contract_details.values() 
                           if 'instrument_token' in details]
            
            # Dynamic token subscription (subscribe/unsubscribe based on ATM changes)
            if option_tokens:
                update_subscribed_tokens(option_tokens, exchange)

            # Get underlying LTP
            underlying_ltp = None
            if underlying_token in latest_tick_data[exchange]:
                underlying_ltp = latest_tick_data[exchange][underlying_token].get('last_price')

            # Estimate time to expiry in years for Black-Scholes
            time_to_expiry_years = None
            if expiry_date:
                expiry_datetime = datetime.combine(
                    expiry_date,
                    datetime.strptime('15:30:00', '%H:%M:%S').time()
                )
                time_to_expiry_seconds = (expiry_datetime - current_iteration_time).total_seconds()
                if time_to_expiry_seconds <= 0:
                    if expiry_date >= current_iteration_time.date():
                        time_to_expiry_seconds = MIN_TIME_TO_EXPIRY_SECONDS
                    else:
                        time_to_expiry_seconds = None
                if time_to_expiry_seconds and time_to_expiry_seconds > 0:
                    time_to_expiry_years = time_to_expiry_seconds / (365 * 24 * 60 * 60)
            
            # Get OI data (WebSocket history + API fallback + Database on restart)
            raw_historical_oi_data = get_oi_data_hybrid(kite, option_contract_details, exchange)
            
            # Log data accumulation status for monitoring
            total_records = sum(len(v) for v in raw_historical_oi_data.values())
            logging.debug(f"{exchange}: Total OI records available: {total_records}")
            
            # Calculate OI differences
            oi_change_data = calculate_oi_differences(raw_historical_oi_data, OI_CHANGE_INTERVALS_MIN)
            
            # Prepare web data
            call_options, put_options = prepare_web_data(
                oi_change_data, 
            option_contract_details,
            current_atm_strike,
            config['strike_difference'],
            config['options_count'],
            exchange,
            underlying_price=underlying_ltp,
            time_to_expiry_years=time_to_expiry_years,
            vix_value=latest_vix_data['value'],
            current_time=current_iteration_time
            )
            
            # Calculate Put Call Ratio (PCR) = Sum of PUT OI / Sum of CALL OI
            total_put_oi = sum(opt['latest_oi'] for opt in put_options if opt['latest_oi'] is not None)
            total_call_oi = sum(opt['latest_oi'] for opt in call_options if opt['latest_oi'] is not None)
            pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else None

            prev_close_price = get_cached_previous_close(exchange)
            price_change = None
            price_change_pct = None
            if underlying_ltp is not None and prev_close_price:
                price_change = underlying_ltp - prev_close_price
                if prev_close_price:
                    price_change_pct = (price_change / prev_close_price) * 100

            # Monitor open positions and auto-exit if needed
            monitor_positions(call_options, put_options, exchange)
            
            # Generate ML prediction if available
            ml_prediction_data = None
            if ml_models_loaded and ml_predictor and ml_feature_engineer and underlying_ltp:
                try:
                    # Get recent data for feature engineering
                    from ml_system.data.data_extractor import DataExtractor
                    extractor = DataExtractor()
                    try:
                        # Get last few records for feature engineering
                        raw_data = extractor.get_time_series_data(exchange, lookback_days=1)
                        if not raw_data.empty:
                            # Engineer features
                            features_df = ml_feature_engineer.engineer_all_features(raw_data)
                            if not features_df.empty:
                                # Get latest features
                                feature_cols = [col for col in features_df.columns 
                                               if col not in ['timestamp', 'future_price', 'price_change', 'price_change_pct', 'direction']]
                                latest_features = features_df[feature_cols].iloc[[-1]]
                                
                                # Make prediction
                                prediction_result = ml_predictor.predict_and_signal(
                                    latest_features,
                                    entry_threshold=0.02,
                                    min_confidence=0.5
                                )
                                
                                # Generate signal
                                signal = ml_signal_generator.generate_signal(
                                    prediction=prediction_result['prediction'],
                                    confidence=prediction_result['confidence'],
                                    current_price=underlying_ltp,
                                    capital=100000.0
                                )
                                
                                ml_prediction_data = {
                                    'prediction': prediction_result['prediction'],
                                    'prediction_pct': prediction_result['prediction'],
                                    'confidence': prediction_result['confidence'],
                                    'signal': signal.signal_type,
                                    'strength': signal.strength,
                                    'reasoning': signal.reasoning
                                }
                    finally:
                        extractor.close()
                except Exception as e:
                    logging.warning(f"{exchange}: ML prediction error: {e}")
            
            # Update global data
            with data_lock:
                latest_oi_data[exchange]['call_options'] = call_options
                latest_oi_data[exchange]['put_options'] = put_options
                latest_oi_data[exchange]['atm_strike'] = int(current_atm_strike)
                latest_oi_data[exchange]['underlying_price'] = round(float(underlying_ltp), 2) if underlying_ltp is not None else None
                latest_oi_data[exchange]['last_update'] = datetime.now().strftime('%H:%M:%S')
                latest_oi_data[exchange]['status'] = 'Live'
                latest_oi_data[exchange]['pcr'] = pcr
                latest_oi_data[exchange]['previous_close'] = round(float(prev_close_price), 2) if prev_close_price else None
                latest_oi_data[exchange]['previous_close_change'] = round(float(price_change), 2) if price_change is not None else None
                latest_oi_data[exchange]['previous_close_change_pct'] = round(float(price_change_pct), 2) if price_change_pct is not None else None
                latest_oi_data[exchange]['vix'] = latest_vix_data['value']
                latest_oi_data[exchange]['diff_thresholds'] = DIFF_THRESHOLDS
                if ml_prediction_data:
                    latest_oi_data[exchange]['ml_prediction'] = ml_prediction_data['prediction']
                    latest_oi_data[exchange]['ml_prediction_pct'] = ml_prediction_data['prediction_pct']
                    latest_oi_data[exchange]['ml_signal'] = ml_prediction_data['signal']
                    latest_oi_data[exchange]['ml_confidence'] = ml_prediction_data['confidence']
                    latest_oi_data[exchange]['ml_strength'] = ml_prediction_data['strength']
                    latest_oi_data[exchange]['ml_reasoning'] = ml_prediction_data['reasoning']
            
            # Save to database (including underlying price and ATM strike)
            current_timestamp = datetime.now()
            if (current_timestamp - last_db_save_time).total_seconds() >= DB_SAVE_INTERVAL_SECONDS:
                db.save_option_chain_snapshot(
                    exchange,
                    call_options,
                    put_options,
                    underlying_price=underlying_ltp,
                    atm_strike=current_atm_strike,
                    timestamp=current_timestamp
                )
                last_db_save_time = current_timestamp
                db_save_counter += 1
            
            # Emit update to all connected clients (include positions data)
            with data_lock:
                emit_data = latest_oi_data[exchange].copy()
                emit_data['open_positions'] = list(open_positions[exchange].values())
                emit_data['total_mtm'] = total_mtm[exchange]
                emit_data['closed_pnl'] = closed_positions_pnl[exchange]
            
            # Emit exchange-specific event
            socketio.emit(f'data_update_{exchange}', emit_data)
            
            logging.info(f"{exchange}: ✓ Data updated. ATM: {current_atm_strike}, PCR: {pcr}")
            time.sleep(UI_REFRESH_INTERVAL_SECONDS)
            
        except Exception as e:
            logging.error(f"{exchange}: Error in data update loop: {e}", exc_info=True)
            with data_lock:
                latest_oi_data[exchange]['status'] = f'Error: {str(e)}'
            time.sleep(UI_REFRESH_INTERVAL_SECONDS)

# ==============================================================================
# --- FLASK ROUTES ---
# ==============================================================================

def login_required(f):
    """Decorator to require authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        global authenticated
        if not session.get('authenticated') or not authenticated:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login')
def login():
    """Login page route."""
    global authenticated, USER_ID
    if session.get('authenticated') and authenticated:
        return redirect(url_for('index'))
    # Pass USER_ID from .env if available (for convenience, but user can override)
    return render_template('login.html', default_user_id=USER_ID if USER_ID else '')

@app.route('/api/login', methods=['POST'])
def api_login():
    """API endpoint for login authentication."""
    global authenticated, kite, current_enctoken
    
    try:
        data = request.get_json()
        user_id = data.get('user_id', '').strip()
        password = data.get('password', '').strip()
        twofa = data.get('twofa', '').strip()
        
        if not user_id or not password or not twofa:
            return jsonify({'success': False, 'error': 'All fields are required'}), 400
        
        if len(twofa) != 6 or not twofa.isdigit():
            return jsonify({'success': False, 'error': '2FA code must be a 6-digit number'}), 400
        
        # Authenticate with Zerodha
        try:
            enctoken = get_enctoken(user_id, password, twofa)
            if not enctoken:
                return jsonify({'success': False, 'error': 'Invalid credentials or 2FA code'}), 401
            
            # Initialize KiteApp
            kite_instance = KiteApp(enctoken=enctoken)
            
            # Verify connection
            profile = kite_instance.profile()
            
            # Store authentication state
            authenticated = True
            current_enctoken = enctoken
            kite = kite_instance
            session['authenticated'] = True
            session['user_id'] = profile.get('user_id')
            session['user_name'] = profile.get('user_name')
 
            # Start initialization in background
            Thread(target=initialize_system_async, daemon=True).start()
            
            return jsonify({
                'success': True,
                'message': f'Authenticated as {profile.get("user_id")}',
                'user_id': profile.get('user_id'),
                'user_name': profile.get('user_name')
            })
        
        except Exception as e:
            logging.error(f"Login error: {e}", exc_info=True)
            return jsonify({'success': False, 'error': f'Authentication failed: {str(e)}'}), 401
    
    except Exception as e:
        logging.error(f"Login API error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'Server error during login'}), 500

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """API endpoint for logout."""
    global authenticated, kite, current_enctoken, kws
    
    authenticated = False
    current_enctoken = None
    kite = None
    session.clear()
    
    # Close WebSocket connections
    if kws:
        try:
            print("\n🔌 Closing WebSocket connection...")
            logging.info("Closing WebSocket connection")
            kws.close()
            print("✓ WebSocket connection closed successfully")
            logging.info("WebSocket connection closed successfully")
        except Exception as e:
            print(f"⚠ Error closing WebSocket: {e}")
            logging.warning(f"Error closing WebSocket: {e}")
    
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/')
@login_required
def index():
    """Main page route - now supports dual exchange tabs."""
    if not authenticated:
        return redirect(url_for('login'))
    return render_template('index.html', 
                          exchanges=['NSE', 'BSE'],
                          exchange_configs=EXCHANGE_CONFIGS,
                          intervals=OI_CHANGE_INTERVALS_MIN,
                          thresholds=PCT_CHANGE_THRESHOLDS)

@app.route('/backtest')
@login_required
def backtest_page():
    """Backtesting page route."""
    return render_template('backtest.html',
                          exchanges=['NSE', 'BSE'],
                          exchange_configs=EXCHANGE_CONFIGS)


@app.route('/analysis')
@login_required
def analysis_page():
    """Historical analysis page route."""
    default_exchange = 'NSE'
    return render_template(
        'analysis.html',
        exchanges=['NSE', 'BSE'],
        default_exchange=default_exchange,
        oi_intervals=[5, 10, 15, 30],
        value_columns=[
            {'value': key, 'label': label}
            for key, label in HEATMAP_VALUE_LABELS.items()
        ]
    )

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """API endpoint to run backtest."""
    try:
        from ml_system.backtesting.backtest_engine import BacktestEngine
        from ml_system.data.data_extractor import DataExtractor
        from ml_system.features.feature_engineer import FeatureEngineer
        from ml_system.training.train_baseline import BaselineTrainer
        import joblib
        
        data = request.get_json()
        
        # Get parameters
        exchange = data.get('exchange', 'NSE')
        initial_capital = float(data.get('initial_capital', 100000))
        entry_threshold = float(data.get('entry_threshold', 0.02))
        exit_threshold = float(data.get('exit_threshold', 0.05))
        stop_loss_pct = float(data.get('stop_loss_pct', 0.03))
        max_holding_periods = int(data.get('max_holding_periods', 30))
        lookback_days = int(data.get('lookback_days', 30))
        commission_rate = float(data.get('commission_rate', 0.0003))
        slippage = float(data.get('slippage', 0.0001))
        position_size_pct = float(data.get('position_size_pct', 0.1))
        
        # Initialize backtester
        backtester = BacktestEngine(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage=slippage,
            position_size_pct=position_size_pct
        )
        
        # Get data and train/load model
        extractor = DataExtractor()
        engineer = FeatureEngineer()
        trainer = BaselineTrainer()
        
        try:
            # Get time series data
            raw_data = extractor.get_time_series_data(exchange, lookback_days=lookback_days)
            
            if raw_data.empty:
                return jsonify({'error': 'No data available for backtesting'}), 400
            
            # Engineer features
            features_df = engineer.engineer_all_features(raw_data)
            
            if features_df.empty:
                return jsonify({'error': 'Feature engineering failed'}), 400
            
            # Prepare data
            feature_cols = [col for col in features_df.columns 
                           if col not in ['timestamp', 'future_price', 'price_change', 'price_change_pct', 'direction']]
            X = features_df[feature_cols]
            y = features_df['price_change_pct']
            prices = features_df['underlying_price']
            timestamps = features_df['timestamp']
            
            # Remove NaN
            valid_mask = ~(X.isna().any(axis=1) | y.isna() | prices.isna())
            X = X[valid_mask]
            y = y[valid_mask]
            prices = prices[valid_mask]
            timestamps = timestamps[valid_mask]
            
            if len(X) == 0:
                return jsonify({'error': 'No valid data after cleaning'}), 400
            
            # Split data (80% train, 20% test)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            prices_test = prices.iloc[split_idx:]
            timestamps_test = timestamps.iloc[split_idx:]
            
            # Load or train model
            model_path = "ml_system/models/random_forest_model.pkl"
            predictions = None
            
            # Ensure trainer has models and scalers dicts
            if not hasattr(trainer, 'models'):
                trainer.models = {}
            if not hasattr(trainer, 'scalers'):
                trainer.scalers = {}
            
            if os.path.exists(model_path):
                try:
                    loaded_model = joblib.load(model_path)
                    trainer.models['random_forest'] = loaded_model
                    
                    scaler_path = "ml_system/models/random_forest_scaler.pkl"
                    if os.path.exists(scaler_path):
                        loaded_scaler = joblib.load(scaler_path)
                        trainer.scalers['random_forest'] = loaded_scaler
                        X_test_scaled = trainer.scalers['random_forest'].transform(X_test)
                        # Convert scaled array back to DataFrame with column names
                        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
                        predictions = trainer.models['random_forest'].predict(X_test_scaled_df)
                    else:
                        # Model exists but no scaler - use model without scaling
                        logging.warning("Random Forest model found but no scaler. Using model without scaling.")
                        # Ensure X_test is a DataFrame with proper column names to avoid warnings
                        if isinstance(X_test, pd.DataFrame):
                            predictions = trainer.models['random_forest'].predict(X_test)
                        else:
                            # Convert to DataFrame if it's not already
                            X_test_df = pd.DataFrame(X_test, columns=X_test.columns if hasattr(X_test, 'columns') else None)
                            predictions = trainer.models['random_forest'].predict(X_test_df)
                except Exception as e:
                    logging.error(f"Error loading random_forest model: {e}", exc_info=True)
                    predictions = None
            
            # If model loading failed or doesn't exist, train new model
            if predictions is None:
                # Train model
                features_df_filtered = features_df.loc[valid_mask]
                baseline_results = trainer.train_all_baselines(
                    features_df_filtered,
                    target_col='price_change_pct',
                    task='regression'
                )
                # Use best model predictions
                if 'random_forest' in trainer.models:
                    if 'random_forest' in trainer.scalers:
                        X_test_scaled = trainer.scalers['random_forest'].transform(X_test)
                        # Convert scaled array back to DataFrame with column names
                        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
                        predictions = trainer.models['random_forest'].predict(X_test_scaled_df)
                    else:
                        # Ensure X_test is a DataFrame with proper column names
                        if isinstance(X_test, pd.DataFrame):
                            predictions = trainer.models['random_forest'].predict(X_test)
                        else:
                            X_test_df = pd.DataFrame(X_test, columns=X_test.columns if hasattr(X_test, 'columns') else None)
                            predictions = trainer.models['random_forest'].predict(X_test_df)
                else:
                    # Use first available model
                    model_name = list(trainer.models.keys())[0]
                    if model_name in trainer.scalers:
                        X_test_scaled = trainer.scalers[model_name].transform(X_test)
                        # Convert scaled array back to DataFrame with column names
                        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
                        predictions = trainer.models[model_name].predict(X_test_scaled_df)
                    else:
                        # Ensure X_test is a DataFrame with proper column names
                        if isinstance(X_test, pd.DataFrame):
                            predictions = trainer.models[model_name].predict(X_test)
                        else:
                            X_test_df = pd.DataFrame(X_test, columns=X_test.columns if hasattr(X_test, 'columns') else None)
                            predictions = trainer.models[model_name].predict(X_test_df)
            
            if predictions is None or len(predictions) == 0:
                return jsonify({'error': 'Failed to generate predictions. Please ensure models are trained.'}), 500
            
            # Run backtest
            result = backtester.backtest_strategy(
                predictions=pd.Series(predictions),
                actual_prices=pd.Series(prices_test.values),
                timestamps=pd.Series(timestamps_test.values),
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                stop_loss_pct=stop_loss_pct,
                max_holding_periods=max_holding_periods
            )
            
            # Prepare response
            trades_data = []
            for trade in result.trades:
                trades_data.append({
                    'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
                    'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'trade_type': trade.trade_type.value,
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'exit_reason': trade.exit_reason
                })
            
            equity_curve_data = []
            for timestamp, equity in result.equity_curve.items():
                equity_curve_data.append({
                    'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                    'equity': float(equity)
                })
            
            response = {
                'success': True,
                'results': {
                    'total_trades': result.total_trades,
                    'winning_trades': result.winning_trades,
                    'losing_trades': result.losing_trades,
                    'total_pnl': float(result.total_pnl),
                    'total_return_pct': float(result.total_return_pct),
                    'win_rate': float(result.win_rate),
                    'avg_win': float(result.avg_win),
                    'avg_loss': float(result.avg_loss),
                    'profit_factor': float(result.profit_factor),
                    'max_drawdown': float(result.max_drawdown),
                    'max_drawdown_pct': float(result.max_drawdown_pct),
                    'sharpe_ratio': float(result.sharpe_ratio),
                    'sortino_ratio': float(result.sortino_ratio),
                    'initial_capital': float(initial_capital),
                    'final_capital': float(initial_capital + result.total_pnl)
                },
                'trades': trades_data,
                'equity_curve': equity_curve_data
            }
            
            return jsonify(response)
        
        finally:
            extractor.close()
    
    except Exception as e:
        logging.error(f"Backtest error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def _ensure_authenticated_api():
    if not session.get('authenticated') or not authenticated:
        return False
    return True


def _parse_datetime_param(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.strptime(value, '%Y-%m-%d %H:%M')
        except ValueError:
            return None


def _serialize_metric(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return value


@app.route('/api/analysis/summary', methods=['GET'])
def api_analysis_summary():
    """Provide aggregated OI summary and heatmap-ready data."""
    if not _ensure_authenticated_api():
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401

    if OIBackAnalysisPipeline is None:
        return jsonify({'success': False, 'error': 'Analysis tools not available'}), 503

    try:
        exchange = request.args.get('exchange', 'NSE').upper()
        if exchange not in EXCHANGE_CONFIGS:
            return jsonify({'success': False, 'error': 'Invalid exchange'}), 400

        start_dt = _parse_datetime_param(request.args.get('start'))
        end_dt = _parse_datetime_param(request.args.get('end'))
        if start_dt and end_dt and start_dt >= end_dt:
            return jsonify({'success': False, 'error': 'Start time must be before end time'}), 400

        resample_rule = request.args.get('resample') or None
        option_type = request.args.get('option_type', 'CE').upper()
        value_column = request.args.get('value_column', 'pct_change_5m')

        pipeline = get_analysis_pipeline()
        snapshots = pipeline.fetch_snapshots(exchange=exchange, start=start_dt, end=end_dt)

        if snapshots.empty:
            return jsonify({'success': True, 'data': None, 'message': 'No data found for the selected window'})

        summary_df = pipeline.build_summary_table(
            snapshots,
            resample_rule=resample_rule,
            include_pct_changes=True
        )

        summary_payload = []
        if not summary_df.empty:
            for _, row in summary_df.iterrows():
                timestamp = row.get('timestamp')
                record = {}
                for key, value in row.items():
                    if key == 'timestamp':
                        continue
                    record[key] = None if pd.isna(value) else _serialize_metric(value)
                record['timestamp'] = timestamp.isoformat() if isinstance(timestamp, datetime) else (
                    timestamp.to_pydatetime().isoformat() if isinstance(timestamp, pd.Timestamp) else str(timestamp)
                )
                summary_payload.append(record)

        heatmap_payload = None
        try:
            heatmap_matrix = pipeline.build_strike_heatmap(
                snapshots,
                option_type=option_type,
                value_column=value_column
            )
            if not heatmap_matrix.empty:
                value_label = HEATMAP_VALUE_LABELS.get(value_column, value_column)
                strikes = [
                    float(strike) if isinstance(strike, (int, float, np.number)) else strike
                    for strike in heatmap_matrix.index.tolist()
                ]
                timestamps = [
                    ts.isoformat() if isinstance(ts, datetime) else (
                        ts.to_pydatetime().isoformat() if isinstance(ts, pd.Timestamp) else str(ts)
                    )
                    for ts in heatmap_matrix.columns.tolist()
                ]
                matrix_values = heatmap_matrix.to_numpy()
                values = [
                    [_serialize_metric(cell) for cell in row]
                    for row in matrix_values
                ]
                heatmap_payload = {
                    'strikes': strikes,
                    'timestamps': timestamps,
                    'values': values,
                    'value_column': value_column,
                    'value_label': value_label
                }
        except ValueError as e:
            logging.warning(f"Heatmap generation failed: {e}")

        response = {
            'success': True,
            'data': {
                'exchange': exchange,
                'summary': summary_payload,
                'heatmap': heatmap_payload,
                'meta': {
                    'records': len(summary_payload),
                    'raw_rows': len(snapshots),
                    'option_type': option_type,
                    'value_column': value_column,
                    'value_label': HEATMAP_VALUE_LABELS.get(value_column, value_column),
                    'resample_rule': resample_rule,
                    'options': {
                        'option_types': ['CE', 'PE'],
                        'value_columns': [
                            {'value': key, 'label': label}
                            for key, label in HEATMAP_VALUE_LABELS.items()
                        ]
                    }
                }
            }
        }

        if not summary_payload:
            response['message'] = 'No summary data available for selected window'

        if heatmap_payload is None:
            response.setdefault('warnings', []).append('Heatmap unavailable for the selected parameters')

        return jsonify(response)

    except FileNotFoundError as e:
        logging.error(f"Analysis pipeline error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    except Exception as e:
        logging.error(f"Analysis summary error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'Failed to generate analysis summary'}), 500

@app.route('/api/data')
def get_data():
    """API endpoint to get current OI data for all exchanges."""
    with data_lock:
        return jsonify({
            'NSE': latest_oi_data['NSE'],
            'BSE': latest_oi_data['BSE']
        })

@app.route('/api/data/<exchange>')
def get_exchange_data(exchange):
    """API endpoint to get current OI data for a specific exchange."""
    if exchange not in ['NSE', 'BSE']:
        return jsonify({'error': 'Invalid exchange'}), 400
    
    with data_lock:
        return jsonify(latest_oi_data[exchange])


@app.route('/api/latest-price/<path:symbol>', methods=['GET'])
def api_latest_price(symbol):
    """Expose latest underlying LTP for the requested symbol (e.g. NSE:NIFTY 50)."""
    decoded_symbol = unquote(symbol)
    normalized_symbol = decoded_symbol.strip().upper()

    if ':' not in normalized_symbol:
        return jsonify({"error": UI_STRINGS['survivor_feed_error_exchange']}), 400

    tick, event_time, exchange_key = get_latest_tick_for_symbol(decoded_symbol)
    if tick is None or exchange_key is None:
        return jsonify({"error": UI_STRINGS['survivor_feed_error_not_found']}), 404

    payload = _build_shared_feed_payload(exchange_key, tick, event_time)
    if payload is None:
        return jsonify({"error": UI_STRINGS['survivor_feed_error_not_found']}), 404

    return jsonify(payload)

@app.route('/api/place_order', methods=['POST'])
def place_order():
    """API endpoint to place a paper trading order (exchange-specific)."""
    global position_counter, open_positions
    
    try:
        data = request.get_json()
        exchange = data.get('exchange', 'NSE')  # Default to NSE if not specified
        symbol = data.get('symbol')
        option_type = data.get('type')  # 'CE' or 'PE'
        side = data.get('side')  # 'B' or 'S'
        price = float(data.get('price'))
        qty = int(data.get('qty', 300))
        
        if exchange not in ['NSE', 'BSE']:
            return jsonify({'success': False, 'error': 'Invalid exchange'}), 400
        
        position_counter[exchange] += 1
        position_id = f"{exchange}_P{position_counter[exchange]:04d}"
        
        entry_time = datetime.now().strftime('%H:%M:%S')
        
        # Create position
        position = {
            'id': position_id,
            'symbol': symbol,
            'type': option_type,
            'side': side,
            'entry_price': price,
            'qty': qty,
            'entry_time': entry_time,
            'current_price': price,
            'mtm': 0.0,
            'exchange': exchange
        }
        
        with data_lock:
            open_positions[exchange][position_id] = position
        
        # Log to file
        side_text = "BUY" if side == 'B' else "SELL"
        logging.info(f"{exchange}: 📊 PAPER TRADE: {side_text} {qty} x {symbol} @ {price} | Position ID: {position_id}")
        
        return jsonify({
            'success': True,
            'message': f'{side_text} order placed',
            'position_id': position_id,
            'position': position
        })
        
    except Exception as e:
        logging.error(f"Error placing order: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/positions')
def get_positions():
    """API endpoint to get open positions, MTM, and P&L for all exchanges."""
    with data_lock:
        return jsonify({
            'NSE': {
                'open_positions': list(open_positions['NSE'].values()),
                'total_mtm': total_mtm['NSE'],
                'closed_pnl': closed_positions_pnl['NSE']
            },
            'BSE': {
                'open_positions': list(open_positions['BSE'].values()),
                'total_mtm': total_mtm['BSE'],
                'closed_pnl': closed_positions_pnl['BSE']
            }
        })

@app.route('/api/positions/<exchange>')
def get_exchange_positions(exchange):
    """API endpoint to get open positions for a specific exchange."""
    if exchange not in ['NSE', 'BSE']:
        return jsonify({'error': 'Invalid exchange'}), 400
    
    with data_lock:
        return jsonify({
            'open_positions': list(open_positions[exchange].values()),
            'total_mtm': total_mtm[exchange],
            'closed_pnl': closed_positions_pnl[exchange]
        })

@socketio.on('connect')
def handle_connect():
    """Handle client connection - send initial data for both exchanges."""
    logging.info('Client connected')
    with data_lock:
        # Send initial data for both exchanges
        emit('data_update_NSE', latest_oi_data['NSE'])
        emit('data_update_BSE', latest_oi_data['BSE'])

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logging.info('Client disconnected')

# ==============================================================================
# --- CLEANUP HANDLERS ---
# ==============================================================================

def cleanup_connections():
    """Cleanup function to close WebSocket connections gracefully."""
    global kws, cleanup_done
    
    # Prevent duplicate cleanup calls
    if cleanup_done:
        return
    
    cleanup_done = True
    
    if kws:
        try:
            print("\n🔌 Closing WebSocket connection...")
            logging.info("Closing WebSocket connection")
            kws.close()
            print("✓ WebSocket connection closed successfully")
            logging.info("WebSocket connection closed successfully")
        except Exception as e:
            print(f"⚠ Error closing WebSocket: {e}")
            logging.warning(f"Error closing WebSocket: {e}")

def signal_handler(signum, frame):
    """Handle termination signals for graceful shutdown."""
    signal_name = signal.Signals(signum).name
    print(f"\n\n⚠ Received signal: {signal_name}")
    logging.info(f"Received signal: {signal_name}")
    cleanup_connections()
    sys.exit(0)

# Register cleanup handlers
atexit.register(cleanup_connections)  # Called on normal program exit
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

# ==============================================================================
# --- AUTO SHUTDOWN MONITOR ---
# ==============================================================================

def start_auto_shutdown_monitor():
    """Start background thread that terminates the application at 15:30 IST."""
    if not AUTO_SHUTDOWN_ENABLED:
        logging.info("Auto-shutdown monitor disabled.")
        return

    def monitor_shutdown():
        global auto_shutdown_triggered
        logging.info("Auto-shutdown monitor started (15:30 IST).")
        target_datetime = None

        while not auto_shutdown_triggered:
            now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))

            if target_datetime is None or now_ist.date() != target_datetime.date():
                target_datetime = now_ist.replace(
                    hour=AUTO_SHUTDOWN_IST_HOUR,
                    minute=AUTO_SHUTDOWN_IST_MINUTE,
                    second=0,
                    microsecond=0
                )

            if now_ist >= target_datetime:
                auto_shutdown_triggered = True
                message = "⏰ Auto shutdown triggered at 15:30 IST. Cleaning up..."
                print(f"\n{message}")
                logging.info("Auto shutdown triggered at 15:30 IST.")
                cleanup_connections()
                os._exit(0)

            time.sleep(AUTO_SHUTDOWN_CHECK_INTERVAL_SECONDS)

    shutdown_thread = Thread(
        target=monitor_shutdown,
        daemon=True,
        name="AutoShutdownMonitor"
    )
    shutdown_thread.start()

# ==============================================================================
# --- MAIN INITIALIZATION ---
# ==============================================================================

def initialize_system_async():
    """Initialize the trading system asynchronously after login."""
    global kite, kws, exchange_instruments, latest_oi_data, oi_history, latest_vix_data, VIX_TOKEN
    global ml_predictor, ml_signal_generator, ml_feature_engineer, ml_models_loaded
    
    if not kite:
        logging.error("Kite instance not available for initialization")
        return
    
    try:
        logging.info("=" * 70)
        logging.info("DUAL EXCHANGE OI TRACKER - NSE (NIFTY) & BSE (SENSEX)")
        logging.info("=" * 70)
        
        # Verify connection
        profile = kite.profile()
        logging.info(f"✓ Connected as: {profile.get('user_id')} ({profile.get('user_name')})")

        # Prime previous close cache for both exchanges
        for exch in ['NSE', 'BSE']:
            prev_close = get_cached_previous_close(exch)
            if prev_close is not None:
                latest_oi_data[exch]['previous_close'] = round(float(prev_close), 2)

        # ==== NSE/NIFTY Instruments ====
        logging.info("\n" + "=" * 70)
        logging.info("FETCHING NSE/NIFTY INSTRUMENTS")
        logging.info("=" * 70)
        
        logging.info("Fetching NFO (NIFTY Options) instruments...")
        nfo_instruments = kite.instruments('NFO')
        if not nfo_instruments:
            logging.error("Failed to fetch NFO instruments.")
            return
        logging.info(f"✓ Fetched {len(nfo_instruments)} NFO instruments")
        
        nse_instruments = kite.instruments('NSE')
        if not nse_instruments:
            logging.error("Failed to fetch NSE instruments.")
            return
        logging.info(f"✓ Fetched {len(nse_instruments)} NSE instruments")

        # Resolve India VIX instrument token dynamically
        vix_aliases = ['INDIAVIX', 'INDIA VIX', 'INDA VIX', 'INDAVIX']
        resolved_vix_token = find_instrument_token_by_aliases(nse_instruments, vix_aliases, 'NSE')
        if resolved_vix_token:
            VIX_TOKEN = resolved_vix_token
            logging.info(f"✓ Resolved INDIAVIX token: {VIX_TOKEN}")
        else:
            VIX_TOKEN = VIX_FALLBACK_TOKEN
            if VIX_TOKEN:
                logging.warning(f"Unable to resolve INDIAVIX token. Using fallback token: {VIX_TOKEN}")
            else:
                logging.warning("Unable to resolve INDIAVIX token. VIX display will remain unavailable.")
        
        # Get NIFTY underlying token
        nifty_config = EXCHANGE_CONFIGS['NSE']
        nifty_token = get_instrument_token_for_symbol(
            nse_instruments, 
            nifty_config['underlying_symbol'], 
            nifty_config['ltp_exchange']
        )
        if not nifty_token:
            logging.error(f"Could not find token for {nifty_config['underlying_symbol']}.")
            return
        logging.info(f"✓ Found NIFTY token: {nifty_token}")
        
        # Get NIFTY nearest expiry
        nifty_expiry_info = get_nearest_weekly_expiry(
            nfo_instruments, 
            nifty_config['underlying_prefix'],
            nifty_config['options_exchange']
        )
        if not nifty_expiry_info:
            logging.error(f"Could not determine nearest expiry for NIFTY.")
            return
        logging.info(f"✓ NIFTY expiry: {nifty_expiry_info['expiry'].strftime('%d-%b-%Y')}")
    
        # Store NSE data
        exchange_instruments['NSE']['underlying_token'] = nifty_token
        exchange_instruments['NSE']['nfo_instruments'] = nfo_instruments
        exchange_instruments['NSE']['expiry_date'] = nifty_expiry_info['expiry']
        exchange_instruments['NSE']['symbol_prefix'] = nifty_expiry_info['symbol_prefix']
        
        # ==== BSE/SENSEX Instruments ====
        logging.info("\n" + "=" * 70)
        logging.info("FETCHING BSE/SENSEX INSTRUMENTS")
        logging.info("=" * 70)
        
        logging.info("Fetching BFO (SENSEX Options) instruments...")
        bfo_instruments = kite.instruments('BFO')
        if not bfo_instruments:
            logging.error("Failed to fetch BFO instruments.")
            return
        logging.info(f"✓ Fetched {len(bfo_instruments)} BFO instruments")
        
        logging.info("Fetching BSE instruments...")
        bse_instruments = kite.instruments('BSE')
        if not bse_instruments:
            logging.error("Failed to fetch BSE instruments.")
            return
        logging.info(f"✓ Fetched {len(bse_instruments)} BSE instruments")
        
        # Get SENSEX underlying token
        sensex_config = EXCHANGE_CONFIGS['BSE']
        sensex_token = get_instrument_token_for_symbol(
            bse_instruments, 
            sensex_config['underlying_symbol'], 
            sensex_config['ltp_exchange']
        )
        if not sensex_token:
            logging.error(f"Could not find token for {sensex_config['underlying_symbol']}.")
            return
        logging.info(f"✓ Found SENSEX token: {sensex_token}")
        
        # Get SENSEX nearest expiry
        sensex_expiry_info = get_nearest_weekly_expiry(
            bfo_instruments, 
            sensex_config['underlying_prefix'],
            sensex_config['options_exchange']
        )
        if not sensex_expiry_info:
            logging.error(f"Could not determine nearest expiry for SENSEX.")
            return
        logging.info(f"✓ SENSEX expiry: {sensex_expiry_info['expiry'].strftime('%d-%b-%Y')}")
    
        # Store BSE data
        exchange_instruments['BSE']['underlying_token'] = sensex_token
        exchange_instruments['BSE']['bfo_instruments'] = bfo_instruments
        exchange_instruments['BSE']['expiry_date'] = sensex_expiry_info['expiry']
        exchange_instruments['BSE']['symbol_prefix'] = sensex_expiry_info['symbol_prefix']
        
        # ==== Database Integration ====
        logging.info("\n" + "=" * 70)
        logging.info("CHECKING DATABASE FOR HISTORICAL DATA")
        logging.info("=" * 70)
        
        # Check if we should load from database for each exchange
        for exchange in ['NSE', 'BSE']:
            if db.should_load_from_db(exchange):
                logging.info(f"✓ {exchange}: Loading historical data from database...")
                oi_history[exchange] = db.load_today_snapshots(exchange)
                logging.info(f"  Loaded {sum(len(v) for v in oi_history[exchange].values())} records")
            else:
                logging.info(f"✓ {exchange}: Starting fresh (no recent data or gap > 30 min)")
        
        # Cleanup old data (30+ days)
        db.cleanup_old_data(days_to_keep=30)
        
        
        # ==== WebSocket Setup ====
        logging.info("\n" + "=" * 70)
        logging.info("SETTING UP WEBSOCKET CONNECTION")
        logging.info("=" * 70)
        
        user_id = profile.get('user_id')
        global current_enctoken
        kws_instance = KiteTicker(api_key="TradeViaPython", access_token=current_enctoken+"&user_id="+user_id)
        global kws
        kws = kws_instance
        
        def on_ticks(ws, ticks):
            """Handle incoming tick data for both exchanges."""
            global latest_tick_data, oi_history, latest_vix_data
            current_time = datetime.now()  # Use timezone-naive for consistency with historical API
            
            for tick in ticks:
                instrument_token = tick.get('instrument_token')
                if not instrument_token:
                    continue

                if VIX_TOKEN and instrument_token == VIX_TOKEN:
                    last_price = tick.get('last_price')
                    if last_price is not None:
                        latest_vix_data['value'] = last_price
                        latest_vix_data['timestamp'] = current_time
                    continue
                
                # Determine which exchange this token belongs to
                exchange = None
                if instrument_token == exchange_instruments['NSE']['underlying_token']:
                    exchange = 'NSE'
                elif instrument_token == exchange_instruments['BSE']['underlying_token']:
                    exchange = 'BSE'
                elif instrument_token in exchange_instruments['NSE'].get('option_tokens', []):
                    exchange = 'NSE'
                elif instrument_token in exchange_instruments['BSE'].get('option_tokens', []):
                    exchange = 'BSE'
                else:
                    # Try to identify by checking both token lists
                    for exch in ['NSE', 'BSE']:
                        if instrument_token in exchange_instruments[exch].get('option_tokens', []):
                            exchange = exch
                            break
                    
                    # If still not found, log and skip (don't default to NSE)
                    if exchange is None:
                        logging.debug(f"Received tick for uncategorized token {instrument_token}")
                        continue
                
                # Store tick data
                feed_payload = None
                latest_tick_data[exchange][instrument_token] = tick
                _record_tick_metadata(exchange, instrument_token, current_time)

                if instrument_token == exchange_instruments[exchange].get('underlying_token'):
                    feed_payload = _build_shared_feed_payload(exchange, tick, current_time)
                
                # Store OI history if available (MODE_FULL provides OI)
                if 'oi' in tick and tick['oi'] is not None:
                    if instrument_token not in oi_history[exchange]:
                        oi_history[exchange][instrument_token] = []
                    
                    # Add new OI record with timestamp
                    oi_history[exchange][instrument_token].append({
                        'date': current_time,
                        'oi': tick['oi']
                    })
                    
                    # Keep only last 60 minutes of OI history
                    cutoff_time = current_time - timedelta(minutes=60)
                    oi_history[exchange][instrument_token] = [
                        record for record in oi_history[exchange][instrument_token]
                        if record['date'] > cutoff_time
                    ]
                    
                    # Log first few OI entries to verify data accumulation
                    if len(oi_history[exchange][instrument_token]) <= 3:
                        logging.info(f"{exchange}: Accumulating OI for token {instrument_token}: {len(oi_history[exchange][instrument_token])} records")

                if feed_payload:
                    socketio.emit(SHARED_FEED_EVENT, feed_payload, namespace=SHARED_FEED_NAMESPACE)
    
        def on_connect(ws, response):
            """Subscribe to both NIFTY and SENSEX underlying tokens on connection."""
            global ws_connected
            ws_connected = True
            logging.info("✓ WebSocket connected!")
            
            # Subscribe to both underlying tokens
            nifty_token = exchange_instruments['NSE']['underlying_token']
            sensex_token = exchange_instruments['BSE']['underlying_token']
            
            tokens = [nifty_token, sensex_token]
            if VIX_TOKEN:
                tokens.append(VIX_TOKEN)
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_QUOTE, tokens)
            if VIX_TOKEN:
                logging.info(f"✓ Subscribed to NIFTY (token: {nifty_token}), SENSEX (token: {sensex_token}), and VIX (token: {VIX_TOKEN})")
            else:
                logging.info(f"✓ Subscribed to NIFTY (token: {nifty_token}) and SENSEX (token: {sensex_token})")
        
        def on_close(ws, code, reason):
            global ws_connected
            ws_connected = False
            logging.warning(f"⚠ WebSocket closed: {code} - {reason}")
        
        def on_error(ws, code, reason):
            logging.error(f"✗ WebSocket error: {code} - {reason}")
        
        kws.on_ticks = on_ticks
        kws.on_connect = on_connect
        kws.on_close = on_close
        kws.on_error = on_error
        
        kws.connect(threaded=True)
        
        # Wait for WebSocket
        logging.info("Waiting for WebSocket connection...")
        wait_count = 0
        while not kws.is_connected() and wait_count < 10:
            time.sleep(1)
            wait_count += 1
        
        if not kws.is_connected():
            logging.error("WebSocket failed to connect.")
            return
        
        logging.info("✓ WebSocket connected successfully!")
        time.sleep(3)  # Wait for initial tick data
        
        # ==== Start Background Data Update Threads ====
        logging.info("\n" + "=" * 70)
        logging.info("STARTING BACKGROUND DATA UPDATE THREADS")
        logging.info("=" * 70)
        
        # Start NSE/NIFTY update thread
        nse_thread = Thread(
            target=run_data_update_loop_exchange, 
            args=('NSE',),
            daemon=True,
            name='NSE-UpdateThread'
        )
        nse_thread.start()
        logging.info("✓ NSE (NIFTY) update thread started")
        
        # Start BSE/SENSEX update thread
        bse_thread = Thread(
            target=run_data_update_loop_exchange, 
            args=('BSE',),
            daemon=True,
            name='BSE-UpdateThread'
        )
        bse_thread.start()
        logging.info("✓ BSE (SENSEX) update thread started")
        
        # Initialize ML System if available
        if ML_SYSTEM_AVAILABLE:
            try:
                logging.info("\n" + "=" * 70)
                logging.info("INITIALIZING ML PREDICTION SYSTEM")
                logging.info("=" * 70)
                ml_predictor = RealTimePredictor()
                ml_signal_generator = SignalGenerator()
                ml_feature_engineer = FeatureEngineer()
                ml_predictor.load_models()
                ml_models_loaded = ml_predictor.is_loaded
                if ml_models_loaded:
                    logging.info("✓ ML models loaded successfully")
                else:
                    logging.warning("⚠️  ML models not found. Train models first using: python3 ml_system/test_phase2.py")
            except Exception as e:
                logging.warning(f"ML System initialization failed: {e}")
                ml_models_loaded = False
        
        logging.info("\n" + "=" * 70)
        logging.info("✓ SYSTEM INITIALIZED SUCCESSFULLY!")
        logging.info("=" * 70)
        logging.info(f"📊 Tracking: NIFTY (NSE) and SENSEX (BSE)")
        logging.info(f"🔄 Refresh interval: {UI_REFRESH_INTERVAL_SECONDS} seconds")
        if ml_models_loaded:
            logging.info("🤖 ML Predictions: Enabled")
        logging.info("💾 Database: Enabled (30-day retention)")
        logging.info("💰 Underlying prices: Saved with every refresh")
    
    except Exception as e:
        logging.error(f"Error during system initialization: {e}", exc_info=True)
        global authenticated
        authenticated = False

if __name__ == '__main__':
    try:
        # Initialize database
        db.initialize_database()

        # Start auto shutdown monitor before launching the server
        #start_auto_shutdown_monitor()
        
        print("\n🌐 Starting Flask-SocketIO server...")
        print("=" * 70)
        print("🔒 Login required at: http://localhost:5000/login")
        print("=" * 70)
        
        # Don't initialize system here - wait for login
        # System will be initialized after successful login via initialize_system_async()
        
        # Get server configuration from environment (with defaults)
        flask_host = os.getenv('FLASK_HOST', '127.0.0.1')
        flask_port = int(os.getenv('FLASK_PORT', 5000))
        
        # For production, allow_unsafe_werkzeug=True is needed (we're behind Nginx reverse proxy)
        socketio.run(app, host=flask_host, port=flask_port, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped by user (Ctrl+C)")
        logging.info("Server stopped by user")
    except Exception as e:
        print(f"\n✗ Critical error: {e}")
        logging.critical(f"Critical error: {e}", exc_info=True)
    finally:
        # Ensure cleanup happens in all scenarios
        print("\n🧹 Performing cleanup...")
        cleanup_connections()
        print("✓ Cleanup complete. Goodbye!")
        logging.info("Application shutdown complete")
