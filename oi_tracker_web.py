# Web-based OI Tracker with Real-time Updates
# This script provides a web interface for tracking Option Open Interest changes

import logging
import sys
import time
import pandas as pd
import signal
import atexit
import os
from datetime import datetime, date, timedelta, timezone
from threading import Thread, Lock
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from kite_trade import *
from kiteconnect import KiteTicker
import database as db
from dotenv import load_dotenv

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

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# --- Login Credentials (loaded from environment variables) ---
USER_ID = os.getenv('ZERODHA_USER_ID')
PASSWORD = os.getenv('ZERODHA_PASSWORD')

# Validate credentials are loaded
if not USER_ID or not PASSWORD:
    print("=" * 70)
    print("ERROR: Credentials not found!")
    print("=" * 70)
    print("\nPlease ensure you have created a .env file with:")
    print("  ZERODHA_USER_ID=your_client_id")
    print("  ZERODHA_PASSWORD=your_password")
    print("\nYou can copy .env.example to .env and fill in your details.")
    print("=" * 70)
    sys.exit(1)

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
OI_CHANGE_INTERVALS_MIN = (5, 10, 15, 30)

# --- Display and Logging ---
REFRESH_INTERVAL_SECONDS = 30  # Refresh interval for data updates
LOG_FILE_NAME = "oi_tracker_web.log"
FILE_LOG_LEVEL = "DEBUG"  # Changed to DEBUG for detailed diagnostics
PCT_CHANGE_THRESHOLDS = {
    5: 8.0,
    10: 10.0,
    15: 15.0,
    30: 25.0
}

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

# Global objects
kite = None
kws = None

# Multi-exchange data structures
latest_tick_data = {
    'NSE': {},
    'BSE': {}
}

oi_history = {
    'NSE': {},  # Store OI history from WebSocket ticks: {token: [(timestamp, oi), ...]}
    'BSE': {}
}

ws_connected = False
data_lock = Lock()
cleanup_done = False  # Flag to prevent duplicate cleanup

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
        'ml_prediction_pct': None
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
        'ml_prediction_pct': None
    }
}

# ML System objects (initialized later)
ml_predictor = None
ml_signal_generator = None
ml_feature_engineer = None
ml_models_loaded = False

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
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

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

def prepare_web_data(oi_report: dict, contract_details: dict, current_atm_strike: float, 
                     strike_step: int, num_strikes_each_side: int, exchange: str):
    """Prepares data structure for web display.
    Displays only num_strikes_each_side on each side, even though we fetched more.
    This ensures edge strikes have historical data from the buffer.
    """
    call_options = []
    put_options = []
    
    # Display only the requested number of strikes (buffer strikes not displayed)
    for i in range(-num_strikes_each_side, num_strikes_each_side + 1):
        strike_val = current_atm_strike + (i * strike_step)
        
        if i == 0: key_suffix = "atm"
        elif i < 0: key_suffix = f"itm{-i}"
        else: key_suffix = f"otm{i}"
        
        # Call option data
        option_key_ce = f"{key_suffix}_ce"
        ce_data = oi_report.get(option_key_ce, {})
        ce_contract = contract_details.get(option_key_ce, {})
        
        ce_latest_oi = ce_data.get('latest_oi')
        ce_latest_oi_time = ce_data.get('latest_oi_timestamp')
        
        # Get Call option LTP from tick data
        ce_token = ce_contract.get('instrument_token')
        ce_ltp = None
        if ce_token and ce_token in latest_tick_data[exchange]:
            ce_ltp = latest_tick_data[exchange][ce_token].get('last_price')
        
        # Add token to call row for database storage
        call_row = {
            'strike': int(ce_contract.get('strike', strike_val)),
            'symbol': ce_contract.get('tradingsymbol', 'N/A'),
            'latest_oi': ce_latest_oi,
            'oi_time': ce_latest_oi_time.strftime("%H:%M:%S") if ce_latest_oi_time else None,
            'strike_type': 'atm' if i == 0 else ('itm' if i < 0 else 'otm'),
            'ltp': ce_ltp,
            'token': ce_token,
            'pct_changes': {}
        }
        
        for interval in OI_CHANGE_INTERVALS_MIN:
            pct_change = ce_data.get(f'pct_diff_{interval}m')
            call_row['pct_changes'][f'{interval}m'] = pct_change
        
        call_options.append(call_row)
        
        # Put option data
        option_key_pe = f"{key_suffix}_pe"
        pe_data = oi_report.get(option_key_pe, {})
        pe_contract = contract_details.get(option_key_pe, {})
        
        pe_latest_oi = pe_data.get('latest_oi')
        pe_latest_oi_time = pe_data.get('latest_oi_timestamp')
        
        # Get Put option LTP from tick data
        pe_token = pe_contract.get('instrument_token')
        pe_ltp = None
        if pe_token and pe_token in latest_tick_data[exchange]:
            pe_ltp = latest_tick_data[exchange][pe_token].get('last_price')
        
        put_row = {
            'strike': int(pe_contract.get('strike', strike_val)),
            'symbol': pe_contract.get('tradingsymbol', 'N/A'),
            'latest_oi': pe_latest_oi,
            'oi_time': pe_latest_oi_time.strftime("%H:%M:%S") if pe_latest_oi_time else None,
            'strike_type': 'atm' if i == 0 else ('itm' if i > 0 else 'otm'),
            'ltp': pe_ltp,
            'token': pe_token,
            'pct_changes': {}
        }
        
        for interval in OI_CHANGE_INTERVALS_MIN:
            pct_change = pe_data.get(f'pct_diff_{interval}m')
            put_row['pct_changes'][f'{interval}m'] = pct_change
        
        put_options.append(put_row)
    
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
    
    logging.info(f"{exchange}: Data update thread started")
    
    while True:
        try:
            logging.info(f"{exchange}: Starting data update iteration")
            
            # Get current ATM strike
            current_atm_strike = get_atm_strike(
                underlying_token, 
                config['strike_difference'],
                exchange
            )
            
            if not current_atm_strike:
                with data_lock:
                    latest_oi_data[exchange]['status'] = 'Error: Could not determine ATM strike'
                time.sleep(REFRESH_INTERVAL_SECONDS)
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
                time.sleep(REFRESH_INTERVAL_SECONDS)
                continue
            
            # Extract option tokens
            option_tokens = [details['instrument_token'] for details in option_contract_details.values() 
                           if 'instrument_token' in details]
            
            # Dynamic token subscription (subscribe/unsubscribe based on ATM changes)
            if option_tokens:
                update_subscribed_tokens(option_tokens, exchange)
            
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
                exchange
            )
            
            # Get underlying LTP
            underlying_ltp = None
            if underlying_token in latest_tick_data[exchange]:
                underlying_ltp = latest_tick_data[exchange][underlying_token].get('last_price')
            
            # Calculate Put Call Ratio (PCR) = Sum of PUT OI / Sum of CALL OI
            total_put_oi = sum(opt['latest_oi'] for opt in put_options if opt['latest_oi'] is not None)
            total_call_oi = sum(opt['latest_oi'] for opt in call_options if opt['latest_oi'] is not None)
            pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else None
            
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
                latest_oi_data[exchange]['underlying_price'] = int(underlying_ltp) if underlying_ltp else None
                latest_oi_data[exchange]['last_update'] = datetime.now().strftime('%H:%M:%S')
                latest_oi_data[exchange]['status'] = 'Live'
                latest_oi_data[exchange]['pcr'] = pcr
                if ml_prediction_data:
                    latest_oi_data[exchange]['ml_prediction'] = ml_prediction_data['prediction']
                    latest_oi_data[exchange]['ml_prediction_pct'] = ml_prediction_data['prediction_pct']
                    latest_oi_data[exchange]['ml_signal'] = ml_prediction_data['signal']
                    latest_oi_data[exchange]['ml_confidence'] = ml_prediction_data['confidence']
                    latest_oi_data[exchange]['ml_strength'] = ml_prediction_data['strength']
                    latest_oi_data[exchange]['ml_reasoning'] = ml_prediction_data['reasoning']
            
            # Save to database (including underlying price and ATM strike)
            current_timestamp = datetime.now()
            db.save_option_chain_snapshot(
                exchange, 
                call_options, 
                put_options, 
                underlying_price=underlying_ltp,
                atm_strike=current_atm_strike,
                timestamp=current_timestamp
            )
            
            # Emit update to all connected clients (include positions data)
            with data_lock:
                emit_data = latest_oi_data[exchange].copy()
                emit_data['open_positions'] = list(open_positions[exchange].values())
                emit_data['total_mtm'] = total_mtm[exchange]
                emit_data['closed_pnl'] = closed_positions_pnl[exchange]
            
            # Emit exchange-specific event
            socketio.emit(f'data_update_{exchange}', emit_data)
            
            logging.info(f"{exchange}: ✓ Data updated. ATM: {current_atm_strike}, PCR: {pcr}")
            time.sleep(REFRESH_INTERVAL_SECONDS)
            
        except Exception as e:
            logging.error(f"{exchange}: Error in data update loop: {e}", exc_info=True)
            with data_lock:
                latest_oi_data[exchange]['status'] = f'Error: {str(e)}'
            time.sleep(REFRESH_INTERVAL_SECONDS)

# ==============================================================================
# --- FLASK ROUTES ---
# ==============================================================================

@app.route('/')
def index():
    """Main page route - now supports dual exchange tabs."""
    return render_template('index.html', 
                          exchanges=['NSE', 'BSE'],
                          exchange_configs=EXCHANGE_CONFIGS,
                          intervals=OI_CHANGE_INTERVALS_MIN,
                          thresholds=PCT_CHANGE_THRESHOLDS)

@app.route('/backtest')
def backtest_page():
    """Backtesting page route."""
    return render_template('backtest.html',
                          exchanges=['NSE', 'BSE'],
                          exchange_configs=EXCHANGE_CONFIGS)

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
                        predictions = trainer.models['random_forest'].predict(X_test_scaled)
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
# --- MAIN INITIALIZATION ---
# ==============================================================================

def initialize_system():
    """Initialize the trading system with dual exchange support and start background threads."""
    global kite, kws, exchange_instruments, latest_oi_data, oi_history
    global ml_predictor, ml_signal_generator, ml_feature_engineer, ml_models_loaded
    
    print("=" * 70)
    print("DUAL EXCHANGE OI TRACKER - NSE (NIFTY) & BSE (SENSEX)")
    print("=" * 70)
    print("\n🔒 Credentials loaded from environment variables (.env file)")
    
    # Get 2FA input
    print(f"\nLogin ID: {USER_ID}")
    twofa_input = input("Enter your 2FA PIN or TOTP code: ").strip()
    if not twofa_input:
        print("2FA code is required. Exiting.")
        sys.exit(1)
    
    # Get enctoken
    print("\nAuthenticating with Zerodha...")
    enctoken = get_enctoken(USER_ID, PASSWORD, twofa_input)
    if not enctoken:
        print("Failed to get enctoken. Check your credentials.")
        sys.exit(1)
    
    print("✓ Successfully obtained enctoken!")
    
    # Initialize KiteApp
    kite = KiteApp(enctoken=enctoken)
    print("✓ KiteApp initialized!")
    
    # Verify connection
    profile = kite.profile()
    print(f"✓ Connected as: {profile.get('user_id')} ({profile.get('user_name')})")
    
    # ==== NSE/NIFTY Instruments ====
    print("\n" + "=" * 70)
    print("FETCHING NSE/NIFTY INSTRUMENTS")
    print("=" * 70)
    
    print("Fetching NFO (NIFTY Options) instruments...")
    nfo_instruments = kite.instruments('NFO')
    if not nfo_instruments:
        print("Failed to fetch NFO instruments. Exiting.")
        sys.exit(1)
    print(f"✓ Fetched {len(nfo_instruments)} NFO instruments")
    
    print("Fetching NSE instruments...")
    nse_instruments = kite.instruments('NSE')
    if not nse_instruments:
        print("Failed to fetch NSE instruments. Exiting.")
        sys.exit(1)
    print(f"✓ Fetched {len(nse_instruments)} NSE instruments")
    
    # Get NIFTY underlying token
    nifty_config = EXCHANGE_CONFIGS['NSE']
    nifty_token = get_instrument_token_for_symbol(
        nse_instruments, 
        nifty_config['underlying_symbol'], 
        nifty_config['ltp_exchange']
    )
    if not nifty_token:
        print(f"Could not find token for {nifty_config['underlying_symbol']}. Exiting.")
        sys.exit(1)
    print(f"✓ Found NIFTY token: {nifty_token}")
    
    # Get NIFTY nearest expiry
    nifty_expiry_info = get_nearest_weekly_expiry(
        nfo_instruments, 
        nifty_config['underlying_prefix'],
        nifty_config['options_exchange']
    )
    if not nifty_expiry_info:
        print(f"Could not determine nearest expiry for NIFTY. Exiting.")
        sys.exit(1)
    print(f"✓ NIFTY expiry: {nifty_expiry_info['expiry'].strftime('%d-%b-%Y')}")
    
    # Store NSE data
    exchange_instruments['NSE']['underlying_token'] = nifty_token
    exchange_instruments['NSE']['nfo_instruments'] = nfo_instruments
    exchange_instruments['NSE']['expiry_date'] = nifty_expiry_info['expiry']
    exchange_instruments['NSE']['symbol_prefix'] = nifty_expiry_info['symbol_prefix']
    
    # ==== BSE/SENSEX Instruments ====
    print("\n" + "=" * 70)
    print("FETCHING BSE/SENSEX INSTRUMENTS")
    print("=" * 70)
    
    print("Fetching BFO (SENSEX Options) instruments...")
    bfo_instruments = kite.instruments('BFO')
    if not bfo_instruments:
        print("Failed to fetch BFO instruments. Exiting.")
        sys.exit(1)
    print(f"✓ Fetched {len(bfo_instruments)} BFO instruments")
    
    print("Fetching BSE instruments...")
    bse_instruments = kite.instruments('BSE')
    if not bse_instruments:
        print("Failed to fetch BSE instruments. Exiting.")
        sys.exit(1)
    print(f"✓ Fetched {len(bse_instruments)} BSE instruments")
    
    # Get SENSEX underlying token
    sensex_config = EXCHANGE_CONFIGS['BSE']
    sensex_token = get_instrument_token_for_symbol(
        bse_instruments, 
        sensex_config['underlying_symbol'], 
        sensex_config['ltp_exchange']
    )
    if not sensex_token:
        print(f"Could not find token for {sensex_config['underlying_symbol']}. Exiting.")
        sys.exit(1)
    print(f"✓ Found SENSEX token: {sensex_token}")
    
    # Get SENSEX nearest expiry
    sensex_expiry_info = get_nearest_weekly_expiry(
        bfo_instruments, 
        sensex_config['underlying_prefix'],
        sensex_config['options_exchange']
    )
    if not sensex_expiry_info:
        print(f"Could not determine nearest expiry for SENSEX. Exiting.")
        sys.exit(1)
    print(f"✓ SENSEX expiry: {sensex_expiry_info['expiry'].strftime('%d-%b-%Y')}")
    
    # Store BSE data
    exchange_instruments['BSE']['underlying_token'] = sensex_token
    exchange_instruments['BSE']['bfo_instruments'] = bfo_instruments
    exchange_instruments['BSE']['expiry_date'] = sensex_expiry_info['expiry']
    exchange_instruments['BSE']['symbol_prefix'] = sensex_expiry_info['symbol_prefix']
    
    # ==== Database Integration ====
    print("\n" + "=" * 70)
    print("CHECKING DATABASE FOR HISTORICAL DATA")
    print("=" * 70)
    
    # Check if we should load from database for each exchange
    for exchange in ['NSE', 'BSE']:
        if db.should_load_from_db(exchange):
            print(f"✓ {exchange}: Loading historical data from database...")
            oi_history[exchange] = db.load_today_snapshots(exchange)
            print(f"  Loaded {sum(len(v) for v in oi_history[exchange].values())} records")
        else:
            print(f"✓ {exchange}: Starting fresh (no recent data or gap > 30 min)")
    
    # Cleanup old data (30+ days)
    db.cleanup_old_data(days_to_keep=30)
    
    
    # ==== WebSocket Setup ====
    print("\n" + "=" * 70)
    print("SETTING UP WEBSOCKET CONNECTION")
    print("=" * 70)
    
    user_id = profile.get('user_id')
    kws = KiteTicker(api_key="TradeViaPython", access_token=enctoken+"&user_id="+user_id)
    
    def on_ticks(ws, ticks):
        """Handle incoming tick data for both exchanges."""
        global latest_tick_data, oi_history
        current_time = datetime.now()  # Use timezone-naive for consistency with historical API
        
        for tick in ticks:
            instrument_token = tick.get('instrument_token')
            if not instrument_token:
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
            latest_tick_data[exchange][instrument_token] = tick
            
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
    
    def on_connect(ws, response):
        """Subscribe to both NIFTY and SENSEX underlying tokens on connection."""
        global ws_connected
        ws_connected = True
        print("✓ WebSocket connected!")
        
        # Subscribe to both underlying tokens
        nifty_token = exchange_instruments['NSE']['underlying_token']
        sensex_token = exchange_instruments['BSE']['underlying_token']
        
        ws.subscribe([nifty_token, sensex_token])
        ws.set_mode(ws.MODE_QUOTE, [nifty_token, sensex_token])
        print(f"✓ Subscribed to NIFTY (token: {nifty_token}) and SENSEX (token: {sensex_token})")
    
    def on_close(ws, code, reason):
        global ws_connected
        ws_connected = False
        print(f"⚠ WebSocket closed: {code} - {reason}")
        logging.warning(f"WebSocket closed: {code} - {reason}")
    
    def on_error(ws, code, reason):
        print(f"✗ WebSocket error: {code} - {reason}")
        logging.error(f"WebSocket error: {code} - {reason}")
    
    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close
    kws.on_error = on_error
    
    kws.connect(threaded=True)
    
    # Wait for WebSocket
    print("Waiting for WebSocket connection...")
    wait_count = 0
    while not kws.is_connected() and wait_count < 10:
        time.sleep(1)
        wait_count += 1
    
    if not kws.is_connected():
        print("WebSocket failed to connect. Exiting.")
        sys.exit(1)
    
    print("✓ WebSocket connected successfully!")
    time.sleep(3)  # Wait for initial tick data
    
    # ==== Start Background Data Update Threads ====
    print("\n" + "=" * 70)
    print("STARTING BACKGROUND DATA UPDATE THREADS")
    print("=" * 70)
    
    # Start NSE/NIFTY update thread
    nse_thread = Thread(
        target=run_data_update_loop_exchange, 
        args=('NSE',),
        daemon=True,
        name='NSE-UpdateThread'
    )
    nse_thread.start()
    print("✓ NSE (NIFTY) update thread started")
    
    # Start BSE/SENSEX update thread
    bse_thread = Thread(
        target=run_data_update_loop_exchange, 
        args=('BSE',),
        daemon=True,
        name='BSE-UpdateThread'
    )
    bse_thread.start()
    print("✓ BSE (SENSEX) update thread started")
    
    # Initialize ML System if available
    if ML_SYSTEM_AVAILABLE:
        try:
            print("\n" + "=" * 70)
            print("INITIALIZING ML PREDICTION SYSTEM")
            print("=" * 70)
            ml_predictor = RealTimePredictor()
            ml_signal_generator = SignalGenerator()
            ml_feature_engineer = FeatureEngineer()
            ml_predictor.load_models()
            ml_models_loaded = ml_predictor.is_loaded
            if ml_models_loaded:
                print("✓ ML models loaded successfully")
            else:
                print("⚠️  ML models not found. Train models first using: python3 ml_system/test_phase2.py")
        except Exception as e:
            logging.warning(f"ML System initialization failed: {e}")
            ml_models_loaded = False
    
    print("\n" + "=" * 70)
    print("✓ SYSTEM INITIALIZED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n🌐 Web interface will be available at: http://localhost:5000")
    print("📊 Tracking: NIFTY (NSE) and SENSEX (BSE)")
    print(f"🔄 Refresh interval: {REFRESH_INTERVAL_SECONDS} seconds")
    if ml_models_loaded:
        print("🤖 ML Predictions: Enabled")
    print("💾 Database: Enabled (30-day retention)")
    print("💰 Underlying prices: Saved with every refresh")
    print("\nPress Ctrl+C to stop the server\n")

if __name__ == '__main__':
    try:
        initialize_system()
        print("\n🌐 Starting Flask-SocketIO server...")
        print("=" * 70)
        
        # Get server configuration from environment (with defaults)
        flask_host = os.getenv('FLASK_HOST', '0.0.0.0')
        flask_port = int(os.getenv('FLASK_PORT', 5555))
        
        socketio.run(app, host=flask_host, port=flask_port, debug=False, use_reloader=False)
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

