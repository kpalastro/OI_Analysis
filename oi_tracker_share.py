# Script to track option OI changes
# This script connects to the Kite API, fetches NFO option chain data for a specified underlying,
# calculates Open Interest (OI) changes over various intervals, and displays this information
# in live-updating tables in the console.

import logging
import os
import sys  # For sys.exit() for critical error handling
import time # For time.sleep() in the live update loop
import json
import pandas as pd
from pandas import json_normalize
from kite_trade import *
from datetime import datetime, date, timedelta, timezone
from kiteconnect import KiteTicker
from dotenv import load_dotenv

# Rich library imports for enhanced terminal output
from rich.console import Console, Group
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.panel import Panel

# Load environment variables from a .env file if present
load_dotenv()

# ==============================================================================
# --- SCRIPT CONFIGURATION ---
# ==============================================================================
# Instructions for Kite Trade Login:
# 1. Set your Zerodha login credentials below:
#    - USER_ID: Your Zerodha login ID
#    - PASSWORD: Your Zerodha login password
#    - TWOFA: Will be requested via user input when script runs (for security)
# 2. These credentials are used to get enctoken for API access
# 3. No API key/secret required with this method

# --- Login Credentials ---
USER_ID = os.getenv("ZERODHA_USER_ID", "")           # Your Zerodha Login ID
PASSWORD = os.getenv("ZERODHA_PASSWORD", "")         # Your Zerodha Login Password
# TWOFA will be requested at runtime via user input

# --- Trading Parameters ---
# -- NIFTY Parameters --
UNDERLYING_SYMBOL = os.getenv("UNDERLYING_SYMBOL", "NIFTY 50")       # Underlying instrument (e.g., "NIFTY 50", "NIFTY BANK")
STRIKE_DIFFERENCE = int(os.getenv("STRIKE_DIFFERENCE", "50"))         # Difference between consecutive strikes (50 for NIFTY, 100 for BANKNIFTY)
OPTIONS_COUNT = int(os.getenv("OPTIONS_COUNT", "5"))                   # Number of ITM/OTM strikes on each side of ATM
# --- Exchange Configuration ---
EXCHANGE_NFO_OPTIONS = os.getenv("EXCHANGE_NFO_OPTIONS", "NFO")  # Exchange for NFO options contracts
EXCHANGE_LTP = os.getenv("EXCHANGE_LTP", "NSE")                  # Exchange for fetching LTP of the underlying


# --- Trading Parameters ---
# --  SENSEX Parameters --
#UNDERLYING_SYMBOL = "SENSEX"       # Underlying instrument (e.g., "NIFTY 50", "NIFTY BANK")
#STRIKE_DIFFERENCE = 100               # Difference between consecutive strikes (50 for NIFTY, 100 for BANKNIFTY)
#OPTIONS_COUNT = 5                    # Number of ITM/OTM strikes to fetch on each side of ATM (e.g., 2 means 2 ITM, 1 ATM, 2 OTM = 5 levels total)
## --- Exchange Configuration ---
#EXCHANGE_NFO_OPTIONS = "BFO"  # Exchange for BFO options contracts
#EXCHANGE_LTP = "BSE"      # Exchange for fetching LTP of the underlying (e.g., BSE for SENSEX)
 


# --- Data Fetching Parameters ---
HISTORICAL_DATA_MINUTES = int(os.getenv("HISTORICAL_DATA_MINUTES", "40"))  # Minutes of historical data for OI calculation

def _parse_int_tuple(env_value: str, default_tuple: tuple) -> tuple:
    try:
        if not env_value:
            return default_tuple
        items = [int(x.strip()) for x in env_value.split(',') if x.strip()]
        return tuple(items) if items else default_tuple
    except Exception:
        return default_tuple

OI_CHANGE_INTERVALS_MIN = _parse_int_tuple(os.getenv("OI_CHANGE_INTERVALS_MIN"), (5, 10, 15, 30))

# --- Display and Logging ---
REFRESH_INTERVAL_SECONDS = int(os.getenv("REFRESH_INTERVAL_SECONDS", "60"))  # Refresh interval in seconds
LOG_FILE_NAME = os.getenv("LOG_FILE_NAME", "oi_tracker.log")                  # Log file name
FILE_LOG_LEVEL = os.getenv("FILE_LOG_LEVEL", "DEBUG, INFO")                   # Logging level string

def _parse_thresholds(env_value: str, default_value: dict) -> dict:
    if not env_value:
        return default_value
    try:
        loaded = json.loads(env_value)
        # Convert string keys to ints if necessary
        return {int(k): float(v) for k, v in loaded.items()}
    except Exception:
        return default_value

PCT_CHANGE_THRESHOLDS = _parse_thresholds(
    os.getenv("PCT_CHANGE_THRESHOLDS"),
    {
        5: 8.0,
        10: 10.0,
        15: 15.0,
        30: 25.0,
    },
)

# Optional siren sound path for alerts; if not provided or not found, sound is skipped
SIREN_SOUND_PATH = os.getenv("SIREN_SOUND_PATH", "")

# Optional Kite WebSocket API key
KITE_API_KEY = os.getenv("KITE_API_KEY", "TradeViaPython")

# Note: Console output is managed by Rich; file logging captures more detailed/background info.

# ==============================================================================
# --- END OF CONFIGURATION ---
# ==============================================================================

# --- Global Initializations ---
# Setup file logging according to configured level and file name
logging.basicConfig(
    level=getattr(logging, FILE_LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    filename=LOG_FILE_NAME,
    filemode='a'  # Append to the log file
)

# Derive underlying prefix (e.g., NIFTY from "NIFTY 50") for instrument searching

now = datetime.now()
current_year_two_digits = now.strftime("%y")
UNDERLYING_PREFIX = UNDERLYING_SYMBOL.split(" ")[0].upper()

# Global KiteApp and Rich Console instances
# Will be initialized after getting enctoken in main()
kite = None  # Will be initialized in main() function
kws = None   # WebSocket connection (if needed)
console = Console() # For Rich text and table display

# Global dictionary to store latest WebSocket tick data
# Key: instrument_token, Value: tick data dictionary
latest_tick_data = {}
ws_connected = False


# --- Core Functions ---

def get_instrument_token_for_symbol(instruments: list, symbol: str, exchange: str):
    """
    Gets the instrument token for a given symbol and exchange.
    
    Args:
        instruments: List of all instruments
        symbol: Trading symbol (e.g., "NIFTY 50", "NIFTY BANK")
        exchange: Exchange name (e.g., "NSE", "NFO")
    
    Returns:
        Instrument token as integer, or None if not found
    """
    try:
        for inst in instruments:
            if inst['tradingsymbol'] == symbol and inst['exchange'] == exchange:
                return inst['instrument_token']
        logging.warning(f"Instrument token not found for {exchange}:{symbol}")
        return None
    except Exception as e:
        logging.error(f"Error finding instrument token for {symbol}: {e}", exc_info=True)
        return None

def get_atm_strike(underlying_token: int, strike_diff: int):
    """
    Fetches the Last Traded Price (LTP) from WebSocket data and calculates the At-The-Money (ATM) strike.

    Args:
        underlying_token: Instrument token for the underlying (e.g., NIFTY 50 token).
        strike_diff: The difference between consecutive option strikes.

    Returns:
        The calculated ATM strike as a float, or None if LTP cannot be fetched or an error occurs.
    """
    global latest_tick_data
    
    try:
        # Get last price from WebSocket tick data
        if underlying_token not in latest_tick_data:
            logging.error(f"No tick data available for token {underlying_token}")
            return None
        
        tick_info = latest_tick_data[underlying_token]
        ltp = tick_info.get('last_price')
        
        if ltp is None:
            logging.error(f"Last price not found in tick data for token {underlying_token}")
            return None
        
        # Calculate ATM strike by rounding LTP to the nearest strike difference
        atm_strike = round(ltp / strike_diff) * strike_diff
        logging.debug(f"LTP from WebSocket: {ltp}, Calculated ATM strike: {atm_strike}")
        return atm_strike
    except Exception as e:
        logging.error(f"Error in get_atm_strike: {e}", exc_info=True)
        return None

def get_nearest_weekly_expiry(instruments: list, underlying_prefix_str: str):
    """
    Finds the nearest future weekly expiry date for the given underlying symbol prefix from a list of instruments.

    Args:
        instruments: A list of instrument dictionaries from Kite API.
        underlying_prefix_str: The prefix of the underlying symbol (e.g., "NIFTY").

    Returns:
        The nearest weekly expiry date as a datetime.date object, or None if no suitable expiry is found.
    """
    today = date.today()
    possible_expiries = set()
    logging.info(f"Searching for nearest weekly expiry for {underlying_prefix_str} among {len(instruments)} instruments.")

    #print(instruments)
    #print(underlying_prefix_str)
    trading_symbol = {} 

    for inst in instruments:
        # Filter for options of the specified underlying
        if inst['name'] == underlying_prefix_str and inst['exchange'] == EXCHANGE_NFO_OPTIONS:
            # Ensure expiry is a date object and is in the future or today
            if isinstance(inst['expiry'], date) and inst['expiry'] >= today:
                possible_expiries.add(inst['expiry'])
                trading_symbol[inst['expiry']] = inst['tradingsymbol']

    #print(possible_expiries) 
    if not possible_expiries:
        logging.error(f"No future expiries found for {underlying_prefix_str}.")
        return None

    # Sort expiries and return the closest one
    nearest_expiry = sorted(list(possible_expiries))[0]
    trading_symbol_of_nearest_expiry = trading_symbol[nearest_expiry]

    symbol_prefix = trading_symbol_of_nearest_expiry[0:len(underlying_prefix_str)+5]
    #print("Trading_symbol = "+symbol_prefix)
    logging.info(f"Nearest weekly expiry for {underlying_prefix_str}: {nearest_expiry}")
    return {"expiry":nearest_expiry, "symbol_prefix":symbol_prefix}

def get_relevant_option_details(instruments: list, atm_strike_val: float, expiry_dt: date, 
                                strike_diff_val: int, opt_count: int, underlying_prefix_str: str, symbol_prefix: str):
    """
    Identifies relevant ITM, ATM, and OTM Call/Put option contract details (tradingsymbol, instrument_token, strike)
    for a given ATM strike and expiry date.

    Args:
        instruments: List of all NFO instrument dictionaries.
        atm_strike_val: The current At-The-Money strike.
        expiry_dt: The expiry date for the options.
        strike_diff_val: The difference between option strikes.
        opt_count: Number of ITM/OTM strikes to fetch on each side of ATM.
        underlying_prefix_str: The prefix of the underlying (e.g., "NIFTY").

    Returns:
        A dictionary where keys are like "atm_ce", "itm1_pe", etc., and values are
        dictionaries containing 'tradingsymbol', 'instrument_token', and 'strike'.
        Returns an empty dictionary if critical inputs are missing.
    """
    relevant_options = {}
    if not expiry_dt or atm_strike_val is None:
        logging.error("Expiry date or ATM strike is None, cannot fetch option details.")
        return relevant_options

    # Format expiry date for Zerodha's trading symbol convention (e.g., NIFTY23OCT19500CE)
    # Year: last two digits. Month: 3-letter uppercase. Day: two digits.
    expiry_str_part = expiry_dt.strftime("%y%b%d").upper() 
    logging.debug(f"Searching for options with expiry: {expiry_dt}, ATM strike: {atm_strike_val}")

    # Iterate from -opt_count (deep ITM for calls / deep OTM for puts) to +opt_count
    for i in range(-opt_count, opt_count + 1):
        current_strike = atm_strike_val + (i * strike_diff_val)
        
        # Construct core part of trading symbols for CE and PE to aid matching
        ce_symbol_pattern_core = f"{symbol_prefix}{int(current_strike)}"
        pe_symbol_pattern_core = f"{symbol_prefix}{int(current_strike)}"

        #print(ce_symbol_pattern_core)
        
        found_ce, found_pe = None, None
        # Search through all instruments for matches
        for inst in instruments:
            # Match instrument by name, strike, expiry date, and echange 
            if inst['name'] == underlying_prefix_str and \
               inst['strike'] == current_strike and \
               inst['expiry'] == expiry_dt and \
               inst['exchange'] == EXCHANGE_NFO_OPTIONS:
                
                # Further match by instrument type (CE/PE) and ensure core symbol pattern is present
                if inst['instrument_type'] == 'CE' and ce_symbol_pattern_core in inst['tradingsymbol']:
                    found_ce = inst
                elif inst['instrument_type'] == 'PE' and pe_symbol_pattern_core in inst['tradingsymbol']:
                    found_pe = inst
            
            # Optimization: if both CE and PE found for this strike, no need to search further for this strike
            if found_ce and found_pe:
                break 
        
        # Determine key suffix (atm, itm1, otm1, etc.) based on position relative to ATM
        if i == 0: key_suffix = "atm"
        elif i < 0: key_suffix = f"itm{-i}" # e.g., i=-1 -> itm1 (strike < ATM)
        else: key_suffix = f"otm{i}"       # e.g., i=1  -> otm1 (strike > ATM)
        
        if found_ce:
            relevant_options[f"{key_suffix}_ce"] = {
                'tradingsymbol': found_ce['tradingsymbol'], 
                'instrument_token': found_ce['instrument_token'], 
                'strike': current_strike
            }
        else:
            logging.warning(f"CE option not found for strike {current_strike}, expiry {expiry_dt}")
        
        if found_pe:
            relevant_options[f"{key_suffix}_pe"] = {
                'tradingsymbol': found_pe['tradingsymbol'], 
                'instrument_token': found_pe['instrument_token'], 
                'strike': current_strike
            }
        else:
            logging.warning(f"PE option not found for strike {current_strike}, expiry {expiry_dt}")
            
    logging.debug(f"Relevant option details identified: {len(relevant_options)} contracts.")
    return relevant_options

def fetch_historical_oi_data(kite_obj, option_details_dict: dict, 
                             minutes_of_data: int = HISTORICAL_DATA_MINUTES):
    """
    Fetches historical OI data (minute interval) for the provided option contracts.

    Args:
        kite_obj: Initialized KiteApp object.
        option_details_dict: Dictionary of option contracts (from get_relevant_option_details).
        minutes_of_data: The duration in minutes for which to fetch historical data.

    Returns:
        A dictionary where keys are option keys (e.g., "atm_ce") and values are lists
        of historical candle data (each candle is a dict). Returns empty list for a contract on error.
    """
    historical_oi_store = {}
    if not option_details_dict:
        logging.warning("No option details provided to fetch_historical_oi_data.")
        return historical_oi_store

    # Calculate date range for historical data API call
    to_date = datetime.now()  # Current local time 
    from_date = to_date - timedelta(minutes=minutes_of_data)

    logging.debug(f"Fetching historical data from {from_date} to {to_date}")

    for option_key, details in option_details_dict.items():
        instrument_token = details.get('instrument_token')
        tradingsymbol = details.get('tradingsymbol')

        if not instrument_token:
            logging.warning(f"Missing instrument_token for {option_key} ({tradingsymbol}). Skipping historical data fetch.")
            historical_oi_store[option_key] = []  # Store empty list for consistency
            continue
        
        try:
            logging.debug(f"Fetching historical OI for {tradingsymbol} (Token: {instrument_token})")
            # Fetch minute-interval data including Open Interest (oi=True)
            # Using kite_trade's historical_data method
            raw_data = kite_obj.historical_data(instrument_token, from_date, to_date, 'minute', continuous=False, oi=True)
            
            # Convert to list of dicts format expected by downstream functions
            if isinstance(raw_data, pd.DataFrame):
                data = raw_data.to_dict('records')
            else:
                data = raw_data if raw_data else []
            
            historical_oi_store[option_key] = data
            logging.debug(f"Fetched {len(data)} records for {tradingsymbol}")
        except Exception as e:
            logging.error(f"Error fetching historical OI for {tradingsymbol} (Token: {instrument_token}): {e}", exc_info=True)
            historical_oi_store[option_key] = []  # Store empty list on error to prevent crashes downstream

    return historical_oi_store

def find_oi_at_timestamp(historical_candles: list, target_time: datetime, 
                          latest_oi_and_time: tuple):
    """
    Finds Open Interest (OI) at or just before a specific target_time from a list of historical candles.
    The historical_candles are assumed to be sorted oldest to newest.

    Args:
        historical_candles: List of candle dictionaries (from Kite API, containing 'date' and 'oi').
        target_time: The target datetime object (timezone-aware) to find OI for.
        latest_oi_and_time: Optional tuple (latest_oi, latest_timestamp). If provided, ensures
                            that the selected candle is not later than this latest_timestamp.
                            This prevents looking "into the future" if target_time is very recent
                            and slightly ahead of the last available candle.

    Returns:
        The Open Interest (int) at the target time, or None if no suitable candle is found.
    """
    if not historical_candles:
        return None

    # Iterate backwards through candles to find the most recent one at or before target_time
    for candle in reversed(historical_candles):
        candle_time = candle['date']  # 'date' field from Kite API is already a timezone-aware datetime object

        if candle_time <= target_time:
            # If latest_oi_and_time is provided, ensure we don't pick a candle
            # whose timestamp is later than the latest_oi_timestamp from the most current data point.
            if latest_oi_and_time and candle_time > latest_oi_and_time[1]:
                continue  # This candle is too new compared to the reference latest OI point
            return candle.get('oi')
            
    # If loop completes, no candle was found at or before target_time (or before the first candle)
    return None

def calculate_oi_differences(raw_historical_data_store: dict, intervals_min: tuple):
    """
    Calculates OI differences between the latest OI and OI at specified past intervals.

    Args:
        raw_historical_data_store: Dictionary of historical candle data for various option contracts.
        intervals_min: A tuple of time intervals in minutes (e.g., (10, 15, 30)) for which to calculate OI change.

    Returns:
        A dictionary structured by option_key, containing 'latest_oi', 'latest_oi_timestamp',
        and 'diff_Xm' for each interval X.
    """
    oi_differences_report = {}
    # Use a consistent, timezone-aware current time for all calculations in this batch
    current_processing_time = datetime.now(timezone.utc)
    logging.debug(f"Calculating OI differences based on current time: {current_processing_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    for option_key, candles_list in raw_historical_data_store.items():
        oi_differences_report[option_key] = {}
        
        latest_oi, latest_oi_timestamp = None, None
        if candles_list:
            # Candles are sorted oldest to newest by API; the last one is the latest.
            latest_candle = candles_list[-1]
            latest_oi = latest_candle.get('oi')
            latest_oi_timestamp = latest_candle.get('date') # This is a datetime object
        
        oi_differences_report[option_key]['latest_oi'] = latest_oi
        oi_differences_report[option_key]['latest_oi_timestamp'] = latest_oi_timestamp

        # If latest_oi is None (e.g., no data for the contract), cannot calculate differences
        if latest_oi is None:
            for interval in intervals_min:
                oi_differences_report[option_key][f'abs_diff_{interval}m'] = None
                oi_differences_report[option_key][f'pct_diff_{interval}m'] = None
            continue # Move to the next option contract

        # Calculate OI at different past intervals
        for interval in intervals_min:
            target_past_time = current_processing_time - timedelta(minutes=interval)
            
            past_oi = find_oi_at_timestamp(
                candles_list,
                target_past_time,
                latest_oi_and_time=(latest_oi, latest_oi_timestamp) # Pass current latest OI info
            )
            
            abs_oi_diff = None
            pct_oi_change = None
            if past_oi is not None:
                abs_oi_diff = latest_oi - past_oi
                if past_oi != 0: # Avoid division by zero for percentage change
                    pct_oi_change = (abs_oi_diff / past_oi) * 100
                # else: pct_oi_change remains None if past_oi is 0 but abs_oi_diff is not (Infinite change)
            else:
                logging.debug(f"Could not find past OI for {option_key} at {interval}m prior ({target_past_time.strftime('%H:%M:%S %Z')}). abs_oi_diff and pct_oi_change will be None.")
            
            oi_differences_report[option_key][f'abs_diff_{interval}m'] = abs_oi_diff
            oi_differences_report[option_key][f'pct_diff_{interval}m'] = pct_oi_change
            
    logging.debug("OI differences calculation complete.")
    return oi_differences_report

def _get_key_suffix(index_from_atm: int, total_options_one_side: int) -> str:
    """
    Helper function to determine the option key suffix (atm, itmX, otmX)
    based on the strike's index relative to the At-The-Money (ATM) strike.
    This mirrors the key generation logic in `get_relevant_option_details`.

    Args:
        index_from_atm: Integer representing the strike's position from ATM.
                        0 for ATM, negative for lower strikes, positive for higher strikes.
        total_options_one_side: Not directly used in current logic but kept for context.

    Returns:
        A string suffix like "atm", "itm1", "otm2".
    """
    if index_from_atm == 0:
        return "atm"
    elif index_from_atm < 0: # Strikes less than ATM
        return f"itm{-index_from_atm}" # e.g., index -1 is itm1
    else: # Strikes greater than ATM
        return f"otm{index_from_atm}"  # e.g., index 1 is otm1

def generate_options_tables(oi_report: dict, contract_details: dict, current_atm_strike: float, 
                            strike_step: int, num_strikes_each_side: int, 
                            change_intervals_list: tuple):
    """
    Generates two Rich Tables (one for Calls, one for Puts) displaying the OI analysis.
    If `current_atm_strike` is None, it returns an error Panel.

    Args:
        oi_report: Dictionary containing calculated OI data (from calculate_oi_differences).
        contract_details: Dictionary containing details of identified option contracts.
        current_atm_strike: The current At-The-Money strike. If None, an error panel is returned.
        strike_step: The difference between option strikes.
        num_strikes_each_side: Number of ITM/OTM strikes to display.
        change_intervals_list: Tuple of intervals (e.g., (10, 15, 30)) for OI change columns.

    Returns:
        A Rich Group object containing the Call and Put tables, or a Rich Panel with an error message.
    """
    if current_atm_strike is None:
        logging.error("Cannot generate tables: current_atm_strike is None.")
        return Panel("[bold red]ATM Strike could not be determined. Tables cannot be generated.[/bold red]", title="Error", border_style="red")

    time_now_str = datetime.now().strftime('%H:%M:%S') # Timestamp for table titles
    
    # Create Call options table
    call_table_title = f"CALL Options OI ({UNDERLYING_SYMBOL} - ATM: {int(current_atm_strike)}) @ {time_now_str}"
    call_table = Table(title=call_table_title, show_lines=True, expand=True)
    
    # Create Put options table
    put_table_title = f"PUT Options OI ({UNDERLYING_SYMBOL} - ATM: {int(current_atm_strike)}) @ {time_now_str}"
    put_table = Table(title=put_table_title, show_lines=True, expand=True)

    # Define common columns for both tables
    cols = ["Strike", "Symbol", "Latest OI", "OI Time"]
    for interval in change_intervals_list: # Dynamically add OI change columns
        cols.append(f"OI %Chg ({interval}m)") # Updated column header
    
    for col_name in cols:
        call_table.add_column(col_name, justify="right")
        put_table.add_column(col_name, justify="right")

    # Iterate through strike levels relative to ATM (-num to +num)
    total_call_threshold_breached = 0
    total_call_cells = 0

    total_put_threshold_breached = 0
    total_put_cells = 0
    for i in range(-num_strikes_each_side, num_strikes_each_side + 1):
        strike_val = current_atm_strike + (i * strike_step)
        key_suffix = _get_key_suffix(i, num_strikes_each_side) # Get "atm", "itmX", "otmX"

        # --- Populate Call Option Row ---
        option_key_ce = f"{key_suffix}_ce" # e.g., "atm_ce", "itm1_ce"
        ce_data = oi_report.get(option_key_ce, {}) # Get data for this call option
        ce_contract = contract_details.get(option_key_ce, {}) # Get contract details
        
        ce_strike_display = str(int(ce_contract.get('strike', strike_val))) # Use actual strike from contract if available
        
        # Style strike price: ATM (cyan), ITM for Calls (lower strikes - green), OTM for Calls (higher strikes - red)
        ce_strike_style = "cyan" if i == 0 else ("green" if i < 0 else "red")
        
        ce_latest_oi = ce_data.get('latest_oi')
        ce_latest_oi_time = ce_data.get('latest_oi_timestamp')

        # Prepare row data for call table
        ce_row_data = [
            Text(ce_strike_display, style=ce_strike_style),
            ce_contract.get('tradingsymbol', 'N/A'),
            f"{ce_latest_oi:,}" if ce_latest_oi is not None else "N/A", # Format OI with comma
            ce_latest_oi_time.strftime("%H:%M:%S %Z") if ce_latest_oi_time else "N/A" # Format time
        ]

        for interval in change_intervals_list: # Add OI change values
            total_call_cells = total_call_cells+1
            pct_oi_change = ce_data.get(f'pct_diff_{interval}m')
            formatted_pct_str = f"{pct_oi_change:+.2f}%" if pct_oi_change is not None else "N/A"
            
            cell_text = Text(formatted_pct_str)
            if pct_oi_change is not None and interval in PCT_CHANGE_THRESHOLDS:
                if abs(pct_oi_change) > PCT_CHANGE_THRESHOLDS[interval]: # Check absolute change against threshold
                    cell_text.stylize("bold red") # Apply style if threshold exceeded
                    total_call_threshold_breached = total_call_threshold_breached+1
            ce_row_data.append(cell_text)
        call_table.add_row(*ce_row_data)

        # --- Populate Put Option Row ---
        option_key_pe = f"{key_suffix}_pe" # e.g., "atm_pe", "itm1_pe"
        pe_data = oi_report.get(option_key_pe, {}) # Get data for this put option
        pe_contract = contract_details.get(option_key_pe, {}) # Get contract details

        pe_strike_display = str(int(pe_contract.get('strike', strike_val)))
        
        # Style strike price: ATM (cyan), ITM for Puts (higher strikes - green), OTM for Puts (lower strikes - red)
        pe_strike_style = "cyan" if i == 0 else ("green" if i > 0 else "red")

        pe_latest_oi = pe_data.get('latest_oi')
        pe_latest_oi_time = pe_data.get('latest_oi_timestamp')

        # Prepare row data for put table
        pe_row_data = [
            Text(pe_strike_display, style=pe_strike_style),
            pe_contract.get('tradingsymbol', 'N/A'),
            f"{pe_latest_oi:,}" if pe_latest_oi is not None else "N/A",
            pe_latest_oi_time.strftime("%H:%M:%S %Z") if pe_latest_oi_time else "N/A"
        ]
        for interval in change_intervals_list: # Add OI percentage change values
            total_put_cells = total_put_cells+1
            pct_oi_change = pe_data.get(f'pct_diff_{interval}m')
            formatted_pct_str = f"{pct_oi_change:+.2f}%" if pct_oi_change is not None else "N/A"

            cell_text = Text(formatted_pct_str)
            if pct_oi_change is not None and interval in PCT_CHANGE_THRESHOLDS:
                if abs(pct_oi_change) > PCT_CHANGE_THRESHOLDS[interval]: # Check absolute change against threshold
                    cell_text.stylize("bold red") # Apply style if threshold exceeded
                    total_put_threshold_breached = total_put_threshold_breached+1
            pe_row_data.append(cell_text)
        put_table.add_row(*pe_row_data)
    if (float(total_put_threshold_breached)/float(total_put_cells) > 0.5) or (float(total_call_threshold_breached)/float(total_call_cells) > 0.5):
        if SIREN_SOUND_PATH and os.path.exists(SIREN_SOUND_PATH):
            os.system(f"afplay {SIREN_SOUND_PATH}")


    return Group(call_table, put_table) # Group tables for simultaneous display in Live


def run_analysis_iteration(kite_conn, nfo_instr: list, nearest_exp_date: date, symbol_prefix: str, underlying_token: int):
    """
    Performs one complete iteration of fetching data, calculating differences, and generating tables.
    This function is called repeatedly by the live update loop.

    Args:
        kite_conn: Initialized KiteApp object.
        nfo_instr: List of all NFO instruments (fetched once at startup).
        nearest_exp_date: The nearest weekly expiry date (determined once at startup).
        underlying_token: Instrument token for the underlying symbol.

    Returns:
        A Rich Group object containing the Call and Put tables for display,
        or a Rich Panel with an error/warning message if issues occur.
    """
    global kws
    
    try:
        logging.debug("Starting new analysis iteration.")
        # 1. Get current ATM strike from WebSocket data
        current_atm_strike = get_atm_strike(underlying_token, STRIKE_DIFFERENCE)

        if not current_atm_strike: # Critical if ATM cannot be determined for this iteration
            logging.error("Could not determine ATM strike for this iteration.")
            return Panel("[bold red]Error: Could not determine ATM strike. Check logs. Waiting for next refresh.[/bold red]", title="Update Error", border_style="red")
        
        # This check should ideally be redundant if main() ensures nearest_exp_date is valid before starting loop
        if not nearest_exp_date:
             logging.error("Nearest expiry date is not available (should not happen if pre-checked).")
             return Panel("[bold red]Error: Nearest expiry date not available. Critical error.[/bold red]", title="Update Error", border_style="red")

        # 2. Identify relevant option contracts around the new ATM strike
        option_contract_details = get_relevant_option_details(
            nfo_instr, current_atm_strike, nearest_exp_date,
            STRIKE_DIFFERENCE, OPTIONS_COUNT, UNDERLYING_PREFIX, symbol_prefix
        )
        
        # If no contracts are found (e.g., due to market close or issues with instrument list for that ATM)
        if not option_contract_details:
            logging.warning(f"Could not retrieve relevant option contracts for ATM {int(current_atm_strike)}.")
            return Panel(f"[bold yellow]Warning: Could not retrieve relevant option contracts for ATM {int(current_atm_strike)}. Waiting for next refresh.[/bold yellow]", title="Update Warning", border_style="yellow")

        # 2.5 Subscribe to option contract tokens via WebSocket for real-time data
        option_tokens = [details['instrument_token'] for details in option_contract_details.values() if 'instrument_token' in details]
        if option_tokens and kws:
            try:
                kws.subscribe(option_tokens)
                kws.set_mode(kws.MODE_QUOTE, option_tokens)
                logging.debug(f"Subscribed to {len(option_tokens)} option tokens via WebSocket")
            except Exception as e:
                logging.warning(f"Failed to subscribe to option tokens: {e}")

        # 3. Fetch historical OI data for these contracts
        raw_historical_oi_data = fetch_historical_oi_data(kite_conn, option_contract_details)
        #print(raw_historical_oi_data)
        
        # 4. Calculate OI differences
        oi_change_data = calculate_oi_differences(raw_historical_oi_data, OI_CHANGE_INTERVALS_MIN)
        
        # 5. Generate Rich tables for display
        table_group = generate_options_tables(
            oi_change_data, option_contract_details, current_atm_strike, 
            STRIKE_DIFFERENCE, OPTIONS_COUNT, OI_CHANGE_INTERVALS_MIN
        )
        logging.debug("Analysis iteration completed successfully.")
        return table_group

    except Exception as e: # Catch any other unexpected errors during the iteration
        logging.error(f"Exception during analysis iteration: {e}", exc_info=True)
        return Panel(f"[bold red]An error occurred during data refresh: {e}. Check logs.[/bold red]", title="Update Error", border_style="red")


def main():
    """
    Main function to run the OI Tracker script.
    Handles initial setup (API connection, instrument fetching) and then enters the live update loop.
    """
    global kite, kws  # Need to modify global kite and kws objects
    
    console.print(f"[bold blue]Starting OI Tracker Script (Log file: {LOG_FILE_NAME})[/bold blue]")

    try:
        # --- Initial Setup ---
        # 1. Get TWOFA from user input
        console.print(f"[bold cyan]Login ID: {USER_ID}[/bold cyan]")
        twofa_input = console.input("[bold cyan]Enter your 2FA PIN or TOTP code: [/bold cyan]").strip()
        if not twofa_input:
            console.print("[bold red]2FA code is required. Exiting.[/bold red]")
            sys.exit(1)
        
        # 2. Get enctoken using kite_trade
        console.print("[bold cyan]Authenticating with Zerodha...[/bold cyan]")
        enctoken = get_enctoken(USER_ID, PASSWORD, twofa_input)
        if not enctoken:
            console.print("[bold red]Failed to get enctoken. Check your credentials and 2FA code.[/bold red]")
            sys.exit(1)
        
        console.print("[bold green]Successfully obtained enctoken![/bold green]")
        logging.info(f"Enctoken obtained: {enctoken[:20]}...")
        
        # 2. Initialize KiteApp with enctoken
        kite = KiteApp(enctoken=enctoken)
        console.print("[bold green]KiteApp initialized successfully![/bold green]")
        
        # 3. Verify connection by fetching profile
        profile = kite.profile()
        console.print(f"[green]Successfully connected for user: {profile.get('user_id')} ({profile.get('user_name')})[/green]")

        # 4. Fetch NFO instruments list (done once at startup)
        console.print(f"Fetching NFO instruments list for {EXCHANGE_NFO_OPTIONS} (once)...")
        nfo_instruments = kite.instruments(EXCHANGE_NFO_OPTIONS)
        if not nfo_instruments:
            console.print(f"[bold red]Failed to fetch NFO instruments from {EXCHANGE_NFO_OPTIONS}. Exiting.[/bold red]")
            logging.critical(f"Failed to fetch NFO instruments from {EXCHANGE_NFO_OPTIONS}.")
            sys.exit(1) # Critical error
        logging.info(f"Fetched {len(nfo_instruments)} NFO instruments.")
        
        # 5. Fetch NSE instruments to get underlying token
        console.print(f"Fetching NSE instruments to get {UNDERLYING_SYMBOL} token...")
        nse_instruments = kite.instruments(EXCHANGE_LTP)
        if not nse_instruments:
            console.print(f"[bold red]Failed to fetch NSE instruments. Exiting.[/bold red]")
            logging.critical(f"Failed to fetch NSE instruments.")
            sys.exit(1)
        
        # 6. Get underlying instrument token
        underlying_token = get_instrument_token_for_symbol(nse_instruments, UNDERLYING_SYMBOL, EXCHANGE_LTP)
        if not underlying_token:
            console.print(f"[bold red]Could not find instrument token for {UNDERLYING_SYMBOL}. Exiting.[/bold red]")
            logging.critical(f"Could not find instrument token for {UNDERLYING_SYMBOL}.")
            sys.exit(1)
        console.print(f"[green]Found {UNDERLYING_SYMBOL} token: {underlying_token}[/green]")
        
        # 7. Setup WebSocket connection
        console.print("[bold cyan]Setting up WebSocket connection...[/bold cyan]")
        user_id = profile.get('user_id')
        kws = KiteTicker(api_key=KITE_API_KEY, access_token=enctoken+"&user_id="+user_id)
        
        # Define WebSocket callbacks
        def on_ticks(ws, ticks):
            """Callback to receive ticks"""
            global latest_tick_data
            for tick in ticks:
                instrument_token = tick.get('instrument_token')
                if instrument_token:
                    latest_tick_data[instrument_token] = tick
                    logging.debug(f"Received tick for token {instrument_token}: LTP={tick.get('last_price')}")
        
        def on_connect(ws, response):
            """Callback on successful connect"""
            global ws_connected
            ws_connected = True
            console.print("[bold green]WebSocket connected successfully![/bold green]")
            logging.info("WebSocket connected")
            # Subscribe to underlying instrument token
            ws.subscribe([underlying_token])
            ws.set_mode(ws.MODE_QUOTE, [underlying_token])
            console.print(f"[green]Subscribed to {UNDERLYING_SYMBOL} (token: {underlying_token})[/green]")
        
        def on_close(ws, code, reason):
            """Callback when connection is closed"""
            global ws_connected
            ws_connected = False
            console.print(f"[bold yellow]WebSocket closed: {code} - {reason}[/bold yellow]")
            logging.warning(f"WebSocket closed: {code} - {reason}")
        
        def on_error(ws, code, reason):
            """Callback when error occurs"""
            console.print(f"[bold red]WebSocket error: {code} - {reason}[/bold red]")
            logging.error(f"WebSocket error: {code} - {reason}")
        
        # Assign callbacks
        kws.on_ticks = on_ticks
        kws.on_connect = on_connect
        kws.on_close = on_close
        kws.on_error = on_error
        
        # Connect WebSocket
        kws.connect(threaded=True)
        
        # Wait for WebSocket to connect
        console.print("[cyan]Waiting for WebSocket connection...[/cyan]")
        wait_count = 0
        while not kws.is_connected() and wait_count < 10:
            time.sleep(1)
            wait_count += 1
        
        if not kws.is_connected():
            console.print("[bold red]WebSocket failed to connect. Exiting.[/bold red]")
            logging.critical("WebSocket failed to connect")
            sys.exit(1)
        
        # Wait a bit for initial tick data
        console.print("[cyan]Waiting for initial tick data...[/cyan]")
        time.sleep(3)

        # 8. Determine nearest weekly expiry (done once at startup)
        # Note: If the script is run over multiple days, this expiry might become outdated.
        # For simplicity, it's fetched once. A more advanced version might re-check periodically.
        return_arr = get_nearest_weekly_expiry(nfo_instruments, UNDERLYING_PREFIX)
        nearest_expiry_date = return_arr['expiry']
        symbol_prefix = return_arr['symbol_prefix']

        if not nearest_expiry_date:
            console.print(f"[bold red]Could not determine nearest weekly expiry for {UNDERLYING_PREFIX}. Exiting.[/bold red]")
            logging.critical(f"Could not determine nearest weekly expiry for {UNDERLYING_PREFIX}.")
            sys.exit(1) # Critical error
        console.print(f"Tracking options for expiry: [bold magenta]{nearest_expiry_date.strftime('%d-%b-%Y')}[/bold magenta]")
        
        console.print(f"[bold green]Setup complete! Starting live OI tracking...[/bold green]")
        console.print(f"Refresh interval: {REFRESH_INTERVAL_SECONDS} seconds. Press Ctrl+C to exit.")
        console.print(f"Underlying: [bold cyan]{UNDERLYING_SYMBOL}[/bold cyan], Strike Difference: [bold cyan]{STRIKE_DIFFERENCE}[/bold cyan], Options Count per side: [bold cyan]{OPTIONS_COUNT}[/bold cyan]")

        # --- Live Update Loop ---
        # refresh_per_second for Live is for UI animation smoothness if any;
        # auto_refresh=False means we control update timing with time.sleep()
        with Live(console=console, refresh_per_second=10, auto_refresh=False) as live: 
            while True:
                logging.info("Starting new live update cycle.")
                # Perform one iteration of analysis
                display_content = run_analysis_iteration(kite, nfo_instruments, nearest_expiry_date, symbol_prefix, underlying_token)
                # Update the live display with the new tables or error panel
                live.update(display_content, refresh=True)
                logging.info(f"Live display updated. Waiting for {REFRESH_INTERVAL_SECONDS} seconds.")
                # Wait for the configured refresh interval
                time.sleep(REFRESH_INTERVAL_SECONDS)

    # Handle user interruption (Ctrl+C)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Script terminated by user (Ctrl+C).[/bold yellow]")
        logging.info("Script terminated by user.")
    # Catch any other unexpected critical errors during the main setup
    except Exception as e: 
        console.print(f"[bold red]An unexpected critical error occurred in the main setup: {e}[/bold red]")
        logging.critical(f"Unexpected critical error in main: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # This block executes whether an exception occurred or not (unless sys.exit was called)
        # Close WebSocket connection if it's open
        if kws:
            try:
                console.print("[cyan]Closing WebSocket connection...[/cyan]")
                kws.close()
                logging.info("WebSocket connection closed")
            except Exception as e:
                logging.warning(f"Error closing WebSocket: {e}")
        
        logging.info("oi_tracker.py script execution process ended.")
        console.print("[bold blue]OI Tracker script finished.[/bold blue]")

if __name__ == "__main__":
    # Entry point of the script
    main()
