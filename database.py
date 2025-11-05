# Database Module for OI Tracker
# Handles SQLite persistence for option chain data and restart recovery

import sqlite3
import logging
from datetime import datetime, timedelta, date
from threading import Lock
import os

# Database file path
DB_FILE = "oi_tracker.db"

# Thread-safe database lock
db_lock = Lock()

def get_db_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn

def initialize_database():
    """Initialize database schema if not exists."""
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Table for option chain snapshots
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS option_chain_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                exchange TEXT NOT NULL,
                strike REAL NOT NULL,
                option_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                oi INTEGER,
                ltp REAL,
                token INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_snapshots_exchange_timestamp 
            ON option_chain_snapshots(exchange, timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_snapshots_token 
            ON option_chain_snapshots(token)
        ''')
        
        # Table for exchange metadata (track last update times and underlying prices)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exchange_metadata (
                exchange TEXT PRIMARY KEY,
                last_update_time TIMESTAMP NOT NULL,
                last_atm_strike REAL,
                last_underlying_price REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logging.info(f"✓ Database initialized: {DB_FILE}")

def save_option_chain_snapshot(exchange, call_options, put_options, underlying_price=None, atm_strike=None, timestamp=None):
    """
    Save complete option chain snapshot to database including underlying price.
    
    Args:
        exchange: 'NSE' or 'BSE'
        call_options: List of call option dicts with strike, symbol, latest_oi, ltp, token
        put_options: List of put option dicts with strike, symbol, latest_oi, ltp, token
        underlying_price: Current price of underlying (NIFTY/SENSEX)
        atm_strike: Current ATM strike
        timestamp: datetime object (defaults to now)
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Prepare data for bulk insert
            records = []
            
            # Add call options
            for opt in call_options:
                if 'token' in opt and opt.get('symbol'):
                    records.append((
                        timestamp,
                        exchange,
                        opt.get('strike'),
                        'CE',
                        opt.get('symbol'),
                        opt.get('latest_oi'),
                        opt.get('ltp'),
                        opt.get('token')
                    ))
            
            # Add put options
            for opt in put_options:
                if 'token' in opt and opt.get('symbol'):
                    records.append((
                        timestamp,
                        exchange,
                        opt.get('strike'),
                        'PE',
                        opt.get('symbol'),
                        opt.get('latest_oi'),
                        opt.get('ltp'),
                        opt.get('token')
                    ))
            
            # Bulk insert
            if records:
                cursor.executemany('''
                    INSERT INTO option_chain_snapshots 
                    (timestamp, exchange, strike, option_type, symbol, oi, ltp, token)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', records)
                
                # Update exchange metadata with underlying price
                cursor.execute('''
                    INSERT OR REPLACE INTO exchange_metadata 
                    (exchange, last_update_time, last_underlying_price, last_atm_strike, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (exchange, timestamp, underlying_price, atm_strike))
                
                conn.commit()
                logging.debug(f"✓ Saved {len(records)} records to DB for {exchange} at {timestamp.strftime('%H:%M:%S')}")
                if underlying_price:
                    logging.debug(f"✓ Saved {exchange} underlying price: {underlying_price}, ATM: {atm_strike}")
            
            conn.close()
        except Exception as e:
            logging.error(f"Error saving snapshot for {exchange}: {e}", exc_info=True)

def get_last_snapshot_time(exchange):
    """
    Get the last snapshot timestamp for an exchange.
    
    Args:
        exchange: 'NSE' or 'BSE'
    
    Returns:
        datetime object or None if no data exists
    """
    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT last_update_time 
                FROM exchange_metadata 
                WHERE exchange = ?
            ''', (exchange,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                # Parse timestamp string to datetime (timezone-naive for consistency)
                timestamp = datetime.fromisoformat(row['last_update_time'])
                if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=None)
                return timestamp
            return None
        except Exception as e:
            logging.error(f"Error getting last snapshot time for {exchange}: {e}", exc_info=True)
            return None

def load_today_snapshots(exchange):
    """
    Load all snapshots for today for a given exchange.
    
    Args:
        exchange: 'NSE' or 'BSE'
    
    Returns:
        Dict with structure: {token: [(timestamp, oi), ...]}
    """
    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get today's date range
            today = date.today()
            start_of_day = datetime.combine(today, datetime.min.time())
            end_of_day = datetime.combine(today, datetime.max.time())
            
            cursor.execute('''
                SELECT timestamp, token, oi
                FROM option_chain_snapshots
                WHERE exchange = ?
                AND timestamp >= ?
                AND timestamp <= ?
                AND oi IS NOT NULL
                ORDER BY timestamp ASC
            ''', (exchange, start_of_day, end_of_day))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Organize by token
            oi_history = {}
            for row in rows:
                token = row['token']
                timestamp = datetime.fromisoformat(row['timestamp'])
                # Ensure timezone-naive for consistency
                if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=None)
                oi = row['oi']
                
                if token not in oi_history:
                    oi_history[token] = []
                
                oi_history[token].append({
                    'date': timestamp,
                    'oi': oi
                })
            
            logging.info(f"✓ Loaded {len(rows)} historical records for {exchange} from DB")
            return oi_history
            
        except Exception as e:
            logging.error(f"Error loading today's snapshots for {exchange}: {e}", exc_info=True)
            return {}

def should_load_from_db(exchange):
    """
    Determine if we should load from database based on 30-minute gap rule.
    
    Args:
        exchange: 'NSE' or 'BSE'
    
    Returns:
        True if gap < 30 minutes (load from DB), False otherwise (fresh start)
    """
    last_time = get_last_snapshot_time(exchange)
    
    if last_time is None:
        logging.info(f"{exchange}: No previous data found in DB - starting fresh")
        return False
    
    current_time = datetime.now()
    time_gap = current_time - last_time
    gap_minutes = time_gap.total_seconds() / 60
    
    # Check if it's the same day
    if last_time.date() != current_time.date():
        logging.info(f"{exchange}: Last data from different day - starting fresh")
        return False
    
    if gap_minutes < 30:
        logging.info(f"{exchange}: Gap is {gap_minutes:.1f} minutes (< 30) - loading from DB")
        return True
    else:
        logging.info(f"{exchange}: Gap is {gap_minutes:.1f} minutes (>= 30) - starting fresh")
        return False

def cleanup_old_data(days_to_keep=30):
    """
    Delete data older than specified days.
    
    Args:
        days_to_keep: Number of days to retain (default 30)
    """
    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            cursor.execute('''
                DELETE FROM option_chain_snapshots
                WHERE timestamp < ?
            ''', (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logging.info(f"✓ Cleaned up {deleted_count} old records (older than {days_to_keep} days)")
            
        except Exception as e:
            logging.error(f"Error cleaning up old data: {e}", exc_info=True)

def get_underlying_price_history(exchange, from_time=None, to_time=None):
    """
    Get underlying price history from exchange_metadata updates.
    
    Args:
        exchange: 'NSE' or 'BSE'
        from_time: Start datetime (optional)
        to_time: End datetime (optional)
    
    Returns:
        Dict with timestamp and price pairs
    """
    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get the latest underlying price for the exchange
            cursor.execute('''
                SELECT last_update_time, last_underlying_price, last_atm_strike
                FROM exchange_metadata
                WHERE exchange = ?
            ''', (exchange,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                # Parse as timezone-naive datetime for consistency
                timestamp = datetime.fromisoformat(row['last_update_time'])
                if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=None)
                
                return {
                    'timestamp': timestamp,
                    'underlying_price': row['last_underlying_price'],
                    'atm_strike': row['last_atm_strike']
                }
            return None
            
        except Exception as e:
            logging.error(f"Error getting underlying price history for {exchange}: {e}", exc_info=True)
            return None

def get_database_stats():
    """Get statistics about database size and records."""
    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Count records per exchange
            cursor.execute('''
                SELECT exchange, COUNT(*) as count
                FROM option_chain_snapshots
                GROUP BY exchange
            ''')
            
            stats = {}
            for row in cursor.fetchall():
                stats[row['exchange']] = row['count']
            
            # Get latest underlying prices
            cursor.execute('''
                SELECT exchange, last_underlying_price, last_atm_strike, last_update_time
                FROM exchange_metadata
            ''')
            
            for row in cursor.fetchall():
                exchange = row['exchange']
                stats[f'{exchange}_underlying_price'] = row['last_underlying_price']
                stats[f'{exchange}_atm_strike'] = row['last_atm_strike']
                stats[f'{exchange}_last_update'] = row['last_update_time']
            
            # Get database file size
            if os.path.exists(DB_FILE):
                file_size_mb = os.path.getsize(DB_FILE) / (1024 * 1024)
                stats['db_size_mb'] = round(file_size_mb, 2)
            
            conn.close()
            return stats
            
        except Exception as e:
            logging.error(f"Error getting database stats: {e}", exc_info=True)
            return {}

# Initialize database on module import
initialize_database()

