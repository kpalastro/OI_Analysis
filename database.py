# Database Module for OI Tracker
# Supports SQLite (default) and PostgreSQL/TimescaleDB backends.

import logging
import os
import sqlite3
from datetime import date, datetime, timedelta
from threading import Lock

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None  # Optional dependency unless PostgreSQL backend is enabled

# -----------------------------------------------------------------------------
# Backend configuration
# -----------------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_DATABASE = os.getenv("DB_DATABASE", "oi_tracker")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

USE_POSTGRES = all([DB_HOST, DB_DATABASE, DB_USER, DB_PASSWORD])

if USE_POSTGRES and psycopg is None:
    raise RuntimeError(
        "psycopg package is required when using PostgreSQL backend. "
        "Install with `pip install psycopg[binary]`."
    )

# SQLite fallback (default)
DB_FILE = os.getenv("SQLITE_DB_PATH", "oi_tracker.db")
PARAM_PLACEHOLDER = "%s" if USE_POSTGRES else "?"

# Thread-safe lock for all DB operations
db_lock = Lock()


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _format_query(query: str) -> str:
    return query.replace("?", "%s") if USE_POSTGRES else query


def execute(cursor, query: str, params=None):
    if params is None:
        cursor.execute(query)
    else:
        cursor.execute(_format_query(query), params)


def executemany(cursor, query: str, params_list):
    cursor.executemany(_format_query(query), params_list)


def _ensure_naive_datetime(value):
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return value


# -----------------------------------------------------------------------------
# Connection management
# -----------------------------------------------------------------------------
def get_db_connection():
    """Create and return a database connection."""
    if USE_POSTGRES:
        conn = psycopg.connect(
            host=DB_HOST,
            port=int(DB_PORT),
            dbname=DB_DATABASE,
            user=DB_USER,
            password=DB_PASSWORD,
            autocommit=False,
            row_factory=dict_row,
        )
        return conn

    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn


# -----------------------------------------------------------------------------
# Schema management
# -----------------------------------------------------------------------------
def initialize_database():
    """Initialize database schema if not exists."""
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            if USE_POSTGRES:
                try:
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                except Exception as exc:
                    logging.warning(f"Could not enable timescaledb extension: {exc}")

                # Check if table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'option_chain_snapshots'
                    ) as exists
                """)
                result = cursor.fetchone()
                table_exists = result['exists'] if result else False
                
                if not table_exists:
                    cursor.execute(
                        """
                        CREATE TABLE option_chain_snapshots (
                            id BIGSERIAL,
                            timestamp TIMESTAMPTZ NOT NULL,
                            exchange TEXT NOT NULL,
                            strike DOUBLE PRECISION NOT NULL,
                            option_type TEXT NOT NULL,
                            symbol TEXT NOT NULL,
                            oi BIGINT,
                            ltp DOUBLE PRECISION,
                            token BIGINT NOT NULL,
                            underlying_price DOUBLE PRECISION,
                            moneyness TEXT,
                            pct_change_3m DOUBLE PRECISION,
                            pct_change_5m DOUBLE PRECISION,
                            pct_change_10m DOUBLE PRECISION,
                            pct_change_15m DOUBLE PRECISION,
                            pct_change_30m DOUBLE PRECISION,
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (id, timestamp)
                        )
                        """
                    )
                    try:
                        cursor.execute(
                            "SELECT create_hypertable('option_chain_snapshots','timestamp', if_not_exists => TRUE);"
                        )
                        logging.info("✓ Created TimescaleDB hypertable: option_chain_snapshots")
                    except Exception as exc:
                        logging.debug(f"Hypertable creation skipped: {exc}")
                else:
                    # Table exists, check if hypertable
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM timescaledb_information.hypertables 
                            WHERE hypertable_name = 'option_chain_snapshots'
                        ) as exists
                    """)
                    result = cursor.fetchone()
                    is_hypertable = result['exists'] if result else False
                    if not is_hypertable:
                        try:
                            cursor.execute(
                                "SELECT create_hypertable('option_chain_snapshots','timestamp', if_not_exists => TRUE);"
                            )
                            logging.info("✓ Converted to TimescaleDB hypertable: option_chain_snapshots")
                        except Exception as exc:
                            logging.debug(f"Hypertable conversion skipped: {exc}")
            else:
                cursor.execute(
                    """
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
                        underlying_price REAL,
                        moneyness TEXT,
                        pct_change_3m REAL,
                        pct_change_5m REAL,
                        pct_change_10m REAL,
                        pct_change_15m REAL,
                        pct_change_30m REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

                # Attempt to backfill missing columns for legacy SQLite databases
                try:
                    cursor.execute("PRAGMA table_info(option_chain_snapshots)")
                    columns = [row[1] for row in cursor.fetchall()]
                    new_columns = {
                        "underlying_price": "REAL",
                        "moneyness": "TEXT",
                        "pct_change_3m": "REAL",
                        "pct_change_5m": "REAL",
                        "pct_change_10m": "REAL",
                        "pct_change_15m": "REAL",
                        "pct_change_30m": "REAL",
                    }
                    for column_name, column_type in new_columns.items():
                        if column_name not in columns:
                            try:
                                cursor.execute(
                                    f"""
                                    ALTER TABLE option_chain_snapshots
                                    ADD COLUMN {column_name} {column_type}
                                    """
                                )
                                logging.info(f"✓ Added column: {column_name}")
                            except sqlite3.OperationalError:
                                pass
                except Exception as e:
                    logging.warning(f"Could not check/add columns: {e}")

            # Common indexes
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_snapshots_exchange_timestamp
                ON option_chain_snapshots(exchange, timestamp)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_snapshots_token
                ON option_chain_snapshots(token)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_snapshots_moneyness
                ON option_chain_snapshots(exchange, moneyness, timestamp)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_snapshots_underlying_price
                ON option_chain_snapshots(exchange, timestamp, underlying_price)
                """
            )

            # Exchange metadata
            if USE_POSTGRES:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS exchange_metadata (
                        exchange TEXT PRIMARY KEY,
                        last_update_time TIMESTAMPTZ NOT NULL,
                        last_atm_strike DOUBLE PRECISION,
                        last_underlying_price DOUBLE PRECISION,
                        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
            else:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS exchange_metadata (
                        exchange TEXT PRIMARY KEY,
                        last_update_time TIMESTAMP NOT NULL,
                        last_atm_strike REAL,
                        last_underlying_price REAL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

            # Alpha prediction history
            if USE_POSTGRES:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS alpha_predictions (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        exchange TEXT NOT NULL,
                        signal TEXT,
                        prediction_code INTEGER,
                        confidence DOUBLE PRECISION,
                        prob_bullish DOUBLE PRECISION,
                        prob_neutral DOUBLE PRECISION,
                        prob_bearish DOUBLE PRECISION,
                        underlying_price DOUBLE PRECISION,
                        atm_strike DOUBLE PRECISION,
                        selected_symbol TEXT,
                        action TEXT,
                        notes TEXT,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
            else:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS alpha_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP NOT NULL,
                        exchange TEXT NOT NULL,
                        signal TEXT,
                        prediction_code INTEGER,
                        confidence REAL,
                        prob_bullish REAL,
                        prob_neutral REAL,
                        prob_bearish REAL,
                        underlying_price REAL,
                        atm_strike REAL,
                        selected_symbol TEXT,
                        action TEXT,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_alpha_predictions_exchange_timestamp
                ON alpha_predictions(exchange, timestamp)
                """
            )

            conn.commit()
            logging.info(f"✓ Database initialized ({'PostgreSQL' if USE_POSTGRES else 'SQLite'})")
        finally:
            conn.close()


# -----------------------------------------------------------------------------
# Domain helpers
# -----------------------------------------------------------------------------
def calculate_moneyness(strike, option_type, underlying_price, atm_strike):
    """
    Calculate moneyness (ITM/ATM/OTM) for an option.
    """
    if atm_strike is not None and strike == atm_strike:
        return "ATM"

    if option_type == "CE":
        if underlying_price is not None:
            if strike < underlying_price:
                return "ITM"
            if strike > underlying_price:
                return "OTM"
        if atm_strike is not None:
            if strike < atm_strike:
                return "ITM"
            if strike > atm_strike:
                return "OTM"
    else:  # PE
        if underlying_price is not None:
            if strike > underlying_price:
                return "ITM"
            if strike < underlying_price:
                return "OTM"
        if atm_strike is not None:
            if strike > atm_strike:
                return "ITM"
            if strike < atm_strike:
                return "OTM"

    return "ATM"


def save_option_chain_snapshot(
    exchange,
    call_options,
    put_options,
    underlying_price=None,
    atm_strike=None,
    timestamp=None,
):
    """
    Save complete option chain snapshot to database.
    """
    if timestamp is None:
        timestamp = datetime.now()

    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            records = []

            for opt in call_options:
                if "token" in opt and opt.get("symbol"):
                    strike = opt.get("strike")
                    pct_changes = opt.get("pct_changes", {})
                    moneyness = calculate_moneyness(strike, "CE", underlying_price, atm_strike)
                    records.append(
                        (
                            timestamp,
                            exchange,
                            strike,
                            "CE",
                            opt.get("symbol"),
                            opt.get("latest_oi"),
                            opt.get("ltp"),
                            opt.get("token"),
                            underlying_price,
                            moneyness,
                            pct_changes.get("3m"),
                            pct_changes.get("5m"),
                            pct_changes.get("10m"),
                            pct_changes.get("15m"),
                            pct_changes.get("30m"),
                        )
                    )

            for opt in put_options:
                if "token" in opt and opt.get("symbol"):
                    strike = opt.get("strike")
                    pct_changes = opt.get("pct_changes", {})
                    moneyness = calculate_moneyness(strike, "PE", underlying_price, atm_strike)
                    records.append(
                        (
                            timestamp,
                            exchange,
                            strike,
                            "PE",
                            opt.get("symbol"),
                            opt.get("latest_oi"),
                            opt.get("ltp"),
                            opt.get("token"),
                            underlying_price,
                            moneyness,
                            pct_changes.get("3m"),
                            pct_changes.get("5m"),
                            pct_changes.get("10m"),
                            pct_changes.get("15m"),
                            pct_changes.get("30m"),
                        )
                    )

            if records:
                executemany(
                    cursor,
                    """
                    INSERT INTO option_chain_snapshots (
                        timestamp,
                        exchange,
                        strike,
                        option_type,
                        symbol,
                        oi,
                        ltp,
                        token,
                        underlying_price,
                        moneyness,
                        pct_change_3m,
                        pct_change_5m,
                        pct_change_10m,
                        pct_change_15m,
                        pct_change_30m
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    records,
                )

                if USE_POSTGRES:
                    execute(
                        cursor,
                        """
                        INSERT INTO exchange_metadata (
                            exchange,
                            last_update_time,
                            last_underlying_price,
                            last_atm_strike,
                            updated_at
                        )
                        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (exchange) DO UPDATE
                        SET last_update_time = EXCLUDED.last_update_time,
                            last_underlying_price = EXCLUDED.last_underlying_price,
                            last_atm_strike = EXCLUDED.last_atm_strike,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (exchange, timestamp, underlying_price, atm_strike),
                    )
                else:
                    execute(
                        cursor,
                        """
                        INSERT OR REPLACE INTO exchange_metadata (
                            exchange,
                            last_update_time,
                            last_underlying_price,
                            last_atm_strike,
                            updated_at
                        )
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        (exchange, timestamp, underlying_price, atm_strike),
                    )

                conn.commit()
                logging.debug(
                    f"✓ Saved {len(records)} records to DB for {exchange} at {timestamp.strftime('%H:%M:%S')}"
                )
        except Exception as e:
            logging.error(f"Error saving snapshot for {exchange}: {e}", exc_info=True)
        finally:
            conn.close()


def save_alpha_prediction(
    exchange,
    signal,
    prediction_code,
    confidence,
    prob_bullish=None,
    prob_neutral=None,
    prob_bearish=None,
    underlying_price=None,
    atm_strike=None,
    selected_symbol=None,
    action=None,
    notes=None,
    timestamp=None,
):
    """Persist an alpha-model prediction for future validation."""
    if timestamp is None:
        timestamp = datetime.now()

    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            execute(
                cursor,
                """
                INSERT INTO alpha_predictions (
                    timestamp,
                    exchange,
                    signal,
                    prediction_code,
                    confidence,
                    prob_bullish,
                    prob_neutral,
                    prob_bearish,
                    underlying_price,
                    atm_strike,
                    selected_symbol,
                    action,
                    notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    exchange,
                    signal,
                    prediction_code,
                    confidence,
                    prob_bullish,
                    prob_neutral,
                    prob_bearish,
                    underlying_price,
                    atm_strike,
                    selected_symbol,
                    action,
                    notes,
                ),
            )
            conn.commit()
        except Exception as exc:
            logging.error(f"Error saving alpha prediction for {exchange}: {exc}", exc_info=True)
        finally:
            conn.close()


# -----------------------------------------------------------------------------
# Retrieval helpers
# -----------------------------------------------------------------------------
def get_last_snapshot_time(exchange):
    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            execute(
                cursor,
                """
                SELECT last_update_time
                FROM exchange_metadata
                WHERE exchange = ?
                """,
                (exchange,),
            )
            row = cursor.fetchone()
            conn.close()

            if row and row.get("last_update_time"):
                return _ensure_naive_datetime(row["last_update_time"])
            return None
        except Exception as e:
            logging.error(f"Error getting last snapshot time for {exchange}: {e}", exc_info=True)
            return None


def load_today_snapshots(exchange):
    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            today = date.today()
            start_of_day = datetime.combine(today, datetime.min.time())
            end_of_day = datetime.combine(today, datetime.max.time())

            execute(
                cursor,
                """
                SELECT timestamp, token, oi
                FROM option_chain_snapshots
                WHERE exchange = ?
                  AND timestamp >= ?
                  AND timestamp <= ?
                  AND oi IS NOT NULL
                ORDER BY timestamp ASC
                """,
                (exchange, start_of_day, end_of_day),
            )

            rows = cursor.fetchall()
            conn.close()

            oi_history = {}
            for row in rows:
                token = row["token"]
                ts = _ensure_naive_datetime(row["timestamp"])
                if token not in oi_history:
                    oi_history[token] = []
                oi_history[token].append({"date": ts, "oi": row["oi"]})

            logging.info(f"✓ Loaded {len(rows)} historical records for {exchange} from DB")
            return oi_history
        except Exception as e:
            logging.error(f"Error loading today's snapshots for {exchange}: {e}", exc_info=True)
            return {}


def get_previous_close_price(exchange):
    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            if USE_POSTGRES:
                execute(
                    cursor,
                    """
                    SELECT underlying_price
                    FROM option_chain_snapshots
                    WHERE exchange = %s
                      AND underlying_price IS NOT NULL
                      AND timestamp::date < CURRENT_DATE
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (exchange,),
                )
            else:
                execute(
                    cursor,
                    """
                    SELECT underlying_price
                    FROM option_chain_snapshots
                    WHERE exchange = ?
                      AND underlying_price IS NOT NULL
                      AND DATE(timestamp) < DATE('now')
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (exchange,),
                )

            row = cursor.fetchone()
            conn.close()
            if row and row.get("underlying_price") is not None:
                return float(row["underlying_price"])
            return None
        except Exception as e:
            logging.warning(f"Could not fetch previous close for {exchange}: {e}")
            return None


def should_load_from_db(exchange):
    last_time = get_last_snapshot_time(exchange)
    if last_time is None:
        logging.info(f"{exchange}: No previous data found in DB - starting fresh")
        return False

    current_time = datetime.now()
    if last_time.date() != current_time.date():
        logging.info(f"{exchange}: Last data from different day - starting fresh")
        return False

    gap_minutes = (current_time - last_time).total_seconds() / 60
    if gap_minutes < 30:
        logging.info(f"{exchange}: Gap is {gap_minutes:.1f} minutes (< 30) - loading from DB")
        return True

    logging.info(f"{exchange}: Gap is {gap_minutes:.1f} minutes (>= 30) - starting fresh")
    return False


def cleanup_old_data(days_to_keep=30):
    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            execute(
                cursor,
                """
                DELETE FROM option_chain_snapshots
                WHERE timestamp < ?
                """,
                (cutoff_date,),
            )
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            if deleted_count > 0:
                logging.info(f"✓ Cleaned up {deleted_count} old records (older than {days_to_keep} days)")
        except Exception as e:
            logging.error(f"Error cleaning up old data: {e}", exc_info=True)


def get_underlying_price_history(exchange, from_time=None, to_time=None):
    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            execute(
                cursor,
                """
                SELECT last_update_time, last_underlying_price, last_atm_strike
                FROM exchange_metadata
                WHERE exchange = ?
                """,
                (exchange,),
            )
            row = cursor.fetchone()
            conn.close()
            if row:
                return {
                    "timestamp": _ensure_naive_datetime(row["last_update_time"]),
                    "underlying_price": row["last_underlying_price"],
                    "atm_strike": row["last_atm_strike"],
                }
            return None
        except Exception as e:
            logging.error(f"Error getting underlying price history for {exchange}: {e}", exc_info=True)
            return None


def get_database_stats():
    with db_lock:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            execute(
                cursor,
                """
                SELECT exchange, COUNT(*) AS count
                FROM option_chain_snapshots
                GROUP BY exchange
                """,
            )
            stats_rows = cursor.fetchall()
            stats = {row["exchange"]: row["count"] for row in stats_rows}

            execute(
                cursor,
                """
                SELECT exchange, last_underlying_price, last_atm_strike, last_update_time
                FROM exchange_metadata
                """,
            )
            for row in cursor.fetchall():
                exchange = row["exchange"]
                stats[f"{exchange}_underlying_price"] = row["last_underlying_price"]
                stats[f"{exchange}_atm_strike"] = row["last_atm_strike"]
                stats[f"{exchange}_last_update"] = row["last_update_time"]

            if not USE_POSTGRES and os.path.exists(DB_FILE):
                file_size_mb = os.path.getsize(DB_FILE) / (1024 * 1024)
                stats["db_size_mb"] = round(file_size_mb, 2)

            conn.close()
            return stats
        except Exception as e:
            logging.error(f"Error getting database stats: {e}", exc_info=True)
            return {}


# Initialize database on module import
initialize_database()

