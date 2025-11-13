#!/usr/bin/env python3
"""
Migration script from SQLite to PostgreSQL/TimescaleDB
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Import database module
try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    logging.error("psycopg package is required. Install with: pip install psycopg[binary]")
    sys.exit(1)

# Get database configuration
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_DATABASE = os.getenv("DB_DATABASE", "oi_tracker")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_FILE = os.getenv("SQLITE_DB_PATH", "oi_tracker.db")

USE_POSTGRES = all([DB_HOST, DB_DATABASE, DB_USER, DB_PASSWORD])

def get_db_connection():
    """Create PostgreSQL connection."""
    if not USE_POSTGRES:
        raise RuntimeError("PostgreSQL configuration not set")
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_DATABASE,
        user=DB_USER,
        password=DB_PASSWORD,
        autocommit=False,
        row_factory=dict_row,
    )

def check_postgres_config():
    """Verify PostgreSQL configuration is set."""
    required_vars = ['DB_HOST', 'DB_DATABASE', 'DB_USER', 'DB_PASSWORD']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logging.error(f"Missing required environment variables: {', '.join(missing)}")
        logging.error("Please set these in your .env file:")
        for var in missing:
            logging.error(f"  {var}=...")
        return False
    
    if not USE_POSTGRES:
        logging.error("PostgreSQL configuration not detected. Check your .env file.")
        return False
    
    logging.info("✓ PostgreSQL configuration found")
    return True

def initialize_schema():
    """Initialize PostgreSQL schema."""
    logging.info("Schema should already be initialized. Skipping...")
    return True

def verify_schema():
    """Verify tables were created."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check for tables
        cursor.execute("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY tablename
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['option_chain_snapshots', 'exchange_metadata', 'alpha_predictions']
        missing = [t for t in expected_tables if t not in tables]
        
        if missing:
            logging.warning(f"Missing tables: {', '.join(missing)}")
            conn.close()
            return False
        
        logging.info(f"✓ Tables verified: {', '.join(tables)}")
        
        # Check hypertable
        cursor.execute("""
            SELECT hypertable_name 
            FROM timescaledb_information.hypertables 
            WHERE hypertable_name = 'option_chain_snapshots'
        """)
        hypertable = cursor.fetchone()
        
        if hypertable:
            logging.info("✓ TimescaleDB hypertable created")
        else:
            logging.warning("⚠ TimescaleDB hypertable not found (will be created on first insert)")
        
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Failed to verify schema: {e}", exc_info=True)
        return False

def check_sqlite_data():
    """Check if SQLite database has data to migrate."""
    if not os.path.exists(DB_FILE):
        logging.info(f"No SQLite database found at {DB_FILE}")
        return 0
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check snapshot count
        cursor.execute("SELECT COUNT(*) FROM option_chain_snapshots")
        snapshot_count = cursor.fetchone()[0]
        
        # Check metadata count
        cursor.execute("SELECT COUNT(*) FROM exchange_metadata")
        metadata_count = cursor.fetchone()[0]
        
        # Check predictions count
        try:
            cursor.execute("SELECT COUNT(*) FROM alpha_predictions")
            predictions_count = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            predictions_count = 0
        
        conn.close()
        
        logging.info(f"SQLite data found:")
        logging.info(f"  - option_chain_snapshots: {snapshot_count:,} rows")
        logging.info(f"  - exchange_metadata: {metadata_count} rows")
        logging.info(f"  - alpha_predictions: {predictions_count} rows")
        
        return snapshot_count + metadata_count + predictions_count
    except Exception as e:
        logging.error(f"Failed to check SQLite data: {e}", exc_info=True)
        return 0

def migrate_data(batch_size=1000):
    """Migrate data from SQLite to PostgreSQL."""
    if not os.path.exists(DB_FILE):
        logging.info("No SQLite database to migrate")
        return True
    
    logging.info("Starting data migration...")
    
    sqlite_conn = sqlite3.connect(DB_FILE)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cursor = sqlite_conn.cursor()
    
    pg_conn = get_db_connection()
    pg_cursor = pg_conn.cursor()
    
    try:
        # Migrate option_chain_snapshots
        logging.info("Migrating option_chain_snapshots...")
        sqlite_cursor.execute("SELECT COUNT(*) FROM option_chain_snapshots")
        total_rows = sqlite_cursor.fetchone()[0]
        
        if total_rows > 0:
            sqlite_cursor.execute("""
                SELECT timestamp, exchange, strike, option_type, symbol, 
                       oi, ltp, token, underlying_price, moneyness,
                       pct_change_3m, pct_change_5m, pct_change_10m, 
                       pct_change_15m, pct_change_30m
                FROM option_chain_snapshots
                ORDER BY timestamp
            """)
            
            migrated = 0
            batch = []
            
            for row in sqlite_cursor:
                batch.append((
                    row['timestamp'],
                    row['exchange'],
                    row['strike'],
                    row['option_type'],
                    row['symbol'],
                    row['oi'],
                    row['ltp'],
                    row['token'],
                    row['underlying_price'],
                    row['moneyness'],
                    row['pct_change_3m'],
                    row['pct_change_5m'],
                    row['pct_change_10m'],
                    row['pct_change_15m'],
                    row['pct_change_30m'],
                ))
                
                if len(batch) >= batch_size:
                    pg_cursor.executemany("""
                        INSERT INTO option_chain_snapshots 
                        (timestamp, exchange, strike, option_type, symbol, oi, ltp, 
                         token, underlying_price, moneyness, pct_change_3m, 
                         pct_change_5m, pct_change_10m, pct_change_15m, pct_change_30m)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, batch)
                    pg_conn.commit()
                    migrated += len(batch)
                    logging.info(f"  Migrated {migrated:,} / {total_rows:,} rows ({migrated*100//total_rows}%)")
                    batch = []
            
            # Insert remaining batch
            if batch:
                pg_cursor.executemany("""
                    INSERT INTO option_chain_snapshots 
                    (timestamp, exchange, strike, option_type, symbol, oi, ltp, 
                     token, underlying_price, moneyness, pct_change_3m, 
                     pct_change_5m, pct_change_10m, pct_change_15m, pct_change_30m)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, batch)
                pg_conn.commit()
                migrated += len(batch)
            
            logging.info(f"✓ Migrated {migrated:,} option_chain_snapshots rows")
        
        # Migrate exchange_metadata
        logging.info("Migrating exchange_metadata...")
        sqlite_cursor.execute("SELECT * FROM exchange_metadata")
        metadata_rows = sqlite_cursor.fetchall()
        
        if metadata_rows:
            for row in metadata_rows:
                pg_cursor.execute("""
                    INSERT INTO exchange_metadata 
                    (exchange, last_update_time, last_atm_strike, last_underlying_price)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (exchange) DO UPDATE SET
                        last_update_time = EXCLUDED.last_update_time,
                        last_atm_strike = EXCLUDED.last_atm_strike,
                        last_underlying_price = EXCLUDED.last_underlying_price,
                        updated_at = CURRENT_TIMESTAMP
                """, (row['exchange'], row['last_update_time'], 
                      row['last_atm_strike'], row['last_underlying_price']))
            pg_conn.commit()
            logging.info(f"✓ Migrated {len(metadata_rows)} exchange_metadata rows")
        
        # Migrate alpha_predictions (if exists)
        try:
            logging.info("Migrating alpha_predictions...")
            sqlite_cursor.execute("SELECT COUNT(*) FROM alpha_predictions")
            predictions_count = sqlite_cursor.fetchone()[0]
            
            if predictions_count > 0:
                sqlite_cursor.execute("""
                    SELECT timestamp, exchange, signal, prediction_code, confidence,
                           prob_bullish, prob_neutral, prob_bearish, underlying_price,
                           atm_strike, selected_symbol, action, notes
                    FROM alpha_predictions
                """)
                
                batch = []
                for row in sqlite_cursor:
                    batch.append((
                        row['timestamp'], row['exchange'], row['signal'],
                        row['prediction_code'], row['confidence'],
                        row['prob_bullish'], row['prob_neutral'], row['prob_bearish'],
                        row['underlying_price'], row['atm_strike'],
                        row['selected_symbol'], row['action'], row['notes']
                    ))
                
                if batch:
                    pg_cursor.executemany("""
                        INSERT INTO alpha_predictions
                        (timestamp, exchange, signal, prediction_code, confidence,
                         prob_bullish, prob_neutral, prob_bearish, underlying_price,
                         atm_strike, selected_symbol, action, notes)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, batch)
                    pg_conn.commit()
                    logging.info(f"✓ Migrated {len(batch)} alpha_predictions rows")
        except sqlite3.OperationalError:
            logging.info("  No alpha_predictions table in SQLite (skipping)")
        
        logging.info("✓ Data migration completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Migration failed: {e}", exc_info=True)
        pg_conn.rollback()
        return False
    finally:
        sqlite_conn.close()
        pg_conn.close()

def verify_migration():
    """Verify migrated data."""
    logging.info("Verifying migrated data...")
    
    try:
        pg_conn = get_db_connection()
        pg_cursor = pg_conn.cursor()
        
        # Count rows
        pg_cursor.execute("SELECT COUNT(*) as count FROM option_chain_snapshots")
        pg_count = pg_cursor.fetchone()['count']
        
        pg_cursor.execute("SELECT COUNT(*) as count FROM exchange_metadata")
        pg_metadata = pg_cursor.fetchone()['count']
        
        pg_cursor.execute("SELECT COUNT(*) as count FROM alpha_predictions")
        pg_predictions = pg_cursor.fetchone()['count']
        
        logging.info(f"PostgreSQL data:")
        logging.info(f"  - option_chain_snapshots: {pg_count:,} rows")
        logging.info(f"  - exchange_metadata: {pg_metadata} rows")
        logging.info(f"  - alpha_predictions: {pg_predictions} rows")
        
        # Check hypertable
        pg_cursor.execute("""
            SELECT hypertable_name, num_chunks 
            FROM timescaledb_information.hypertables 
            WHERE hypertable_name = 'option_chain_snapshots'
        """)
        hypertable = pg_cursor.fetchone()
        
        if hypertable:
            logging.info(f"✓ Hypertable active with {hypertable['num_chunks']} chunks")
        
        pg_conn.close()
        return True
    except Exception as e:
        logging.error(f"Verification failed: {e}", exc_info=True)
        return False

def main():
    """Main migration function."""
    print("=" * 60)
    print("SQLite to PostgreSQL/TimescaleDB Migration")
    print("=" * 60)
    print()
    
    # Step 1: Check configuration
    if not check_postgres_config():
        sys.exit(1)
    
    # Step 2: Initialize schema
    if not initialize_schema():
        sys.exit(1)
    
    # Step 3: Verify schema
    if not verify_schema():
        logging.warning("Schema verification failed, but continuing...")
    
    # Step 4: Check SQLite data
    sqlite_row_count = check_sqlite_data()
    
    if sqlite_row_count == 0:
        logging.info("No data to migrate. Migration complete!")
        return
    
    # Step 5: Ask for confirmation
    print()
    response = input(f"Migrate {sqlite_row_count:,} rows from SQLite to PostgreSQL? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        logging.info("Migration cancelled by user")
        return
    
    # Step 6: Migrate data
    if not migrate_data():
        logging.error("Migration failed!")
        sys.exit(1)
    
    # Step 7: Verify migration
    verify_migration()
    
    print()
    print("=" * 60)
    print("✓ Migration completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Test the application with PostgreSQL backend")
    print("2. Verify data integrity")
    print("3. Once confirmed, you can backup/remove the SQLite database")
    print()

if __name__ == "__main__":
    main()

