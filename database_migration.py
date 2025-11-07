"""
Database Migration Script
Adds new columns to option_chain_snapshots table as per FINAL_SCHEMA_SUMMARY.md
"""

import sqlite3
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_FILE = "oi_tracker.db"

def migrate_database():
    """Migrate database to add new columns."""
    if not os.path.exists(DB_FILE):
        logger.info("Database file does not exist. It will be created with new schema.")
        return
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(option_chain_snapshots)")
        columns = [row[1] for row in cursor.fetchall()]
        
        logger.info(f"Current columns: {columns}")
        
        # Add new columns if they don't exist
        new_columns = {
            'underlying_price': 'REAL',
            'moneyness': 'TEXT',
            'pct_change_5m': 'REAL',
            'pct_change_10m': 'REAL',
            'pct_change_15m': 'REAL',
            'pct_change_30m': 'REAL'
        }
        
        for column_name, column_type in new_columns.items():
            if column_name not in columns:
                logger.info(f"Adding column: {column_name} ({column_type})")
                try:
                    cursor.execute(f'''
                        ALTER TABLE option_chain_snapshots 
                        ADD COLUMN {column_name} {column_type}
                    ''')
                    logger.info(f"✓ Added column: {column_name}")
                except sqlite3.OperationalError as e:
                    logger.warning(f"Could not add column {column_name}: {e}")
            else:
                logger.info(f"Column {column_name} already exists")
        
        conn.commit()
        logger.info("✓ Database migration completed!")
        
        # Create indexes for new columns
        try:
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_snapshots_moneyness 
                ON option_chain_snapshots(exchange, moneyness, timestamp)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_snapshots_underlying_price 
                ON option_chain_snapshots(exchange, timestamp, underlying_price)
            ''')
            conn.commit()
            logger.info("✓ Indexes created for new columns")
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")
    
    except Exception as e:
        logger.error(f"Migration error: {e}", exc_info=True)
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database()

