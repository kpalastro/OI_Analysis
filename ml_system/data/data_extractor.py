"""
Phase 1: Data Extraction Module
Extracts historical data from the database and prepares it for feature engineering.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExtractor:
    """Extracts and prepares data from the OI tracker database."""
    
    def __init__(self, db_path: str = "oi_tracker.db"):
        """
        Initialize the data extractor.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        logger.info(f"Connected to database: {self.db_path}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def get_exchange_data(
        self, 
        exchange: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Extract option chain snapshots for a specific exchange.
        
        Args:
            exchange: 'NSE' or 'BSE'
            start_date: Start datetime (default: 30 days ago)
            end_date: End datetime (default: now)
        
        Returns:
            DataFrame with columns: timestamp, exchange, strike, option_type, 
            symbol, oi, ltp, token
        """
        if not self.conn:
            self.connect()
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        query = """
            SELECT timestamp, exchange, strike, option_type, symbol, oi, ltp, token
            FROM option_chain_snapshots
            WHERE exchange = ?
            AND timestamp >= ?
            AND timestamp <= ?
            ORDER BY timestamp ASC, strike ASC, option_type ASC
        """
        
        df = pd.read_sql_query(
            query, 
            self.conn, 
            params=(exchange, start_date, end_date),
            parse_dates=['timestamp']
        )
        
        logger.info(f"Extracted {len(df)} records for {exchange} from {start_date} to {end_date}")
        return df
    
    def get_underlying_prices(
        self,
        exchange: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Extract underlying price history from exchange_metadata.
        
        Args:
            exchange: 'NSE' or 'BSE'
            start_date: Start datetime
            end_date: End datetime
        
        Returns:
            DataFrame with underlying price and ATM strike over time
        """
        if not self.conn:
            self.connect()
        
        # Get all snapshots to extract underlying prices from timestamps
        query = """
            SELECT DISTINCT timestamp
            FROM option_chain_snapshots
            WHERE exchange = ?
            AND timestamp >= ?
            AND timestamp <= ?
            ORDER BY timestamp ASC
        """
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        timestamps_df = pd.read_sql_query(
            query,
            self.conn,
            params=(exchange, start_date, end_date),
            parse_dates=['timestamp']
        )
        
        # Get metadata for each timestamp (approximate by getting closest)
        prices = []
        for _, row in timestamps_df.iterrows():
            ts = row['timestamp']
            meta_query = """
                SELECT last_update_time, last_underlying_price, last_atm_strike
                FROM exchange_metadata
                WHERE exchange = ?
                ORDER BY ABS(julianday(?) - julianday(last_update_time)) ASC
                LIMIT 1
            """
            meta = pd.read_sql_query(
                meta_query,
                self.conn,
                params=(exchange, ts)
            )
            if not meta.empty:
                prices.append({
                    'timestamp': ts,
                    'underlying_price': meta.iloc[0]['last_underlying_price'],
                    'atm_strike': meta.iloc[0]['last_atm_strike']
                })
        
        df = pd.DataFrame(prices)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Extracted {len(df)} underlying price records for {exchange}")
        return df
    
    def get_aggregated_option_data(
        self,
        exchange: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get aggregated option data grouped by timestamp.
        This creates one row per timestamp with aggregated metrics.
        
        Args:
            exchange: 'NSE' or 'BSE'
            start_date: Start datetime
            end_date: End datetime
        
        Returns:
            DataFrame with aggregated metrics per timestamp
        """
        df = self.get_exchange_data(exchange, start_date, end_date)
        
        if df.empty:
            return pd.DataFrame()
        
        # Group by timestamp and calculate aggregations
        grouped = df.groupby('timestamp').agg({
            'oi': ['sum', 'mean', 'std', 'min', 'max'],
            'ltp': ['mean', 'std', 'min', 'max'],
            'strike': ['min', 'max', 'mean'],
            'token': 'count'
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                         for col in grouped.columns.values]
        
        # Separate calls and puts
        calls_df = df[df['option_type'] == 'CE'].groupby('timestamp').agg({
            'oi': 'sum',
            'ltp': 'mean'
        }).reset_index()
        calls_df.columns = ['timestamp', 'call_oi_total', 'call_ltp_avg']
        
        puts_df = df[df['option_type'] == 'PE'].groupby('timestamp').agg({
            'oi': 'sum',
            'ltp': 'mean'
        }).reset_index()
        puts_df.columns = ['timestamp', 'put_oi_total', 'put_ltp_avg']
        
        # Merge
        result = grouped.merge(calls_df, on='timestamp', how='left')
        result = result.merge(puts_df, on='timestamp', how='left')
        
        # Calculate Put-Call Ratio
        result['pcr'] = result['put_oi_total'] / (result['call_oi_total'] + 1e-10)
        
        # Merge with underlying prices
        underlying_df = self.get_underlying_prices(exchange, start_date, end_date)
        if not underlying_df.empty:
            result = result.merge(underlying_df, on='timestamp', how='left')
        
        logger.info(f"Created aggregated dataset with {len(result)} timestamps for {exchange}")
        return result
    
    def get_time_series_data(
        self,
        exchange: str,
        lookback_days: int = 30,
        target_minutes: int = 15
    ) -> pd.DataFrame:
        """
        Get time series data ready for ML training.
        Creates sequences with features and targets.
        
        Args:
            exchange: 'NSE' or 'BSE'
            lookback_days: Number of days to look back
            target_minutes: Minutes ahead to predict
        
        Returns:
            DataFrame with features and target columns
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Get aggregated data
        df = self.get_aggregated_option_data(exchange, start_date, end_date)
        
        if df.empty:
            return pd.DataFrame()
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Create target: future underlying price change
        if 'underlying_price' in df.columns:
            df['future_price'] = df['underlying_price'].shift(-target_minutes)
            df['price_change'] = df['future_price'] - df['underlying_price']
            df['price_change_pct'] = (df['price_change'] / df['underlying_price']) * 100
            
            # Direction target (1 for up, 0 for down, -1 for neutral)
            df['direction'] = 0
            df.loc[df['price_change_pct'] > 0.1, 'direction'] = 1  # Up
            df.loc[df['price_change_pct'] < -0.1, 'direction'] = -1  # Down
        
        # Remove rows with NaN targets (last rows)
        df = df.dropna(subset=['price_change'])
        
        logger.info(f"Created time series dataset with {len(df)} samples for {exchange}")
        return df
    
    def export_to_csv(
        self,
        exchange: str,
        output_path: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ):
        """
        Export data to CSV for external analysis.
        
        Args:
            exchange: 'NSE' or 'BSE'
            output_path: Path to save CSV file
            start_date: Start datetime
            end_date: End datetime
        """
        df = self.get_time_series_data(exchange)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} records to {output_path}")


if __name__ == "__main__":
    # Example usage
    extractor = DataExtractor()
    
    try:
        # Extract NSE data
        nse_data = extractor.get_time_series_data('NSE', lookback_days=7)
        print(f"\nNSE Data Shape: {nse_data.shape}")
        print(f"NSE Columns: {list(nse_data.columns)}")
        if not nse_data.empty:
            print(f"\nNSE Data Sample:\n{nse_data.head()}")
        
        # Extract BSE data
        bse_data = extractor.get_time_series_data('BSE', lookback_days=7)
        print(f"\nBSE Data Shape: {bse_data.shape}")
        print(f"BSE Columns: {list(bse_data.columns)}")
        if not bse_data.empty:
            print(f"\nBSE Data Sample:\n{bse_data.head()}")
    
    finally:
        extractor.close()

