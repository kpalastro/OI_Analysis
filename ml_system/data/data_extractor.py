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
        
        # Convert datetime to string for SQLite
        start_str = start_date.strftime('%Y-%m-%d %H:%M:%S') if isinstance(start_date, datetime) else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d %H:%M:%S') if isinstance(end_date, datetime) else str(end_date)
        
        query = """
            SELECT timestamp, exchange, strike, option_type, symbol, oi, ltp, token,
                   underlying_price, moneyness, pct_change_5m, pct_change_10m, 
                   pct_change_15m, pct_change_30m
            FROM option_chain_snapshots
            WHERE exchange = ?
            AND timestamp >= ?
            AND timestamp <= ?
            ORDER BY timestamp ASC, strike ASC, option_type ASC
        """
        
        df = pd.read_sql_query(
            query, 
            self.conn, 
            params=(exchange, start_str, end_str),
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
        Extract underlying price history from option chain snapshots.
        Uses the underlying_price column directly if available, otherwise infers from ATM strikes.
        
        Args:
            exchange: 'NSE' or 'BSE'
            start_date: Start datetime
            end_date: End datetime
        
        Returns:
            DataFrame with underlying price and ATM strike over time
        """
        if not self.conn:
            self.connect()
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        # Convert datetime to string for SQLite
        start_str = start_date.strftime('%Y-%m-%d %H:%M:%S') if isinstance(start_date, datetime) else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d %H:%M:%S') if isinstance(end_date, datetime) else str(end_date)
        
        # Try to get underlying_price directly from database (new schema)
        query = """
            SELECT DISTINCT
                timestamp,
                underlying_price,
                strike
            FROM option_chain_snapshots
            WHERE exchange = ?
            AND timestamp >= ?
            AND timestamp <= ?
            AND underlying_price IS NOT NULL
            AND moneyness = 'ATM'
            ORDER BY timestamp ASC
        """
        
        prices_df = pd.read_sql_query(
            query,
            self.conn,
            params=(exchange, start_str, end_str),
            parse_dates=['timestamp']
        )
        
        if not prices_df.empty:
            # Group by timestamp and take median underlying_price (in case of duplicates)
            prices_df = prices_df.groupby('timestamp').agg({
                'underlying_price': 'median',
                'strike': 'first'  # ATM strike
            }).reset_index()
            prices_df.columns = ['timestamp', 'underlying_price', 'atm_strike']
            logger.info(f"Extracted {len(prices_df)} underlying price records for {exchange}")
            return prices_df
        
        # Fallback: Infer from strike distribution (old method for backward compatibility)
        logger.warning(f"No underlying_price data found for {exchange}, inferring from strikes...")
        
        query = """
            SELECT 
                timestamp,
                strike,
                SUM(oi) as total_oi,
                COUNT(*) as option_count
            FROM option_chain_snapshots
            WHERE exchange = ?
            AND timestamp >= ?
            AND timestamp <= ?
            AND oi IS NOT NULL
            GROUP BY timestamp, strike
            ORDER BY timestamp ASC, total_oi DESC
        """
        
        strike_data = pd.read_sql_query(
            query,
            self.conn,
            params=(exchange, start_str, end_str),
            parse_dates=['timestamp']
        )
        
        if strike_data.empty:
            logger.warning(f"No strike data found for {exchange}")
            return pd.DataFrame(columns=['timestamp', 'underlying_price', 'atm_strike'])
        
        # For each timestamp, find the ATM strike and infer underlying price
        prices_list = []
        
        for timestamp in strike_data['timestamp'].unique():
            timestamp_data = strike_data[strike_data['timestamp'] == timestamp]
            
            # Get all strikes for this timestamp
            strikes = sorted(timestamp_data['strike'].unique())
            
            if len(strikes) == 0:
                continue
            
            # Method: Use the median strike as ATM (most stable)
            # The median strike in the option chain is typically close to ATM
            median_strike = timestamp_data['strike'].median()
            
            # Alternative: Use strike with highest OI (but this can be noisy)
            max_oi_row = timestamp_data.loc[timestamp_data['total_oi'].idxmax()]
            max_oi_strike = max_oi_row['strike']
            
            # Use median as primary (more stable), but if max OI is very close, use it
            if abs(max_oi_strike - median_strike) <= 25:  # Within 25 points
                atm_strike = max_oi_strike
            else:
                atm_strike = median_strike
            
            # Infer underlying price from ATM strike
            # ATM is calculated as: round(underlying_price / strike_diff) * strike_diff
            # So underlying_price ≈ ATM_strike (for NIFTY with 50 strike diff)
            # For more accuracy, we could reverse the calculation, but this approximation works
            underlying_price = atm_strike
            
            prices_list.append({
                'timestamp': timestamp,
                'underlying_price': underlying_price,
                'atm_strike': atm_strike
            })
        
        prices_df = pd.DataFrame(prices_list)
        
        if not prices_df.empty:
            # Sort by timestamp
            prices_df = prices_df.sort_values('timestamp').reset_index(drop=True)
            
            # Smooth the prices (moving average) to reduce noise
            window_size = min(5, len(prices_df) // 10 + 1)
            if window_size > 1:
                prices_df['underlying_price'] = prices_df['underlying_price'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
                prices_df['atm_strike'] = prices_df['atm_strike'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
        
        logger.info(f"Extracted {len(prices_df)} underlying price records for {exchange}")
        if not prices_df.empty:
            price_range = prices_df['underlying_price'].max() - prices_df['underlying_price'].min()
            logger.info(f"Price range: {price_range:.2f} points (min: {prices_df['underlying_price'].min():.2f}, max: {prices_df['underlying_price'].max():.2f})")
        
        return prices_df
    
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
            # Since data is sampled every ~30 seconds, we need to shift by number of rows
            # target_minutes = 15 means we want 15 minutes ahead
            # If refresh interval is 30 seconds, that's 30 rows ahead (15 * 60 / 30)
            # But we'll use a more flexible approach: shift by a reasonable number of rows
            # Estimate rows per minute (assuming ~2 rows per minute based on 30s refresh)
            rows_per_minute = 2  # Approximate
            shift_rows = max(1, int(target_minutes * rows_per_minute))
            
            df['future_price'] = df['underlying_price'].shift(-shift_rows)
            df['price_change'] = df['future_price'] - df['underlying_price']
            df['price_change_pct'] = (df['price_change'] / (df['underlying_price'] + 1e-10)) * 100
            
            # Direction target (1 for up, -1 for down, 0 for neutral)
            # Use a smaller threshold since we're using ATM strikes (less precise)
            threshold = 0.05  # 0.05% change threshold
            df['direction'] = 0
            df.loc[df['price_change_pct'] > threshold, 'direction'] = 1  # Up
            df.loc[df['price_change_pct'] < -threshold, 'direction'] = -1  # Down
        
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

