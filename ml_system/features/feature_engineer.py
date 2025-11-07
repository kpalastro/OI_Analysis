"""
Phase 1: Feature Engineering Module
Creates advanced features from raw OI and price data for ML model training.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineers features from raw OI and price data."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_names = []
    
    def create_oi_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create OI momentum and rate of change features.
        
        Args:
            df: DataFrame with OI columns
        
        Returns:
            DataFrame with added momentum features
        """
        if 'call_oi_total' in df.columns:
            # Call OI momentum
            df['call_oi_change'] = df['call_oi_total'].diff()
            df['call_oi_change_pct'] = df['call_oi_total'].pct_change() * 100
            df['call_oi_momentum_5'] = df['call_oi_total'].diff(5)
            df['call_oi_momentum_15'] = df['call_oi_total'].diff(15)
            df['call_oi_acceleration'] = df['call_oi_change'].diff()
        
        if 'put_oi_total' in df.columns:
            # Put OI momentum
            df['put_oi_change'] = df['put_oi_total'].diff()
            df['put_oi_change_pct'] = df['put_oi_total'].pct_change() * 100
            df['put_oi_momentum_5'] = df['put_oi_total'].diff(5)
            df['put_oi_momentum_15'] = df['put_oi_total'].diff(15)
            df['put_oi_acceleration'] = df['put_oi_change'].diff()
        
        # Net OI change
        if 'call_oi_total' in df.columns and 'put_oi_total' in df.columns:
            df['net_oi_change'] = df['put_oi_total'] - df['call_oi_total']
            df['net_oi_change_pct'] = df['net_oi_change'].pct_change() * 100
        
        logger.info("Created OI momentum features")
        return df
    
    def create_pcr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Put-Call Ratio (PCR) related features.
        
        Args:
            df: DataFrame with PCR column
        
        Returns:
            DataFrame with added PCR features
        """
        if 'pcr' in df.columns:
            # PCR momentum
            df['pcr_change'] = df['pcr'].diff()
            df['pcr_change_pct'] = df['pcr'].pct_change() * 100
            df['pcr_momentum_5'] = df['pcr'].diff(5)
            df['pcr_momentum_15'] = df['pcr'].diff(15)
            
            # PCR moving averages
            df['pcr_ma_5'] = df['pcr'].rolling(window=5).mean()
            df['pcr_ma_15'] = df['pcr'].rolling(window=15).mean()
            df['pcr_ma_30'] = df['pcr'].rolling(window=30).mean()
            
            # PCR deviation from mean
            df['pcr_deviation'] = df['pcr'] - df['pcr_ma_15']
            df['pcr_zscore'] = (df['pcr'] - df['pcr_ma_15']) / (df['pcr'].rolling(window=15).std() + 1e-10)
        
        logger.info("Created PCR features")
        return df
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features (momentum, volatility, etc.).
        
        Args:
            df: DataFrame with underlying_price column
        
        Returns:
            DataFrame with added price features
        """
        if 'underlying_price' in df.columns:
            # Price momentum
            df['price_change'] = df['underlying_price'].diff()
            df['price_change_pct'] = df['underlying_price'].pct_change() * 100
            df['price_momentum_5'] = df['underlying_price'].pct_change(5) * 100
            df['price_momentum_15'] = df['underlying_price'].pct_change(15) * 100
            df['price_acceleration'] = df['price_change_pct'].diff()
            
            # Moving averages
            df['price_ma_5'] = df['underlying_price'].rolling(window=5).mean()
            df['price_ma_15'] = df['underlying_price'].rolling(window=15).mean()
            df['price_ma_30'] = df['underlying_price'].rolling(window=30).mean()
            
            # Price deviation from MA
            df['price_deviation_ma5'] = (df['underlying_price'] - df['price_ma_5']) / df['price_ma_5'] * 100
            df['price_deviation_ma15'] = (df['underlying_price'] - df['price_ma_15']) / df['price_ma_15'] * 100
            
            # Volatility (rolling standard deviation)
            df['price_volatility_5'] = df['price_change_pct'].rolling(window=5).std()
            df['price_volatility_15'] = df['price_change_pct'].rolling(window=15).std()
            df['price_volatility_30'] = df['price_change_pct'].rolling(window=30).std()
            
            # RSI-like indicator (simplified)
            gains = df['price_change_pct'].where(df['price_change_pct'] > 0, 0)
            losses = -df['price_change_pct'].where(df['price_change_pct'] < 0, 0)
            avg_gain = gains.rolling(window=14).mean()
            avg_loss = losses.rolling(window=14).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
        
        logger.info("Created price features")
        return df
    
    def create_atm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ATM (At-The-Money) strike related features.
        
        Args:
            df: DataFrame with atm_strike and underlying_price columns
        
        Returns:
            DataFrame with added ATM features
        """
        if 'atm_strike' in df.columns and 'underlying_price' in df.columns:
            # Distance from ATM
            df['distance_from_atm'] = df['underlying_price'] - df['atm_strike']
            df['distance_from_atm_pct'] = (df['distance_from_atm'] / df['atm_strike']) * 100
            
            # ATM change
            df['atm_change'] = df['atm_strike'].diff()
            df['atm_change_pct'] = df['atm_strike'].pct_change() * 100
            
            # ATM momentum
            df['atm_momentum_5'] = df['atm_strike'].diff(5)
            df['atm_momentum_15'] = df['atm_strike'].diff(15)
        
        logger.info("Created ATM features")
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features (hour, day, etc.).
        
        Args:
            df: DataFrame with timestamp column
        
        Returns:
            DataFrame with added time features
        """
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Extract time components
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
            
            # Minutes since market open (assuming 9:15 AM)
            df['minutes_since_open'] = (df['hour'] - 9) * 60 + (df['minute'] - 15)
            df['minutes_since_open'] = df['minutes_since_open'].clip(lower=0)
            
            # Market session (morning, afternoon, closing)
            df['session'] = 0  # Default
            df.loc[(df['hour'] >= 9) & (df['hour'] < 12), 'session'] = 1  # Morning
            df.loc[(df['hour'] >= 12) & (df['hour'] < 15), 'session'] = 2  # Afternoon
            df.loc[(df['hour'] >= 15) & (df['hour'] < 16), 'session'] = 3  # Closing
        
        logger.info("Created time features")
        return df
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-related features from OI data.
        
        Args:
            df: DataFrame with OI columns
        
        Returns:
            DataFrame with added volume features
        """
        if 'oi_sum' in df.columns:
            # Total OI momentum
            df['total_oi_change'] = df['oi_sum'].diff()
            df['total_oi_change_pct'] = df['oi_sum'].pct_change() * 100
            
            # OI concentration (if we had strike-level data, would calculate here)
            # For now, use variance as proxy
            if 'oi_std' in df.columns:
                df['oi_concentration'] = df['oi_std'] / (df['oi_mean'] + 1e-10)
        
        logger.info("Created volume features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different variables.
        
        Args:
            df: DataFrame with various features
        
        Returns:
            DataFrame with added interaction features
        """
        # PCR * Price momentum
        if 'pcr' in df.columns and 'price_change_pct' in df.columns:
            df['pcr_price_interaction'] = df['pcr'] * df['price_change_pct']
        
        # OI change * Price change
        if 'call_oi_change_pct' in df.columns and 'price_change_pct' in df.columns:
            df['call_oi_price_interaction'] = df['call_oi_change_pct'] * df['price_change_pct']
        
        if 'put_oi_change_pct' in df.columns and 'price_change_pct' in df.columns:
            df['put_oi_price_interaction'] = df['put_oi_change_pct'] * df['price_change_pct']
        
        # Volatility * OI change
        if 'price_volatility_15' in df.columns and 'net_oi_change_pct' in df.columns:
            df['volatility_oi_interaction'] = df['price_volatility_15'] * df['net_oi_change_pct']
        
        logger.info("Created interaction features")
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: Raw DataFrame from data extractor
        
        Returns:
            DataFrame with all engineered features
        """
        logger.info(f"Starting feature engineering on {len(df)} rows")
        
        # Apply all feature engineering steps
        df = self.create_oi_momentum_features(df)
        df = self.create_pcr_features(df)
        df = self.create_price_features(df)
        df = self.create_atm_features(df)
        df = self.create_time_features(df)
        df = self.create_volume_features(df)
        df = self.create_interaction_features(df)
        
        # Remove rows with too many NaN values (from rolling windows)
        initial_rows = len(df)
        df = df.dropna()
        removed_rows = initial_rows - len(df)
        
        logger.info(f"Feature engineering complete. Removed {removed_rows} rows with NaN values.")
        logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
        
        # Store feature names (excluding target columns)
        target_cols = ['price_change', 'price_change_pct', 'direction', 'future_price']
        self.feature_names = [col for col in df.columns if col not in target_cols and col != 'timestamp']
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        return self.feature_names


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ml_system.data.data_extractor import DataExtractor
    
    extractor = DataExtractor()
    engineer = FeatureEngineer()
    
    try:
        # Get raw data
        raw_data = extractor.get_time_series_data('NSE', lookback_days=7)
        
        if not raw_data.empty:
            # Engineer features
            features_df = engineer.engineer_all_features(raw_data)
            
            print(f"\nEngineered Features Shape: {features_df.shape}")
            print(f"\nFeature Columns ({len(engineer.get_feature_names())}):")
            for i, feat in enumerate(engineer.get_feature_names(), 1):
                print(f"{i}. {feat}")
            
            print(f"\nSample Data:\n{features_df[['timestamp', 'underlying_price', 'pcr', 'price_change_pct'] + engineer.get_feature_names()[:5]].head()}")
    
    finally:
        extractor.close()

