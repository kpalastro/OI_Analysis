"""
Configuration file for ML Prediction System
"""

# Database configuration
DB_PATH = "oi_tracker.db"

# Data extraction configuration
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_TARGET_MINUTES = 15  # Predict 15 minutes ahead

# Feature engineering configuration
FEATURE_WINDOWS = {
    'short_term': 5,
    'medium_term': 15,
    'long_term': 30
}

# Model configuration
MODEL_CONFIG = {
    'train_test_split': 0.8,
    'validation_split': 0.1,
    'random_state': 42,
    'sequence_length': 60,  # Number of time steps for LSTM
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 10
}

# Target configuration
TARGET_CONFIG = {
    'direction_threshold': 0.1,  # 0.1% change to classify as up/down
    'prediction_horizons': [5, 15, 30, 60],  # Minutes ahead to predict
}

# Feature selection
FEATURES_TO_EXCLUDE = [
    'timestamp',
    'future_price',
    'price_change',
    'price_change_pct',
    'direction'
]

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FILE = "ml_system.log"

