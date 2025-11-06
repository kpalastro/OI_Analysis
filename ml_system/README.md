# ML Prediction System for OI Tracker

## Overview

This ML system uses historical Open Interest (OI) data to predict index movements and generate trading signals for options trading.

## Phase 1: Data Pipeline & Feature Engineering ✅

### Components

1. **Data Extraction** (`data/data_extractor.py`)
   - Extracts historical data from SQLite database
   - Creates time series datasets
   - Handles multiple exchanges (NSE/BSE)

2. **Feature Engineering** (`features/feature_engineer.py`)
   - OI momentum features
   - PCR (Put-Call Ratio) features
   - Price momentum and volatility
   - ATM strike features
   - Time-based features
   - Interaction features

### Usage

```python
from ml_system.data.data_extractor import DataExtractor
from ml_system.features.feature_engineer import FeatureEngineer

# Extract data
extractor = DataExtractor()
raw_data = extractor.get_time_series_data('NSE', lookback_days=30)

# Engineer features
engineer = FeatureEngineer()
features_df = engineer.engineer_all_features(raw_data)

# Get feature names
feature_names = engineer.get_feature_names()
```

### Installation

```bash
# Install ML dependencies
pip install -r ml_system/requirements_ml.txt
```

## Phase 2: Baseline Models (Next)

- Linear regression baseline
- Random Forest
- XGBoost
- Performance evaluation

## Phase 3: Advanced Models (Planned)

- LSTM/Transformer for time series
- Ensemble methods
- Multi-task learning

## Phase 4: Backtesting (Planned)

- Historical performance testing
- Risk metrics calculation
- Trade simulation

## Phase 5: Real-time System (Planned)

- Live prediction pipeline
- Signal generation
- Integration with trading platform

## Directory Structure

```
ml_system/
├── data/              # Data extraction
├── features/          # Feature engineering
├── models/            # Model definitions
├── training/          # Training scripts
├── predictions/       # Prediction pipeline
├── utils/             # Utility functions
└── config.py          # Configuration
```

## Current Status

✅ Phase 1 Complete: Data extraction and feature engineering modules are ready.

## Next Steps

1. Test data extraction with your database
2. Validate feature engineering
3. Create baseline models
4. Evaluate initial performance

