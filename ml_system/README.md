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

## Phase 2: Baseline Models ✅

### Components

1. **Baseline Training** (`training/train_baseline.py`)
   - Linear Regression
   - Ridge Regression
   - Random Forest (Regression & Classification)
   - XGBoost (if available)
   - Time-based train/test split
   - Model comparison and selection

2. **Model Evaluation** (`training/evaluate_models.py`)
   - Comprehensive metrics (R², MAE, RMSE, MAPE)
   - Direction accuracy for price predictions
   - Feature importance analysis
   - Visualization tools
   - Evaluation reports

### Usage

```python
from ml_system.training.train_baseline import BaselineTrainer
from ml_system.training.evaluate_models import ModelEvaluator

# Train all baseline models
trainer = BaselineTrainer()
results = trainer.train_all_baselines(features_df, target_col='price_change_pct', task='regression')

# Evaluate models
evaluator = ModelEvaluator()
evaluator.generate_report(results)

# Get best model
best_name, best_result = trainer.get_best_model('test_r2')
```

### Testing

```bash
python3 ml_system/test_phase2.py
```

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

✅ **Phase 1 Complete**: Data extraction and feature engineering modules are ready.
✅ **Phase 2 Complete**: Baseline models (Linear, Ridge, Random Forest, XGBoost) with evaluation tools.

## Next Steps

1. ✅ Test data extraction with your database
2. ✅ Validate feature engineering
3. ✅ Create baseline models
4. ✅ Evaluate initial performance
5. **Next**: Phase 3 - Advanced Models (LSTM/Transformer)

