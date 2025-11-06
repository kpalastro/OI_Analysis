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

## Phase 3: Advanced Models ✅

### Components

1. **LSTM Model** (`models/lstm_model.py`)
   - Deep learning for time series prediction
   - Multi-layer LSTM with dropout and batch normalization
   - Sequence-based learning (captures temporal patterns)
   - Early stopping and learning rate scheduling

2. **Ensemble Model** (`models/ensemble_model.py`)
   - Weighted average ensemble
   - Stacking ensemble (with meta-learner)
   - Combines baseline and advanced models
   - Performance-based weight calculation

3. **Advanced Training** (`training/train_advanced.py`)
   - Trains LSTM models
   - Creates ensembles from multiple models
   - Integrates with baseline models

### Usage

```python
from ml_system.training.train_advanced import AdvancedTrainer
from ml_system.training.train_baseline import BaselineTrainer

# Train baseline models first
baseline_trainer = BaselineTrainer()
baseline_results = baseline_trainer.train_all_baselines(features_df, ...)

# Train advanced models
advanced_trainer = AdvancedTrainer()
advanced_results = advanced_trainer.train_all_advanced(
    X_train, y_train, X_test, y_test,
    baseline_results,
    train_lstm=True,
    create_ensemble=True
)
```

### Testing

```bash
python3 ml_system/test_phase3.py
```

### Requirements

- TensorFlow (for LSTM): `pip install tensorflow`
- All Phase 1 & 2 dependencies

## Phase 4: Backtesting & Real-time System ✅

### Components

1. **Backtesting Engine** (`backtesting/backtest_engine.py`)
   - Historical strategy simulation
   - Realistic trade execution (slippage, commissions)
   - Risk metrics (Sharpe ratio, Sortino ratio, max drawdown)
   - Position sizing with confidence-based adjustments
   - Stop-loss and take-profit management

2. **Real-time Predictor** (`predictions/realtime_predictor.py`)
   - Live prediction service
   - Model loading and inference
   - Ensemble prediction support
   - Confidence score calculation

3. **Signal Generator** (`predictions/signal_generator.py`)
   - Trading signal generation (BUY/SELL/HOLD)
   - Risk management (stop-loss, take-profit)
   - Position sizing recommendations
   - Signal filtering and quality control

4. **Performance Monitor** (`monitoring/performance_monitor.py`)
   - Prediction accuracy tracking
   - Model drift detection
   - Performance metrics calculation
   - Signal distribution analysis

### Usage

```python
from ml_system.backtesting.backtest_engine import BacktestEngine
from ml_system.predictions.realtime_predictor import RealTimePredictor
from ml_system.predictions.signal_generator import SignalGenerator

# Backtesting
backtester = BacktestEngine(initial_capital=100000.0)
result = backtester.backtest_strategy(predictions, actual_prices, timestamps)
backtester.print_results(result)

# Real-time prediction
predictor = RealTimePredictor()
predictor.load_models()
result = predictor.predict_and_signal(features_df)

# Signal generation
signal_gen = SignalGenerator()
signal = signal_gen.generate_signal(prediction, confidence, current_price)
```

### Testing

```bash
python3 ml_system/test_phase4.py
```


## Directory Structure

```
ml_system/
├── data/              # Data extraction
├── features/          # Feature engineering
├── models/            # Model definitions (LSTM, Ensemble)
├── training/          # Training scripts
├── backtesting/       # Backtesting engine
├── predictions/       # Real-time prediction & signals
├── monitoring/        # Performance monitoring
├── utils/             # Utility functions
└── config.py          # Configuration
```

## Current Status

✅ **Phase 1 Complete**: Data extraction and feature engineering modules are ready.
✅ **Phase 2 Complete**: Baseline models (Linear, Ridge, Random Forest, XGBoost) with evaluation tools.
✅ **Phase 3 Complete**: Advanced models (LSTM, Ensemble) for improved predictions.
✅ **Phase 4 Complete**: Backtesting engine and real-time prediction system.

## Next Steps

1. ✅ Test data extraction with your database
2. ✅ Validate feature engineering
3. ✅ Create baseline models
4. ✅ Evaluate initial performance
5. ✅ Implement advanced models (LSTM, Ensemble)
6. ✅ Build backtesting and real-time system
7. **Next**: Integration with trading platform or further optimization

