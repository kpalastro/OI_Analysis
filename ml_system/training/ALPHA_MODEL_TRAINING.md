# Alpha Model Training Pipeline

This module implements rigorous walk-forward validation for training classification models on option chain features.

## Overview

The training pipeline:
- Uses **walk-forward validation** to prevent data leakage
- Trains **XGBoost** or **LightGBM** classifiers for 3-class prediction (Bullish/Neutral/Bearish)
- Generates comprehensive metrics and feature importance analysis
- Saves trained models with scalers and metadata for production use

## Quick Start

### Basic Training

Train an XGBoost model on NSE data with default settings:

```bash
python -m ml_system.training.train_alpha_model --exchange NSE
```

### Custom Configuration

```bash
# Train LightGBM model on BSE with custom validation windows
python -m ml_system.training.train_alpha_model \
    --exchange BSE \
    --model-type lightgbm \
    --initial-train-days 30 \
    --test-days 7 \
    --step-days 7 \
    --lookback-days 60
```

## Command-Line Options

| Option | Choices | Default | Description |
|--------|---------|---------|-------------|
| `--exchange` | NSE, BSE | NSE | Exchange to train on |
| `--model-type` | xgboost, lightgbm | xgboost | Model algorithm |
| `--lookback-days` | integer | 30 | Total historical data to use |
| `--initial-train-days` | integer | 30 | Days for first training fold |
| `--test-days` | integer | 7 | Days for each test fold |
| `--step-days` | integer | 7 | Days to step forward between folds |
| `--output-dir` | string | ml_system/models/alpha | Output directory |

## Walk-Forward Validation

Walk-forward validation ensures:
- **No data leakage**: Training data always precedes test data chronologically
- **Realistic performance**: Simulates how the model would perform in production
- **Multiple test periods**: Each fold tests on a distinct time period

### Example Validation Strategy

With `--initial-train-days 30 --test-days 7 --step-days 7`:

- **Fold 0**: Train on Days 1-30, Test on Days 31-37
- **Fold 1**: Train on Days 1-37, Test on Days 38-44
- **Fold 2**: Train on Days 1-44, Test on Days 45-51
- ... and so on

## Output Files

After training, the following files are saved to the output directory:

- `alpha_model_{model_type}.pkl` - Trained model
- `alpha_scaler_{model_type}.pkl` - Feature scaler
- `alpha_feature_names.pkl` - List of feature names (for validation)
- `alpha_model_metadata.pkl` - Training metadata (accuracy, best fold, etc.)
- `walk_forward_results.csv` - Per-fold metrics
- `feature_importance.csv` - Feature importance rankings

## Model Output

The model predicts one of three classes:
- **1**: Bullish (price increase > 0.2% in target window)
- **0**: Neutral (price change between -0.2% and +0.2%)
- **-1**: Bearish (price decrease < -0.2%)

The model also provides probability distributions for each class via `predict_proba()`.

## Metrics

The training pipeline reports:
- **Per-fold metrics**: Accuracy, Precision, Recall, F1 for each class
- **Overall metrics**: Aggregated across all folds
- **Confusion matrix**: Shows prediction vs actual distribution
- **Feature importance**: Top features driving predictions

## Programmatic Usage

```python
from ml_system.training.train_alpha_model import AlphaModelTrainer
from ml_system.data.feature_pipeline import FeaturePipeline

# Build features
pipeline = FeaturePipeline()
df = pipeline.build_feature_set('NSE')

# Train model
trainer = AlphaModelTrainer(model_type='xgboost')
results = trainer.train_with_walk_forward(
    df,
    initial_train_days=30,
    test_days=7,
    step_days=7
)

# Save best model
best_fold = results['best_fold']['fold']
best_model = trainer.models[f'fold_{best_fold}']
trainer.save_model(
    best_model,
    trainer.scaler,
    results['feature_names'],
    {'model_type': 'xgboost', 'exchange': 'NSE'}
)
```

## Requirements

- **XGBoost**: `pip install xgboost` (for XGBoost models)
- **LightGBM**: `pip install lightgbm` (for LightGBM models)
- **scikit-learn**: For preprocessing and metrics
- **pandas/numpy**: For data manipulation

## Notes

- The pipeline automatically handles feature scaling (StandardScaler)
- Missing values in features are automatically dropped
- The best model (highest accuracy fold) is saved for production
- Feature importance is computed from the best model
- All splits are validated for chronological correctness (no leakage)

## Next Steps

After training:
1. Review `feature_importance.csv` to understand which features matter most
2. Check `walk_forward_results.csv` for consistent performance across folds
3. Use the saved model in the prediction serving API (Phase 3)

