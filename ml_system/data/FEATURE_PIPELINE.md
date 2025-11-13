# Feature Engineering Pipeline

This module generates engineered features from raw option chain snapshots stored in the SQLite database.

## Overview

The `FeaturePipeline` class transforms raw option chain data into a structured feature set suitable for machine learning models. It:

- Aggregates OI metrics (call/put totals, ITM ratios, PCR)
- Computes lagged deltas at multiple time horizons (1m, 3m, 5m, 10m, 15m)
- Generates forward-looking target labels for classification
- Saves outputs to CSV and/or SQLite for downstream training

## Quick Start

### Basic Usage

Generate features for NSE with default settings (30 days lookback, 15-minute target):

```bash
python -m ml_system.data.feature_pipeline --exchange NSE
```

### Custom Configuration

```bash
# Generate features for BSE with 60 days of history
python -m ml_system.data.feature_pipeline \
    --exchange BSE \
    --lookback-days 60 \
    --target-minutes 30

# Output only CSV format
python -m ml_system.data.feature_pipeline \
    --exchange NSE \
    --output-format csv

# Use a custom database path
python -m ml_system.data.feature_pipeline \
    --exchange NSE \
    --db-path /path/to/custom.db
```

## Command-Line Options

| Option | Choices | Default | Description |
|--------|---------|---------|-------------|
| `--exchange` | NSE, BSE | NSE | Exchange to process |
| `--lookback-days` | integer | 30 | Number of days of historical data to use |
| `--target-minutes` | integer | 15 | Forward-looking prediction window |
| `--output-format` | csv, sqlite, both | both | Output format |
| `--db-path` | string | oi_tracker.db | Path to source database |

## Generated Features

### Aggregated OI Metrics
- `total_oi`: Total open interest across all strikes
- `call_oi_total` / `put_oi_total`: Total call/put OI
- `call_itm_oi` / `put_itm_oi`: ITM call/put OI
- `call_otm_oi` / `put_otm_oi`: OTM call/put OI
- `oi_concentration`: Standard deviation of OI across strikes

### Ratio Features
- `pcr`: Put-Call Ratio (put OI / call OI)
- `itm_ratio`: ITM Put OI / ITM Call OI
- `net_itm_pressure`: Net directional pressure from ITM options

### Lagged Delta Features
For each metric (call_oi_total, put_oi_total, call_itm_oi, put_itm_oi, pcr, itm_ratio, underlying_price), the pipeline computes change over multiple time horizons:
- `{metric}_chg_1m`: 1-minute change
- `{metric}_chg_3m`: 3-minute change
- `{metric}_chg_5m`: 5-minute change
- `{metric}_chg_10m`: 10-minute change
- `{metric}_chg_15m`: 15-minute change

### Target Variables
- `future_return_pct`: Percentage change in underlying price over target window
- `future_direction_label`: Classification label
  - `1`: Bullish (>0.2% increase)
  - `0`: Neutral (-0.2% to +0.2%)
  - `-1`: Bearish (<-0.2% decrease)

## Output Files

### CSV Format
Saved to: `ml_system/data/feature_sets/{exchange}_features_{target_minutes}m.csv`

### SQLite Format
Saved to: `ml_system/data/feature_store.db` in table `feature_sets`

The SQLite table includes an index on `(exchange, timestamp)` for efficient querying.

## Programmatic Usage

```python
from ml_system.data.feature_pipeline import FeaturePipeline, FeaturePipelineConfig

# Create custom configuration
config = FeaturePipelineConfig(
    lookback_days=60,
    target_minutes=30,
    lag_minutes=(1, 3, 5, 10, 15, 30)
)

# Build feature set
pipeline = FeaturePipeline(config)
try:
    feature_df = pipeline.build_feature_set('NSE')
    
    # Save to CSV
    csv_path = pipeline.save_feature_set(feature_df, 'NSE')
    
    # Save to SQLite
    db_path = pipeline.save_feature_set_sqlite(feature_df, 'NSE')
finally:
    pipeline.extractor.close()
```

## Integration with Training Pipeline

The generated feature sets can be used directly with the existing training pipeline:

```python
from ml_system.data.feature_pipeline import FeaturePipeline
from ml_system.training.train_baseline import BaselineTrainer

# Generate features
pipeline = FeaturePipeline()
feature_df = pipeline.build_feature_set('NSE')

# Train model
trainer = BaselineTrainer()
results = trainer.train_all_baselines(
    feature_df,
    target_col='future_direction_label',
    task='classification'
)
```

## Notes

- The pipeline assumes approximately 2 rows per minute of data (30-second sampling cadence)
- Rows with missing target values (future returns) are automatically dropped
- Feature computation is deterministic and can be re-run safely
- The SQLite feature store accumulates data across runs (no automatic cleanup)

