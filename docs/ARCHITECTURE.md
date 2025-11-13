# BSE / NSE OI Tracker – System Architecture

This document describes the end‑to‑end architecture of the OI Tracker project: how real‑time market data flows through the system, how it is persisted and transformed, and how it powers analytics, automation, and the web UI. It also documents the current SQLite schema.

---

## 1. High‑Level Overview

```
┌──────────────┐
│ Zerodha APIs │  (login, instruments, historical data)
└──────┬───────┘
       │
       │  (KiteTicker websocket ticks: prices, OI, volume)
┌──────▼────────────────────────────────────────────────────────────────────┐
│  oi_tracker_web.py                                                        │
│  • Login / session management                                             │
│  • Dynamic token subscription per exchange                                │
│  • run_data_update_loop_exchange():                                       │
│      - Market-hours gating                                                │
│      - Merge websocket + REST snapshots                                   │
│      - Compute pct-change analytics, Greeks, totals                       │
│      - Persist to SQLite & update caches                                  │
│      - Run Alpha model + baseline ML predictors                           │
│      - Manage paper-trade positions (manual + strategy-driven)            │
│      - Emit data via Flask-SocketIO                                       │
└──────────────┬────────────────────────────────────────────────────────────┘
               │
               │  (persisted snapshots, metadata)
       ┌───────▼────────────────┐
       │  SQLite `oi_tracker.db`│
       │  • option_chain_snapshots                                     │
       │  • exchange_metadata                                         │
       └──────────┬──────────────┘
                  │
                  │  (DataExtractor, FeaturePipeline)
┌─────────────────▼──────────────────────┐
│  ml_system/                             │
│  • data/feature_pipeline.py             │
│  • training/train_alpha_model.py        │
│  • serving/model_server.py              │
│  • serving/realtime_feature_builder.py  │
│  • predictions/signal_generator.py      │
│  • backtesting/backtest_engine.py       │
└─────────────────┬──────────────────────┘
                  │
                  │  (environment-configurable parameters)
┌─────────────────▼──────────────────────┐
│  Web UI (Flask + Socket.IO)            │
│  templates/index.html                  │
│  • interactive dashboard               │
│  • Alpha signal & confidence card       │
│  • Paper-trade positions panel          │
│  • Admin APIs (/api/place_order, etc.)  │
└────────────────────────────────────────┘
```

Key capabilities:
- **Real-time market visualization** for NIFTY (NSE) and SENSEX (BSE).
- **SQLite persistence** with restart recovery and analytical backfills.
- **ML pipelines** for baseline models, advanced alpha model (LightGBM/XGBoost) and walk-forward validation.
- **Automated Alpha strategy** that generates paper-trade signals, recorded alongside manual trades.
- **Socket.IO dashboard** with option-chain analytics, Greeks, PCR, delta metrics, and position tracking.

---

## 2. Runtime Data Flow

1. **Data ingestion**
   - Login credentials are supplied via environment variables or the login UI.
   - `kite_trade.py` handles login and enctoken fetch.
   - `KiteTicker` websocket streams ticks for underlying indices and subscribed option tokens.
   - REST API is used to backfill historical OI when websocket history is short (bootstrapping).

2. **Processing loop**
   - `run_data_update_loop_exchange()` (per exchange in its own thread) enforces trading-session windows (09:15–15:30 IST).
   - Option contracts around ATM are resolved dynamically using instrument metadata.
   - Raw OI history is merged (`get_oi_data_hybrid`) and normalized.
   - Analytics (`calculate_oi_differences`, theoretical pricing, IV, delta deltas) are computed.

3. **Persistence**
   - `database.save_option_chain_snapshot()` batches writes to `option_chain_snapshots`.
   - `exchange_metadata` stores last update time, underlying price, ATM strike per exchange.
   - Daily restart uses `load_today_snapshots` and `should_load_from_db` to recover state.

4. **Model inference**
   - Baseline ML predictor (`ml_system/predictions/realtime_predictor.py`) runs if legacy models are loaded.
   - Alpha model pipeline:
     - `build_alpha_snapshot_dataframe()` rebuilds training-like features from the current window.
     - `RealTimeFeatureBuilder` matches training feature order and handles lagged deltas.
     - Model/scaler metadata are loaded via `ml_system.serving.model_server.get_artifacts()`.
     - Predictions produce `{signal, confidence, probabilities}`.

5. **Automated Alpha strategy**
   - `execute_alpha_strategy()` converts Alpha signals into paper-trade positions using `register_paper_trade_position()`.
   - Confidence threshold and quantity are configurable via `ALPHA_CONFIDENCE_THRESHOLD`, `ALPHA_POSITION_QTY`, `ALPHA_MIN_OPTION_PRICE`.
   - Positions auto-close on signal reversal or manual/automatic stop (`monitor_positions`).

6. **UI emission**
   - Updated state is stored in `latest_oi_data` and emitted via Socket.IO channel `data_update_<exchange>`.
   - Clients render tables, charts, and Alpha card updates in `templates/index.html`.

7. **Backtesting & monitoring**
   - Historical data feeds DataExtractor → FeaturePipeline → BacktestEngine.
   - Logs (`oi_tracker_web.log`) capture paper-trade actions, signal changes, and errors.

---

## 3. Component Overview

| Layer | Module / Path | Responsibilities |
|-------|---------------|------------------|
| **Web & Control Plane** | `oi_tracker_web.py` | Flask app, login flow, websocket event loops, option-chain analytics, persistence, ML integration, paper trading, Socket.IO emission. |
| **Data Persistence** | `database.py` | SQLite schema management, snapshot writes, metadata tracking, cleanup helpers. |
| **ML Feature Engineering** | `ml_system/data/feature_pipeline.py` | Aggregations, lagged deltas, target labels, CLI & docs. |
| **Model Training** | `ml_system/training/train_alpha_model.py` | Walk-forward validation, LightGBM/XGBoost, metrics, artifact persistence. |
| **Model Serving** | `ml_system/serving/model_server.py`<br>`ml_system/serving/realtime_feature_builder.py` | FastAPI server (optional), cached artifact loader, realtime feature generation for inference. |
| **Prediction + Signals** | `ml_system/predictions/realtime_predictor.py`<br>`ml_system/predictions/signal_generator.py` | Baseline regression/classification predictions, trading signal heuristics. |
| **Backtesting & Monitoring** | `ml_system/backtesting/backtest_engine.py`<br>`ml_system/monitoring/performance_monitor.py` | Strategy evaluation, position accounting, metric dashboards. |
| **UI** | `templates/index.html` + `static/style.css` | Dashboard widgets, Alpha card, tables, charts, positions. |
| **Deployment** | `.github/workflows/deploy.yml`, `deploy/` | PM2-based deployment scripts, server setup, hot reload. |

---

## 4. Database Schema

### 4.1 `option_chain_snapshots`

Stores every 30-second snapshot of the watched option chain for both exchanges.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PRIMARY KEY | Auto-increment row id. |
| `timestamp` | TIMESTAMP | Snapshot time (UTC-naive). |
| `exchange` | TEXT | `'NSE'` or `'BSE'`. |
| `strike` | REAL | Strike price. |
| `option_type` | TEXT | `'CE'` or `'PE'`. |
| `symbol` | TEXT | Zerodha tradingsymbol. |
| `oi` | INTEGER | Open interest. |
| `ltp` | REAL | Last traded price. |
| `token` | INTEGER | Instrument token (primary lookup key). |
| `underlying_price` | REAL | Last known underlying (NIFTY/SENSEX). |
| `moneyness` | TEXT | Computed classification (ITM/ATM/OTM). |
| `pct_change_3m` ... `pct_change_30m` | REAL | Percentage OI change over rolling windows (minutes). |
| `created_at` | TIMESTAMP | Auto-set insertion timestamp. |

**Indexes**
- `idx_snapshots_exchange_timestamp (exchange, timestamp)`
- `idx_snapshots_token (token)`
- `idx_snapshots_moneyness (exchange, moneyness, timestamp)`
- `idx_snapshots_underlying_price (exchange, timestamp, underlying_price)`

### 4.2 `exchange_metadata`

Tracks latest heartbeat per exchange to support restart recovery and analytics.

| Column | Type | Description |
|--------|------|-------------|
| `exchange` | TEXT PRIMARY KEY | Exchange identifier. |
| `last_update_time` | TIMESTAMP | Last snapshot timestamp. |
| `last_atm_strike` | REAL | Most recent ATM strike. |
| `last_underlying_price` | REAL | Most recent underlying price. |
| `updated_at` | TIMESTAMP | Automatic update time. |

### Other helpers
- `get_previous_close_price(exchange)` fetches previous trading-day underlying close (used for UI deltas).
- `cleanup_old_data(days)` removes stale rows (default 30 days).
- `get_database_stats()` surfaces record counts, file size, metadata (useful for health dashboards).

---

## 5. Real-time Engine Details (`oi_tracker_web.py`)

1. **Trading session gating**
   - Market window: 09:15–15:30 IST (configurable).
   - Outside trading hours, the loop sleeps, sets status to *Pre-open* or *Market closed*, and skips API calls.

2. **Instrument selection**
   - `get_relevant_option_details()` picks strikes around ATM with configurable `options_count` (default ±5 strikes).
   - Token subscriptions are adjusted using `update_subscribed_tokens()` to follow ATM drift.

3. **Analytics computed per loop**
   - OI percentage changes for (3m, 5m, 10m, 15m, 30m).
   - Implied volatility via Black-Scholes.
   - Theoretical price and delta vs theo.
   - PCR, ITM/OTM splits, ATM derived metrics, delta percentage for ITM call/put baskets.

4. **Automated Alpha strategy**
   - Maintains `alpha_strategy_state` per exchange.
   - `execute_alpha_strategy()`:
     - Filters by confidence + price floor.
     - Chooses ATM call or put (prefers actual ATM, falls back to first with price).
     - Opens paper trade via `register_paper_trade_position()`.
     - Closes position on neutral signal or signal flip.
   - Trades appear in UI’s *Open Positions* cards and follow existing MTM/auto-exit logic.

5. **Manual paper trading**
   - `/api/place_order` and `/api/positions` provide JSON APIs.
   - Front-end modal supports BUY/SELL simulation; `monitor_positions()` handles MTM and thresholds.

6. **Socket.IO channels**
   - `data_update_NSE` / `data_update_BSE`: full payload per exchange on every refresh.
   - `position_closed`: emits closed position info for toast/alert display.

---

## 6. Machine Learning Ecosystem (`ml_system/`)

### 6.1 Feature pipeline
- `FeaturePipelineConfig` controls lookback window, lag intervals, target horizon, output destinations.
- CLI usage:  
  ```bash
  python -m ml_system.data.feature_pipeline --exchange NSE --lookback-days 30 --output-format both
  ```
- Outputs CSV to `ml_system/data/feature_sets/` and optional SQLite feature store.
- Documentation: `ml_system/data/FEATURE_PIPELINE.md`.

### 6.2 Model training (Alpha)
- `train_alpha_model.py`:
  - Walk-forward splits (configurable train/test windows).
  - Trains LightGBM or XGBoost multiclass classifier.
  - Logs fold metrics and confusion matrices.
  - Saves artifacts to `ml_system/models/alpha/` (`alpha_model_*.pkl`, scaler, feature names, metadata).
- Docs: `ml_system/training/ALPHA_MODEL_TRAINING.md`.

### 6.3 Model serving
- `ml_system/serving/model_server.py` exposes `/predict` and `/health` endpoints (optional FastAPI deployment).
- `get_artifacts()` caches model/scaler/feature names for both API and in-process use.
- `RealTimeFeatureBuilder` reconstructs features from live snapshots, ensures consistent column ordering, and sanitises NaNs/Infs.

### 6.4 Baseline predictors & backtesting
- `realtime_predictor.py` loads older models (Linear, Ridge, RandomForest, LSTM) for comparison.
- `signal_generator.py` creates trade recommendations based on regression outputs.
- `backtest_engine.py` + `PerformanceMonitor` evaluate strategy performance and record metrics.
- `ml_system/test_phase4.py` ties data loading, model training, backtesting, prediction, and monitoring into a regression test for the end-to-end ML stack.

---

## 7. Web UI (`templates/index.html`)

### Key elements
- **Info bar**: displays underlying price, previous close delta, PCR, VIX, ITM delta %, Alpha card when active.
- **Alpha card**: reveals signal (Bullish/Neutral/Bearish), confidence %, probability breakdown for each class. Hidden when model confidence is below threshold or no signal generated.
- **Option chain table**: symmetrical layout for calls/puts with OI % change heatmaps, theoretical pricing, strikes.
- **Positions panel**: lists manual and Alpha-generated paper trades with MTM updates.
- **Modals & actions**: manual BUY/SELL commands via `/api/place_order`.

### Client-side logic
- `updateUI` merges server payloads into DOM.
- `setDeltaValue`, `formatPriceDiff` handle conditional formatting.
- Alpha UI toggles rely on `alpha_signal`, `alpha_confidence`, `alpha_probabilities` fields.

---

## 8. Configuration & Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `ZERODHA_USER_ID`, `ZERODHA_PASSWORD` | Optional login prefill. | – |
| `DIFF_POS_THRESHOLD`, `DIFF_NEG_THRESHOLD` | Price diff highlighting thresholds. | 3.0 / 2.0 |
| `ALPHA_CONFIDENCE_THRESHOLD` | Minimum probability to place Alpha trade. | 0.6 |
| `ALPHA_POSITION_QTY` | Quantity for Alpha paper trades. | 50 |
| `ALPHA_MIN_OPTION_PRICE` | Skip options cheaper than this. | 5.0 |
| `OI_TRACKER_DB_PATH` | Custom DB location for analysis pipeline. | `oi_tracker.db` |
| `FLASK_SECRET_KEY` | Flask session signing. | `your-secret-key…` |

Additional pipeline options (`FEATURE_EXCHANGE`, etc.) are documented in pipeline & training READMEs.

---

## 9. Deployment Notes

1. **Local development**
   - Install dependencies: `pip install -r requirements.txt` (web) and `pip install -r ml_system/requirements_ml.txt` (ML).
   - Run server: `python oi_tracker_web.py` (ensuring `.env` contains credentials).

2. **Production**
   - GitHub Actions workflow `.github/workflows/deploy.yml` deploys via SSH and PM2 (no sudo required).
   - `deploy/setup_server.sh` provisions Python, Node, PM2, environment dependencies.
   - `deploy/nginx.conf.template` provides reverse-proxy config.

3. **Monitoring**
   - Logs: `oi_tracker_web.log` (rotated via PM2 config).
   - `get_database_stats()` can be exposed via API for dashboards.
   - Alpha strategy prints structured log lines for both entries and exits.

---

## 10. Reference Commands

```bash
# Generate features
python -m ml_system.data.feature_pipeline --exchange NSE --lookback-days 30

# Train alpha model with walk-forward validation
python -m ml_system.training.train_alpha_model --exchange NSE --model-type lightgbm

# Serve model via FastAPI (optional)
ALPHA_MODEL_DIR=ml_system/models/alpha uvicorn ml_system.serving.model_server:app

# Run Phase 4 end-to-end regression test
python -m ml_system.test_phase4
```

---

## 11. Future Enhancements

- Add historical Alpha strategy P&L charts to the UI.
- Expose REST/GraphQL endpoints for feature exports and model metrics.
- Automate daily feature generation & model retraining via scheduled jobs.
- Integrate alerting (email/Slack) for confidence spikes, unusual PCR, or Alpha flips.

---

This document should provide a complete blueprint for onboarding new developers, auditing the data model, or extending the system to additional exchanges or strategies.


