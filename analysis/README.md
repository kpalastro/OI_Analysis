## OI Back-Analysis Toolkit

This package contains helper utilities to explore the historical option-chain
data stored in `oi_tracker.db`.  Use it to study how changes in Open Interest
(OI) interact with the underlying index price.

### Key Components

- `OIBackAnalysisPipeline`: loads snapshots from SQLite and produces
  analysis-ready time series, lagged features, and strike-vs-time matrices.
- `visualizations.py`: reusable Plotly chart builders (line charts, scatter,
  heatmaps) for notebooks or future UI integration.
- `strings.py`: centralised text catalogue to keep the codebase localisation-ready.

### Quick Start

```python
from analysis import OIBackAnalysisPipeline
from analysis.visualizations import plot_underlying_vs_oi, plot_heatmap

pipeline = OIBackAnalysisPipeline()
snapshots = pipeline.fetch_snapshots(
    exchange="NSE",
    start="2025-01-02 09:15",
    end="2025-01-02 15:30",
)

summary = pipeline.build_summary_table(snapshots, resample_rule="5min")
lagged = pipeline.build_lagged_features(summary, columns=["net_oi"], lags=[1, 2, 3])

fig = plot_underlying_vs_oi(summary, exchange="NSE")
fig.show()

heatmap_matrix = pipeline.build_strike_heatmap(snapshots, option_type="CE")
heatmap_fig = plot_heatmap(heatmap_matrix, title_suffix="NSE Calls – 5m Δ OI")
heatmap_fig.show()
```

### Suggested Analyses

- Correlate lagged `net_oi_change` with future `underlying_return_pct`.
- Compare OI surges across ITM/ATM/OTM buckets.
- Build intraday heatmaps to highlight strike-level pressure zones.

These outputs can feed both exploratory notebooks and UI widgets (e.g.,
historical insights tab) planned for the next iterations.

