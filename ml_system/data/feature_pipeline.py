"""
Feature engineering pipeline for option-chain data.

Generates aggregated features, ratios, and lagged deltas from the
`option_chain_snapshots` table stored in the SQLite database.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .data_extractor import DataExtractor


DEFAULT_LAG_MINUTES = (1, 3, 5, 10, 15)


@dataclass
class FeaturePipelineConfig:
    db_path: str = "oi_tracker.db"
    output_dir: Path = Path("ml_system/data/feature_sets")
    feature_db_path: Optional[str] = "ml_system/data/feature_store.db"
    lookback_days: int = 30
    rows_per_minute: int = 2  # Approx. 30-second sampling cadence
    target_minutes: int = 15
    lag_minutes: tuple[int, ...] = field(default_factory=lambda: DEFAULT_LAG_MINUTES)


class FeaturePipeline:
    """
    Builds engineered features for downstream modelling.

    The pipeline loads raw option chain snapshots, aggregates them to
    timestamp-level features, computes ratio-based and lagged metrics,
    and saves the resulting dataset to CSV and/or SQLite.
    """

    def __init__(self, config: Optional[FeaturePipelineConfig] = None):
        self.config = config or FeaturePipelineConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = DataExtractor(db_path=self.config.db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_feature_set(self, exchange: str) -> pd.DataFrame:
        """Generate features for the given exchange."""
        end_date = pd.Timestamp.utcnow()
        start_date = end_date - pd.Timedelta(days=self.config.lookback_days)

        raw_df = self.extractor.get_exchange_data(
            exchange=exchange,
            start_date=start_date.to_pydatetime(),
            end_date=end_date.to_pydatetime(),
        )

        if raw_df.empty:
            raise ValueError(f"No data available for exchange {exchange}")

        features = self._aggregate_features(raw_df)
        features = self._add_lagged_features(features)
        features = self._add_targets(features)
        required_cols = [
            "future_return_pct",
            "future_direction_label",
            "pcr",
            "itm_ratio",
        ]
        features = features.dropna(subset=required_cols).reset_index()

        metadata = {
            "exchange": exchange,
            "lookback_days": self.config.lookback_days,
            "rows_per_minute": self.config.rows_per_minute,
            "target_minutes": self.config.target_minutes,
        }
        features.attrs.update(metadata)

        return features

    def save_feature_set(self, df: pd.DataFrame, exchange: str) -> Path:
        """Persist the generated dataset to CSV."""
        filename = f"{exchange.lower()}_features_{self.config.target_minutes}m.csv"
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        return path

    def save_feature_set_sqlite(
        self,
        df: pd.DataFrame,
        exchange: str,
        table_name: Optional[str] = None,
        if_exists: str = "replace",
    ) -> Optional[Path]:
        """Persist the dataset to an auxiliary SQLite feature store."""
        db_path = self.config.feature_db_path
        if not db_path:
            return None

        table = table_name or f"{exchange.lower()}_features"
        feature_db = Path(db_path)
        feature_db.parent.mkdir(parents=True, exist_ok=True)

        import sqlite3

        with sqlite3.connect(feature_db) as conn:
            df.to_sql(table, conn, if_exists=if_exists, index=False)

        return feature_db

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        df["is_call"] = df["option_type"] == "CE"
        df["is_put"] = df["option_type"] == "PE"
        df["is_itm"] = df["moneyness"] == "ITM"

        grouped = df.groupby("timestamp").agg(
            total_oi=("oi", "sum"),
            oi_std=("oi", "std"),
            underlying_price=("underlying_price", "median"),
            strike_count=("strike", "nunique"),
        )

        # Call/put specific aggregations
        iv_column = None
        for candidate in ("iv", "iv%", "implied_volatility"):
            if candidate in df.columns:
                iv_column = candidate
                break

        def aggregate_side(sub_df: pd.DataFrame, mask: pd.Series) -> pd.Series:
            filtered = sub_df.loc[mask]
            if filtered.empty:
                return pd.Series(
                    {
                        "oi_sum": 0.0,
                        "oi_itm_sum": 0.0,
                        "oi_otm_sum": 0.0,
                        "avg_iv": np.nan,
                    }
                )
            return pd.Series(
                {
                    "oi_sum": filtered["oi"].sum(),
                    "oi_itm_sum": filtered.loc[filtered["is_itm"], "oi"].sum(),
                    "oi_otm_sum": filtered.loc[filtered["moneyness"] == "OTM", "oi"].sum(),
                    "avg_iv": filtered[iv_column].mean() if iv_column else np.nan,
                }
            )

        call_features = df.groupby("timestamp", group_keys=False).apply(
            lambda g: aggregate_side(g, g["is_call"]),
            include_groups=False,
        )
        call_features = call_features.rename(
            columns={
                "oi_sum": "call_oi_total",
                "oi_itm_sum": "call_itm_oi",
                "oi_otm_sum": "call_otm_oi",
                "avg_iv": "call_avg_iv",
            }
        )

        put_features = df.groupby("timestamp", group_keys=False).apply(
            lambda g: aggregate_side(g, g["is_put"]),
            include_groups=False,
        )
        put_features = put_features.rename(
            columns={
                "oi_sum": "put_oi_total",
                "oi_itm_sum": "put_itm_oi",
                "oi_otm_sum": "put_otm_oi",
                "avg_iv": "put_avg_iv",
            }
        )

        features = grouped.join(call_features).join(put_features)

        # Ratios and composite metrics
        features["pcr"] = features["put_oi_total"] / features["call_oi_total"].replace(0, np.nan)
        features["itm_ratio"] = features["put_itm_oi"] / features["call_itm_oi"].replace(0, np.nan)
        features["oi_concentration"] = features["oi_std"] / features["total_oi"].replace(0, np.nan)
        features["net_itm_pressure"] = features["call_itm_oi"] - features["put_itm_oi"]
        features["net_oi"] = features["call_oi_total"] - features["put_oi_total"]

        return features

    def _add_lagged_features(self, features: pd.DataFrame) -> pd.DataFrame:
        lagged = features.copy()
        for minutes in self.config.lag_minutes:
            shift_rows = max(1, minutes * self.config.rows_per_minute)
            for col in [
                "call_oi_total",
                "put_oi_total",
                "call_itm_oi",
                "put_itm_oi",
                "pcr",
                "itm_ratio",
                "underlying_price",
            ]:
                change_col = f"{col}_chg_{minutes}m"
                lagged[change_col] = lagged[col] - lagged[col].shift(shift_rows)
        return lagged

    def _add_targets(self, features: pd.DataFrame) -> pd.DataFrame:
        target_shift = max(1, self.config.target_minutes * self.config.rows_per_minute)
        features = features.copy()
        future_price = features["underlying_price"].shift(-target_shift)
        features["future_return_pct"] = (
            (future_price - features["underlying_price"])
            / features["underlying_price"].replace(0, np.nan)
        ) * 100

        # Classification label: 1 bullish, -1 bearish, 0 neutral
        bullish_threshold = 0.2
        bearish_threshold = -0.2
        label = pd.Series(0, index=features.index)
        label[features["future_return_pct"] > bullish_threshold] = 1
        label[features["future_return_pct"] < bearish_threshold] = -1
        features["future_direction_label"] = label
        return features


def run_pipeline(exchange: str = "NSE") -> Path:
    pipeline = FeaturePipeline()
    feature_df = pipeline.build_feature_set(exchange)
    output_path = pipeline.save_feature_set(feature_df, exchange)
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate feature sets from option chain snapshots"
    )
    parser.add_argument(
        "--exchange",
        choices=["NSE", "BSE"],
        default="NSE",
        help="Exchange to process (default: NSE)"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Number of days to look back (default: 30)"
    )
    parser.add_argument(
        "--target-minutes",
        type=int,
        default=15,
        help="Forward-looking target window in minutes (default: 15)"
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "sqlite", "both"],
        default="both",
        help="Output format (default: both)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="oi_tracker.db",
        help="Path to source SQLite database (default: oi_tracker.db)"
    )
    
    args = parser.parse_args()
    
    config = FeaturePipelineConfig(
        db_path=args.db_path,
        lookback_days=args.lookback_days,
        target_minutes=args.target_minutes
    )
    
    pipeline = FeaturePipeline(config)
    
    try:
        print(f"Building feature set for {args.exchange}...")
        print(f"  Lookback: {args.lookback_days} days")
        print(f"  Target: {args.target_minutes} minutes ahead")
        
        feature_df = pipeline.build_feature_set(args.exchange)
        print(f"  Generated {len(feature_df)} feature rows")
        
        if args.output_format in ["csv", "both"]:
            csv_path = pipeline.save_feature_set(feature_df, args.exchange)
            print(f"  ✓ CSV saved: {csv_path}")
        
        if args.output_format in ["sqlite", "both"]:
            db_path = pipeline.save_feature_set_sqlite(feature_df, args.exchange)
            print(f"  ✓ SQLite saved: {db_path}")
        
        print(f"\n✅ Feature pipeline completed for {args.exchange}")
        
    finally:
        pipeline.extractor.close()

