"""
Real-time feature builder for the alpha model serving layer.

Recomputes the same engineered features that were used during training,
so that the live option-chain snapshots can be transformed into the
expected feature vector order for inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ml_system.data.feature_pipeline import FeaturePipelineConfig


SnapshotInput = Union[pd.DataFrame, Iterable[Dict[str, object]]]


@dataclass
class RealTimeFeatureBuilderConfig:
    """Configuration for the real-time feature builder."""

    rows_per_minute: int
    lag_minutes: Tuple[int, ...]

    @classmethod
    def from_pipeline_config(cls, config: FeaturePipelineConfig) -> "RealTimeFeatureBuilderConfig":
        return cls(
            rows_per_minute=config.rows_per_minute,
            lag_minutes=tuple(config.lag_minutes),
        )


class RealTimeFeatureBuilder:
    """
    Transform recent option-chain snapshots into a feature vector that matches
    the training feature schema.
    """

    def __init__(
        self,
        feature_names: List[str],
        config: Optional[RealTimeFeatureBuilderConfig] = None,
    ) -> None:
        self.feature_names = feature_names
        pipeline_cfg = config or RealTimeFeatureBuilderConfig.from_pipeline_config(
            FeaturePipelineConfig()
        )
        self.rows_per_minute = pipeline_cfg.rows_per_minute
        self.lag_minutes = pipeline_cfg.lag_minutes

        # Derived requirements
        self.max_lag_rows = max(self.lag_minutes) * self.rows_per_minute if self.lag_minutes else 1

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def build_feature_vector(self, snapshots: SnapshotInput) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Convert recent snapshots into the model-ready feature vector.

        Args:
            snapshots: Iterable of dicts or DataFrame with option chain entries.

        Returns:
            (feature_vector, feature_map)

        Raises:
            ValueError: if insufficient history or required columns missing.
        """
        df = self._prepare_input(snapshots)
        if df.empty:
            raise ValueError("Snapshot input is empty – cannot build features.")

        aggregated = self._aggregate_features(df)
        if len(aggregated) <= self.max_lag_rows:
            raise ValueError(
                "Not enough historical aggregated rows to compute lagged features. "
                f"Need > {self.max_lag_rows}, received {len(aggregated)}."
            )

        enriched = self._add_lagged_features(aggregated)
        latest_row = enriched.iloc[-1]
        feature_map = latest_row.to_dict()

        feature_vector = np.array(
            [float(feature_map.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float32,
        )

        return feature_vector, feature_map

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _prepare_input(self, snapshots: SnapshotInput) -> pd.DataFrame:
        if isinstance(snapshots, pd.DataFrame):
            df = snapshots.copy()
        else:
            df = pd.DataFrame(list(snapshots))

        if df.empty:
            return df

        if "timestamp" not in df.columns:
            raise ValueError("Input snapshots must include a 'timestamp' column.")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")
        return df

    def _aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["is_call"] = df["option_type"] == "CE"
        df["is_put"] = df["option_type"] == "PE"
        df["is_itm"] = df.get("moneyness", "") == "ITM"

        grouped = df.groupby("timestamp").agg(
            total_oi=("oi", "sum"),
            oi_std=("oi", "std"),
            underlying_price=("underlying_price", "median"),
            strike_count=("strike", "nunique"),
        )

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
                    "oi_itm_sum": filtered.loc[filtered["moneyness"] == "ITM", "oi"].sum(),
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
        features["pcr"] = features["put_oi_total"] / features["call_oi_total"].replace(0, np.nan)
        features["itm_ratio"] = features["put_itm_oi"] / features["call_itm_oi"].replace(0, np.nan)
        features["oi_concentration"] = features["oi_std"] / features["total_oi"].replace(0, np.nan)
        features["net_itm_pressure"] = features["call_itm_oi"] - features["put_itm_oi"]
        features["net_oi"] = features["call_oi_total"] - features["put_oi_total"]

        return features

    def _add_lagged_features(self, features: pd.DataFrame) -> pd.DataFrame:
        lagged = features.copy()
        for minutes in self.lag_minutes:
            shift_rows = max(1, minutes * self.rows_per_minute)
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
        return lagged.fillna(0.0)


__all__ = ["RealTimeFeatureBuilder", "RealTimeFeatureBuilderConfig"]

