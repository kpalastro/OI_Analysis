"""
Utilities to prepare historical OI datasets for exploratory analysis.

The pipeline encapsulates common transformations required to study the
relationship between option open interest (OI) dynamics and the
underlying index price.  Typical outputs include time-series tables with
aggregated OI per option type/moneyness, weighted percentage-change
metrics, and strike-vs-time matrices for heatmap visualisations.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_DB_ENV_VAR = "OI_TRACKER_DB_PATH"
DEFAULT_DB_FILENAME = "oi_tracker.db"


@dataclass(frozen=True)
class OISummaryConfig:
    """Configuration for aggregating OI summary time-series."""

    exchange: str
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    strikes: Optional[Sequence[float]] = None
    option_types: Optional[Sequence[str]] = None
    resample_rule: Optional[str] = None  # e.g. "5min", "15min"
    include_pct_changes: bool = True
    include_heatmap_data: bool = True


class OIBackAnalysisPipeline:
    """
    High-level helper for extracting and transforming option-chain snapshots.

    Example
    -------
    >>> pipeline = OIBackAnalysisPipeline()
    >>> snapshots = pipeline.fetch_snapshots(exchange="NSE",
    ...                                      start="2025-01-06 09:15",
    ...                                      end="2025-01-06 15:30")
    >>> summary = pipeline.build_summary_table(snapshots)
    >>> heatmap = pipeline.build_strike_heatmap(snapshots, option_type="CE")
    """

    def __init__(self, db_path: Optional[os.PathLike] = None) -> None:
        if db_path is None:
            db_path = os.getenv(DEFAULT_DB_ENV_VAR, DEFAULT_DB_FILENAME)
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------
    def fetch_snapshots(
        self,
        exchange: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        strikes: Optional[Sequence[float]] = None,
        option_types: Optional[Sequence[str]] = None,
        columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        Retrieve raw snapshots for the requested filters.

        Parameters
        ----------
        exchange:
            Either "NSE" or "BSE".
        start / end:
            Optional datetime bounds (inclusive start, exclusive end).
        strikes:
            Optional list of strikes to keep.
        option_types:
            Option types to include (defaults to both "CE" and "PE").
        columns:
            Optional subset of columns to fetch. When omitted the default
            analytical columns are selected.
        """

        if not exchange:
            raise ValueError("exchange must be provided")

        if columns is None:
            columns = (
                "timestamp",
                "exchange",
                "strike",
                "option_type",
                "symbol",
                "oi",
                "ltp",
                "underlying_price",
                "moneyness",
                "pct_change_5m",
                "pct_change_10m",
                "pct_change_15m",
                "pct_change_30m",
            )

        query, params = self._build_snapshot_query(
            exchange=exchange,
            columns=columns,
            start=start,
            end=end,
            strikes=strikes,
            option_types=option_types,
        )

        with self._open_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=["timestamp"])

        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Normalise enum fields
        df["option_type"] = df["option_type"].str.upper()
        if "moneyness" in df.columns:
            df["moneyness"] = df["moneyness"].str.upper()

        return df

    # ------------------------------------------------------------------
    # Transformation helpers
    # ------------------------------------------------------------------
    def build_summary_table(
        self,
        snapshots: pd.DataFrame,
        *,
        resample_rule: Optional[str] = None,
        include_pct_changes: bool = True,
    ) -> pd.DataFrame:
        """
        Produce a time-series summary combining OI totals and underlying price.

        The returned DataFrame contains, at minimum:
            - call_oi / put_oi
            - net_oi (PUT - CALL)
            - pcr (safe division)
            - underlying_price, underlying_change, underlying_return_pct
            - per-moneyness OI columns (e.g. call_oi_atm, put_oi_itm)
            - weighted OI pct-change metrics (optional)
        """

        if snapshots.empty:
            return pd.DataFrame()

        df = snapshots.copy()
        df.set_index("timestamp", inplace=True)

        # Option-type totals
        totals = df.pivot_table(
            index="timestamp",
            columns="option_type",
            values="oi",
            aggfunc="sum",
            fill_value=0,
        )
        totals.columns = [f"{col.lower()}_oi" for col in totals.columns]

        summary = totals

        # Underlying price
        underlying = df.groupby("timestamp")["underlying_price"].mean()
        summary = summary.join(underlying.rename("underlying_price"), how="left")
        summary["underlying_change"] = summary["underlying_price"].diff()
        summary["underlying_return_pct"] = (
            summary["underlying_price"].pct_change() * 100.0
        )

        # Net OI, PCR
        summary["net_oi"] = summary.get("put_oi", 0.0) - summary.get("call_oi", 0.0)
        summary["total_oi"] = summary.get("put_oi", 0.0) + summary.get("call_oi", 0.0)
        summary["pcr"] = summary.apply(
            lambda row: row["put_oi"] / row["call_oi"]
            if row.get("call_oi") not in (None, 0)
            else np.nan,
            axis=1,
        )

        # Moneyness splits
        if "moneyness" in df.columns:
            money = (
                df.pivot_table(
                    index="timestamp",
                    columns=["option_type", "moneyness"],
                    values="oi",
                    aggfunc="sum",
                    fill_value=0,
                )
                .sort_index(axis=1)
            )
            money.columns = [
                f"{opt.lower()}_oi_{mny.lower()}" for opt, mny in money.columns
            ]
            summary = summary.join(money, how="left")

        if include_pct_changes:
            pct_cols = [
                col for col in df.columns if col.startswith("pct_change_") and col != "pct_change"
            ]
            if pct_cols:
                weights = df["oi"].where(df["oi"] > 0, other=np.nan)
                pct_data: Mapping[str, pd.Series] = {}

                for opt in sorted(df["option_type"].unique()):
                    opt_mask = df["option_type"] == opt
                    opt_weights = weights[opt_mask]
                    opt_weights = opt_weights.groupby(level=0).mean()
                    opt_grouped = df.loc[opt_mask]
                    grouped = opt_grouped.groupby("timestamp")
                    for pct_col in pct_cols:
                        if pct_col not in opt_grouped:
                            continue
                        series = grouped.apply(
                            lambda g, col=pct_col: _weighted_mean(
                                g[col], opt_weights.loc[g.index]
                            )
                        )
                        pct_data[
                            f"{opt.lower()}_{pct_col}_weighted"
                        ] = series.replace([np.inf, -np.inf], np.nan)

                if pct_data:
                    pct_frame = pd.DataFrame(pct_data)
                    summary = summary.join(pct_frame, how="left")

        if resample_rule:
            summary = (
                summary.resample(resample_rule)
                .mean()
                .interpolate(limit_direction="both")
                .ffill()
            )

        summary.reset_index(inplace=True)
        return summary

    def build_strike_heatmap(
        self,
        snapshots: pd.DataFrame,
        *,
        option_type: str = "CE",
        value_column: str = "pct_change_5m",
        aggfunc: str = "mean",
    ) -> pd.DataFrame:
        """
        Create a strike vs time matrix (suitable for heatmaps).

        Parameters
        ----------
        option_type:
            Filter to "CE" or "PE".
        value_column:
            Column to pivot (default: 5-minute pct change). Use "oi" or "ltp"
            when raw values are preferred.
        aggfunc:
            Aggregation function passed to `pivot_table`.
        """

        if snapshots.empty:
            return pd.DataFrame()

        option_type = option_type.upper()
        data = snapshots[snapshots["option_type"] == option_type].copy()
        if data.empty:
            return pd.DataFrame()

        if value_column not in data.columns:
            raise ValueError(f"Column '{value_column}' not found in snapshots DataFrame")

        matrix = data.pivot_table(
            index="strike",
            columns="timestamp",
            values=value_column,
            aggfunc=aggfunc,
        ).sort_index()

        return matrix

    def build_lagged_features(
        self,
        summary_table: pd.DataFrame,
        columns: Sequence[str],
        lags: Sequence[int],
    ) -> pd.DataFrame:
        """
        Append lagged versions of selected columns (in rows).

        Useful for correlation studies such as comparing ΔOI at lag +1
        to future underlying returns.
        """

        if summary_table.empty:
            return summary_table.copy()

        df = summary_table.copy()
        df.set_index("timestamp", inplace=True)

        for col in columns:
            if col not in df.columns:
                continue
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        df.reset_index(inplace=True)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _build_snapshot_query(
        self,
        *,
        exchange: str,
        columns: Iterable[str],
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        strikes: Optional[Sequence[float]],
        option_types: Optional[Sequence[str]],
    ) -> Tuple[str, Tuple]:
        base_columns = ", ".join(columns)
        query = [
            f"SELECT {base_columns} FROM option_chain_snapshots WHERE exchange = ?"
        ]
        params: List = [exchange.upper()]

        if start is not None:
            start_ts = pd.Timestamp(start).to_pydatetime()
            query.append("AND timestamp >= ?")
            params.append(start_ts)

        if end is not None:
            end_ts = pd.Timestamp(end).to_pydatetime()
            query.append("AND timestamp < ?")
            params.append(end_ts)

        if strikes:
            placeholders = ", ".join(["?"] * len(strikes))
            query.append(f"AND strike IN ({placeholders})")
            params.extend(strikes)

        if option_types:
            cleaned = [opt.upper() for opt in option_types]
            placeholders = ", ".join(["?"] * len(cleaned))
            query.append(f"AND option_type IN ({placeholders})")
            params.extend(cleaned)

        query.append("ORDER BY timestamp ASC, strike ASC")
        return " ".join(query), tuple(params)


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    if values.empty:
        return np.nan

    aligned = pd.DataFrame({"value": values})
    aligned["weight"] = weights

    mask = aligned["value"].notna() & aligned["weight"].notna()
    if not mask.any():
        return np.nan

    v = aligned.loc[mask, "value"].to_numpy()
    w = aligned.loc[mask, "weight"].to_numpy()
    total = w.sum()

    if total == 0 or np.isnan(total):
        return float(np.nanmean(v))

    return float(np.dot(v, w) / total)

