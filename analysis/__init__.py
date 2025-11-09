"""
Analysis utilities for exploring historical option-chain data.

This package currently provides helper classes to extract aggregated
time series suitable for visualising how open interest (OI) interacts
with the underlying instrument price.
"""

from .oi_backanalysis import OIBackAnalysisPipeline, OISummaryConfig

__all__ = ["OIBackAnalysisPipeline", "OISummaryConfig"]

