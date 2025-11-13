"""
Walk-Forward Validation Framework for Time-Series Data

Implements rigorous time-based cross-validation that prevents data leakage
by ensuring training data always precedes test data chronologically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@dataclass
class WalkForwardSplit:
    """Represents a single train/test split in walk-forward validation."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_indices: pd.Index
    test_indices: pd.Index
    fold_number: int


class WalkForwardValidator:
    """
    Implements walk-forward validation for time-series data.
    
    This ensures that:
    - Training data always comes before test data chronologically
    - No future information leaks into training
    - Each fold tests on a distinct time period
    
    Example:
        # Train on Month 1, test on Month 2
        # Train on Months 1-2, test on Month 3
        # Train on Months 1-3, test on Month 4
        # etc.
    """
    
    def __init__(
        self,
        initial_train_days: int = 30,
        test_days: int = 7,
        step_days: Optional[int] = None,
        min_train_samples: int = 100,
        min_test_samples: int = 10
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            initial_train_days: Days of data for first training fold
            test_days: Days of data for each test fold
            step_days: Days to step forward between folds (default: test_days)
            min_train_samples: Minimum samples required in training set
            min_test_samples: Minimum samples required in test set
        """
        self.initial_train_days = initial_train_days
        self.test_days = test_days
        self.step_days = step_days or test_days
        self.min_train_samples = min_train_samples
        self.min_test_samples = min_test_samples
    
    def split(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> Iterator[WalkForwardSplit]:
        """
        Generate walk-forward train/test splits.
        
        Args:
            df: DataFrame with time-series data
            timestamp_col: Name of timestamp column
            
        Yields:
            WalkForwardSplit objects with train/test indices
        """
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame")
        
        # Ensure DataFrame is sorted by timestamp
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        
        if len(df_sorted) == 0:
            return
        
        # Get time range
        timestamps = pd.to_datetime(df_sorted[timestamp_col])
        start_time = timestamps.min()
        end_time = timestamps.max()
        
        # Calculate time deltas
        initial_train_delta = timedelta(days=self.initial_train_days)
        test_delta = timedelta(days=self.test_days)
        step_delta = timedelta(days=self.step_days)
        
        # Generate folds
        fold_number = 0
        train_end = start_time + initial_train_delta
        
        while train_end < end_time:
            test_start = train_end
            test_end = min(test_start + test_delta, end_time)
            
            # Get indices for this fold
            train_mask = (timestamps >= start_time) & (timestamps < train_end)
            test_mask = (timestamps >= test_start) & (timestamps < test_end)
            
            train_indices = df_sorted.index[train_mask]
            test_indices = df_sorted.index[test_mask]
            
            # Validate minimum sample requirements
            if len(train_indices) < self.min_train_samples:
                break
            
            if len(test_indices) < self.min_test_samples:
                # Skip this fold but continue
                train_end += step_delta
                continue
            
            yield WalkForwardSplit(
                train_start=start_time,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_indices=train_indices,
                test_indices=test_indices,
                fold_number=fold_number
            )
            
            # Move forward
            train_end += step_delta
            fold_number += 1
    
    def get_split_summary(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Get a summary of all splits without actually splitting.
        
        Useful for understanding the validation strategy before training.
        
        Returns:
            DataFrame with columns: fold, train_start, train_end, test_start, 
            test_end, train_samples, test_samples
        """
        splits = []
        for split in self.split(df, timestamp_col):
            splits.append({
                'fold': split.fold_number,
                'train_start': split.train_start,
                'train_end': split.train_end,
                'test_start': split.test_start,
                'test_end': split.test_end,
                'train_samples': len(split.train_indices),
                'test_samples': len(split.test_indices),
                'train_days': (split.train_end - split.train_start).days,
                'test_days': (split.test_end - split.test_start).days
            })
        
        return pd.DataFrame(splits)


def validate_walk_forward_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_col: str = 'timestamp'
) -> bool:
    """
    Validate that a train/test split is chronologically correct.
    
    Ensures no data leakage by checking that all training timestamps
    come before all test timestamps.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        timestamp_col: Name of timestamp column
        
    Returns:
        True if split is valid (no leakage), False otherwise
    """
    if timestamp_col not in train_df.columns or timestamp_col not in test_df.columns:
        return False
    
    train_max = pd.to_datetime(train_df[timestamp_col]).max()
    test_min = pd.to_datetime(test_df[timestamp_col]).min()
    
    return train_max < test_min


if __name__ == "__main__":
    # Example usage
    from ml_system.data.feature_pipeline import FeaturePipeline
    
    print("=" * 60)
    print("Walk-Forward Validation Example")
    print("=" * 60)
    
    # Load feature data
    pipeline = FeaturePipeline()
    try:
        df = pipeline.build_feature_set('NSE')
        
        if df.empty:
            print("No data available")
            exit(1)
        
        print(f"\nLoaded {len(df)} feature rows")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Create validator
        validator = WalkForwardValidator(
            initial_train_days=30,
            test_days=7,
            step_days=7
        )
        
        # Get split summary
        summary = validator.get_split_summary(df)
        print(f"\nGenerated {len(summary)} walk-forward folds:")
        print(summary.to_string(index=False))
        
        # Validate splits
        print("\n" + "=" * 60)
        print("Validating splits (checking for data leakage)...")
        print("=" * 60)
        
        all_valid = True
        for split in validator.split(df):
            train_df = df.loc[split.train_indices]
            test_df = df.loc[split.test_indices]
            
            is_valid = validate_walk_forward_split(train_df, test_df)
            all_valid = all_valid and is_valid
            
            status = "✓" if is_valid else "✗"
            print(f"Fold {split.fold_number}: {status} "
                  f"Train: {len(train_df)} samples, "
                  f"Test: {len(test_df)} samples")
        
        if all_valid:
            print("\n✅ All splits are valid (no data leakage detected)")
        else:
            print("\n❌ Some splits have data leakage!")
            
    finally:
        pipeline.extractor.close()

