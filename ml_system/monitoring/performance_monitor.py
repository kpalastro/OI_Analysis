"""
Phase 4: Performance Monitor
Tracks model performance and prediction accuracy over time.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitors model performance and prediction accuracy."""
    
    def __init__(self, log_file: str = "ml_system/monitoring/performance_log.json"):
        """
        Initialize performance monitor.
        
        Args:
            log_file: Path to log file for storing performance data
        """
        self.log_file = log_file
        self.predictions_log = []
        self.performance_metrics = {}
        
        # Create directory if needed
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Load existing log if available
        self.load_log()
    
    def log_prediction(
        self,
        timestamp: datetime,
        prediction: float,
        actual: Optional[float] = None,
        confidence: float = 1.0,
        signal: str = "HOLD"
    ):
        """
        Log a prediction for later evaluation.
        
        Args:
            timestamp: Prediction timestamp
            prediction: Predicted price change
            actual: Actual price change (if available)
            confidence: Confidence score
            signal: Generated signal
        """
        entry = {
            'timestamp': timestamp.isoformat(),
            'prediction': prediction,
            'actual': actual,
            'confidence': confidence,
            'signal': signal,
            'error': None,
            'direction_correct': None
        }
        
        # Calculate error if actual is available
        if actual is not None:
            entry['error'] = abs(prediction - actual)
            entry['direction_correct'] = np.sign(prediction) == np.sign(actual)
        
        self.predictions_log.append(entry)
        
        # Save periodically (every 10 entries)
        if len(self.predictions_log) % 10 == 0:
            self.save_log()
    
    def calculate_metrics(
        self,
        lookback_days: int = 7
    ) -> Dict:
        """
        Calculate performance metrics from logged predictions.
        
        Args:
            lookback_days: Number of days to look back
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.predictions_log:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(self.predictions_log)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by lookback period
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        df = df[df['timestamp'] >= cutoff_date]
        
        # Filter entries with actual values
        df_with_actual = df[df['actual'].notna()]
        
        if len(df_with_actual) == 0:
            logger.warning("No predictions with actual values for metrics calculation")
            return {}
        
        # Calculate metrics
        metrics = {
            'total_predictions': len(df),
            'predictions_with_actual': len(df_with_actual),
            'mean_absolute_error': df_with_actual['error'].mean() if 'error' in df_with_actual.columns else None,
            'mean_squared_error': (df_with_actual['error'] ** 2).mean() if 'error' in df_with_actual.columns else None,
            'direction_accuracy': df_with_actual['direction_correct'].mean() if 'direction_correct' in df_with_actual.columns else None,
            'avg_confidence': df['confidence'].mean(),
            'signal_distribution': df['signal'].value_counts().to_dict()
        }
        
        # Calculate accuracy by signal type
        if 'direction_correct' in df_with_actual.columns:
            signal_accuracy = {}
            for signal in df_with_actual['signal'].unique():
                signal_df = df_with_actual[df_with_actual['signal'] == signal]
                if len(signal_df) > 0:
                    signal_accuracy[signal] = signal_df['direction_correct'].mean()
            metrics['signal_accuracy'] = signal_accuracy
        
        self.performance_metrics = metrics
        return metrics
    
    def detect_drift(
        self,
        window_size: int = 50,
        threshold: float = 0.1
    ) -> Dict:
        """
        Detect model performance drift.
        
        Args:
            window_size: Size of rolling window
            threshold: Performance degradation threshold
        
        Returns:
            Dictionary with drift detection results
        """
        if len(self.predictions_log) < window_size * 2:
            return {'drift_detected': False, 'reason': 'Insufficient data'}
        
        df = pd.DataFrame(self.predictions_log)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[df['actual'].notna()].sort_values('timestamp')
        
        if len(df) < window_size * 2:
            return {'drift_detected': False, 'reason': 'Insufficient data with actuals'}
        
        # Calculate accuracy in recent vs older window
        recent = df.tail(window_size)
        older = df.iloc[-window_size * 2:-window_size]
        
        recent_accuracy = recent['direction_correct'].mean() if 'direction_correct' in recent.columns else None
        older_accuracy = older['direction_correct'].mean() if 'direction_correct' in older.columns else None
        
        if recent_accuracy is None or older_accuracy is None:
            return {'drift_detected': False, 'reason': 'No direction accuracy data'}
        
        accuracy_drop = older_accuracy - recent_accuracy
        
        drift_detected = accuracy_drop > threshold
        
        return {
            'drift_detected': drift_detected,
            'recent_accuracy': recent_accuracy,
            'older_accuracy': older_accuracy,
            'accuracy_drop': accuracy_drop,
            'threshold': threshold
        }
    
    def save_log(self):
        """Save prediction log to file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.predictions_log, f, indent=2)
            logger.debug(f"Saved {len(self.predictions_log)} log entries")
        except Exception as e:
            logger.error(f"Error saving log: {e}")
    
    def load_log(self):
        """Load prediction log from file."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.predictions_log = json.load(f)
                logger.info(f"Loaded {len(self.predictions_log)} log entries")
            except Exception as e:
                logger.error(f"Error loading log: {e}")
                self.predictions_log = []
        else:
            self.predictions_log = []
    
    def print_metrics(self, metrics: Optional[Dict] = None):
        """Print performance metrics."""
        if metrics is None:
            metrics = self.performance_metrics
        
        if not metrics:
            print("No metrics available")
            return
        
        print("\n" + "=" * 60)
        print("PERFORMANCE METRICS")
        print("=" * 60)
        print(f"Total Predictions:        {metrics.get('total_predictions', 0)}")
        print(f"With Actual Values:        {metrics.get('predictions_with_actual', 0)}")
        
        if metrics.get('mean_absolute_error') is not None:
            print(f"Mean Absolute Error:       {metrics['mean_absolute_error']:.4f}")
        
        if metrics.get('direction_accuracy') is not None:
            print(f"Direction Accuracy:        {metrics['direction_accuracy']:.2%}")
        
        print(f"Average Confidence:        {metrics.get('avg_confidence', 0):.2%}")
        
        if 'signal_distribution' in metrics:
            print(f"\nSignal Distribution:")
            for signal, count in metrics['signal_distribution'].items():
                print(f"  {signal}: {count}")
        
        if 'signal_accuracy' in metrics:
            print(f"\nSignal Accuracy:")
            for signal, accuracy in metrics['signal_accuracy'].items():
                print(f"  {signal}: {accuracy:.2%}")
        
        print("=" * 60)


if __name__ == "__main__":
    # Example usage
    monitor = PerformanceMonitor()
    
    # Simulate some predictions
    base_time = datetime.now()
    for i in range(10):
        monitor.log_prediction(
            timestamp=base_time + timedelta(minutes=i * 15),
            prediction=0.03 + np.random.normal(0, 0.01),
            actual=0.025 + np.random.normal(0, 0.01),
            confidence=0.7 + np.random.uniform(-0.1, 0.1),
            signal="BUY" if i % 3 == 0 else "HOLD"
        )
    
    # Calculate metrics
    metrics = monitor.calculate_metrics()
    monitor.print_metrics(metrics)
    
    # Check for drift
    drift = monitor.detect_drift()
    print(f"\nDrift Detected: {drift.get('drift_detected', False)}")

