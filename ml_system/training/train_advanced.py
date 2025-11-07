"""
Phase 3: Advanced Model Training
Trains LSTM and ensemble models for improved prediction accuracy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from sklearn.model_selection import train_test_split

from ml_system.models.lstm_model import LSTMPredictor, TENSORFLOW_AVAILABLE
from ml_system.models.ensemble_model import EnsemblePredictor, StackingEnsemble
from ml_system.training.train_baseline import BaselineTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedTrainer:
    """Trains advanced models (LSTM, Ensemble)."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize advanced trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
    
    def train_lstm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sequence_length: int = 60,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Optional[Dict]:
        """
        Train LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            sequence_length: LSTM sequence length
            epochs: Training epochs
            batch_size: Batch size
        
        Returns:
            Dictionary with training results
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Skipping LSTM training.")
            return None
        
        logger.info("=" * 60)
        logger.info("Training LSTM Model")
        logger.info("=" * 60)
        
        try:
            # Create LSTM model
            lstm = LSTMPredictor(
                sequence_length=sequence_length,
                hidden_units=[128, 64, 32],
                dropout_rate=0.2,
                learning_rate=0.001
            )
            
            # Train
            history = lstm.train(
                X_train, y_train,
                X_val=X_test, y_val=y_test,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.0,  # Using explicit validation set
                verbose=1
            )
            
            # Evaluate
            eval_results = lstm.evaluate(X_test, y_test)
            
            # Get predictions
            predictions = lstm.predict(X_test)
            valid_mask = ~np.isnan(predictions)
            predictions = predictions[valid_mask]
            
            results = {
                'model_name': 'LSTM',
                'model': lstm,
                'train_mse': history['history']['loss'][-1],
                'test_mse': eval_results['mse'],
                'test_mae': eval_results['mae'],
                'test_rmse': eval_results['rmse'],
                'test_r2': eval_results['r2'],
                'direction_accuracy': eval_results['direction_accuracy'],
                'predictions': predictions,
                'history': history
            }
            
            self.models['lstm'] = lstm
            logger.info(f"LSTM - Test R²: {results['test_r2']:.4f}, MAE: {results['test_mae']:.4f}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error training LSTM: {e}", exc_info=True)
            return None
    
    def create_ensemble(
        self,
        baseline_results: Dict,
        lstm_result: Optional[Dict] = None,
        method: str = 'weighted_average'
    ) -> Dict:
        """
        Create ensemble from baseline and advanced models.
        
        Args:
            baseline_results: Results from baseline models
            lstm_result: Results from LSTM model (optional)
            method: Ensemble method ('weighted_average' or 'stacking')
        
        Returns:
            Dictionary with ensemble results
        """
        logger.info("=" * 60)
        logger.info(f"Creating {method} Ensemble")
        logger.info("=" * 60)
        
        if method == 'weighted_average':
            ensemble = EnsemblePredictor(ensemble_method='weighted_average')
            
            # Add baseline models (using their predictions)
            for name, result in baseline_results.items():
                if 'predictions' in result:
                    ensemble.add_model(
                        name=name,
                        model=None,  # We'll use predictions directly
                        performance=result.get('test_r2', 0)
                    )
                    ensemble.model_performances[name] = result.get('test_r2', 0)
            
            # Add LSTM if available
            if lstm_result and 'predictions' in lstm_result:
                ensemble.add_model(
                    name='lstm',
                    model=None,
                    performance=lstm_result.get('test_r2', 0)
                )
                ensemble.model_performances['lstm'] = lstm_result.get('test_r2', 0)
            
            # Calculate weights
            ensemble.calculate_weights_from_performance()
            
            # For weighted average, we need to combine predictions
            # This is a simplified version - in production, you'd use actual model objects
            all_predictions = []
            all_weights = []
            
            for name in ensemble.models.keys():
                if name in baseline_results and 'predictions' in baseline_results[name]:
                    all_predictions.append(baseline_results[name]['predictions'])
                    all_weights.append(ensemble.weights[name])
                elif name == 'lstm' and lstm_result and 'predictions' in lstm_result:
                    all_predictions.append(lstm_result['predictions'])
                    all_weights.append(ensemble.weights[name])
            
            if not all_predictions:
                logger.warning("No valid predictions for ensemble")
                return {}
            
            # Ensure all predictions have same length
            min_len = min(len(p) for p in all_predictions)
            all_predictions = [p[:min_len] for p in all_predictions]
            
            # Weighted average
            weights_array = np.array(all_weights)
            weights_array = weights_array / weights_array.sum()  # Normalize
            
            ensemble_pred = np.zeros(min_len)
            for pred, weight in zip(all_predictions, weights_array):
                ensemble_pred += pred[:min_len] * weight
            
            # Calculate metrics (would need true y_test)
            results = {
                'model_name': f'Ensemble ({method})',
                'ensemble': ensemble,
                'predictions': ensemble_pred,
                'weights': dict(zip(ensemble.models.keys(), weights_array))
            }
            
            logger.info("Ensemble created successfully!")
            for name, weight in results['weights'].items():
                logger.info(f"  {name}: {weight:.4f}")
            
            return results
        
        elif method == 'stacking':
            # Stacking ensemble would require actual model objects
            logger.warning("Stacking ensemble requires model objects. Use weighted_average for now.")
            return {}
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def train_all_advanced(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        baseline_results: Dict,
        train_lstm: bool = True,
        create_ensemble: bool = True
    ) -> Dict:
        """
        Train all advanced models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            baseline_results: Results from baseline models
            train_lstm: Whether to train LSTM
            create_ensemble: Whether to create ensemble
        
        Returns:
            Dictionary with all results
        """
        all_results = {}
        
        # Train LSTM
        if train_lstm:
            lstm_result = self.train_lstm(X_train, y_train, X_test, y_test)
            if lstm_result:
                all_results['lstm'] = lstm_result
        
        # Create ensemble
        if create_ensemble:
            ensemble_result = self.create_ensemble(
                baseline_results,
                all_results.get('lstm'),
                method='weighted_average'
            )
            if ensemble_result:
                all_results['ensemble'] = ensemble_result
        
        self.results = all_results
        return all_results
    
    def save_models(self, output_dir: str = "ml_system/models"):
        """Save trained models."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save LSTM
        if 'lstm' in self.models:
            lstm_path = os.path.join(output_dir, "lstm_model.keras")
            self.models['lstm'].save_model(lstm_path)
        
        # Save ensemble
        if 'ensemble' in self.results and 'ensemble' in self.results['ensemble']:
            ensemble_path = os.path.join(output_dir, "ensemble_config.pkl")
            self.results['ensemble']['ensemble'].save_ensemble(ensemble_path)
        
        logger.info(f"Saved advanced models to {output_dir}")


if __name__ == "__main__":
    # Example usage
    from ml_system.data.data_extractor import DataExtractor
    from ml_system.features.feature_engineer import FeatureEngineer
    
    extractor = DataExtractor()
    engineer = FeatureEngineer()
    baseline_trainer = BaselineTrainer()
    advanced_trainer = AdvancedTrainer()
    
    try:
        # Get and engineer data
        raw_data = extractor.get_time_series_data('NSE', lookback_days=30)
        features_df = engineer.engineer_all_features(raw_data)
        
        # Prepare data
        feature_cols = [col for col in features_df.columns 
                       if col not in ['timestamp', 'future_price', 'price_change', 'price_change_pct', 'direction']]
        X = features_df[feature_cols]
        y = features_df['price_change_pct']
        
        # Remove NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Split data (time-based)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train baseline models
        baseline_results = baseline_trainer.train_all_baselines(
            features_df.iloc[valid_mask],
            target_col='price_change_pct',
            task='regression'
        )
        
        # Train advanced models
        advanced_results = advanced_trainer.train_all_advanced(
            X_train, y_train, X_test, y_test,
            baseline_results,
            train_lstm=True,
            create_ensemble=True
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("Advanced Models Summary")
        print("=" * 60)
        for name, result in advanced_results.items():
            if 'test_r2' in result:
                print(f"{result['model_name']}: R² = {result['test_r2']:.4f}, MAE = {result['test_mae']:.4f}")
    
    finally:
        extractor.close()

