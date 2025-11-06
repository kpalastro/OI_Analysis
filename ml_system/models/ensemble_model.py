"""
Phase 3: Ensemble Model
Combines multiple models for improved prediction accuracy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Ensemble model that combines predictions from multiple models."""
    
    def __init__(self, ensemble_method: str = 'weighted_average'):
        """
        Initialize ensemble predictor.
        
        Args:
            ensemble_method: Method to combine predictions
                - 'average': Simple average
                - 'weighted_average': Weighted average based on performance
                - 'stacking': Use meta-learner (not implemented yet)
        """
        self.ensemble_method = ensemble_method
        self.models = {}
        self.weights = {}
        self.model_performances = {}
    
    def add_model(self, name: str, model, weight: float = 1.0, performance: float = None):
        """
        Add a model to the ensemble.
        
        Args:
            name: Model name/identifier
            model: Trained model object
            weight: Weight for this model (for weighted average)
            performance: Performance metric (e.g., R²) for weight calculation
        """
        self.models[name] = model
        self.weights[name] = weight
        if performance is not None:
            self.model_performances[name] = performance
        
        logger.info(f"Added model '{name}' to ensemble (weight: {weight:.4f})")
    
    def calculate_weights_from_performance(self):
        """Calculate weights based on model performance."""
        if not self.model_performances:
            logger.warning("No performance metrics available. Using equal weights.")
            return
        
        # Convert performance to weights (higher performance = higher weight)
        # Normalize so weights sum to 1
        total_performance = sum(max(0, perf) for perf in self.model_performances.values())
        
        if total_performance > 0:
            for name in self.models.keys():
                perf = max(0, self.model_performances.get(name, 0))
                self.weights[name] = perf / total_performance
        else:
            # Equal weights if all performances are negative or zero
            equal_weight = 1.0 / len(self.models)
            for name in self.models.keys():
                self.weights[name] = equal_weight
        
        logger.info("Recalculated weights from performance metrics")
        for name, weight in self.weights.items():
            logger.info(f"  {name}: {weight:.4f}")
    
    def predict(self, X: pd.DataFrame, return_individual: bool = False) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Feature data
            return_individual: If True, return individual model predictions too
        
        Returns:
            Ensemble predictions (and optionally individual predictions)
        """
        if not self.models:
            raise ValueError("No models in ensemble. Add models first.")
        
        individual_predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                    # Handle different prediction formats
                    if isinstance(pred, np.ndarray):
                        if pred.ndim > 1:
                            pred = pred.flatten()
                    else:
                        pred = np.array(pred)
                    individual_predictions[name] = pred
                else:
                    logger.warning(f"Model '{name}' does not have predict method. Skipping.")
            except Exception as e:
                logger.error(f"Error predicting with model '{name}': {e}")
                continue
        
        if not individual_predictions:
            raise ValueError("No valid predictions obtained from any model.")
        
        # Combine predictions
        if self.ensemble_method == 'average':
            ensemble_pred = np.mean(list(individual_predictions.values()), axis=0)
        
        elif self.ensemble_method == 'weighted_average':
            # Calculate weighted average
            weighted_sum = np.zeros(len(list(individual_predictions.values())[0]))
            total_weight = 0
            
            for name, pred in individual_predictions.items():
                weight = self.weights.get(name, 1.0)
                weighted_sum += pred * weight
                total_weight += weight
            
            ensemble_pred = weighted_sum / total_weight if total_weight > 0 else weighted_sum
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        if return_individual:
            return ensemble_pred, individual_predictions
        else:
            return ensemble_pred
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        return_individual: bool = False
    ) -> Dict:
        """
        Evaluate ensemble performance.
        
        Args:
            X: Feature data
            y: True targets
            return_individual: If True, return individual model metrics too
        
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        predictions = self.predict(X, return_individual=return_individual)
        
        if return_individual:
            ensemble_pred, individual_preds = predictions
        else:
            ensemble_pred = predictions
            individual_preds = {}
        
        # Ensemble metrics
        mse = mean_squared_error(y, ensemble_pred)
        mae = mean_absolute_error(y, ensemble_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, ensemble_pred)
        direction_accuracy = np.mean(np.sign(y.values) == np.sign(ensemble_pred))
        
        results = {
            'ensemble': {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'direction_accuracy': direction_accuracy
            }
        }
        
        # Individual model metrics
        if return_individual and individual_preds:
            results['individual'] = {}
            for name, pred in individual_preds.items():
                results['individual'][name] = {
                    'mse': mean_squared_error(y, pred),
                    'mae': mean_absolute_error(y, pred),
                    'rmse': np.sqrt(mean_squared_error(y, pred)),
                    'r2': r2_score(y, pred),
                    'direction_accuracy': np.mean(np.sign(y.values) == np.sign(pred))
                }
        
        return results
    
    def save_ensemble(self, filepath: str):
        """Save ensemble configuration."""
        ensemble_data = {
            'ensemble_method': self.ensemble_method,
            'weights': self.weights,
            'model_performances': self.model_performances,
            'model_names': list(self.models.keys())
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Saved ensemble configuration to {filepath}")
    
    def load_ensemble(self, filepath: str):
        """Load ensemble configuration."""
        ensemble_data = joblib.load(filepath)
        self.ensemble_method = ensemble_data['ensemble_method']
        self.weights = ensemble_data['weights']
        self.model_performances = ensemble_data['model_performances']
        logger.info(f"Loaded ensemble configuration from {filepath}")


class StackingEnsemble:
    """Stacking ensemble with meta-learner."""
    
    def __init__(self, meta_learner=None):
        """
        Initialize stacking ensemble.
        
        Args:
            meta_learner: Meta-learner model (default: Linear Regression)
        """
        from sklearn.linear_model import LinearRegression
        
        self.base_models = {}
        self.meta_learner = meta_learner or LinearRegression()
        self.is_fitted = False
    
    def add_base_model(self, name: str, model):
        """Add a base model."""
        self.base_models[name] = model
        logger.info(f"Added base model '{name}' to stacking ensemble")
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ):
        """
        Fit stacking ensemble using validation set for meta-features.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (used to generate meta-features)
            y_val: Validation targets
        """
        logger.info("Training stacking ensemble...")
        
        # Generate meta-features from base models
        meta_features = []
        
        for name, model in self.base_models.items():
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X_val)
                    if isinstance(pred, np.ndarray) and pred.ndim > 1:
                        pred = pred.flatten()
                    meta_features.append(pred)
                else:
                    logger.warning(f"Model '{name}' does not have predict method. Skipping.")
            except Exception as e:
                logger.error(f"Error with model '{name}': {e}")
                continue
        
        if not meta_features:
            raise ValueError("No valid base model predictions obtained.")
        
        # Stack meta-features
        X_meta = np.column_stack(meta_features)
        
        # Train meta-learner
        self.meta_learner.fit(X_meta, y_val)
        self.is_fitted = True
        
        logger.info("Stacking ensemble training complete!")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using stacking ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Generate meta-features
        meta_features = []
        
        for name, model in self.base_models.items():
            try:
                pred = model.predict(X)
                if isinstance(pred, np.ndarray) and pred.ndim > 1:
                    pred = pred.flatten()
                meta_features.append(pred)
            except Exception as e:
                logger.error(f"Error predicting with model '{name}': {e}")
                continue
        
        if not meta_features:
            raise ValueError("No valid predictions obtained.")
        
        # Stack and predict with meta-learner
        X_meta = np.column_stack(meta_features)
        return self.meta_learner.predict(X_meta)


if __name__ == "__main__":
    # Example usage
    from ml_system.training.train_baseline import BaselineTrainer
    from ml_system.data.data_extractor import DataExtractor
    from ml_system.features.feature_engineer import FeatureEngineer
    
    extractor = DataExtractor()
    engineer = FeatureEngineer()
    trainer = BaselineTrainer()
    
    try:
        # Get and engineer data
        raw_data = extractor.get_time_series_data('NSE', lookback_days=30)
        features_df = engineer.engineer_all_features(raw_data)
        
        # Train baseline models
        results = trainer.train_all_baselines(features_df, target_col='price_change_pct', task='regression')
        
        # Create ensemble
        ensemble = EnsemblePredictor(ensemble_method='weighted_average')
        
        # Add models to ensemble
        for name, result in results.items():
            if 'predictions' in result:
                # In real usage, you'd load the saved models
                # For demo, we'll use the results
                ensemble.add_model(
                    name=name,
                    model=None,  # Would be loaded model
                    performance=result.get('test_r2', 0)
                )
        
        # Calculate weights from performance
        ensemble.calculate_weights_from_performance()
        
        print("Ensemble created successfully!")
    
    finally:
        extractor.close()

