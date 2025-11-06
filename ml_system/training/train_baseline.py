"""
Phase 2: Baseline Model Training
Trains baseline models (Linear Regression, Random Forest, XGBoost) for comparison.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
import joblib
import logging
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineTrainer:
    """Trains and evaluates baseline ML models."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the baseline trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.results = {}
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'price_change_pct',
        test_size: float = 0.2,
        use_time_split: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features and targets
            target_col: Name of target column
            test_size: Proportion of data for testing
            use_time_split: Use time-based split (recommended for time series)
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Get feature columns (exclude target and metadata)
        exclude_cols = ['timestamp', 'future_price', 'price_change', 'price_change_pct', 'direction']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Remove rows with NaN in target
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Prepared data: {len(X)} samples, {len(feature_cols)} features")
        logger.info(f"Target: {target_col}, Range: [{y.min():.4f}, {y.max():.4f}]")
        
        # Time-based split (recommended for time series)
        if use_time_split and 'timestamp' in df.columns:
            # Sort by timestamp
            sorted_indices = df[valid_mask].sort_values('timestamp').index
            split_idx = int(len(sorted_indices) * (1 - test_size))
            train_indices = sorted_indices[:split_idx]
            test_indices = sorted_indices[split_idx:]
            
            X_train = X.loc[train_indices]
            X_test = X.loc[test_indices]
            y_train = y.loc[train_indices]
            y_test = y.loc[test_indices]
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
        
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_linear_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        scale_features: bool = True
    ) -> Dict:
        """Train Linear Regression model."""
        logger.info("Training Linear Regression...")
        
        # Scale features
        if scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['linear'] = scaler
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        self.models['linear'] = model
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Metrics
        results = {
            'model_name': 'Linear Regression',
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'predictions': y_pred_test
        }
        
        logger.info(f"Linear Regression - Test R²: {results['test_r2']:.4f}, MAE: {results['test_mae']:.4f}")
        return results
    
    def train_ridge_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        alpha: float = 1.0
    ) -> Dict:
        """Train Ridge Regression model."""
        logger.info("Training Ridge Regression...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['ridge'] = scaler
        
        model = Ridge(alpha=alpha, random_state=self.random_state)
        model.fit(X_train_scaled, y_train)
        self.models['ridge'] = model
        
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        results = {
            'model_name': 'Ridge Regression',
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'predictions': y_pred_test
        }
        
        logger.info(f"Ridge Regression - Test R²: {results['test_r2']:.4f}, MAE: {results['test_mae']:.4f}")
        return results
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        task: str = 'regression'
    ) -> Dict:
        """Train Random Forest model."""
        logger.info(f"Training Random Forest ({task})...")
        
        if task == 'regression':
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        if task == 'regression':
            results = {
                'model_name': 'Random Forest (Regression)',
                'train_mse': mean_squared_error(y_train, y_pred_train),
                'test_mse': mean_squared_error(y_test, y_pred_test),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'predictions': y_pred_test,
                'feature_importance': dict(zip(self.feature_names, model.feature_importances_))
            }
            logger.info(f"Random Forest - Test R²: {results['test_r2']:.4f}, MAE: {results['test_mae']:.4f}")
        else:
            results = {
                'model_name': 'Random Forest (Classification)',
                'train_accuracy': accuracy_score(y_train, y_pred_train),
                'test_accuracy': accuracy_score(y_test, y_pred_test),
                'predictions': y_pred_test,
                'classification_report': classification_report(y_test, y_pred_test),
                'feature_importance': dict(zip(self.feature_names, model.feature_importances_))
            }
            logger.info(f"Random Forest - Test Accuracy: {results['test_accuracy']:.4f}")
        
        return results
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task: str = 'regression'
    ) -> Optional[Dict]:
        """Train XGBoost model."""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available. Skipping.")
            return None
        
        logger.info(f"Training XGBoost ({task})...")
        
        if task == 'regression':
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        if task == 'regression':
            results = {
                'model_name': 'XGBoost (Regression)',
                'train_mse': mean_squared_error(y_train, y_pred_train),
                'test_mse': mean_squared_error(y_test, y_pred_test),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'predictions': y_pred_test,
                'feature_importance': dict(zip(self.feature_names, model.feature_importances_))
            }
            logger.info(f"XGBoost - Test R²: {results['test_r2']:.4f}, MAE: {results['test_mae']:.4f}")
        else:
            results = {
                'model_name': 'XGBoost (Classification)',
                'train_accuracy': accuracy_score(y_train, y_pred_train),
                'test_accuracy': accuracy_score(y_test, y_pred_test),
                'predictions': y_pred_test,
                'classification_report': classification_report(y_test, y_pred_test),
                'feature_importance': dict(zip(self.feature_names, model.feature_importances_))
            }
            logger.info(f"XGBoost - Test Accuracy: {results['test_accuracy']:.4f}")
        
        return results
    
    def train_all_baselines(
        self,
        df: pd.DataFrame,
        target_col: str = 'price_change_pct',
        task: str = 'regression'
    ) -> Dict:
        """
        Train all baseline models and compare.
        
        Args:
            df: DataFrame with features and targets
            target_col: Target column name
            task: 'regression' or 'classification'
        
        Returns:
            Dictionary with all model results
        """
        logger.info("=" * 60)
        logger.info("Training All Baseline Models")
        logger.info("=" * 60)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        
        all_results = {}
        
        # Train regression models
        if task == 'regression':
            all_results['linear'] = self.train_linear_regression(X_train, y_train, X_test, y_test)
            all_results['ridge'] = self.train_ridge_regression(X_train, y_train, X_test, y_test)
            all_results['random_forest'] = self.train_random_forest(X_train, y_train, X_test, y_test, task='regression')
            
            if XGBOOST_AVAILABLE:
                xgb_result = self.train_xgboost(X_train, y_train, X_test, y_test, task='regression')
                if xgb_result:
                    all_results['xgboost'] = xgb_result
        
        # Train classification models
        else:
            all_results['random_forest'] = self.train_random_forest(X_train, y_train, X_test, y_test, task='classification')
            
            if XGBOOST_AVAILABLE:
                xgb_result = self.train_xgboost(X_train, y_train, X_test, y_test, task='classification')
                if xgb_result:
                    all_results['xgboost'] = xgb_result
        
        self.results = all_results
        
        # Print summary
        self.print_summary(all_results, task)
        
        return all_results
    
    def print_summary(self, results: Dict, task: str = 'regression'):
        """Print summary of all model results."""
        logger.info("\n" + "=" * 60)
        logger.info("Model Comparison Summary")
        logger.info("=" * 60)
        
        if task == 'regression':
            logger.info(f"{'Model':<25} {'Test R²':<12} {'Test MAE':<12} {'Test MSE':<12}")
            logger.info("-" * 60)
            for name, result in results.items():
                logger.info(f"{result['model_name']:<25} {result['test_r2']:>11.4f} {result['test_mae']:>11.4f} {result['test_mse']:>11.4f}")
        else:
            logger.info(f"{'Model':<25} {'Test Accuracy':<15}")
            logger.info("-" * 40)
            for name, result in results.items():
                logger.info(f"{result['model_name']:<25} {result['test_accuracy']:>14.4f}")
        
        logger.info("=" * 60)
    
    def save_models(self, output_dir: str = "ml_system/models"):
        """Save trained models and scalers."""
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = os.path.join(output_dir, f"{name}_model.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Saved model: {model_path}")
        
        for name, scaler in self.scalers.items():
            scaler_path = os.path.join(output_dir, f"{name}_scaler.pkl")
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved scaler: {scaler_path}")
        
        # Save feature names
        feature_path = os.path.join(output_dir, "feature_names.pkl")
        joblib.dump(self.feature_names, feature_path)
        logger.info(f"Saved feature names: {feature_path}")
    
    def get_best_model(self, metric: str = 'test_r2') -> Tuple[str, Dict]:
        """
        Get the best model based on a metric.
        
        Args:
            metric: Metric to compare ('test_r2', 'test_mae', 'test_accuracy')
        
        Returns:
            Tuple of (model_name, results_dict)
        """
        if not self.results:
            return None, None
        
        # For regression, higher R² is better, lower MAE/MSE is better
        if metric == 'test_r2':
            best_name = max(self.results.keys(), key=lambda k: self.results[k].get(metric, -np.inf))
        else:
            best_name = min(self.results.keys(), key=lambda k: self.results[k].get(metric, np.inf))
        
        return best_name, self.results[best_name]


if __name__ == "__main__":
    # Example usage
    from ml_system.data.data_extractor import DataExtractor
    from ml_system.features.feature_engineer import FeatureEngineer
    
    extractor = DataExtractor()
    engineer = FeatureEngineer()
    trainer = BaselineTrainer()
    
    try:
        # Get and engineer data
        print("Loading and engineering data...")
        raw_data = extractor.get_time_series_data('NSE', lookback_days=30)
        
        if raw_data.empty:
            print("No data found!")
            exit(1)
        
        features_df = engineer.engineer_all_features(raw_data)
        
        if features_df.empty:
            print("Feature engineering produced empty dataset!")
            exit(1)
        
        # Train all baseline models
        print("\nTraining baseline models...")
        results = trainer.train_all_baselines(features_df, target_col='price_change_pct', task='regression')
        
        # Save models
        print("\nSaving models...")
        trainer.save_models()
        
        # Get best model
        best_name, best_result = trainer.get_best_model('test_r2')
        print(f"\nBest model: {best_name}")
        print(f"Test R²: {best_result['test_r2']:.4f}")
        print(f"Test MAE: {best_result['test_mae']:.4f}")
    
    finally:
        extractor.close()

