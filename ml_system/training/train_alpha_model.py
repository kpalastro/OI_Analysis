"""
Alpha Model Training with Walk-Forward Validation

Trains classification models (XGBoost/LightGBM) on option chain features
using rigorous walk-forward validation to prevent data leakage.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from ml_system.data.feature_pipeline import FeaturePipeline
from ml_system.training.walk_forward_validator import WalkForwardValidator, validate_walk_forward_split

warnings.filterwarnings('ignore')


class AlphaModelTrainer:
    """
    Trains alpha models using walk-forward validation.
    """
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        output_dir: Path = Path('ml_system/models/alpha'),
        random_state: int = 42
    ):
        """
        Initialize trainer.
        
        Args:
            model_type: 'xgboost' or 'lightgbm'
            output_dir: Directory to save models and reports
            random_state: Random seed
        """
        if model_type == 'xgboost' and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        if model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")
        
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = []
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'future_direction_label'
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features and target for training.
        
        Args:
            df: Feature DataFrame
            target_col: Name of target column
            
        Returns:
            X (features), y (target), feature_names
        """
        # Exclude metadata and target columns
        exclude_cols = [
            'timestamp',
            'future_price',
            'price_change',
            'future_return_pct',
            target_col
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Remove rows with missing target
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Remove rows with any NaN features
        feature_mask = ~X.isna().any(axis=1)
        X = X[feature_mask]
        y = y[feature_mask]
        
        return X, y, feature_cols
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> xgb.XGBClassifier:
        """Train XGBoost classifier."""
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'eval_metric': 'mlogloss'
        }
        
        model = xgb.XGBClassifier(**params)
        
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
        
        return model
    
    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> lgb.LGBMClassifier:
        """Train LightGBM classifier."""
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'verbosity': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_names=['validation'],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )
        else:
            model.fit(X_train, y_train)
        
        return model
    
    def train_with_walk_forward(
        self,
        df: pd.DataFrame,
        initial_train_days: int = 30,
        test_days: int = 7,
        step_days: int = 7
    ) -> Dict:
        """
        Train model using walk-forward validation.
        
        Args:
            df: Feature DataFrame with timestamp column
            initial_train_days: Days for initial training fold
            test_days: Days for each test fold
            step_days: Days to step forward between folds
            
        Returns:
            Dictionary with training results and metrics
        """
        print("=" * 60)
        print("Alpha Model Training with Walk-Forward Validation")
        print("=" * 60)
        
        # Prepare features
        X, y, feature_names = self.prepare_features(df)
        
        # Create a DataFrame with features and timestamps for splitting
        feature_df = df.loc[X.index].copy()
        
        print(f"\nPrepared {len(X)} samples with {len(feature_names)} features")
        print(f"Target distribution:\n{y.value_counts().sort_index()}")
        
        # Create validator
        validator = WalkForwardValidator(
            initial_train_days=initial_train_days,
            test_days=test_days,
            step_days=step_days
        )
        
        # Get split summary
        summary = validator.get_split_summary(feature_df)
        print(f"\nWalk-forward validation: {len(summary)} folds")
        if len(summary) > 0:
            print(summary[['fold', 'train_samples', 'test_samples', 'train_days', 'test_days']].to_string(index=False))
        else:
            print("⚠️  No valid folds generated. Check data availability and split parameters.")
            return {}
        
        # Train on each fold
        fold_results = []
        all_predictions = []
        all_actuals = []
        
        for split in validator.split(feature_df):
            print(f"\n{'=' * 60}")
            print(f"Fold {split.fold_number}")
            print(f"Train: {split.train_start.date()} to {split.train_end.date()} ({len(split.train_indices)} samples)")
            print(f"Test:  {split.test_start.date()} to {split.test_end.date()} ({len(split.test_indices)} samples)")
            
            # Get train/test data
            train_df = feature_df.loc[split.train_indices]
            test_df = feature_df.loc[split.test_indices]
            
            # Validate no leakage
            if not validate_walk_forward_split(train_df, test_df):
                print("⚠️  WARNING: Data leakage detected! Skipping fold.")
                continue
            
            # Get feature matrices (use original indices from split)
            X_train = X.loc[split.train_indices]
            y_train = y.loc[split.train_indices]
            X_test = X.loc[split.test_indices]
            y_test = y.loc[split.test_indices]
            
            # Scale features (fit on train, transform test - no leakage)
            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train)
            X_test_scaled = fold_scaler.transform(X_test)
            
            # Train model
            print(f"Training {self.model_type}...")
            if self.model_type == 'xgboost':
                model = self.train_xgboost(
                    pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index),
                    y_train,
                    pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index),
                    y_test
                )
            else:  # lightgbm
                model = self.train_lightgbm(
                    pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index),
                    y_train,
                    pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index),
                    y_test
                )
            
            # Store scaler from best fold (will be updated later)
            if split.fold_number == 0:
                self.scaler = fold_scaler
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, y_pred, average=None, zero_division=0
            )
            
            # Store results
            fold_result = {
                'fold': split.fold_number,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'accuracy': accuracy,
                'precision_bearish': precision[0] if len(precision) > 0 else 0,
                'precision_neutral': precision[1] if len(precision) > 1 else 0,
                'precision_bullish': precision[2] if len(precision) > 2 else 0,
                'recall_bearish': recall[0] if len(recall) > 0 else 0,
                'recall_neutral': recall[1] if len(recall) > 1 else 0,
                'recall_bullish': recall[2] if len(recall) > 2 else 0,
                'f1_bearish': f1[0] if len(f1) > 0 else 0,
                'f1_neutral': f1[1] if len(f1) > 1 else 0,
                'f1_bullish': f1[2] if len(f1) > 2 else 0,
            }
            fold_results.append(fold_result)
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: Bearish={precision[0]:.3f}, Neutral={precision[1]:.3f}, Bullish={precision[2]:.3f}")
            print(f"  Recall:    Bearish={recall[0]:.3f}, Neutral={recall[1]:.3f}, Bullish={recall[2]:.3f}")
            
            # Store for overall metrics
            all_predictions.extend(y_pred)
            all_actuals.extend(y_test.values)
            
            # Save fold model
            self.models[f'fold_{split.fold_number}'] = model
        
        # Overall metrics
        print(f"\n{'=' * 60}")
        print("Overall Results (All Folds Combined)")
        print("=" * 60)
        
        overall_accuracy = accuracy_score(all_actuals, all_predictions)
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            all_actuals, all_predictions, average=None, zero_division=0
        )
        
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        print(f"\nPer-Class Metrics:")
        print(f"  Bearish: Precision={overall_precision[0]:.3f}, Recall={overall_recall[0]:.3f}, F1={overall_f1[0]:.3f}")
        print(f"  Neutral: Precision={overall_precision[1]:.3f}, Recall={overall_recall[1]:.3f}, F1={overall_f1[1]:.3f}")
        print(f"  Bullish: Precision={overall_precision[2]:.3f}, Recall={overall_recall[2]:.3f}, F1={overall_f1[2]:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(all_actuals, all_predictions)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Bearish Neutral Bullish")
        print(f"Actual Bearish   {cm[0,0]:5d}   {cm[0,1]:5d}   {cm[0,2]:5d}")
        print(f"       Neutral    {cm[1,0]:5d}   {cm[1,1]:5d}   {cm[1,2]:5d}")
        print(f"       Bullish    {cm[2,0]:5d}   {cm[2,1]:5d}   {cm[2,2]:5d}")
        
        # Save best model (use last fold or best performing)
        best_fold = max(fold_results, key=lambda x: x['accuracy'])
        best_model = self.models[f"fold_{best_fold['fold']}"]
        
        # Feature importance analysis
        print(f"\n{'=' * 60}")
        print("Feature Importance (Top 20)")
        print("=" * 60)
        
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(importance_df.head(20).to_string(index=False))
            
            # Save feature importance
            importance_path = self.output_dir / 'feature_importance.csv'
            importance_df.to_csv(importance_path, index=False)
            print(f"\n✓ Saved feature importance to {importance_path}")
        else:
            print("Feature importance not available for this model type")
        
        # Save results
        results_df = pd.DataFrame(fold_results)
        results_path = self.output_dir / 'walk_forward_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\n✓ Saved fold results to {results_path}")
        
        return {
            'fold_results': fold_results,
            'overall_accuracy': overall_accuracy,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'confusion_matrix': cm,
            'best_fold': best_fold,
            'feature_names': feature_names
        }
    
    def save_model(
        self,
        model,
        scaler,
        feature_names: List[str],
        metadata: Dict
    ):
        """Save trained model with scaler and metadata."""
        model_path = self.output_dir / f'alpha_model_{self.model_type}.pkl'
        scaler_path = self.output_dir / f'alpha_scaler_{self.model_type}.pkl'
        feature_path = self.output_dir / 'alpha_feature_names.pkl'
        metadata_path = self.output_dir / 'alpha_model_metadata.pkl'
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(feature_names, feature_path)
        joblib.dump(metadata, metadata_path)
        
        print(f"\n✓ Saved model to {model_path}")
        print(f"✓ Saved scaler to {scaler_path}")
        print(f"✓ Saved feature names to {feature_path}")
        print(f"✓ Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train alpha model with walk-forward validation"
    )
    parser.add_argument(
        '--exchange',
        choices=['NSE', 'BSE'],
        default='NSE',
        help='Exchange to train on (default: NSE)'
    )
    parser.add_argument(
        '--model-type',
        choices=['xgboost', 'lightgbm'],
        default='xgboost',
        help='Model type (default: xgboost)'
    )
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=30,
        help='Days of historical data to use (default: 30)'
    )
    parser.add_argument(
        '--initial-train-days',
        type=int,
        default=30,
        help='Days for initial training fold (default: 30)'
    )
    parser.add_argument(
        '--test-days',
        type=int,
        default=7,
        help='Days for each test fold (default: 7)'
    )
    parser.add_argument(
        '--step-days',
        type=int,
        default=7,
        help='Days to step forward between folds (default: 7)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='ml_system/models/alpha',
        help='Output directory for models (default: ml_system/models/alpha)'
    )
    
    args = parser.parse_args()
    
    # Build feature set
    print("Building feature set...")
    from ml_system.data.feature_pipeline import FeaturePipelineConfig
    config = FeaturePipelineConfig(lookback_days=args.lookback_days)
    pipeline = FeaturePipeline(config)
    try:
        df = pipeline.build_feature_set(args.exchange)
        
        if df.empty:
            print("❌ No data available for training")
            return
        
        # Train model
        trainer = AlphaModelTrainer(
            model_type=args.model_type,
            output_dir=Path(args.output_dir)
        )
        
        results = trainer.train_with_walk_forward(
            df,
            initial_train_days=args.initial_train_days,
            test_days=args.test_days,
            step_days=args.step_days
        )
        
        if not results or 'best_fold' not in results:
            print("\n❌ Training failed - no valid folds generated")
            return
        
        # Save best model
        best_fold_num = results['best_fold']['fold']
        best_model = trainer.models[f'fold_{best_fold_num}']
        
        trainer.save_model(
            best_model,
            trainer.scaler,
            results['feature_names'],
            {
                'model_type': args.model_type,
                'exchange': args.exchange,
                'overall_accuracy': results['overall_accuracy'],
                'best_fold': best_fold_num,
                'feature_count': len(results['feature_names'])
            }
        )
        
        print("\n✅ Training completed successfully!")
        
    finally:
        pipeline.extractor.close()


if __name__ == "__main__":
    main()

