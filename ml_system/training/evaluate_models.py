"""
Phase 2: Model Evaluation
Provides comprehensive evaluation metrics and visualizations for trained models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates and visualizes model performance."""
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_results = {}
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> Dict:
        """
        Evaluate regression model performance.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Name of the model
        
        Returns:
            Dictionary with evaluation metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100  # Mean Absolute Percentage Error
        
        # Direction accuracy (if predicting price changes)
        if len(y_true) > 0:
            direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))
        else:
            direction_accuracy = 0.0
        
        results = {
            'model_name': model_name,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_accuracy
        }
        
        logger.info(f"{model_name} - R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, Direction Acc: {direction_accuracy:.4f}")
        
        return results
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> Dict:
        """
        Evaluate classification model performance.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Name of the model
        
        Returns:
            Dictionary with evaluation metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
    
    def plot_predictions_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ):
        """
        Plot predictions vs actual values.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Name of the model
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name} - Predictions vs Actual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² to plot
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {save_path}")
        
        plt.close()
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ):
        """
        Plot residuals (errors).
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Name of the model
            save_path: Path to save the plot
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'{model_name} - Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{model_name} - Residuals Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {save_path}")
        
        plt.close()
    
    def plot_feature_importance(
        self,
        feature_importance: Dict,
        model_name: str = "Model",
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance.
        
        Args:
            feature_importance: Dictionary of feature names and importance scores
            model_name: Name of the model
            top_n: Number of top features to show
            save_path: Path to save the plot
        """
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        features, importances = zip(*top_features)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title(f'{model_name} - Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {save_path}")
        
        plt.close()
    
    def compare_models(
        self,
        results: Dict,
        metric: str = 'test_r2',
        save_path: Optional[str] = None
    ):
        """
        Compare multiple models on a metric.
        
        Args:
            results: Dictionary of model results
            metric: Metric to compare
            save_path: Path to save the plot
        """
        model_names = []
        metric_values = []
        
        for name, result in results.items():
            if metric in result:
                model_names.append(result.get('model_name', name))
                metric_values.append(result[metric])
        
        if not model_names:
            logger.warning(f"Metric {metric} not found in results")
            return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(model_names)), metric_values)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {save_path}")
        
        plt.close()
    
    def generate_report(
        self,
        results: Dict,
        output_dir: str = "ml_system/training/reports"
    ):
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Dictionary of model results
            output_dir: Directory to save reports
        """
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Generating evaluation report...")
        
        # Create comparison DataFrame
        comparison_data = []
        for name, result in results.items():
            row = {'Model': result.get('model_name', name)}
            for key, value in result.items():
                if key not in ['model_name', 'predictions', 'feature_importance', 'classification_report']:
                    if isinstance(value, (int, float)):
                        row[key] = value
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        comparison_path = os.path.join(output_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Saved comparison table: {comparison_path}")
        
        # Generate plots
        if 'test_r2' in comparison_df.columns:
            self.compare_models(results, 'test_r2', 
                              os.path.join(output_dir, "model_comparison_r2.png"))
        
        if 'test_mae' in comparison_df.columns:
            self.compare_models(results, 'test_mae',
                              os.path.join(output_dir, "model_comparison_mae.png"))
        
        logger.info(f"Evaluation report generated in: {output_dir}")


if __name__ == "__main__":
    # Example usage
    from ml_system.training.train_baseline import BaselineTrainer
    from ml_system.data.data_extractor import DataExtractor
    from ml_system.features.feature_engineer import FeatureEngineer
    
    extractor = DataExtractor()
    engineer = FeatureEngineer()
    trainer = BaselineTrainer()
    evaluator = ModelEvaluator()
    
    try:
        # Get and engineer data
        raw_data = extractor.get_time_series_data('NSE', lookback_days=30)
        features_df = engineer.engineer_all_features(raw_data)
        
        # Train models
        results = trainer.train_all_baselines(features_df, target_col='price_change_pct', task='regression')
        
        # Generate evaluation report
        evaluator.generate_report(results)
        
    finally:
        extractor.close()

