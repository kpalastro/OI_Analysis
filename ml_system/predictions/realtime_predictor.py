"""
Phase 4: Real-time Prediction Service
Provides live predictions using trained models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimePredictor:
    """Real-time prediction service for live trading."""
    
    def __init__(self, model_dir: str = "ml_system/models"):
        """
        Initialize real-time predictor.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.model_weights = {}
        self.is_loaded = False
    
    def load_models(self, model_names: Optional[List[str]] = None):
        """
        Load trained models.
        
        Args:
            model_names: List of model names to load (None = load all available)
        """
        import glob
        
        if model_names is None:
            # Auto-detect available models
            model_files = glob.glob(os.path.join(self.model_dir, "*_model.pkl"))
            model_names = [os.path.basename(f).replace("_model.pkl", "") for f in model_files]
        
        logger.info(f"Loading models: {model_names}")
        
        for name in model_names:
            model_path = os.path.join(self.model_dir, f"{name}_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{name}_scaler.pkl")
            
            if os.path.exists(model_path):
                try:
                    self.models[name] = joblib.load(model_path)
                    logger.info(f"Loaded model: {name}")
                    
                    if os.path.exists(scaler_path):
                        self.scalers[name] = joblib.load(scaler_path)
                        logger.info(f"Loaded scaler: {name}")
                except Exception as e:
                    logger.error(f"Error loading {name}: {e}")
        
        # Load feature names
        feature_path = os.path.join(self.model_dir, "feature_names.pkl")
        if os.path.exists(feature_path):
            self.feature_names = joblib.load(feature_path)
            logger.info(f"Loaded {len(self.feature_names)} feature names")
        
        # Load ensemble weights if available
        ensemble_path = os.path.join(self.model_dir, "ensemble_config.pkl")
        if os.path.exists(ensemble_path):
            ensemble_data = joblib.load(ensemble_path)
            self.model_weights = ensemble_data.get('weights', {})
            logger.info(f"Loaded ensemble weights: {self.model_weights}")
        
        self.is_loaded = True
        logger.info(f"Successfully loaded {len(self.models)} models")
    
    def predict(
        self,
        features: pd.DataFrame,
        use_ensemble: bool = True,
        return_confidence: bool = False
    ) -> Dict:
        """
        Make predictions on new data.
        
        Args:
            features: DataFrame with feature columns
            use_ensemble: If True, use ensemble prediction
            return_confidence: If True, return confidence scores
        
        Returns:
            Dictionary with predictions and metadata
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Ensure features match expected feature names
        missing_features = set(self.feature_names) - set(features.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Fill missing features with 0
            for feat in missing_features:
                features[feat] = 0
        
        # Select only required features
        X = features[self.feature_names].copy()
        
        # Get predictions from each model
        individual_predictions = {}
        
        for name, model in self.models.items():
            try:
                # Scale features if scaler available
                if name in self.scalers:
                    X_scaled = self.scalers[name].transform(X)
                else:
                    X_scaled = X
                
                if hasattr(model, "feature_names_in_"):
                    if isinstance(X_scaled, np.ndarray):
                        X_input = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
                    else:
                        X_input = X_scaled[self.feature_names]
                else:
                    if isinstance(X_scaled, pd.DataFrame):
                        X_input = X_scaled.values
                    else:
                        X_input = X_scaled
                
                # Predict
                pred = model.predict(X_input)
                if isinstance(pred, np.ndarray) and pred.ndim > 1:
                    pred = pred.flatten()
                
                individual_predictions[name] = pred[0] if len(pred) > 0 else 0.0
                
            except Exception as e:
                logger.error(f"Error predicting with {name}: {e}")
                continue
        
        if not individual_predictions:
            raise ValueError("No valid predictions obtained from any model.")
        
        # Ensemble prediction
        if use_ensemble and self.model_weights:
            ensemble_pred = 0.0
            total_weight = 0.0
            
            for name, pred in individual_predictions.items():
                weight = self.model_weights.get(name, 1.0 / len(individual_predictions))
                ensemble_pred += pred * weight
                total_weight += weight
            
            final_prediction = ensemble_pred / total_weight if total_weight > 0 else ensemble_pred
        else:
            # Use best model or average
            if individual_predictions:
                final_prediction = np.mean(list(individual_predictions.values()))
            else:
                final_prediction = 0.0
        
        # Calculate confidence (variance of predictions)
        if return_confidence and len(individual_predictions) > 1:
            pred_values = list(individual_predictions.values())
            confidence = 1.0 / (1.0 + np.std(pred_values))  # Higher std = lower confidence
        else:
            confidence = 1.0
        
        result = {
            'prediction': final_prediction,
            'confidence': confidence,
            'individual_predictions': individual_predictions,
            'timestamp': datetime.now(),
            'model_count': len(individual_predictions)
        }
        
        return result
    
    def generate_signal(
        self,
        prediction: float,
        confidence: float,
        entry_threshold: float = 0.02,
        min_confidence: float = 0.5
    ) -> Dict:
        """
        Generate trading signal from prediction.
        
        Args:
            prediction: Predicted price change (percentage)
            confidence: Confidence score (0-1)
            entry_threshold: Minimum predicted change to generate signal
            min_confidence: Minimum confidence to generate signal
        
        Returns:
            Dictionary with signal information
        """
        signal = "HOLD"
        signal_strength = 0.0
        
        if confidence < min_confidence:
            return {
                'signal': 'HOLD',
                'reason': 'Low confidence',
                'strength': 0.0,
                'prediction': prediction,
                'confidence': confidence
            }
        
        if prediction > entry_threshold:
            signal = "BUY"
            signal_strength = min(1.0, (prediction / entry_threshold) * confidence)
        elif prediction < -entry_threshold:
            signal = "SELL"
            signal_strength = min(1.0, (abs(prediction) / entry_threshold) * confidence)
        
        return {
            'signal': signal,
            'reason': f'Predicted change: {prediction:.4f}%',
            'strength': signal_strength,
            'prediction': prediction,
            'confidence': confidence,
            'entry_threshold': entry_threshold
        }
    
    def predict_and_signal(
        self,
        features: pd.DataFrame,
        entry_threshold: float = 0.02,
        min_confidence: float = 0.5
    ) -> Dict:
        """
        Make prediction and generate signal in one call.
        
        Args:
            features: Feature DataFrame
            entry_threshold: Entry threshold for signals
            min_confidence: Minimum confidence for signals
        
        Returns:
            Combined prediction and signal dictionary
        """
        prediction_result = self.predict(features, return_confidence=True)
        signal_result = self.generate_signal(
            prediction_result['prediction'],
            prediction_result['confidence'],
            entry_threshold,
            min_confidence
        )
        
        return {
            **prediction_result,
            **signal_result
        }


if __name__ == "__main__":
    # Example usage
    from ml_system.data.data_extractor import DataExtractor
    from ml_system.features.feature_engineer import FeatureEngineer
    
    extractor = DataExtractor()
    engineer = FeatureEngineer()
    predictor = RealTimePredictor()
    
    try:
        # Load models
        predictor.load_models()
        
        # Get latest data
        raw_data = extractor.get_time_series_data('NSE', lookback_days=1)
        features_df = engineer.engineer_all_features(raw_data)
        
        if not features_df.empty:
            # Get latest row
            latest_features = features_df.iloc[[-1]]
            
            # Make prediction
            result = predictor.predict_and_signal(latest_features)
            
            print(f"\nPrediction: {result['prediction']:.4f}%")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Signal: {result['signal']}")
            print(f"Strength: {result['strength']:.4f}")
    
    finally:
        extractor.close()

