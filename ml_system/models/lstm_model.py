"""
Phase 3: LSTM Model for Time Series Prediction
Deep learning model using LSTM to capture temporal patterns in OI data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
import logging
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
    # Create dummy types for type hints when TensorFlow is not available
    keras_Model = keras.Model
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Create dummy classes for type hints
    class keras_Model:
        pass
    keras = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMPredictor:
    """LSTM model for time series prediction of price movements."""
    
    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = None,
        hidden_units: List[int] = [128, 64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Number of time steps to look back
            n_features: Number of features (will be set during training)
            hidden_units: List of hidden units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model. Install with: pip install tensorflow")
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = None
        self.feature_names = []
    
    def create_sequences(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Feature data (n_samples, n_features)
            targets: Target values (n_samples,)
            sequence_length: Length of sequences
        
        Returns:
            X: Sequences (n_sequences, sequence_length, n_features)
            y: Targets (n_sequences,)
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(targets[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> keras_Model:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: (sequence_length, n_features)
        
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.hidden_units[0],
            return_sequences=len(self.hidden_units) > 1,
            input_shape=input_shape
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for units in self.hidden_units[1:-1]:
            model.add(LSTM(units=units, return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Last LSTM layer (if multiple layers)
        if len(self.hidden_units) > 1:
            model.add(LSTM(units=self.hidden_units[-1], return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Dense output layer
        model.add(Dense(units=32, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(units=1))  # Regression output
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1
    ) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split if no validation set provided
            verbose: Verbosity level
        
        Returns:
            Training history dictionary
        """
        from sklearn.preprocessing import StandardScaler
        
        logger.info("Preparing data for LSTM training...")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        self.n_features = len(self.feature_names)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(
            X_train_scaled,
            y_train.values,
            self.sequence_length
        )
        
        logger.info(f"Created {len(X_seq)} sequences of length {self.sequence_length}")
        
        # Build model
        input_shape = (self.sequence_length, self.n_features)
        self.model = self.build_model(input_shape)
        
        logger.info("LSTM Model Architecture:")
        self.model.summary(print_fn=logger.info)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self.create_sequences(
                X_val_scaled,
                y_val.values,
                self.sequence_length
            )
            validation_data = (X_val_seq, y_val_seq)
        elif validation_split > 0:
            # Split sequences for validation
            split_idx = int(len(X_seq) * (1 - validation_split))
            X_val_seq = X_seq[split_idx:]
            y_val_seq = y_seq[split_idx:]
            X_seq = X_seq[:split_idx]
            y_seq = y_seq[:split_idx]
            validation_data = (X_val_seq, y_val_seq)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        logger.info("Training LSTM model...")
        history = self.model.fit(
            X_seq, y_seq,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        logger.info("LSTM training complete!")
        
        return {
            'history': history.history,
            'n_sequences': len(X_seq),
            'n_features': self.n_features
        }
    
    def predict(
        self,
        X: pd.DataFrame,
        return_sequences: bool = False
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature data
            return_sequences: If True, return predictions for all sequences
        
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq, _ = self.create_sequences(
            X_scaled,
            np.zeros(len(X_scaled)),  # Dummy targets
            self.sequence_length
        )
        
        # Predict
        predictions = self.model.predict(X_seq, verbose=0)
        
        if return_sequences:
            return predictions
        else:
            # Return predictions aligned with original data
            # Pad with NaN for first sequence_length samples
            padded = np.full(len(X), np.nan)
            padded[self.sequence_length:] = predictions.flatten()
            return padded
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X: Feature data
            y: True targets
        
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        predictions = self.predict(X)
        
        # Remove NaN predictions (from padding)
        valid_mask = ~np.isnan(predictions)
        y_true = y.values[valid_mask]
        y_pred = predictions[valid_mask]
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Direction accuracy
        direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'n_samples': len(y_true)
        }
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Use .keras format (newer format) instead of .h5 (legacy)
        if filepath.endswith('.h5'):
            filepath = filepath.replace('.h5', '.keras')
        
        # Save model in native Keras format
        self.model.save(filepath)
        
        # Save scaler and metadata
        import joblib
        scaler_path = filepath.replace('.keras', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        metadata_path = filepath.replace('.keras', '_metadata.pkl')
        joblib.dump({
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'hidden_units': self.hidden_units
        }, metadata_path)
        
        logger.info(f"Saved model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        import joblib
        
        # Support both .keras and .h5 formats for backward compatibility
        if not os.path.exists(filepath):
            # Try .keras if .h5 not found
            if filepath.endswith('.h5'):
                keras_path = filepath.replace('.h5', '.keras')
                if os.path.exists(keras_path):
                    filepath = keras_path
        
        self.model = keras.models.load_model(filepath)
        
        # Load scaler and metadata (try both extensions)
        base_path = filepath.replace('.keras', '').replace('.h5', '')
        scaler_path = base_path + '_scaler.pkl'
        metadata_path = base_path + '_metadata.pkl'
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.sequence_length = metadata['sequence_length']
            self.n_features = metadata['n_features']
            self.feature_names = metadata['feature_names']
            self.hidden_units = metadata['hidden_units']
        
        logger.info(f"Loaded model from {filepath}")


if __name__ == "__main__":
    # Example usage
    from ml_system.data.data_extractor import DataExtractor
    from ml_system.features.feature_engineer import FeatureEngineer
    from sklearn.model_selection import train_test_split
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Cannot run LSTM example.")
        exit(1)
    
    extractor = DataExtractor()
    engineer = FeatureEngineer()
    
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
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False  # Don't shuffle time series
        )
        
        # Train LSTM
        lstm = LSTMPredictor(sequence_length=30, hidden_units=[64, 32])
        history = lstm.train(X_train, y_train, epochs=50, batch_size=32)
        
        # Evaluate
        results = lstm.evaluate(X_test, y_test)
        print(f"\nLSTM Results:")
        print(f"R²: {results['r2']:.4f}")
        print(f"MAE: {results['mae']:.4f}")
        print(f"Direction Accuracy: {results['direction_accuracy']:.4f}")
        
        # Save model
        lstm.save_model("ml_system/models/lstm_model.keras")
    
    finally:
        extractor.close()

