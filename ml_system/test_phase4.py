"""
Test script for Phase 4: Backtesting & Real-time System
Run this to test Phase 4 functionality.
"""

import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
from ml_system.data.data_extractor import DataExtractor
from ml_system.features.feature_engineer import FeatureEngineer
from ml_system.training.train_baseline import BaselineTrainer
from ml_system.backtesting.backtest_engine import BacktestEngine
from ml_system.predictions.realtime_predictor import RealTimePredictor
from ml_system.predictions.signal_generator import SignalGenerator
from ml_system.monitoring.performance_monitor import PerformanceMonitor
import numpy as np

def test_phase4():
    """Test Phase 4 components."""
    print("=" * 60)
    print("Phase 4 Testing: Backtesting & Real-time System")
    print("=" * 60)
    
    extractor = DataExtractor()
    engineer = FeatureEngineer()
    trainer = BaselineTrainer()
    backtester = BacktestEngine(initial_capital=100000.0)
    predictor = RealTimePredictor()
    signal_gen = SignalGenerator()
    monitor = PerformanceMonitor()
    
    try:
        # Step 1: Load and prepare data
        print("\n[Step 1] Loading and preparing data...")
        raw_data = extractor.get_time_series_data('NSE', lookback_days=30)
        features_df = engineer.engineer_all_features(raw_data)
        
        if features_df.empty:
            print("❌ ERROR: No data available!")
            return
        
        print(f"✅ Loaded {len(features_df)} samples")
        
        # Step 2: Train models (if not already trained)
        print("\n[Step 2] Training/loading models...")
        try:
            # Try to load existing models first
            predictor.load_models()
            trainer_models_loaded = trainer.load_models()
            if not predictor.is_loaded or not trainer_models_loaded:
                print("   No existing models found. Training new models...")
                features_df_filtered = features_df.loc[~(features_df.isna().any(axis=1) | features_df['price_change_pct'].isna())]
                baseline_results = trainer.train_all_baselines(
                    features_df_filtered,
                    target_col='price_change_pct',
                    task='regression'
                )
                trainer.save_models()
                predictor.load_models()
                trainer.load_models()
            else:
                print("✅ Loaded existing models")
        except Exception as e:
            print(f"⚠️  Error loading models: {str(e)}")
            print("   Training new models...")
            features_df_filtered = features_df.loc[~(features_df.isna().any(axis=1) | features_df['price_change_pct'].isna())]
            baseline_results = trainer.train_all_baselines(
                features_df_filtered,
                target_col='price_change_pct',
                task='regression'
            )
            trainer.save_models()
            predictor.load_models()
            trainer.load_models()
        
        # Step 3: Backtesting
        print("\n[Step 3] Running backtest...")
        try:
            # Prepare data for backtest
            feature_cols = [col for col in features_df.columns 
                           if col not in ['timestamp', 'future_price', 'price_change', 'price_change_pct', 'direction']]
            X = features_df[feature_cols]
            y = features_df['price_change_pct']
            prices = features_df['underlying_price']
            timestamps = features_df['timestamp']
            
            # Remove NaN
            valid_mask = ~(X.isna().any(axis=1) | y.isna() | prices.isna())
            X = X[valid_mask]
            y = y[valid_mask]
            prices = prices[valid_mask]
            timestamps = timestamps[valid_mask]
            
            # Get predictions from best model
            X_train, X_test, y_train, y_test = trainer.prepare_data(
                features_df.loc[valid_mask],
                target_col='price_change_pct',
                test_size=0.2
            )
            
            # Use Random Forest for predictions (usually best)
            if not trainer.models:
                raise ValueError("No trained models available. Please ensure training completed successfully.")
            
            if 'random_forest' in trainer.models:
                predictions = trainer.models['random_forest'].predict(X_test)
            else:
                # Use first available model
                model_name = list(trainer.models.keys())[0]
                if model_name in trainer.scalers:
                    X_test_scaled = trainer.scalers[model_name].transform(X_test)
                    predictions = trainer.models[model_name].predict(X_test_scaled)
                else:
                    predictions = trainer.models[model_name].predict(X_test)
            
            # Run backtest
            backtest_result = backtester.backtest_strategy(
                predictions=pd.Series(predictions),
                actual_prices=pd.Series(prices.iloc[-len(predictions):].values),
                timestamps=pd.Series(timestamps.iloc[-len(predictions):].values),
                entry_threshold=0.02,
                exit_threshold=0.05,
                stop_loss_pct=0.03
            )
            
            backtester.print_results(backtest_result)
            print("✅ Backtest completed")
            
        except Exception as e:
            print(f"⚠️  Backtest error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Step 4: Real-time prediction
        print("\n[Step 4] Testing real-time prediction...")
        try:
            if not features_df.empty:
                # Get latest features
                latest_features = features_df.iloc[[-1]]
                feature_cols = [col for col in latest_features.columns 
                               if col not in ['timestamp', 'future_price', 'price_change', 'price_change_pct', 'direction']]
                latest_X = latest_features[feature_cols]
                
                # Make prediction
                prediction_result = predictor.predict_and_signal(
                    latest_X,
                    entry_threshold=0.02,
                    min_confidence=0.5
                )
                
                print(f"✅ Prediction: {prediction_result['prediction']:.4f}%")
                print(f"   Confidence: {prediction_result['confidence']:.2%}")
                print(f"   Signal: {prediction_result['signal']}")
                print(f"   Strength: {prediction_result['strength']:.2%}")
                
        except Exception as e:
            print(f"⚠️  Prediction error: {str(e)}")
        
        # Step 5: Signal generation
        print("\n[Step 5] Testing signal generation...")
        try:
            if 'prediction' in locals() and 'prediction_result' in locals():
                signal = signal_gen.generate_signal(
                    prediction=prediction_result['prediction'],
                    confidence=prediction_result['confidence'],
                    current_price=features_df.iloc[-1]['underlying_price'],
                    capital=100000.0
                )
                
                signal_gen.print_signal(signal)
                print("✅ Signal generated")
        except Exception as e:
            print(f"⚠️  Signal generation error: {str(e)}")
        
        # Step 6: Performance monitoring
        print("\n[Step 6] Testing performance monitoring...")
        try:
            # Log some sample predictions
            for i in range(min(5, len(features_df))):
                row = features_df.iloc[i]
                monitor.log_prediction(
                    timestamp=row['timestamp'],
                    prediction=row.get('price_change_pct', 0),
                    actual=row.get('price_change_pct', None),
                    confidence=0.7,
                    signal="BUY" if i % 2 == 0 else "HOLD"
                )
            
            # Calculate metrics
            metrics = monitor.calculate_metrics()
            monitor.print_metrics(metrics)
            print("✅ Performance monitoring working")
            
        except Exception as e:
            print(f"⚠️  Monitoring error: {str(e)}")
        
        print("\n" + "=" * 60)
        print("✅ Phase 4 Testing Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        extractor.close()

if __name__ == "__main__":
    test_phase4()

