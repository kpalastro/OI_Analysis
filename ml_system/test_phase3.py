"""
Test script for Phase 3: Advanced Models
Run this to test Phase 3 functionality.
"""

import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from ml_system.data.data_extractor import DataExtractor
from ml_system.features.feature_engineer import FeatureEngineer
from ml_system.training.train_baseline import BaselineTrainer
from ml_system.training.train_advanced import AdvancedTrainer
import numpy as np

def test_phase3():
    """Test Phase 3 components."""
    print("=" * 60)
    print("Phase 3 Testing: Advanced Models (LSTM & Ensemble)")
    print("=" * 60)
    
    extractor = DataExtractor()
    engineer = FeatureEngineer()
    baseline_trainer = BaselineTrainer()
    advanced_trainer = AdvancedTrainer()
    
    try:
        # Step 1: Load and engineer data
        print("\n[Step 1] Loading and engineering data...")
        raw_data = extractor.get_time_series_data('NSE', lookback_days=30)
        
        if raw_data.empty:
            print("❌ ERROR: No data found!")
            return
        
        print(f"✅ Loaded {len(raw_data)} raw records")
        
        features_df = engineer.engineer_all_features(raw_data)
        
        if features_df.empty:
            print("❌ ERROR: Feature engineering produced empty dataset!")
            return
        
        print(f"✅ Engineered {len(features_df)} samples with {len(engineer.get_feature_names())} features")
        
        # Step 2: Prepare data
        print("\n[Step 2] Preparing data for training...")
        feature_cols = [col for col in features_df.columns 
                       if col not in ['timestamp', 'future_price', 'price_change', 'price_change_pct', 'direction']]
        X = features_df[feature_cols]
        y = features_df['price_change_pct']
        
        # Remove NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Time-based split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"✅ Train set: {len(X_train)} samples")
        print(f"✅ Test set: {len(X_test)} samples")
        print(f"   Target range: [{y_train.min():.4f}, {y_train.max():.4f}]")
        
        # Step 3: Train baseline models
        print("\n[Step 3] Training baseline models...")
        baseline_results = baseline_trainer.train_all_baselines(
            features_df.iloc[valid_mask],
            target_col='price_change_pct',
            task='regression'
        )
        print(f"✅ Trained {len(baseline_results)} baseline models")
        
        # Step 4: Train LSTM
        print("\n[Step 4] Training LSTM model...")
        try:
            lstm_result = advanced_trainer.train_lstm(
                X_train, y_train, X_test, y_test,
                sequence_length=30,
                epochs=20,  # Reduced for testing
                batch_size=32
            )
            
            if lstm_result:
                print(f"✅ LSTM trained successfully")
                print(f"   Test R²: {lstm_result['test_r2']:.4f}")
                print(f"   Test MAE: {lstm_result['test_mae']:.4f}")
                print(f"   Direction Accuracy: {lstm_result['direction_accuracy']:.4f}")
            else:
                print("⚠️  LSTM training skipped (TensorFlow not available or error)")
        except Exception as e:
            print(f"⚠️  LSTM training failed: {str(e)}")
            lstm_result = None
        
        # Step 5: Create ensemble
        print("\n[Step 5] Creating ensemble...")
        try:
            ensemble_result = advanced_trainer.create_ensemble(
                baseline_results,
                lstm_result,
                method='weighted_average'
            )
            
            if ensemble_result:
                print(f"✅ Ensemble created successfully")
                print(f"   Model weights:")
                for name, weight in ensemble_result.get('weights', {}).items():
                    print(f"      {name}: {weight:.4f}")
            else:
                print("⚠️  Ensemble creation skipped")
        except Exception as e:
            print(f"⚠️  Ensemble creation failed: {str(e)}")
        
        # Step 6: Save models
        print("\n[Step 6] Saving models...")
        try:
            advanced_trainer.save_models()
            print("✅ Models saved successfully")
        except Exception as e:
            print(f"⚠️  Error saving models: {str(e)}")
        
        print("\n" + "=" * 60)
        print("✅ Phase 3 Testing Complete!")
        print("=" * 60)
        
        # Summary
        print("\nModel Performance Summary:")
        print("-" * 60)
        for name, result in baseline_results.items():
            if 'test_r2' in result:
                print(f"{result['model_name']:<30} R²: {result['test_r2']:>7.4f}  MAE: {result['test_mae']:>7.4f}")
        
        if lstm_result:
            print(f"{lstm_result['model_name']:<30} R²: {lstm_result['test_r2']:>7.4f}  MAE: {lstm_result['test_mae']:>7.4f}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        extractor.close()

if __name__ == "__main__":
    test_phase3()

