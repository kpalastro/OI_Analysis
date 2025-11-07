"""
Test script for Phase 2: Baseline Models
Run this to test Phase 2 functionality.
"""

import sys
import os

# Add parent directory to path so we can import ml_system
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from ml_system.data.data_extractor import DataExtractor
from ml_system.features.feature_engineer import FeatureEngineer
from ml_system.training.train_baseline import BaselineTrainer
from ml_system.training.evaluate_models import ModelEvaluator

def test_phase2():
    """Test Phase 2 components."""
    print("=" * 60)
    print("Phase 2 Testing: Baseline Model Training")
    print("=" * 60)
    
    extractor = DataExtractor()
    engineer = FeatureEngineer()
    trainer = BaselineTrainer()
    evaluator = ModelEvaluator()
    
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
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            features_df, 
            target_col='price_change_pct',
            test_size=0.2
        )
        
        print(f"✅ Train set: {len(X_train)} samples")
        print(f"✅ Test set: {len(X_test)} samples")
        print(f"   Target range: [{y_train.min():.4f}, {y_train.max():.4f}]")
        
        # Step 3: Train baseline models
        print("\n[Step 3] Training baseline models...")
        results = trainer.train_all_baselines(
            features_df,
            target_col='price_change_pct',
            task='regression'
        )
        
        print(f"✅ Trained {len(results)} models")
        
        # Step 4: Evaluate models
        print("\n[Step 4] Evaluating models...")
        for name, result in results.items():
            if 'predictions' in result:
                y_test_actual = y_test.values
                y_test_pred = result['predictions']
                
                eval_result = evaluator.evaluate_regression(
                    y_test_actual,
                    y_test_pred,
                    result['model_name']
                )
                print(f"   {result['model_name']}: R² = {eval_result['r2']:.4f}, MAE = {eval_result['mae']:.4f}")
        
        # Step 5: Get best model
        print("\n[Step 5] Best Model Analysis:")
        best_name, best_result = trainer.get_best_model('test_r2')
        if best_name:
            print(f"   Best Model: {best_result['model_name']}")
            print(f"   Test R²: {best_result['test_r2']:.4f}")
            print(f"   Test MAE: {best_result['test_mae']:.4f}")
            print(f"   Test RMSE: {np.sqrt(best_result['test_mse']):.4f}")
        
        # Step 6: Feature importance (if available)
        print("\n[Step 6] Feature Importance Analysis:")
        for name, result in results.items():
            if 'feature_importance' in result:
                importance = result['feature_importance']
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"   {result['model_name']} - Top 5 Features:")
                for feat, imp in top_features:
                    print(f"      {feat}: {imp:.4f}")
        
        # Step 7: Save models
        print("\n[Step 7] Saving models...")
        trainer.save_models()
        print("✅ Models saved successfully")
        
        # Step 8: Generate evaluation report
        print("\n[Step 8] Generating evaluation report...")
        evaluator.generate_report(results)
        print("✅ Evaluation report generated")
        
        print("\n" + "=" * 60)
        print("✅ Phase 2 Testing Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        extractor.close()

if __name__ == "__main__":
    import numpy as np
    test_phase2()

