"""
Test script for Phase 1: Data Extraction and Feature Engineering
Run this to verify Phase 1 is working correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_system.data.data_extractor import DataExtractor
from ml_system.features.feature_engineer import FeatureEngineer
import pandas as pd

def test_phase1():
    """Test Phase 1 components."""
    print("=" * 60)
    print("Phase 1 Testing: Data Extraction & Feature Engineering")
    print("=" * 60)
    
    extractor = DataExtractor()
    engineer = FeatureEngineer()
    
    try:
        # Test 1: Data Extraction
        print("\n[Test 1] Extracting NSE data...")
        nse_data = extractor.get_time_series_data('NSE', lookback_days=7)
        
        if nse_data.empty:
            print("⚠️  WARNING: No NSE data found. Check your database.")
            return
        
        print(f"✅ Extracted {len(nse_data)} records")
        print(f"   Columns: {list(nse_data.columns)[:10]}...")
        
        # Test 2: Feature Engineering
        print("\n[Test 2] Engineering features...")
        features_df = engineer.engineer_all_features(nse_data.copy())
        
        if features_df.empty:
            print("⚠️  WARNING: Feature engineering produced empty dataset.")
            return
        
        print(f"✅ Created {len(features_df)} samples with {len(features_df.columns)} columns")
        
        # Test 3: Feature Summary
        print("\n[Test 3] Feature Summary:")
        feature_names = engineer.get_feature_names()
        print(f"   Total features: {len(feature_names)}")
        print(f"   Sample features: {feature_names[:10]}")
        
        # Test 4: Data Quality Check
        print("\n[Test 4] Data Quality Check:")
        print(f"   Missing values: {features_df.isnull().sum().sum()}")
        print(f"   Infinite values: {pd.isinf(features_df.select_dtypes(include=[float, int])).sum().sum()}")
        
        # Test 5: Target Distribution
        if 'direction' in features_df.columns:
            print("\n[Test 5] Target Distribution:")
            direction_counts = features_df['direction'].value_counts()
            print(f"   Up (1): {direction_counts.get(1, 0)}")
            print(f"   Down (-1): {direction_counts.get(-1, 0)}")
            print(f"   Neutral (0): {direction_counts.get(0, 0)}")
        
        # Test 6: Sample Data
        print("\n[Test 6] Sample Data:")
        sample_cols = ['timestamp', 'underlying_price', 'pcr', 'price_change_pct']
        if all(col in features_df.columns for col in sample_cols):
            print(features_df[sample_cols].head())
        
        print("\n" + "=" * 60)
        print("✅ Phase 1 Testing Complete!")
        print("=" * 60)
        
        # Test BSE data
        print("\n[Bonus] Testing BSE data extraction...")
        bse_data = extractor.get_time_series_data('BSE', lookback_days=7)
        if not bse_data.empty:
            print(f"✅ BSE data: {len(bse_data)} records")
        else:
            print("⚠️  No BSE data found")
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        extractor.close()

if __name__ == "__main__":
    test_phase1()

