"""
Debug script to identify and fix model issues
Focuses on data quality and feature engineering problems
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import os

def debug_training_data(data_dir="enhanced_market_data"):
    """Debug the training data to identify separation issues"""
    print("="*60)
    print("DEBUGGING TRAINING DATA")
    print("="*60)
    
    try:
        # Load the original data
        import glob
        order_files = glob.glob(f"{data_dir}/orders_*.csv")
        if not order_files:
            print(f"✗ No order files found in {data_dir}")
            return None
            
        latest_order_file = max(order_files, key=os.path.getctime)
        orders_df = pd.read_csv(latest_order_file)
        
        print(f"✓ Loaded orders data: {orders_df.shape}")
        print(f"Columns: {list(orders_df.columns)}")
        
        # Check for data quality issues
        print(f"\nData Quality Check:")
        print(f"  Duplicate rows: {orders_df.duplicated().sum()}")
        print(f"  Missing values: {orders_df.isnull().sum().sum()}")
        
        # Check key features for variation
        key_features = ['quantity', 'price', 'mid_price', 'spread', 'volatility', 'momentum']
        available_features = [f for f in key_features if f in orders_df.columns]
        
        print(f"\nFeature Variation Analysis:")
        for feature in available_features:
            values = orders_df[feature].dropna()
            if len(values) > 0:
                print(f"  {feature}:")
                print(f"    Count: {len(values)}")
                print(f"    Unique values: {values.nunique()}")
                print(f"    Range: [{values.min():.6f}, {values.max():.6f}]")
                print(f"    Std: {values.std():.6f}")
                print(f"    CV: {values.std()/abs(values.mean()):.6f}" if values.mean() != 0 else "    CV: inf")
                
                # Check for constant or near-constant values
                if values.nunique() == 1:
                    print(f"    ⚠️  CONSTANT VALUES!")
                elif values.std() < 1e-10:
                    print(f"    ⚠️  NEAR-CONSTANT VALUES!")
        
        return orders_df
        
    except Exception as e:
        print(f"✗ Error loading training data: {e}")
        return None

def debug_feature_engineering(orders_df):
    """Debug the feature engineering process"""
    print("\n" + "="*60)
    print("DEBUGGING FEATURE ENGINEERING")
    print("="*60)
    
    if orders_df is None:
        print("✗ No data available for feature engineering debug")
        return None
    
    # Simulate the feature engineering process
    features_df = pd.DataFrame(index=orders_df.index)
    
    # Basic features
    if 'quantity' in orders_df.columns:
        features_df['order_size'] = orders_df['quantity']
        features_df['log_order_size'] = np.log1p(orders_df['quantity'])
        
        print(f"Order size features:")
        print(f"  order_size range: [{features_df['order_size'].min():.2f}, {features_df['order_size'].max():.2f}]")
        print(f"  log_order_size range: [{features_df['log_order_size'].min():.2f}, {features_df['log_order_size'].max():.2f}]")
    
    # Check for problematic feature combinations
    if 'mid_price' in orders_df.columns:
        features_df['mid_price'] = orders_df['mid_price']
        
        # Check if mid_price is constant
        if orders_df['mid_price'].nunique() == 1:
            print("⚠️  WARNING: mid_price is constant!")
            
        if 'price' in orders_df.columns:
            price_diff = orders_df['price'] - orders_df['mid_price']
            features_df['price_deviation'] = price_diff / (orders_df['mid_price'] + 1e-8)
            
            print(f"Price deviation stats:")
            print(f"  Range: [{features_df['price_deviation'].min():.6f}, {features_df['price_deviation'].max():.6f}]")
            print(f"  Std: {features_df['price_deviation'].std():.6f}")
            
            # Check for extreme values that might cause separation
            extreme_mask = np.abs(features_df['price_deviation']) > 10
            if extreme_mask.any():
                print(f"  ⚠️  {extreme_mask.sum()} extreme price deviations found!")
    
    # Temporal features - check for patterns
    if 'timestamp' in orders_df.columns:
        timestamps = pd.to_datetime(orders_df['timestamp'])
        features_df['hour_of_day'] = timestamps.dt.hour
        
        hour_distribution = features_df['hour_of_day'].value_counts().sort_index()
        print(f"\nHour distribution:")
        print(hour_distribution)
        
        # Check if all orders are from same hour (would create separation)
        if features_df['hour_of_day'].nunique() == 1:
            print("⚠️  WARNING: All orders from same hour!")
    
    # Remove infinite and NaN values
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    nan_counts = features_df.isnull().sum()
    if nan_counts.any():
        print(f"\nNaN/Inf values found:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"  {col}: {count} NaN values")
    
    features_df = features_df.fillna(0)
    
    print(f"\nFinal features shape: {features_df.shape}")
    
    return features_df

def debug_clustering_on_real_data(features_df):
    """Test clustering on the actual training data"""
    print("\n" + "="*60)
    print("DEBUGGING CLUSTERING ON REAL DATA")
    print("="*60)
    
    if features_df is None or features_df.empty:
        print("✗ No features available for clustering debug")
        return
    
    # Take a sample to speed up analysis
    sample_size = min(5000, len(features_df))
    sample_df = features_df.sample(n=sample_size, random_state=42)
    
    print(f"Testing clustering on {sample_size} samples...")
    
    # Test different scalers
    scalers = {
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'none': None
    }
    
    for scaler_name, scaler in scalers.items():
        print(f"\n--- Testing with {scaler_name} scaler ---")
        
        if scaler is not None:
            X_scaled = scaler.fit_transform(sample_df)
        else:
            X_scaled = sample_df.values
        
        print(f"Scaled data shape: {X_scaled.shape}")
        print(f"Scaled data range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
        print(f"Scaled data std: {X_scaled.std():.3f}")
        
        # Test K-means with different cluster numbers
        for k in [3, 5, 8]:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                # Check cluster distribution
                unique_labels, counts = np.unique(labels, return_counts=True)
                
                print(f"\n  K={k}:")
                print(f"    Clusters found: {len(unique_labels)}")
                print(f"    Cluster sizes: {counts}")
                print(f"    Size ratios: {counts/counts.sum()}")
                
                # Check for degenerate clustering (one huge cluster + tiny ones)
                max_cluster_ratio = counts.max() / counts.sum()
                if max_cluster_ratio > 0.95:
                    print(f"    ⚠️  Degenerate clustering: {max_cluster_ratio:.3f} in largest cluster")
                
                if len(unique_labels) > 1:
                    try:
                        sil_score = silhouette_score(X_scaled, labels)
                        cal_score = calinski_harabasz_score(X_scaled, labels)
                        db_score = davies_bouldin_score(X_scaled, labels)
                        
                        print(f"    Silhouette: {sil_score:.4f}")
                        print(f"    Calinski-Harabasz: {cal_score:.1f}")
                        print(f"    Davies-Bouldin: {db_score:.4f}")
                        
                        # Flag suspicious scores
                        if sil_score > 0.8:
                            print(f"    🚨 SUSPICIOUS: Very high silhouette score!")
                        if cal_score > 1e5:
                            print(f"    🚨 SUSPICIOUS