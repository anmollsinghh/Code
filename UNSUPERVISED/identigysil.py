"""
CLUSTERING PROBLEM DIAGNOSIS AND FIXES
=====================================
Addressing perfect silhouette scores (1.000) in toxicity detection model
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

class ClusteringDiagnostics:
    """Diagnose and fix clustering issues in toxicity detection"""
    
    def __init__(self):
        self.problematic_features = []
        self.diagnostic_results = {}
    
    def diagnose_perfect_scores(self, features_df, labels):
        """Diagnose why clustering gives perfect scores"""
        
        print("DIAGNOSING PERFECT CLUSTERING SCORES")
        print("="*50)
        
        # 1. Check for features causing artificial separation
        print("\n1. CHECKING FOR ARTIFICIAL SEPARATION")
        print("-"*40)
        
        separating_features = []
        
        for col in features_df.columns:
            cluster_stats = []
            for cluster_id in np.unique(labels):
                cluster_mask = labels == cluster_id
                if cluster_mask.sum() > 0:
                    cluster_data = features_df[col][cluster_mask]
                    cluster_stats.append({
                        'cluster': cluster_id,
                        'mean': cluster_data.mean(),
                        'std': cluster_data.std(),
                        'min': cluster_data.min(),
                        'max': cluster_data.max(),
                        'size': len(cluster_data)
                    })
            
            if len(cluster_stats) > 1:
                # Check for non-overlapping clusters
                ranges = [(stat['min'], stat['max']) for stat in cluster_stats]
                non_overlapping = True
                
                for i in range(len(ranges)):
                    for j in range(i+1, len(ranges)):
                        if not (ranges[i][1] < ranges[j][0] or ranges[j][1] < ranges[i][0]):
                            non_overlapping = False
                            break
                    if not non_overlapping:
                        break
                
                if non_overlapping:
                    separating_features.append({
                        'feature': col,
                        'cluster_stats': cluster_stats,
                        'separation_type': 'non_overlapping'
                    })
                
                # Check for extreme separation ratios
                means = [stat['mean'] for stat in cluster_stats]
                mean_range = max(means) - min(means)
                overall_std = features_df[col].std()
                
                if overall_std > 0:
                    separation_ratio = mean_range / overall_std
                    if separation_ratio > 10:  # Extremely high separation
                        separating_features.append({
                            'feature': col,
                            'separation_ratio': separation_ratio,
                            'cluster_stats': cluster_stats,
                            'separation_type': 'extreme_ratio'
                        })
        
        self.problematic_features = separating_features
        
        print(f"Found {len(separating_features)} problematic features:")
        for feat_info in separating_features[:5]:  # Show top 5
            feature_name = feat_info['feature']
            sep_type = feat_info['separation_type']
            
            if sep_type == 'non_overlapping':
                print(f"  {feature_name}: Non-overlapping clusters")
                for stat in feat_info['cluster_stats']:
                    print(f"    Cluster {stat['cluster']}: [{stat['min']:.3f}, {stat['max']:.3f}] "
                          f"(n={stat['size']})")
            elif sep_type == 'extreme_ratio':
                ratio = feat_info['separation_ratio']
                print(f"  {feature_name}: Extreme separation ratio = {ratio:.1f}")
        
        return separating_features
    
    def identify_root_causes(self, features_df):
        """Identify specific root causes of artificial separation"""
        
        print("\n2. IDENTIFYING ROOT CAUSES")
        print("-"*40)
        
        issues = []
        
        # Check for binary features disguised as continuous
        for col in features_df.columns:
            unique_vals = features_df[col].nunique()
            if unique_vals <= 10:  # Likely categorical
                val_counts = features_df[col].value_counts()
                dominant_values = val_counts.head(3)
                
                if dominant_values.iloc[0] / len(features_df) > 0.8:  # One value dominates
                    issues.append({
                        'feature': col,
                        'issue': 'dominant_value',
                        'dominant_pct': dominant_values.iloc[0] / len(features_df),
                        'unique_count': unique_vals
                    })
        
        # Check for features with extreme outliers
        for col in features_df.select_dtypes(include=[np.number]).columns:
            q99 = features_df[col].quantile(0.99)
            q01 = features_df[col].quantile(0.01)
            iqr = features_df[col].quantile(0.75) - features_df[col].quantile(0.25)
            
            if iqr > 0:
                outlier_range = (q99 - q01) / iqr
                if outlier_range > 20:  # Extreme outlier range
                    issues.append({
                        'feature': col,
                        'issue': 'extreme_outliers',
                        'outlier_range_ratio': outlier_range,
                        'q01': q01,
                        'q99': q99
                    })
        
        # Check for features with scale differences
        scales = {}
        for col in features_df.select_dtypes(include=[np.number]).columns:
            magnitude = np.log10(abs(features_df[col].max()) + 1e-10)
            scales[col] = magnitude
        
        if scales:
            max_scale = max(scales.values())
            min_scale = min(scales.values())
            scale_difference = max_scale - min_scale
            
            if scale_difference > 5:  # Orders of magnitude difference
                issues.append({
                    'issue': 'scale_mismatch',
                    'scale_difference': scale_difference,
                    'max_scale_feature': max(scales, key=scales.get),
                    'min_scale_feature': min(scales, key=scales.get)
                })
        
        print(f"Identified {len(issues)} root cause issues:")
        for issue in issues:
            if issue.get('issue') == 'dominant_value':
                print(f"  {issue['feature']}: {issue['dominant_pct']*100:.1f}% dominated by one value")
            elif issue.get('issue') == 'extreme_outliers':
                print(f"  {issue['feature']}: Extreme outlier range ratio = {issue['outlier_range_ratio']:.1f}")
            elif issue.get('issue') == 'scale_mismatch':
                print(f"  Scale mismatch: {issue['scale_difference']:.1f} orders of magnitude difference")
                print(f"    Largest: {issue['max_scale_feature']}")
                print(f"    Smallest: {issue['min_scale_feature']}")
        
        return issues

class FixedFeatureEngineer:
    """Fixed version of feature engineering to prevent artificial clustering"""
    
    def __init__(self):
        self.feature_names = []
        self.scaling_info = {}
    
    def engineer_robust_features(self, orders_df):
        """Engineer features with proper handling to avoid artificial clustering"""
        
        print("ENGINEERING ROBUST FEATURES (FIXED)")
        print("="*40)
        
        features_df = pd.DataFrame(index=orders_df.index)
        
        # 1. Basic features with robust scaling
        if 'quantity' in orders_df.columns:
            # Log transform with offset to handle zeros
            features_df['log_order_size'] = np.log1p(orders_df['quantity'])
            
            # Robust size categories instead of raw size
            size_percentiles = orders_df['quantity'].quantile([0.2, 0.4, 0.6, 0.8])
            features_df['size_category'] = pd.cut(
                orders_df['quantity'], 
                bins=[-np.inf] + size_percentiles.tolist() + [np.inf],
                labels=[0, 1, 2, 3, 4]
            ).astype(float)
        
        # 2. Spread features with proper normalization
        if 'spread' in orders_df.columns and 'mid_price' in orders_df.columns:
            # Relative spread instead of absolute
            spread = orders_df['spread']
            mid_price = orders_df['mid_price']
            
            # Handle division by zero gracefully
            relative_spread = spread / (mid_price + 1e-8)
            
            # Cap extreme values to prevent outlier-driven clustering
            relative_spread = np.clip(relative_spread, 0, relative_spread.quantile(0.99))
            features_df['relative_spread'] = relative_spread
            
            # Spread percentile rank (more robust than raw values)
            features_df['spread_rank'] = spread.rank(pct=True)
        
        # 3. Time-based features with proper encoding
        if 'timestamp' in orders_df.columns:
            timestamps = pd.to_datetime(orders_df['timestamp'])
            
            # Cyclical encoding (prevents time-based clustering artifacts)
            hour = timestamps.dt.hour
            features_df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            features_df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            
            # Robust inter-arrival time
            time_diffs = orders_df['timestamp'].diff().fillna(0)
            time_diffs = np.clip(time_diffs, 0, time_diffs.quantile(0.95))  # Cap extremes
            features_df['log_inter_arrival'] = np.log1p(time_diffs)
        
        # 4. Market microstructure with robust handling
        if 'volatility' in orders_df.columns:
            vol = orders_df['volatility']
            # Use rank transformation to make more robust
            features_df['volatility_rank'] = vol.rank(pct=True)
            
            # Volatility categories instead of raw values
            vol_percentiles = vol.quantile([0.25, 0.5, 0.75])
            features_df['volatility_category'] = pd.cut(
                vol,
                bins=[-np.inf] + vol_percentiles.tolist() + [np.inf],
                labels=[0, 1, 2, 3]
            ).astype(float)
        
        # 5. Order book features with normalization
        if 'order_book_imbalance' in orders_df.columns:
            imbalance = orders_df['order_book_imbalance']
            
            # Clip extreme imbalances
            imbalance_clipped = np.clip(imbalance, 
                                      imbalance.quantile(0.01), 
                                      imbalance.quantile(0.99))
            features_df['imbalance_clipped'] = imbalance_clipped
            
            # Sign of imbalance (robust directional feature)
            features_df['imbalance_sign'] = np.sign(imbalance)
        
        # 6. Interaction features (carefully constructed)
        if 'size_category' in features_df.columns and 'volatility_rank' in features_df.columns:
            # Multiplicative interaction (bounded)
            features_df['size_vol_interaction'] = (
                features_df['size_category'] * features_df['volatility_rank']
            )
        
        # 7. Rolling features with proper window handling
        if 'log_order_size' in features_df.columns:
            # Use shorter windows to avoid over-smoothing
            for window in [5, 10]:
                rolling_mean = features_df['log_order_size'].rolling(
                    window, min_periods=1).mean()
                features_df[f'size_ma_{window}'] = rolling_mean
                
                # Z-score relative to recent history
                rolling_std = features_df['log_order_size'].rolling(
                    window, min_periods=1).std()
                features_df[f'size_zscore_{window}'] = (
                    (features_df['log_order_size'] - rolling_mean) / 
                    (rolling_std + 1e-8)
                )
                # Clip extreme z-scores
                features_df[f'size_zscore_{window}'] = np.clip(
                    features_df[f'size_zscore_{window}'], -3, 3
                )
        
        # Final cleaning and validation
        features_df = features_df.fillna(0)
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        # Remove any remaining constant features
        constant_cols = []
        for col in features_df.columns:
            if features_df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            features_df = features_df.drop(columns=constant_cols)
            print(f"Removed {len(constant_cols)} constant features")
        
        # Validate feature ranges
        print("\nFeature validation:")
        for col in features_df.columns:
            col_min, col_max = features_df[col].min(), features_df[col].max()
            col_std = features_df[col].std()
            print(f"  {col}: range=[{col_min:.3f}, {col_max:.3f}], std={col_std:.3f}")
        
        self.feature_names = features_df.columns.tolist()
        return features_df

class RobustClusteringValidator:
    """Validate clustering results with realistic expectations"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_clustering_quality(self, X_scaled, labels):
        """Validate clustering with realistic quality checks"""
        
        print("\nVALIDATING CLUSTERING QUALITY")
        print("-"*40)
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)
        
        # Realistic thresholds for financial market data
        quality_assessment = {
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski,
        }
        
        # Quality flags
        flags = []
        
        if silhouette > 0.8:
            flags.append("WARNING: Suspiciously high silhouette score")
        elif silhouette < 0.2:
            flags.append("WARNING: Poor cluster separation")
        
        if davies_bouldin < 0.1:
            flags.append("WARNING: Unrealistically low Davies-Bouldin score")
        
        if calinski > 1000000:  # Arbitrary large threshold
            flags.append("WARNING: Extremely high Calinski-Harabasz score")
        
        # Expected ranges for market data clustering
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"  Expected range: 0.2 - 0.7 (realistic for market data)")
        print(f"  Status: {'✓ Realistic' if 0.2 <= silhouette <= 0.7 else '⚠ Suspicious'}")
        
        print(f"\nDavies-Bouldin Score: {davies_bouldin:.3f}")
        print(f"  Expected range: 0.5 - 2.0 (lower is better)")
        print(f"  Status: {'✓ Realistic' if 0.5 <= davies_bouldin <= 2.0 else '⚠ Suspicious'}")
        
        print(f"\nCalinski-Harabasz Score: {calinski:.1f}")
        print(f"  Expected range: 100 - 100,000 (higher is better)")
        print(f"  Status: {'✓ Realistic' if 100 <= calinski <= 100000 else '⚠ Suspicious'}")
        
        if flags:
            print(f"\nQuality Flags:")
            for flag in flags:
                print(f"  {flag}")
        
        quality_assessment['flags'] = flags
        quality_assessment['is_realistic'] = len(flags) == 0
        
        self.validation_results = quality_assessment
        return quality_assessment

# Example usage and fixes
def demonstrate_fixes():
    """Demonstrate the fixes for clustering issues"""
    
    print("DEMONSTRATING CLUSTERING FIXES")
    print("="*50)
    
    # Create synthetic problematic data (similar to what's causing issues)
    np.random.seed(42)
    n_samples = 1000
    
    # Feature 1: Binary-like feature with extreme separation
    feature1 = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Feature 2: Continuous but with extreme outliers
    feature2 = np.random.normal(0, 1, n_samples)
    feature2[feature1 == 1] += 100  # Create artificial separation
    
    # Create problematic feature matrix
    problematic_features = pd.DataFrame({
        'binary_like': feature1,
        'extreme_outliers': feature2,
        'normal_feature': np.random.normal(0, 1, n_samples)
    })
    
    print("Testing problematic features:")
    
    # Test clustering on problematic data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(problematic_features)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    # Diagnose issues
    diagnostics = ClusteringDiagnostics()
    separating_features = diagnostics.diagnose_perfect_scores(problematic_features, labels)
    root_causes = diagnostics.identify_root_causes(problematic_features)
    
    # Validate with realistic checker
    validator = RobustClusteringValidator()
    quality_assessment = validator.validate_clustering_quality(X_scaled, labels)
    
    print("\n" + "="*50)
    print("SUMMARY OF FIXES NEEDED:")
    print("="*50)
    
    print("\n1. Feature Engineering Fixes:")
    print("   - Use rank transformations instead of raw values")
    print("   - Clip extreme outliers at 95th/99th percentiles")
    print("   - Convert continuous features to robust categories")
    print("   - Use cyclical encoding for temporal features")
    print("   - Apply proper normalization for ratio features")
    
    print("\n2. Clustering Validation Fixes:")
    print("   - Set realistic quality thresholds")
    print("   - Flag suspicious perfect scores")
    print("   - Monitor for artificial separation")
    print("   - Use multiple validation metrics")
    
    print("\n3. Preprocessing Fixes:")
    print("   - Remove features with >90% zeros")
    print("   - Handle division by zero in ratio calculations")
    print("   - Cap interaction features to prevent explosion")
    print("   - Validate feature distributions before clustering")

if __name__ == "__main__":
    demonstrate_fixes()