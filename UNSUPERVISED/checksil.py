"""
Investigation: Why Silhouette Score = 1.000 (Suspicious Perfect Score)
======================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def investigate_clustering_quality(features_df):
    """Investigate why silhouette score is suspiciously perfect"""
    
    print("INVESTIGATING SUSPICIOUS SILHOUETTE SCORE = 1.000")
    print("="*60)
    
    # 1. Check data properties
    print("\n1. DATA PROPERTIES ANALYSIS")
    print("-" * 40)
    
    print(f"Feature matrix shape: {features_df.shape}")
    print(f"Number of unique rows: {features_df.drop_duplicates().shape[0]}")
    print(f"Percentage unique: {features_df.drop_duplicates().shape[0] / len(features_df) * 100:.1f}%")
    
    # Check for constant features
    constant_features = []
    for col in features_df.columns:
        if features_df[col].nunique() <= 1:
            constant_features.append(col)
    
    print(f"Constant features: {len(constant_features)}")
    if constant_features:
        print(f"Constant feature names: {constant_features[:5]}...")
    
    # Check for highly sparse features
    sparse_features = []
    for col in features_df.columns:
        zero_pct = (features_df[col] == 0).mean()
        if zero_pct > 0.9:
            sparse_features.append((col, zero_pct))
    
    print(f"Highly sparse features (>90% zeros): {len(sparse_features)}")
    if sparse_features:
        print("Top sparse features:")
        for feat, pct in sorted(sparse_features, key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {feat}: {pct*100:.1f}% zeros")
    
    # 2. Examine feature distributions
    print("\n2. FEATURE DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    # Check for binary-like features
    binary_like = []
    for col in features_df.columns:
        unique_vals = features_df[col].nunique()
        if unique_vals <= 5:
            binary_like.append((col, unique_vals, features_df[col].unique()[:5]))
    
    print(f"Binary/categorical-like features: {len(binary_like)}")
    if binary_like:
        print("Examples:")
        for feat, n_unique, vals in binary_like[:5]:
            print(f"  {feat}: {n_unique} unique values {vals}")
    
    # 3. Test clustering with different preprocessing
    print("\n3. CLUSTERING SENSITIVITY ANALYSIS")
    print("-" * 40)
    
    # Remove constant and highly sparse features
    clean_features = features_df.copy()
    cols_to_remove = constant_features + [feat for feat, _ in sparse_features]
    clean_features = clean_features.drop(columns=cols_to_remove, errors='ignore')
    
    print(f"Features after cleaning: {clean_features.shape[1]}")
    
    # Test different scalers and clustering
    results = {}
    
    scalers = {
        'StandardScaler': StandardScaler(),
        'RobustScaler': RobustScaler(),
        'MinMaxScaler': None  # Will implement manually
    }
    
    for scaler_name, scaler in scalers.items():
        if scaler_name == 'MinMaxScaler':
            # Manual MinMax to avoid division by zero
            X_scaled = clean_features.copy()
            for col in X_scaled.columns:
                col_min = X_scaled[col].min()
                col_max = X_scaled[col].max()
                if col_max - col_min > 1e-8:  # Avoid division by zero
                    X_scaled[col] = (X_scaled[col] - col_min) / (col_max - col_min)
                else:
                    X_scaled[col] = 0
        else:
            X_scaled = pd.DataFrame(
                scaler.fit_transform(clean_features),
                columns=clean_features.columns,
                index=clean_features.index
            )
        
        # Test different k values
        for k in [3, 5, 7, 9]:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                if len(set(labels)) > 1:  # Valid clustering
                    sil_score = silhouette_score(X_scaled, labels)
                    sil_samples = silhouette_samples(X_scaled, labels)
                    
                    results[f"{scaler_name}_k{k}"] = {
                        'silhouette_score': sil_score,
                        'min_sample_score': sil_samples.min(),
                        'max_sample_score': sil_samples.max(),
                        'labels': labels,
                        'n_clusters_actual': len(set(labels))
                    }
            except Exception as e:
                print(f"Error with {scaler_name} k={k}: {e}")
    
    # Display results
    print("\nClustering Results:")
    print(f"{'Method':<20} {'Silhouette':<12} {'Min Sample':<12} {'Max Sample':<12} {'Clusters':<8}")
    print("-" * 65)
    
    for method, res in results.items():
        print(f"{method:<20} {res['silhouette_score']:<12.6f} "
              f"{res['min_sample_score']:<12.6f} {res['max_sample_score']:<12.6f} "
              f"{res['n_clusters_actual']:<8}")
    
    # 4. Investigate perfect score case
    perfect_scores = [method for method, res in results.items() 
                     if abs(res['silhouette_score'] - 1.0) < 1e-6]
    
    if perfect_scores:
        print(f"\n4. INVESTIGATING PERFECT SCORES")
        print("-" * 40)
        print(f"Methods with perfect scores: {perfect_scores}")
        
        # Analyze the perfect case
        method = perfect_scores[0]
        labels = results[method]['labels']
        
        print(f"\nAnalyzing {method}:")
        print(f"Cluster distribution: {np.bincount(labels)}")
        
        # Check if clusters are separated by simple rules
        if 'StandardScaler' in method:
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(clean_features),
                columns=clean_features.columns
            )
        else:
            X_scaled = clean_features
        
        # Find features that perfectly separate clusters
        separating_features = []
        for col in X_scaled.columns:
            cluster_means = []
            for cluster_id in set(labels):
                cluster_mask = labels == cluster_id
                if cluster_mask.sum() > 0:
                    cluster_mean = X_scaled[col][cluster_mask].mean()
                    cluster_means.append(cluster_mean)
            
            # Check if means are well separated
            if len(cluster_means) > 1:
                mean_separation = max(cluster_means) - min(cluster_means)
                overall_std = X_scaled[col].std()
                if overall_std > 0:
                    separation_ratio = mean_separation / overall_std
                    if separation_ratio > 3:  # Highly separated
                        separating_features.append((col, separation_ratio))
        
        separating_features.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nFeatures causing perfect separation:")
        for feat, ratio in separating_features[:10]:
            print(f"  {feat}: separation ratio = {ratio:.2f}")
    
    return results, clean_features

def plot_clustering_investigation(features_df, results):
    """Create diagnostic plots for clustering investigation"""
    
    # Find the method that gave perfect score
    perfect_method = None
    for method, res in results.items():
        if abs(res['silhouette_score'] - 1.0) < 1e-6:
            perfect_method = method
            break
    
    if perfect_method is None:
        print("No perfect silhouette scores found!")
        return
    
    labels = results[perfect_method]['labels']
    
    # Clean and scale features
    clean_features = features_df.select_dtypes(include=[np.number]).fillna(0)
    
    # Remove constant columns
    clean_features = clean_features.loc[:, clean_features.var() > 1e-8]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean_features)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Clustering Investigation: {perfect_method}', fontsize=16)
    
    # Plot 1: PCA with cluster labels
    scatter = axes[0,0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.7)
    axes[0,0].set_title('Clusters in PCA Space')
    axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.colorbar(scatter, ax=axes[0,0])
    
    # Plot 2: Cluster size distribution
    cluster_counts = np.bincount(labels)
    axes[0,1].bar(range(len(cluster_counts)), cluster_counts)
    axes[0,1].set_title('Cluster Size Distribution')
    axes[0,1].set_xlabel('Cluster ID')
    axes[0,1].set_ylabel('Number of Points')
    
    # Plot 3: Silhouette analysis
    from sklearn.metrics import silhouette_samples
    silhouette_vals = silhouette_samples(X_scaled, labels)
    
    y_lower = 10
    for i in range(len(set(labels))):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
        
        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.tab10(i / len(set(labels)))
        axes[1,0].fill_betweenx(np.arange(y_lower, y_upper),
                               0, cluster_silhouette_vals,
                               facecolor=color, edgecolor=color, alpha=0.7)
        
        axes[1,0].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    axes[1,0].set_title('Silhouette Plot')
    axes[1,0].set_xlabel('Silhouette Coefficient Values')
    axes[1,0].set_ylabel('Cluster Label')
    
    # Add average silhouette score line
    avg_score = silhouette_vals.mean()
    axes[1,0].axvline(x=avg_score, color="red", linestyle="--", 
                     label=f'Average Score: {avg_score:.3f}')
    axes[1,0].legend()
    
    # Plot 4: Feature variance by cluster
    feature_vars = []
    for i in range(len(set(labels))):
        cluster_mask = labels == i
        if cluster_mask.sum() > 1:
            cluster_data = X_scaled[cluster_mask]
            cluster_var = np.var(cluster_data, axis=0).mean()
            feature_vars.append(cluster_var)
        else:
            feature_vars.append(0)
    
    axes[1,1].bar(range(len(feature_vars)), feature_vars)
    axes[1,1].set_title('Average Feature Variance by Cluster')
    axes[1,1].set_xlabel('Cluster ID')
    axes[1,1].set_ylabel('Average Variance')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Main investigation function
def run_clustering_investigation(data_dir="enhanced_market_data"):
    """Run the complete clustering investigation"""
    
    print("CLUSTERING QUALITY INVESTIGATION")
    print("="*50)
    
    # Load data (simplified version)
    import glob
    order_files = glob.glob(f"{data_dir}/orders_*.csv")
    if not order_files:
        print("No order files found!")
        return
    
    latest_order_file = max(order_files, key=lambda x: x.split('_')[-1])
    orders_df = pd.read_csv(latest_order_file)
    
    # Create a simplified feature set for investigation
    feature_columns = [
        'order_size', 'log_order_size', 'is_market_order', 'is_buy_order',
        'relative_spread', 'volatility', 'momentum_3', 'order_book_imbalance',
        'arrival_rate', 'size_spread_interaction'
    ]
    
    # Filter to available columns
    available_features = [col for col in feature_columns if col in orders_df.columns]
    
    if not available_features:
        print("No suitable features found in the data!")
        return
    
    features_df = orders_df[available_features].fillna(0)
    
    print(f"Investigating {len(features_df)} orders with {len(available_features)} features")
    
    # Run investigation
    results, clean_features = investigate_clustering_quality(features_df)
    
    # Create diagnostic plots
    if results:
        plot_clustering_investigation(clean_features, results)
    
    return results

if __name__ == "__main__":
    results = run_clustering_investigation()