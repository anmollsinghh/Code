"""
Production Unsupervised Toxicity Detection System
Uses only publicly observable market data for real-world deployment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
from scipy import stats
import joblib
import glob
import os

class ProductionFeatureEngineer:
    
    def __init__(self):
        self.feature_names = []
        
    def load_and_merge_data(self, data_dir="enhanced_market_data"):
        print("Loading market data...")
        
        order_files = glob.glob(f"{data_dir}/orders_*.csv")
        lob_files = glob.glob(f"{data_dir}/lob_snapshots_*.csv")
        trade_files = glob.glob(f"{data_dir}/trades_*.csv")
        
        if not order_files:
            raise FileNotFoundError(f"No order files found in {data_dir}")
        
        latest_order_file = max(order_files, key=os.path.getctime)
        orders_df = pd.read_csv(latest_order_file)
        
        public_order_columns = [
            'timestamp', 'order_id', 'order_type', 'side', 'price', 'quantity',
            'mid_price', 'spread', 'distance_from_mid', 'is_aggressive',
            'volatility', 'momentum', 'order_book_imbalance', 'time_since_last_trade'
        ]
        
        available_columns = [col for col in public_order_columns if col in orders_df.columns]
        orders_df = orders_df[available_columns].copy()
        
        print(f"Orders data: {len(orders_df)} records")
        
        lob_df = pd.DataFrame()
        if lob_files:
            latest_lob_file = max(lob_files, key=os.path.getctime)
            lob_df = pd.read_csv(latest_lob_file)
            print(f"LOB data: {len(lob_df)} snapshots")
        
        trades_df = pd.DataFrame()
        if trade_files:
            latest_trade_file = max(trade_files, key=os.path.getctime)
            trades_df = pd.read_csv(latest_trade_file)
            public_trade_columns = ['timestamp', 'price', 'quantity']
            available_trade_columns = [col for col in public_trade_columns if col in trades_df.columns]
            trades_df = trades_df[available_trade_columns].copy()
            print(f"Trades data: {len(trades_df)} trades")
        
        return orders_df, lob_df, trades_df
    
    def engineer_public_features(self, orders_df, lob_df, trades_df):
        print("Engineering features from public market data...")
        
        features_df = pd.DataFrame(index=orders_df.index)
        
        # Basic order features
        features_df['order_size'] = orders_df['quantity']
        features_df['log_order_size'] = np.log1p(orders_df['quantity'])
        features_df['is_market_order'] = (orders_df['order_type'] == 'MARKET').astype(int)
        features_df['is_buy_order'] = (orders_df['side'] == 'BUY').astype(int)
        
        # Price features
        if 'mid_price' in orders_df.columns:
            mid_price = orders_df['mid_price']
            features_df['mid_price'] = mid_price
            
            if 'price' in orders_df.columns:
                order_price = orders_df['price'].fillna(mid_price)
                features_df['price_deviation'] = (order_price - mid_price) / mid_price
                features_df['abs_price_deviation'] = np.abs(features_df['price_deviation'])
        
        # Spread features
        if 'spread' in orders_df.columns:
            features_df['spread'] = orders_df['spread']
            features_df['relative_spread'] = orders_df['spread'] / (mid_price + 1e-8)
        
        # Aggressiveness
        if 'is_aggressive' in orders_df.columns:
            features_df['is_aggressive'] = orders_df['is_aggressive'].astype(int)
        elif 'distance_from_mid' in orders_df.columns:
            features_df['is_aggressive'] = (np.abs(orders_df['distance_from_mid']) < 0.001).astype(int)
        
        # Temporal features
        if 'timestamp' in orders_df.columns:
            time_diffs = orders_df['timestamp'].diff().fillna(1)
            features_df['inter_arrival_time'] = time_diffs
            features_df['arrival_rate'] = 1 / (time_diffs + 1e-8)
            features_df['arrival_intensity_10'] = features_df['arrival_rate'].rolling(10, min_periods=1).mean()
            features_df['arrival_intensity_50'] = features_df['arrival_rate'].rolling(50, min_periods=1).mean()
        
        # Market microstructure features
        if 'volatility' in orders_df.columns:
            features_df['volatility'] = orders_df['volatility']
        else:
            if 'mid_price' in orders_df.columns:
                returns = mid_price.pct_change().fillna(0)
                features_df['volatility'] = returns.rolling(20, min_periods=1).std()
        
        if 'momentum' in orders_df.columns:
            features_df['momentum'] = orders_df['momentum']
        else:
            if 'mid_price' in orders_df.columns:
                features_df['momentum_3'] = mid_price.pct_change(3).fillna(0)
                features_df['momentum_10'] = mid_price.pct_change(10).fillna(0)
        
        if 'order_book_imbalance' in orders_df.columns:
            features_df['order_book_imbalance'] = orders_df['order_book_imbalance']
        
        # LOB features
        if not lob_df.empty:
            lob_features = self._extract_lob_features(lob_df, orders_df)
            for col in lob_features.columns:
                if col not in features_df.columns:
                    features_df[col] = lob_features[col]
        
        # Trade features
        if not trades_df.empty:
            trade_features = self._extract_trade_features(trades_df, orders_df)
            for col in trade_features.columns:
                if col not in features_df.columns:
                    features_df[col] = trade_features[col]
        
        # Rolling features
        features_df = self._add_rolling_features(features_df)
        
        # Interaction features
        features_df = self._add_interaction_features(features_df)
        
        features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)
        self.feature_names = features_df.columns.tolist()
        
        print(f"Generated {len(self.feature_names)} public market features")
        return features_df
    
    def _extract_lob_features(self, lob_df, orders_df):
        lob_features = pd.DataFrame(index=orders_df.index)
        
        if 'timestamp' in lob_df.columns and 'timestamp' in orders_df.columns:
            merged = pd.merge_asof(
                orders_df[['timestamp']].reset_index(),
                lob_df,
                on='timestamp',
                direction='backward'
            ).set_index('index')
            
            if 'imbalance' in merged.columns:
                lob_features['lob_imbalance'] = merged['imbalance']
            
            for level in range(1, 6):
                bid_price_col = f'bid_price_{level}'
                ask_price_col = f'ask_price_{level}'
                bid_size_col = f'bid_size_{level}'
                ask_size_col = f'ask_size_{level}'
                
                if all(col in merged.columns for col in [bid_price_col, ask_price_col, bid_size_col, ask_size_col]):
                    bid_size = merged[bid_size_col].fillna(0)
                    ask_size = merged[ask_size_col].fillna(0)
                    total_size = bid_size + ask_size
                    level_imbalance = (bid_size - ask_size) / (total_size + 1e-8)
                    lob_features[f'imbalance_L{level}'] = level_imbalance
                    lob_features[f'bid_depth_L{level}'] = bid_size
                    lob_features[f'ask_depth_L{level}'] = ask_size
            
            total_bid_depth = sum(merged[f'bid_size_{i}'].fillna(0) for i in range(1, 6) 
                                if f'bid_size_{i}' in merged.columns)
            total_ask_depth = sum(merged[f'ask_size_{i}'].fillna(0) for i in range(1, 6) 
                                if f'ask_size_{i}' in merged.columns)
            
            lob_features['total_bid_depth'] = total_bid_depth
            lob_features['total_ask_depth'] = total_ask_depth
            lob_features['depth_ratio'] = total_bid_depth / (total_ask_depth + 1e-8)
        
        return lob_features.fillna(0)
    
    def _extract_trade_features(self, trades_df, orders_df):
        trade_features = pd.DataFrame(index=orders_df.index)
        
        if 'timestamp' in trades_df.columns and 'timestamp' in orders_df.columns:
            for idx, order in orders_df.iterrows():
                timestamp = order['timestamp']
                
                recent_trades = trades_df[
                    (trades_df['timestamp'] >= timestamp - 10) & 
                    (trades_df['timestamp'] <= timestamp)
                ]
                
                if not recent_trades.empty:
                    trade_features.loc[idx, 'recent_trade_volume'] = recent_trades['quantity'].sum()
                    trade_features.loc[idx, 'recent_trade_count'] = len(recent_trades)
                    trade_features.loc[idx, 'avg_recent_trade_size'] = recent_trades['quantity'].mean()
                    
                    if len(recent_trades) > 1:
                        price_change = (recent_trades['price'].iloc[-1] - recent_trades['price'].iloc[0])
                        trade_features.loc[idx, 'recent_price_impact'] = price_change / recent_trades['price'].iloc[0]
                    
                    time_span = recent_trades['timestamp'].max() - recent_trades['timestamp'].min()
                    if time_span > 0:
                        trade_features.loc[idx, 'trade_frequency'] = len(recent_trades) / time_span
        
        return trade_features.fillna(0)
    
    def _add_rolling_features(self, features_df):
        key_features = ['order_size', 'relative_spread', 'volatility']
        available_features = [f for f in key_features if f in features_df.columns]
        
        for feature in available_features:
            for window in [5, 20, 50]:
                features_df[f'{feature}_ma_{window}'] = features_df[feature].rolling(window, min_periods=1).mean()
                features_df[f'{feature}_std_{window}'] = features_df[feature].rolling(window, min_periods=1).std()
                
                ma_col = f'{feature}_ma_{window}'
                std_col = f'{feature}_std_{window}'
                features_df[f'{feature}_zscore_{window}'] = (
                    (features_df[feature] - features_df[ma_col]) / (features_df[std_col] + 1e-8)
                )
        
        return features_df
    
    def _add_interaction_features(self, features_df):
        if 'order_size' in features_df.columns and 'relative_spread' in features_df.columns:
            features_df['size_spread_interaction'] = features_df['order_size'] * features_df['relative_spread']
        
        if 'volatility' in features_df.columns and 'is_aggressive' in features_df.columns:
            features_df['vol_aggression_interaction'] = features_df['volatility'] * features_df['is_aggressive']
        
        if 'order_book_imbalance' in features_df.columns and 'is_buy_order' in features_df.columns:
            features_df['imbalance_direction_interaction'] = (
                features_df['order_book_imbalance'] * (2 * features_df['is_buy_order'] - 1)
            )
        
        return features_df

class ProductionToxicityDetector:
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.ensemble_weights = {}
        self.performance_metrics = {}
        
    def prepare_features(self, features_df, correlation_threshold=0.95):
        print("Preparing features for unsupervised learning...")
        
        selected_features = self._remove_correlated_features(features_df, correlation_threshold)
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(selected_features)
        
        self.scalers['main'] = scaler
        self.feature_selector = selected_features.columns.tolist()
        
        print(f"Selected {len(self.feature_selector)} features after correlation filtering")
        
        return X_scaled, selected_features
    
    def _remove_correlated_features(self, features_df, threshold=0.95):
        corr_matrix = features_df.corr().abs()
        
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        features_to_drop = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > threshold)
        ]
        
        selected_features = features_df.drop(columns=features_to_drop)
        
        print(f"Removed {len(features_to_drop)} highly correlated features")
        return selected_features
    
    def train_ensemble_detectors(self, X_scaled, contamination_rates=[0.05, 0.10]):
        print("Training ensemble of unsupervised detectors...")
        
        detectors = {}
        
        for contamination in contamination_rates:
            print(f"Training detectors with contamination rate: {contamination}")
            
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=200,
                max_samples='auto',
                bootstrap=True
            )
            iso_forest.fit(X_scaled)
            detectors[f'isolation_forest_{contamination}'] = iso_forest
            
            lof = LocalOutlierFactor(
                n_neighbors=min(20, len(X_scaled) // 10),
                contamination=contamination,
                novelty=True
            )
            lof.fit(X_scaled)
            detectors[f'lof_{contamination}'] = lof
            
            elliptic = EllipticEnvelope(
                contamination=contamination,
                random_state=42
            )
            elliptic.fit(X_scaled)
            detectors[f'elliptic_{contamination}'] = elliptic
        
        gmm = GaussianMixture(
            n_components=min(10, len(X_scaled) // 50),
            random_state=42,
            covariance_type='full'
        )
        gmm.fit(X_scaled)
        detectors['gmm'] = gmm
        
        clustering_detector = self._train_clustering_detector(X_scaled)
        detectors['clustering'] = clustering_detector
        
        self.models = detectors
        
        return detectors
    
    def _train_clustering_detector(self, X_scaled):
        print("Training clustering-based detector...")
        
        best_k = self._optimize_kmeans_clusters(X_scaled)
        
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        distances = np.min(kmeans.transform(X_scaled), axis=1)
        
        clustering_detector = {
            'kmeans': kmeans,
            'distance_threshold': np.percentile(distances, 95)
        }
        
        return clustering_detector
    
    def _optimize_kmeans_clusters(self, X_scaled, max_k=20):
        k_range = range(2, min(max_k, len(X_scaled) // 20))
        best_score = -1
        best_k = 3
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                if len(set(labels)) > 1:
                    score = silhouette_score(X_scaled, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            except:
                continue
        
        print(f"Optimal number of clusters: {best_k} (silhouette score: {best_score:.3f})")
        return best_k
    
    def calculate_ensemble_scores(self, X_scaled):
        print("Calculating ensemble anomaly scores...")
        
        individual_scores = {}
        
        for name, model in self.models.items():
            try:
                if 'isolation_forest' in name:
                    scores = -model.decision_function(X_scaled)
                elif 'lof' in name:
                    scores = -model.score_samples(X_scaled)
                elif 'elliptic' in name:
                    scores = -model.decision_function(X_scaled)
                elif name == 'gmm':
                    scores = -model.score_samples(X_scaled)
                elif name == 'clustering':
                    distances = np.min(model['kmeans'].transform(X_scaled), axis=1)
                    scores = distances
                else:
                    continue
                
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                individual_scores[name] = scores
                
            except Exception as e:
                print(f"Error calculating scores for {name}: {e}")
                continue
        
        if individual_scores:
            weights = {name: 1.0 / len(individual_scores) for name in individual_scores}
            self.ensemble_weights = weights
            
            ensemble_scores = np.zeros(len(X_scaled))
            for name, scores in individual_scores.items():
                ensemble_scores += weights[name] * scores
        else:
            ensemble_scores = np.zeros(len(X_scaled))
        
        return ensemble_scores, individual_scores
    
    def evaluate_detectors(self, X_scaled, ensemble_scores, individual_scores):
        print("Evaluating detector performance...")
        
        metrics = {}
        
        if 'clustering' in self.models:
            kmeans = self.models['clustering']['kmeans']
            cluster_labels = kmeans.predict(X_scaled)
            
            if len(set(cluster_labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(X_scaled, cluster_labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(X_scaled, cluster_labels)
        
        anomaly_threshold = np.percentile(ensemble_scores, 95)
        anomaly_labels = ensemble_scores > anomaly_threshold
        
        individual_anomaly_rates = {}
        for name, scores in individual_scores.items():
            threshold = np.percentile(scores, 95)
            individual_anomaly_rates[name] = (scores > threshold).mean()
        
        metrics['ensemble_anomaly_rate'] = anomaly_labels.mean()
        metrics['individual_anomaly_rates'] = individual_anomaly_rates
        
        metrics['ensemble_score_mean'] = ensemble_scores.mean()
        metrics['ensemble_score_std'] = ensemble_scores.std()
        metrics['ensemble_score_skewness'] = stats.skew(ensemble_scores)
        metrics['ensemble_score_kurtosis'] = stats.kurtosis(ensemble_scores)
        
        self.performance_metrics = metrics
        
        return metrics
    
    def save_model(self, save_dir="production_models"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{save_dir}/toxicity_detector_{timestamp}.joblib"
        
        model_package = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_selector': self.feature_selector,
            'ensemble_weights': self.ensemble_weights,
            'performance_metrics': self.performance_metrics,
            'timestamp': timestamp,
            'version': '1.0'
        }
        
        joblib.dump(model_package, model_path)
        
        print(f"Model saved to: {model_path}")
        return model_path

def create_visualizations(features_df, ensemble_scores, individual_scores, detector, save_dir="production_plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Feature correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = features_df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='RdYlBu_r', center=0)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_correlations_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Anomaly score distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0,0].hist(ensemble_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(np.percentile(ensemble_scores, 95), color='red', linestyle='--', 
                     linewidth=2, label='95th Percentile')
    axes[0,0].axvline(np.percentile(ensemble_scores, 99), color='darkred', linestyle='--', 
                     linewidth=2, label='99th Percentile')
    axes[0,0].set_xlabel('Ensemble Anomaly Score')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Ensemble Anomaly Score Distribution')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    detector_names = list(individual_scores.keys())[:3]
    for i, name in enumerate(detector_names):
        if i < 3:
            axes[0,1].hist(individual_scores[name], bins=30, alpha=0.5, 
                          label=name.replace('_', ' ').title()[:15])
    axes[0,1].set_xlabel('Anomaly Score')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Individual Detector Scores')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(ensemble_scores, alpha=0.7, color='blue', linewidth=0.8)
    axes[1,0].axhline(np.percentile(ensemble_scores, 95), color='red', linestyle='--', 
                     label='Anomaly Threshold')
    axes[1,0].set_xlabel('Order Sequence')
    axes[1,0].set_ylabel('Anomaly Score')
    axes[1,0].set_title('Anomaly Scores Over Time')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    score_data = [individual_scores[name] for name in detector_names[:5]]
    score_labels = [name.replace('_', ' ').title()[:12] for name in detector_names[:5]]
    axes[1,1].boxplot(score_data, labels=score_labels)
    axes[1,1].set_ylabel('Anomaly Score')
    axes[1,1].set_title('Detector Score Distributions')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/anomaly_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # PCA visualization
    if len(features_df.columns) > 2:
        plt.figure(figsize=(12, 5))
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df.fillna(0))
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.subplot(1, 2, 1)
        anomaly_threshold = np.percentile(ensemble_scores, 95)
        normal_mask = ensemble_scores <= anomaly_threshold
        anomaly_mask = ensemble_scores > anomaly_threshold
        
        plt.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], 
                   c='lightblue', alpha=0.6, s=20, label='Normal Orders')
        if anomaly_mask.sum() > 0:
            plt.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], 
                       c='red', alpha=0.8, s=40, marker='D', label='Anomalous Orders')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('Order Behaviour in PCA Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=ensemble_scores, 
                             cmap='viridis', alpha=0.7, s=25)
        plt.colorbar(scatter, label='Anomaly Score')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('Anomaly Score Intensity')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/pca_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # Feature importance
    if detector.feature_selector and len(detector.feature_selector) > 0:
        anomaly_threshold = np.percentile(ensemble_scores, 95)
        anomaly_mask = ensemble_scores > anomaly_threshold
        
        if anomaly_mask.sum() > 0:
            feature_diffs = {}
            for feature in features_df.columns:
                normal_mean = features_df[feature][~anomaly_mask].mean()
                anomaly_mean = features_df[feature][anomaly_mask].mean()
                normal_std = features_df[feature][~anomaly_mask].std()
                
                if normal_std > 0:
                    diff_score = abs(anomaly_mean - normal_mean) / normal_std
                    feature_diffs[feature] = diff_score
            
            if feature_diffs:
                sorted_features = sorted(feature_diffs.items(), key=lambda x: x[1], reverse=True)
                top_features = sorted_features[:15]
                
                plt.figure(figsize=(12, 8))
                features_names = [f[0] for f in top_features]
                feature_scores = [f[1] for f in top_features]
                
                plt.barh(range(len(features_names)), feature_scores, color='steelblue', alpha=0.7)
                plt.yticks(range(len(features_names)), 
                          [name.replace('_', ' ').title()[:25] for name in features_names])
                plt.xlabel('Anomaly Discrimination Score')
                plt.title('Feature Importance for Anomaly Detection', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                plt.savefig(f"{save_dir}/feature_importance_{timestamp}.png", dpi=300, bbox_inches='tight')
                plt.show()

def main_production_pipeline(data_dir="enhanced_market_data"):
    print("="*80)
    print("PRODUCTION UNSUPERVISED TOXICITY DETECTION")
    print("Using only publicly observable market data")
    print("="*80)
    
    # Load and prepare data
    print("\n1. LOADING MARKET DATA")
    print("-" * 40)
    
    feature_engineer = ProductionFeatureEngineer()
    orders_df, lob_df, trades_df = feature_engineer.load_and_merge_data(data_dir)
    
    # Engineer features
    print("\n2. FEATURE ENGINEERING")
    print("-" * 40)
    
    features_df = feature_engineer.engineer_public_features(orders_df, lob_df, trades_df)
    
    # Prepare features for training
    print("\n3. FEATURE PREPARATION")
    print("-" * 40)
    
    detector = ProductionToxicityDetector()
    X_scaled, selected_features = detector.prepare_features(features_df)
    
    print(f"Final feature set: {X_scaled.shape}")
    
    # Train ensemble detectors
    print("\n4. TRAINING ENSEMBLE DETECTORS")
    print("-" * 40)
    
    detectors = detector.train_ensemble_detectors(X_scaled)
    
    # Calculate ensemble scores
    print("\n5. CALCULATING ENSEMBLE SCORES")
    print("-" * 40)
    
    ensemble_scores, individual_scores = detector.calculate_ensemble_scores(X_scaled)
    
    # Evaluate performance
    print("\n6. EVALUATING PERFORMANCE")
    print("-" * 40)
    
    metrics = detector.evaluate_detectors(X_scaled, ensemble_scores, individual_scores)
    
    # Create visualizations
    print("\n7. CREATING VISUALIZATIONS")
    print("-" * 40)
    
    create_visualizations(selected_features, ensemble_scores, individual_scores, detector)
    
    # Save model
    print("\n8. SAVING PRODUCTION MODEL")
    print("-" * 40)
    
    model_path = detector.save_model()
    
    # Print summary results
    print("\n" + "="*80)
    print("PRODUCTION MODEL TRAINING COMPLETED")
    print("="*80)
    
    print(f"\nModel Performance Summary:")
    print(f"  Ensemble anomaly rate: {metrics.get('ensemble_anomaly_rate', 0)*100:.2f}%")
    print(f"  Number of features: {len(detector.feature_selector)}")
    print(f"  Number of detectors: {len(detector.models)}")
    
    if 'silhouette_score' in metrics:
        print(f"  Clustering quality (silhouette): {metrics['silhouette_score']:.3f}")
    
    print(f"\nEnsemble Composition:")
    for name, weight in detector.ensemble_weights.items():
        print(f"  {name}: {weight:.3f}")
    
    print(f"\nDetection Statistics:")
    print(f"  Score range: {ensemble_scores.min():.4f} - {ensemble_scores.max():.4f}")
    print(f"  95th percentile: {np.percentile(ensemble_scores, 95):.4f}")
    print(f"  99th percentile: {np.percentile(ensemble_scores, 99):.4f}")
    
    print(f"\nModel saved to: {model_path}")
    
    return detector, ensemble_scores, metrics

class ProductionInference:
    
    def __init__(self, model_path):
        self.model_package = joblib.load(model_path)
        self.models = self.model_package['models']
        self.scalers = self.model_package['scalers']
        self.feature_selector = self.model_package['feature_selector']
        self.ensemble_weights = self.model_package['ensemble_weights']
        
        print(f"Loaded model trained on {self.model_package['timestamp']}")
        print(f"Model version: {self.model_package['version']}")
    
    def predict_toxicity_score(self, features_dict):
        features_df = pd.DataFrame([features_dict])
        
        missing_features = set(self.feature_selector) - set(features_df.columns)
        for feature in missing_features:
            features_df[feature] = 0
        
        features_df = features_df[self.feature_selector]
        
        X_scaled = self.scalers['main'].transform(features_df)
        
        ensemble_score = 0.0
        
        for name, model in self.models.items():
            try:
                if 'isolation_forest' in name:
                    score = -model.decision_function(X_scaled)[0]
                elif 'lof' in name:
                    score = -model.score_samples(X_scaled)[0]
                elif 'elliptic' in name:
                    score = -model.decision_function(X_scaled)[0]
                elif name == 'gmm':
                    score = -model.score_samples(X_scaled)[0]
                elif name == 'clustering':
                    distance = np.min(model['kmeans'].transform(X_scaled), axis=1)[0]
                    score = distance
                else:
                    continue
                
                score = max(0, min(1, score))
                
                weight = self.ensemble_weights.get(name, 0)
                ensemble_score += weight * score
                
            except Exception as e:
                continue
        
        return ensemble_score
    
    def is_toxic(self, features_dict, threshold=0.7):
        toxicity_score = self.predict_toxicity_score(features_dict)
        return toxicity_score > threshold, toxicity_score

class ToxicityAwareMarketMaker:
    
    def __init__(self, model_path, base_spread_bps=50, inventory_limit=40):
        self.toxicity_detector = ProductionInference(model_path)
        self.base_spread_bps = base_spread_bps
        self.inventory_limit = inventory_limit
        self.inventory = 0
        self.capital = 100000
        
        self.toxicity_multiplier_base = 2.0
        self.toxicity_threshold = 0.7
        
        self.trades = []
        self.pnl_history = []
        self.toxicity_scores = []
        self.spread_adjustments = []
        
        print("Toxicity-aware market maker initialized")
    
    def generate_orders(self, mid_price, order_features):
        toxicity_score = self.toxicity_detector.predict_toxicity_score(order_features)
        is_toxic = toxicity_score > self.toxicity_threshold
        
        if is_toxic:
            spread_multiplier = 1 + (toxicity_score * self.toxicity_multiplier_base)
        else:
            spread_multiplier = 1.0
        
        adjusted_spread_bps = self.base_spread_bps * spread_multiplier
        half_spread = mid_price * (adjusted_spread_bps / 10000) / 2
        
        inventory_skew = (self.inventory / self.inventory_limit) * 0.02 * mid_price if self.inventory_limit > 0 else 0
        
        bid_price = max(mid_price - half_spread - inventory_skew, 0.01)
        ask_price = max(mid_price + half_spread - inventory_skew, bid_price * 1.001)
        
        self.toxicity_scores.append(toxicity_score)
        self.spread_adjustments.append(adjusted_spread_bps)
        
        return {
            'bid_price': bid_price,
            'ask_price': ask_price,
            'bid_size': 2,
            'ask_size': 2,
            'toxicity_score': toxicity_score,
            'spread_bps': adjusted_spread_bps,
            'is_toxic': is_toxic
        }
    
    def record_trade(self, trade_price, trade_size, is_buy, toxicity_score):
        if is_buy:
            self.capital -= trade_price * trade_size
            self.inventory += trade_size
        else:
            self.capital += trade_price * trade_size
            self.inventory -= trade_size
        
        trade_pnl = trade_price * trade_size * (1 if not is_buy else -1)
        
        self.trades.append({
            'price': trade_price,
            'size': trade_size,
            'is_buy': is_buy,
            'pnl': trade_pnl,
            'toxicity_score': toxicity_score
        })
        
        mtm_value = self.capital + self.inventory * trade_price
        self.pnl_history.append(mtm_value)
    
    def get_performance_summary(self):
        if not self.pnl_history:
            return {}
        
        total_return = (self.pnl_history[-1] / 100000 - 1) * 100
        avg_toxicity_score = np.mean(self.toxicity_scores) if self.toxicity_scores else 0
        avg_spread = np.mean(self.spread_adjustments) if self.spread_adjustments else self.base_spread_bps
        
        toxic_trades = [t for t in self.trades if t['toxicity_score'] > self.toxicity_threshold]
        normal_trades = [t for t in self.trades if t['toxicity_score'] <= self.toxicity_threshold]
        
        return {
            'total_return_pct': total_return,
            'total_trades': len(self.trades),
            'toxic_trades': len(toxic_trades),
            'normal_trades': len(normal_trades),
            'avg_toxicity_score': avg_toxicity_score,
            'avg_spread_bps': avg_spread,
            'final_inventory': self.inventory,
            'toxic_trade_pnl': sum(t['pnl'] for t in toxic_trades) if toxic_trades else 0,
            'normal_trade_pnl': sum(t['pnl'] for t in normal_trades) if normal_trades else 0
        }

def run_backtest_with_toxicity_detection(model_path, orders_df, lob_df):
    print("Running backtest with toxicity detection...")
    
    mm = ToxicityAwareMarketMaker(model_path)
    
    for idx, order in orders_df.iterrows():
        if idx % 100 == 0:
            print(f"Processing order {idx}/{len(orders_df)}")
        
        mid_price = order.get('mid_price', 250.0)
        
        order_features = {
            'order_size': order.get('quantity', 1),
            'log_order_size': np.log1p(order.get('quantity', 1)),
            'is_market_order': 1 if order.get('order_type') == 'MARKET' else 0,
            'is_buy_order': 1 if order.get('side') == 'BUY' else 0,
            'relative_spread': order.get('spread', 0.01) / (mid_price + 1e-8),
            'volatility': order.get('volatility', 0.01),
            'momentum_3': order.get('momentum', 0.0),
            'order_book_imbalance': order.get('order_book_imbalance', 0.0),
            'arrival_rate': 0.5,
            'size_spread_interaction': order.get('quantity', 1) * order.get('spread', 0.01) / (mid_price + 1e-8)
        }
        
        quotes = mm.generate_orders(mid_price, order_features)
        
        if order.get('order_type') == 'MARKET':
            if order.get('side') == 'BUY':
                trade_price = quotes['ask_price']
                mm.record_trade(trade_price, order.get('quantity', 1), False, quotes['toxicity_score'])
            else:
                trade_price = quotes['bid_price']
                mm.record_trade(trade_price, order.get('quantity', 1), True, quotes['toxicity_score'])
    
    performance = mm.get_performance_summary()
    
    print("\nBacktest Results:")
    print(f"Total Return: {performance['total_return_pct']:.2f}%")
    print(f"Total Trades: {performance['total_trades']}")
    print(f"Toxic Trades: {performance['toxic_trades']} ({performance['toxic_trades']/performance['total_trades']*100:.1f}%)")
    print(f"Average Spread: {performance['avg_spread_bps']:.1f} bps")
    print(f"Toxic Trade P&L: {performance['toxic_trade_pnl']:.2f}")
    print(f"Normal Trade P&L: {performance['normal_trade_pnl']:.2f}")
    
    return mm, performance

if __name__ == "__main__":
    try:
        detector, ensemble_scores, metrics = main_production_pipeline()
        
        print("\n" + "="*80)
        print("SUCCESS: Production model training completed!")
        print("="*80)
        
    except Exception as e:
        print(f"Error in production pipeline: {e}")
        import traceback
        traceback.print_exc()