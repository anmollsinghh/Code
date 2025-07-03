"""
Section 4.4: Realistic Unsupervised Learning for Toxicity Detection - NASDAQ ITCH Data
=====================================================================================
Implementation using ONLY publicly observable market data from NASDAQ ITCH feed
Adapted for the specific CSV format provided
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core libraries for unsupervised learning
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import joblib
import glob
import os

class NASDAQToxicityDetector:
    """
    4.4.1 & 4.4.2: Realistic unsupervised toxicity detection for NASDAQ ITCH data
    Uses only publicly observable market microstructure data
    """
    
    def __init__(self):
        self.scalers = {}
        self.models = {}
        self.feature_names = []
        self.results = {}
        
    def load_nasdaq_data(self, message_file, orderbook_file):
        """
        Load NASDAQ ITCH data from the provided CSV files
        """
        print(f"Loading NASDAQ data from {message_file} and {orderbook_file}")
        
        # Load message data (orders)
        orders_df = pd.read_csv(message_file)
        print(f"Loaded {len(orders_df)} messages")
        print(f"Message columns: {list(orders_df.columns)}")
        print(f"Message types: {orders_df['Type'].value_counts()}")
        
        # Load orderbook data (LOB snapshots)
        lob_df = pd.read_csv(orderbook_file)
        print(f"Loaded {len(lob_df)} LOB snapshots")
        print(f"LOB columns: {list(lob_df.columns)}")
        
        # Clean and standardise column names
        orders_df = self._clean_message_data(orders_df)
        lob_df = self._clean_lob_data(lob_df)
        
        return orders_df, lob_df
    
    def _clean_message_data(self, orders_df):
        """Clean and standardise message data"""
        # Rename columns to match expected format
        column_mapping = {
            'Time': 'timestamp',
            'Type': 'message_type',
            'Order ID': 'order_id',
            'Size': 'quantity',
            'Price': 'price',
            'Direction': 'direction'
        }
        
        orders_df = orders_df.rename(columns=column_mapping)
        
        # Convert direction to standard format
        orders_df['side'] = orders_df['direction'].map({1: 'BUY', -1: 'SELL'})
        
        # Convert prices from cents to pounds (assuming NASDAQ data is in cents)
        if orders_df['price'].max() > 10000:  # Likely in cents
            orders_df['price'] = orders_df['price'] / 100.0
        
        # Add order type classification
        # Type 1 = Add order, Type 4 = Execute, Type 5 = Cancel, etc.
        orders_df['order_type'] = orders_df['message_type'].map({
            1: 'ADD',
            4: 'EXECUTE', 
            5: 'CANCEL',
            6: 'DELETE',
            7: 'REPLACE'
        }).fillna('OTHER')
        
        # Filter to only ADD orders for analysis (new order placement)
        orders_df = orders_df[orders_df['message_type'] == 1].copy()
        
        print(f"After filtering to ADD orders: {len(orders_df)} orders")
        
        return orders_df
    
    def _clean_lob_data(self, lob_df):
        # Ensure original columns are as expected
        expected_cols = ['Ask Price 1', 'Ask Size 1', 'Bid Price 1', 'Bid Size 1']
        for col in expected_cols:
            if col not in lob_df.columns:
                raise KeyError(f"Column '{col}' not found in LOB data!")

        # Rename columns explicitly and directly
        lob_df = lob_df.rename(columns={
            'Ask Price 1': 'ask_price_1',
            'Ask Size 1': 'ask_size_1',
            'Bid Price 1': 'bid_price_1',
            'Bid Size 1': 'bid_size_1',
            'Ask Price 2': 'ask_price_2',
            'Ask Size 2': 'ask_size_2',
            'Bid Price 2': 'bid_price_2',
            'Bid Size 2': 'bid_size_2',
            'Ask Price 3': 'ask_price_3',
            'Ask Size 3': 'ask_size_3',
            'Bid Price 3': 'bid_price_3',
            'Bid Size 3': 'bid_size_3',
            'Ask Price 4': 'ask_price_4',
            'Ask Size 4': 'ask_size_4',
            'Bid Price 4': 'bid_price_4',
            'Bid Size 4': 'bid_size_4',
            'Ask Price 5': 'ask_price_5',
            'Ask Size 5': 'ask_size_5',
            'Bid Price 5': 'bid_price_5',
            'Bid Size 5': 'bid_size_5',
            'Ask Price 6': 'ask_price_6',
            'Ask Size 6': 'ask_size_6',
            'Bid Price 6': 'bid_price_6',
            'Bid Size 6': 'bid_size_6',
            'Ask Price 7': 'ask_price_7',
            'Ask Size 7': 'ask_size_7',
            'Bid Price 7': 'bid_price_7',
            'Bid Size 7': 'bid_size_7',
            'Ask Price 8': 'ask_price_8',
            'Ask Size 8': 'ask_size_8',
            'Bid Price 8': 'bid_price_8',
            'Bid Size 8': 'bid_size_8',
            'Ask Price 9': 'ask_price_9',
            'Ask Size 9': 'ask_size_9',
            'Bid Price 9': 'bid_price_9',
            'Bid Size 9': 'bid_size_9',
            'Ask Price 10': 'ask_price_10',
            'Ask Size 10': 'ask_size_10',
            'Bid Price 10': 'bid_price_10',
            'Bid Size 10': 'bid_size_10'
        })

        # Confirm renaming
        print("Renamed columns:", lob_df.columns.tolist())

        # Calculate market metrics safely
        lob_df['mid_price'] = (lob_df['bid_price_1'] + lob_df['ask_price_1']) / 2
        lob_df['spread'] = lob_df['ask_price_1'] - lob_df['bid_price_1']
        lob_df['imbalance'] = (
            (lob_df['bid_size_1'] - lob_df['ask_size_1']) /
            (lob_df['bid_size_1'] + lob_df['ask_size_1'] + 1e-8)
        )

        lob_df['lob_sequence'] = range(len(lob_df))

        return lob_df

        
    def engineer_nasdaq_features(self, orders_df, lob_df):
        """
        Engineer features from NASDAQ ITCH data
        """
        print("Engineering features from NASDAQ ITCH data...")
        print(f"Input shapes: orders={orders_df.shape}, lob={lob_df.shape}")
        
        # Align orders with LOB snapshots by sequence
        # Each order gets the corresponding LOB snapshot
        min_length = min(len(orders_df), len(lob_df))
        orders_subset = orders_df.iloc[:min_length].copy()
        lob_subset = lob_df.iloc[:min_length].copy()
        
        print(f"Aligned data length: {min_length}")
        
        # Merge order data with corresponding LOB snapshot
        orders_subset.reset_index(drop=True, inplace=True)
        lob_subset.reset_index(drop=True, inplace=True)
        merged_df = pd.concat([orders_subset, lob_subset], axis=1)
        
        # Calculate market microstructure features
        features_df = self._calculate_nasdaq_features(merged_df)
        
        print(f"Final feature matrix shape: {features_df.shape}")
        print(f"Feature names: {list(features_df.columns)}")
        return features_df
    
    def _calculate_nasdaq_features(self, merged_df):
        """Calculate comprehensive features from NASDAQ data"""
        
        features = pd.DataFrame(index=merged_df.index)
        
        print(f"Available columns: {list(merged_df.columns)}")
        
        # 1. ORDER-LEVEL FEATURES
        
        # Order characteristics
        features['order_size'] = merged_df['quantity'].fillna(0)
        features['order_price'] = merged_df['price'].fillna(0)
        
        # Price relative to mid
        mid_price = merged_df['mid_price'].fillna(merged_df['price'])
        features['distance_from_mid'] = abs(merged_df['price'] - mid_price) / (mid_price + 1e-8)
        
        # Order aggressiveness
        spread = merged_df['spread'].fillna(0.01)
        features['relative_spread'] = spread / (mid_price + 1e-8)
        features['price_aggressiveness'] = features['distance_from_mid'] / (features['relative_spread'] + 1e-8)
        
        # Side encoding
        features['side_numeric'] = merged_df['side'].map({'BUY': 1, 'SELL': -1}).fillna(0)
        
        # 2. LOB-DERIVED FEATURES
        
        # Book imbalance at multiple levels
        for level in [1, 2, 3, 5]:
            bid_col = f'bid_size_{level}'
            ask_col = f'ask_size_{level}'
            if bid_col in merged_df.columns and ask_col in merged_df.columns:
                bid_size = merged_df[bid_col].fillna(0)
                ask_size = merged_df[ask_col].fillna(0)
                total_size = bid_size + ask_size
                features[f'imbalance_level_{level}'] = ((bid_size - ask_size) / 
                                                       (total_size + 1e-8))
            else:
                features[f'imbalance_level_{level}'] = 0
        
        # Total depth on each side
        features['total_bid_depth'] = 0
        features['total_ask_depth'] = 0
        
        for level in range(1, 11):  # Up to 10 levels available
            bid_col = f'bid_size_{level}'
            ask_col = f'ask_size_{level}'
            if bid_col in merged_df.columns:
                features['total_bid_depth'] += merged_df[bid_col].fillna(0)
            if ask_col in merged_df.columns:
                features['total_ask_depth'] += merged_df[ask_col].fillna(0)
        
        features['depth_ratio'] = (features['total_bid_depth'] / 
                                  (features['total_ask_depth'] + 1e-8))
        
        # Weighted mid price
        bid_price_1 = merged_df['bid_price_1'].fillna(0)
        ask_price_1 = merged_df['ask_price_1'].fillna(0)
        bid_size_1 = merged_df['bid_size_1'].fillna(1)
        ask_size_1 = merged_df['ask_size_1'].fillna(1)
        
        total_size_1 = bid_size_1 + ask_size_1
        features['weighted_mid_price'] = ((bid_price_1 * ask_size_1 + ask_price_1 * bid_size_1) / 
                                         (total_size_1 + 1e-8))
        
        # Price level concentration (Herfindahl-style)
        features['bid_concentration'] = self._calculate_concentration(merged_df, 'bid')
        features['ask_concentration'] = self._calculate_concentration(merged_df, 'ask')
        
        # 3. TEMPORAL FEATURES
        
        # Price momentum and volatility
        features['price_momentum_5'] = mid_price.pct_change(periods=5).fillna(0)
        features['price_momentum_10'] = mid_price.pct_change(periods=10).fillna(0)
        features['volatility_5'] = mid_price.rolling(5).std().fillna(0)
        features['volatility_20'] = mid_price.rolling(20).std().fillna(0)
        
        # Spread dynamics
        features['spread_momentum'] = spread.pct_change(periods=5).fillna(0)
        features['spread_volatility'] = spread.rolling(10).std().fillna(0)
        
        # Order flow imbalance
        features['order_flow_imbalance_5'] = features['side_numeric'].rolling(5).sum().fillna(0)
        features['order_flow_imbalance_10'] = features['side_numeric'].rolling(10).sum().fillna(0)
        features['order_flow_imbalance_20'] = features['side_numeric'].rolling(20).sum().fillna(0)
        
        # 4. INTERACTION FEATURES
        
        # Size relative to market depth
        total_depth = features['total_bid_depth'] + features['total_ask_depth']
        features['size_vs_depth'] = features['order_size'] / (total_depth + 1)
        
        # Order intensity (based on timestamp differences)
        if 'timestamp' in merged_df.columns:
            time_diff = merged_df['timestamp'].diff().fillna(0)
            features['time_since_last_order'] = time_diff
            features['order_arrival_rate'] = 1 / (time_diff + 1e-8)
        else:
            features['time_since_last_order'] = 1.0  # Constant for sequence data
            features['order_arrival_rate'] = 1.0
        
        # 5. ADVANCED MARKET MICROSTRUCTURE FEATURES
        
        # Effective spread
        features['effective_spread'] = 2 * features['distance_from_mid']
        
        # Market impact proxy
        features['market_impact_proxy'] = (features['order_size'] * features['price_aggressiveness'] / 
                                          (total_depth + 1))
        
        # Liquidity consumption rate
        features['liquidity_consumption'] = features['order_size'] / (merged_df['ask_size_1'].fillna(0) + merged_df['bid_size_1'].fillna(0) + 1)

        
        # Price pressure
        side_adjusted_size = features['order_size'] * features['side_numeric']
        features['price_pressure'] = side_adjusted_size.rolling(10).sum() / (total_depth + 1)
        
        # Relative order size
        typical_size = features['order_size'].rolling(50).median().fillna(features['order_size'].median())
        features['relative_order_size'] = features['order_size'] / (typical_size + 1)
        
        # Book shape features
        features['ask_slope'] = self._calculate_book_slope(merged_df, 'ask')
        features['bid_slope'] = self._calculate_book_slope(merged_df, 'bid')
        
        # 6. REGIME INDICATORS
        
        # Volatility regime
        vol_ma = features['volatility_20'].rolling(50).mean()
        features['volatility_regime'] = (features['volatility_20'] / (vol_ma + 1e-8)).fillna(1)
        
        # Spread regime
        spread_ma = features['relative_spread'].rolling(50).mean()
        features['spread_regime'] = (features['relative_spread'] / (spread_ma + 1e-8)).fillna(1)
        
        # Clean and return features
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        print(f"Generated {len(features.columns)} features")
        return features
    
    def _calculate_concentration(self, merged_df, side):
        """Calculate liquidity concentration (Herfindahl-style)"""
        concentration = pd.Series(0.0, index=merged_df.index)
        
        # Calculate total depth for normalisation
        total_depth = pd.Series(0.0, index=merged_df.index)
        for level in range(1, 11):
            size_col = f'{side}_size_{level}'
            if size_col in merged_df.columns:
                total_depth += merged_df[size_col].fillna(0)
        
        # Calculate concentration
        for level in range(1, 11):
            size_col = f'{side}_size_{level}'
            if size_col in merged_df.columns:
                share = merged_df[size_col].fillna(0) / (total_depth + 1e-8)
                concentration += share ** 2
        
        return concentration
    
    def _calculate_book_slope(self, merged_df, side):
        """Calculate order book slope (price impact per unit depth)"""
        slopes = pd.Series(0.0, index=merged_df.index)
        
        # Use first 5 levels to calculate slope
        prices = []
        depths = []
        
        for level in range(1, 6):
            price_col = f'{side}_price_{level}'
            size_col = f'{side}_size_{level}'
            
            if price_col in merged_df.columns and size_col in merged_df.columns:
                prices.append(merged_df[price_col].fillna(0))
                if level == 1:
                    depths.append(merged_df[size_col].fillna(0))
                else:
                    depths.append(depths[-1] + merged_df[size_col].fillna(0))
        
        if len(prices) >= 2:
            # Calculate slope between first and last level
            price_diff = abs(prices[-1] - prices[0])
            depth_diff = depths[-1] - depths[0]
            slopes = price_diff / (depth_diff + 1e-8)
        
        return slopes.fillna(0)
    
    def implement_clustering(self, features_df):
        """
        Implement clustering on engineered features
        """
        print("Implementing clustering on NASDAQ features...")
        
        # Select uncorrelated features
        selected_features = self._select_features(features_df)
        X = selected_features.values
        
        # Robust scaling
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['clustering'] = scaler
        self.feature_names = selected_features.columns.tolist()
        
        # K-means clustering
        kmeans_results = self._optimize_kmeans(X_scaled)
        
        # DBSCAN clustering
        dbscan_results = self._optimize_dbscan(X_scaled)
        
        return kmeans_results, dbscan_results, selected_features
    
    def _select_features(self, features_df, correlation_threshold=0.9):
        """Select features with reasonable variance and low correlation"""
        
        # Remove constant features
        feature_vars = features_df.var()
        variable_features = features_df.loc[:, feature_vars > 1e-8]
        
        if len(variable_features.columns) == 0:
            print("Warning: No variable features found, using all features")
            return features_df
        
        # Remove highly correlated features
        corr_matrix = variable_features.corr().abs()
        
        # Find features to remove
        features_to_remove = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    # Remove feature with lower variance
                    feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    if variable_features[feat1].var() > variable_features[feat2].var():
                        features_to_remove.add(feat2)
                    else:
                        features_to_remove.add(feat1)
        
        selected_features = variable_features.drop(columns=list(features_to_remove))
        print(f"Selected {len(selected_features.columns)} features from {len(features_df.columns)}")
        
        return selected_features
    
    def _optimize_kmeans(self, X_scaled):
        """Optimize K-means parameters"""
        results = {}
        max_k = min(15, len(X_scaled) // 20, len(X_scaled))
        k_range = range(2, max_k + 1)
        
        best_score = -1
        best_k = None
        
        for k in k_range:
            if k >= len(X_scaled):
                continue
                
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            labels = kmeans.fit_predict(X_scaled)
            
            if len(set(labels)) > 1:
                try:
                    silhouette = silhouette_score(X_scaled, labels)
                    davies_bouldin = davies_bouldin_score(X_scaled, labels)
                    
                    results[k] = {
                        'model': kmeans,
                        'labels': labels,
                        'silhouette': silhouette,
                        'davies_bouldin': davies_bouldin,
                        'inertia': kmeans.inertia_
                    }
                    
                    if silhouette > best_score:
                        best_score = silhouette
                        best_k = k
                        
                except Exception as e:
                    print(f"Error with k={k}: {e}")
                    continue
        
        if best_k:
            self.models['kmeans'] = results[best_k]['model']
            print(f"Optimal K-means: k={best_k}, Silhouette={results[best_k]['silhouette']:.3f}")
        
        return results
    
    def _optimize_dbscan(self, X_scaled):
        """Optimize DBSCAN parameters"""
        results = {}
        
        # Parameter ranges suitable for financial data
        eps_values = np.linspace(0.1, 2.0, 10)
        min_samples_values = [5, 10, 15, 20]
        
        best_score = -1
        best_params = None
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                if min_samples >= len(X_scaled):
                    continue
                    
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(X_scaled)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters >= 2:
                        non_noise_mask = labels != -1
                        if non_noise_mask.sum() > min_samples:
                            silhouette = silhouette_score(X_scaled[non_noise_mask], 
                                                         labels[non_noise_mask])
                            noise_ratio = (labels == -1).mean()
                            adjusted_score = silhouette * (1 - noise_ratio * 0.3)
                            
                            key = f"eps_{eps:.2f}_min_{min_samples}"
                            results[key] = {
                                'model': dbscan,
                                'labels': labels,
                                'silhouette': silhouette,
                                'adjusted_score': adjusted_score,
                                'n_clusters': n_clusters,
                                'noise_ratio': noise_ratio,
                                'eps': eps,
                                'min_samples': min_samples
                            }
                            
                            if adjusted_score > best_score:
                                best_score = adjusted_score
                                best_params = (eps, min_samples)
                                
                except Exception as e:
                    continue
        
        if best_params:
            best_key = f"eps_{best_params[0]:.2f}_min_{best_params[1]}"
            self.models['dbscan'] = results[best_key]['model']
            print(f"Optimal DBSCAN: eps={best_params[0]:.2f}, min_samples={best_params[1]}")
            print(f"Clusters: {results[best_key]['n_clusters']}, Noise: {results[best_key]['noise_ratio']:.1%}")
        
        return results

class NASDAQAnomalyDetector:
    """
    4.4.3: Anomaly detection for NASDAQ data
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
    
    def implement_anomaly_detection(self, features_df, contamination=0.05):
        """
        Implement ensemble anomaly detection
        """
        print("Implementing anomaly detection on NASDAQ features...")
        
        # Prepare features
        X = features_df.values
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['ensemble'] = scaler
        
        # Ensemble of detectors
        detectors = {
            'isolation_forest': IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto'
            ),
            'lof': LocalOutlierFactor(
                n_neighbors=min(20, len(X_scaled) // 5),
                contamination=contamination,
                novelty=True
            )
        }
        
        # Fit and score
        anomaly_scores = {}
        
        for name, detector in detectors.items():
            try:
                detector.fit(X_scaled)
                
                if name == 'lof':
                    scores = -detector.score_samples(X_scaled)
                else:
                    scores = -detector.decision_function(X_scaled)
                
                # Normalise scores
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                anomaly_scores[name] = scores
                self.models[name] = detector
                
            except Exception as e:
                print(f"Error with {name}: {e}")
                continue
        
        if not anomaly_scores:
            print("Warning: No anomaly detectors succeeded")
            return np.zeros(len(features_df)), np.zeros(len(features_df)), {}
        
        # Ensemble scoring
        ensemble_scores = np.mean([anomaly_scores[name] for name in anomaly_scores], axis=0)
        
        # Determine anomalies
        threshold = np.percentile(ensemble_scores, (1 - contamination) * 100)
        anomaly_labels = (ensemble_scores > threshold).astype(int)
        
        self.thresholds['ensemble'] = threshold
        
        print(f"Detected {anomaly_labels.sum()} anomalies ({anomaly_labels.mean()*100:.1f}%)")
        
        return ensemble_scores, anomaly_labels, anomaly_scores

def evaluate_nasdaq_detection(features_df, cluster_labels, anomaly_scores):
    """
    Evaluate detection using market-based metrics
    """
    print("Evaluating NASDAQ toxicity detection...")
    
    results = {}
    
    # Price impact evaluation
    if 'market_impact_proxy' in features_df.columns:
        impact = features_df['market_impact_proxy']
        anomaly_threshold = np.percentile(anomaly_scores, 95)
        anomaly_mask = anomaly_scores > anomaly_threshold
        
        if anomaly_mask.sum() > 0:
            results['impact_analysis'] = {
                'anomaly_avg_impact': impact[anomaly_mask].mean(),
                'normal_avg_impact': impact[~anomaly_mask].mean(),
                'impact_ratio': impact[anomaly_mask].mean() / (impact[~anomaly_mask].mean() + 1e-8)
            }
    
    # Volatility analysis
    if 'volatility_20' in features_df.columns:
        volatility = features_df['volatility_20']
        correlation = np.corrcoef(anomaly_scores, volatility)[0, 1] if len(set(volatility)) > 1 else 0
        
        results['volatility_analysis'] = {
            'anomaly_volatility_correlation': correlation
        }
    
    # Size analysis
    if 'relative_order_size' in features_df.columns:
        rel_size = features_df['relative_order_size']
        anomaly_threshold = np.percentile(anomaly_scores, 95)
        anomaly_mask = anomaly_scores > anomaly_threshold
        
        if anomaly_mask.sum() > 0:
            results['size_analysis'] = {
                'anomaly_avg_size': rel_size[anomaly_mask].mean(),
                'normal_avg_size': rel_size[~anomaly_mask].mean(),
                'size_ratio': rel_size[anomaly_mask].mean() / (rel_size[~anomaly_mask].mean() + 1e-8)
            }
    
    # Aggressiveness analysis
    if 'price_aggressiveness' in features_df.columns:
        aggressiveness = features_df['price_aggressiveness']
        anomaly_threshold = np.percentile(anomaly_scores, 95)
        anomaly_mask = anomaly_scores > anomaly_threshold
        
        if anomaly_mask.sum() > 0:
            results['aggressiveness_analysis'] = {
                'anomaly_avg_aggr': aggressiveness[anomaly_mask].mean(),
                'normal_avg_aggr': aggressiveness[~anomaly_mask].mean(),
                'aggr_ratio': aggressiveness[anomaly_mask].mean() / (aggressiveness[~anomaly_mask].mean() + 1e-8)
            }
    
    return results

def plot_nasdaq_results(features_df, cluster_labels, anomaly_scores, evaluation_results):
    """Generate visualizations for NASDAQ results"""
    
    # Create results directory
    save_dir = "nasdaq_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. PCA visualization
    if len(features_df.columns) > 1:
        pca = PCA(n_components=2)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Clustering results
        if cluster_labels is not None:
            unique_labels = set(cluster_labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = cluster_labels == label
                label_name = 'Noise' if label == -1 else f'Cluster {label}'
                marker = 'x' if label == -1 else 'o'
                axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                               c=[color], label=label_name, alpha=0.7, marker=marker)
            
            axes[0].set_title('NASDAQ Order Clusters')
            axes[0].legend()
        
        # Anomaly detection
        anomaly_threshold = np.percentile(anomaly_scores, 95)
        anomaly_mask = anomaly_scores > anomaly_threshold
        
        axes[1].scatter(X_pca[~anomaly_mask, 0], X_pca[~anomaly_mask, 1], 
                       c='lightblue', label='Normal Orders', alpha=0.5)
        if anomaly_mask.sum() > 0:
            axes[1].scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], 
                           c='red', label='Anomalous Orders', alpha=0.8, marker='D')
        
        axes[1].set_title('NASDAQ Anomaly Detection')
        axes[1].legend()
        
        for ax in axes:
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/nasdaq_pca_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # 2. Feature distributions
    key_features = ['order_size', 'price_aggressiveness', 'relative_spread', 'imbalance_level_1']
    available_features = [f for f in key_features if f in features_df.columns]
    
    if available_features:
        n_features = len(available_features)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        anomaly_threshold = np.percentile(anomaly_scores, 95)
        anomaly_mask = anomaly_scores > anomaly_threshold
        
        for i, feature in enumerate(available_features[:4]):
            if i < len(axes):
                normal_data = features_df[feature][~anomaly_mask]
                anomaly_data = features_df[feature][anomaly_mask]
                
                axes[i].hist(normal_data, bins=30, alpha=0.6, label='Normal Orders', 
                            color='lightblue', density=True)
                if len(anomaly_data) > 0:
                    axes[i].hist(anomaly_data, bins=20, alpha=0.8, label='Anomalous Orders', 
                                color='red', density=True)
                
                axes[i].set_xlabel(feature.replace('_', ' ').title())
                axes[i].set_ylabel('Density')
                axes[i].set_title(f'{feature.replace("_", " ").title()} Distribution')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/nasdaq_feature_distributions_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. Anomaly scores over time
    plt.figure(figsize=(15, 6))
    order_sequence = np.arange(len(anomaly_scores))
    
    plt.plot(order_sequence, anomaly_scores, alpha=0.7, color='blue', linewidth=0.5)
    anomaly_threshold = np.percentile(anomaly_scores, 95)
    plt.axhline(anomaly_threshold, color='red', linestyle='--', 
               label=f'Anomaly Threshold (95th percentile)')
    
    plt.xlabel('Order Sequence')
    plt.ylabel('Anomaly Score')
    plt.title('NASDAQ Anomaly Scores Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/nasdaq_anomaly_timeline_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Market microstructure patterns
    if 'mid_price' in features_df.columns or any('price' in col for col in features_df.columns):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price patterns
        if 'mid_price' in features_df.columns:
            axes[0, 0].plot(features_df['mid_price'], alpha=0.7)
            axes[0, 0].set_title('Mid Price Evolution')
            axes[0, 0].set_ylabel('Price (£)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Spread patterns
        if 'relative_spread' in features_df.columns:
            axes[0, 1].plot(features_df['relative_spread'], alpha=0.7, color='green')
            axes[0, 1].set_title('Relative Spread Evolution')
            axes[0, 1].set_ylabel('Relative Spread')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Imbalance patterns
        if 'imbalance_level_1' in features_df.columns:
            axes[1, 0].plot(features_df['imbalance_level_1'], alpha=0.7, color='purple')
            axes[1, 0].set_title('Order Book Imbalance (Level 1)')
            axes[1, 0].set_ylabel('Imbalance')
            axes[1, 0].axhline(0, color='black', linestyle='-', alpha=0.5)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Volatility patterns
        if 'volatility_20' in features_df.columns:
            axes[1, 1].plot(features_df['volatility_20'], alpha=0.7, color='orange')
            axes[1, 1].set_title('20-Period Volatility')
            axes[1, 1].set_ylabel('Volatility')
            axes[1, 1].grid(True, alpha=0.3)
        
        for ax in axes.flatten():
            ax.set_xlabel('Order Sequence')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/nasdaq_microstructure_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()

def print_nasdaq_evaluation(evaluation_results):
    """Print evaluation results for NASDAQ data"""
    
    print("\nNASDAQ EVALUATION RESULTS:")
    print("=" * 50)
    
    if 'impact_analysis' in evaluation_results:
        impact = evaluation_results['impact_analysis']
        print(f"\nMarket Impact Analysis:")
        print(f"  Anomalous orders avg impact: {impact['anomaly_avg_impact']:.6f}")
        print(f"  Normal orders avg impact: {impact['normal_avg_impact']:.6f}")
        print(f"  Impact ratio: {impact['impact_ratio']:.2f}x")
    
    if 'volatility_analysis' in evaluation_results:
        vol = evaluation_results['volatility_analysis']
        print(f"\nVolatility Analysis:")
        print(f"  Anomaly-volatility correlation: {vol['anomaly_volatility_correlation']:.3f}")
    
    if 'size_analysis' in evaluation_results:
        size = evaluation_results['size_analysis']
        print(f"\nOrder Size Analysis:")
        print(f"  Anomalous orders avg size: {size['anomaly_avg_size']:.3f}")
        print(f"  Normal orders avg size: {size['normal_avg_size']:.3f}")
        print(f"  Size ratio: {size['size_ratio']:.2f}x")
    
    if 'aggressiveness_analysis' in evaluation_results:
        aggr = evaluation_results['aggressiveness_analysis']
        print(f"\nPrice Aggressiveness Analysis:")
        print(f"  Anomalous orders avg aggressiveness: {aggr['anomaly_avg_aggr']:.3f}")
        print(f"  Normal orders avg aggressiveness: {aggr['normal_avg_aggr']:.3f}")
        print(f"  Aggressiveness ratio: {aggr['aggr_ratio']:.2f}x")

def save_nasdaq_models(detector, anomaly_detector, features_df, evaluation_results, 
                      anomaly_scores, cluster_labels):
    """Save models and results"""
    
    save_dir = "nasdaq_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Results summary
    results_summary = {
        'timestamp': timestamp,
        'data_source': 'NASDAQ_ITCH',
        'n_features': len(features_df.columns),
        'n_samples': len(features_df),
        'feature_names': features_df.columns.tolist(),
        'anomaly_rate': (anomaly_scores > np.percentile(anomaly_scores, 95)).mean(),
        'evaluation_results': evaluation_results
    }
    
    if cluster_labels is not None:
        results_summary['n_clusters'] = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        results_summary['noise_ratio'] = (cluster_labels == -1).mean() if -1 in cluster_labels else 0
    
    # Save data
    save_data = {
        'toxicity_detector': {
            'models': detector.models,
            'scalers': detector.scalers,
            'feature_names': detector.feature_names
        },
        'anomaly_detector': {
            'models': anomaly_detector.models,
            'scalers': anomaly_detector.scalers,
            'thresholds': anomaly_detector.thresholds
        },
        'results_summary': results_summary,
        'anomaly_scores': anomaly_scores,
        'cluster_labels': cluster_labels,
        'features_df': features_df
    }
    
    joblib.dump(save_data, f"{save_dir}/nasdaq_models_{timestamp}.joblib")
    
    print(f"\nModels saved to: {save_dir}/nasdaq_models_{timestamp}.joblib")
    print(f"Features: {results_summary['n_features']}")
    print(f"Samples: {results_summary['n_samples']}")
    print(f"Anomaly rate: {results_summary['anomaly_rate']:.1%}")
    
    if 'n_clusters' in results_summary:
        print(f"Clusters: {results_summary['n_clusters']}")

def main_nasdaq_pipeline():
    """
    Main pipeline for NASDAQ ITCH toxicity detection
    """
    print("="*80)
    print("NASDAQ ITCH REALISTIC TOXICITY DETECTION")
    print("Using publicly observable NASDAQ market data")
    print("="*80)
    
    # File paths
    message_file = "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    orderbook_file = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    
    # Check files exist
    if not os.path.exists(message_file):
        print(f"Error: Message file {message_file} not found")
        return None, None, None
    
    if not os.path.exists(orderbook_file):
        print(f"Error: Orderbook file {orderbook_file} not found")
        return None, None, None
    
    # Initialize detectors
    detector = NASDAQToxicityDetector()
    anomaly_detector = NASDAQAnomalyDetector()
    
    # Load NASDAQ data
    print("\n4.4.1: LOADING NASDAQ ITCH DATA")
    print("-" * 35)
    orders_df, lob_df = detector.load_nasdaq_data(message_file, orderbook_file)
    
    # Engineer features
    print("\n4.4.2: FEATURE ENGINEERING FROM NASDAQ DATA")
    print("-" * 45)
    features_df = detector.engineer_nasdaq_features(orders_df, lob_df)
    
    if len(features_df) == 0:
        print("Error: No features generated")
        return None, None, None
    
    # Implement clustering
    print(f"\n4.4.3: CLUSTERING NASDAQ ORDER BEHAVIOUR")
    print("-" * 40)
    kmeans_results, dbscan_results, selected_features = detector.implement_clustering(features_df)
    
    # Get best clustering labels
    cluster_labels = None
    if kmeans_results:
        best_k = max(kmeans_results.keys(), key=lambda k: kmeans_results[k]['silhouette'])
        cluster_labels = kmeans_results[best_k]['labels']
        print(f"Using K-means clustering with {best_k} clusters")
    
    # Implement anomaly detection
    print(f"\n4.4.4: ANOMALY DETECTION ON NASDAQ ORDERS")
    print("-" * 40)
    anomaly_scores, anomaly_labels, individual_scores = anomaly_detector.implement_anomaly_detection(
        selected_features, contamination=0.05
    )
    
    # Evaluation
    print(f"\n4.4.5: EVALUATION USING MARKET METRICS")
    print("-" * 37)
    evaluation_results = evaluate_nasdaq_detection(selected_features, cluster_labels, anomaly_scores)
    
    # Print results
    print_nasdaq_evaluation(evaluation_results)
    
    # Generate visualizations
    print("\n4.4.6: VISUALIZATION OF RESULTS")
    print("-" * 32)
    plot_nasdaq_results(selected_features, cluster_labels, anomaly_scores, evaluation_results)
    
    # Save results
    save_nasdaq_models(detector, anomaly_detector, selected_features, evaluation_results,
                      anomaly_scores, cluster_labels)
    
    print("\n" + "="*80)
    print("NASDAQ ITCH TOXICITY DETECTION COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return detector, anomaly_detector, evaluation_results

if __name__ == "__main__":
    try:
        detector, anomaly_detector, results = main_nasdaq_pipeline()
        
        if detector is not None:
            print("\nNASDAQ pipeline completed successfully!")
            print("\nKey Features:")
            print("- Adapted for NASDAQ ITCH message format")
            print("- Uses only publicly observable market data")
            print("- Comprehensive microstructure feature engineering")
            print("- Ensemble anomaly detection")
            print("- Market-based evaluation metrics")
            print("- Production-ready for live NASDAQ data")
        else:
            print("Pipeline failed - check file paths and data format")
        
    except Exception as e:
        print(f"Error in NASDAQ pipeline: {e}")
        import traceback
        traceback.print_exc()