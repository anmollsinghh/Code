"""
Enhanced Market Toxicity Detection System - Advanced Technical Improvements
Includes ensemble calibration, feature importance aggregation, dimensionality reduction,
real-time optimization, synthetic anomaly injection, and sequence modeling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import OneClassSVM
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from scipy import stats
from scipy.spatial.distance import cdist
import joblib
import glob
import os
import optuna
from optuna.samplers import TPESampler
import numba
from numba import jit
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    try:
        from umap import UMAP
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False
        print("UMAP not available - UMAP embedding features disabled")

# Try to import advanced libraries (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - LSTM autoencoder features disabled")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available - SHAP feature importance disabled")

class AdvancedFeatureEngineer:
    """Enhanced feature engineering with real-time optimization"""
    
    def __init__(self):
        self.feature_names = []
        self.feature_importance_scores = {}
        self.incremental_stats = {}
        
    @staticmethod
    @jit(nopython=True)
    def welford_update(mean, var, count, new_value):
        """Numba-optimized incremental statistics using Welford's method"""
        count += 1
        delta = new_value - mean
        mean += delta / count
        delta2 = new_value - mean
        var += delta * delta2
        return mean, var, count
    
    @staticmethod
    @jit(nopython=True)
    def rolling_stats_optimized(values, window):
        """Optimized rolling statistics computation"""
        n = len(values)
        means = np.zeros(n)
        stds = np.zeros(n)
        
        for i in range(n):
            start_idx = max(0, i - window + 1)
            window_values = values[start_idx:i+1]
            means[i] = np.mean(window_values)
            stds[i] = np.std(window_values)
            
        return means, stds
    
    def load_market_data(self, data_dir="enhanced_market_data"):
        """Load market data with enhanced validation"""
        print(f"Loading market data from {data_dir}...")
        
        order_files = glob.glob(f"{data_dir}/orders_*.csv")
        lob_files = glob.glob(f"{data_dir}/lob_snapshots_*.csv")
        trade_files = glob.glob(f"{data_dir}/trades_*.csv")
        
        if not order_files:
            raise FileNotFoundError(f"No order files found in {data_dir}")
        
        # Load most recent files
        latest_order_file = max(order_files, key=os.path.getctime)
        orders_df = pd.read_csv(latest_order_file)
        print(f"Orders data: {len(orders_df)} records from {os.path.basename(latest_order_file)}")
        
        lob_df = pd.DataFrame()
        if lob_files:
            latest_lob_file = max(lob_files, key=os.path.getctime)
            lob_df = pd.read_csv(latest_lob_file)
            print(f"LOB data: {len(lob_df)} snapshots from {os.path.basename(latest_lob_file)}")
        
        trades_df = pd.DataFrame()
        if trade_files:
            latest_trade_file = max(trade_files, key=os.path.getctime)
            trades_df = pd.read_csv(latest_trade_file)
            print(f"Trades data: {len(trades_df)} trades from {os.path.basename(latest_trade_file)}")
        
        return orders_df, lob_df, trades_df
    
    def extract_optimized_features(self, orders_df, lob_df, trades_df):
        """Extract features with real-time optimizations"""
        print("Extracting optimized features for real-time deployment...")
        
        features_df = pd.DataFrame(index=orders_df.index)
        
        # Basic features (optimized)
        features_df['order_size'] = orders_df['quantity']
        features_df['log_order_size'] = np.log1p(orders_df['quantity'])
        features_df['order_size_percentile'] = orders_df['quantity'].rank(pct=True)
        features_df['is_market_order'] = (orders_df['order_type'] == 'MARKET').astype(int)
        features_df['is_buy'] = (orders_df['side'] == 'BUY').astype(int)
        
        # Optimized rolling features using Numba
        if 'quantity' in orders_df.columns:
            values = orders_df['quantity'].values
            for window in [5, 10, 20]:
                means, stds = self.rolling_stats_optimized(values, window)
                features_df[f'order_size_ma_{window}'] = means
                features_df[f'order_size_std_{window}'] = stds
                features_df[f'order_size_zscore_{window}'] = (values - means) / (stds + 1e-8)
        
        # Price features
        if 'mid_price' in orders_df.columns:
            mid_price = orders_df['mid_price']
            features_df['mid_price'] = mid_price
            features_df['log_mid_price'] = np.log(mid_price)
            features_df['mid_price_returns'] = mid_price.pct_change().fillna(0)
            
            # Optimized price momentum
            for window in [5, 10, 20]:
                price_values = mid_price.values
                means, stds = self.rolling_stats_optimized(price_values, window)
                features_df[f'price_ma_{window}'] = means
                features_df[f'price_volatility_{window}'] = stds
                features_df[f'price_momentum_{window}'] = (price_values - means) / (means + 1e-8)
        
        # Enhanced timing features
        if 'timestamp' in orders_df.columns:
            timestamps = pd.to_datetime(orders_df['timestamp']) if not pd.api.types.is_datetime64_any_dtype(orders_df['timestamp']) else orders_df['timestamp']
            
            # Optimized inter-arrival times
            time_diffs = timestamps.diff().dt.total_seconds().fillna(1) if hasattr(timestamps.iloc[0], 'hour') else pd.Series(range(len(timestamps))).diff().fillna(1)
            features_df['inter_arrival_time'] = time_diffs
            features_df['log_inter_arrival_time'] = np.log1p(time_diffs)
            features_df['arrival_rate'] = 1 / (time_diffs + 1e-8)
            
            # Optimized arrival intensity
            arrival_values = features_df['arrival_rate'].values
            for window in [5, 10, 20]:
                means, stds = self.rolling_stats_optimized(arrival_values, window)
                features_df[f'arrival_intensity_{window}'] = means
                features_df[f'arrival_volatility_{window}'] = stds
        
        # Market microstructure features
        for feature_name in ['volatility', 'momentum', 'order_book_imbalance']:
            if feature_name in orders_df.columns:
                feature_values = orders_df[feature_name].values
                features_df[feature_name] = feature_values
                features_df[f'abs_{feature_name}'] = np.abs(feature_values)
                features_df[f'{feature_name}_percentile'] = orders_df[feature_name].rank(pct=True)
                
                # Optimized rolling statistics
                for window in [5, 10, 20]:
                    means, stds = self.rolling_stats_optimized(feature_values, window)
                    features_df[f'{feature_name}_ma_{window}'] = means
                    features_df[f'{feature_name}_std_{window}'] = stds
        
        # Enhanced LOB features
        if not lob_df.empty:
            lob_features = self._extract_optimized_lob_features(lob_df, orders_df)
            for col in lob_features.columns:
                if col not in features_df.columns:
                    features_df[col] = lob_features[col]
        
        # Enhanced trade features (optimized)
        if not trades_df.empty:
            trade_features = self._extract_optimized_trade_features(trades_df, orders_df)
            for col in trade_features.columns:
                if col not in features_df.columns:
                    features_df[col] = trade_features[col]
        
        # Sequence features for LSTM
        features_df = self._add_sequence_features(features_df)
        
        # Synthetic anomaly features for evaluation
        features_df = self._add_synthetic_anomaly_features(features_df)
        
        # Clean up
        features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)
        self.feature_names = features_df.columns.tolist()
        
        print(f"Generated {len(self.feature_names)} optimized features")
        return features_df
    
    def _extract_optimized_lob_features(self, lob_df, orders_df):
        """Extract LOB features with optimization"""
        lob_features = pd.DataFrame(index=orders_df.index)
        
        if 'timestamp' in lob_df.columns and 'timestamp' in orders_df.columns:
            merged = pd.merge_asof(
                orders_df[['timestamp']].reset_index(),
                lob_df,
                on='timestamp',
                direction='backward'
            ).set_index('index')
            
            # Vectorized depth calculations
            bid_cols = [f'bid_size_{i}' for i in range(1, 6) if f'bid_size_{i}' in merged.columns]
            ask_cols = [f'ask_size_{i}' for i in range(1, 6) if f'ask_size_{i}' in merged.columns]
            
            if bid_cols and ask_cols:
                total_bid_depth = merged[bid_cols].fillna(0).sum(axis=1)
                total_ask_depth = merged[ask_cols].fillna(0).sum(axis=1)
                
                lob_features['total_bid_depth'] = total_bid_depth
                lob_features['total_ask_depth'] = total_ask_depth
                lob_features['depth_imbalance'] = (total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth + 1e-8)
                lob_features['depth_ratio'] = total_bid_depth / (total_ask_depth + 1e-8)
                
                # Level-specific features (optimized)
                for i, (bid_col, ask_col) in enumerate(zip(bid_cols, ask_cols), 1):
                    bid_size = merged[bid_col].fillna(0)
                    ask_size = merged[ask_col].fillna(0)
                    lob_features[f'imbalance_L{i}'] = (bid_size - ask_size) / (bid_size + ask_size + 1e-8)
        
        return lob_features.fillna(0)
    
    def _extract_optimized_trade_features(self, trades_df, orders_df):
        """Extract trade features with optimization"""
        trade_features = pd.DataFrame(index=orders_df.index)
        
        # Vectorized approach for trade feature calculation
        if 'timestamp' in trades_df.columns and 'timestamp' in orders_df.columns:
            try:
                orders_ts = pd.to_datetime(orders_df['timestamp'])
                trades_ts = pd.to_datetime(trades_df['timestamp'])
                
                # Sample for performance (optimized selection)
                sample_size = min(len(orders_df), 500)  # Reduced for optimization
                sample_indices = np.random.choice(len(orders_df), sample_size, replace=False)
                
                for idx in sample_indices:
                    order_time = orders_ts.iloc[idx]
                    
                    # Vectorized time window calculations
                    for window_seconds in [30, 60]:  # Reduced windows for optimization
                        start_time = order_time - pd.Timedelta(seconds=window_seconds)
                        mask = (trades_ts >= start_time) & (trades_ts <= order_time)
                        recent_trades = trades_df[mask]
                        
                        if not recent_trades.empty and 'quantity' in recent_trades.columns:
                            trade_features.loc[idx, f'trade_count_{window_seconds}'] = len(recent_trades)
                            trade_features.loc[idx, f'trade_volume_{window_seconds}'] = recent_trades['quantity'].sum()
                            
                            if len(recent_trades) > 1:
                                trade_features.loc[idx, f'trade_size_std_{window_seconds}'] = recent_trades['quantity'].std()
            except Exception:
                pass
        
        return trade_features.fillna(0)
    
    def _add_sequence_features(self, features_df):
        """Add features for sequence modeling"""
        # Sequence patterns for LSTM autoencoder
        if 'order_size' in features_df.columns:
            # Size burst patterns
            size_values = features_df['order_size'].values
            for window in [3, 5]:
                # Rolling coefficient of variation
                means, stds = self.rolling_stats_optimized(size_values, window)
                features_df[f'size_cv_{window}'] = stds / (means + 1e-8)
                
                # Momentum patterns
                if window > 2:
                    momentum = np.gradient(means)
                    features_df[f'size_momentum_{window}'] = np.concatenate([np.zeros(window-1), momentum[window-1:]])
        
        # Arrival pattern sequences
        if 'arrival_rate' in features_df.columns:
            arrival_values = features_df['arrival_rate'].values
            for window in [3, 5]:
                means, stds = self.rolling_stats_optimized(arrival_values, window)
                features_df[f'arrival_burst_{window}'] = (arrival_values > means + 2 * stds).astype(int)
        
        return features_df
    
    def _add_synthetic_anomaly_features(self, features_df):
        """Add synthetic anomaly indicators for evaluation"""
        n_samples = len(features_df)
        
        # Synthetic spoofing pattern
        features_df['synthetic_spoofing'] = 0
        if 'order_size' in features_df.columns and 'inter_arrival_time' in features_df.columns:
            # Large orders with rapid succession
            large_orders = features_df['order_size'] > features_df['order_size'].quantile(0.9)
            fast_arrival = features_df['inter_arrival_time'] < features_df['inter_arrival_time'].quantile(0.1)
            spoofing_pattern = large_orders & fast_arrival
            features_df['synthetic_spoofing'] = spoofing_pattern.astype(int)
        
        # Synthetic momentum ignition
        features_df['synthetic_momentum_ignition'] = 0
        if 'momentum' in features_df.columns and 'order_size' in features_df.columns:
            high_momentum = abs(features_df['momentum']) > features_df['momentum'].abs().quantile(0.95)
            large_size = features_df['order_size'] > features_df['order_size'].quantile(0.8)
            momentum_ignition = high_momentum & large_size
            features_df['synthetic_momentum_ignition'] = momentum_ignition.astype(int)
        
        # Synthetic layering pattern
        features_df['synthetic_layering'] = 0
        if 'order_size' in features_df.columns:
            # Multiple similar-sized orders in succession
            size_similarity = features_df['order_size'].rolling(5, min_periods=1).std()
            low_variance = size_similarity < size_similarity.quantile(0.2)
            features_df['synthetic_layering'] = low_variance.astype(int)
        
        return features_df

class LSTMAutoencoder:
    """LSTM Autoencoder for sequence anomaly detection"""
    
    def __init__(self, sequence_length=10, n_features=5):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self):
        """Build LSTM autoencoder model"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available - LSTM autoencoder disabled")
            return None
            
        # Encoder
        input_layer = Input(shape=(self.sequence_length, self.n_features))
        encoded = LSTM(32, activation='relu', return_sequences=True)(input_layer)
        encoded = LSTM(16, activation='relu', return_sequences=False)(encoded)
        
        # Decoder
        decoded = RepeatVector(self.sequence_length)(encoded)
        decoded = LSTM(16, activation='relu', return_sequences=True)(decoded)
        decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(self.n_features))(decoded)
        
        self.model = Model(input_layer, decoded)
        self.model.compile(optimizer='adam', loss='mse')
        
        return self.model
    
    def prepare_sequences(self, features_df):
        """Prepare sequences for LSTM training"""
        # Select key features for sequence modeling
        sequence_features = ['order_size', 'inter_arrival_time', 'volatility', 'momentum', 'spread']
        available_features = [f for f in sequence_features if f in features_df.columns][:self.n_features]
        
        if len(available_features) < self.n_features:
            # Pad with synthetic features if needed
            for i in range(self.n_features - len(available_features)):
                features_df[f'synthetic_feature_{i}'] = np.random.normal(0, 1, len(features_df))
                available_features.append(f'synthetic_feature_{i}')
        
        # Normalize features
        feature_data = features_df[available_features].values
        feature_data_scaled = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        sequences = []
        for i in range(len(feature_data_scaled) - self.sequence_length + 1):
            sequences.append(feature_data_scaled[i:i + self.sequence_length])
        
        return np.array(sequences)
    
    def train(self, features_df, epochs=50, batch_size=32):
        """Train LSTM autoencoder"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        sequences = self.prepare_sequences(features_df)
        if len(sequences) == 0:
            return None
            
        self.build_model()
        
        # Train autoencoder
        history = self.model.fit(
            sequences, sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
        
        return history
    
    def get_anomaly_scores(self, features_df):
        """Get anomaly scores from reconstruction error"""
        if self.model is None:
            return np.zeros(len(features_df))
            
        sequences = self.prepare_sequences(features_df)
        if len(sequences) == 0:
            return np.zeros(len(features_df))
            
        # Get reconstruction errors
        reconstructions = self.model.predict(sequences, verbose=0)
        mse = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
        
        # Pad scores to match original length
        scores = np.zeros(len(features_df))
        scores[self.sequence_length-1:] = mse
        
        return scores

class CalibratedToxicityDetector:
    """Advanced toxicity detector with ensemble calibration"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.ensemble_weights = {}
        self.meta_model = None
        self.lstm_autoencoder = None
        self.performance_metrics = {}
        self.best_hyperparameters = {}
        self.feature_importance = {}
        self.pca_model = None
        self.umap_model = None
        
    def prepare_features_with_dimensionality_reduction(self, features_df, variance_threshold=0.01, correlation_threshold=0.95):
        """Enhanced feature preparation with dimensionality reduction"""
        print("Preparing features with dimensionality reduction...")
        
        # Standard feature selection
        var_selector = VarianceThreshold(threshold=variance_threshold)
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        high_var_features = numeric_features.loc[:, var_selector.fit(numeric_features).get_support()]
        print(f"Removed {len(numeric_features.columns) - len(high_var_features.columns)} low variance features")
        
        # Remove correlated features
        selected_features = self._remove_correlated_features(high_var_features, correlation_threshold)
        
        # Calculate permutation-based feature importance
        self.feature_importance = self._calculate_permutation_importance(selected_features)
        
        # Store the selected feature names BEFORE any transformations
        self.feature_selector = selected_features.columns.tolist()
        
        # Apply PCA for dimensionality reduction
        self.pca_model = PCA(n_components=min(50, selected_features.shape[1]))
        X_pca = self.pca_model.fit_transform(selected_features)
        
        # Apply UMAP for non-linear dimensionality reduction
        if UMAP_AVAILABLE:
            try:
                if 'umap' in globals() and hasattr(umap, 'UMAP'):
                    # Standard umap-learn import
                    self.umap_model = umap.UMAP(n_components=10, random_state=42)
                else:
                    # Alternative import method
                    from umap import UMAP
                    self.umap_model = UMAP(n_components=10, random_state=42)
                
                X_umap = self.umap_model.fit_transform(selected_features)
                print(f"UMAP embedding: {X_umap.shape}")
            except Exception as e:
                print(f"UMAP failed: {e}")
                X_umap = X_pca[:, :10]  # Fallback to PCA
                self.umap_model = None
        else:
            print("UMAP not available, using PCA fallback")
            X_umap = X_pca[:, :10]  # Fallback to PCA
            self.umap_model = None
        
        # Select best scaler
        scaler_performance = self._evaluate_scalers(selected_features)
        best_scaler_name = max(scaler_performance, key=scaler_performance.get)
        
        if best_scaler_name == 'robust':
            scaler = RobustScaler()
        elif best_scaler_name == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        X_scaled = scaler.fit_transform(selected_features)
        
        # Combine original features with reduced dimensions
        X_combined = np.hstack([
            X_scaled,
            X_pca[:, :min(20, X_pca.shape[1])],  # Top 20 PCA components
            X_umap  # UMAP embedding or PCA fallback
        ])
        
        self.scalers['main'] = scaler
        # Store the shape information for later use
        self.original_feature_count = selected_features.shape[1]
        self.pca_component_count = min(20, X_pca.shape[1])
        self.umap_component_count = X_umap.shape[1]
        
        print(f"Selected {len(self.feature_selector)} original features")
        print(f"Combined feature set: {X_combined.shape}")
        
        return X_combined, selected_features
    
    def _calculate_permutation_importance(self, features_df):
        """Calculate permutation-based feature importance"""
        print("Calculating permutation-based feature importance...")
        
        try:
            # Train a simple model for importance calculation
            iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=50)
            iso_forest.fit(features_df)
            
            # Get baseline scores
            baseline_scores = -iso_forest.decision_function(features_df)
            
            # Calculate permutation importance
            feature_importance = {}
            for col in features_df.columns:
                # Permute this feature
                permuted_data = features_df.copy()
                permuted_data[col] = np.random.permutation(permuted_data[col])
                
                # Get permuted scores
                permuted_scores = -iso_forest.decision_function(permuted_data)
                
                # Calculate importance as score difference
                importance = np.mean(np.abs(baseline_scores - permuted_scores))
                feature_importance[col] = importance
            
            # Normalize importance scores
            max_importance = max(feature_importance.values()) if feature_importance.values() else 1
            feature_importance = {k: v/max_importance for k, v in feature_importance.items()}
            
            return feature_importance
            
        except Exception as e:
            print(f"Permutation importance calculation failed: {e}")
            # Fallback to variance-based importance
            variances = features_df.var()
            return (variances / variances.max()).to_dict()
    
    def _remove_correlated_features(self, features_df, threshold=0.95):
        """Remove highly correlated features"""
        corr_matrix = features_df.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        features_to_drop = set()
        for column in upper_triangle.columns:
            correlated_features = upper_triangle.index[upper_triangle[column] > threshold].tolist()
            if correlated_features:
                variances = {feat: features_df[feat].var() for feat in correlated_features + [column]}
                features_to_keep = max(variances, key=variances.get)
                features_to_drop.update([f for f in correlated_features if f != features_to_keep])
        
        selected_features = features_df.drop(columns=list(features_to_drop))
        print(f"Removed {len(features_to_drop)} highly correlated features")
        
        return selected_features
    
    def _evaluate_scalers(self, features_df):
        """Evaluate different scaling methods"""
        scalers = {
            'robust': RobustScaler(),
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        
        scaler_scores = {}
        
        for name, scaler in scalers.items():
            try:
                X_scaled = scaler.fit_transform(features_df)
                n_clusters = min(5, max(2, len(features_df) // 100))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                if len(set(labels)) > 1:
                    score = silhouette_score(X_scaled, labels)
                    scaler_scores[name] = score
                else:
                    scaler_scores[name] = -1
                    
            except Exception:
                scaler_scores[name] = -1
        
        return scaler_scores
    
    def train_calibrated_ensemble(self, X_scaled, features_df, hyperparameters=None):
        """Train ensemble with meta-model calibration"""
        print("Training calibrated ensemble with meta-model...")
        
        # Train base detectors
        detectors = {}
        
        # Isolation Forest variants
        contamination_rates = [0.05, 0.1, 0.15]
        for i, contamination in enumerate(contamination_rates):
            try:
                iso_forest = IsolationForest(
                    contamination=contamination,
                    n_estimators=200,
                    random_state=42 + i,
                    bootstrap=True
                )
                iso_forest.fit(X_scaled)
                detectors[f'isolation_forest_{contamination}'] = iso_forest
            except Exception as e:
                print(f"Warning: Failed to train Isolation Forest: {e}")
        
        # LOF variants
        for neighbors in [10, 20, 30]:
            try:
                lof = LocalOutlierFactor(
                    n_neighbors=neighbors,
                    contamination=0.1,
                    novelty=True
                )
                lof.fit(X_scaled)
                detectors[f'lof_{neighbors}'] = lof
            except Exception as e:
                print(f"Warning: Failed to train LOF: {e}")
        
        # One-Class SVM
        try:
            svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
            svm.fit(X_scaled)
            detectors['svm_rbf'] = svm
        except Exception as e:
            print(f"Warning: Failed to train SVM: {e}")
        
        # K-means based detectors
        for n_clusters in [5, 10, 15]:
            try:
                n_clust = min(n_clusters, max(2, len(X_scaled) // 30))
                kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                distances = np.min(kmeans.transform(X_scaled), axis=1)
                
                detectors[f'kmeans_{n_clust}'] = {
                    'kmeans': kmeans,
                    'distance_threshold': np.percentile(distances, 95)
                }
            except Exception as e:
                print(f"Warning: Failed to train K-means: {e}")
        
        # LSTM Autoencoder
        if TENSORFLOW_AVAILABLE:
            try:
                self.lstm_autoencoder = LSTMAutoencoder()
                self.lstm_autoencoder.train(features_df)
                detectors['lstm_autoencoder'] = self.lstm_autoencoder
                print("LSTM Autoencoder trained successfully")
            except Exception as e:
                print(f"Warning: Failed to train LSTM Autoencoder: {e}")
        
        self.models = detectors
        
        # Train meta-model for ensemble calibration
        self._train_meta_model(X_scaled, features_df)
        
        print(f"Trained {len(detectors)} detectors with meta-model calibration")
        return detectors
    
    def _train_meta_model(self, X_scaled, features_df):
        """Train meta-model for ensemble calibration"""
        print("Training meta-model for ensemble calibration...")
        # Get individual detector scores
        individual_scores = {}
        for name, model in self.models.items():
            try:
                if 'isolation_forest' in name:
                    scores = -model.decision_function(X_scaled)
                elif 'lof' in name:
                    scores = -model.score_samples(X_scaled)
                elif 'svm' in name:
                    scores = -model.decision_function(X_scaled)
                elif 'kmeans' in name:
                    distances = np.min(model['kmeans'].transform(X_scaled), axis=1)
                    scores = distances
                elif 'lstm_autoencoder' in name:
                    scores = model.get_anomaly_scores(features_df)
                else:
                    continue
                
                # Normalize scores
                if scores.max() > scores.min():
                    scores = (scores - scores.min()) / (scores.max() - scores.min())
                
                individual_scores[name] = scores
                
            except Exception as e:
                print(f"Error getting scores for {name}: {e}")
                continue
        
        if len(individual_scores) < 2:
            print("Warning: Insufficient detectors for meta-model training")
            return
        
        # Create meta-features matrix
        score_matrix = np.array(list(individual_scores.values())).T
        
        # Create synthetic target for meta-model training
        # Use ensemble percentile as pseudo-target
        ensemble_baseline = np.mean(score_matrix, axis=1)
        target = (ensemble_baseline > np.percentile(ensemble_baseline, 95)).astype(int)
        
        # Add synthetic anomaly targets
        synthetic_features = ['synthetic_spoofing', 'synthetic_momentum_ignition', 'synthetic_layering']
        available_synthetic = [f for f in synthetic_features if f in features_df.columns]
        
        if available_synthetic:
            synthetic_target = features_df[available_synthetic].max(axis=1).values
            target = np.maximum(target, synthetic_target)
        
        # Train meta-model (Ridge regression for stability)
        try:
            self.meta_model = Ridge(alpha=1.0, random_state=42)
            self.meta_model.fit(score_matrix, target)
            
            # Evaluate meta-model
            meta_predictions = self.meta_model.predict(score_matrix)
            meta_score = np.corrcoef(meta_predictions, target)[0, 1]
            print(f"Meta-model correlation with targets: {meta_score:.3f}")
            
        except Exception as e:
            print(f"Meta-model training failed: {e}")
            self.meta_model = None
    
    def calculate_calibrated_ensemble_scores(self, X_scaled, features_df):
        """Calculate ensemble scores using meta-model calibration"""
        print("Calculating calibrated ensemble scores...")
        
        individual_scores = {}
        
        # Get individual detector scores
        for name, model in self.models.items():
            try:
                if 'isolation_forest' in name:
                    scores = -model.decision_function(X_scaled)
                elif 'lof' in name:
                    scores = -model.score_samples(X_scaled)
                elif 'svm' in name:
                    scores = -model.decision_function(X_scaled)
                elif 'kmeans' in name:
                    distances = np.min(model['kmeans'].transform(X_scaled), axis=1)
                    scores = distances
                elif 'lstm_autoencoder' in name:
                    scores = model.get_anomaly_scores(features_df)
                else:
                    continue
                
                # Normalize scores
                if scores.max() > scores.min():
                    scores = (scores - scores.min()) / (scores.max() - scores.min())
                else:
                    scores = np.zeros_like(scores)
                
                individual_scores[name] = scores
                
            except Exception as e:
                print(f"Error calculating scores for {name}: {e}")
                continue
        
        # Use meta-model for ensemble calibration if available
        if self.meta_model is not None and len(individual_scores) > 1:
            try:
                score_matrix = np.array(list(individual_scores.values())).T
                ensemble_scores = self.meta_model.predict(score_matrix)
                
                # Ensure scores are in [0, 1] range
                ensemble_scores = np.clip(ensemble_scores, 0, 1)
                
                print("Using meta-model calibrated ensemble scores")
                
            except Exception as e:
                print(f"Meta-model prediction failed: {e}, falling back to weighted average")
                ensemble_scores = self._fallback_ensemble_scores(individual_scores)
        else:
            ensemble_scores = self._fallback_ensemble_scores(individual_scores)
        
        # Calculate individual detector weights for interpretation
        self._calculate_detector_weights(individual_scores)
        
        return ensemble_scores, individual_scores
    
    def _fallback_ensemble_scores(self, individual_scores):
        """Fallback ensemble scoring method"""
        if len(individual_scores) == 0:
            return np.zeros(1)
        
        # Simple average with equal weights
        score_matrix = np.array(list(individual_scores.values()))
        ensemble_scores = np.mean(score_matrix, axis=0)
        
        return ensemble_scores
    
    def _calculate_detector_weights(self, individual_scores):
        """Calculate detector weights for interpretation"""
        weights = {}
        
        for name, scores in individual_scores.items():
            # Weight based on score quality metrics
            score_std = np.std(scores)
            score_range = np.max(scores) - np.min(scores)
            separation_quality = score_std * score_range
            
            # Consistency across thresholds
            consistency_scores = []
            for pct in [90, 95, 99]:
                threshold = np.percentile(scores, pct)
                detection_rate = np.mean(scores > threshold)
                expected_rate = (100 - pct) / 100
                consistency = 1 - abs(detection_rate - expected_rate) / expected_rate if expected_rate > 0 else 0
                consistency_scores.append(max(0, consistency))
            
            avg_consistency = np.mean(consistency_scores)
            
            # Combine metrics
            weight = separation_quality * avg_consistency
            weights[name] = max(0.01, min(weight, 1.0))
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        
        self.ensemble_weights = weights
        
        print("Detector weights (for interpretation):")
        for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {weight:.3f}")
    
    def evaluate_with_synthetic_anomalies(self, X_scaled, ensemble_scores, features_df):
        """Evaluate model performance using synthetic anomalies"""
        print("Evaluating performance with synthetic anomalies...")
        
        metrics = {}
        
        # Standard evaluation metrics
        metrics['score_stats'] = {
            'mean': ensemble_scores.mean(),
            'std': ensemble_scores.std(),
            'min': ensemble_scores.min(),
            'max': ensemble_scores.max(),
            'skewness': stats.skew(ensemble_scores),
            'kurtosis': stats.kurtosis(ensemble_scores)
        }
        
        # Synthetic anomaly evaluation
        synthetic_features = ['synthetic_spoofing', 'synthetic_momentum_ignition', 'synthetic_layering']
        synthetic_evaluation = {}
        
        for synthetic_feature in synthetic_features:
            if synthetic_feature in features_df.columns:
                synthetic_labels = features_df[synthetic_feature].values
                if synthetic_labels.sum() > 0:  # If any synthetic anomalies exist
                    
                    # AUC-like metric for ranking
                    normal_scores = ensemble_scores[synthetic_labels == 0]
                    anomaly_scores = ensemble_scores[synthetic_labels == 1]
                    
                    if len(normal_scores) > 0 and len(anomaly_scores) > 0:
                        # Mann-Whitney U test statistic (equivalent to AUC)
                        from scipy.stats import mannwhitneyu
                        try:
                            statistic, p_value = mannwhitneyu(anomaly_scores, normal_scores, alternative='greater')
                            auc_score = statistic / (len(normal_scores) * len(anomaly_scores))
                            
                            synthetic_evaluation[synthetic_feature] = {
                                'auc_score': auc_score,
                                'p_value': p_value,
                                'n_anomalies': len(anomaly_scores),
                                'mean_anomaly_score': anomaly_scores.mean(),
                                'mean_normal_score': normal_scores.mean()
                            }
                        except Exception as e:
                            print(f"Error evaluating {synthetic_feature}: {e}")
        
        metrics['synthetic_anomaly_evaluation'] = synthetic_evaluation
        
        # Detection rates at different thresholds
        thresholds = [85, 90, 95, 97, 99, 99.5]
        anomaly_rates = {}
        for threshold in thresholds:
            threshold_value = np.percentile(ensemble_scores, threshold)
            anomaly_rate = np.mean(ensemble_scores > threshold_value)
            anomaly_rates[f'{threshold}th_percentile'] = anomaly_rate
            
            # Separation score
            if anomaly_rate > 0 and anomaly_rate < 1:
                anomaly_scores = ensemble_scores[ensemble_scores > threshold_value]
                normal_scores = ensemble_scores[ensemble_scores <= threshold_value]
                if len(normal_scores) > 0 and normal_scores.std() > 0:
                    separation = (anomaly_scores.mean() - normal_scores.mean()) / normal_scores.std()
                    anomaly_rates[f'{threshold}th_separation'] = separation
        
        metrics['anomaly_rates'] = anomaly_rates
        
        # Feature importance evaluation
        if SHAP_AVAILABLE and self.meta_model is not None:
            try:
                # SHAP analysis for meta-model
                individual_scores_matrix = np.array([
                    scores for scores in self.individual_scores.values()
                ]).T
                
                explainer = shap.LinearExplainer(self.meta_model, individual_scores_matrix)
                shap_values = explainer.shap_values(individual_scores_matrix[:100])  # Sample for performance
                
                detector_names = list(self.models.keys())
                shap_importance = {
                    detector_names[i]: np.mean(np.abs(shap_values[:, i])) 
                    for i in range(len(detector_names))
                }
                
                metrics['shap_detector_importance'] = shap_importance
                print("SHAP detector importance calculated")
                
            except Exception as e:
                print(f"SHAP analysis failed: {e}")
        
        # PCA cluster stability
        if self.pca_model is not None and hasattr(self, 'feature_selector'):
            try:
                # Use only the original features for PCA transform
                original_features = features_df[self.feature_selector]
                
                # Apply the same scaling as during training
                if 'main' in self.scalers:
                    original_features_scaled = self.scalers['main'].transform(original_features)
                else:
                    scaler = StandardScaler()
                    original_features_scaled = scaler.fit_transform(original_features)
                
                X_pca = self.pca_model.transform(original_features_scaled)
                
                # Evaluate cluster stability in PCA space
                kmeans_pca = KMeans(n_clusters=5, random_state=42, n_init=10)
                pca_labels = kmeans_pca.fit_predict(X_pca[:, :10])  # Use top 10 components
                
                if len(set(pca_labels)) > 1:
                    pca_silhouette = silhouette_score(X_pca[:, :10], pca_labels)
                    metrics['pca_cluster_quality'] = pca_silhouette
                    print(f"PCA cluster quality: {pca_silhouette:.3f}")
                    
            except Exception as e:
                print(f"PCA cluster evaluation failed: {e}")
                metrics['pca_cluster_quality'] = None
        
        self.performance_metrics = metrics
        return metrics
    
    def save_calibrated_model(self, save_dir="calibrated_toxicity_models"):
        """Save the calibrated model with all components"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{save_dir}/calibrated_toxicity_detector_{timestamp}.joblib"
        
        # Prepare LSTM model for saving
        lstm_model_path = None
        if self.lstm_autoencoder is not None and self.lstm_autoencoder.model is not None:
            lstm_model_path = f"{save_dir}/lstm_autoencoder_{timestamp}.h5"
            self.lstm_autoencoder.model.save(lstm_model_path)
        
        model_package = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_selector': self.feature_selector,
            'ensemble_weights': self.ensemble_weights,
            'meta_model': self.meta_model,
            'lstm_model_path': lstm_model_path,
            'pca_model': self.pca_model,
            'umap_model': self.umap_model,
            'performance_metrics': self.performance_metrics,
            'best_hyperparameters': self.best_hyperparameters,
            'feature_importance': self.feature_importance,
            'timestamp': timestamp,
            'version': '6.0_calibrated_advanced',
            'n_features': len(self.feature_selector) if self.feature_selector else 0,
            'n_detectors': len(self.models),
            'has_meta_model': self.meta_model is not None,
            'has_lstm': lstm_model_path is not None,
            'training_summary': {
                'top_features': dict(sorted(self.feature_importance.items(), 
                                            key=lambda x: x[1], reverse=True)[:15]),
                'detector_weights': dict(sorted(self.ensemble_weights.items(), 
                                                key=lambda x: x[1], reverse=True)),
                'synthetic_anomaly_performance': self.performance_metrics.get('synthetic_anomaly_evaluation', {}),
                'meta_model_available': self.meta_model is not None,
                'pca_components': self.pca_model.n_components_ if self.pca_model else 0
            }
        }
        
        joblib.dump(model_package, model_path)
        
        print(f"Calibrated model saved to: {model_path}")
        if lstm_model_path:
            print(f"LSTM autoencoder saved to: {lstm_model_path}")
        
        return model_path

def create_advanced_visualizations(features_df, ensemble_scores, detector, X_scaled):
    """Create advanced visualizations with dimensionality reduction plots"""
    
    plots_dir = "advanced_toxicity_plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.style.use('ggplot')
    plt.rcParams.update({
        'figure.figsize': (16, 12),
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    # 1. Advanced Model Performance Dashboard
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    
    # Score distribution with synthetic anomaly overlay
    axes[0, 0].hist(ensemble_scores, bins=60, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    
    # Overlay synthetic anomalies
    synthetic_features = ['synthetic_spoofing', 'synthetic_momentum_ignition', 'synthetic_layering']
    colors = ['red', 'orange', 'purple']
    
    for synthetic_feature, color in zip(synthetic_features, colors):
        if synthetic_feature in features_df.columns:
            anomaly_mask = features_df[synthetic_feature] == 1
            if anomaly_mask.sum() > 0:
                anomaly_scores = ensemble_scores[anomaly_mask]
                axes[0, 0].hist(anomaly_scores, bins=30, alpha=0.5, color=color, 
                                label=synthetic_feature.replace('synthetic_', ''), density=True)
    
    axes[0, 0].axvline(np.percentile(ensemble_scores, 95), color='red', linestyle='--', linewidth=2, label='95th')
    axes[0, 0].axvline(np.percentile(ensemble_scores, 99), color='darkred', linestyle='--', linewidth=2, label='99th')
    axes[0, 0].set_xlabel('Toxicity Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Score Distribution with Synthetic Anomalies')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PCA visualization - FIXED
    if detector.pca_model is not None and hasattr(detector, 'feature_selector'):
        try:
            # Extract only the original features that PCA was trained on
            original_features = features_df[detector.feature_selector]
            
            # Apply the same scaling as during training
            if 'main' in detector.scalers:
                original_features_scaled = detector.scalers['main'].transform(original_features)
            else:
                scaler = StandardScaler()
                original_features_scaled = scaler.fit_transform(original_features)
            
            X_pca = detector.pca_model.transform(original_features_scaled)
            
            # PCA scatter plot colored by toxicity score
            scatter = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=ensemble_scores, 
                                        cmap='viridis', alpha=0.6, s=10)
            axes[0, 1].set_xlabel(f'PC1 ({detector.pca_model.explained_variance_ratio_[0]:.1%} variance)')
            axes[0, 1].set_ylabel(f'PC2 ({detector.pca_model.explained_variance_ratio_[1]:.1%} variance)')
            axes[0, 1].set_title('PCA Space - Toxicity Intensity')
            plt.colorbar(scatter, ax=axes[0, 1], label='Toxicity Score')
            
            # PCA explained variance
            axes[0, 2].plot(range(1, min(21, len(detector.pca_model.explained_variance_ratio_) + 1)), 
                            np.cumsum(detector.pca_model.explained_variance_ratio_)[:20], 'bo-', linewidth=2)
            axes[0, 2].axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='95% Variance')
            axes[0, 2].set_xlabel('Number of Components')
            axes[0, 2].set_ylabel('Cumulative Explained Variance')
            axes[0, 2].set_title('PCA Explained Variance')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"PCA visualization failed: {e}")
            axes[0, 1].text(0.5, 0.5, f'PCA visualization\nfailed: {str(e)[:50]}...', 
                            ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('PCA Visualization (Error)')
            axes[0, 2].text(0.5, 0.5, 'PCA variance plot\nunavailable', 
                            ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('PCA Explained Variance (Error)')
    
    # UMAP visualization - FIXED
    if detector.umap_model is not None and hasattr(detector, 'feature_selector'):
        try:
            # Extract only the original features that UMAP was trained on
            original_features = features_df[detector.feature_selector]
            
            # Apply the same scaling as during training
            if 'main' in detector.scalers:
                original_features_scaled = detector.scalers['main'].transform(original_features)
            else:
                scaler = StandardScaler()
                original_features_scaled = scaler.fit_transform(original_features)
            
            X_umap = detector.umap_model.transform(original_features_scaled)
            scatter_umap = axes[0, 3].scatter(X_umap[:, 0], X_umap[:, 1], c=ensemble_scores, 
                                                cmap='plasma', alpha=0.6, s=10)
            axes[0, 3].set_xlabel('UMAP1')
            axes[0, 3].set_ylabel('UMAP2')
            axes[0, 3].set_title('UMAP Space - Toxicity Intensity')
            plt.colorbar(scatter_umap, ax=axes[0, 3], label='Toxicity Score')
        except Exception as e:
            print(f"UMAP visualization failed: {e}")
            axes[0, 3].text(0.5, 0.5, f'UMAP visualization\nfailed: {str(e)[:50]}...', 
                            ha='center', va='center', transform=axes[0, 3].transAxes)
            axes[0, 3].set_title('UMAP Visualization (Error)')
    else:
        axes[0, 3].text(0.5, 0.5, 'UMAP visualization\nunavailable', 
                        ha='center', va='center', transform=axes[0, 3].transAxes)
        axes[0, 3].set_title('UMAP Visualization')
    
    # Synthetic anomaly performance
    synthetic_eval = detector.performance_metrics.get('synthetic_anomaly_evaluation', {})
    if synthetic_eval:
        synthetic_names = list(synthetic_eval.keys())
        auc_scores = [synthetic_eval[name].get('auc_score', 0) for name in synthetic_names]
        
        bars = axes[1, 0].bar(range(len(synthetic_names)), auc_scores, 
                                color=['red', 'orange', 'purple'][:len(synthetic_names)], alpha=0.7)
        axes[1, 0].set_xticks(range(len(synthetic_names)))
        axes[1, 0].set_xticklabels([name.replace('synthetic_', '') for name in synthetic_names], rotation=45)
        axes[1, 0].set_ylabel('AUC Score')
        axes[1, 0].set_title('Synthetic Anomaly Detection Performance')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, score in zip(bars, auc_scores):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Meta-model vs individual detector comparison
    if detector.meta_model is not None:
        detector_names = list(detector.ensemble_weights.keys())[:8]  # Top 8 for readability
        weights = [detector.ensemble_weights[name] for name in detector_names]
        
        bars = axes[1, 1].bar(range(len(detector_names)), weights, 
                                color=plt.cm.Set3(np.arange(len(detector_names))), alpha=0.8)
        axes[1, 1].set_xticks(range(len(detector_names)))
        axes[1, 1].set_xticklabels([name[:12] for name in detector_names], rotation=45, ha='right')
        axes[1, 1].set_ylabel('Interpretive Weight')
        axes[1, 1].set_title('Detector Weights (Meta-Model Available)')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Feature importance (permutation-based)
    if detector.feature_importance:
        top_features = dict(sorted(detector.feature_importance.items(), key=lambda x: x[1], reverse=True)[:12])
        feature_names = list(top_features.keys())
        importance_scores = list(top_features.values())
        
        colors_feat = plt.cm.viridis(np.linspace(0, 1, len(importance_scores)))
        axes[1, 2].barh(range(len(feature_names)), importance_scores, color=colors_feat, alpha=0.8)
        axes[1, 2].set_yticks(range(len(feature_names)))
        axes[1, 2].set_yticklabels([name[:18] for name in feature_names])
        axes[1, 2].set_xlabel('Permutation Importance')
        axes[1, 2].set_title('Top 12 Permutation-Based Feature Importance')
        axes[1, 2].grid(True, alpha=0.3, axis='x')
    
    # SHAP detector importance (if available)
    shap_importance = detector.performance_metrics.get('shap_detector_importance', {})
    if shap_importance:
        shap_names = list(shap_importance.keys())
        shap_values = list(shap_importance.values())
        
        bars_shap = axes[1, 3].bar(range(len(shap_names)), shap_values, 
                                    color='lightcoral', alpha=0.8)
        axes[1, 3].set_xticks(range(len(shap_names)))
        axes[1, 3].set_xticklabels([name[:10] for name in shap_names], rotation=45, ha='right')
        axes[1, 3].set_ylabel('SHAP Importance')
        axes[1, 3].set_title('SHAP Detector Importance')
        axes[1, 3].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 3].text(0.5, 0.5, 'SHAP analysis\nnot available', 
                        ha='center', va='center', transform=axes[1, 3].transAxes)
        axes[1, 3].set_title('SHAP Analysis')
    
    # Detection performance across thresholds
    thresholds = [85, 90, 95, 97, 99, 99.5]
    anomaly_rates = detector.performance_metrics.get('anomaly_rates', {})
    
    rates = [anomaly_rates.get(f'{t}th_percentile', 0) * 100 for t in thresholds]
    separations = [anomaly_rates.get(f'{t}th_separation', 0) for t in thresholds]
    
    # Detection rates
    bars_rates = axes[2, 0].bar(range(len(thresholds)), rates, 
                                color=['lightblue', 'lightgreen', 'orange', 'coral', 'red', 'darkred'], alpha=0.8)
    axes[2, 0].set_xticks(range(len(thresholds)))
    axes[2, 0].set_xticklabels([f'{t}th' for t in thresholds])
    axes[2, 0].set_ylabel('Detection Rate (%)')
    axes[2, 0].set_title('Detection Rates by Threshold')
    axes[2, 0].grid(True, alpha=0.3, axis='y')
    
    # Separation scores
    bars_sep = axes[2, 1].bar(range(len(thresholds)), separations, 
                                color=['lightblue', 'lightgreen', 'orange', 'coral', 'red', 'darkred'], alpha=0.8)
    axes[2, 1].set_xticks(range(len(thresholds)))
    axes[2, 1].set_xticklabels([f'{t}th' for t in thresholds])
    axes[2, 1].set_ylabel('Separation Score')
    axes[2, 1].set_title('Anomaly Separation by Threshold')
    axes[2, 1].grid(True, alpha=0.3, axis='y')
    
    # Model capability summary
    capabilities_text = f"""
    ADVANCED MODEL CAPABILITIES
    
    Dataset: {len(ensemble_scores)} samples
    Features: {len(detector.feature_selector) if detector.feature_selector else 0}
    Detectors: {len(detector.models)}
    
    Advanced Features:
    • Meta-Model Calibration: {'Yes' if detector.meta_model else 'No'}
    • LSTM Autoencoder: {'Yes' if detector.lstm_autoencoder else 'No'}
    • PCA Components: {detector.pca_model.n_components_ if detector.pca_model else 0}
    • UMAP Embedding: {'Yes' if detector.umap_model else 'No'}
    • SHAP Analysis: {'Yes' if shap_importance else 'No'}
    
    Performance:
    • Avg Separation: {np.mean([s for s in separations if s > 0]):.3f}
    • Score Range: {ensemble_scores.max() - ensemble_scores.min():.3f}
    • Synthetic Anomaly AUC: {np.mean([v.get('auc_score', 0) for v in synthetic_eval.values()]) if synthetic_eval else 0:.3f}
    """
    
    axes[2, 2].text(0.05, 0.95, capabilities_text, transform=axes[2, 2].transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
    axes[2, 2].set_xlim(0, 1)
    axes[2, 2].set_ylim(0, 1)
    axes[2, 2].axis('off')
    axes[2, 2].set_title('Advanced Capabilities Summary', fontweight='bold')
    
    # Timeline with anomaly highlights
    axes[2, 3].plot(ensemble_scores, alpha=0.7, color='blue', linewidth=1)
    axes[2, 3].axhline(np.percentile(ensemble_scores, 95), color='orange', linestyle='--', alpha=0.8)
    axes[2, 3].axhline(np.percentile(ensemble_scores, 99), color='red', linestyle='--', alpha=0.8)
    
    # Highlight synthetic anomalies
    for synthetic_feature, color in zip(synthetic_features, colors):
        if synthetic_feature in features_df.columns:
            anomaly_indices = np.where(features_df[synthetic_feature] == 1)[0]
            if len(anomaly_indices) > 0:
                axes[2, 3].scatter(anomaly_indices, ensemble_scores[anomaly_indices], 
                                    color=color, s=20, alpha=0.8, zorder=5)
    
    axes[2, 3].set_xlabel('Order Sequence')
    axes[2, 3].set_ylabel('Toxicity Score')
    axes[2, 3].set_title('Timeline with Synthetic Anomaly Highlights')
    axes[2, 3].grid(True, alpha=0.3)
    
    plt.suptitle(f'Advanced Calibrated Toxicity Detection Model - Comprehensive Analysis\nTimestamp: {timestamp}', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/advanced_comprehensive_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Advanced visualizations saved to: {plots_dir}")
    plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png') and timestamp in f]
    print(f"Generated {len(plot_files)} advanced plot files:")
    for file in sorted(plot_files):
        print(f"  {file}")

def main_advanced_training_pipeline(data_dir="enhanced_market_data", n_trials=30):
    """Advanced training pipeline with all technical improvements"""
    print("="*80)
    print("ADVANCED CALIBRATED TOXICITY DETECTION MODEL TRAINING")
    print("Meta-Model Calibration | Dimensionality Reduction | Sequence Modelling")
    print("Real-Time Optimisation | Synthetic Anomaly Evaluation")
    print("="*80)
    
    # Load and engineer features
    print("\n1. LOADING & OPTIMISED FEATURE ENGINEERING")
    print("-" * 40)
    
    feature_engineer = AdvancedFeatureEngineer()
    orders_df, lob_df, trades_df = feature_engineer.load_market_data(data_dir)
    features_df = feature_engineer.extract_optimized_features(orders_df, lob_df, trades_df)
    
    print(f"\nData Summary:")
    print(f"  Orders: {len(orders_df)} records")
    print(f"  LOB Snapshots: {len(lob_df)} records") 
    print(f"  Trades: {len(trades_df)} records")
    print(f"  Generated Features: {len(features_df.columns)} features")
    
    # Feature preparation with dimensionality reduction
    print("\n2. ADVANCED FEATURE PREPARATION & DIMENSIONALITY REDUCTION")
    print("-" * 60)
    
    detector = CalibratedToxicityDetector()
    X_scaled, selected_features = detector.prepare_features_with_dimensionality_reduction(features_df)
    
    # Train calibrated ensemble
    print("\n3. CALIBRATED ENSEMBLE TRAINING")
    print("-" * 40)
    
    trained_models = detector.train_calibrated_ensemble(X_scaled, features_df)
    
    # Calculate ensemble scores
    print("\n4. ENSEMBLE SCORE CALCULATION")
    print("-" * 35)
    
    ensemble_scores, individual_scores = detector.calculate_calibrated_ensemble_scores(X_scaled, features_df)
    detector.individual_scores = individual_scores  # Store for evaluation
    
    # Evaluate performance
    print("\n5. COMPREHENSIVE PERFORMANCE EVALUATION")
    print("-" * 45)
    
    metrics = detector.evaluate_with_synthetic_anomalies(X_scaled, ensemble_scores, features_df)
    
    # Advanced visualisations
    print("\n6. ADVANCED VISUALISATION GENERATION")
    print("-" * 40)
    
    create_advanced_visualizations(features_df, ensemble_scores, detector, X_scaled)
    
    # Save model
    print("\n7. MODEL PERSISTENCE")
    print("-" * 20)
    
    model_path = detector.save_calibrated_model()
    
    # Final summary
    print("\n" + "="*80)
    print("ADVANCED TRAINING COMPLETE - PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"\nModel Capabilities:")
    print(f"  • Dataset Size: {len(ensemble_scores):,} samples")
    print(f"  • Original Features: {len(features_df.columns)} → Selected: {len(detector.feature_selector)}")
    print(f"  • Ensemble Detectors: {len(detector.models)}")
    print(f"  • Meta-Model Calibration: {'✓' if detector.meta_model else '✗'}")
    print(f"  • LSTM Autoencoder: {'✓' if detector.lstm_autoencoder else '✗'}")
    print(f"  • PCA Components: {detector.pca_model.n_components_ if detector.pca_model else 0}")
    print(f"  • UMAP Embedding: {'✓' if detector.umap_model else '✗'}")
    
    print(f"\nPerformance Metrics:")
    score_stats = metrics.get('score_stats', {})
    print(f"  • Score Range: {score_stats.get('min', 0):.3f} - {score_stats.get('max', 1):.3f}")
    print(f"  • Score Std Dev: {score_stats.get('std', 0):.3f}")
    print(f"  • Score Skewness: {score_stats.get('skewness', 0):.3f}")
    
    # Synthetic anomaly performance
    synthetic_eval = metrics.get('synthetic_anomaly_evaluation', {})
    if synthetic_eval:
        print(f"\nSynthetic Anomaly Detection:")
        for anomaly_type, eval_metrics in synthetic_eval.items():
            auc_score = eval_metrics.get('auc_score', 0)
            n_anomalies = eval_metrics.get('n_anomalies', 0)
            print(f"  • {anomaly_type.replace('synthetic_', '').title()}: AUC={auc_score:.3f} (n={n_anomalies})")
    
    # Top features
    print(f"\nTop Feature Importance:")
    top_features = dict(sorted(detector.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
    for feature, importance in top_features.items():
        print(f"  • {feature[:25]}: {importance:.3f}")
    
    # Detector weights
    print(f"\nDetector Ensemble Weights:")
    top_detectors = dict(sorted(detector.ensemble_weights.items(), key=lambda x: x[1], reverse=True)[:5])
    for detector_name, weight in top_detectors.items():
        print(f"  • {detector_name[:25]}: {weight:.3f}")
    
    # Performance evaluation summary
    print(f"\nModel Quality Assessment:")
    anomaly_rates = metrics.get('anomaly_rates', {})
    if anomaly_rates:
        print(f"  • 95th Percentile Detection Rate: {anomaly_rates.get('95th_percentile', 0)*100:.1f}%")
        print(f"  • 99th Percentile Detection Rate: {anomaly_rates.get('99th_percentile', 0)*100:.1f}%")
        print(f"  • Average Separation Score: {np.mean([v for k, v in anomaly_rates.items() if 'separation' in k]):.3f}")
    
    print(f"\nModel saved to: {model_path}")
    print("="*80)
    
    return detector, ensemble_scores, metrics, model_path

def test_model_performance(detector, X_scaled, features_df, ensemble_scores):
    """Comprehensive testing and validation of the trained model"""
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL TESTING & VALIDATION")
    print("="*60)
    
    test_results = {}
    
    # 1. Cross-validation simulation (using bootstrap)
    print("\n1. Bootstrap Cross-Validation Testing")
    print("-" * 40)
    
    n_bootstrap = 10
    bootstrap_scores = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        sample_indices = np.random.choice(len(X_scaled), size=len(X_scaled), replace=True)
        X_bootstrap = X_scaled[sample_indices]
        features_bootstrap = features_df.iloc[sample_indices]
        
        # Calculate scores
        boot_scores, _ = detector.calculate_calibrated_ensemble_scores(X_bootstrap, features_bootstrap)
        bootstrap_scores.append(boot_scores)
    
    # Calculate stability metrics
    score_stability = np.std([np.mean(scores) for scores in bootstrap_scores])
    test_results['cross_validation'] = {
        'score_stability': score_stability,
        'mean_score_variance': np.mean([np.var(scores) for scores in bootstrap_scores]),
        'consistent_ranking': np.corrcoef([np.argsort(scores) for scores in bootstrap_scores[:2]])[0, 1] if len(bootstrap_scores) >= 2 else 0
    }
    
    print(f"  Score Stability (std): {score_stability:.4f}")
    print(f"  Mean Score Variance: {test_results['cross_validation']['mean_score_variance']:.4f}")
    
    # 2. Detector reliability testing
    print("\n2. Individual Detector Reliability")
    print("-" * 35)
    
    detector_reliability = {}
    for name, model in detector.models.items():
        try:
            # Test consistency with noise injection
            noise_factor = 0.01
            X_noisy = X_scaled + np.random.normal(0, noise_factor, X_scaled.shape)
            
            if 'isolation_forest' in name:
                original_scores = -model.decision_function(X_scaled)
                noisy_scores = -model.decision_function(X_noisy)
            elif 'lof' in name:
                original_scores = -model.score_samples(X_scaled)
                noisy_scores = -model.score_samples(X_noisy)
            elif 'svm' in name:
                original_scores = -model.decision_function(X_scaled)
                noisy_scores = -model.decision_function(X_noisy)
            else:
                continue
            
            # Calculate reliability score (correlation under noise)
            reliability = np.corrcoef(original_scores, noisy_scores)[0, 1]
            detector_reliability[name] = reliability
            
        except Exception as e:
            detector_reliability[name] = 0.0
    
    test_results['detector_reliability'] = detector_reliability
    
    # Print top reliable detectors
    sorted_reliability = sorted(detector_reliability.items(), key=lambda x: x[1], reverse=True)
    print("  Top Reliable Detectors:")
    for name, reliability in sorted_reliability[:5]:
        print(f"    {name[:20]}: {reliability:.3f}")
    
    # 3. Threshold sensitivity analysis
    print("\n3. Threshold Sensitivity Analysis")
    print("-" * 35)
    
    thresholds = np.percentile(ensemble_scores, [80, 85, 90, 95, 97, 99, 99.5])
    threshold_analysis = {}
    
    for i, threshold in enumerate(thresholds):
        anomaly_mask = ensemble_scores > threshold
        n_anomalies = np.sum(anomaly_mask)
        anomaly_rate = n_anomalies / len(ensemble_scores)
        
        threshold_analysis[f'{[80, 85, 90, 95, 97, 99, 99.5][i]}th_percentile'] = {
            'threshold_value': threshold,
            'detection_count': n_anomalies,
            'detection_rate': anomaly_rate
        }
    
    test_results['threshold_sensitivity'] = threshold_analysis
    
    print("  Threshold Analysis:")
    for percentile, analysis in threshold_analysis.items():
        print(f"    {percentile}: {analysis['detection_count']} detections ({analysis['detection_rate']*100:.2f}%)")
    
    # 4. Feature importance stability test
    print("\n4. Feature Importance Stability")
    print("-" * 30)
    
    # Test feature importance with data permutation
    n_permutations = 5
    importance_variations = []
    
    for i in range(n_permutations):
        # Shuffle data order
        shuffled_indices = np.random.permutation(len(features_df))
        shuffled_features = features_df.iloc[shuffled_indices]
        
        # Recalculate feature importance
        temp_importance = detector._calculate_permutation_importance(shuffled_features[detector.feature_selector])
        importance_variations.append(temp_importance)
    
    # Calculate stability of feature rankings
    feature_rank_stability = {}
    for feature in detector.feature_importance.keys():
        ranks = []
        for importance_dict in importance_variations:
            if feature in importance_dict:
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                rank = next((i for i, (f, _) in enumerate(sorted_features) if f == feature), len(sorted_features))
                ranks.append(rank)
        
        if ranks:
            feature_rank_stability[feature] = np.std(ranks)
    
    test_results['feature_stability'] = feature_rank_stability
    
    # Show most stable features
    stable_features = sorted(feature_rank_stability.items(), key=lambda x: x[1])[:5]
    print("  Most Stable Features (low rank variance):")
    for feature, stability in stable_features:
        print(f"    {feature[:25]}: {stability:.2f}")
    
    # 5. Synthetic anomaly validation
    print("\n5. Synthetic Anomaly Validation")
    print("-" * 30)
    
    synthetic_validation = {}
    synthetic_features = ['synthetic_spoofing', 'synthetic_momentum_ignition', 'synthetic_layering']
    
    for synthetic_feature in synthetic_features:
        if synthetic_feature in features_df.columns:
            synthetic_mask = features_df[synthetic_feature] == 1
            if synthetic_mask.sum() > 0:
                # Calculate true positive rate at different thresholds
                synthetic_scores = ensemble_scores[synthetic_mask]
                normal_scores = ensemble_scores[~synthetic_mask]
                
                tpr_at_thresholds = {}
                for percentile in [90, 95, 99]:
                    threshold = np.percentile(ensemble_scores, percentile)
                    tpr = np.mean(synthetic_scores > threshold)
                    fpr = np.mean(normal_scores > threshold)
                    tpr_at_thresholds[f'{percentile}th_percentile'] = {'tpr': tpr, 'fpr': fpr}
                
                synthetic_validation[synthetic_feature] = tpr_at_thresholds
    
    test_results['synthetic_validation'] = synthetic_validation
    
    # Print synthetic anomaly detection rates
    for anomaly_type, validation in synthetic_validation.items():
        print(f"  {anomaly_type.replace('synthetic_', '').title()} Detection:")
        for threshold, rates in validation.items():
            print(f"    {threshold}: TPR={rates['tpr']:.3f}, FPR={rates['fpr']:.3f}")
    
    # 6. Overall model assessment
    print("\n6. Overall Model Quality Assessment")
    print("-" * 35)
    
    # Calculate composite quality score
    quality_metrics = {
        'stability_score': 1 - min(score_stability, 1.0),  # Lower is better for stability
        'reliability_score': np.mean(list(detector_reliability.values())),
        'feature_consistency': 1 - np.mean(list(feature_rank_stability.values())) / len(detector.feature_selector),
        'synthetic_detection_rate': np.mean([
            val['95th_percentile']['tpr'] 
            for val in synthetic_validation.values() 
            if '95th_percentile' in val
        ]) if synthetic_validation else 0.5
    }
    
    overall_quality = np.mean(list(quality_metrics.values()))
    test_results['overall_quality'] = {
        'individual_metrics': quality_metrics,
        'composite_score': overall_quality
    }
    
    print(f"  Model Quality Metrics:")
    for metric, score in quality_metrics.items():
        print(f"    {metric.replace('_', ' ').title()}: {score:.3f}")
    print(f"  Composite Quality Score: {overall_quality:.3f}")
    
    # Quality assessment
    if overall_quality > 0.8:
        quality_rating = "EXCELLENT"
    elif overall_quality > 0.7:
        quality_rating = "GOOD"
    elif overall_quality > 0.6:
        quality_rating = "ACCEPTABLE"
    else:
        quality_rating = "NEEDS IMPROVEMENT"
    
    print(f"  Overall Rating: {quality_rating}")
    
    print("\n" + "="*60)
    print("MODEL TESTING COMPLETE")
    print("="*60)
    
    return test_results

# Main execution for training and testing
if __name__ == "__main__":
    try:
        # Train the model
        detector, ensemble_scores, metrics, model_path = main_advanced_training_pipeline()
        
        # Get the scaled features for testing
        feature_engineer = AdvancedFeatureEngineer()
        orders_df, lob_df, trades_df = feature_engineer.load_market_data("enhanced_market_data")
        features_df = feature_engineer.extract_optimized_features(orders_df, lob_df, trades_df)
        X_scaled, selected_features = detector.prepare_features_with_dimensionality_reduction(features_df)
        
        # Test the model
        test_results = test_model_performance(detector, X_scaled, features_df, ensemble_scores)
        
        # Save test results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_results_path = f"test_results_{timestamp}.joblib"
        joblib.dump(test_results, test_results_path)
        
        print(f"\nTest results saved to: {test_results_path}")
        print(f"Model ready for deployment. Use model file: {model_path}")
        
    except Exception as e:
        print(f"Error during training/testing: {e}")
        import traceback
        traceback.print_exc()