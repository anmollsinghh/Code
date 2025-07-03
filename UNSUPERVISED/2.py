"""
Enhanced Production Unsupervised Toxicity Detection System - FINAL FIXED VERSION
Fixed all issues including rolling rank, Series comparison, and GMM fitting errors
Uses only publicly observable market data for real-world deployment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA, FastICA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import ParameterGrid
from scipy import stats
from scipy.spatial.distance import cdist
import joblib
import glob
import os
from itertools import combinations
import optuna
from optuna.samplers import TPESampler

class EnhancedFeatureEngineer:
    
    def __init__(self):
        self.feature_names = []
        self.feature_importance_scores = {}
        
    def load_and_merge_data(self, data_dir="enhanced_market_data"):
        """Load market data ensuring only publicly observable features"""
        print("Loading market data...")
        
        order_files = glob.glob(f"{data_dir}/orders_*.csv")
        lob_files = glob.glob(f"{data_dir}/lob_snapshots_*.csv")
        trade_files = glob.glob(f"{data_dir}/trades_*.csv")
        
        if not order_files:
            raise FileNotFoundError(f"No order files found in {data_dir}")
        
        latest_order_file = max(order_files, key=os.path.getctime)
        orders_df = pd.read_csv(latest_order_file)
        
        # Only use publicly observable order features
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
            # Only use publicly observable trade features
            public_trade_columns = ['timestamp', 'price', 'quantity']
            available_trade_columns = [col for col in public_trade_columns if col in trades_df.columns]
            trades_df = trades_df[available_trade_columns].copy()
            print(f"Trades data: {len(trades_df)} trades")
        
        return orders_df, lob_df, trades_df
    
    def engineer_advanced_features(self, orders_df, lob_df, trades_df):
        """Enhanced feature engineering with more sophisticated features"""
        print("Engineering advanced features from public market data...")
        
        features_df = pd.DataFrame(index=orders_df.index)
        
        # Basic order features
        features_df['order_size'] = orders_df['quantity']
        features_df['log_order_size'] = np.log1p(orders_df['quantity'])
        features_df['sqrt_order_size'] = np.sqrt(orders_df['quantity'])
        features_df['is_market_order'] = (orders_df['order_type'] == 'MARKET').astype(int)
        features_df['is_buy_order'] = (orders_df['side'] == 'BUY').astype(int)
        
        # Price features with more variations
        if 'mid_price' in orders_df.columns:
            mid_price = orders_df['mid_price']
            features_df['mid_price'] = mid_price
            features_df['log_mid_price'] = np.log(mid_price + 1e-8)
            
            if 'price' in orders_df.columns:
                order_price = orders_df['price'].fillna(mid_price)
                features_df['price_deviation'] = (order_price - mid_price) / (mid_price + 1e-8)
                features_df['abs_price_deviation'] = np.abs(features_df['price_deviation'])
                features_df['squared_price_deviation'] = features_df['price_deviation'] ** 2
        
        # Enhanced spread features
        if 'spread' in orders_df.columns:
            spread = orders_df['spread']
            features_df['spread'] = spread
            features_df['relative_spread'] = spread / (mid_price + 1e-8)
            features_df['log_spread'] = np.log1p(spread)
            # Fixed: Calculate rolling percentile using quantile
            features_df['spread_percentile'] = spread.rolling(100, min_periods=1).apply(
                lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
            )
        
        # Aggressiveness features
        if 'is_aggressive' in orders_df.columns:
            features_df['is_aggressive'] = orders_df['is_aggressive'].astype(int)
        elif 'distance_from_mid' in orders_df.columns:
            features_df['is_aggressive'] = (np.abs(orders_df['distance_from_mid']) < 0.001).astype(int)
            features_df['distance_from_mid'] = orders_df['distance_from_mid']
            features_df['abs_distance_from_mid'] = np.abs(orders_df['distance_from_mid'])
        
        # Enhanced temporal features
        if 'timestamp' in orders_df.columns:
            timestamps = pd.to_datetime(orders_df['timestamp'])
            
            # Time-based features
            features_df['hour_of_day'] = timestamps.dt.hour
            features_df['minute_of_hour'] = timestamps.dt.minute
            features_df['day_of_week'] = timestamps.dt.dayofweek
            
            # Cyclical encoding for time features
            features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour_of_day'] / 24)
            features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour_of_day'] / 24)
            features_df['minute_sin'] = np.sin(2 * np.pi * features_df['minute_of_hour'] / 60)
            features_df['minute_cos'] = np.cos(2 * np.pi * features_df['minute_of_hour'] / 60)
            
            # Inter-arrival time features
            time_diffs = orders_df['timestamp'].diff().fillna(1)
            features_df['inter_arrival_time'] = time_diffs
            features_df['log_inter_arrival_time'] = np.log1p(time_diffs)
            features_df['arrival_rate'] = 1 / (time_diffs + 1e-8)
            
            # Multiple time window arrival intensities
            for window in [5, 10, 20, 50, 100]:
                features_df[f'arrival_intensity_{window}'] = features_df['arrival_rate'].rolling(
                    window, min_periods=1).mean()
                features_df[f'arrival_volatility_{window}'] = features_df['arrival_rate'].rolling(
                    window, min_periods=1).std()
        
        # Enhanced market microstructure features
        if 'volatility' in orders_df.columns:
            vol = orders_df['volatility']
            features_df['volatility'] = vol
            features_df['log_volatility'] = np.log1p(vol)
            # Fixed: Calculate rolling percentile
            features_df['vol_percentile'] = vol.rolling(100, min_periods=1).apply(
                lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
            )
        else:
            if 'mid_price' in orders_df.columns:
                returns = mid_price.pct_change().fillna(0)
                for window in [5, 10, 20, 50]:
                    features_df[f'volatility_{window}'] = returns.rolling(window, min_periods=1).std()
                    # Fixed: Calculate rolling percentile
                    features_df[f'volatility_percentile_{window}'] = features_df[f'volatility_{window}'].rolling(
                        100, min_periods=1).apply(
                        lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
                    )
        
        # Enhanced momentum features
        if 'momentum' in orders_df.columns:
            mom = orders_df['momentum']
            features_df['momentum'] = mom
            features_df['abs_momentum'] = np.abs(mom)
            features_df['momentum_sign'] = np.sign(mom)
        else:
            if 'mid_price' in orders_df.columns:
                for period in [3, 5, 10, 20]:
                    mom = mid_price.pct_change(period).fillna(0)
                    features_df[f'momentum_{period}'] = mom
                    features_df[f'abs_momentum_{period}'] = np.abs(mom)
                    features_df[f'momentum_sign_{period}'] = np.sign(mom)
        
        # Order book imbalance features
        if 'order_book_imbalance' in orders_df.columns:
            imbalance = orders_df['order_book_imbalance']
            features_df['order_book_imbalance'] = imbalance
            features_df['abs_imbalance'] = np.abs(imbalance)
            features_df['imbalance_sign'] = np.sign(imbalance)
            # Fixed: Calculate rolling percentile
            features_df['imbalance_percentile'] = imbalance.rolling(100, min_periods=1).apply(
                lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
            )
        
        # Enhanced LOB features
        if not lob_df.empty:
            lob_features = self._extract_enhanced_lob_features(lob_df, orders_df)
            for col in lob_features.columns:
                if col not in features_df.columns:
                    features_df[col] = lob_features[col]
        
        # Enhanced trade features
        if not trades_df.empty:
            trade_features = self._extract_enhanced_trade_features(trades_df, orders_df)
            for col in trade_features.columns:
                if col not in features_df.columns:
                    features_df[col] = trade_features[col]
        
        # Multi-scale rolling features
        features_df = self._add_multi_scale_rolling_features(features_df)
        
        # Advanced interaction features
        features_df = self._add_advanced_interaction_features(features_df)
        
        # Technical indicators
        features_df = self._add_technical_indicators(features_df)
        
        # Statistical features
        features_df = self._add_statistical_features(features_df)
        
        features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)
        self.feature_names = features_df.columns.tolist()
        
        print(f"Generated {len(self.feature_names)} advanced public market features")
        return features_df
    
    def _extract_enhanced_lob_features(self, lob_df, orders_df):
        """Extract more sophisticated LOB features"""
        lob_features = pd.DataFrame(index=orders_df.index)
        
        if 'timestamp' in lob_df.columns and 'timestamp' in orders_df.columns:
            merged = pd.merge_asof(
                orders_df[['timestamp']].reset_index(),
                lob_df,
                on='timestamp',
                direction='backward'
            ).set_index('index')
            
            # Multi-level imbalance features
            for level in range(1, 6):
                bid_price_col = f'bid_price_{level}'
                ask_price_col = f'ask_price_{level}'
                bid_size_col = f'bid_size_{level}'
                ask_size_col = f'ask_size_{level}'
                
                if all(col in merged.columns for col in [bid_price_col, ask_price_col, bid_size_col, ask_size_col]):
                    bid_size = merged[bid_size_col].fillna(0)
                    ask_size = merged[ask_size_col].fillna(0)
                    bid_price = merged[bid_price_col].fillna(0)
                    ask_price = merged[ask_price_col].fillna(0)
                    
                    total_size = bid_size + ask_size
                    level_imbalance = (bid_size - ask_size) / (total_size + 1e-8)
                    
                    lob_features[f'imbalance_L{level}'] = level_imbalance
                    lob_features[f'bid_depth_L{level}'] = bid_size
                    lob_features[f'ask_depth_L{level}'] = ask_size
                    lob_features[f'depth_ratio_L{level}'] = bid_size / (ask_size + 1e-8)
                    lob_features[f'price_level_{level}'] = (ask_price + bid_price) / 2
                    
                    if level > 1:
                        lob_features[f'spread_L{level}'] = ask_price - bid_price
                        lob_features[f'relative_spread_L{level}'] = (ask_price - bid_price) / (
                            (ask_price + bid_price) / 2 + 1e-8)
            
            # Aggregate depth features
            total_bid_depth = pd.Series(0, index=merged.index)
            total_ask_depth = pd.Series(0, index=merged.index)
            
            for i in range(1, 6):
                if f'bid_size_{i}' in merged.columns:
                    total_bid_depth += merged[f'bid_size_{i}'].fillna(0)
                if f'ask_size_{i}' in merged.columns:
                    total_ask_depth += merged[f'ask_size_{i}'].fillna(0)
            
            lob_features['total_bid_depth'] = total_bid_depth
            lob_features['total_ask_depth'] = total_ask_depth
            lob_features['total_depth'] = total_bid_depth + total_ask_depth
            lob_features['depth_ratio'] = total_bid_depth / (total_ask_depth + 1e-8)
            lob_features['depth_imbalance'] = (total_bid_depth - total_ask_depth) / (
                total_bid_depth + total_ask_depth + 1e-8)
            
            # Weighted average prices (vectorized)
            weighted_bid_sum = pd.Series(0.0, index=merged.index)
            weighted_ask_sum = pd.Series(0.0, index=merged.index)
            
            for i in range(1, 6):
                if f'bid_price_{i}' in merged.columns and f'bid_size_{i}' in merged.columns:
                    weighted_bid_sum += merged[f'bid_price_{i}'].fillna(0) * merged[f'bid_size_{i}'].fillna(0)
                if f'ask_price_{i}' in merged.columns and f'ask_size_{i}' in merged.columns:
                    weighted_ask_sum += merged[f'ask_price_{i}'].fillna(0) * merged[f'ask_size_{i}'].fillna(0)
            
            # Calculate weighted average prices with proper handling of zero depth
            lob_features['weighted_bid_price'] = weighted_bid_sum / (total_bid_depth + 1e-8)
            lob_features['weighted_ask_price'] = weighted_ask_sum / (total_ask_depth + 1e-8)
            
            # Replace inf values with 0
            lob_features['weighted_bid_price'] = lob_features['weighted_bid_price'].replace([np.inf, -np.inf], 0)
            lob_features['weighted_ask_price'] = lob_features['weighted_ask_price'].replace([np.inf, -np.inf], 0)
        
        return lob_features.fillna(0)
    
    def _extract_enhanced_trade_features(self, trades_df, orders_df):
        """Extract more sophisticated trade-based features"""
        trade_features = pd.DataFrame(index=orders_df.index)
        
        if 'timestamp' in trades_df.columns and 'timestamp' in orders_df.columns:
            for idx, order in orders_df.iterrows():
                timestamp = order['timestamp']
                
                # Multiple time windows for trade analysis
                for window in [5, 10, 30]:
                    recent_trades = trades_df[
                        (trades_df['timestamp'] >= timestamp - window) & 
                        (trades_df['timestamp'] <= timestamp)
                    ]
                    
                    if not recent_trades.empty:
                        trade_features.loc[idx, f'trade_volume_{window}'] = recent_trades['quantity'].sum()
                        trade_features.loc[idx, f'trade_count_{window}'] = len(recent_trades)
                        trade_features.loc[idx, f'avg_trade_size_{window}'] = recent_trades['quantity'].mean()
                        trade_features.loc[idx, f'trade_size_std_{window}'] = recent_trades['quantity'].std()
                        
                        if len(recent_trades) > 1:
                            price_returns = recent_trades['price'].pct_change().dropna()
                            if len(price_returns) > 0:
                                trade_features.loc[idx, f'trade_volatility_{window}'] = price_returns.std()
                        
                        time_span = recent_trades['timestamp'].max() - recent_trades['timestamp'].min()
                        if time_span > 0:
                            trade_features.loc[idx, f'trade_frequency_{window}'] = len(recent_trades) / time_span
                        
                        # VWAP calculation
                        if recent_trades['quantity'].sum() > 0:
                            vwap = (recent_trades['price'] * recent_trades['quantity']).sum() / recent_trades['quantity'].sum()
                            trade_features.loc[idx, f'vwap_{window}'] = vwap
        
        return trade_features.fillna(0)
    
    def _add_multi_scale_rolling_features(self, features_df):
        """Add rolling features across multiple time scales"""
        key_features = ['order_size', 'relative_spread', 'volatility', 'momentum', 'order_book_imbalance']
        available_features = [f for f in key_features if f in features_df.columns]
        
        for feature in available_features:
            for window in [5, 10, 20, 50, 100]:
                # Basic rolling statistics
                features_df[f'{feature}_ma_{window}'] = features_df[feature].rolling(window, min_periods=1).mean()
                features_df[f'{feature}_std_{window}'] = features_df[feature].rolling(window, min_periods=1).std()
                features_df[f'{feature}_min_{window}'] = features_df[feature].rolling(window, min_periods=1).min()
                features_df[f'{feature}_max_{window}'] = features_df[feature].rolling(window, min_periods=1).max()
                
                # Z-score and percentile features
                ma_col = f'{feature}_ma_{window}'
                std_col = f'{feature}_std_{window}'
                features_df[f'{feature}_zscore_{window}'] = (
                    (features_df[feature] - features_df[ma_col]) / (features_df[std_col] + 1e-8)
                )
                
                # Fixed: Calculate rolling percentile manually
                features_df[f'{feature}_percentile_{window}'] = features_df[feature].rolling(
                    window, min_periods=1).apply(
                    lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
                )
                
                # Trend features
                if window >= 10:
                    features_df[f'{feature}_trend_{window}'] = (
                        features_df[feature] - features_df[feature].shift(window // 2)
                    ) / (features_df[feature].shift(window // 2) + 1e-8)
        
        return features_df
    
    def _add_advanced_interaction_features(self, features_df):
        """Add sophisticated interaction features"""
        # Size-based interactions
        if 'order_size' in features_df.columns:
            for feature in ['relative_spread', 'volatility', 'abs_momentum', 'abs_imbalance']:
                if feature in features_df.columns:
                    features_df[f'size_{feature}_interaction'] = features_df['order_size'] * features_df[feature]
                    features_df[f'log_size_{feature}_interaction'] = features_df['log_order_size'] * features_df[feature]
        
        # Directional interactions
        if 'is_buy_order' in features_df.columns:
            directional_multiplier = 2 * features_df['is_buy_order'] - 1
            for feature in ['order_book_imbalance', 'momentum', 'price_deviation']:
                if feature in features_df.columns:
                    features_df[f'{feature}_directional'] = features_df[feature] * directional_multiplier
        
        # Volatility-based interactions
        if 'volatility' in features_df.columns:
            for feature in ['is_aggressive', 'relative_spread', 'abs_momentum']:
                if feature in features_df.columns:
                    features_df[f'vol_{feature}_interaction'] = features_df['volatility'] * features_df[feature]
        
        # Time-based interactions
        if 'hour_of_day' in features_df.columns:
            # Market session indicators
            features_df['is_opening_session'] = ((features_df['hour_of_day'] >= 9) & 
                                               (features_df['hour_of_day'] < 11)).astype(int)
            features_df['is_closing_session'] = ((features_df['hour_of_day'] >= 15) & 
                                               (features_df['hour_of_day'] < 17)).astype(int)
            
            for feature in ['order_size', 'volatility', 'relative_spread']:
                if feature in features_df.columns:
                    features_df[f'{feature}_opening_interaction'] = (
                        features_df[feature] * features_df['is_opening_session']
                    )
                    features_df[f'{feature}_closing_interaction'] = (
                        features_df[feature] * features_df['is_closing_session']
                    )
        
        return features_df
    
    def _add_technical_indicators(self, features_df):
        """Add technical analysis indicators"""
        if 'mid_price' in features_df.columns:
            mid_price = features_df['mid_price']
            
            # Simple moving averages
            for period in [5, 10, 20]:
                sma = mid_price.rolling(period, min_periods=1).mean()
                features_df[f'sma_{period}'] = sma
                features_df[f'price_sma_ratio_{period}'] = mid_price / (sma + 1e-8)
                features_df[f'price_above_sma_{period}'] = (mid_price > sma).astype(int)
            
            # Exponential moving averages
            for span in [5, 10, 20]:
                ema = mid_price.ewm(span=span).mean()
                features_df[f'ema_{span}'] = ema
                features_df[f'price_ema_ratio_{span}'] = mid_price / (ema + 1e-8)
            
            # Bollinger Bands
            for period in [10, 20]:
                rolling_mean = mid_price.rolling(period, min_periods=1).mean()
                rolling_std = mid_price.rolling(period, min_periods=1).std()
                
                upper_band = rolling_mean + (rolling_std * 2)
                lower_band = rolling_mean - (rolling_std * 2)
                
                features_df[f'bb_upper_{period}'] = upper_band
                features_df[f'bb_lower_{period}'] = lower_band
                features_df[f'bb_position_{period}'] = (mid_price - lower_band) / (upper_band - lower_band + 1e-8)
                features_df[f'bb_squeeze_{period}'] = (upper_band - lower_band) / (rolling_mean + 1e-8)
            
            # RSI-like indicators
            for period in [5, 14]:
                delta = mid_price.diff()
                gain = (delta.where(delta > 0, 0)).rolling(period, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=1).mean()
                rs = gain / (loss + 1e-8)
                features_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        return features_df
    
    def _add_statistical_features(self, features_df):
        """Add statistical distribution features"""
        numerical_features = features_df.select_dtypes(include=[np.number]).columns
        
        for feature in numerical_features[:10]:  # Limit to prevent feature explosion
            if features_df[feature].std() > 0:
                # Distribution moments
                features_df[f'{feature}_skewness'] = features_df[feature].rolling(50, min_periods=10).skew()
                features_df[f'{feature}_kurtosis'] = features_df[feature].rolling(50, min_periods=10).kurt()
                
                # Quantile features
                for q in [0.1, 0.25, 0.75, 0.9]:
                    features_df[f'{feature}_q{int(q*100)}'] = features_df[feature].rolling(
                        50, min_periods=10).quantile(q)
        
        return features_df

class AdvancedToxicityDetector:
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.ensemble_weights = {}
        self.performance_metrics = {}
        self.best_hyperparameters = {}
        
    def prepare_features_with_selection(self, features_df, correlation_threshold=0.95, 
                                      variance_threshold=0.01):
        """Enhanced feature preparation with multiple selection methods"""
        print("Preparing features with advanced selection...")
        
        # Remove low variance features
        from sklearn.feature_selection import VarianceThreshold
        var_selector = VarianceThreshold(threshold=variance_threshold)
        high_var_features = features_df.loc[:, var_selector.fit(features_df).get_support()]
        print(f"Removed {len(features_df.columns) - len(high_var_features.columns)} low variance features")
        
        # Remove highly correlated features
        selected_features = self._remove_correlated_features(high_var_features, correlation_threshold)
        
        # Test different scalers
        scaler_performance = self._evaluate_scalers(selected_features)
        best_scaler_name = max(scaler_performance, key=scaler_performance.get)
        
        print(f"Selected scaler: {best_scaler_name}")
        
        if best_scaler_name == 'robust':
            scaler = RobustScaler()
        elif best_scaler_name == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        X_scaled = scaler.fit_transform(selected_features)
        
        self.scalers['main'] = scaler
        self.feature_selector = selected_features.columns.tolist()
        
        print(f"Selected {len(self.feature_selector)} features after advanced selection")
        
        return X_scaled, selected_features
    
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
                
                # Quick clustering evaluation
                kmeans = KMeans(n_clusters=min(5, len(features_df) // 100), random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                if len(set(labels)) > 1:
                    score = silhouette_score(X_scaled, labels)
                    scaler_scores[name] = score
                else:
                    scaler_scores[name] = -1
                    
            except Exception:
                scaler_scores[name] = -1
        
        return scaler_scores
    
    def _remove_correlated_features(self, features_df, threshold=0.95):
        """Enhanced correlation-based feature removal"""
        corr_matrix = features_df.corr().abs()
        
        # Create mask for upper triangle
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop based on correlation
        features_to_drop = set()
        for column in upper_triangle.columns:
            correlated_features = upper_triangle.index[upper_triangle[column] > threshold].tolist()
            if correlated_features:
                # Keep the feature with higher variance
                variances = {feat: features_df[feat].var() for feat in correlated_features + [column]}
                features_to_keep = max(variances, key=variances.get)
                features_to_drop.update([f for f in correlated_features if f != features_to_keep])
        
        selected_features = features_df.drop(columns=list(features_to_drop))
        print(f"Removed {len(features_to_drop)} highly correlated features")
        
        return selected_features
    
    def optimize_hyperparameters(self, X_scaled, n_trials=100):
        """Use Optuna for hyperparameter optimization"""
        
        print(f"Optimizing hyperparameters with {n_trials} trials...")
        
        def objective(trial):
            # Sample hyperparameters
            contamination = trial.suggest_float('contamination', 0.01, 0.2)
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_samples = trial.suggest_categorical('max_samples', ['auto', 0.5, 0.7, 0.9])
            n_neighbors = trial.suggest_int('n_neighbors', 5, 50)
            n_clusters = trial.suggest_int('n_clusters', 3, min(20, len(X_scaled) // 20))
            
            try:
                # Train multiple detectors with these parameters
                iso_forest = IsolationForest(
                    contamination=contamination,
                    n_estimators=n_estimators,
                    max_samples=max_samples,
                    random_state=42,
                    bootstrap=True
                )
                iso_forest.fit(X_scaled)
                
                lof = LocalOutlierFactor(
                    n_neighbors=n_neighbors,
                    contamination=contamination,
                    novelty=True
                )
                lof.fit(X_scaled)
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                
                # Calculate ensemble score
                iso_scores = -iso_forest.decision_function(X_scaled)
                lof_scores = -lof.score_samples(X_scaled)
                cluster_distances = np.min(kmeans.transform(X_scaled), axis=1)
                
                # Normalize scores
                iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8)
                lof_scores = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-8)
                cluster_scores = (cluster_distances - cluster_distances.min()) / (cluster_distances.max() - cluster_distances.min() + 1e-8)
                
                ensemble_scores = (iso_scores + lof_scores + cluster_scores) / 3
                
                # Evaluate clustering quality
                if len(set(cluster_labels)) > 1:
                    silhouette = silhouette_score(X_scaled, cluster_labels)
                    calinski = calinski_harabasz_score(X_scaled, cluster_labels)
                    davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
                    
                    # Combine multiple metrics (higher is better for silhouette and calinski, lower for davies_bouldin)
                    score = silhouette + (calinski / 1000) - davies_bouldin
                else:
                    score = -1
                
                # Add ensemble diversity bonus
                score_correlations = np.corrcoef([iso_scores, lof_scores, cluster_scores])
                avg_correlation = np.mean(score_correlations[np.triu_indices_from(score_correlations, k=1)])
                diversity_bonus = 1 - abs(avg_correlation)
                score += diversity_bonus
                
                return score
                
            except Exception:
                return -10
        
        # Run optimization
        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_hyperparameters = study.best_params
        print(f"Best hyperparameters: {self.best_hyperparameters}")
        print(f"Best score: {study.best_value:.4f}")
        
        return self.best_hyperparameters
    
    def train_optimized_ensemble_detectors(self, X_scaled, hyperparameters=None):
        """Train ensemble using optimized hyperparameters"""
        print("Training optimized ensemble detectors...")
        
        if hyperparameters is None:
            hyperparameters = self.best_hyperparameters
        
        detectors = {}
        
        # Isolation Forest with optimized parameters
        iso_forest = IsolationForest(
            contamination=hyperparameters.get('contamination', 0.05),
            n_estimators=hyperparameters.get('n_estimators', 200),
            max_samples=hyperparameters.get('max_samples', 'auto'),
            random_state=42,
            bootstrap=True
        )
        iso_forest.fit(X_scaled)
        detectors['isolation_forest_optimized'] = iso_forest
        
        # LOF with optimized parameters
        lof = LocalOutlierFactor(
            n_neighbors=hyperparameters.get('n_neighbors', 20),
            contamination=hyperparameters.get('contamination', 0.05),
            novelty=True
        )
        lof.fit(X_scaled)
        detectors['lof_optimized'] = lof
        
        # Multiple contamination rates for robustness
        for contamination in [0.03, 0.07, 0.1]:
            # Additional Isolation Forest
            iso_forest_multi = IsolationForest(
                contamination=contamination,
                n_estimators=hyperparameters.get('n_estimators', 200),
                random_state=42 + int(contamination * 100),
                bootstrap=True
            )
            iso_forest_multi.fit(X_scaled)
            detectors[f'isolation_forest_{contamination}'] = iso_forest_multi
            
            # Elliptic Envelope
            try:
                elliptic = EllipticEnvelope(
                    contamination=contamination,
                    random_state=42
                )
                elliptic.fit(X_scaled)
                detectors[f'elliptic_{contamination}'] = elliptic
            except Exception as e:
                print(f"Warning: Failed to train Elliptic Envelope with contamination {contamination}: {e}")
        
        # Gaussian Mixture Models with regularization
        for n_components in [3, 5, 8]:
            try:
                n_comp = min(n_components, len(X_scaled) // 50)
                if n_comp < 2:
                    continue
                    
                gmm = GaussianMixture(
                    n_components=n_comp,
                    random_state=42,
                    covariance_type='full',
                    reg_covar=1e-4  # Add regularization to prevent singular covariance
                )
                gmm.fit(X_scaled)
                detectors[f'gmm_{n_comp}'] = gmm
            except Exception as e:
                print(f"Warning: Failed to train GMM with {n_comp} components: {e}")
        
        # Bayesian Gaussian Mixture with regularization
        try:
            n_comp = min(10, len(X_scaled) // 30)
            if n_comp >= 2:
                bgmm = BayesianGaussianMixture(
                    n_components=n_comp,
                    random_state=42,
                    covariance_type='full',
                    reg_covar=1e-4  # Add regularization
                )
                bgmm.fit(X_scaled)
                detectors['bayesian_gmm'] = bgmm
        except Exception as e:
            print(f"Warning: Failed to train Bayesian GMM: {e}")
        
        # Multiple clustering approaches
        clustering_detectors = self._train_advanced_clustering_detectors(X_scaled, hyperparameters)
        detectors.update(clustering_detectors)
        
        # DBSCAN-based detector
        dbscan_detector = self._train_dbscan_detector(X_scaled)
        if dbscan_detector:
            detectors['dbscan'] = dbscan_detector
        
        self.models = detectors
        print(f"Trained {len(detectors)} optimized detectors")
        
        return detectors
    
    def _train_advanced_clustering_detectors(self, X_scaled, hyperparameters):
        """Train multiple advanced clustering-based detectors"""
        clustering_detectors = {}
        
        # Optimized K-means
        best_k = hyperparameters.get('n_clusters', self._optimize_kmeans_clusters(X_scaled))
        
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(X_scaled)
        distances = np.min(kmeans.transform(X_scaled), axis=1)
        
        clustering_detectors['kmeans_optimized'] = {
            'kmeans': kmeans,
            'distance_threshold': np.percentile(distances, 95),
            'cluster_sizes': np.bincount(cluster_labels)
        }
        
        # Agglomerative clustering
        for linkage in ['ward', 'complete']:
            try:
                agg_clustering = AgglomerativeClustering(
                    n_clusters=best_k,
                    linkage=linkage
                )
                agg_labels = agg_clustering.fit_predict(X_scaled)
                
                # Calculate distances to cluster centres
                cluster_centres = []
                for i in range(best_k):
                    cluster_points = X_scaled[agg_labels == i]
                    if len(cluster_points) > 0:
                        cluster_centres.append(np.mean(cluster_points, axis=0))
                
                if cluster_centres:
                    cluster_centres = np.array(cluster_centres)
                    distances = np.min(cdist(X_scaled, cluster_centres), axis=1)
                    
                    clustering_detectors[f'agglomerative_{linkage}'] = {
                        'cluster_centres': cluster_centres,
                        'distance_threshold': np.percentile(distances, 95),
                        'labels': agg_labels
                    }
                    
            except Exception as e:
                print(f"Failed to train agglomerative clustering with {linkage}: {e}")
                continue
        
        return clustering_detectors
    
    def _train_dbscan_detector(self, X_scaled):
        """Train DBSCAN-based anomaly detector"""
        print("Training DBSCAN detector...")
        
        # Optimize DBSCAN parameters
        best_score = -1
        best_params = None
        
        # Parameter search for DBSCAN
        eps_values = np.logspace(-2, 1, 10)
        min_samples_values = [3, 5, 10, 15, 20]
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(X_scaled)
                    
                    # Count outliers (label -1)
                    n_outliers = np.sum(labels == -1)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    
                    # Good DBSCAN should have reasonable number of clusters and outliers
                    if 2 <= n_clusters <= 10 and 0.01 <= n_outliers / len(X_scaled) <= 0.15:
                        # Calculate silhouette score for non-outlier points
                        non_outlier_mask = labels != -1
                        if np.sum(non_outlier_mask) > 10 and len(set(labels[non_outlier_mask])) > 1:
                            score = silhouette_score(X_scaled[non_outlier_mask], labels[non_outlier_mask])
                            if score > best_score:
                                best_score = score
                                best_params = {'eps': eps, 'min_samples': min_samples}
                                
                except Exception:
                    continue
        
        if best_params:
            dbscan = DBSCAN(**best_params)
            labels = dbscan.fit_predict(X_scaled)
            
            return {
                'dbscan': dbscan,
                'labels': labels,
                'params': best_params
            }
        
        return None
    
    def _optimize_kmeans_clusters(self, X_scaled, max_k=20):
        """Enhanced K-means cluster optimization using multiple metrics"""
        k_range = range(2, min(max_k, len(X_scaled) // 20))
        
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        inertias = []
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                if len(set(labels)) > 1:
                    sil_score = silhouette_score(X_scaled, labels)
                    cal_score = calinski_harabasz_score(X_scaled, labels)
                    db_score = davies_bouldin_score(X_scaled, labels)
                    
                    silhouette_scores.append(sil_score)
                    calinski_scores.append(cal_score)
                    davies_bouldin_scores.append(db_score)
                    inertias.append(kmeans.inertia_)
                else:
                    silhouette_scores.append(-1)
                    calinski_scores.append(0)
                    davies_bouldin_scores.append(10)
                    inertias.append(float('inf'))
                    
            except Exception:
                silhouette_scores.append(-1)
                calinski_scores.append(0)
                davies_bouldin_scores.append(10)
                inertias.append(float('inf'))
        
        # Normalize scores and combine
        if max(silhouette_scores) > min(silhouette_scores):
            norm_sil = [(s - min(silhouette_scores)) / (max(silhouette_scores) - min(silhouette_scores)) 
                       for s in silhouette_scores]
        else:
            norm_sil = [0] * len(silhouette_scores)
        
        if max(calinski_scores) > min(calinski_scores):
            norm_cal = [(s - min(calinski_scores)) / (max(calinski_scores) - min(calinski_scores)) 
                       for s in calinski_scores]
        else:
            norm_cal = [0] * len(calinski_scores)
        
        if max(davies_bouldin_scores) > min(davies_bouldin_scores):
            norm_db = [1 - (s - min(davies_bouldin_scores)) / (max(davies_bouldin_scores) - min(davies_bouldin_scores)) 
                      for s in davies_bouldin_scores]
        else:
            norm_db = [1] * len(davies_bouldin_scores)
        
        # Combined score (equal weights)
        combined_scores = [(s + c + d) / 3 for s, c, d in zip(norm_sil, norm_cal, norm_db)]
        
        best_k_idx = np.argmax(combined_scores)
        best_k = list(k_range)[best_k_idx]
        
        print(f"Optimal number of clusters: {best_k}")
        print(f"  Silhouette score: {silhouette_scores[best_k_idx]:.3f}")
        print(f"  Calinski-Harabasz score: {calinski_scores[best_k_idx]:.3f}")
        print(f"  Davies-Bouldin score: {davies_bouldin_scores[best_k_idx]:.3f}")
        
        return best_k
    
    def calculate_weighted_ensemble_scores(self, X_scaled):
        """Calculate ensemble scores with adaptive weighting"""
        print("Calculating weighted ensemble anomaly scores...")
        
        individual_scores = {}
        individual_weights = {}
        
        # Calculate scores for each detector
        for name, model in self.models.items():
            try:
                if 'isolation_forest' in name:
                    scores = -model.decision_function(X_scaled)
                elif 'lof' in name:
                    scores = -model.score_samples(X_scaled)
                elif 'elliptic' in name:
                    scores = -model.decision_function(X_scaled)
                elif 'gmm' in name or 'bayesian_gmm' in name:
                    scores = -model.score_samples(X_scaled)
                elif 'kmeans' in name:
                    distances = np.min(model['kmeans'].transform(X_scaled), axis=1)
                    scores = distances
                elif 'agglomerative' in name:
                    distances = np.min(cdist(X_scaled, model['cluster_centres']), axis=1)
                    scores = distances
                elif 'dbscan' in name:
                    # For DBSCAN, calculate distance to nearest core point
                    labels = model['labels']
                    scores = np.zeros(len(X_scaled))
                    for i, point in enumerate(X_scaled):
                        if labels[i] == -1:  # Outlier
                            scores[i] = 1.0
                        else:
                            # Distance to cluster centre
                            cluster_points = X_scaled[labels == labels[i]]
                            cluster_centre = np.mean(cluster_points, axis=0)
                            scores[i] = np.linalg.norm(point - cluster_centre)
                else:
                    continue
                
                # Normalize scores
                if scores.max() > scores.min():
                    scores = (scores - scores.min()) / (scores.max() - scores.min())
                else:
                    scores = np.zeros_like(scores)
                
                individual_scores[name] = scores
                
                # Calculate weight based on performance
                weight = self._calculate_detector_weight(scores, X_scaled)
                individual_weights[name] = weight
                
            except Exception as e:
                print(f"Error calculating scores for {name}: {e}")
                continue
        
        # Normalize weights
        total_weight = sum(individual_weights.values())
        if total_weight > 0:
            individual_weights = {name: weight / total_weight 
                                for name, weight in individual_weights.items()}
        else:
            individual_weights = {name: 1.0 / len(individual_scores) 
                                for name in individual_scores}
        
        # Calculate weighted ensemble scores
        ensemble_scores = np.zeros(len(X_scaled))
        for name, scores in individual_scores.items():
            weight = individual_weights[name]
            ensemble_scores += weight * scores
        
        self.ensemble_weights = individual_weights
        
        print(f"Ensemble weights:")
        for name, weight in sorted(individual_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {weight:.3f}")
        
        return ensemble_scores, individual_scores
    
    def _calculate_detector_weight(self, scores, X_scaled):
        """Calculate weight for individual detector based on performance"""
        try:
            # Weight based on score distribution quality
            score_std = np.std(scores)
            score_range = np.max(scores) - np.min(scores)
            
            # Prefer detectors with good separation
            separation_score = score_std * score_range
            
            # Weight based on anomaly detection at different thresholds
            performance_scores = []
            for threshold_pct in [90, 95, 99]:
                threshold = np.percentile(scores, threshold_pct)
                anomaly_rate = np.mean(scores > threshold)
                expected_rate = (100 - threshold_pct) / 100
                
                # Prefer rates close to expected
                rate_score = 1 - abs(anomaly_rate - expected_rate) / expected_rate
                performance_scores.append(max(0, rate_score))
            
            avg_performance = np.mean(performance_scores)
            
            # Combine separation and performance
            weight = separation_score * avg_performance
            
            return max(0.1, weight)  # Minimum weight threshold
            
        except Exception:
            return 0.1
    
    def evaluate_enhanced_performance(self, X_scaled, ensemble_scores, individual_scores):
        """Enhanced performance evaluation with multiple metrics"""
        print("Evaluating enhanced detector performance...")
        
        metrics = {}
        
        # Clustering quality metrics
        if 'kmeans_optimized' in self.models:
            kmeans = self.models['kmeans_optimized']['kmeans']
            cluster_labels = kmeans.predict(X_scaled)
            
            if len(set(cluster_labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(X_scaled, cluster_labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(X_scaled, cluster_labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, cluster_labels)
        
        # Anomaly detection metrics at multiple thresholds
        anomaly_metrics = {}
        for threshold_pct in [90, 95, 97, 99]:
            threshold = np.percentile(ensemble_scores, threshold_pct)
            anomaly_labels = ensemble_scores > threshold
            anomaly_rate = anomaly_labels.mean()
            
            anomaly_metrics[f'anomaly_rate_{threshold_pct}th'] = anomaly_rate
            
            # Score statistics for anomalies vs normal
            if anomaly_labels.sum() > 0:
                anomaly_scores = ensemble_scores[anomaly_labels]
                normal_scores = ensemble_scores[~anomaly_labels]
                
                anomaly_metrics[f'anomaly_score_mean_{threshold_pct}th'] = anomaly_scores.mean()
                anomaly_metrics[f'normal_score_mean_{threshold_pct}th'] = normal_scores.mean()
                anomaly_metrics[f'separation_{threshold_pct}th'] = (
                    anomaly_scores.mean() - normal_scores.mean()
                ) / (normal_scores.std() + 1e-8)
        
        metrics['anomaly_detection'] = anomaly_metrics
        
        # Individual detector performance
        individual_performance = {}
        for name, scores in individual_scores.items():
            individual_performance[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max(),
                'skewness': stats.skew(scores),
                'kurtosis': stats.kurtosis(scores),
                'weight': self.ensemble_weights.get(name, 0)
            }
        
        metrics['individual_detectors'] = individual_performance
        
        # Ensemble quality metrics
        metrics['ensemble_statistics'] = {
            'score_mean': ensemble_scores.mean(),
            'score_std': ensemble_scores.std(),
            'score_range': ensemble_scores.max() - ensemble_scores.min(),
            'score_skewness': stats.skew(ensemble_scores),
            'score_kurtosis': stats.kurtosis(ensemble_scores),
            'effective_detectors': len([w for w in self.ensemble_weights.values() if w > 0.05])
        }
        
        # Diversity metrics
        if len(individual_scores) > 1:
            score_matrix = np.array(list(individual_scores.values())).T
            correlation_matrix = np.corrcoef(score_matrix.T)
            
            # Average pairwise correlation (lower is better for diversity)
            avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            metrics['ensemble_diversity'] = {
                'avg_correlation': avg_correlation,
                'diversity_score': 1 - abs(avg_correlation)
            }
        
        self.performance_metrics = metrics
        
        return metrics
    
    def save_enhanced_model(self, save_dir="enhanced_production_models"):
        """Save the enhanced model with comprehensive metadata"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{save_dir}/enhanced_toxicity_detector_{timestamp}.joblib"
        
        model_package = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_selector': self.feature_selector,
            'ensemble_weights': self.ensemble_weights,
            'performance_metrics': self.performance_metrics,
            'best_hyperparameters': self.best_hyperparameters,
            'timestamp': timestamp,
            'version': '2.0_enhanced',
            'n_features': len(self.feature_selector),
            'n_detectors': len(self.models)
        }
        
        joblib.dump(model_package, model_path)
        
        print(f"Enhanced model saved to: {model_path}")
        return model_path

def create_enhanced_visualizations(features_df, ensemble_scores, individual_scores, 
                                 detector, save_dir="enhanced_production_plots"):
    """Create comprehensive visualizations for the enhanced model"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Enhanced feature analysis
    plt.figure(figsize=(16, 12))
    
    # Feature correlation heatmap with clustering
    plt.subplot(2, 3, 1)
    # Select a subset of features for visualization
    feature_subset = features_df.columns[:20]
    correlation_matrix = features_df[feature_subset].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='RdYlBu_r', 
                center=0, square=True, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix (Top 20)', fontweight='bold')
    
    # Feature importance based on anomaly discrimination
    plt.subplot(2, 3, 2)
    anomaly_threshold = np.percentile(ensemble_scores, 95)
    anomaly_mask = ensemble_scores > anomaly_threshold
    
    if anomaly_mask.sum() > 0:
        feature_importance = {}
        for feature in features_df.columns[:20]:  # Top 20 features
            normal_mean = features_df[feature][~anomaly_mask].mean()
            anomaly_mean = features_df[feature][anomaly_mask].mean()
            normal_std = features_df[feature][~anomaly_mask].std()
            
            if normal_std > 0:
                importance = abs(anomaly_mean - normal_mean) / normal_std
                feature_importance[feature] = importance
        
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features_names = [f[0] for f in sorted_features]
            feature_scores = [f[1] for f in sorted_features]
            
            plt.barh(range(len(features_names)), feature_scores, color='steelblue', alpha=0.7)
            plt.yticks(range(len(features_names)), 
                      [name.replace('_', ' ')[:20] for name in features_names])
            plt.xlabel('Discrimination Score')
            plt.title('Top Discriminative Features')
            plt.grid(True, alpha=0.3, axis='x')
    
    # Ensemble score distribution with multiple thresholds
    plt.subplot(2, 3, 3)
    plt.hist(ensemble_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    
    percentiles = [90, 95, 99]
    colors = ['orange', 'red', 'darkred']
    for pct, color in zip(percentiles, colors):
        threshold = np.percentile(ensemble_scores, pct)
        plt.axvline(threshold, color=color, linestyle='--', linewidth=2, 
                   label=f'{pct}th Percentile')
    
    plt.xlabel('Ensemble Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Ensemble Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Detector weight visualization
    plt.subplot(2, 3, 4)
    weights = list(detector.ensemble_weights.values())
    detector_names = [name.replace('_', ' ')[:15] for name in detector.ensemble_weights.keys()]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(weights)))
    bars = plt.bar(range(len(weights)), weights, color=colors, alpha=0.7)
    plt.xticks(range(len(detector_names)), detector_names, rotation=45, ha='right')
    plt.ylabel('Ensemble Weight')
    plt.title('Detector Weights in Ensemble')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Time series of anomaly scores
    plt.subplot(2, 3, 5)
    plt.plot(ensemble_scores, alpha=0.7, color='blue', linewidth=0.8)
    plt.axhline(np.percentile(ensemble_scores, 95), color='red', linestyle='--', 
               label='95th Percentile')
    plt.axhline(np.percentile(ensemble_scores, 99), color='darkred', linestyle='--', 
               label='99th Percentile')
    plt.xlabel('Order Sequence')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Scores Timeline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Individual detector score correlations
    plt.subplot(2, 3, 6)
    if len(individual_scores) > 1:
        score_matrix = np.array(list(individual_scores.values())).T
        correlation_matrix = np.corrcoef(score_matrix.T)
        
        im = plt.imshow(correlation_matrix, cmap='RdYlBu_r', aspect='auto')
        plt.colorbar(im, shrink=0.8)
        
        detector_names_short = [name[:10] for name in individual_scores.keys()]
        plt.xticks(range(len(detector_names_short)), detector_names_short, rotation=45)
        plt.yticks(range(len(detector_names_short)), detector_names_short)
        plt.title('Detector Score Correlations')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/enhanced_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Advanced PCA visualization with anomaly highlighting
    if len(features_df.columns) > 2:
        plt.figure(figsize=(15, 10))
        
        # Prepare data for PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df.fillna(0))
        
        # Multi-component PCA analysis
        plt.subplot(2, 3, 1)
        pca = PCA()
        pca.fit(X_scaled)
        
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(range(1, min(21, len(cumvar) + 1)), cumvar[:20], 'bo-')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2D PCA with anomaly colouring
        plt.subplot(2, 3, 2)
        pca_2d = PCA(n_components=2)
        X_pca = pca_2d.fit_transform(X_scaled)
        
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=ensemble_scores, 
                             cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Anomaly Score')
        plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('PCA Space - Anomaly Intensity')
        plt.grid(True, alpha=0.3)
        
        # 3D PCA visualization
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.subplot(2, 3, 3, projection='3d')
        pca_3d = PCA(n_components=3)
        X_pca_3d = pca_3d.fit_transform(X_scaled)
        
        anomaly_threshold = np.percentile(ensemble_scores, 95)
        normal_mask = ensemble_scores <= anomaly_threshold
        anomaly_mask = ensemble_scores > anomaly_threshold
        
        ax.scatter(X_pca_3d[normal_mask, 0], X_pca_3d[normal_mask, 1], X_pca_3d[normal_mask, 2],
                  c='lightblue', alpha=0.6, s=10, label='Normal')
        if anomaly_mask.sum() > 0:
            ax.scatter(X_pca_3d[anomaly_mask, 0], X_pca_3d[anomaly_mask, 1], X_pca_3d[anomaly_mask, 2],
                      c='red', alpha=0.8, s=30, marker='D', label='Anomalous')
        
        ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})')
        ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})')
        ax.set_title('3D PCA Space')
        ax.legend()
        
        # Anomaly score distributions by percentile groups
        plt.subplot(2, 3, 4)
        percentile_groups = {
            '0-90th': ensemble_scores <= np.percentile(ensemble_scores, 90),
            '90-95th': (ensemble_scores > np.percentile(ensemble_scores, 90)) & 
                      (ensemble_scores <= np.percentile(ensemble_scores, 95)),
            '95-99th': (ensemble_scores > np.percentile(ensemble_scores, 95)) & 
                      (ensemble_scores <= np.percentile(ensemble_scores, 99)),
            '99th+': ensemble_scores > np.percentile(ensemble_scores, 99)
        }
        
        group_data = []
        group_labels = []
        for label, mask in percentile_groups.items():
            if mask.sum() > 0:
                group_data.append(ensemble_scores[mask])
                group_labels.append(f'{label}\n(n={mask.sum()})')
        
        plt.boxplot(group_data, labels=group_labels)
        plt.ylabel('Anomaly Score')
        plt.title('Score Distribution by Percentile Groups')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Individual detector performance comparison
        plt.subplot(2, 3, 5)
        detector_names = list(individual_scores.keys())[:8]  # Top 8 detectors
        score_data = [individual_scores[name] for name in detector_names]
        
        parts = plt.violinplot(score_data, positions=range(len(detector_names)), showmeans=True)
        plt.xticks(range(len(detector_names)), 
                  [name.replace('_', ' ')[:12] for name in detector_names], rotation=45)
        plt.ylabel('Anomaly Score')
        plt.title('Detector Score Distributions')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Ensemble vs individual detector ROC-style analysis
        plt.subplot(2, 3, 6)
        thresholds = np.percentile(ensemble_scores, np.arange(50, 100, 2))
        
        # Calculate detection rates at different thresholds
        detection_rates = []
        for threshold in thresholds:
            detection_rate = np.mean(ensemble_scores > threshold)
            detection_rates.append(detection_rate)
        
        plt.plot(thresholds, detection_rates, 'b-', linewidth=2, label='Ensemble')
        
        # Compare with best individual detector
        best_detector = max(detector.ensemble_weights, key=detector.ensemble_weights.get)
        if best_detector in individual_scores:
            best_scores = individual_scores[best_detector]
            best_thresholds = np.percentile(best_scores, np.arange(50, 100, 2))
            best_detection_rates = []
            for threshold in best_thresholds:
                detection_rate = np.mean(best_scores > threshold)
                best_detection_rates.append(detection_rate)
            
            plt.plot(best_thresholds, best_detection_rates, 'r--', linewidth=2, 
                    label=f'Best Individual ({best_detector[:15]})')
        
        plt.xlabel('Anomaly Score Threshold')
        plt.ylabel('Detection Rate')
        plt.title('Detection Rate vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/advanced_pca_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Performance metrics visualization
    plt.figure(figsize=(15, 8))
    
    # Hyperparameter optimization results (if available)
    if hasattr(detector, 'best_hyperparameters') and detector.best_hyperparameters:
        plt.subplot(2, 3, 1)
        params = detector.best_hyperparameters
        param_names = list(params.keys())
        param_values = list(params.values())
        
        # Convert non-numeric values for visualization
        numeric_values = []
        for val in param_values:
            if isinstance(val, (int, float)):
                numeric_values.append(val)
            else:
                numeric_values.append(hash(str(val)) % 100)  # Simple hash for categorical
        
        plt.bar(range(len(param_names)), numeric_values, color='lightcoral', alpha=0.7)
        plt.xticks(range(len(param_names)), 
                  [name.replace('_', ' ')[:10] for name in param_names], rotation=45)
        plt.ylabel('Parameter Value')
        plt.title('Optimized Hyperparameters')
        plt.grid(True, alpha=0.3, axis='y')
    
    # Model complexity vs performance
    plt.subplot(2, 3, 2)
    if hasattr(detector, 'performance_metrics') and 'individual_detectors' in detector.performance_metrics:
        individual_perf = detector.performance_metrics['individual_detectors']
        
        detector_complexities = []
        detector_weights = []
        detector_labels = []
        
        for name, metrics in individual_perf.items():
            # Estimate complexity based on detector type
            if 'isolation_forest' in name:
                complexity = 3
            elif 'lof' in name:
                complexity = 4
            elif 'gmm' in name or 'bayesian' in name:
                complexity = 5
            elif 'clustering' in name or 'kmeans' in name:
                complexity = 2
            elif 'dbscan' in name:
                complexity = 3
            else:
                complexity = 1
            
            detector_complexities.append(complexity)
            detector_weights.append(metrics.get('weight', 0))
            detector_labels.append(name[:15])
        
        scatter = plt.scatter(detector_complexities, detector_weights, 
                            s=[w*500 for w in detector_weights], alpha=0.6, c=detector_weights, 
                            cmap='viridis')
        plt.xlabel('Model Complexity')
        plt.ylabel('Ensemble Weight')
        plt.title('Complexity vs Performance')
        plt.colorbar(scatter, label='Weight')
        plt.grid(True, alpha=0.3)
    
    # Score correlation heatmap for top detectors
    plt.subplot(2, 3, 3)
    if len(individual_scores) > 1:
        # Select top detectors by weight
        top_detectors = sorted(detector.ensemble_weights.items(), 
                             key=lambda x: x[1], reverse=True)[:8]
        top_detector_names = [name for name, _ in top_detectors]
        
        top_scores = {name: individual_scores[name] for name in top_detector_names 
                     if name in individual_scores}
        
        if len(top_scores) > 1:
            score_matrix = np.array(list(top_scores.values())).T
            correlation_matrix = np.corrcoef(score_matrix.T)
            
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f',
                       cmap='RdYlBu_r', center=0, square=True,
                       xticklabels=[name[:8] for name in top_scores.keys()],
                       yticklabels=[name[:8] for name in top_scores.keys()])
            plt.title('Top Detector Correlations')
    
    # Anomaly characteristics analysis
    plt.subplot(2, 3, 4)
    if 'order_size' in features_df.columns and 'relative_spread' in features_df.columns:
        anomaly_threshold = np.percentile(ensemble_scores, 95)
        anomaly_mask = ensemble_scores > anomaly_threshold
        
        plt.scatter(features_df['order_size'][~anomaly_mask], 
                   features_df['relative_spread'][~anomaly_mask],
                   alpha=0.6, s=10, label='Normal', c='lightblue')
        
        if anomaly_mask.sum() > 0:
            plt.scatter(features_df['order_size'][anomaly_mask], 
                       features_df['relative_spread'][anomaly_mask],
                       alpha=0.8, s=30, label='Anomalous', c='red', marker='D')
        
        plt.xlabel('Order Size')
        plt.ylabel('Relative Spread')
        plt.title('Anomalies in Order Size vs Spread')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xscale('log')
    
    # Performance over time (if temporal features available)
    plt.subplot(2, 3, 5)
    if 'hour_of_day' in features_df.columns:
        hourly_anomaly_rates = []
        hours = sorted(features_df['hour_of_day'].unique())
        
        for hour in hours:
            hour_mask = features_df['hour_of_day'] == hour
            if hour_mask.sum() > 0:
                hour_scores = ensemble_scores[hour_mask]
                anomaly_rate = np.mean(hour_scores > np.percentile(ensemble_scores, 95))
                hourly_anomaly_rates.append(anomaly_rate)
            else:
                hourly_anomaly_rates.append(0)
        
        plt.plot(hours, hourly_anomaly_rates, 'bo-', linewidth=2, markersize=6)
        plt.xlabel('Hour of Day')
        plt.ylabel('Anomaly Rate')
        plt.title('Anomaly Rate by Hour')
        plt.grid(True, alpha=0.3)
        plt.xticks(hours[::2])  # Show every other hour
    
    # Model stability analysis
    plt.subplot(2, 3, 6)
    if len(individual_scores) >= 3:
        # Calculate rolling correlation between top 3 detectors
        top_3_detectors = sorted(detector.ensemble_weights.items(), 
                               key=lambda x: x[1], reverse=True)[:3]
        
        correlations = []
        window_size = min(100, len(ensemble_scores) // 10)
        
        for i in range(window_size, len(ensemble_scores)):
            window_scores = []
            for name, _ in top_3_detectors:
                if name in individual_scores:
                    window_scores.append(individual_scores[name][i-window_size:i])
            
            if len(window_scores) >= 2:
                corr_matrix = np.corrcoef(window_scores)
                avg_corr = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
                correlations.append(avg_corr)
        
        if correlations:
            plt.plot(range(window_size, window_size + len(correlations)), 
                    correlations, linewidth=2, color='green')
            plt.xlabel('Sample Index')
            plt.ylabel('Average Detector Correlation')
            plt.title('Model Stability Over Time')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Moderate Correlation')
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/performance_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

def main_enhanced_production_pipeline(data_dir="enhanced_market_data", n_optimization_trials=50):
    """Enhanced production pipeline with hyperparameter optimization"""
    print("="*80)
    print("ENHANCED PRODUCTION UNSUPERVISED TOXICITY DETECTION")
    print("With Hyperparameter Optimization and Advanced Ensemble Methods")
    print("Using only publicly observable market data")
    print("="*80)
    
    # Load and prepare data
    print("\n1. LOADING MARKET DATA")
    print("-" * 40)
    
    feature_engineer = EnhancedFeatureEngineer()
    orders_df, lob_df, trades_df = feature_engineer.load_and_merge_data(data_dir)
    
    # Enhanced feature engineering
    print("\n2. ADVANCED FEATURE ENGINEERING")
    print("-" * 40)
    
    features_df = feature_engineer.engineer_advanced_features(orders_df, lob_df, trades_df)
    
    # Advanced feature preparation
    print("\n3. ADVANCED FEATURE PREPARATION")
    print("-" * 40)
    
    detector = AdvancedToxicityDetector()
    X_scaled, selected_features = detector.prepare_features_with_selection(features_df)
    
    print(f"Final feature set: {X_scaled.shape}")
    
    # Hyperparameter optimization
    print("\n4. HYPERPARAMETER OPTIMIZATION")
    print("-" * 40)
    
    best_params = detector.optimize_hyperparameters(X_scaled, n_trials=n_optimization_trials)
    
    # Train optimized ensemble detectors
    print("\n5. TRAINING OPTIMIZED ENSEMBLE DETECTORS")
    print("-" * 40)
    
    detectors = detector.train_optimized_ensemble_detectors(X_scaled, best_params)
    
    # Calculate weighted ensemble scores
    print("\n6. CALCULATING WEIGHTED ENSEMBLE SCORES")
    print("-" * 40)
    
    ensemble_scores, individual_scores = detector.calculate_weighted_ensemble_scores(X_scaled)
    
    # Enhanced performance evaluation
    print("\n7. ENHANCED PERFORMANCE EVALUATION")
    print("-" * 40)
    
    metrics = detector.evaluate_enhanced_performance(X_scaled, ensemble_scores, individual_scores)
    
    # Create enhanced visualizations
    print("\n8. CREATING ENHANCED VISUALIZATIONS")
    print("-" * 40)
    
    create_enhanced_visualizations(selected_features, ensemble_scores, individual_scores, detector)
    
    # Save enhanced model
    print("\n9. SAVING ENHANCED PRODUCTION MODEL")
    print("-" * 40)
    
    model_path = detector.save_enhanced_model()
    
    # Print comprehensive results summary
    print("\n" + "="*80)
    print("ENHANCED PRODUCTION MODEL TRAINING COMPLETED")
    print("="*80)
    
    print(f"\nModel Performance Summary:")
    print(f"  Ensemble anomaly rate (95th): {metrics.get('anomaly_detection', {}).get('anomaly_rate_95th', 0)*100:.2f}%")
    print(f"  Ensemble anomaly rate (99th): {metrics.get('anomaly_detection', {}).get('anomaly_rate_99th', 0)*100:.2f}%")
    print(f"  Number of features: {len(detector.feature_selector)}")
    print(f"  Number of detectors: {len(detector.models)}")
    print(f"  Effective detectors (>5% weight): {metrics.get('ensemble_statistics', {}).get('effective_detectors', 0)}")
    
    if 'silhouette_score' in metrics:
        print(f"  Clustering quality (silhouette): {metrics['silhouette_score']:.3f}")
    if 'calinski_harabasz_score' in metrics:
        print(f"  Clustering quality (Calinski-Harabasz): {metrics['calinski_harabasz_score']:.1f}")
    
    print(f"\nEnsemble Diversity:")
    if 'ensemble_diversity' in metrics:
        print(f"  Average correlation: {metrics['ensemble_diversity']['avg_correlation']:.3f}")
        print(f"  Diversity score: {metrics['ensemble_diversity']['diversity_score']:.3f}")
    
    print(f"\nTop Performing Detectors:")
    top_detectors = sorted(detector.ensemble_weights.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, weight in top_detectors:
        print(f"  {name}: {weight:.3f}")
    
    print(f"\nOptimized Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\nDetection Statistics:")
    ensemble_stats = metrics.get('ensemble_statistics', {})
    print(f"  Score range: {ensemble_stats.get('score_mean', 0) - ensemble_stats.get('score_std', 0):.4f} - {ensemble_stats.get('score_mean', 0) + ensemble_stats.get('score_std', 0):.4f}")
    print(f"  Score skewness: {ensemble_stats.get('score_skewness', 0):.3f}")
    print(f"  Score kurtosis: {ensemble_stats.get('score_kurtosis', 0):.3f}")
    
    anomaly_detection = metrics.get('anomaly_detection', {})
    for threshold in [95, 99]:
        rate = anomaly_detection.get(f'anomaly_rate_{threshold}th', 0)
        separation = anomaly_detection.get(f'separation_{threshold}th', 0)
        print(f"  {threshold}th percentile: {rate*100:.2f}% rate, {separation:.2f} separation")
    
    print(f"\nModel saved to: {model_path}")
    
    return detector, ensemble_scores, metrics

class EnhancedProductionInference:
    """Enhanced inference class with better error handling and performance tracking"""
    
    def __init__(self, model_path):
        self.model_package = joblib.load(model_path)
        self.models = self.model_package['models']
        self.scalers = self.model_package['scalers']
        self.feature_selector = self.model_package['feature_selector']
        self.ensemble_weights = self.model_package['ensemble_weights']
        self.best_hyperparameters = self.model_package.get('best_hyperparameters', {})
        
        print(f"Loaded enhanced model trained on {self.model_package['timestamp']}")
        print(f"Model version: {self.model_package['version']}")
        print(f"Number of features: {self.model_package['n_features']}")
        print(f"Number of detectors: {self.model_package['n_detectors']}")
        
        # Performance tracking
        self.prediction_count = 0
        self.prediction_times = []
        
    def predict_toxicity_score(self, features_dict):
        """Enhanced prediction with performance tracking"""
        import time
        start_time = time.time()
        
        try:
            # Prepare features
            features_df = pd.DataFrame([features_dict])
            
            # Handle missing features
            missing_features = set(self.feature_selector) - set(features_df.columns)
            for feature in missing_features:
                features_df[feature] = 0
            
            # Select and order features
            features_df = features_df[self.feature_selector]
            
            # Scale features
            X_scaled = self.scalers['main'].transform(features_df)
            
            # Calculate ensemble score
            ensemble_score = 0.0
            successful_detectors = 0
            
            for name, model in self.models.items():
                try:
                    weight = self.ensemble_weights.get(name, 0)
                    if weight == 0:
                        continue
                    
                    if 'isolation_forest' in name:
                        score = -model.decision_function(X_scaled)[0]
                    elif 'lof' in name:
                        score = -model.score_samples(X_scaled)[0]
                    elif 'elliptic' in name:
                        score = -model.decision_function(X_scaled)[0]
                    elif 'gmm' in name or 'bayesian_gmm' in name:
                        score = -model.score_samples(X_scaled)[0]
                    elif 'kmeans' in name:
                        distance = np.min(model['kmeans'].transform(X_scaled), axis=1)[0]
                        score = distance
                    elif 'agglomerative' in name:
                        distances = np.min(cdist(X_scaled, model['cluster_centres']), axis=1)
                        score = distances[0]
                    elif 'dbscan' in name:
                        # For DBSCAN, calculate distance to nearest core point
                        # This is a simplified approach for single prediction
                        score = 0.5  # Default moderate score
                    else:
                        continue
                    
                    # Normalize and bound score
                    score = max(0, min(1, score))
                    ensemble_score += weight * score
                    successful_detectors += 1
                    
                except Exception as e:
                    # Silently continue if individual detector fails
                    continue
            
            # Normalize by successful detectors
            if successful_detectors > 0:
                ensemble_score = ensemble_score
            else:
                ensemble_score = 0.0
            
            # Track performance
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            self.prediction_count += 1
            
            return ensemble_score
            
        except Exception as e:
            print(f"Error in toxicity prediction: {e}")
            return 0.0
    
    def is_toxic(self, features_dict, threshold=0.7):
        """Enhanced toxicity classification with confidence scoring"""
        toxicity_score = self.predict_toxicity_score(features_dict)
        
        # Calculate confidence based on score distance from threshold
        confidence = abs(toxicity_score - threshold) / threshold
        confidence = min(1.0, confidence)
        
        return toxicity_score > threshold, toxicity_score, confidence
    
    def get_performance_stats(self):
        """Get inference performance statistics"""
        if not self.prediction_times:
            return {}
        
        return {
            'total_predictions': self.prediction_count,
            'avg_prediction_time': np.mean(self.prediction_times),
            'median_prediction_time': np.median(self.prediction_times),
            'max_prediction_time': np.max(self.prediction_times),
            'min_prediction_time': np.min(self.prediction_times)
        }

if __name__ == "__main__":
    try:
        # Run enhanced pipeline with hyperparameter optimization
        detector, ensemble_scores, metrics = main_enhanced_production_pipeline(
            data_dir="enhanced_market_data",
            n_optimization_trials=50  # Adjust based on computational budget
        )
        
        print("\n" + "="*80)
        print("SUCCESS: Enhanced production model training completed!")
        print("="*80)
        
    except Exception as e:
        print(f"Error in enhanced production pipeline: {e}")
        import traceback
        traceback.print_exc()
