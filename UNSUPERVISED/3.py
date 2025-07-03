"""
Enhanced Production Unsupervised Toxicity Detection System - FIXED VERSION
Fixed timestamp handling, rolling calculations, and GMM fitting errors
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
    
    def rolling_percentile(self, series, window):
        """Calculate rolling percentile using quantile comparison"""
        def percentile_calc(x):
            if len(x) <= 1:
                return 0.5
            return (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1)
        return series.rolling(window, min_periods=1).apply(percentile_calc, raw=False)
        
    def load_and_merge_data(self, data_dir="enhanced_market_data"):
        """Load market data ensuring only publicly observable features"""
        print("Loading market data...")
        
        order_files = glob.glob(f"{data_dir}/orders_*.csv")
        lob_files = glob.glob(f"{data_dir}/lob_snapshots_*.csv")
        trade_files = glob.glob(f"{data_dir}/trades_*.csv")
        
        if not order_files:
            print(f"No order files found in {data_dir}. Creating synthetic data for testing...")
            return self._create_synthetic_data()
        
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
    
    def _create_synthetic_data(self):
        """Create synthetic market data for testing"""
        print("Creating synthetic market data for testing...")
        np.random.seed(42)
        
        n_orders = 1000
        timestamps = pd.date_range(start='2024-01-01 09:30:00', periods=n_orders, freq='1s')
        
        orders_df = pd.DataFrame({
            'timestamp': timestamps,
            'order_id': range(n_orders),
            'order_type': np.random.choice(['LIMIT', 'MARKET'], n_orders, p=[0.8, 0.2]),
            'side': np.random.choice(['BUY', 'SELL'], n_orders),
            'price': 100 + np.cumsum(np.random.normal(0, 0.1, n_orders)),
            'quantity': np.random.lognormal(3, 1, n_orders),
            'mid_price': 100 + np.cumsum(np.random.normal(0, 0.1, n_orders)),
            'spread': np.random.exponential(0.02, n_orders),
            'distance_from_mid': np.random.normal(0, 0.5, n_orders),
            'is_aggressive': np.random.choice([0, 1], n_orders, p=[0.7, 0.3]),
            'volatility': np.random.exponential(0.1, n_orders),
            'momentum': np.random.normal(0, 0.05, n_orders),
            'order_book_imbalance': np.random.normal(0, 0.3, n_orders),
            'time_since_last_trade': np.random.exponential(1, n_orders)
        })
        
        # Create some LOB data
        n_lob = 100
        lob_timestamps = pd.date_range(start='2024-01-01 09:30:00', periods=n_lob, freq='10s')
        lob_df = pd.DataFrame({
            'timestamp': lob_timestamps,
            'bid_price_1': 100 + np.cumsum(np.random.normal(0, 0.1, n_lob)) - 0.01,
            'ask_price_1': 100 + np.cumsum(np.random.normal(0, 0.1, n_lob)) + 0.01,
            'bid_size_1': np.random.exponential(1000, n_lob),
            'ask_size_1': np.random.exponential(1000, n_lob)
        })
        
        # Create some trade data
        n_trades = 200
        trade_timestamps = timestamps[::5][:n_trades]  # Every 5th order timestamp
        trades_df = pd.DataFrame({
            'timestamp': trade_timestamps,
            'price': 100 + np.cumsum(np.random.normal(0, 0.1, n_trades)),
            'quantity': np.random.lognormal(2, 1, n_trades)
        })
        
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
            if 'mid_price' in orders_df.columns:
                features_df['relative_spread'] = spread / (mid_price + 1e-8)
            features_df['log_spread'] = np.log1p(spread)
            # Fixed: Calculate rolling percentile using quantile comparison
            def rolling_percentile(series, window):
                def percentile_calc(x):
                    if len(x) <= 1:
                        return 0.5
                    return (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1)
                return series.rolling(window, min_periods=1).apply(percentile_calc, raw=False)
            
            features_df['spread_percentile'] = self.rolling_percentile(spread, 100)
        
        # Aggressiveness features
        if 'is_aggressive' in orders_df.columns:
            features_df['is_aggressive'] = orders_df['is_aggressive'].astype(int)
        elif 'distance_from_mid' in orders_df.columns:
            features_df['is_aggressive'] = (np.abs(orders_df['distance_from_mid']) < 0.001).astype(int)
            features_df['distance_from_mid'] = orders_df['distance_from_mid']
            features_df['abs_distance_from_mid'] = np.abs(orders_df['distance_from_mid'])
        
        # Enhanced temporal features - FIXED timestamp handling
        if 'timestamp' in orders_df.columns:
            # Convert to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(orders_df['timestamp']):
                timestamps = pd.to_datetime(orders_df['timestamp'])
            else:
                timestamps = orders_df['timestamp']
            
            # Time-based features
            features_df['hour_of_day'] = timestamps.dt.hour
            features_df['minute_of_hour'] = timestamps.dt.minute
            features_df['day_of_week'] = timestamps.dt.dayofweek
            
            # Cyclical encoding for time features
            features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour_of_day'] / 24)
            features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour_of_day'] / 24)
            features_df['minute_sin'] = np.sin(2 * np.pi * features_df['minute_of_hour'] / 60)
            features_df['minute_cos'] = np.cos(2 * np.pi * features_df['minute_of_hour'] / 60)
            
            # Inter-arrival time features - FIXED
            time_diffs = timestamps.diff().dt.total_seconds().fillna(1)
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
            # Fixed: Calculate rolling percentile using quantile comparison
            features_df['vol_percentile'] = self.rolling_percentile(vol, 100)
        else:
            if 'mid_price' in orders_df.columns:
                returns = mid_price.pct_change().fillna(0)
                for window in [5, 10, 20, 50]:
                    features_df[f'volatility_{window}'] = returns.rolling(window, min_periods=1).std()
                    # Fixed: Calculate rolling percentile using quantile comparison
                    features_df[f'volatility_percentile_{window}'] = self.rolling_percentile(
                        features_df[f'volatility_{window}'], 100)
        
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
            # Fixed: Calculate rolling percentile using quantile comparison
            features_df['imbalance_percentile'] = self.rolling_percentile(imbalance, 100)
        
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
            # Convert timestamps if needed
            if not pd.api.types.is_datetime64_any_dtype(lob_df['timestamp']):
                lob_df['timestamp'] = pd.to_datetime(lob_df['timestamp'])
            if not pd.api.types.is_datetime64_any_dtype(orders_df['timestamp']):
                orders_timestamps = pd.to_datetime(orders_df['timestamp'])
            else:
                orders_timestamps = orders_df['timestamp']
            
            orders_with_ts = orders_df.copy()
            orders_with_ts['timestamp'] = orders_timestamps
            
            merged = pd.merge_asof(
                orders_with_ts[['timestamp']].reset_index(),
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
        
        return lob_features.fillna(0)
    
    def _extract_enhanced_trade_features(self, trades_df, orders_df):
        """Extract more sophisticated trade-based features"""
        trade_features = pd.DataFrame(index=orders_df.index)
        
        if 'timestamp' in trades_df.columns and 'timestamp' in orders_df.columns:
            # Convert timestamps if needed
            if not pd.api.types.is_datetime64_any_dtype(trades_df['timestamp']):
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            if not pd.api.types.is_datetime64_any_dtype(orders_df['timestamp']):
                orders_timestamps = pd.to_datetime(orders_df['timestamp'])
            else:
                orders_timestamps = orders_df['timestamp']
            
            for idx, order_timestamp in enumerate(orders_timestamps):
                # Multiple time windows for trade analysis
                for window_seconds in [5, 10, 30]:
                    start_time = order_timestamp - pd.Timedelta(seconds=window_seconds)
                    recent_trades = trades_df[
                        (trades_df['timestamp'] >= start_time) & 
                        (trades_df['timestamp'] <= order_timestamp)
                    ]
                    
                    if not recent_trades.empty:
                        trade_features.loc[idx, f'trade_volume_{window_seconds}'] = recent_trades['quantity'].sum()
                        trade_features.loc[idx, f'trade_count_{window_seconds}'] = len(recent_trades)
                        trade_features.loc[idx, f'avg_trade_size_{window_seconds}'] = recent_trades['quantity'].mean()
                        
                        if len(recent_trades) > 1:
                            trade_features.loc[idx, f'trade_size_std_{window_seconds}'] = recent_trades['quantity'].std()
                            
                            if 'price' in recent_trades.columns:
                                price_returns = recent_trades['price'].pct_change().dropna()
                                if len(price_returns) > 0:
                                    trade_features.loc[idx, f'trade_volatility_{window_seconds}'] = price_returns.std()
                        
                        # Time span analysis
                        if len(recent_trades) > 1:
                            time_span_seconds = (recent_trades['timestamp'].max() - recent_trades['timestamp'].min()).total_seconds()
                            if time_span_seconds > 0:
                                trade_features.loc[idx, f'trade_frequency_{window_seconds}'] = len(recent_trades) / time_span_seconds
                        
                        # VWAP calculation
                        if 'price' in recent_trades.columns and recent_trades['quantity'].sum() > 0:
                            vwap = (recent_trades['price'] * recent_trades['quantity']).sum() / recent_trades['quantity'].sum()
                            trade_features.loc[idx, f'vwap_{window_seconds}'] = vwap
        
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
                
                # Z-score features
                ma_col = f'{feature}_ma_{window}'
                std_col = f'{feature}_std_{window}'
                features_df[f'{feature}_zscore_{window}'] = (
                    (features_df[feature] - features_df[ma_col]) / (features_df[std_col] + 1e-8)
                )
                
                # Fixed: Calculate rolling percentile using quantile comparison
                features_df[f'{feature}_percentile_{window}'] = self.rolling_percentile(
                    features_df[feature], window)
                
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
                    if 'log_order_size' in features_df.columns:
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
        
        return features_df
    
    def _add_statistical_features(self, features_df):
        """Add statistical distribution features"""
        numerical_features = features_df.select_dtypes(include=[np.number]).columns[:10]  # Limit for performance
        
        for feature in numerical_features:
            if features_df[feature].std() > 0:
                # Distribution moments
                features_df[f'{feature}_skewness'] = features_df[feature].rolling(50, min_periods=10).skew()
                features_df[f'{feature}_kurtosis'] = features_df[feature].rolling(50, min_periods=10).kurt()
        
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
        
        # Ensure we have numeric data only
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        high_var_features = numeric_features.loc[:, var_selector.fit(numeric_features).get_support()]
        print(f"Removed {len(numeric_features.columns) - len(high_var_features.columns)} low variance features")
        
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
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
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
        try:
            iso_forest = IsolationForest(
                contamination=hyperparameters.get('contamination', 0.05),
                n_estimators=hyperparameters.get('n_estimators', 200),
                max_samples=hyperparameters.get('max_samples', 'auto'),
                random_state=42,
                bootstrap=True
            )
            iso_forest.fit(X_scaled)
            detectors['isolation_forest_optimized'] = iso_forest
        except Exception as e:
            print(f"Warning: Failed to train Isolation Forest: {e}")
        
        # LOF with optimized parameters
        try:
            lof = LocalOutlierFactor(
                n_neighbors=hyperparameters.get('n_neighbors', 20),
                contamination=hyperparameters.get('contamination', 0.05),
                novelty=True
            )
            lof.fit(X_scaled)
            detectors['lof_optimized'] = lof
        except Exception as e:
            print(f"Warning: Failed to train LOF: {e}")
        
        # Multiple contamination rates for robustness
        for contamination in [0.03, 0.07, 0.1]:
            # Additional Isolation Forest
            try:
                iso_forest_multi = IsolationForest(
                    contamination=contamination,
                    n_estimators=hyperparameters.get('n_estimators', 200),
                    random_state=42 + int(contamination * 100),
                    bootstrap=True
                )
                iso_forest_multi.fit(X_scaled)
                detectors[f'isolation_forest_{contamination}'] = iso_forest_multi
            except Exception as e:
                print(f"Warning: Failed to train Isolation Forest with contamination {contamination}: {e}")
            
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
        
        # Gaussian Mixture Models with regularization - FIXED
        for n_components in [3, 5, 8]:
            try:
                n_comp = min(n_components, max(2, len(X_scaled) // 50))
                if n_comp < 2:
                    continue
                
                # Add regularization and better initialization
                gmm = GaussianMixture(
                    n_components=n_comp,
                    random_state=42,
                    covariance_type='tied',  # Changed to tied for better stability
                    reg_covar=1e-3,  # Increased regularization
                    init_params='kmeans',
                    max_iter=200
                )
                gmm.fit(X_scaled)
                detectors[f'gmm_{n_comp}'] = gmm
            except Exception as e:
                print(f"Warning: Failed to train GMM with {n_comp} components: {e}")
        
        # Bayesian Gaussian Mixture with regularization - FIXED
        try:
            n_comp = min(10, max(2, len(X_scaled) // 30))
            if n_comp >= 2:
                bgmm = BayesianGaussianMixture(
                    n_components=n_comp,
                    random_state=42,
                    covariance_type='tied',  # Changed to tied for better stability
                    reg_covar=1e-3,  # Increased regularization
                    init_params='kmeans',
                    max_iter=200
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
        
        try:
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
            cluster_labels = kmeans.fit_predict(X_scaled)
            distances = np.min(kmeans.transform(X_scaled), axis=1)
            
            clustering_detectors['kmeans_optimized'] = {
                'kmeans': kmeans,
                'distance_threshold': np.percentile(distances, 95),
                'cluster_sizes': np.bincount(cluster_labels)
            }
        except Exception as e:
            print(f"Warning: Failed to train K-means: {e}")
        
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
            try:
                dbscan = DBSCAN(**best_params)
                labels = dbscan.fit_predict(X_scaled)
                
                return {
                    'dbscan': dbscan,
                    'labels': labels,
                    'params': best_params
                }
            except Exception:
                return None
        
        return None
    
    def _optimize_kmeans_clusters(self, X_scaled, max_k=20):
        """Enhanced K-means cluster optimization using multiple metrics"""
        k_range = range(2, min(max_k, max(3, len(X_scaled) // 20)))
        
        if len(k_range) == 0:
            return 3  # Default
        
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        
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
                else:
                    silhouette_scores.append(-1)
                    calinski_scores.append(0)
                    davies_bouldin_scores.append(10)
                    
            except Exception:
                silhouette_scores.append(-1)
                calinski_scores.append(0)
                davies_bouldin_scores.append(10)
        
        # Find best k based on silhouette score
        if silhouette_scores:
            best_k_idx = np.argmax(silhouette_scores)
            best_k = list(k_range)[best_k_idx]
            
            print(f"Optimal number of clusters: {best_k}")
            print(f"  Silhouette score: {silhouette_scores[best_k_idx]:.3f}")
            
            return best_k
        
        return 3  # Default fallback
    
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
                        if i < len(labels) and labels[i] == -1:  # Outlier
                            scores[i] = 1.0
                        else:
                            # Distance to cluster centre
                            if i < len(labels):
                                cluster_points = X_scaled[labels == labels[i]]
                                if len(cluster_points) > 0:
                                    cluster_centre = np.mean(cluster_points, axis=0)
                                    scores[i] = np.linalg.norm(point - cluster_centre)
                else:
                    continue
                
                # Normalize scores
                if len(scores) > 0 and scores.max() > scores.min():
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
                if expected_rate > 0:
                    rate_score = 1 - abs(anomaly_rate - expected_rate) / expected_rate
                    performance_scores.append(max(0, rate_score))
                else:
                    performance_scores.append(0)
            
            avg_performance = np.mean(performance_scores) if performance_scores else 0
            
            # Combine separation and performance
            weight = separation_score * (avg_performance + 0.1)  # Add small constant to avoid zero
            
            return max(0.1, weight)  # Minimum weight threshold
            
        except Exception:
            return 0.1
    
    def evaluate_enhanced_performance(self, X_scaled, ensemble_scores, individual_scores):
        """Enhanced performance evaluation with multiple metrics"""
        print("Evaluating enhanced detector performance...")
        
        metrics = {}
        
        # Clustering quality metrics
        if 'kmeans_optimized' in self.models:
            try:
                kmeans = self.models['kmeans_optimized']['kmeans']
                cluster_labels = kmeans.predict(X_scaled)
                
                if len(set(cluster_labels)) > 1:
                    metrics['silhouette_score'] = silhouette_score(X_scaled, cluster_labels)
                    metrics['davies_bouldin_score'] = davies_bouldin_score(X_scaled, cluster_labels)
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, cluster_labels)
            except Exception as e:
                print(f"Error calculating clustering metrics: {e}")
        
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
                if normal_scores.std() > 0:
                    anomaly_metrics[f'separation_{threshold_pct}th'] = (
                        anomaly_scores.mean() - normal_scores.mean()
                    ) / normal_scores.std()
        
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
            try:
                score_matrix = np.array(list(individual_scores.values())).T
                correlation_matrix = np.corrcoef(score_matrix.T)
                
                # Average pairwise correlation (lower is better for diversity)
                avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
                metrics['ensemble_diversity'] = {
                    'avg_correlation': avg_correlation,
                    'diversity_score': 1 - abs(avg_correlation)
                }
            except Exception as e:
                print(f"Error calculating diversity metrics: {e}")
        
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
            'version': '2.0_enhanced_fixed',
            'n_features': len(self.feature_selector) if self.feature_selector else 0,
            'n_detectors': len(self.models)
        }
        
        joblib.dump(model_package, model_path)
        
        print(f"Enhanced model saved to: {model_path}")
        return model_path

def main_enhanced_production_pipeline(data_dir="enhanced_market_data", n_optimization_trials=25):
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
    
    # Save enhanced model
    print("\n8. SAVING ENHANCED PRODUCTION MODEL")
    print("-" * 40)
    
    model_path = detector.save_enhanced_model()
    
    # Print comprehensive results summary
    print("\n" + "="*80)
    print("ENHANCED PRODUCTION MODEL TRAINING COMPLETED")
    print("="*80)
    
    print(f"\nModel Performance Summary:")
    anomaly_detection = metrics.get('anomaly_detection', {})
    print(f"  Ensemble anomaly rate (95th): {anomaly_detection.get('anomaly_rate_95th', 0)*100:.2f}%")
    print(f"  Ensemble anomaly rate (99th): {anomaly_detection.get('anomaly_rate_99th', 0)*100:.2f}%")
    print(f"  Number of features: {len(detector.feature_selector) if detector.feature_selector else 0}")
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
    
    print(f"\nModel saved to: {model_path}")
    
    return detector, ensemble_scores, metrics

if __name__ == "__main__":
    try:
        # Run enhanced pipeline with hyperparameter optimization
        detector, ensemble_scores, metrics = main_enhanced_production_pipeline(
            data_dir="enhanced_market_data",
            n_optimization_trials=25  # Reduced for faster execution
        )
        
        print("\n" + "="*80)
        print("SUCCESS: Enhanced production model training completed!")
        print("="*80)
        
    except Exception as e:
        print(f"Error in enhanced production pipeline: {e}")
        import traceback
        traceback.print_exc()