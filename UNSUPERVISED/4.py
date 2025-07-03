"""
Enhanced Production Unsupervised Toxicity Detection System - IMPROVED VERSION
Added comprehensive plotting, model interpretation, and performance monitoring
Saves all plots separately with publication-quality formatting
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

plt.style.use('ggplot')
sns.set_palette("husl")

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
        """Create enhanced synthetic market data with realistic patterns"""
        print("Creating enhanced synthetic market data...")
        np.random.seed(42)
        
        n_orders = 2000  # Increased sample size
        timestamps = pd.date_range(start='2024-01-01 09:30:00', periods=n_orders, freq='1s')
        
        # Create more realistic price evolution with trends and volatility clustering
        price_base = 100
        volatility = 0.02 + 0.03 * np.abs(np.sin(np.arange(n_orders) * 0.01))  # Time-varying volatility
        price_returns = np.random.normal(0, volatility)
        mid_prices = price_base * np.exp(np.cumsum(price_returns))
        
        # Add some toxic patterns (for testing detection)
        toxic_indices = np.random.choice(n_orders, size=int(0.05 * n_orders), replace=False)
        
        orders_df = pd.DataFrame({
            'timestamp': timestamps,
            'order_id': range(n_orders),
            'order_type': np.random.choice(['LIMIT', 'MARKET'], n_orders, p=[0.85, 0.15]),
            'side': np.random.choice(['BUY', 'SELL'], n_orders),
            'price': mid_prices + np.random.normal(0, 0.05, n_orders),
            'quantity': np.random.lognormal(3, 1, n_orders),
            'mid_price': mid_prices,
            'spread': 0.01 + np.random.exponential(0.02, n_orders),
            'distance_from_mid': np.random.normal(0, 0.5, n_orders),
            'is_aggressive': np.random.choice([0, 1], n_orders, p=[0.75, 0.25]),
            'volatility': volatility,
            'momentum': np.random.normal(0, 0.05, n_orders),
            'order_book_imbalance': np.random.normal(0, 0.3, n_orders),
            'time_since_last_trade': np.random.exponential(1, n_orders)
        })
        
        # Inject toxic patterns
        for idx in toxic_indices:
            # Large order sizes (potential market manipulation)
            orders_df.loc[idx, 'quantity'] *= np.random.uniform(5, 15)
            # Aggressive timing
            orders_df.loc[idx, 'time_since_last_trade'] *= 0.1
            # Price impact
            orders_df.loc[idx, 'volatility'] *= np.random.uniform(3, 8)
        
        # Enhanced LOB data
        n_lob = 200
        lob_timestamps = pd.date_range(start='2024-01-01 09:30:00', periods=n_lob, freq='10s')
        lob_df = pd.DataFrame({
            'timestamp': lob_timestamps,
            'bid_price_1': mid_prices[:n_lob] - 0.01,
            'ask_price_1': mid_prices[:n_lob] + 0.01,
            'bid_size_1': np.random.exponential(1000, n_lob),
            'ask_size_1': np.random.exponential(1000, n_lob),
            'bid_price_2': mid_prices[:n_lob] - 0.02,
            'ask_price_2': mid_prices[:n_lob] + 0.02,
            'bid_size_2': np.random.exponential(800, n_lob),
            'ask_size_2': np.random.exponential(800, n_lob)
        })
        
        # Enhanced trade data
        n_trades = 400
        trade_timestamps = timestamps[::5][:n_trades]
        trades_df = pd.DataFrame({
            'timestamp': trade_timestamps,
            'price': mid_prices[:n_trades] + np.random.normal(0, 0.01, n_trades),
            'quantity': np.random.lognormal(2.5, 1, n_trades)
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
        
        # Enhanced size features
        features_df['order_size_zscore'] = (orders_df['quantity'] - orders_df['quantity'].mean()) / orders_df['quantity'].std()
        features_df['order_size_percentile'] = orders_df['quantity'].rank(pct=True)
        
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
            features_df['spread_percentile'] = self.rolling_percentile(spread, 100)
        
        # Aggressiveness features
        if 'is_aggressive' in orders_df.columns:
            features_df['is_aggressive'] = orders_df['is_aggressive'].astype(int)
        elif 'distance_from_mid' in orders_df.columns:
            features_df['is_aggressive'] = (np.abs(orders_df['distance_from_mid']) < 0.001).astype(int)
            features_df['distance_from_mid'] = orders_df['distance_from_mid']
            features_df['abs_distance_from_mid'] = np.abs(orders_df['distance_from_mid'])
        
        # Enhanced temporal features
        if 'timestamp' in orders_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(orders_df['timestamp']):
                timestamps = pd.to_datetime(orders_df['timestamp'])
            else:
                timestamps = orders_df['timestamp']
            
            # Time-based features
            features_df['hour_of_day'] = timestamps.dt.hour
            features_df['minute_of_hour'] = timestamps.dt.minute
            features_df['day_of_week'] = timestamps.dt.dayofweek
            features_df['second_of_minute'] = timestamps.dt.second
            
            # Enhanced cyclical encoding
            features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour_of_day'] / 24)
            features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour_of_day'] / 24)
            features_df['minute_sin'] = np.sin(2 * np.pi * features_df['minute_of_hour'] / 60)
            features_df['minute_cos'] = np.cos(2 * np.pi * features_df['minute_of_hour'] / 60)
            
            # Inter-arrival time features
            time_diffs = timestamps.diff().dt.total_seconds().fillna(1)
            features_df['inter_arrival_time'] = time_diffs
            features_df['log_inter_arrival_time'] = np.log1p(time_diffs)
            features_df['arrival_rate'] = 1 / (time_diffs + 1e-8)
            features_df['arrival_acceleration'] = features_df['arrival_rate'].diff().fillna(0)
            
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
            features_df['vol_percentile'] = self.rolling_percentile(vol, 100)
            features_df['vol_regime'] = (vol > vol.quantile(0.75)).astype(int)
        
        # Enhanced momentum features
        if 'momentum' in orders_df.columns:
            mom = orders_df['momentum']
            features_df['momentum'] = mom
            features_df['abs_momentum'] = np.abs(mom)
            features_df['momentum_sign'] = np.sign(mom)
            features_df['momentum_squared'] = mom ** 2
        
        # Order book imbalance features
        if 'order_book_imbalance' in orders_df.columns:
            imbalance = orders_df['order_book_imbalance']
            features_df['order_book_imbalance'] = imbalance
            features_df['abs_imbalance'] = np.abs(imbalance)
            features_df['imbalance_sign'] = np.sign(imbalance)
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
        
        # Market regime features
        features_df = self._add_market_regime_features(features_df)
        
        features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)
        self.feature_names = features_df.columns.tolist()
        
        print(f"Generated {len(self.feature_names)} advanced public market features")
        return features_df
    
    def _extract_enhanced_lob_features(self, lob_df, orders_df):
        """Extract more sophisticated LOB features"""
        lob_features = pd.DataFrame(index=orders_df.index)
        
        if 'timestamp' in lob_df.columns and 'timestamp' in orders_df.columns:
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
            
            # Multi-level depth and imbalance features
            total_bid_depth = pd.Series(0, index=merged.index)
            total_ask_depth = pd.Series(0, index=merged.index)
            
            for level in range(1, 6):
                bid_price_col = f'bid_price_{level}'
                ask_price_col = f'ask_price_{level}'
                bid_size_col = f'bid_size_{level}'
                ask_size_col = f'ask_size_{level}'
                
                if all(col in merged.columns for col in [bid_size_col, ask_size_col]):
                    bid_size = merged[bid_size_col].fillna(0)
                    ask_size = merged[ask_size_col].fillna(0)
                    
                    total_bid_depth += bid_size
                    total_ask_depth += ask_size
                    
                    total_size = bid_size + ask_size
                    level_imbalance = (bid_size - ask_size) / (total_size + 1e-8)
                    
                    lob_features[f'imbalance_L{level}'] = level_imbalance
                    lob_features[f'bid_depth_L{level}'] = bid_size
                    lob_features[f'ask_depth_L{level}'] = ask_size
                    lob_features[f'depth_ratio_L{level}'] = bid_size / (ask_size + 1e-8)
                    
                    if all(col in merged.columns for col in [bid_price_col, ask_price_col]):
                        bid_price = merged[bid_price_col].fillna(0)
                        ask_price = merged[ask_price_col].fillna(0)
                        lob_features[f'spread_L{level}'] = ask_price - bid_price
            
            # Aggregate depth features
            lob_features['total_bid_depth'] = total_bid_depth
            lob_features['total_ask_depth'] = total_ask_depth
            lob_features['total_depth'] = total_bid_depth + total_ask_depth
            lob_features['depth_imbalance'] = (total_bid_depth - total_ask_depth) / (
                total_bid_depth + total_ask_depth + 1e-8)
        
        return lob_features.fillna(0)
    
    def _extract_enhanced_trade_features(self, trades_df, orders_df):
        """Extract more sophisticated trade-based features"""
        trade_features = pd.DataFrame(index=orders_df.index)
        
        if 'timestamp' in trades_df.columns and 'timestamp' in orders_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(trades_df['timestamp']):
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            if not pd.api.types.is_datetime64_any_dtype(orders_df['timestamp']):
                orders_timestamps = pd.to_datetime(orders_df['timestamp'])
            else:
                orders_timestamps = orders_df['timestamp']
            
            for idx, order_timestamp in enumerate(orders_timestamps[:min(len(orders_timestamps), 1000)]):  # Limit for performance
                for window_seconds in [10, 30, 60]:
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
                                    trade_features.loc[idx, f'trade_skewness_{window_seconds}'] = price_returns.skew()
                        
                        # Enhanced VWAP and price impact features
                        if 'price' in recent_trades.columns and recent_trades['quantity'].sum() > 0:
                            vwap = (recent_trades['price'] * recent_trades['quantity']).sum() / recent_trades['quantity'].sum()
                            trade_features.loc[idx, f'vwap_{window_seconds}'] = vwap
                            
                            # Price impact (last price vs VWAP)
                            if len(recent_trades) > 0:
                                last_price = recent_trades['price'].iloc[-1]
                                trade_features.loc[idx, f'price_impact_{window_seconds}'] = (last_price - vwap) / (vwap + 1e-8)
        
        return trade_features.fillna(0)
    
    def _add_multi_scale_rolling_features(self, features_df):
        """Add rolling features across multiple time scales"""
        key_features = ['order_size', 'relative_spread', 'volatility', 'momentum', 'order_book_imbalance']
        available_features = [f for f in key_features if f in features_df.columns]
        
        for feature in available_features:
            for window in [5, 10, 20, 50]:
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
                
                # Rolling percentile
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
            for feature in ['relative_spread', 'volatility', 'abs_momentum']:
                if feature in features_df.columns:
                    features_df[f'size_{feature}_interaction'] = features_df['order_size'] * features_df[feature]
        
        # Volatility-based interactions
        if 'volatility' in features_df.columns:
            for feature in ['is_aggressive', 'relative_spread']:
                if feature in features_df.columns:
                    features_df[f'vol_{feature}_interaction'] = features_df['volatility'] * features_df[feature]
        
        return features_df
    
    def _add_technical_indicators(self, features_df):
        """Add technical analysis indicators"""
        if 'mid_price' in features_df.columns:
            mid_price = features_df['mid_price']
            
            # Enhanced moving averages
            for period in [5, 10, 20]:
                sma = mid_price.rolling(period, min_periods=1).mean()
                ema = mid_price.ewm(span=period).mean()
                
                features_df[f'sma_{period}'] = sma
                features_df[f'ema_{period}'] = ema
                features_df[f'price_sma_ratio_{period}'] = mid_price / (sma + 1e-8)
                features_df[f'sma_ema_diff_{period}'] = (sma - ema) / (ema + 1e-8)
            
            # Bollinger Bands with additional features
            for period in [10, 20]:
                rolling_mean = mid_price.rolling(period, min_periods=1).mean()
                rolling_std = mid_price.rolling(period, min_periods=1).std()
                
                upper_band = rolling_mean + (rolling_std * 2)
                lower_band = rolling_mean - (rolling_std * 2)
                
                features_df[f'bb_position_{period}'] = (mid_price - lower_band) / (upper_band - lower_band + 1e-8)
                features_df[f'bb_squeeze_{period}'] = (upper_band - lower_band) / (rolling_mean + 1e-8)
                features_df[f'bb_breakout_{period}'] = ((mid_price > upper_band) | (mid_price < lower_band)).astype(int)
        
        return features_df
    
    def _add_statistical_features(self, features_df):
        """Add statistical distribution features"""
        numerical_features = features_df.select_dtypes(include=[np.number]).columns[:10]
        
        for feature in numerical_features:
            if features_df[feature].std() > 0:
                # Distribution moments
                features_df[f'{feature}_skewness'] = features_df[feature].rolling(50, min_periods=10).skew()
                features_df[f'{feature}_kurtosis'] = features_df[feature].rolling(50, min_periods=10).kurt()
        
        return features_df
    
    def _add_market_regime_features(self, features_df):
        """Add market regime identification features"""
        if 'volatility' in features_df.columns and 'order_size' in features_df.columns:
            # Volatility regime
            vol_quantiles = features_df['volatility'].quantile([0.33, 0.67])
            features_df['vol_regime_low'] = (features_df['volatility'] <= vol_quantiles.iloc[0]).astype(int)
            features_df['vol_regime_high'] = (features_df['volatility'] >= vol_quantiles.iloc[1]).astype(int)
            
            # Size regime
            size_quantiles = features_df['order_size'].quantile([0.33, 0.67])
            features_df['size_regime_large'] = (features_df['order_size'] >= size_quantiles.iloc[1]).astype(int)
            features_df['size_regime_small'] = (features_df['order_size'] <= size_quantiles.iloc[0]).astype(int)
        
        return features_df

class AdvancedToxicityDetector:
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.ensemble_weights = {}
        self.performance_metrics = {}
        self.best_hyperparameters = {}
        self.feature_importance = {}
        
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
        
        # Calculate feature importance for anomaly detection
        self.feature_importance = self._calculate_feature_importance(selected_features)
        
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
    
    def _calculate_feature_importance(self, features_df):
        """Calculate feature importance using multiple methods"""
        print("Calculating feature importance...")
        
        feature_importance = {}
        
        # Method 1: Variance-based importance
        variances = features_df.var()
        normalized_variances = variances / variances.max()

        # Method 2: Isolation Forest feature importance
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_forest.fit(features_df)
            anomaly_scores = iso_forest.decision_function(features_df)
            
            # Calculate correlation between each feature and anomaly scores
            correlations = {}
            for col in features_df.columns:
                corr = abs(np.corrcoef(features_df[col], anomaly_scores)[0, 1])
                correlations[col] = corr if not np.isnan(corr) else 0
            
            # Normalize correlations
            max_corr = max(correlations.values()) if correlations.values() else 1
            normalized_correlations = {k: v/max_corr for k, v in correlations.items()}
            
        except Exception:
            normalized_correlations = {col: 0 for col in features_df.columns}
        
        # Combine importance scores
        for col in features_df.columns:
            variance_score = normalized_variances.get(col, 0)
            correlation_score = normalized_correlations.get(col, 0)
            feature_importance[col] = (variance_score + correlation_score) / 2
        
        return feature_importance
    
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
    
    def optimize_hyperparameters(self, X_scaled, n_trials=50):
        """Use Optuna for hyperparameter optimization"""
        
        print(f"Optimizing hyperparameters with {n_trials} trials...")
        
        def objective(trial):
            # Sample hyperparameters
            contamination = trial.suggest_float('contamination', 0.01, 0.2)
            n_estimators = trial.suggest_int('n_estimators', 100, 500)
            max_samples = trial.suggest_categorical('max_samples', ['auto', 0.5, 0.7, 0.9])
            n_neighbors = trial.suggest_int('n_neighbors', 5, 50)
            n_clusters = trial.suggest_int('n_clusters', 3, min(25, len(X_scaled) // 20))
            
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
                
                # Evaluate clustering quality
                if len(set(cluster_labels)) > 1:
                    silhouette = silhouette_score(X_scaled, cluster_labels)
                    calinski = calinski_harabasz_score(X_scaled, cluster_labels)
                    davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
                    
                    # Combine multiple metrics
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
        
        # Enhanced Isolation Forest variants
        contamination_rates = [0.01, 0.1, 0.25]
        for i, contamination in enumerate(contamination_rates):
            try:
                iso_forest = IsolationForest(
                    contamination=contamination,
                    n_estimators=hyperparameters.get('n_estimators', 200),
                    max_samples=hyperparameters.get('max_samples', 'auto'),
                    random_state=42 + i,
                    bootstrap=True
                )
                iso_forest.fit(X_scaled)
                detectors[f'isolation_forest_{contamination}'] = iso_forest
            except Exception as e:
                print(f"Warning: Failed to train Isolation Forest with contamination {contamination}: {e}")
        
        # Enhanced LOF variants
        neighbor_counts = [5, 10, 20, 30]
        for neighbors in neighbor_counts:
            try:
                lof = LocalOutlierFactor(
                    n_neighbors=neighbors,
                    contamination=hyperparameters.get('contamination', 0.1),
                    novelty=True
                )
                lof.fit(X_scaled)
                detectors[f'lof_{neighbors}'] = lof
            except Exception as e:
                print(f"Warning: Failed to train LOF with {neighbors} neighbors: {e}")
        
        # Enhanced Gaussian Mixture Models
        component_counts = [3, 5, 8, 12]
        for n_components in component_counts:
            try:
                n_comp = min(n_components, max(2, len(X_scaled) // 50))
                if n_comp < 2:
                    continue
                
                gmm = GaussianMixture(
                    n_components=n_comp,
                    random_state=42,
                    covariance_type='tied',
                    reg_covar=1e-3,
                    init_params='kmeans',
                    max_iter=200
                )
                gmm.fit(X_scaled)
                detectors[f'gmm_{n_comp}'] = gmm
            except Exception as e:
                print(f"Warning: Failed to train GMM with {n_comp} components: {e}")
        
        # Enhanced clustering approaches
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
        
        # Optimized K-means variants
        cluster_counts = [5, 10, 15, 20]
        for n_clusters in cluster_counts:
            try:
                n_clust = min(n_clusters, max(2, len(X_scaled) // 30))
                if n_clust < 2:
                    continue
                
                kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init=20)
                cluster_labels = kmeans.fit_predict(X_scaled)
                distances = np.min(kmeans.transform(X_scaled), axis=1)
                
                clustering_detectors[f'kmeans_{n_clust}'] = {
                    'kmeans': kmeans,
                    'distance_threshold': np.percentile(distances, 95),
                    'cluster_sizes': np.bincount(cluster_labels)
                }
            except Exception as e:
                print(f"Warning: Failed to train K-means with {n_clust} clusters: {e}")
        
        # Agglomerative clustering variants
        for linkage in ['ward', 'complete', 'average']:
            try:
                best_k = min(10, max(3, len(X_scaled) // 100))
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
                print(f"Warning: Failed to train agglomerative clustering with {linkage}: {e}")
        
        return clustering_detectors
    
    def _train_dbscan_detector(self, X_scaled):
        """Train DBSCAN-based anomaly detector with enhanced parameter search"""
        print("Training DBSCAN detector...")
        
        best_score = -1
        best_params = None
        
        # Enhanced parameter search
        eps_values = np.logspace(-2, 1, 15)
        min_samples_values = [3, 5, 8, 12, 20]
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(X_scaled)
                    
                    n_outliers = np.sum(labels == -1)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    
                    if 2 <= n_clusters <= 15 and 0.01 <= n_outliers / len(X_scaled) <= 0.2:
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
                    'params': best_params,
                    'score': best_score
                }
            except Exception:
                return None
        
        return None
    
    def calculate_weighted_ensemble_scores(self, X_scaled):
        """Calculate ensemble scores with enhanced adaptive weighting"""
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
                elif 'gmm' in name:
                    scores = -model.score_samples(X_scaled)
                elif 'kmeans' in name:
                    distances = np.min(model['kmeans'].transform(X_scaled), axis=1)
                    scores = distances
                elif 'agglomerative' in name:
                    distances = np.min(cdist(X_scaled, model['cluster_centres']), axis=1)
                    scores = distances
                elif 'dbscan' in name:
                    labels = model['labels']
                    scores = np.zeros(len(X_scaled))
                    for i in range(len(X_scaled)):
                        if i < len(labels) and labels[i] == -1:
                            scores[i] = 1.0
                        else:
                            if i < len(labels):
                                cluster_points = X_scaled[labels == labels[i]]
                                if len(cluster_points) > 0:
                                    cluster_centre = np.mean(cluster_points, axis=0)
                                    scores[i] = np.linalg.norm(X_scaled[i] - cluster_centre)
                else:
                    continue
                
                # Enhanced normalization
                if len(scores) > 0 and scores.max() > scores.min():
                    # Use robust normalization
                    q25, q75 = np.percentile(scores, [25, 75])
                    iqr = q75 - q25
                    if iqr > 0:
                        scores = (scores - q25) / iqr
                        scores = np.clip(scores, 0, 3)  # Cap extreme values
                    else:
                        scores = (scores - scores.min()) / (scores.max() - scores.min())
                else:
                    scores = np.zeros_like(scores)
                
                individual_scores[name] = scores
                
                # Enhanced weight calculation
                weight = self._calculate_enhanced_detector_weight(scores, X_scaled, name)
                individual_weights[name] = weight
                
            except Exception as e:
                print(f"Error calculating scores for {name}: {e}")
                continue
        
        # Enhanced weight normalization with stability constraints
        total_weight = sum(individual_weights.values())
        if total_weight > 0:
            # Prevent any single detector from dominating
            max_weight = 0.3  # Maximum 30% weight for any detector
            normalized_weights = {}
            for name, weight in individual_weights.items():
                normalized_weights[name] = min(weight / total_weight, max_weight)
            
            # Renormalize after capping
            total_capped_weight = sum(normalized_weights.values())
            individual_weights = {name: weight / total_capped_weight 
                                for name, weight in normalized_weights.items()}
        else:
            individual_weights = {name: 1.0 / len(individual_scores) 
                                for name in individual_scores}
        
        # Calculate weighted ensemble scores
        ensemble_scores = np.zeros(len(X_scaled))
        for name, scores in individual_scores.items():
            weight = individual_weights[name]
            ensemble_scores += weight * scores
        
        self.ensemble_weights = individual_weights
        
        print(f"Enhanced ensemble weights:")
        for name, weight in sorted(individual_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {weight:.3f}")
        
        return ensemble_scores, individual_scores
    
    def _calculate_enhanced_detector_weight(self, scores, X_scaled, detector_name):
        """Enhanced weight calculation with multiple criteria"""
        try:
            # Criterion 1: Score distribution quality
            score_std = np.std(scores)
            score_range = np.max(scores) - np.min(scores)
            separation_score = score_std * score_range
            
            # Criterion 2: Anomaly detection consistency across thresholds
            performance_scores = []
            for threshold_pct in [90, 95, 99]:
                threshold = np.percentile(scores, threshold_pct)
                anomaly_rate = np.mean(scores > threshold)
                expected_rate = (100 - threshold_pct) / 100
                
                if expected_rate > 0:
                    rate_score = 1 - abs(anomaly_rate - expected_rate) / expected_rate
                    performance_scores.append(max(0, rate_score))
            
            avg_performance = np.mean(performance_scores) if performance_scores else 0
            
            # Criterion 3: Detector type bonus (prefer diverse methods)
            type_bonus = 1.0
            if 'isolation_forest' in detector_name:
                type_bonus = 1.1  # Slight preference for isolation forests
            elif 'lof' in detector_name:
                type_bonus = 1.05
            elif 'gmm' in detector_name:
                type_bonus = 1.0
            elif 'dbscan' in detector_name:
                type_bonus = 0.9  # Slightly lower for DBSCAN due to potential instability
            
            # Criterion 4: Score stability (prefer less volatile detectors)
            score_volatility = np.std(np.diff(scores))
            stability_score = 1 / (1 + score_volatility)
            
            # Combine all criteria
            weight = separation_score * avg_performance * type_bonus * stability_score
            
            return max(0.05, min(weight, 2.0))  # Constrain weights
            
        except Exception:
            return 0.1
    
    def evaluate_enhanced_performance(self, X_scaled, ensemble_scores, individual_scores):
        """Enhanced performance evaluation with comprehensive metrics"""
        print("Evaluating enhanced detector performance...")
        
        metrics = {}
        
        # Enhanced clustering quality metrics
        clustering_metrics = self._evaluate_clustering_quality(X_scaled)
        if clustering_metrics:
            metrics.update(clustering_metrics)
        
        # Enhanced anomaly detection metrics
        anomaly_metrics = self._evaluate_anomaly_detection_quality(ensemble_scores)
        metrics['anomaly_detection'] = anomaly_metrics
        
        # Enhanced individual detector performance
        individual_performance = self._evaluate_individual_detectors(individual_scores)
        metrics['individual_detectors'] = individual_performance
        
        # Enhanced ensemble quality metrics
        ensemble_metrics = self._evaluate_ensemble_quality(ensemble_scores, individual_scores)
        metrics['ensemble_statistics'] = ensemble_metrics
        
        # Enhanced diversity metrics
        diversity_metrics = self._evaluate_ensemble_diversity(individual_scores)
        if diversity_metrics:
            metrics['ensemble_diversity'] = diversity_metrics
        
        # Feature importance metrics
        if self.feature_importance:
            metrics['feature_importance'] = dict(sorted(
                self.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:20])  # Top 20 features
        
        self.performance_metrics = metrics
        
        return metrics
    
    def _evaluate_clustering_quality(self, X_scaled):
        """Evaluate clustering quality across multiple detectors"""
        clustering_metrics = {}
        
        for name, model in self.models.items():
            if 'kmeans' in name and isinstance(model, dict):
                try:
                    kmeans = model['kmeans']
                    cluster_labels = kmeans.predict(X_scaled)
                    
                    if len(set(cluster_labels)) > 1:
                        sil_score = silhouette_score(X_scaled, cluster_labels)
                        db_score = davies_bouldin_score(X_scaled, cluster_labels)
                        ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
                        
                        clustering_metrics[f'{name}_silhouette'] = sil_score
                        clustering_metrics[f'{name}_davies_bouldin'] = db_score
                        clustering_metrics[f'{name}_calinski_harabasz'] = ch_score
                        
                except Exception:
                    continue
        
        # Overall clustering quality (average across all clustering methods)
        if clustering_metrics:
            sil_scores = [v for k, v in clustering_metrics.items() if 'silhouette' in k]
            if sil_scores:
                clustering_metrics['average_silhouette'] = np.mean(sil_scores)
        
        return clustering_metrics
    
    def _evaluate_anomaly_detection_quality(self, ensemble_scores):
        """Enhanced anomaly detection quality evaluation"""
        anomaly_metrics = {}
        
        # Multi-threshold analysis
        thresholds = [85, 90, 95, 97, 99, 99.5]
        for threshold_pct in thresholds:
            threshold = np.percentile(ensemble_scores, threshold_pct)
            anomaly_labels = ensemble_scores > threshold
            anomaly_rate = anomaly_labels.mean()
            
            anomaly_metrics[f'anomaly_rate_{threshold_pct}th'] = anomaly_rate
            
            if anomaly_labels.sum() > 0:
                anomaly_scores = ensemble_scores[anomaly_labels]
                normal_scores = ensemble_scores[~anomaly_labels]
                
                anomaly_metrics[f'anomaly_score_mean_{threshold_pct}th'] = anomaly_scores.mean()
                anomaly_metrics[f'normal_score_mean_{threshold_pct}th'] = normal_scores.mean()
                
                if normal_scores.std() > 0:
                    separation = (anomaly_scores.mean() - normal_scores.mean()) / normal_scores.std()
                    anomaly_metrics[f'separation_{threshold_pct}th'] = separation
        
        # Score distribution analysis
        anomaly_metrics['score_entropy'] = stats.entropy(
            np.histogram(ensemble_scores, bins=50)[0] + 1e-10
        )
        
        return anomaly_metrics
    
    def _evaluate_individual_detectors(self, individual_scores):
        """Enhanced individual detector performance evaluation"""
        individual_performance = {}
        
        for name, scores in individual_scores.items():
            try:
                performance = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'min': scores.min(),
                    'max': scores.max(),
                    'range': scores.max() - scores.min(),
                    'skewness': stats.skew(scores),
                    'kurtosis': stats.kurtosis(scores),
                    'weight': self.ensemble_weights.get(name, 0),
                    'iqr': np.percentile(scores, 75) - np.percentile(scores, 25),
                    'coefficient_variation': scores.std() / (scores.mean() + 1e-8)
                }
                
                # Detection consistency across thresholds
                consistency_scores = []
                for pct in [90, 95, 99]:
                    threshold = np.percentile(scores, pct)
                    detection_rate = np.mean(scores > threshold)
                    expected_rate = (100 - pct) / 100
                    consistency = 1 - abs(detection_rate - expected_rate) / expected_rate
                    consistency_scores.append(max(0, consistency))
                
                performance['consistency'] = np.mean(consistency_scores)
                individual_performance[name] = performance
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                continue
        
        return individual_performance
    
    def _evaluate_ensemble_quality(self, ensemble_scores, individual_scores):
        """Enhanced ensemble quality evaluation"""
        ensemble_metrics = {
            'score_mean': ensemble_scores.mean(),
            'score_std': ensemble_scores.std(),
            'score_range': ensemble_scores.max() - ensemble_scores.min(),
            'score_skewness': stats.skew(ensemble_scores),
            'score_kurtosis': stats.kurtosis(ensemble_scores),
            'effective_detectors': len([w for w in self.ensemble_weights.values() if w > 0.05]),
            'weight_concentration': max(self.ensemble_weights.values()),  # Highest individual weight
            'weight_entropy': stats.entropy(list(self.ensemble_weights.values()) + [1e-10])
        }
        
        # Ensemble stability (how much ensemble varies from individual detectors)
        if len(individual_scores) > 1:
            individual_means = [scores.mean() for scores in individual_scores.values()]
            ensemble_stability = 1 - (abs(ensemble_scores.mean() - np.mean(individual_means)) / 
                                    (np.std(individual_means) + 1e-8))
            ensemble_metrics['stability'] = max(0, ensemble_stability)
        
        return ensemble_metrics
    
    def _evaluate_ensemble_diversity(self, individual_scores):
        """Enhanced ensemble diversity evaluation"""
        if len(individual_scores) < 2:
            return None
        
        try:
            score_matrix = np.array(list(individual_scores.values())).T
            correlation_matrix = np.corrcoef(score_matrix.T)
            
            # Remove NaN values
            correlation_matrix = np.nan_to_num(correlation_matrix)
            
            # Diversity metrics
            upper_triangle_indices = np.triu_indices_from(correlation_matrix, k=1)
            correlations = correlation_matrix[upper_triangle_indices]
            
            diversity_metrics = {
                'avg_correlation': np.mean(correlations),
                'max_correlation': np.max(correlations),
                'min_correlation': np.min(correlations),
                'correlation_std': np.std(correlations),
                'diversity_score': 1 - abs(np.mean(correlations)),
                'pairwise_disagreement': np.mean(correlations < 0.5)  # Fraction of pairs with low correlation
            }
            
            return diversity_metrics
            
        except Exception as e:
            print(f"Error calculating diversity metrics: {e}")
            return None
    
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
            'feature_importance': self.feature_importance,
            'timestamp': timestamp,
            'version': '3.0_enhanced_comprehensive',
            'n_features': len(self.feature_selector) if self.feature_selector else 0,
            'n_detectors': len(self.models),
            'training_summary': {
                'top_features': dict(sorted(self.feature_importance.items(), 
                                            key=lambda x: x[1], reverse=True)[:10]),
                'detector_weights': dict(sorted(self.ensemble_weights.items(), 
                                                key=lambda x: x[1], reverse=True)),
                'ensemble_diversity': self.performance_metrics.get('ensemble_diversity', {}),
                'best_clustering_score': max([v for k, v in self.performance_metrics.items() 
                                                if 'silhouette' in k and isinstance(v, (int, float))], 
                                            default=0)
            }
        }
        
        joblib.dump(model_package, model_path)
        
        print(f"Enhanced model saved to: {model_path}")
        return model_path

class ComprehensivePlotter:
    """Enhanced plotting class with separate, publication-quality plots"""
    
    def __init__(self, save_dir="enhanced_production_plots"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set publication-quality defaults
        plt.rcParams.update({
            'figure.figsize': (12, 8),
           'font.size': 12,
           'axes.titlesize': 14,
           'axes.labelsize': 12,
           'xtick.labelsize': 10,
           'ytick.labelsize': 10,
           'legend.fontsize': 10,
           'figure.dpi': 300,
           'savefig.dpi': 300,
           'savefig.bbox': 'tight',
           'savefig.facecolor': 'white'
       })
   
    def plot_feature_analysis(self, features_df, feature_importance, detector):
        """Create comprehensive feature analysis plots"""
        
        # 1. Feature Importance Plot
        plt.figure(figsize=(14, 8))
        top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20])
        
        features_names = list(top_features.keys())
        feature_scores = list(top_features.values())
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(features_names)))
        bars = plt.barh(range(len(features_names)), feature_scores, color=colors, alpha=0.8)
        
        plt.yticks(range(len(features_names)), [name.replace('_', ' ')[:25] for name in features_names])
        plt.xlabel('Feature Importance Score')
        plt.title('Top 20 Most Important Features for Toxicity Detection', fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, feature_scores)):
            plt.text(score + 0.001, i, f'{score:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/01_feature_importance_{self.timestamp}.png")
        plt.close()
        
        # 2. Feature Correlation Network
        plt.figure(figsize=(14, 10))
        
        # Select top features for correlation analysis
        top_feature_names = list(top_features.keys())[:15]
        if len(top_feature_names) > 1:
            correlation_matrix = features_df[top_feature_names].corr()
            
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            im = plt.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            # Add correlation values
            for i in range(len(correlation_matrix)):
                for j in range(len(correlation_matrix)):
                    if not mask[i, j] and abs(correlation_matrix.iloc[i, j]) > 0.3:
                        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                                ha='center', va='center', fontsize=8)
            
            plt.xticks(range(len(top_feature_names)), 
                        [name.replace('_', ' ')[:15] for name in top_feature_names], 
                        rotation=45, ha='right')
            plt.yticks(range(len(top_feature_names)), 
                        [name.replace('_', ' ')[:15] for name in top_feature_names])
            
            plt.colorbar(im, label='Correlation Coefficient')
            plt.title('Feature Correlation Matrix (Top 15 Features)', fontweight='bold', pad=20)
            
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/02_feature_correlation_{self.timestamp}.png")
        plt.close()
        
        # 3. Feature Distribution Analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        key_features = ['order_size', 'volatility', 'spread', 'momentum', 'order_book_imbalance', 'inter_arrival_time']
        available_features = [f for f in key_features if f in features_df.columns]
        
        for i, feature in enumerate(available_features[:6]):
            if i < len(axes):
                ax = axes[i]
                
                # Create histogram with KDE
                feature_data = features_df[feature].dropna()
                if len(feature_data) > 0:
                    ax.hist(feature_data, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
                    
                    # Add KDE line
                    try:
                        from scipy.stats import gaussian_kde
                        kde = gaussian_kde(feature_data)
                        x_range = np.linspace(feature_data.min(), feature_data.max(), 200)
                        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                    except:
                        pass
                    
                    ax.set_xlabel(feature.replace('_', ' ').title())
                    ax.set_ylabel('Density')
                    ax.set_title(f'Distribution: {feature.replace("_", " ").title()}', fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
        
        # Remove empty subplots
        for i in range(len(available_features), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Key Feature Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/03_feature_distributions_{self.timestamp}.png")
        plt.close()
    
    def plot_ensemble_analysis(self, individual_scores, ensemble_scores, detector):
        """Create comprehensive ensemble analysis plots"""
        
        # 1. Detector Weights Visualization
        plt.figure(figsize=(14, 8))
        
        weights = list(detector.ensemble_weights.values())
        detector_names = list(detector.ensemble_weights.keys())
        
        # Sort by weight
        sorted_items = sorted(zip(detector_names, weights), key=lambda x: x[1], reverse=True)
        detector_names, weights = zip(*sorted_items)
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(weights)))
        bars = plt.bar(range(len(weights)), weights, color=colors, alpha=0.8)
        
        plt.xticks(range(len(detector_names)), 
                    [name.replace('_', ' ')[:20] for name in detector_names], 
                    rotation=45, ha='right')
        plt.ylabel('Ensemble Weight')
        plt.title('Detector Weights in Ensemble (Sorted by Weight)', fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, weight in zip(bars, weights):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/04_detector_weights_{self.timestamp}.png")
        plt.close()
        
        # 2. Score Correlation Heatmap
        plt.figure(figsize=(14, 12))
        
        if len(individual_scores) > 1:
            # Select top detectors by weight for visualization
            top_detectors = sorted(detector.ensemble_weights.items(), key=lambda x: x[1], reverse=True)[:12]
            top_detector_names = [name for name, _ in top_detectors]
            
            score_data = {name: individual_scores[name] for name in top_detector_names 
                            if name in individual_scores}
            
            if len(score_data) > 1:
                score_df = pd.DataFrame(score_data)
                correlation_matrix = score_df.corr()
                
                # Create heatmap
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.3f',
                            cmap='RdBu_r', center=0, square=True, cbar_kws={"shrink": .8})
                
                plt.title('Detector Score Correlations (Top 12 Detectors)', fontweight='bold', pad=20)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/05_score_correlations_{self.timestamp}.png")
        plt.close()
        
        # 3. Ensemble vs Individual Detector Performance
        plt.figure(figsize=(16, 10))
        
        # Create subplot for score distributions
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Ensemble score distribution
        ax1.hist(ensemble_scores, bins=50, alpha=0.7, color='darkblue', edgecolor='black', density=True)
        ax1.axvline(np.percentile(ensemble_scores, 95), color='red', linestyle='--', linewidth=2, label='95th Percentile')
        ax1.axvline(np.percentile(ensemble_scores, 99), color='darkred', linestyle='--', linewidth=2, label='99th Percentile')
        ax1.set_xlabel('Ensemble Anomaly Score')
        ax1.set_ylabel('Density')
        ax1.set_title('Ensemble Score Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Individual detector score ranges
        if len(individual_scores) > 1:
            detector_names = list(individual_scores.keys())[:10]  # Top 10 for readability
            score_ranges = []
            score_means = []
            
            for name in detector_names:
                scores = individual_scores[name]
                score_ranges.append([scores.min(), scores.max()])
                score_means.append(scores.mean())
            
            # Plot score ranges
            for i, (name, (min_score, max_score), mean_score) in enumerate(zip(detector_names, score_ranges, score_means)):
                ax2.plot([min_score, max_score], [i, i], 'b-', linewidth=3, alpha=0.6)
                ax2.plot(mean_score, i, 'ro', markersize=6)
            
            ax2.set_yticks(range(len(detector_names)))
            ax2.set_yticklabels([name.replace('_', ' ')[:15] for name in detector_names])
            ax2.set_xlabel('Score Range')
            ax2.set_title('Individual Detector Score Ranges', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # Detector performance vs weight
        if len(individual_scores) > 1:
            weights = [detector.ensemble_weights.get(name, 0) for name in individual_scores.keys()]
            score_stds = [individual_scores[name].std() for name in individual_scores.keys()]
            score_means = [individual_scores[name].mean() for name in individual_scores.keys()]
            
            scatter = ax3.scatter(weights, score_stds, c=score_means, cmap='viridis', 
                                s=100, alpha=0.7, edgecolors='black')
            ax3.set_xlabel('Ensemble Weight')
            ax3.set_ylabel('Score Standard Deviation')
            ax3.set_title('Detector Weight vs Score Variability', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax3, label='Mean Score')
        
        # Anomaly rate by threshold
        thresholds = np.percentile(ensemble_scores, np.arange(80, 100, 1))
        anomaly_rates = [np.mean(ensemble_scores > threshold) for threshold in thresholds]
        
        ax4.plot(np.arange(80, 100, 1), np.array(anomaly_rates) * 100, 'g-', linewidth=2, marker='o')
        ax4.set_xlabel('Percentile Threshold')
        ax4.set_ylabel('Anomaly Rate (%)')
        ax4.set_title('Anomaly Detection Rate by Threshold', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(80, 99)
        
        plt.suptitle('Ensemble Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/06_ensemble_performance_{self.timestamp}.png")
        plt.close()
    
    def plot_anomaly_analysis(self, features_df, ensemble_scores, individual_scores):
        """Create comprehensive anomaly analysis plots"""
        
        # 1. Anomaly Score Timeline
        plt.figure(figsize=(16, 8))
        
        plt.plot(ensemble_scores, alpha=0.7, color='blue', linewidth=1)
        plt.axhline(np.percentile(ensemble_scores, 95), color='orange', linestyle='--', 
                    linewidth=2, label='95th Percentile')
        plt.axhline(np.percentile(ensemble_scores, 99), color='red', linestyle='--', 
                    linewidth=2, label='99th Percentile')
        plt.axhline(np.percentile(ensemble_scores, 99.9), color='darkred', linestyle='--', 
                    linewidth=2, label='99.9th Percentile')
        
        plt.xlabel('Order Sequence')
        plt.ylabel('Anomaly Score')
        plt.title('Anomaly Scores Over Time', fontweight='bold', pad=20)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Highlight top anomalies
        top_anomalies = np.where(ensemble_scores > np.percentile(ensemble_scores, 99.5))[0]
        if len(top_anomalies) > 0:
            plt.scatter(top_anomalies, ensemble_scores[top_anomalies], 
                        color='red', s=50, alpha=0.8, zorder=5, label='Top 0.5% Anomalies')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/07_anomaly_timeline_{self.timestamp}.png")
        plt.close()
        
        # 2. Anomaly Characteristics Analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Define anomaly threshold
        anomaly_threshold = np.percentile(ensemble_scores, 95)
        anomaly_mask = ensemble_scores > anomaly_threshold
        
        key_features = ['order_size', 'volatility', 'spread', 'momentum', 'inter_arrival_time', 'relative_spread']
        available_features = [f for f in key_features if f in features_df.columns]
        
        for i, feature in enumerate(available_features[:6]):
            if i < len(axes):
                ax = axes[i]
                
                normal_data = features_df[feature][~anomaly_mask].dropna()
                anomaly_data = features_df[feature][anomaly_mask].dropna()
                
                if len(normal_data) > 0 and len(anomaly_data) > 0:
                    # Create overlapping histograms
                    ax.hist(normal_data, bins=30, alpha=0.6, label='Normal', 
                            color='lightblue', density=True, edgecolor='black')
                    ax.hist(anomaly_data, bins=30, alpha=0.6, label='Anomalous', 
                            color='red', density=True, edgecolor='black')
                    
                    ax.set_xlabel(feature.replace('_', ' ').title())
                    ax.set_ylabel('Density')
                    ax.set_title(f'{feature.replace("_", " ").title()}: Normal vs Anomalous', fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Add statistics
                    normal_mean = normal_data.mean()
                    anomaly_mean = anomaly_data.mean()
                    ax.axvline(normal_mean, color='blue', linestyle=':', alpha=0.8, label=f'Normal Mean: {normal_mean:.3f}')
                    ax.axvline(anomaly_mean, color='red', linestyle=':', alpha=0.8, label=f'Anomaly Mean: {anomaly_mean:.3f}')
        
        # Remove empty subplots
        for i in range(len(available_features), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Feature Distributions: Normal vs Anomalous Orders', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/08_anomaly_characteristics_{self.timestamp}.png")
        plt.close()
        
        # 3. Anomaly Detection ROC-style Analysis
        plt.figure(figsize=(14, 8))
        
        # Calculate detection rates at different thresholds
        thresholds = np.percentile(ensemble_scores, np.arange(50, 100, 1))
        detection_rates = []
        false_positive_rates = []
        
        for threshold in thresholds:
            detected = ensemble_scores > threshold
            detection_rate = np.mean(detected)
            detection_rates.append(detection_rate)
            
            # Approximate false positive rate (assuming top 5% are true positives)
            true_anomalies = ensemble_scores > np.percentile(ensemble_scores, 95)
            false_positives = detected & ~true_anomalies
            false_positive_rate = np.mean(false_positives) / np.mean(~true_anomalies) if np.mean(~true_anomalies) > 0 else 0
            false_positive_rates.append(false_positive_rate)
        
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(50, 100, 1), np.array(detection_rates) * 100, 'b-', linewidth=2, marker='o')
        plt.xlabel('Percentile Threshold')
        plt.ylabel('Detection Rate (%)')
        plt.title('Detection Rate by Threshold', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(false_positive_rates, detection_rates, 'g-', linewidth=2, marker='o')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-style Curve', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/09_detection_performance_{self.timestamp}.png")
        plt.close()
    
    def plot_model_diagnostics(self, X_scaled, ensemble_scores, detector):
        """Create model diagnostic plots"""
        
        # 1. PCA Analysis with Anomaly Overlay
        if X_scaled.shape[1] > 2:
            plt.figure(figsize=(16, 6))
            
            # PCA transformation
            pca = PCA()
            X_pca_full = pca.fit_transform(X_scaled)
            
            # Explained variance plot
            plt.subplot(1, 3, 1)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            plt.plot(range(1, min(21, len(cumvar) + 1)), cumvar[:20], 'bo-', linewidth=2)
            plt.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='95% Variance')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('PCA Explained Variance', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 2D PCA with anomaly coloring
            plt.subplot(1, 3, 2)
            pca_2d = PCA(n_components=2)
            X_pca_2d = pca_2d.fit_transform(X_scaled)
            
            scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=ensemble_scores, 
                                    cmap='viridis', alpha=0.6, s=20)
            plt.colorbar(scatter, label='Anomaly Score')
            plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
            plt.title('PCA Space - Anomaly Intensity', fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Anomaly clusters in PCA space
            plt.subplot(1, 3, 3)
            anomaly_threshold = np.percentile(ensemble_scores, 95)
            normal_mask = ensemble_scores <= anomaly_threshold
            anomaly_mask = ensemble_scores > anomaly_threshold
            
            plt.scatter(X_pca_2d[normal_mask, 0], X_pca_2d[normal_mask, 1],
                        c='lightblue', alpha=0.6, s=15, label='Normal')
            if anomaly_mask.sum() > 0:
                plt.scatter(X_pca_2d[anomaly_mask, 0], X_pca_2d[anomaly_mask, 1],
                            c='red', alpha=0.8, s=40, marker='D', label='Anomalous')
            
            plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
            plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
            plt.title('PCA Space - Normal vs Anomalous', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/10_pca_analysis_{self.timestamp}.png")
            plt.close()
        
        # 2. Model Performance Metrics
        if hasattr(detector, 'performance_metrics') and detector.performance_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Ensemble diversity metrics
            if 'ensemble_diversity' in detector.performance_metrics:
                diversity = detector.performance_metrics['ensemble_diversity']
                ax = axes[0, 0]
                
                metrics = ['avg_correlation', 'max_correlation', 'correlation_std', 'diversity_score']
                values = [diversity.get(metric, 0) for metric in metrics]
                metric_labels = ['Avg Correlation', 'Max Correlation', 'Correlation Std', 'Diversity Score']
                
                bars = ax.bar(metric_labels, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'], alpha=0.8)
                ax.set_ylabel('Score')
                ax.set_title('Ensemble Diversity Metrics', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{value:.3f}', ha='center', va='bottom')
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Individual detector performance
            if 'individual_detectors' in detector.performance_metrics:
                ax = axes[0, 1]
                individual_perf = detector.performance_metrics['individual_detectors']
                
                detector_names = list(individual_perf.keys())[:10]  # Top 10
                consistencies = [individual_perf[name].get('consistency', 0) for name in detector_names]
                weights = [individual_perf[name].get('weight', 0) for name in detector_names]
                
                scatter = ax.scatter(weights, consistencies, s=100, alpha=0.7, c=range(len(detector_names)), cmap='tab10')
                ax.set_xlabel('Ensemble Weight')
                ax.set_ylabel('Detection Consistency')
                ax.set_title('Detector Weight vs Consistency', fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add labels for top performers
                for i, name in enumerate(detector_names[:5]):
                    ax.annotate(name.replace('_', ' ')[:15], (weights[i], consistencies[i]), 
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Anomaly detection rates
            if 'anomaly_detection' in detector.performance_metrics:
                ax = axes[1, 0]
                anomaly_metrics = detector.performance_metrics['anomaly_detection']
                
                thresholds = [90, 95, 97, 99]
                rates = [anomaly_metrics.get(f'anomaly_rate_{th}th', 0) * 100 for th in thresholds]
                
                bars = ax.bar([f'{th}th' for th in thresholds], rates, 
                                color=['lightblue', 'orange', 'red', 'darkred'], alpha=0.8)
                ax.set_ylabel('Anomaly Rate (%)')
                ax.set_title('Anomaly Detection Rates by Threshold', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, rate in zip(bars, rates):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                            f'{rate:.2f}%', ha='center', va='bottom')
            
            # Clustering quality metrics
            ax = axes[1, 1]
            clustering_scores = []
            clustering_names = []
            
            for metric_name, value in detector.performance_metrics.items():
                if 'silhouette' in metric_name and isinstance(value, (int, float)):
                    clustering_scores.append(value)
                    clustering_names.append(metric_name.replace('_silhouette', '').replace('_', ' '))
            
            if clustering_scores:
                bars = ax.bar(range(len(clustering_names)), clustering_scores, 
                                color='lightgreen', alpha=0.8)
                ax.set_xticks(range(len(clustering_names)))
                ax.set_xticklabels(clustering_names, rotation=45, ha='right')
                ax.set_ylabel('Silhouette Score')
                ax.set_title('Clustering Quality by Method', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, score in zip(bars, clustering_scores):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{score:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.suptitle('Model Performance Diagnostics', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/11_model_diagnostics_{self.timestamp}.png")
            plt.close()
    
    def plot_hyperparameter_analysis(self, detector):
        """Create hyperparameter optimization analysis plots"""
        
        if not hasattr(detector, 'best_hyperparameters') or not detector.best_hyperparameters:
            return
        
        plt.figure(figsize=(14, 8))
        
        # Hyperparameter values visualization
        params = detector.best_hyperparameters
        param_names = list(params.keys())
        param_values = []
        
        # Convert parameters to numerical values for visualization
        for param, value in params.items():
            if isinstance(value, (int, float)):
                param_values.append(value)
            elif value == 'auto':
                param_values.append(1)  # Placeholder for 'auto'
            else:
                param_values.append(hash(str(value)) % 100)  # Hash for other string values
        
        # Normalize values for better visualization
        if len(param_values) > 1:
            normalized_values = [(v - min(param_values)) / (max(param_values) - min(param_values)) 
                                for v in param_values]
        else:
            normalized_values = param_values
        
        bars = plt.bar(range(len(param_names)), normalized_values, 
                        color=plt.cm.viridis(np.linspace(0, 1, len(param_names))), alpha=0.8)
        
        plt.xticks(range(len(param_names)), 
                    [name.replace('_', ' ').title() for name in param_names], rotation=45, ha='right')
        plt.ylabel('Normalized Parameter Value')
        plt.title('Optimized Hyperparameters', fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add actual values as text
        for i, (bar, param_name, original_value) in enumerate(zip(bars, param_names, params.values())):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{original_value}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/12_hyperparameters_{self.timestamp}.png")
        plt.close()
    
    def create_summary_report(self, detector, ensemble_scores, features_df):
        """Create a comprehensive summary report plot"""
        
        fig = plt.figure(figsize=(20, 14))
        
        # Create a complex layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Model Overview (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Key metrics summary
        metrics_text = f"""
        MODEL PERFORMANCE SUMMARY
        
        • Total Features: {len(detector.feature_selector) if detector.feature_selector else 0}
        • Active Detectors: {len(detector.models)}
        • Ensemble Diversity: {detector.performance_metrics.get('ensemble_diversity', {}).get('diversity_score', 0):.3f}
        
        • Anomaly Rate (95th): {detector.performance_metrics.get('anomaly_detection', {}).get('anomaly_rate_95th', 0)*100:.2f}%
        • Anomaly Rate (99th): {detector.performance_metrics.get('anomaly_detection', {}).get('anomaly_rate_99th', 0)*100:.2f}%
        
        • Top Detector: {max(detector.ensemble_weights, key=detector.ensemble_weights.get) if detector.ensemble_weights else 'N/A'}
        • Best Weight: {max(detector.ensemble_weights.values()) if detector.ensemble_weights else 0:.3f}
        """
        
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('Model Summary', fontsize=14, fontweight='bold')
        
        # 2. Score distribution (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.hist(ensemble_scores, bins=40, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.percentile(ensemble_scores, 95), color='orange', linestyle='--', linewidth=2, label='95th')
        ax2.axvline(np.percentile(ensemble_scores, 99), color='red', linestyle='--', linewidth=2, label='99th')
        ax2.set_xlabel('Ensemble Anomaly Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Anomaly Score Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Top detector weights (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        if detector.ensemble_weights:
            top_5_detectors = sorted(detector.ensemble_weights.items(), key=lambda x: x[1], reverse=True)[:5]
            names, weights = zip(*top_5_detectors)
            
            bars = ax3.barh(range(len(names)), weights, color=plt.cm.Set3(np.arange(len(names))))
            ax3.set_yticks(range(len(names)))
            ax3.set_yticklabels([name.replace('_', ' ')[:20] for name in names])
            ax3.set_xlabel('Ensemble Weight')
            ax3.set_title('Top 5 Detector Weights', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, weight in zip(bars, weights):
                ax3.text(weight + 0.005, bar.get_y() + bar.get_height()/2, 
                        f'{weight:.3f}', va='center', fontsize=10)
        
        # 4. Feature importance (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        if hasattr(detector, 'feature_importance') and detector.feature_importance:
            top_features = dict(sorted(detector.feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True)[:8])
            
            names = list(top_features.keys())
            scores = list(top_features.values())
            
            bars = ax4.barh(range(len(names)), scores, color=plt.cm.viridis(np.linspace(0, 1, len(names))))
            ax4.set_yticks(range(len(names)))
            ax4.set_yticklabels([name.replace('_', ' ')[:15] for name in names])
            ax4.set_xlabel('Importance Score')
            ax4.set_title('Top 8 Feature Importance', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        # 5. Anomaly timeline (bottom left)
        ax5 = fig.add_subplot(gs[2, :2])
        sample_indices = np.linspace(0, len(ensemble_scores)-1, min(1000, len(ensemble_scores)), dtype=int)
        sampled_scores = ensemble_scores[sample_indices]
        
        ax5.plot(sample_indices, sampled_scores, alpha=0.7, color='blue', linewidth=1)
        ax5.axhline(np.percentile(ensemble_scores, 95), color='orange', linestyle='--', alpha=0.8)
        ax5.axhline(np.percentile(ensemble_scores, 99), color='red', linestyle='--', alpha=0.8)
        ax5.set_xlabel('Sample Index')
        ax5.set_ylabel('Anomaly Score')
        ax5.set_title('Anomaly Score Timeline (Sampled)', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Detection rates by threshold (bottom right)
        ax6 = fig.add_subplot(gs[2, 2:])
        thresholds = [85, 90, 95, 97, 99, 99.5]
        rates = []
        for threshold in thresholds:
            rate = np.mean(ensemble_scores > np.percentile(ensemble_scores, threshold)) * 100
            rates.append(rate)
        
        bars = ax6.bar(range(len(thresholds)), rates, 
                        color=['lightblue', 'lightgreen', 'orange', 'coral', 'red', 'darkred'], alpha=0.8)
        ax6.set_xticks(range(len(thresholds)))
        ax6.set_xticklabels([f'{t}th' for t in thresholds])
        ax6.set_ylabel('Detection Rate (%)')
        ax6.set_title('Detection Rates by Threshold', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, rate in zip(bars, rates):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{rate:.2f}%', ha='center', va='bottom', fontsize=9)
        
        # 7. Model diagnostics (bottom)
        ax7 = fig.add_subplot(gs[3, :])
        
        # Create diagnostic summary
        diagnostics_text = f"""
        DIAGNOSTIC SUMMARY:
        
        Ensemble Composition: {len(detector.models)} detectors trained successfully
        Feature Selection: {len(detector.feature_selector) if detector.feature_selector else 0} features selected from original set
        Optimization: Hyperparameters optimized using {detector.best_hyperparameters.get('n_estimators', 'N/A')} estimators
        
        Performance Indicators:
        • Score Range: {ensemble_scores.min():.3f} - {ensemble_scores.max():.3f}
        • Score Mean: {ensemble_scores.mean():.3f} ± {ensemble_scores.std():.3f}
        • Skewness: {stats.skew(ensemble_scores):.3f} (higher values indicate more extreme anomalies)
        
        Quality Metrics:
        • Ensemble Diversity: {detector.performance_metrics.get('ensemble_diversity', {}).get('diversity_score', 0):.3f}/1.0
        • Weight Concentration: {max(detector.ensemble_weights.values()) if detector.ensemble_weights else 0:.3f} (lower is better)
        • Effective Detectors: {len([w for w in detector.ensemble_weights.values() if w > 0.05])} contributing meaningfully
        """
        
        ax7.text(0.02, 0.98, diagnostics_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        ax7.set_xlim(0, 1)
        ax7.set_ylim(0, 1)
        ax7.axis('off')
        ax7.set_title('System Diagnostics', fontsize=14, fontweight='bold')
        
        # Overall title
        fig.suptitle(f'Enhanced Toxicity Detection System - Comprehensive Report\nGenerated: {self.timestamp}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig(f"{self.save_dir}/00_comprehensive_summary_{self.timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nComprehensive summary report saved: {self.save_dir}/00_comprehensive_summary_{self.timestamp}.png")
    
    def generate_all_plots(self, features_df, ensemble_scores, individual_scores, detector, X_scaled):
        """Generate all plots in sequence"""
        
        print(f"\n{'='*60}")
        print("GENERATING COMPREHENSIVE VISUALIZATION SUITE")
        print(f"{'='*60}")
        
        plot_functions = [
            ("Feature Analysis", lambda: self.plot_feature_analysis(features_df, detector.feature_importance, detector)),
            ("Ensemble Analysis", lambda: self.plot_ensemble_analysis(individual_scores, ensemble_scores, detector)),
            ("Anomaly Analysis", lambda: self.plot_anomaly_analysis(features_df, ensemble_scores, individual_scores)),
            ("Model Diagnostics", lambda: self.plot_model_diagnostics(X_scaled, ensemble_scores, detector)),
            ("Hyperparameter Analysis", lambda: self.plot_hyperparameter_analysis(detector)),
            ("Summary Report", lambda: self.create_summary_report(detector, ensemble_scores, features_df))
        ]
        
        for plot_name, plot_function in plot_functions:
            try:
                print(f"Creating {plot_name}...")
                plot_function()
                print(f"✓ {plot_name} completed")
            except Exception as e:
                print(f"✗ Error creating {plot_name}: {e}")
        
        print(f"\n✓ All plots saved to: {self.save_dir}")
        print(f"✓ Timestamp: {self.timestamp}")
        
        # List all generated files
        plot_files = [f for f in os.listdir(self.save_dir) if f.endswith('.png') and self.timestamp in f]
        print(f"\nGenerated {len(plot_files)} plot files:")
        for file in sorted(plot_files):
            print(f"  • {file}")

def main_enhanced_production_pipeline(data_dir="enhanced_market_data", n_optimization_trials=50):
    """Enhanced production pipeline with comprehensive plotting"""
    print("="*80)
    print("ENHANCED PRODUCTION UNSUPERVISED TOXICITY DETECTION v3.0")
    print("With Hyperparameter Optimization, Advanced Ensemble Methods, and Comprehensive Plotting")
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
    
    # Generate comprehensive visualizations
    print("\n8. GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("-" * 40)
    
    plotter = ComprehensivePlotter()
    plotter.generate_all_plots(selected_features, ensemble_scores, individual_scores, detector, X_scaled)
    
    # Save enhanced model
    print("\n9. SAVING ENHANCED PRODUCTION MODEL")
    print("-" * 40)
    
    model_path = detector.save_enhanced_model()
    
    # Print comprehensive results summary
    print("\n" + "="*80)
    print("ENHANCED PRODUCTION MODEL TRAINING COMPLETED")
    print("="*80)
    
    print(f"\n📊 MODEL PERFORMANCE SUMMARY:")
    anomaly_detection = metrics.get('anomaly_detection', {})
    ensemble_stats = metrics.get('ensemble_statistics', {})
    diversity_stats = metrics.get('ensemble_diversity', {})
    
    print(f"  • Dataset Size: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    print(f"  • Ensemble Composition: {len(detector.models)} detectors")
    print(f"  • Effective Contributors: {ensemble_stats.get('effective_detectors', 0)} detectors (>5% weight)")
    
    print(f"\n🎯 ANOMALY DETECTION PERFORMANCE:")
    for threshold in [90, 95, 97, 99]:
        rate = anomaly_detection.get(f'anomaly_rate_{threshold}th', 0)
        separation = anomaly_detection.get(f'separation_{threshold}th', 0)
        print(f"  • {threshold}th percentile: {rate*100:.2f}% detection rate, {separation:.2f} separation score")
    
    print(f"\n🔍 ENSEMBLE QUALITY:")
    print(f"  • Diversity Score: {diversity_stats.get('diversity_score', 0):.3f}/1.0")
    print(f"  • Average Correlation: {diversity_stats.get('avg_correlation', 0):.3f}")
    print(f"  • Weight Concentration: {ensemble_stats.get('weight_concentration', 0):.3f}")
    print(f"  • Score Range: {ensemble_stats.get('score_range', 0):.4f}")
    
    print(f"\n🏆 TOP PERFORMING DETECTORS:")
    top_detectors = sorted(detector.ensemble_weights.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (name, weight) in enumerate(top_detectors, 1):
        print(f"  {i}. {name.replace('_', ' ').title()}: {weight:.3f}")
    
    print(f"\n🔧 OPTIMIZED HYPERPARAMETERS:")
    for param, value in best_params.items():
        print(f"  • {param.replace('_', ' ').title()}: {value}")
    
    print(f"\n📁 OUTPUTS:")
    print(f"  • Model: {model_path}")
    print(f"  • Plots: {plotter.save_dir}")
    print(f"  • Timestamp: {plotter.timestamp}")
    
    # Feature importance summary
    if hasattr(detector, 'feature_importance') and detector.feature_importance:
        print(f"\n⭐ TOP FEATURES FOR TOXICITY DETECTION:")
        top_features = sorted(detector.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"  {i}. {feature.replace('_', ' ').title()}: {importance:.4f}")
    
    return detector, ensemble_scores, metrics, plotter

class EnhancedProductionInference:
    """Enhanced inference class with comprehensive monitoring"""
    
    def __init__(self, model_path):
        self.model_package = joblib.load(model_path)
        self.models = self.model_package['models']
        self.scalers = self.model_package['scalers']
        self.feature_selector = self.model_package['feature_selector']
        self.ensemble_weights = self.model_package['ensemble_weights']
        self.feature_importance = self.model_package.get('feature_importance', {})
        
        print(f"🚀 Loaded enhanced model v{self.model_package.get('version', 'unknown')}")
        print(f"📅 Training timestamp: {self.model_package['timestamp']}")
        print(f"🔧 Features: {self.model_package['n_features']}")
        print(f"🎯 Detectors: {self.model_package['n_detectors']}")
        
        # Performance tracking
        self.prediction_count = 0
        self.prediction_times = []
        self.prediction_history = []
        
    def predict_toxicity_score(self, features_dict, return_breakdown=False):
        """Enhanced prediction with optional detailed breakdown"""
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
            
            # Calculate ensemble score with breakdown
            individual_contributions = {}
            ensemble_score = 0.0
            successful_detectors = 0
            
            for name, model in self.models.items():
                try:
                    weight = self.ensemble_weights.get(name, 0)
                    if weight == 0:
                        continue
                    
                    # Calculate detector-specific score
                    if 'isolation_forest' in name:
                        score = -model.decision_function(X_scaled)[0]
                    elif 'lof' in name:
                        score = -model.score_samples(X_scaled)[0]
                    elif 'gmm' in name:
                        score = -model.score_samples(X_scaled)[0]
                    elif 'kmeans' in name:
                        distance = np.min(model['kmeans'].transform(X_scaled), axis=1)[0]
                        score = distance
                    elif 'agglomerative' in name:
                        distances = np.min(cdist(X_scaled, model['cluster_centres']), axis=1)
                        score = distances[0]
                    else:
                        score = 0.5  # Default
                    
                    # Normalize and contribute to ensemble
                    score = max(0, min(1, score))
                    contribution = weight * score
                    individual_contributions[name] = {
                        'raw_score': score,
                        'weight': weight,
                        'contribution': contribution
                    }
                    
                    ensemble_score += contribution
                    successful_detectors += 1
                    
                except Exception:
                    continue
            
            # Track performance
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            self.prediction_count += 1
            
            # Store prediction history
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'score': ensemble_score,
                'features': dict(features_dict),
                'prediction_time': prediction_time
            })
            
            if return_breakdown:
                return ensemble_score, individual_contributions
            else:
                return ensemble_score
            
        except Exception as e:
            print(f"Error in toxicity prediction: {e}")
            return 0.0 if not return_breakdown else (0.0, {})
    
    def classify_toxicity(self, features_dict, threshold=0.7, confidence_levels=True):
        """Enhanced classification with confidence levels"""
        score, breakdown = self.predict_toxicity_score(features_dict, return_breakdown=True)
        
        # Determine classification
        is_toxic = score > threshold
        
        # Calculate confidence metrics
        confidence_metrics = {}
        if confidence_levels:
            # Agreement among detectors
            detector_agreements = [1 if contrib['raw_score'] > threshold else 0 
                                    for contrib in breakdown.values()]
            agreement_rate = np.mean(detector_agreements) if detector_agreements else 0
            
            # Score distance from threshold
            distance_confidence = abs(score - threshold) / threshold
            
            # Weighted contribution consistency
            contributions = [contrib['contribution'] for contrib in breakdown.values()]
            contribution_std = np.std(contributions) if len(contributions) > 1 else 0
            consistency = 1 / (1 + contribution_std)
            
            confidence_metrics = {
                'detector_agreement': agreement_rate,
                'threshold_distance': distance_confidence,
                'contribution_consistency': consistency,
                'overall_confidence': np.mean([agreement_rate, distance_confidence, consistency])
            }
        
        return {
            'is_toxic': is_toxic,
            'toxicity_score': score,
            'threshold': threshold,
            'confidence_metrics': confidence_metrics,
            'detector_breakdown': breakdown
        }
    
    def get_performance_stats(self):
        """Get comprehensive inference performance statistics"""
        if not self.prediction_times:
            return {}
        
        recent_scores = [pred['score'] for pred in self.prediction_history[-100:]]  # Last 100 predictions
        
        return {
            'prediction_performance': {
                'total_predictions': self.prediction_count,
                'avg_prediction_time': np.mean(self.prediction_times),
                'median_prediction_time': np.median(self.prediction_times),
                'max_prediction_time': np.max(self.prediction_times),
                'min_prediction_time': np.min(self.prediction_times)
            },
            'score_statistics': {
                'recent_score_mean': np.mean(recent_scores) if recent_scores else 0,
                'recent_score_std': np.std(recent_scores) if recent_scores else 0,
                'recent_score_range': [np.min(recent_scores), np.max(recent_scores)] if recent_scores else [0, 0]
            },
            'model_info': {
                'version': self.model_package.get('version'),
                'training_timestamp': self.model_package['timestamp'],
                'n_features': self.model_package['n_features'],
                'n_detectors': self.model_package['n_detectors']
            }
        }

if __name__ == "__main__":
    try:
        # Run enhanced pipeline with comprehensive plotting
        detector, ensemble_scores, metrics, plotter = main_enhanced_production_pipeline(
            data_dir="enhanced_market_data",
            n_optimization_trials=50
        )
        
        print("\n" + "="*80)
        print("🎉 SUCCESS: Enhanced production model with comprehensive plotting completed!")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error in enhanced production pipeline: {e}")
        import traceback
        traceback.print_exc()