"""
Enhanced Market Toxicity Detection System - Optimized for Real Market Data
Uses only publicly observable market data for unsupervised toxicity detection
Designed specifically for market maker spread optimization
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
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
from scipy.spatial.distance import cdist
import joblib
import glob
import os
import optuna
from optuna.samplers import TPESampler

class MarketDataFeatureEngineer:
    """Extract features from your specific market data format"""
    
    def __init__(self):
        self.feature_names = []
        self.feature_importance_scores = {}
    
    def load_market_data(self, data_dir="enhanced_market_data"):
        """Load market data from your specific file structure"""
        print(f"Loading market data from {data_dir}...")
        
        # Find all data files with your naming convention
        order_files = glob.glob(f"{data_dir}/orders_*.csv")
        lob_files = glob.glob(f"{data_dir}/lob_snapshots_*.csv")
        trade_files = glob.glob(f"{data_dir}/trades_*.csv")
        
        if not order_files:
            raise FileNotFoundError(f"No order files found in {data_dir}. Expected files like 'orders_{{timestamp}}.csv'")
        
        # Use the most recent files based on creation time
        latest_order_file = max(order_files, key=os.path.getctime)
        orders_df = pd.read_csv(latest_order_file)
        print(f"Orders data: {len(orders_df)} records from {os.path.basename(latest_order_file)}")
        
        # Load LOB data
        lob_df = pd.DataFrame()
        if lob_files:
            latest_lob_file = max(lob_files, key=os.path.getctime)
            lob_df = pd.read_csv(latest_lob_file)
            print(f"LOB data: {len(lob_df)} snapshots from {os.path.basename(latest_lob_file)}")
        else:
            print("No LOB snapshot files found")
        
        # Load trades data
        trades_df = pd.DataFrame()
        if trade_files:
            latest_trade_file = max(trade_files, key=os.path.getctime)
            trades_df = pd.read_csv(latest_trade_file)
            print(f"Trades data: {len(trades_df)} trades from {os.path.basename(latest_trade_file)}")
        else:
            print("No trade files found")
        
        return orders_df, lob_df, trades_df
    
    def _create_sample_data(self):
        """This method is removed - we only use your actual data"""
        pass
    
    def extract_public_features(self, orders_df, lob_df, trades_df):
        """Extract comprehensive features from publicly observable market data"""
        print("Extracting public market features...")
        
        features_df = pd.DataFrame(index=orders_df.index)
        
        # Basic order characteristics (always public)
        features_df['order_size'] = orders_df['quantity']
        features_df['log_order_size'] = np.log1p(orders_df['quantity'])
        features_df['sqrt_order_size'] = np.sqrt(orders_df['quantity'])
        features_df['order_size_zscore'] = (orders_df['quantity'] - orders_df['quantity'].mean()) / orders_df['quantity'].std()
        features_df['order_size_percentile'] = orders_df['quantity'].rank(pct=True)
        
        # Order type features
        features_df['is_market_order'] = (orders_df['order_type'] == 'MARKET').astype(int)
        features_df['is_limit_order'] = (orders_df['order_type'] == 'LIMIT').astype(int)
        features_df['is_buy'] = (orders_df['side'] == 'BUY').astype(int)
        features_df['is_sell'] = (orders_df['side'] == 'SELL').astype(int)
        
        # Price-related features (if available)
        if 'mid_price' in orders_df.columns:
            mid_price = orders_df['mid_price']
            features_df['mid_price'] = mid_price
            features_df['log_mid_price'] = np.log(mid_price)
            features_df['mid_price_change'] = mid_price.diff().fillna(0)
            features_df['mid_price_returns'] = mid_price.pct_change().fillna(0)
            
            # Price momentum and trend
            for window in [5, 10, 20]:
                features_df[f'price_ma_{window}'] = mid_price.rolling(window, min_periods=1).mean()
                features_df[f'price_trend_{window}'] = (mid_price - features_df[f'price_ma_{window}']) / features_df[f'price_ma_{window}']
                features_df[f'price_momentum_{window}'] = mid_price.rolling(window, min_periods=1).apply(
                    lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) > 1 and x.iloc[0] != 0 else 0)
            
            # Price relative to order
            if 'price' in orders_df.columns:
                order_price = orders_df['price'].fillna(mid_price)
                features_df['price_deviation'] = (order_price - mid_price) / mid_price
                features_df['abs_price_deviation'] = np.abs(features_df['price_deviation'])
                features_df['price_aggressiveness'] = np.where(
                    orders_df['side'] == 'BUY',
                    np.maximum(0, (order_price - mid_price) / mid_price),
                    np.maximum(0, (mid_price - order_price) / mid_price)
                )
        
        # Spread features (if available)
        if 'spread' in orders_df.columns:
            spread = orders_df['spread']
            features_df['spread'] = spread
            features_df['log_spread'] = np.log1p(spread)
            if 'mid_price' in orders_df.columns:
                features_df['relative_spread'] = spread / mid_price
            
            # Rolling spread statistics
            for window in [5, 10, 20]:
                features_df[f'spread_ma_{window}'] = spread.rolling(window, min_periods=1).mean()
                features_df[f'spread_std_{window}'] = spread.rolling(window, min_periods=1).std()
                # Fix: Use a custom rolling percentile function
                def rolling_rank_pct(series, window):
                    def rank_pct(x):
                        if len(x) <= 1:
                            return 0.5
                        return (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1)
                    return series.rolling(window, min_periods=1).apply(rank_pct, raw=False)
                
                features_df[f'spread_percentile_{window}'] = rolling_rank_pct(spread, window)
        
        # Timing features
        if 'timestamp' in orders_df.columns:
            timestamps = pd.to_datetime(orders_df['timestamp']) if not pd.api.types.is_datetime64_any_dtype(orders_df['timestamp']) else orders_df['timestamp']
            
            # Time-based patterns
            if hasattr(timestamps.iloc[0], 'hour'):
                features_df['hour'] = timestamps.dt.hour
                features_df['minute'] = timestamps.dt.minute
                features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
                features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
                features_df['minute_sin'] = np.sin(2 * np.pi * features_df['minute'] / 60)
                features_df['minute_cos'] = np.cos(2 * np.pi * features_df['minute'] / 60)
            
            # Inter-arrival times
            time_diffs = timestamps.diff().dt.total_seconds().fillna(1) if hasattr(timestamps.iloc[0], 'hour') else pd.Series(range(len(timestamps))).diff().fillna(1)
            features_df['inter_arrival_time'] = time_diffs
            features_df['log_inter_arrival_time'] = np.log1p(time_diffs)
            features_df['arrival_rate'] = 1 / (time_diffs + 1e-8)
            
            # Arrival intensity over different windows
            for window in [5, 10, 20, 50]:
                features_df[f'arrival_intensity_{window}'] = features_df['arrival_rate'].rolling(window, min_periods=1).mean()
                features_df[f'arrival_volatility_{window}'] = features_df['arrival_rate'].rolling(window, min_periods=1).std()
        
        # Market microstructure features (if available)
        if 'volatility' in orders_df.columns:
            vol = orders_df['volatility']
            features_df['volatility'] = vol
            features_df['log_volatility'] = np.log1p(vol)
            features_df['vol_percentile'] = vol.rank(pct=True)
            features_df['vol_regime_high'] = (vol > vol.quantile(0.75)).astype(int)
            features_df['vol_regime_low'] = (vol <= vol.quantile(0.25)).astype(int)
            
            # Rolling volatility features
            for window in [5, 10, 20]:
                features_df[f'vol_ma_{window}'] = vol.rolling(window, min_periods=1).mean()
                features_df[f'vol_std_{window}'] = vol.rolling(window, min_periods=1).std()
        
        if 'momentum' in orders_df.columns:
            mom = orders_df['momentum']
            features_df['momentum'] = mom
            features_df['abs_momentum'] = np.abs(mom)
            features_df['momentum_sign'] = np.sign(mom)
            features_df['momentum_squared'] = mom ** 2
            
            # Rolling momentum features
            for window in [5, 10, 20]:
                features_df[f'momentum_ma_{window}'] = mom.rolling(window, min_periods=1).mean()
                features_df[f'momentum_std_{window}'] = mom.rolling(window, min_periods=1).std()
        
        if 'order_book_imbalance' in orders_df.columns:
            imbalance = orders_df['order_book_imbalance']
            features_df['imbalance'] = imbalance
            features_df['abs_imbalance'] = np.abs(imbalance)
            features_df['imbalance_sign'] = np.sign(imbalance)
            features_df['imbalance_percentile'] = imbalance.rank(pct=True)
            
            # Rolling imbalance features
            for window in [5, 10, 20]:
                features_df[f'imbalance_ma_{window}'] = imbalance.rolling(window, min_periods=1).mean()
                features_df[f'imbalance_std_{window}'] = imbalance.rolling(window, min_periods=1).std()
        
        # Aggressiveness features
        if 'is_aggressive' in orders_df.columns:
            features_df['is_aggressive'] = orders_df['is_aggressive'].astype(int)
        elif 'distance_from_mid' in orders_df.columns:
            features_df['distance_from_mid'] = orders_df['distance_from_mid']
            features_df['abs_distance_from_mid'] = np.abs(orders_df['distance_from_mid'])
            features_df['is_aggressive'] = (np.abs(orders_df['distance_from_mid']) < 0.001).astype(int)
        
        # Time since last trade
        if 'time_since_last_trade' in orders_df.columns:
            time_since = orders_df['time_since_last_trade']
            features_df['time_since_last_trade'] = time_since
            features_df['log_time_since_trade'] = np.log1p(time_since)
            features_df['trade_urgency'] = 1 / (time_since + 1e-8)
        
        # LOB features (if available)
        if not lob_df.empty:
            lob_features = self._extract_lob_features(lob_df, orders_df)
            for col in lob_features.columns:
                if col not in features_df.columns:
                    features_df[col] = lob_features[col]
        
        # Trade features (if available)
        if not trades_df.empty:
            trade_features = self._extract_trade_features(trades_df, orders_df)
            for col in trade_features.columns:
                if col not in features_df.columns:
                    features_df[col] = trade_features[col]
        
        # Market regime features
        features_df = self._add_market_regime_features(features_df)
        
        # Rolling statistics for key features
        features_df = self._add_rolling_features(features_df)
        
        # Interaction features
        features_df = self._add_interaction_features(features_df)
        
        # Clean up
        features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)
        self.feature_names = features_df.columns.tolist()
        
        print(f"Generated {len(self.feature_names)} public market features")
        return features_df
    
    def _extract_lob_features(self, lob_df, orders_df):
        """Extract features from order book snapshots"""
        lob_features = pd.DataFrame(index=orders_df.index)
        
        if 'timestamp' in lob_df.columns and 'timestamp' in orders_df.columns:
            # Merge LOB data with orders based on timestamp
            merged = pd.merge_asof(
                orders_df[['timestamp']].reset_index(),
                lob_df,
                on='timestamp',
                direction='backward'
            ).set_index('index')
            
            # Extract bid/ask features for multiple levels
            for level in range(1, 6):  # Up to 5 levels
                bid_price_col = f'bid_price_{level}'
                ask_price_col = f'ask_price_{level}'
                bid_size_col = f'bid_size_{level}'
                ask_size_col = f'ask_size_{level}'
                
                if all(col in merged.columns for col in [bid_size_col, ask_size_col]):
                    bid_size = merged[bid_size_col].fillna(0)
                    ask_size = merged[ask_size_col].fillna(0)
                    
                    # Size features
                    lob_features[f'bid_size_L{level}'] = bid_size
                    lob_features[f'ask_size_L{level}'] = ask_size
                    lob_features[f'total_size_L{level}'] = bid_size + ask_size
                    lob_features[f'size_imbalance_L{level}'] = (bid_size - ask_size) / (bid_size + ask_size + 1e-8)
                    lob_features[f'size_ratio_L{level}'] = bid_size / (ask_size + 1e-8)
                    
                    # Price features if available
                    if all(col in merged.columns for col in [bid_price_col, ask_price_col]):
                        bid_price = merged[bid_price_col].fillna(0)
                        ask_price = merged[ask_price_col].fillna(0)
                        lob_features[f'spread_L{level}'] = ask_price - bid_price
                        if level == 1:
                            lob_features['lob_mid_price'] = (bid_price + ask_price) / 2
            
            # Aggregate features across all levels
            bid_cols = [col for col in merged.columns if 'bid_size' in col]
            ask_cols = [col for col in merged.columns if 'ask_size' in col]
            
            if bid_cols and ask_cols:
                total_bid_depth = merged[bid_cols].fillna(0).sum(axis=1)
                total_ask_depth = merged[ask_cols].fillna(0).sum(axis=1)
                
                lob_features['total_bid_depth'] = total_bid_depth
                lob_features['total_ask_depth'] = total_ask_depth
                lob_features['total_depth'] = total_bid_depth + total_ask_depth
                lob_features['depth_imbalance'] = (total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth + 1e-8)
                lob_features['depth_ratio'] = total_bid_depth / (total_ask_depth + 1e-8)
        
        return lob_features.fillna(0)
    
    def _extract_trade_features(self, trades_df, orders_df):
        """Extract features from recent trade activity"""
        trade_features = pd.DataFrame(index=orders_df.index)
        
        if 'timestamp' in trades_df.columns and 'timestamp' in orders_df.columns:
            # Handle timestamp conversion more carefully
            try:
                orders_ts = pd.to_datetime(orders_df['timestamp'])
                trades_ts = pd.to_datetime(trades_df['timestamp'])
                use_datetime = True
            except:
                # If datetime conversion fails, use numeric timestamps
                orders_ts = orders_df['timestamp']
                trades_ts = trades_df['timestamp']
                use_datetime = False
            
            # Calculate trade features for each order (sample subset for performance)
            sample_size = min(len(orders_df), 1000)
            sample_indices = np.random.choice(len(orders_df), sample_size, replace=False)
            
            for idx in sample_indices:
                order_time = orders_ts.iloc[idx]
                
                # Look at trades in different time windows
                for window_seconds in [10, 30, 60]:
                    try:
                        if use_datetime:
                            # Use pd.Timedelta for datetime objects
                            start_time = order_time - pd.Timedelta(seconds=window_seconds)
                            recent_trades = trades_df[
                                (trades_ts >= start_time) & 
                                (trades_ts <= order_time)
                            ]
                        else:
                            # Use numeric subtraction for numeric timestamps
                            start_time = order_time - window_seconds
                            recent_trades = trades_df[
                                (trades_df['timestamp'] >= start_time) & 
                                (trades_df['timestamp'] <= order_time)
                            ]
                        
                        if not recent_trades.empty and 'quantity' in recent_trades.columns:
                            trade_features.loc[idx, f'trade_count_{window_seconds}'] = len(recent_trades)
                            trade_features.loc[idx, f'trade_volume_{window_seconds}'] = recent_trades['quantity'].sum()
                            trade_features.loc[idx, f'avg_trade_size_{window_seconds}'] = recent_trades['quantity'].mean()
                            
                            if len(recent_trades) > 1:
                                trade_features.loc[idx, f'trade_size_std_{window_seconds}'] = recent_trades['quantity'].std()
                            
                            # Price-based features if available
                            if 'price' in recent_trades.columns:
                                prices = recent_trades['price']
                                if len(prices) > 1:
                                    price_returns = prices.pct_change().dropna()
                                    if len(price_returns) > 0:
                                        trade_features.loc[idx, f'trade_volatility_{window_seconds}'] = price_returns.std()
                                        trade_features.loc[idx, f'trade_momentum_{window_seconds}'] = price_returns.mean()
                                
                                # VWAP if both price and quantity available
                                if recent_trades['quantity'].sum() > 0:
                                    vwap = (prices * recent_trades['quantity']).sum() / recent_trades['quantity'].sum()
                                    trade_features.loc[idx, f'vwap_{window_seconds}'] = vwap
                    except Exception as e:
                        # Skip this window if there's an error
                        continue
        
        return trade_features.fillna(0)
    
    def _add_market_regime_features(self, features_df):
        """Add market regime identification features"""
        # Size-based regimes
        if 'order_size' in features_df.columns:
            size_quantiles = features_df['order_size'].quantile([0.8, 0.95, 0.99])
            features_df['large_order'] = (features_df['order_size'] >= size_quantiles.iloc[0]).astype(int)
            features_df['very_large_order'] = (features_df['order_size'] >= size_quantiles.iloc[1]).astype(int)
            features_df['extreme_order'] = (features_df['order_size'] >= size_quantiles.iloc[2]).astype(int)
        
        # Volatility-based regimes
        if 'volatility' in features_df.columns:
            vol_quantiles = features_df['volatility'].quantile([0.33, 0.67])
            features_df['low_vol_regime'] = (features_df['volatility'] <= vol_quantiles.iloc[0]).astype(int)
            features_df['high_vol_regime'] = (features_df['volatility'] >= vol_quantiles.iloc[1]).astype(int)
        
        # Spread-based regimes
        if 'spread' in features_df.columns:
            spread_quantiles = features_df['spread'].quantile([0.33, 0.67])
            features_df['tight_spread_regime'] = (features_df['spread'] <= spread_quantiles.iloc[0]).astype(int)
            features_df['wide_spread_regime'] = (features_df['spread'] >= spread_quantiles.iloc[1]).astype(int)
        
        return features_df
    
    def _add_rolling_features(self, features_df):
        """Add rolling statistical features for key variables"""
        key_features = ['order_size', 'spread', 'volatility', 'momentum', 'arrival_rate']
        available_features = [f for f in key_features if f in features_df.columns]
        
        for feature in available_features:
            for window in [5, 10, 20]:
                # Basic rolling statistics
                features_df[f'{feature}_ma_{window}'] = features_df[feature].rolling(window, min_periods=1).mean()
                features_df[f'{feature}_std_{window}'] = features_df[feature].rolling(window, min_periods=1).std()
                features_df[f'{feature}_min_{window}'] = features_df[feature].rolling(window, min_periods=1).min()
                features_df[f'{feature}_max_{window}'] = features_df[feature].rolling(window, min_periods=1).max()
                
                # Z-score
                ma_col = f'{feature}_ma_{window}'
                std_col = f'{feature}_std_{window}'
                features_df[f'{feature}_zscore_{window}'] = (
                    (features_df[feature] - features_df[ma_col]) / (features_df[std_col] + 1e-8)
                )
                
                # Percentile ranking
                def rolling_rank_pct(series, window):
                    def rank_pct(x):
                        if len(x) <= 1:
                            return 0.5
                        return (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1)
                    return series.rolling(window, min_periods=1).apply(rank_pct, raw=False)
                
                features_df[f'{feature}_rank_{window}'] = rolling_rank_pct(features_df[feature], window)
        
        return features_df
    
    def _add_interaction_features(self, features_df):
        """Add interaction features between key variables"""
        # Size-based interactions
        if 'order_size' in features_df.columns:
            if 'spread' in features_df.columns:
                features_df['size_spread_interaction'] = features_df['order_size'] * features_df['spread']
            if 'volatility' in features_df.columns:
                features_df['size_vol_interaction'] = features_df['order_size'] * features_df['volatility']
            if 'is_aggressive' in features_df.columns:
                features_df['size_aggression_interaction'] = features_df['order_size'] * features_df['is_aggressive']
        
        # Volatility-based interactions
        if 'volatility' in features_df.columns:
            if 'is_aggressive' in features_df.columns:
                features_df['vol_aggression_interaction'] = features_df['volatility'] * features_df['is_aggressive']
            if 'arrival_rate' in features_df.columns:
                features_df['vol_arrival_interaction'] = features_df['volatility'] * features_df['arrival_rate']
        
        # Time-based interactions
        if 'arrival_rate' in features_df.columns and 'order_size' in features_df.columns:
            features_df['arrival_size_interaction'] = features_df['arrival_rate'] * features_df['order_size']
        
        return features_df

class OptimizedToxicityDetector:
    """Optimized ensemble detector for market toxicity using only public data"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.ensemble_weights = {}
        self.performance_metrics = {}
        self.best_hyperparameters = {}
        self.feature_importance = {}
        
    def prepare_features(self, features_df, variance_threshold=0.01, correlation_threshold=0.95):
        """Prepare features with selection and scaling"""
        print("Preparing features with advanced selection...")
        
        # Remove low variance features
        var_selector = VarianceThreshold(threshold=variance_threshold)
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        high_var_features = numeric_features.loc[:, var_selector.fit(numeric_features).get_support()]
        print(f"Removed {len(numeric_features.columns) - len(high_var_features.columns)} low variance features")
        
        # Remove highly correlated features
        selected_features = self._remove_correlated_features(high_var_features, correlation_threshold)
        
        # Calculate feature importance
        self.feature_importance = self._calculate_feature_importance(selected_features)
        
        # Select best scaler
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
    
    def _remove_correlated_features(self, features_df, threshold=0.95):
        """Remove highly correlated features"""
        corr_matrix = features_df.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
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
    
    def _calculate_feature_importance(self, features_df):
        """Calculate feature importance using multiple methods"""
        print("Calculating feature importance...")
        
        feature_importance = {}
        
        # Method 1: Variance-based importance
        variances = features_df.var()
        normalized_variances = variances / variances.max()

        # Method 2: Isolation Forest feature importance
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
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
    
    def optimize_hyperparameters(self, X_scaled, n_trials=50):
        """Optimize hyperparameters using Optuna"""
        print(f"Optimizing hyperparameters with {n_trials} trials...")
        
        def objective(trial):
            contamination = trial.suggest_float('contamination', 0.01, 0.2)
            n_estimators = trial.suggest_int('n_estimators', 100, 300)
            n_neighbors = trial.suggest_int('n_neighbors', 5, 30)
            n_clusters = trial.suggest_int('n_clusters', 3, min(15, len(X_scaled) // 30))
            
            try:
                # Train multiple detectors with these parameters
                iso_forest = IsolationForest(
                    contamination=contamination,
                    n_estimators=n_estimators,
                    random_state=42
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
                
                # Evaluate clustering quality
                if len(set(cluster_labels)) > 1:
                    silhouette = silhouette_score(X_scaled, cluster_labels)
                    calinski = calinski_harabasz_score(X_scaled, cluster_labels)
                    davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
                    
                    # Combine multiple metrics
                    score = silhouette + (calinski / 1000) - davies_bouldin
                else:
                    score = -1
                
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
    
    def train_ensemble_detectors(self, X_scaled, hyperparameters=None):
        """Train ensemble of toxicity detectors"""
        print("Training ensemble detectors...")
        
        if hyperparameters is None:
            hyperparameters = self.best_hyperparameters
        
        detectors = {}
        
        # Isolation Forest variants with different contamination rates
        contamination_rates = [0.05, 0.1, 0.15]
        for i, contamination in enumerate(contamination_rates):
            try:
                iso_forest = IsolationForest(
                    contamination=contamination,
                    n_estimators=hyperparameters.get('n_estimators', 200),
                    random_state=42 + i,
                    bootstrap=True
                )
                iso_forest.fit(X_scaled)
                detectors[f'isolation_forest_{contamination}'] = iso_forest
            except Exception as e:
                print(f"Warning: Failed to train Isolation Forest with contamination {contamination}: {e}")
        
        # Local Outlier Factor variants
        neighbor_counts = [5, 10, 20]
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
        
        # Gaussian Mixture Models
        component_counts = [3, 5, 8]
        for n_components in component_counts:
            try:
                n_comp = min(n_components, max(2, len(X_scaled) // 50))
                if n_comp < 2:
                    continue
                
                gmm = GaussianMixture(
                    n_components=n_comp,
                    random_state=42,
                    covariance_type='tied'
                )
                gmm.fit(X_scaled)
                detectors[f'gmm_{n_comp}'] = gmm
            except Exception as e:
                print(f"Warning: Failed to train GMM with {n_comp} components: {e}")
        
        # K-means based detectors
        cluster_counts = [5, 8, 12]
        for n_clusters in cluster_counts:
            try:
                n_clust = min(n_clusters, max(2, len(X_scaled) // 30))
                if n_clust < 2:
                    continue
                
                kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                distances = np.min(kmeans.transform(X_scaled), axis=1)
                
                detectors[f'kmeans_{n_clust}'] = {
                    'kmeans': kmeans,
                    'distance_threshold': np.percentile(distances, 95),
                    'cluster_sizes': np.bincount(cluster_labels)
                }
            except Exception as e:
                print(f"Warning: Failed to train K-means with {n_clust} clusters: {e}")
        
        # DBSCAN detector
        dbscan_detector = self._train_dbscan_detector(X_scaled)
        if dbscan_detector:
            detectors['dbscan'] = dbscan_detector
        
        self.models = detectors
        print(f"Trained {len(detectors)} detectors")
        
        return detectors
    
    def _train_dbscan_detector(self, X_scaled):
        """Train DBSCAN detector with parameter search"""
        best_score = -1
        best_params = None
        
        eps_values = np.logspace(-1.5, 0.5, 10)
        min_samples_values = [5, 10, 15]
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(X_scaled)
                    
                    n_outliers = np.sum(labels == -1)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    
                    if 2 <= n_clusters <= 10 and 0.01 <= n_outliers / len(X_scaled) <= 0.2:
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
    
    def calculate_ensemble_scores(self, X_scaled):
        """Calculate ensemble toxicity scores with adaptive weighting"""
        print("Calculating ensemble scores...")
        
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
                elif 'dbscan' in name:
                    labels = model['labels']
                    scores = np.zeros(len(X_scaled))
                    for i in range(min(len(X_scaled), len(labels))):
                        if labels[i] == -1:
                            scores[i] = 1.0
                        else:
                            cluster_points = X_scaled[labels == labels[i]]
                            if len(cluster_points) > 0:
                                cluster_centre = np.mean(cluster_points, axis=0)
                                scores[i] = np.linalg.norm(X_scaled[i] - cluster_centre)
                else:
                    continue
                
                # Robust normalization
                if len(scores) > 0 and scores.max() > scores.min():
                    q25, q75 = np.percentile(scores, [25, 75])
                    iqr = q75 - q25
                    if iqr > 0:
                        scores = (scores - q25) / iqr
                        scores = np.clip(scores, 0, 3)
                    else:
                        scores = (scores - scores.min()) / (scores.max() - scores.min())
                else:
                    scores = np.zeros_like(scores)
                
                individual_scores[name] = scores
                
                # Calculate weight based on score quality
                weight = self._calculate_detector_weight(scores, name)
                individual_weights[name] = weight
                
            except Exception as e:
                print(f"Error calculating scores for {name}: {e}")
                continue
        
        # Normalize weights
        total_weight = sum(individual_weights.values())
        if total_weight > 0:
            # Prevent any single detector from dominating
            max_weight = 0.4
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
        
        print("Detector weights:")
        for name, weight in sorted(individual_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {weight:.3f}")
        
        return ensemble_scores, individual_scores
    
    def _calculate_detector_weight(self, scores, detector_name):
        """Calculate detector weight based on multiple criteria"""
        try:
            # Score distribution quality
            score_std = np.std(scores)
            score_range = np.max(scores) - np.min(scores)
            separation_score = score_std * score_range
            
            # Anomaly detection consistency
            performance_scores = []
            for threshold_pct in [90, 95, 99]:
                threshold = np.percentile(scores, threshold_pct)
                anomaly_rate = np.mean(scores > threshold)
                expected_rate = (100 - threshold_pct) / 100
                
                if expected_rate > 0:
                    rate_score = 1 - abs(anomaly_rate - expected_rate) / expected_rate
                    performance_scores.append(max(0, rate_score))
            
            avg_performance = np.mean(performance_scores) if performance_scores else 0
            
            # Detector type preference
            type_bonus = 1.0
            if 'isolation_forest' in detector_name:
                type_bonus = 1.1
            elif 'lof' in detector_name:
                type_bonus = 1.05
            elif 'gmm' in detector_name:
                type_bonus = 1.0
            elif 'dbscan' in detector_name:
                type_bonus = 0.9
            
            # Score stability
            score_volatility = np.std(np.diff(scores))
            stability_score = 1 / (1 + score_volatility)
            
            # Combine all criteria
            weight = separation_score * avg_performance * type_bonus * stability_score
            
            return max(0.05, min(weight, 2.0))
            
        except Exception:
            return 0.1
    
    def evaluate_performance(self, X_scaled, ensemble_scores, individual_scores):
        """Evaluate detector performance with comprehensive metrics"""
        print("Evaluating performance...")
        
        metrics = {}
        
        # Score distribution analysis
        metrics['score_stats'] = {
            'mean': ensemble_scores.mean(),
            'std': ensemble_scores.std(),
            'min': ensemble_scores.min(),
            'max': ensemble_scores.max(),
            'skewness': stats.skew(ensemble_scores),
            'kurtosis': stats.kurtosis(ensemble_scores)
        }
        
        # Anomaly detection rates at different thresholds
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
        
        # Clustering quality for K-means detectors
        clustering_scores = {}
        for name, model in self.models.items():
            if 'kmeans' in name and isinstance(model, dict):
                try:
                    kmeans = model['kmeans']
                    cluster_labels = kmeans.predict(X_scaled)
                    if len(set(cluster_labels)) > 1:
                        sil_score = silhouette_score(X_scaled, cluster_labels)
                        clustering_scores[f'{name}_silhouette'] = sil_score
                except Exception:
                    continue
        
        metrics['clustering_quality'] = clustering_scores
        
        # Individual detector performance
        individual_performance = {}
        for name, scores in individual_scores.items():
            try:
                performance = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'range': scores.max() - scores.min(),
                    'weight': self.ensemble_weights.get(name, 0),
                    'consistency': self._calculate_consistency(scores)
                }
                individual_performance[name] = performance
            except Exception:
                continue
        
        metrics['individual_detectors'] = individual_performance
        
        # Ensemble diversity
        if len(individual_scores) > 1:
            try:
                score_matrix = np.array(list(individual_scores.values())).T
                correlation_matrix = np.corrcoef(score_matrix.T)
                correlation_matrix = np.nan_to_num(correlation_matrix)
                
                upper_triangle_indices = np.triu_indices_from(correlation_matrix, k=1)
                correlations = correlation_matrix[upper_triangle_indices]
                
                diversity_metrics = {
                    'avg_correlation': np.mean(correlations),
                    'max_correlation': np.max(correlations),
                    'diversity_score': 1 - abs(np.mean(correlations))
                }
                
                metrics['ensemble_diversity'] = diversity_metrics
            except Exception:
                pass
        
        self.performance_metrics = metrics
        return metrics
    
    def _calculate_consistency(self, scores):
        """Calculate detection consistency across thresholds"""
        consistency_scores = []
        for pct in [90, 95, 99]:
            threshold = np.percentile(scores, pct)
            detection_rate = np.mean(scores > threshold)
            expected_rate = (100 - pct) / 100
            consistency = 1 - abs(detection_rate - expected_rate) / expected_rate
            consistency_scores.append(max(0, consistency))
        
        return np.mean(consistency_scores)
    
    def save_model(self, save_dir="toxicity_models"):
        """Save the trained model with comprehensive metadata"""
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
            'best_hyperparameters': self.best_hyperparameters,
            'feature_importance': self.feature_importance,
            'timestamp': timestamp,
            'version': '4.0_market_optimized',
            'n_features': len(self.feature_selector) if self.feature_selector else 0,
            'n_detectors': len(self.models),
            'training_summary': {
                'top_features': dict(sorted(self.feature_importance.items(), 
                                            key=lambda x: x[1], reverse=True)[:10]),
                'detector_weights': dict(sorted(self.ensemble_weights.items(), 
                                                key=lambda x: x[1], reverse=True)),
                'best_clustering_score': max([v for k, v in self.performance_metrics.items() 
                                            if 'silhouette' in k and isinstance(v, (int, float))], 
                                            default=0)
            }
        }
        
        joblib.dump(model_package, model_path)
        
        print(f"Model saved to: {model_path}")
        return model_path

class ToxicityInference:
    """Production inference class for real-time toxicity detection"""
    
    def __init__(self, model_path):
        self.model_package = joblib.load(model_path)
        self.models = self.model_package['models']
        self.scalers = self.model_package['scalers']
        self.feature_selector = self.model_package['feature_selector']
        self.ensemble_weights = self.model_package['ensemble_weights']
        self.feature_importance = self.model_package.get('feature_importance', {})
        
        print(f"Loaded model v{self.model_package.get('version', 'unknown')}")
        print(f"Training timestamp: {self.model_package['timestamp']}")
        print(f"Features: {len(self.feature_selector)}")
        print(f"Detectors: {len(self.models)}")
        
        # Performance tracking
        self.prediction_count = 0
        self.prediction_times = []
        
    def predict_toxicity_score(self, features_dict, return_breakdown=False):
        """Predict toxicity score for new order"""
        import time
        start_time = time.time()
        
        try:
            # Prepare features
            features_df = pd.DataFrame([features_dict])
            
            # Handle missing features
            for feature in self.feature_selector:
                if feature not in features_df.columns:
                    features_df[feature] = 0
            
            # Select and order features
            features_df = features_df[self.feature_selector]
            
            # Scale features
            X_scaled = self.scalers['main'].transform(features_df)
            
            # Calculate ensemble score
            individual_contributions = {}
            ensemble_score = 0.0
            
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
                    elif 'dbscan' in name:
                        score = 0.5  # Default for DBSCAN in inference
                    else:
                        score = 0.0
                    
                    # Normalize and contribute to ensemble
                    score = max(0, min(1, score))
                    contribution = weight * score
                    individual_contributions[name] = {
                        'raw_score': score,
                        'weight': weight,
                        'contribution': contribution
                    }
                    
                    ensemble_score += contribution
                    
                except Exception:
                    continue
            
            # Track performance
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            self.prediction_count += 1
            
            if return_breakdown:
                return ensemble_score, individual_contributions
            else:
                return ensemble_score
            
        except Exception as e:
            print(f"Error in toxicity prediction: {e}")
            return 0.0 if not return_breakdown else (0.0, {})
    
    def classify_toxicity(self, features_dict, threshold=0.7):
        """Classify order as toxic or not with confidence metrics"""
        score, breakdown = self.predict_toxicity_score(features_dict, return_breakdown=True)
        
        is_toxic = score > threshold
        
        # Calculate confidence
        detector_agreements = [1 if contrib['raw_score'] > threshold else 0 
                             for contrib in breakdown.values()]
        agreement_rate = np.mean(detector_agreements) if detector_agreements else 0
        
        return {
            'is_toxic': is_toxic,
            'toxicity_score': score,
            'threshold': threshold,
            'confidence': agreement_rate,
            'detector_breakdown': breakdown
        }
    
    def get_performance_stats(self):
        """Get inference performance statistics"""
        if not self.prediction_times:
            return {}
        
        return {
            'total_predictions': self.prediction_count,
            'avg_prediction_time': np.mean(self.prediction_times),
            'median_prediction_time': np.median(self.prediction_times),
            'model_version': self.model_package.get('version'),
            'training_timestamp': self.model_package['timestamp']
        }

def create_performance_plots(features_df, ensemble_scores, individual_scores, detector):
    """Create comprehensive performance visualization plots"""
    
    # Create plots directory
    plots_dir = "toxicity_plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set publication-quality style
    plt.style.use('ggplot')
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    # 1. Comprehensive Analysis Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Score distribution
    axes[0, 0].hist(ensemble_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.percentile(ensemble_scores, 95), color='red', linestyle='--', label='95th Percentile')
    axes[0, 0].axvline(np.percentile(ensemble_scores, 99), color='darkred', linestyle='--', label='99th Percentile')
    axes[0, 0].set_xlabel('Toxicity Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Ensemble Toxicity Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Detector weights
    weights = list(detector.ensemble_weights.values())
    detector_names = list(detector.ensemble_weights.keys())
    colors = plt.cm.Set3(np.arange(len(weights)))
    
    bars = axes[0, 1].bar(range(len(weights)), weights, color=colors, alpha=0.8)
    axes[0, 1].set_xticks(range(len(detector_names)))
    axes[0, 1].set_xticklabels([name[:15] for name in detector_names], rotation=45, ha='right')
    axes[0, 1].set_ylabel('Ensemble Weight')
    axes[0, 1].set_title('Detector Weights in Ensemble')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, weight in zip(bars, weights):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                        f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Score timeline
    axes[0, 2].plot(ensemble_scores, alpha=0.7, color='blue', linewidth=1)
    axes[0, 2].axhline(np.percentile(ensemble_scores, 95), color='orange', linestyle='--', alpha=0.8)
    axes[0, 2].axhline(np.percentile(ensemble_scores, 99), color='red', linestyle='--', alpha=0.8)
    axes[0, 2].set_xlabel('Order Sequence')
    axes[0, 2].set_ylabel('Toxicity Score')
    axes[0, 2].set_title('Toxicity Scores Over Time')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Feature importance
    if hasattr(detector, 'feature_importance') and detector.feature_importance:
        top_features = dict(sorted(detector.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
        feature_names = list(top_features.keys())
        importance_scores = list(top_features.values())
        
        axes[1, 0].barh(range(len(feature_names)), importance_scores, color='lightgreen', alpha=0.8)
        axes[1, 0].set_yticks(range(len(feature_names)))
        axes[1, 0].set_yticklabels([name[:20] for name in feature_names])
        axes[1, 0].set_xlabel('Importance Score')
        axes[1, 0].set_title('Top 10 Feature Importance')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
    else:
        # Show score statistics instead
        stats_data = detector.performance_metrics.get('score_stats', {})
        if stats_data:
            metrics = ['mean', 'std', 'skewness', 'kurtosis']
            values = [stats_data.get(metric, 0) for metric in metrics]
            axes[1, 0].bar(metrics, values, color='lightcoral', alpha=0.8)
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].set_title('Score Statistics')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Detection rates by threshold
    # Detection rates by threshold
    thresholds = [85, 90, 95, 97, 99, 99.5]
    rates = []
    for threshold in thresholds:
        rate = np.mean(ensemble_scores > np.percentile(ensemble_scores, threshold)) * 100
        rates.append(rate)
    
    bars = axes[1, 1].bar(range(len(thresholds)), rates, 
                          color=['lightblue', 'lightgreen', 'orange', 'coral', 'red', 'darkred'], alpha=0.8)
    axes[1, 1].set_xticks(range(len(thresholds)))
    axes[1, 1].set_xticklabels([f'{t}th' for t in thresholds])
    axes[1, 1].set_ylabel('Detection Rate (%)')
    axes[1, 1].set_title('Detection Rates by Threshold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for bar, rate in zip(bars, rates):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Score correlations heatmap
    if len(individual_scores) > 1:
        # Select top detectors by weight for visualization
        top_detectors = sorted(detector.ensemble_weights.items(), key=lambda x: x[1], reverse=True)[:8]
        top_detector_names = [name for name, _ in top_detectors]
        
        score_data = {name: individual_scores[name] for name in top_detector_names 
                      if name in individual_scores}
        
        if len(score_data) > 1:
            score_df = pd.DataFrame(score_data)
            correlation_matrix = score_df.corr()
            
            im = axes[1, 2].imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            # Add correlation values
            for i in range(len(correlation_matrix)):
                for j in range(len(correlation_matrix)):
                    if abs(correlation_matrix.iloc[i, j]) > 0.3:
                        axes[1, 2].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                                        ha='center', va='center', fontsize=8)
            
            axes[1, 2].set_xticks(range(len(top_detector_names)))
            axes[1, 2].set_xticklabels([name[:10] for name in top_detector_names], rotation=45, ha='right')
            axes[1, 2].set_yticks(range(len(top_detector_names)))
            axes[1, 2].set_yticklabels([name[:10] for name in top_detector_names])
            axes[1, 2].set_title('Detector Score Correlations')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[1, 2], shrink=0.8)
            cbar.set_label('Correlation Coefficient')
    
    plt.suptitle(f'Toxicity Detection Model Analysis - {timestamp}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/comprehensive_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Anomaly Characteristics Plot
    if len(features_df.columns) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Define anomaly threshold
        anomaly_threshold = np.percentile(ensemble_scores, 95)
        anomaly_mask = ensemble_scores > anomaly_threshold
        
        key_features = ['order_size', 'volatility', 'spread', 'momentum', 'inter_arrival_time', 'arrival_rate']
        available_features = [f for f in key_features if f in features_df.columns][:6]
        
        for i, feature in enumerate(available_features):
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
                    ax.set_title(f'{feature.replace("_", " ").title()}: Normal vs Anomalous')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Add mean lines
                    if len(normal_data) > 0:
                        ax.axvline(normal_data.mean(), color='blue', linestyle=':', alpha=0.8)
                    if len(anomaly_data) > 0:
                        ax.axvline(anomaly_data.mean(), color='red', linestyle=':', alpha=0.8)
        
        # Remove empty subplots
        for i in range(len(available_features), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Feature Distributions: Normal vs Anomalous Orders', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/anomaly_characteristics_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Performance plots saved to: {plots_dir}")
    
    # List generated files
    plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png') and timestamp in f]
    print(f"Generated {len(plot_files)} plot files:")
    for file in sorted(plot_files):
        print(f"  {file}")

def main_training_pipeline(data_dir="enhanced_market_data", n_trials=30):
    """Main training pipeline for toxicity detection using your actual data"""
    print("="*80)
    print("MARKET TOXICITY DETECTION MODEL TRAINING")
    print("Using Your Actual Market Simulation Data")
    print("="*80)
    
    # Step 1: Load and engineer features from your data
    print("\n1. LOADING YOUR MARKET DATA")
    print("-" * 40)
    
    feature_engineer = MarketDataFeatureEngineer()
    orders_df, lob_df, trades_df = feature_engineer.load_market_data(data_dir)
    
    print(f"\nData Summary:")
    print(f"  Orders: {len(orders_df)} records")
    print(f"  LOB Snapshots: {len(lob_df)} records")
    print(f"  Trades: {len(trades_df)} records")
    
    # Step 2: Extract features from your data
    print("\n2. EXTRACTING PUBLIC FEATURES FROM YOUR DATA")
    print("-" * 40)
    
    features_df = feature_engineer.extract_public_features(orders_df, lob_df, trades_df)
    
    # Step 3: Prepare features
    print("\n3. FEATURE PREPARATION AND SELECTION")
    print("-" * 40)
    
    detector = OptimizedToxicityDetector()
    X_scaled, selected_features = detector.prepare_features(features_df)
    
    print(f"Final feature set: {X_scaled.shape}")
    
    # Step 4: Optimize hyperparameters
    print("\n4. HYPERPARAMETER OPTIMIZATION")
    print("-" * 40)
    
    best_params = detector.optimize_hyperparameters(X_scaled, n_trials=n_trials)
    
    # Step 5: Train ensemble detectors
    print("\n5. TRAINING ENSEMBLE DETECTORS")
    print("-" * 40)
    
    detectors = detector.train_ensemble_detectors(X_scaled, best_params)
    
    # Step 6: Calculate ensemble scores
    print("\n6. CALCULATING ENSEMBLE SCORES")
    print("-" * 40)
    
    ensemble_scores, individual_scores = detector.calculate_ensemble_scores(X_scaled)
    
    # Step 7: Evaluate performance
    print("\n7. PERFORMANCE EVALUATION")
    print("-" * 40)
    
    metrics = detector.evaluate_performance(X_scaled, ensemble_scores, individual_scores)
    
    # Step 8: Generate visualizations
    print("\n8. GENERATING VISUALIZATIONS")
    print("-" * 40)
    
    create_performance_plots(selected_features, ensemble_scores, individual_scores, detector)
    
    # Step 9: Save model
    print("\n9. SAVING MODEL")
    print("-" * 40)
    
    model_path = detector.save_model()
    
    # Print comprehensive results summary
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
    
    print(f"\nMODEL PERFORMANCE SUMMARY:")
    print(f"  Dataset Size: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    print(f"  Ensemble Composition: {len(detector.models)} detectors")
    
    # Print anomaly detection rates
    anomaly_rates = metrics.get('anomaly_rates', {})
    print(f"\nANOMALY DETECTION RATES:")
    for threshold_key, rate in anomaly_rates.items():
        if 'separation' not in threshold_key:
            print(f"  {threshold_key}: {rate*100:.2f}%")
    
    # Print separation scores
    print(f"\nSEPARATION SCORES:")
    for threshold_key, score in anomaly_rates.items():
        if 'separation' in threshold_key:
            print(f"  {threshold_key}: {score:.3f}")
    
    # Print top detectors
    print(f"\nTOP PERFORMING DETECTORS:")
    top_detectors = sorted(detector.ensemble_weights.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (name, weight) in enumerate(top_detectors, 1):
        print(f"  {i}. {name.replace('_', ' ').title()}: {weight:.3f}")
    
    # Print feature importance
    if hasattr(detector, 'feature_importance') and detector.feature_importance:
        print(f"\nTOP FEATURES FOR TOXICITY DETECTION:")
        top_features = sorted(detector.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"  {i}. {feature.replace('_', ' ').title()}: {importance:.4f}")
    
    # Print clustering quality
    clustering_quality = metrics.get('clustering_quality', {})
    if clustering_quality:
        print(f"\nCLUSTERING QUALITY:")
        for detector_name, score in clustering_quality.items():
            print(f"  {detector_name}: {score:.3f}")
    
    # Print ensemble diversity
    ensemble_diversity = metrics.get('ensemble_diversity', {})
    if ensemble_diversity:
        print(f"\nENSEMBLE DIVERSITY:")
        print(f"  Average Correlation: {ensemble_diversity.get('avg_correlation', 0):.3f}")
        print(f"  Diversity Score: {ensemble_diversity.get('diversity_score', 0):.3f}")
    
    print(f"\nMODEL SAVED TO: {model_path}")
    print(f"\nREADY FOR DEPLOYMENT IN MARKET MAKER SPREAD OPTIMIZATION")
    
    return detector, ensemble_scores, metrics

# Example usage and testing
if __name__ == "__main__":
    try:
        # Run the training pipeline using your actual data
        print("Starting toxicity detection model training with your market simulation data...")
        
        # Check if data directory exists
        data_dir = "enhanced_market_data"
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory '{data_dir}' not found. Please ensure your data files are in this directory.")
        
        # List available data files
        order_files = glob.glob(f"{data_dir}/orders_*.csv")
        lob_files = glob.glob(f"{data_dir}/lob_snapshots_*.csv")
        trade_files = glob.glob(f"{data_dir}/trades_*.csv")
        
        print(f"\nFound data files:")
        print(f"  Order files: {len(order_files)}")
        for f in order_files:
            print(f"    - {os.path.basename(f)}")
        print(f"  LOB files: {len(lob_files)}")
        for f in lob_files:
            print(f"    - {os.path.basename(f)}")
        print(f"  Trade files: {len(trade_files)}")
        for f in trade_files:
            print(f"    - {os.path.basename(f)}")
        
        if not order_files:
            raise FileNotFoundError("No order files found. Expected files like 'orders_20241201_120000.csv'")
        
        # Train the model using your data
        detector, scores, metrics = main_training_pipeline(
            data_dir=data_dir,
            n_trials=30  # Adjust number of optimization trials as needed
        )
        
        print("\n" + "="*80)
        print("SUCCESS: Training completed with your market data!")
        print("="*80)
        
        
    except FileNotFoundError as e:
        print(f"Data Error: {e}")
        print("\nPlease ensure:")
        print("1. The 'enhanced_market_data' directory exists")
        print("2. Your data files are named correctly:")
        print("   - orders_{{timestamp}}.csv")
        print("   - lob_snapshots_{{timestamp}}.csv") 
        print("   - trades_{{timestamp}}.csv")
        print("3. Files contain the expected columns based on your data format")
        
    except Exception as e:
        print(f"Training Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting tips:")
        print("1. Check data file formats and column names")
        print("2. Ensure sufficient data for training")
        print("3. Verify data directory path is correct")