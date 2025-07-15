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
from sklearn.svm import OneClassSVM
from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold
from scipy import stats
from scipy.spatial.distance import cdist
import joblib
import glob
import os
import optuna
from optuna.samplers import TPESampler

class MarketDataFeatureEngineer:
    """Feature engineering for toxicity detection patterns"""
    
    def __init__(self):
        self.feature_names = []
        self.feature_importance_scores = {}
    
    def load_market_data(self, data_dir="market_data"):
        """Load market data from specific file structure"""
        print(f"Loading market data from {data_dir}...")
        
        order_files = glob.glob(f"{data_dir}/orders_*.csv")
        lob_files = glob.glob(f"{data_dir}/lob_snapshots_*.csv")
        trade_files = glob.glob(f"{data_dir}/trades_*.csv")
        
        if not order_files:
            raise FileNotFoundError(f"No order files found in {data_dir}. Expected files like 'orders_{{timestamp}}.csv'")
        
        latest_order_file = max(order_files, key=os.path.getctime)
        orders_df = pd.read_csv(latest_order_file)
        print(f"Orders data: {len(orders_df)} records from {os.path.basename(latest_order_file)}")
        
        lob_df = pd.DataFrame()
        if lob_files:
            latest_lob_file = max(lob_files, key=os.path.getctime)
            lob_df = pd.read_csv(latest_lob_file)
            print(f"LOB data: {len(lob_df)} snapshots from {os.path.basename(latest_lob_file)}")
        else:
            print("No LOB snapshot files found")
        
        trades_df = pd.DataFrame()
        if trade_files:
            latest_trade_file = max(trade_files, key=os.path.getctime)
            trades_df = pd.read_csv(latest_trade_file)
            print(f"Trades data: {len(trades_df)} trades from {os.path.basename(latest_trade_file)}")
        else:
            print("No trade files found")
        
        return orders_df, lob_df, trades_df
    
    def extract_features(self, orders_df, lob_df, trades_df):
        """Extract comprehensive features for toxicity detection"""
        print("Extracting public market features...")
        
        features_df = pd.DataFrame(index=orders_df.index)
        
        # Basic order characteristics
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
        
        # Size regime features
        size_quantiles = orders_df['quantity'].quantile([0.8, 0.9, 0.95, 0.99])
        features_df['large_order'] = (orders_df['quantity'] >= size_quantiles.iloc[0]).astype(int)
        features_df['very_large_order'] = (orders_df['quantity'] >= size_quantiles.iloc[1]).astype(int)
        features_df['extreme_order'] = (orders_df['quantity'] >= size_quantiles.iloc[2]).astype(int)
        features_df['massive_order'] = (orders_df['quantity'] >= size_quantiles.iloc[3]).astype(int)
        
        # Order size clustering features
        size_bins = pd.qcut(orders_df['quantity'], q=10, duplicates='drop')
        features_df['size_bin'] = size_bins.cat.codes
        features_df['size_concentration'] = features_df.groupby('size_bin')['order_size'].transform('count')
        
        # Price-related features
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
                features_df[f'price_volatility_{window}'] = mid_price.rolling(window, min_periods=1).std()
            
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
        
        # Spread features
        if 'spread' in orders_df.columns:
            spread = orders_df['spread']
            features_df['spread'] = spread
            features_df['log_spread'] = np.log1p(spread)
            features_df['spread_percentile'] = spread.rank(pct=True)
            if 'mid_price' in orders_df.columns:
                features_df['relative_spread'] = spread / mid_price
            
            # Rolling spread statistics
            for window in [5, 10, 20]:
                features_df[f'spread_ma_{window}'] = spread.rolling(window, min_periods=1).mean()
                features_df[f'spread_std_{window}'] = spread.rolling(window, min_periods=1).std()
                features_df[f'spread_range_{window}'] = spread.rolling(window, min_periods=1).max() - spread.rolling(window, min_periods=1).min()
        
        # Timing features
        if 'timestamp' in orders_df.columns:
            timestamps = pd.to_datetime(orders_df['timestamp']) if not pd.api.types.is_datetime64_any_dtype(orders_df['timestamp']) else orders_df['timestamp']
            
            # Inter-arrival times
            time_diffs = timestamps.diff().dt.total_seconds().fillna(1) if hasattr(timestamps.iloc[0], 'hour') else pd.Series(range(len(timestamps))).diff().fillna(1)
            features_df['inter_arrival_time'] = time_diffs
            features_df['log_inter_arrival_time'] = np.log1p(time_diffs)
            features_df['arrival_rate'] = 1 / (time_diffs + 1e-8)
            
            # Arrival intensity patterns
            for window in [5, 10, 20, 50]:
                features_df[f'arrival_intensity_{window}'] = features_df['arrival_rate'].rolling(window, min_periods=1).mean()
                features_df[f'arrival_volatility_{window}'] = features_df['arrival_rate'].rolling(window, min_periods=1).std()
                features_df[f'arrival_burst_{window}'] = (features_df['arrival_rate'] > features_df[f'arrival_intensity_{window}'] + 2 * features_df[f'arrival_volatility_{window}']).astype(int)
            
            # Time-of-day effects
            if hasattr(timestamps.iloc[0], 'hour'):
                features_df['hour'] = timestamps.dt.hour
                features_df['minute'] = timestamps.dt.minute
                features_df['is_market_open'] = ((features_df['hour'] >= 9) & (features_df['hour'] < 16)).astype(int)
                features_df['is_market_close'] = ((features_df['hour'] >= 15) & (features_df['hour'] < 16)).astype(int)
        
        # Market microstructure features
        if 'volatility' in orders_df.columns:
            vol = orders_df['volatility']
            features_df['volatility'] = vol
            features_df['log_volatility'] = np.log1p(vol)
            features_df['vol_percentile'] = vol.rank(pct=True)
            features_df['vol_zscore'] = (vol - vol.mean()) / vol.std()
            
            # Volatility regimes
            vol_quantiles = vol.quantile([0.33, 0.67, 0.9])
            features_df['low_vol_regime'] = (vol <= vol_quantiles.iloc[0]).astype(int)
            features_df['high_vol_regime'] = (vol >= vol_quantiles.iloc[1]).astype(int)
            features_df['extreme_vol_regime'] = (vol >= vol_quantiles.iloc[2]).astype(int)
            
            # Volatility clustering
            for window in [5, 10, 20]:
                features_df[f'vol_ma_{window}'] = vol.rolling(window, min_periods=1).mean()
                features_df[f'vol_ratio_{window}'] = vol / features_df[f'vol_ma_{window}']
        
        if 'momentum' in orders_df.columns:
            mom = orders_df['momentum']
            features_df['momentum'] = mom
            features_df['abs_momentum'] = np.abs(mom)
            features_df['momentum_sign'] = np.sign(mom)
            features_df['momentum_squared'] = mom ** 2
            features_df['momentum_percentile'] = mom.rank(pct=True)
        
        if 'order_book_imbalance' in orders_df.columns:
            imbalance = orders_df['order_book_imbalance']
            features_df['imbalance'] = imbalance
            features_df['abs_imbalance'] = np.abs(imbalance)
            features_df['imbalance_sign'] = np.sign(imbalance)
            features_df['imbalance_percentile'] = imbalance.rank(pct=True)
            features_df['imbalance_zscore'] = (imbalance - imbalance.mean()) / imbalance.std()
        
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
        
        # Toxic order flow features from literature
        features_df = self._add_toxic_order_patterns(features_df)
        
        # Sequential pattern features
        features_df = self._add_sequential_patterns(features_df)
        
        # Market impact features
        features_df = self._add_market_impact_features(features_df)
        
        # Rolling features for key variables
        features_df = self._add_rolling_features(features_df)
        
        # Interaction features
        features_df = self._add_interaction_features(features_df)
        
        # Clean up
        features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)
        self.feature_names = features_df.columns.tolist()
        
        print(f"Generated {len(self.feature_names)} public market features")
        return features_df
    
    def _extract_lob_features(self, lob_df, orders_df):
        """Extract LOB features"""
        lob_features = pd.DataFrame(index=orders_df.index)
        
        if 'timestamp' in lob_df.columns and 'timestamp' in orders_df.columns:
            merged = pd.merge_asof(
                orders_df[['timestamp']].reset_index(),
                lob_df,
                on='timestamp',
                direction='backward'
            ).set_index('index')
            
            total_bid_depth = pd.Series(0, index=merged.index)
            total_ask_depth = pd.Series(0, index=merged.index)
            
            for level in range(1, 6):
                bid_size_col = f'bid_size_{level}'
                ask_size_col = f'ask_size_{level}'
                bid_price_col = f'bid_price_{level}'
                ask_price_col = f'ask_price_{level}'
                
                if all(col in merged.columns for col in [bid_size_col, ask_size_col]):
                    bid_size = merged[bid_size_col].fillna(0)
                    ask_size = merged[ask_size_col].fillna(0)
                    
                    total_bid_depth += bid_size
                    total_ask_depth += ask_size
                    
                    lob_features[f'bid_depth_L{level}'] = bid_size
                    lob_features[f'ask_depth_L{level}'] = ask_size
                    lob_features[f'depth_imbalance_L{level}'] = (bid_size - ask_size) / (bid_size + ask_size + 1e-8)
                    lob_features[f'depth_ratio_L{level}'] = bid_size / (ask_size + 1e-8)
                
                # Price level features
                if bid_price_col in merged.columns and ask_price_col in merged.columns:
                    bid_price = merged[bid_price_col].fillna(0)
                    ask_price = merged[ask_price_col].fillna(0)
                    lob_features[f'spread_L{level}'] = ask_price - bid_price
            
            lob_features['total_bid_depth'] = total_bid_depth
            lob_features['total_ask_depth'] = total_ask_depth
            lob_features['total_depth'] = total_bid_depth + total_ask_depth
            lob_features['depth_imbalance'] = (total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth + 1e-8)
            lob_features['depth_ratio'] = total_bid_depth / (total_ask_depth + 1e-8)
            
            # Weighted depth features
            for level in range(1, 4):
                weight = 1 / level
                lob_features[f'weighted_bid_depth_{level}'] = weight * lob_features.get(f'bid_depth_L{level}', 0)
                lob_features[f'weighted_ask_depth_{level}'] = weight * lob_features.get(f'ask_depth_L{level}', 0)
            
            # Depth concentration
            if total_bid_depth.sum() > 0:
                lob_features['bid_depth_concentration'] = lob_features.get('bid_depth_L1', 0) / (total_bid_depth + 1e-8)
            if total_ask_depth.sum() > 0:
                lob_features['ask_depth_concentration'] = lob_features.get('ask_depth_L1', 0) / (total_ask_depth + 1e-8)
        
        return lob_features.fillna(0)
    
    def _extract_trade_features(self, trades_df, orders_df):
        """Extract trade-based features"""
        trade_features = pd.DataFrame(index=orders_df.index)
        
        if 'timestamp' in trades_df.columns and 'timestamp' in orders_df.columns:
            try:
                orders_ts = pd.to_datetime(orders_df['timestamp'])
                trades_ts = pd.to_datetime(trades_df['timestamp'])
                use_datetime = True
            except:
                orders_ts = orders_df['timestamp']
                trades_ts = trades_df['timestamp']
                use_datetime = False
            
            sample_size = min(len(orders_df), 1000)
            sample_indices = np.random.choice(len(orders_df), sample_size, replace=False)
            
            for idx in sample_indices:
                order_time = orders_ts.iloc[idx]
                
                for window_seconds in [10, 30, 60, 300]:
                    try:
                        if use_datetime:
                            start_time = order_time - pd.Timedelta(seconds=window_seconds)
                            recent_trades = trades_df[
                                (trades_ts >= start_time) & 
                                (trades_ts <= order_time)
                            ]
                        else:
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
                                trade_features.loc[idx, f'trade_intensity_{window_seconds}'] = len(recent_trades) / window_seconds
                            
                            # Buy/sell trade imbalance
                            if 'side' in recent_trades.columns:
                                buy_volume = recent_trades[recent_trades['side'] == 'BUY']['quantity'].sum()
                                sell_volume = recent_trades[recent_trades['side'] == 'SELL']['quantity'].sum()
                                total_volume = buy_volume + sell_volume
                                if total_volume > 0:
                                    trade_features.loc[idx, f'trade_imbalance_{window_seconds}'] = (buy_volume - sell_volume) / total_volume
                            
                            # Price features
                            if 'price' in recent_trades.columns:
                                prices = recent_trades['price']
                                if len(prices) > 1:
                                    price_returns = prices.pct_change().dropna()
                                    if len(price_returns) > 0:
                                        trade_features.loc[idx, f'trade_volatility_{window_seconds}'] = price_returns.std()
                                        trade_features.loc[idx, f'trade_momentum_{window_seconds}'] = price_returns.mean()
                                        trade_features.loc[idx, f'trade_skewness_{window_seconds}'] = price_returns.skew()
                                
                                # VWAP
                                if recent_trades['quantity'].sum() > 0:
                                    vwap = (prices * recent_trades['quantity']).sum() / recent_trades['quantity'].sum()
                                    trade_features.loc[idx, f'vwap_{window_seconds}'] = vwap
                    except Exception:
                        continue
        
        return trade_features.fillna(0)
    
    def _add_toxic_order_patterns(self, features_df):
        """Add patterns associated with toxic order flow from literature"""
        
        # Layering and spoofing patterns
        if 'order_size' in features_df.columns:
            # Large orders followed by cancellations (proxy)
            large_threshold = features_df['order_size'].quantile(0.9)
            features_df['potential_layering'] = (features_df['order_size'] > large_threshold).astype(int)
            
            # Order size clustering (iceberg detection)
            size_rounded = (features_df['order_size'] / 100).round() * 100
            features_df['order_size_clustering'] = features_df.groupby(size_rounded)['order_size'].transform('count')
            features_df['iceberg_indicator'] = (features_df['order_size_clustering'] > 5).astype(int)
            
            # Unusual size patterns
            size_ma_short = features_df['order_size'].rolling(5, min_periods=1).mean()
            size_ma_long = features_df['order_size'].rolling(20, min_periods=1).mean()
            features_df['size_divergence'] = (size_ma_short - size_ma_long) / size_ma_long
            features_df['size_spike'] = (features_df['order_size'] > 3 * size_ma_long).astype(int)
        
        # Momentum ignition patterns
        if 'mid_price_change' in features_df.columns and 'order_size' in features_df.columns:
            # Large orders with significant price impact
            features_df['momentum_ignition'] = (
                (features_df['order_size'] > features_df['order_size'].quantile(0.9)) &
                (abs(features_df['mid_price_change']) > features_df['mid_price_change'].std() * 2)
            ).astype(int)
            
            # Price acceleration after large orders
            price_accel = features_df['mid_price_change'].diff()
            features_df['price_acceleration'] = price_accel
            features_df['acceleration_burst'] = (
                (features_df['order_size'] > features_df['order_size'].quantile(0.8)) &
                (abs(price_accel) > price_accel.std() * 2)
            ).astype(int)
        
        # Quote stuffing patterns
        if 'arrival_rate' in features_df.columns:
            arrival_ma = features_df['arrival_rate'].rolling(10, min_periods=1).mean()
            arrival_std = features_df['arrival_rate'].rolling(10, min_periods=1).std()
            features_df['quote_stuffing'] = (
                features_df['arrival_rate'] > arrival_ma + 3 * arrival_std
            ).astype(int)
            
            # Sustained high frequency activity
            high_freq_window = features_df['arrival_rate'].rolling(20, min_periods=1).mean()
            features_df['sustained_high_freq'] = (
                high_freq_window > features_df['arrival_rate'].quantile(0.95)
            ).astype(int)
        
        # Cross-market manipulation proxies
        if 'spread' in features_df.columns and 'volatility' in features_df.columns:
            # Unusual spread-volatility relationship
            spread_vol_ratio = features_df['spread'] / (features_df['volatility'] + 1e-8)
            features_df['spread_vol_anomaly'] = (
                abs(spread_vol_ratio - spread_vol_ratio.median()) > 2 * spread_vol_ratio.std()
            ).astype(int)
        
        # Wash trading indicators
        if 'order_size' in features_df.columns and 'is_buy' in features_df.columns:
            # Repeated similar size orders alternating direction
            size_rounded = (features_df['order_size'] / 50).round() * 50
            direction_changes = features_df['is_buy'].diff().abs()
            features_df['wash_trading_pattern'] = (
                (features_df['order_size_clustering'] > 3) &
                (direction_changes.rolling(5, min_periods=1).sum() > 2)
            ).astype(int)
        
        # Pump and dump patterns
        if 'order_size' in features_df.columns and 'mid_price_change' in features_df.columns:
            # Large order sequences with directional price movement
            large_order_sequence = features_df['large_order'].rolling(5, min_periods=1).sum()
            cumulative_price_change = features_df['mid_price_change'].rolling(5, min_periods=1).sum()
            features_df['pump_dump_pattern'] = (
                (large_order_sequence >= 3) &
                (abs(cumulative_price_change) > features_df['mid_price_change'].std() * 3)
            ).astype(int)
        
        # Front running indicators
        if 'trade_volume_60' in features_df.columns and 'order_size' in features_df.columns:
            # Large orders preceding volume spikes
            volume_ma = features_df['trade_volume_60'].rolling(10, min_periods=1).mean()
            volume_spike = features_df['trade_volume_60'] > volume_ma * 2
            large_order_lead = features_df['large_order'].shift(1).fillna(0)
            features_df['front_running_pattern'] = (
                (large_order_lead == 1) & volume_spike
            ).astype(int)
        
        # Order book manipulation
        if 'depth_imbalance' in features_df.columns:
            # Extreme imbalances that reverse quickly
            imbalance_extreme = abs(features_df['depth_imbalance']) > 0.8
            imbalance_reversal = (features_df['depth_imbalance'] * features_df['depth_imbalance'].shift(1)) < 0
            features_df['order_book_manipulation'] = (
                imbalance_extreme & imbalance_reversal.shift(-1).fillna(False)
            ).astype(int)
        
        return features_df
    
    def _add_sequential_patterns(self, features_df):
        """Add sequential pattern detection for manipulation"""
        if 'order_size' in features_df.columns:
            # Size acceleration patterns
            size_diff = features_df['order_size'].diff()
            size_accel = size_diff.diff()
            features_df['size_acceleration'] = size_accel
            features_df['size_momentum_burst'] = (size_accel > size_accel.quantile(0.95)).astype(int)
            
            # Consecutive large order pattern
            large_order_threshold = features_df['order_size'].quantile(0.9)
            is_large = (features_df['order_size'] > large_order_threshold).astype(int)
            features_df['consecutive_large_orders'] = is_large.rolling(3, min_periods=1).sum()
            
            # Order size patterns (geometric sequences)
            size_ratio = features_df['order_size'] / features_df['order_size'].shift(1).fillna(1)
            features_df['size_ratio_pattern'] = (
                (size_ratio > 1.5) | (size_ratio < 0.67)
            ).astype(int)
            
            # Fibonacci-like sequences (common in algorithmic manipulation)
            for lag in [2, 3]:
                fib_pattern = (
                    features_df['order_size'] == 
                    features_df['order_size'].shift(1) + features_df['order_size'].shift(lag)
                )
                features_df[f'fibonacci_pattern_lag{lag}'] = fib_pattern.astype(int)
        
        if 'arrival_rate' in features_df.columns:
            # Arrival rate bursts
            arrival_ma = features_df['arrival_rate'].rolling(10, min_periods=1).mean()
            arrival_std = features_df['arrival_rate'].rolling(10, min_periods=1).std()
            features_df['arrival_burst'] = ((features_df['arrival_rate'] - arrival_ma) > 2 * arrival_std).astype(int)
            
            # Rhythmic patterns (algorithmic fingerprints)
            for period in [5, 10, 20]:
                periodic_pattern = features_df['arrival_rate'].rolling(period).apply(
                    lambda x: 1 if len(set(np.round(x, 2))) <= 3 else 0, raw=True
                )
                features_df[f'rhythmic_pattern_{period}'] = periodic_pattern.fillna(0)
        
        # Cross-variable sequential patterns
        if 'is_buy' in features_df.columns:
            # Buy-sell alternation patterns
            direction_pattern = features_df['is_buy'].rolling(6).apply(
                lambda x: 1 if len(set(x)) == 2 and all(x[i] != x[i+1] for i in range(len(x)-1)) else 0,
                raw=True
            )
            features_df['alternating_direction'] = direction_pattern.fillna(0)
        
        return features_df
    
    def _add_market_impact_features(self, features_df):
        """Add market impact and manipulation indicators"""
        if 'mid_price_change' in features_df.columns and 'order_size' in features_df.columns:
            # Price impact per unit size
            features_df['impact_per_size'] = features_df['mid_price_change'] / (features_df['order_size'] + 1e-8)
            features_df['abnormal_impact'] = (abs(features_df['impact_per_size']) > 
                                            abs(features_df['impact_per_size']).quantile(0.95)).astype(int)
            
            # Delayed impact patterns
            for lag in [1, 2, 3]:
                delayed_impact = features_df['mid_price_change'].shift(-lag)
                features_df[f'delayed_impact_{lag}'] = delayed_impact
                
                # Correlation between order size and delayed impact
                correlation_window = 20
                rolling_corr = features_df['order_size'].rolling(correlation_window).corr(
                    delayed_impact.rolling(correlation_window)
                )
                features_df[f'size_delayed_impact_corr_{lag}'] = rolling_corr.fillna(0)
           
            # Permanent vs temporary impact
            short_impact = features_df['mid_price_change'].rolling(3, min_periods=1).sum()
            long_impact = features_df['mid_price_change'].rolling(10, min_periods=1).sum()
            features_df['impact_permanence'] = long_impact / (short_impact + 1e-8)
            
            # Impact asymmetry (different impact for buys vs sells)
            if 'is_buy' in features_df.columns:
                buy_impact = features_df['mid_price_change'] * features_df['is_buy']
                sell_impact = features_df['mid_price_change'] * (1 - features_df['is_buy'])
                features_df['buy_impact'] = buy_impact
                features_df['sell_impact'] = sell_impact
                features_df['impact_asymmetry'] = abs(buy_impact) - abs(sell_impact)
        
        # Size relative to recent trading
        if 'trade_volume_60' in features_df.columns and 'order_size' in features_df.columns:
            features_df['size_vs_recent_volume'] = features_df['order_size'] / (features_df['trade_volume_60'] + 1e-8)
            features_df['dominant_order'] = (features_df['size_vs_recent_volume'] > 0.5).astype(int)
            
            # Market share indicators
            features_df['market_share'] = features_df['order_size'] / (features_df['trade_volume_60'] + features_df['order_size'])
            features_df['high_market_share'] = (features_df['market_share'] > 0.3).astype(int)
        
        # Liquidity consumption patterns
        if 'total_depth' in features_df.columns:
            features_df['liquidity_consumption'] = features_df['order_size'] / (features_df['total_depth'] + 1e-8)
            features_df['liquidity_depletion'] = (features_df['liquidity_consumption'] > 0.2).astype(int)
        
        return features_df
    
    def _add_rolling_features(self, features_df):
        """Add rolling statistical features"""
        key_features = ['order_size', 'spread', 'volatility', 'momentum', 'arrival_rate', 'imbalance']
        available_features = [f for f in key_features if f in features_df.columns]
        
        for feature in available_features:
            for window in [5, 10, 20]:
                features_df[f'{feature}_ma_{window}'] = features_df[feature].rolling(window, min_periods=1).mean()
                features_df[f'{feature}_std_{window}'] = features_df[feature].rolling(window, min_periods=1).std()
                features_df[f'{feature}_min_{window}'] = features_df[feature].rolling(window, min_periods=1).min()
                features_df[f'{feature}_max_{window}'] = features_df[feature].rolling(window, min_periods=1).max()
                features_df[f'{feature}_range_{window}'] = features_df[f'{feature}_max_{window}'] - features_df[f'{feature}_min_{window}']
                
                # Z-score
                ma_col = f'{feature}_ma_{window}'
                std_col = f'{feature}_std_{window}'
                features_df[f'{feature}_zscore_{window}'] = (
                    (features_df[feature] - features_df[ma_col]) / (features_df[std_col] + 1e-8)
                )
                
                # Percentile rank
                features_df[f'{feature}_percentile_{window}'] = features_df[feature].rolling(window).rank(pct=True)
                
                # Trend indicators
                features_df[f'{feature}_trend_{window}'] = (
                    features_df[feature] - features_df[feature].shift(window)
                ) / (features_df[feature].shift(window) + 1e-8)
        
        return features_df
    
    def _add_interaction_features(self, features_df):
        """Add interaction features between key variables"""
        # Size-based interactions
        if 'order_size' in features_df.columns:
            if 'spread' in features_df.columns:
                features_df['size_spread_interaction'] = features_df['order_size'] * features_df['spread']
            if 'volatility' in features_df.columns:
                features_df['size_vol_interaction'] = features_df['order_size'] * features_df['volatility']
            if 'arrival_rate' in features_df.columns:
                features_df['arrival_size_interaction'] = features_df['arrival_rate'] * features_df['order_size']
                
            # Size-time interactions
            if 'inter_arrival_time' in features_df.columns:
                features_df['size_time_interaction'] = features_df['order_size'] / (features_df['inter_arrival_time'] + 1e-8)
        
        # Volatility-based interactions
        if 'volatility' in features_df.columns:
            if 'arrival_rate' in features_df.columns:
                features_df['vol_arrival_interaction'] = features_df['volatility'] * features_df['arrival_rate']
            if 'spread' in features_df.columns:
                features_df['vol_spread_interaction'] = features_df['volatility'] * features_df['spread']
        
        # Depth-based interactions
        if 'depth_imbalance' in features_df.columns:
            if 'order_size' in features_df.columns:
                features_df['imbalance_size_interaction'] = features_df['depth_imbalance'] * features_df['order_size']
            if 'spread' in features_df.columns:
                features_df['imbalance_spread_interaction'] = features_df['depth_imbalance'] * features_df['spread']
        
        # Multi-way interactions
        if all(col in features_df.columns for col in ['order_size', 'volatility', 'spread']):
            features_df['size_vol_spread_interaction'] = (
                features_df['order_size'] * features_df['volatility'] * features_df['spread']
            )
        
        return features_df

class AdvancedToxicityDetector:
    """Ensemble detector with multiple algorithms and cross-validation"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.ensemble_weights = {}
        self.performance_metrics = {}
        self.best_hyperparameters = {}
        self.feature_importance = {}
        self.cv_results = {}
        self.validation_scores = {}
        
    def prepare_features(self, features_df, variance_threshold=0.01, correlation_threshold=0.95):
        """Feature preparation with selection"""
        print("Preparing features with selection...")
        
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
        
        print(f"Selected {len(self.feature_selector)} features after selection")
        
        return X_scaled, selected_features
    
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
    
    def _calculate_feature_importance(self, features_df):
        """Calculate feature importance using multiple methods"""
        print("Calculating feature importance...")
        
        feature_importance = {}
        
        # Variance-based importance
        variances = features_df.var()
        normalized_variances = variances / variances.max()

        # Isolation Forest feature importance
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
            iso_forest.fit(features_df)
            anomaly_scores = iso_forest.decision_function(features_df)
            
            correlations = {}
            for col in features_df.columns:
                corr = abs(np.corrcoef(features_df[col], anomaly_scores)[0, 1])
                correlations[col] = corr if not np.isnan(corr) else 0
            
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
    
    def optimize_hyperparameters_with_cv(self, X_scaled, n_trials=30, cv_folds=5):
        """Optimize hyperparameters using Optuna with cross-validation"""
        print(f"Optimizing hyperparameters with {n_trials} trials and {cv_folds}-fold CV...")
        
        cv_splitter = TimeSeriesSplit(n_splits=cv_folds)
        
        def objective(trial):
            contamination = trial.suggest_float('contamination', 0.01, 0.2)
            n_estimators = trial.suggest_int('n_estimators', 100, 300)
            n_neighbors = trial.suggest_int('n_neighbors', 5, 30)
            n_clusters = trial.suggest_int('n_clusters', 3, min(15, len(X_scaled) // 30))
            
            cv_scores = []
            
            for train_idx, test_idx in cv_splitter.split(X_scaled):
                try:
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    
                    iso_forest = IsolationForest(
                        contamination=contamination,
                        n_estimators=n_estimators,
                        random_state=42
                    )
                    iso_forest.fit(X_train)
                    
                    lof = LocalOutlierFactor(
                        n_neighbors=n_neighbors,
                        contamination=contamination,
                        novelty=True
                    )
                    lof.fit(X_train)
                    
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(X_train)
                    
                    if len(set(cluster_labels)) > 1:
                        test_cluster_labels = kmeans.predict(X_test)
                        
                        if len(set(test_cluster_labels)) > 1:
                            silhouette = silhouette_score(X_test, test_cluster_labels)
                            calinski = calinski_harabasz_score(X_test, test_cluster_labels)
                            davies_bouldin = davies_bouldin_score(X_test, test_cluster_labels)
                            
                            fold_score = silhouette + (calinski / 1000) - davies_bouldin
                            cv_scores.append(fold_score)
                        else:
                            cv_scores.append(-1)
                    else:
                        cv_scores.append(-1)
                        
                except Exception:
                    cv_scores.append(-10)
            
            return np.mean(cv_scores) if cv_scores else -10
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        self.best_hyperparameters = study.best_params
        print(f"Best hyperparameters: {self.best_hyperparameters}")
        print(f"Best CV score: {study.best_value:.4f}")
        
        return self.best_hyperparameters
    
    def train_ensemble_with_cv(self, X_scaled, hyperparameters=None, cv_folds=5):
        """Train ensemble of toxicity detectors with cross-validation"""
        print(f"Training ensemble detectors with {cv_folds}-fold cross-validation...")
        
        if hyperparameters is None:
            hyperparameters = self.best_hyperparameters
        
        cv_splitter = TimeSeriesSplit(n_splits=cv_folds)
        detectors = {}
        cv_results = {}
        
        # Isolation Forest variants with CV
        contamination_rates = [0.025, 0.05, 0.1, 0.15]
        for i, contamination in enumerate(contamination_rates):
            detector_name = f'isolation_forest_{contamination}'
            cv_scores = []
            
            for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X_scaled)):
                try:
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    
                    iso_forest = IsolationForest(
                        contamination=contamination,
                        n_estimators=hyperparameters.get('n_estimators', 200),
                        random_state=42 + i,
                        bootstrap=True,
                        n_jobs=-1
                    )
                    iso_forest.fit(X_train)
                    
                    train_scores = -iso_forest.decision_function(X_train)
                    test_scores = -iso_forest.decision_function(X_test)
                    
                    score_range = np.max(test_scores) - np.min(test_scores)
                    score_std = np.std(test_scores)
                    validation_score = score_range * score_std
                    
                    cv_scores.append(validation_score)
                    
                except Exception:
                    cv_scores.append(0)
            
            if np.mean(cv_scores) > 0:
                try:
                    final_iso_forest = IsolationForest(
                        contamination=contamination,
                        n_estimators=hyperparameters.get('n_estimators', 200),
                        random_state=42 + i,
                        bootstrap=True,
                        n_jobs=-1
                    )
                    final_iso_forest.fit(X_scaled)
                    detectors[detector_name] = final_iso_forest
                    cv_results[detector_name] = {
                        'mean_cv_score': np.mean(cv_scores),
                        'std_cv_score': np.std(cv_scores),
                        'fold_scores': cv_scores
                    }
                except Exception:
                    pass
        
        # LOF variants with CV
        neighbor_counts = [5, 10, 20]
        for neighbors in neighbor_counts:
            detector_name = f'lof_{neighbors}'
            cv_scores = []
            
            for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X_scaled)):
                try:
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    
                    lof = LocalOutlierFactor(
                        n_neighbors=neighbors,
                        contamination=hyperparameters.get('contamination', 0.1),
                        novelty=True,
                        n_jobs=-1
                    )
                    lof.fit(X_train)
                    
                    test_scores = -lof.score_samples(X_test)
                    
                    score_range = np.max(test_scores) - np.min(test_scores)
                    score_std = np.std(test_scores)
                    validation_score = score_range * score_std
                    
                    cv_scores.append(validation_score)
                    
                except Exception:
                    cv_scores.append(0)
            
            if np.mean(cv_scores) > 0:
                try:
                    final_lof = LocalOutlierFactor(
                        n_neighbors=neighbors,
                        contamination=hyperparameters.get('contamination', 0.1),
                        novelty=True,
                        n_jobs=-1
                    )
                    final_lof.fit(X_scaled)
                    detectors[detector_name] = final_lof
                    cv_results[detector_name] = {
                        'mean_cv_score': np.mean(cv_scores),
                        'std_cv_score': np.std(cv_scores),
                        'fold_scores': cv_scores
                    }
                except Exception:
                    pass
        
        # One-Class SVM variants with CV
        svm_configs = [
            {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.05},
            {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.1},
            {'kernel': 'poly', 'degree': 3, 'nu': 0.1}
        ]
        
        for i, config in enumerate(svm_configs):
            detector_name = f'svm_{config["kernel"]}_{config["nu"]}'
            cv_scores = []
            
            for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X_scaled)):
                try:
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    
                    svm = OneClassSVM(**config)
                    svm.fit(X_train)
                    
                    test_scores = -svm.decision_function(X_test)
                    
                    score_range = np.max(test_scores) - np.min(test_scores)
                    score_std = np.std(test_scores)
                    validation_score = score_range * score_std
                    
                    cv_scores.append(validation_score)
                    
                except Exception:
                    cv_scores.append(0)
            
            if np.mean(cv_scores) > 0:
                try:
                    final_svm = OneClassSVM(**config)
                    final_svm.fit(X_scaled)
                    detectors[detector_name] = final_svm
                    cv_results[detector_name] = {
                        'mean_cv_score': np.mean(cv_scores),
                        'std_cv_score': np.std(cv_scores),
                        'fold_scores': cv_scores
                    }
                except Exception:
                    pass
        
        # K-means based detectors with CV
        cluster_counts = [5, 8, 12]
        for n_clusters in cluster_counts:
            detector_name = f'kmeans_{n_clusters}'
            cv_scores = []
            
            for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X_scaled)):
                try:
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    
                    n_clust = min(n_clusters, max(2, len(X_train) // 30))
                    if n_clust < 2:
                        continue
                    
                    kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(X_train)
                    
                    test_labels = kmeans.predict(X_test)
                    
                    if len(set(test_labels)) > 1:
                        silhouette = silhouette_score(X_test, test_labels)
                        cv_scores.append(silhouette)
                    else:
                        cv_scores.append(0)
                        
                except Exception:
                    cv_scores.append(0)
            
            if np.mean(cv_scores) > 0:
                try:
                    n_clust = min(n_clusters, max(2, len(X_scaled) // 30))
                    if n_clust >= 2:
                        final_kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
                        cluster_labels = final_kmeans.fit_predict(X_scaled)
                        distances = np.min(final_kmeans.transform(X_scaled), axis=1)
                        
                        detectors[detector_name] = {
                            'kmeans': final_kmeans,
                            'distance_threshold': np.percentile(distances, 95),
                            'cluster_sizes': np.bincount(cluster_labels)
                        }
                        cv_results[detector_name] = {
                            'mean_cv_score': np.mean(cv_scores),
                            'std_cv_score': np.std(cv_scores),
                            'fold_scores': cv_scores
                        }
                except Exception:
                    pass
        
        self.models = detectors
        self.cv_results = cv_results
        
        print(f"Trained {len(detectors)} detectors with cross-validation")
        
        return detectors
    
    def calculate_ensemble_scores_with_validation(self, X_scaled):
        """Calculate ensemble toxicity scores with validation-based weighting"""
        print("Calculating ensemble scores with validation-based weighting...")
        
        individual_scores = {}
        individual_weights = {}
        
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
                else:
                    continue
                
                # Normalisation
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
                
                cv_performance = self.cv_results.get(name, {})
                cv_weight = self._calculate_cv_weight(scores, name, cv_performance)
                individual_weights[name] = cv_weight
                
            except Exception:
                continue
        
        # Normalise weights
        total_weight = sum(individual_weights.values())
        if total_weight > 0:
            max_weight = 0.3
            normalized_weights = {}
            for name, weight in individual_weights.items():
                normalized_weights[name] = min(weight / total_weight, max_weight)
            
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
        self.validation_scores = individual_scores
        
        print("Detector weights (including CV performance):")
        for name, weight in sorted(individual_weights.items(), key=lambda x: x[1], reverse=True):
            cv_score = self.cv_results.get(name, {}).get('mean_cv_score', 0)
            print(f"  {name}: {weight:.3f} (CV: {cv_score:.3f})")
        
        return ensemble_scores, individual_scores
    
    def _calculate_cv_weight(self, scores, detector_name, cv_performance):
        """Calculate weight with numerical stability"""
        try:
            score_std = np.std(scores)
            score_range = np.max(scores) - np.min(scores)
            separation_score = score_std * score_range
            
            cv_score = cv_performance.get('mean_cv_score', 0)
            cv_std = cv_performance.get('std_cv_score', 1)
            
            if cv_score > 1e6 or np.isnan(cv_score) or np.isinf(cv_score):
                cv_score = np.log1p(cv_score) if cv_score > 0 else 0
            
            cv_score = min(cv_score, 100)
            cv_stability = 1 / (1 + cv_std) if cv_std > 0 and cv_std < 1e6 else 0.1
            
            type_bonus = 1.0
            if 'isolation_forest' in detector_name:
                type_bonus = 1.2
            elif 'lof' in detector_name:
                type_bonus = 0.8
            elif 'svm' in detector_name:
                type_bonus = 1.1
            elif 'kmeans' in detector_name:
                type_bonus = 1.15
            
            score_volatility = np.std(np.diff(scores))
            stability_score = 1 / (1 + score_volatility)
            
            normalized_cv_score = cv_score / (1 + cv_score)
            
            weight = (separation_score * 0.25 + 
                    normalized_cv_score * 0.15 +
                    cv_stability * 0.25 +
                    stability_score * 0.15 + 
                    type_bonus * 0.2)
            
            return max(0.01, min(weight, 1.5))
            
        except Exception:
            return 0.05
    
    def evaluate_comprehensive_performance_with_cv(self, X_scaled, ensemble_scores, individual_scores):
        """Comprehensive performance evaluation including CV results"""
        print("Evaluating comprehensive detector performance with CV analysis...")
        
        metrics = {}
        
        # Score distribution analysis
        metrics['score_stats'] = {
            'mean': ensemble_scores.mean(),
            'std': ensemble_scores.std(),
            'min': ensemble_scores.min(),
            'max': ensemble_scores.max(),
            'skewness': stats.skew(ensemble_scores),
            'kurtosis': stats.kurtosis(ensemble_scores),
            'iqr': np.percentile(ensemble_scores, 75) - np.percentile(ensemble_scores, 25)
        }
        
        # Anomaly detection rates with separation scores
        thresholds = [85, 90, 95, 97, 99, 99.5]
        anomaly_rates = {}
        for threshold in thresholds:
            threshold_value = np.percentile(ensemble_scores, threshold)
            anomaly_rate = np.mean(ensemble_scores > threshold_value)
            anomaly_rates[f'{threshold}th_percentile'] = anomaly_rate
            
            if anomaly_rate > 0 and anomaly_rate < 1:
                anomaly_scores = ensemble_scores[ensemble_scores > threshold_value]
                normal_scores = ensemble_scores[ensemble_scores <= threshold_value]
                if len(normal_scores) > 0 and normal_scores.std() > 0:
                    separation = (anomaly_scores.mean() - normal_scores.mean()) / normal_scores.std()
                    anomaly_rates[f'{threshold}th_separation'] = separation
        
        metrics['anomaly_rates'] = anomaly_rates
        
        # Cross-validation performance analysis
        cv_performance = {}
        for name, cv_results in self.cv_results.items():
            cv_performance[name] = {
                'mean_score': cv_results['mean_cv_score'],
                'std_score': cv_results['std_cv_score'],
                'stability': 1 / (1 + cv_results['std_cv_score']) if cv_results['std_cv_score'] > 0 else 1,
                'consistency': len([s for s in cv_results['fold_scores'] if s > 0]) / len(cv_results['fold_scores']),
                'best_fold': max(cv_results['fold_scores']),
                'worst_fold': min(cv_results['fold_scores'])
            }
        
        metrics['cv_performance'] = cv_performance
        
        # Clustering quality assessment
        clustering_scores = {}
        for name, model in self.models.items():
            if 'kmeans' in name and isinstance(model, dict):
                try:
                    kmeans = model['kmeans']
                    cluster_labels = kmeans.predict(X_scaled)
                    if len(set(cluster_labels)) > 1:
                        sil_score = silhouette_score(X_scaled, cluster_labels)
                        db_score = davies_bouldin_score(X_scaled, cluster_labels)
                        ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
                        
                        clustering_scores[f'{name}_silhouette'] = sil_score
                        clustering_scores[f'{name}_davies_bouldin'] = db_score
                        clustering_scores[f'{name}_calinski_harabasz'] = ch_score
                except Exception:
                    continue
        
        metrics['clustering_quality'] = clustering_scores
        
        # Individual detector performance with CV integration
        individual_performance = {}
        for name, scores in individual_scores.items():
            try:
                cv_results = self.cv_results.get(name, {})
                performance = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'range': scores.max() - scores.min(),
                    'weight': self.ensemble_weights.get(name, 0),
                    'consistency': self._calculate_consistency(scores),
                    'stability': 1 / (1 + np.std(np.diff(scores))),
                    'cv_mean': cv_results.get('mean_cv_score', 0),
                    'cv_std': cv_results.get('std_cv_score', 0),
                    'cv_stability': cv_performance.get(name, {}).get('stability', 0),
                    'cv_consistency': cv_performance.get(name, {}).get('consistency', 0)
                }
                individual_performance[name] = performance
            except Exception:
                continue
        
        metrics['individual_detectors'] = individual_performance
        
        # Ensemble diversity metrics
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
                    'min_correlation': np.min(correlations),
                    'correlation_std': np.std(correlations),
                    'diversity_score': 1 - abs(np.mean(correlations)),
                    'pairwise_disagreement': np.mean(correlations < 0.5)
                }
                
                metrics['ensemble_diversity'] = diversity_metrics
            except Exception:
                pass
        
        # Cross-validation stability metrics
        cv_stability_metrics = {
            'overall_cv_mean': np.mean([cv['mean_cv_score'] for cv in self.cv_results.values()]),
            'overall_cv_std': np.mean([cv['std_cv_score'] for cv in self.cv_results.values()]),
            'detector_consistency': np.mean([cv_performance[name]['consistency'] 
                                            for name in cv_performance.keys()]),
            'best_performing_detector': max(cv_performance.items(), 
                                            key=lambda x: x[1]['mean_score'])[0] if cv_performance else None,
            'most_stable_detector': max(cv_performance.items(), 
                                        key=lambda x: x[1]['stability'])[0] if cv_performance else None
        }
        
        metrics['cv_stability'] = cv_stability_metrics
        
        if self.feature_importance:
            metrics['feature_importance'] = dict(sorted(
                self.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:20])
        
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
    
    def save_model_with_cv_results(self, save_dir="toxicity_models"):
        """Save the model with comprehensive metadata including CV results"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{save_dir}/toxicity_detector_cv_{timestamp}.joblib"
        
        model_package = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_selector': self.feature_selector,
            'ensemble_weights': self.ensemble_weights,
            'performance_metrics': self.performance_metrics,
            'best_hyperparameters': self.best_hyperparameters,
            'feature_importance': self.feature_importance,
            'cv_results': self.cv_results,
            'validation_scores': self.validation_scores,
            'timestamp': timestamp,
            'version': '1.0_production_cv',
            'n_features': len(self.feature_selector) if self.feature_selector else 0,
            'n_detectors': len(self.models),
            'cv_summary': {
                'mean_cv_performance': np.mean([cv['mean_cv_score'] for cv in self.cv_results.values()]),
                'cv_stability': np.mean([1/(1+cv['std_cv_score']) for cv in self.cv_results.values()]),
                'best_detector_cv': max(self.cv_results.items(), 
                                        key=lambda x: x[1]['mean_cv_score'])[0] if self.cv_results else None,
                'most_stable_detector_cv': min(self.cv_results.items(), 
                                                key=lambda x: x[1]['std_cv_score'])[0] if self.cv_results else None
            }
        }
        
        joblib.dump(model_package, model_path)
        
        print(f"Model with CV results saved to: {model_path}")
        return model_path

def create_cv_visualizations(features_df, ensemble_scores, individual_scores, detector):
    """Create comprehensive performance visualizations including CV results"""
    
    plots_dir = "toxicity_plots_cv"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.style.use('ggplot')
    plt.rcParams.update({
        'figure.figsize': (14, 10),
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    # Cross-Validation Performance Dashboard
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # CV scores comparison
    cv_names = list(detector.cv_results.keys())
    cv_means = [detector.cv_results[name]['mean_cv_score'] for name in cv_names]
    cv_stds = [detector.cv_results[name]['std_cv_score'] for name in cv_names]
    
    sorted_items = sorted(zip(cv_names, cv_means, cv_stds), key=lambda x: x[1], reverse=True)
    cv_names_sorted, cv_means_sorted, cv_stds_sorted = zip(*sorted_items)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(cv_names_sorted)))
    bars = axes[0, 0].bar(range(len(cv_names_sorted)), cv_means_sorted, 
                            yerr=cv_stds_sorted, color=colors, alpha=0.8, capsize=5)
    axes[0, 0].set_xticks(range(len(cv_names_sorted)))
    axes[0, 0].set_xticklabels([name[:12] for name in cv_names_sorted], rotation=45, ha='right')
    axes[0, 0].set_ylabel('CV Score')
    axes[0, 0].set_title('Cross-Validation Performance (Mean ± Std)')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    for bar, mean, std in zip(bars, cv_means_sorted, cv_stds_sorted):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01, 
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    
    # CV stability vs performance scatter
    cv_performance = detector.performance_metrics.get('cv_performance', {})
    if cv_performance:
        stabilities = [cv_performance[name]['stability'] for name in cv_performance.keys()]
        means = [cv_performance[name]['mean_score'] for name in cv_performance.keys()]
        consistencies = [cv_performance[name]['consistency'] for name in cv_performance.keys()]
        
        scatter = axes[0, 1].scatter(stabilities, means, c=consistencies, 
                                    s=100, alpha=0.7, cmap='plasma')
        axes[0, 1].set_xlabel('CV Stability')
        axes[0, 1].set_ylabel('CV Mean Score')
        axes[0, 1].set_title('CV Performance vs Stability')
        axes[0, 1].grid(True, alpha=0.3)
        
        for name, x, y in zip(cv_performance.keys(), stabilities, means):
            axes[0, 1].annotate(name[:8], (x, y), xytext=(5, 5), 
                                textcoords='offset points', fontsize=8)
        
        plt.colorbar(scatter, ax=axes[0, 1], label='Consistency')
    
    # Ensemble weights vs CV performance
    weights = list(detector.ensemble_weights.values())
    detector_names = list(detector.ensemble_weights.keys())
    cv_scores = [detector.cv_results.get(name, {}).get('mean_cv_score', 0) for name in detector_names]
    
    axes[0, 2].scatter(cv_scores, weights, alpha=0.7, s=100, color='orange')
    axes[0, 2].set_xlabel('CV Score')
    axes[0, 2].set_ylabel('Ensemble Weight')
    axes[0, 2].set_title('Ensemble Weight vs CV Performance')
    axes[0, 2].grid(True, alpha=0.3)
    
    if len(cv_scores) > 1:
        z = np.polyfit(cv_scores, weights, 1)
        p = np.poly1d(z)
        axes[0, 2].plot(sorted(cv_scores), p(sorted(cv_scores)), "r--", alpha=0.8)
    
    # Score distribution
    axes[1, 0].hist(ensemble_scores, bins=60, alpha=0.7, color='skyblue', 
                    edgecolor='black', density=True)
    axes[1, 0].axvline(np.percentile(ensemble_scores, 95), color='red', 
                        linestyle='--', linewidth=2, label='95th')
    axes[1, 0].axvline(np.percentile(ensemble_scores, 99), color='darkred', 
                        linestyle='--', linewidth=2, label='99th')
    axes[1, 0].set_xlabel('Toxicity Score')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Toxicity Score Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Feature importance
    if hasattr(detector, 'feature_importance') and detector.feature_importance:
        top_features = dict(sorted(detector.feature_importance.items(), 
                                    key=lambda x: x[1], reverse=True)[:12])
        feature_names = list(top_features.keys())
        importance_scores = list(top_features.values())
        
        colors_feat = plt.cm.plasma(np.linspace(0, 1, len(importance_scores)))
        axes[1, 1].barh(range(len(feature_names)), importance_scores, 
                        color=colors_feat, alpha=0.8)
        axes[1, 1].set_yticks(range(len(feature_names)))
        axes[1, 1].set_yticklabels([name[:18] for name in feature_names])
        axes[1, 1].set_xlabel('Importance Score')
        axes[1, 1].set_title('Top 12 Feature Importance')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    # Model summary
    cv_summary = detector.performance_metrics.get('cv_stability', {})
    performance_summary = detector.performance_metrics.get('score_stats', {})
    
    summary_text = f"""
    MODEL SUMMARY (with Cross-Validation)
    
    Dataset: {len(ensemble_scores)} samples
    Features: {len(detector.feature_selector) if detector.feature_selector else 0}
    Detectors: {len(detector.models)}
    CV Folds: 5
    
    Score Statistics:
    • Mean: {performance_summary.get('mean', 0):.3f}
    • Std: {performance_summary.get('std', 0):.3f}
    • Skewness: {performance_summary.get('skewness', 0):.3f}
    
    Cross-Validation Results:
    • Overall CV Mean: {cv_summary.get('overall_cv_mean', 0):.3f}
    • Overall CV Std: {cv_summary.get('overall_cv_std', 0):.3f}
    • Detector Consistency: {cv_summary.get('detector_consistency', 0):.3f}
    • Best Detector: {cv_summary.get('best_performing_detector', 'N/A')[:15]}
    • Most Stable: {cv_summary.get('most_stable_detector', 'N/A')[:15]}
    """
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    axes[1, 2].set_title('CV Model Summary', fontweight='bold')
    
    # Additional plots
    axes[2, 0].axis('off')
    axes[2, 1].axis('off')
    axes[2, 2].axis('off')
    
    plt.suptitle(f'Toxicity Detection Model - Cross-Validation Analysis\nTimestamp: {timestamp}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/cv_comprehensive_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Detailed CV Performance Analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # CV score distributions by detector type
    detector_types = {}
    for name, results in detector.cv_results.items():
        detector_type = name.split('_')[0]
        if detector_type not in detector_types:
            detector_types[detector_type] = []
        detector_types[detector_type].extend(results['fold_scores'])
    
    ax_idx = 0
    for detector_type, scores in detector_types.items():
        if ax_idx < len(axes) and scores:
            axes[ax_idx].hist(scores, bins=20, alpha=0.7, label=detector_type, 
                            color=plt.cm.Set3(ax_idx))
            axes[ax_idx].axvline(np.mean(scores), color='red', linestyle='--', 
                                label=f'Mean: {np.mean(scores):.3f}')
            axes[ax_idx].set_xlabel('CV Score')
            axes[ax_idx].set_ylabel('Frequency')
            axes[ax_idx].set_title(f'{detector_type.title()} CV Score Distribution')
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)
            ax_idx += 1
    
    for i in range(ax_idx, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Detector Type CV Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/cv_detector_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"CV visualisations saved to: {plots_dir}")
    
    plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png') and timestamp in f]
    print(f"Generated {len(plot_files)} CV plot files:")
    for file in sorted(plot_files):
        print(f"  {file}")

def create_summary_report(detector, ensemble_scores, metrics):
    """Create summary report with actionable insights"""
    
    print("\n" + "="*80)
    print(" MODEL ANALYSIS & RECOMMENDATIONS")
    print("="*80)
    
    cv_performance = metrics.get('cv_performance', {})
    
    stable_detectors = []
    unstable_detectors = []
    
    for name, perf in cv_performance.items():
        if perf['stability'] > 0.5 and perf['mean_score'] < 1000:
            stable_detectors.append((name, perf))
        else:
            unstable_detectors.append((name, perf))
    
    print(f"\n DETECTOR STABILITY ANALYSIS:")
    print(f"  Stable Detectors: {len(stable_detectors)}")
    print(f"  Unstable Detectors: {len(unstable_detectors)}")
    
    print(f"\n STABLE DETECTORS (Recommended for production):")
    for name, perf in sorted(stable_detectors, key=lambda x: x[1]['mean_score'], reverse=True)[:5]:
        weight = detector.ensemble_weights.get(name, 0)
        print(f"  • {name.replace('_', ' ').title()}: Weight={weight:.3f}, CV={perf['mean_score']:.3f}, Stability={perf['stability']:.3f}")
    
    print(f"\n UNSTABLE DETECTORS (Require attention):")
    for name, perf in unstable_detectors[:3]:
        weight = detector.ensemble_weights.get(name, 0)
        print(f"  • {name.replace('_', ' ').title()}: Weight={weight:.3f}, CV={perf['mean_score']:.1e}, Stability={perf['stability']:.3f}")
    
    print(f"\n TOP MARKET MANIPULATION INDICATORS:")
    if hasattr(detector, 'feature_importance') and detector.feature_importance:
        top_features = sorted(detector.feature_importance.items(), key=lambda x: x[1], reverse=True)[:8]
        for i, (feature, importance) in enumerate(top_features, 1):
            feature_type = ""
            if 'size' in feature.lower():
                feature_type = "[ORDER SIZE]"
            elif 'arrival' in feature.lower():
                feature_type = "[TIMING]"
            elif 'depth' in feature.lower():
                feature_type = "[MARKET DEPTH]"
            elif 'spread' in feature.lower():
                feature_type = "[SPREAD]"
            elif 'volatility' in feature.lower():
                feature_type = "[VOLATILITY]"
            
            print(f"  {i}. {feature.replace('_', ' ').title()} {feature_type}: {importance:.4f}")
    
    anomaly_rates = metrics.get('anomaly_rates', {})
    print(f"\n ANOMALY DETECTION EFFECTIVENESS:")
    
    thresholds = [95, 99, 99.5]
    for threshold in thresholds:
        rate = anomaly_rates.get(f'{threshold}th_percentile', 0) * 100
        separation = anomaly_rates.get(f'{threshold}th_separation', 0)
        
        effectiveness = ""
        if separation > 3.0:
            effectiveness = "EXCELLENT"
        elif separation > 2.0:
            effectiveness = "GOOD"
        elif separation > 1.0:
            effectiveness = "FAIR"
        else:
            effectiveness = "POOR"
        
        print(f"  {threshold}th percentile: {rate:.1f}% detection, {separation:.3f} separation [{effectiveness}]")
    
    print(f"\n MODEL RECOMMENDATIONS:")
    
    diversity_score = metrics.get('ensemble_diversity', {}).get('diversity_score', 0)
    if diversity_score > 0.8:
        print("  Good ensemble diversity - model robust to individual detector failures")
    elif diversity_score > 0.6:
        print("  Good ensemble diversity - consider adding more diverse detectors")
    else:
        print("  Mixed stability - monitor unstable detectors closely")
    
    stable_count = len(stable_detectors)
    total_count = len(cv_performance)
    stability_ratio = stable_count / total_count if total_count > 0 else 0
    
    # Feature recommendations
    top_feature_types = {}
    if hasattr(detector, 'feature_importance') and detector.feature_importance:
        for feature, importance in detector.feature_importance.items():
            if 'size' in feature.lower():
                top_feature_types['Order Size'] = top_feature_types.get('Order Size', 0) + importance
            elif 'arrival' in feature.lower() or 'timing' in feature.lower():
                top_feature_types['Timing'] = top_feature_types.get('Timing', 0) + importance
            elif 'depth' in feature.lower():
                top_feature_types['Market Depth'] = top_feature_types.get('Market Depth', 0) + importance
            elif 'spread' in feature.lower():
                top_feature_types['Spread'] = top_feature_types.get('Spread', 0) + importance
    
    print(f"\n FEATURE CATEGORY IMPORTANCE:")
    for category, total_importance in sorted(top_feature_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {category}: {total_importance:.3f}")
    
    print(f"\n DEPLOYMENT READINESS:")
    score_stats = metrics.get('score_stats', {})
    
    if score_stats.get('std', 0) > 0.3:
        print("  Good score discrimination - can distinguish normal from toxic orders")
    else:
        print("  Low score discrimination - may have difficulty separating anomalies")
    
    if stability_ratio > 0.7 and diversity_score > 0.7:
        print("  READY FOR PRODUCTION - Model meets stability and diversity requirements")
        print("  • Deploy with confidence monitoring on unstable detectors")
        print("  • Set up alerts for detection rates outside expected ranges")
    elif stability_ratio > 0.5:
        print("  READY FOR PILOT TESTING - Monitor closely before full deployment")
        print("  • Run parallel with existing systems initially")
        print("  • Focus monitoring on unstable detectors")
    else:
        print("  NOT READY FOR PRODUCTION - Requires further tuning")
        print("  • Retrain with different hyperparameters")
        print("  • Consider removing unstable detectors")

def main_training_pipeline_with_cv(data_dir="market_data", n_trials=30, cv_folds=5):
    """Main training pipeline with comprehensive cross-validation"""
    print("="*80)
    print("MARKET TOXICITY DETECTION MODEL TRAINING WITH CROSS-VALIDATION")
    print("Feature Engineering & Ensemble Methods with CV")
    print("="*80)
    
    # Step 1: Load market data
    print("\n1. LOADING MARKET DATA")
    print("-" * 40)
    
    feature_engineer = MarketDataFeatureEngineer()
    orders_df, lob_df, trades_df = feature_engineer.load_market_data(data_dir)
    
    print(f"\nData Summary:")
    print(f"  Orders: {len(orders_df)} records")
    print(f"  LOB Snapshots: {len(lob_df)} records")
    print(f"  Trades: {len(trades_df)} records")
    
    # Step 2: Feature extraction
    print("\n2. FEATURE EXTRACTION")
    print("-" * 40)
    
    features_df = feature_engineer.extract_features(orders_df, lob_df, trades_df)
    
    # Step 3: Feature preparation
    print("\n3. FEATURE PREPARATION")
    print("-" * 40)
    
    detector = AdvancedToxicityDetector()
    X_scaled, selected_features = detector.prepare_features(features_df)
    
    print(f"Final feature set: {X_scaled.shape}")
    
    # Step 4: Hyperparameter optimization with CV
    print("\n4. HYPERPARAMETER OPTIMIZATION WITH CROSS-VALIDATION")
    print("-" * 40)
    
    best_params = detector.optimize_hyperparameters_with_cv(X_scaled, n_trials=n_trials, cv_folds=cv_folds)
    
    # Step 5: Train ensemble with CV
    print("\n5. TRAINING ENSEMBLE WITH CROSS-VALIDATION")
    print("-" * 40)
    
    detectors = detector.train_ensemble_with_cv(X_scaled, best_params, cv_folds=cv_folds)
    
    # Step 6: Calculate ensemble scores with validation
    print("\n6. CALCULATING ENSEMBLE SCORES WITH VALIDATION")
    print("-" * 40)
    
    ensemble_scores, individual_scores = detector.calculate_ensemble_scores_with_validation(X_scaled)
    
    # Step 7: Comprehensive performance evaluation with CV
    print("\n7. COMPREHENSIVE PERFORMANCE EVALUATION WITH CV")
    print("-" * 40)
    
    metrics = detector.evaluate_comprehensive_performance_with_cv(X_scaled, ensemble_scores, individual_scores)
    
    # Step 8: Generate CV visualisations
    print("\n8. GENERATING CV VISUALISATIONS")
    print("-" * 40)
    
    create_cv_visualizations(selected_features, ensemble_scores, individual_scores, detector)
    
    # Step 9: Save model with CV results
    print("\n9. SAVING MODEL WITH CV RESULTS")
    print("-" * 40)
    
    model_path = detector.save_model_with_cv_results()
    
    # Print comprehensive results summary
    print("\n" + "="*80)
    print(" CROSS-VALIDATION TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
    
    print(f"\n MODEL PERFORMANCE SUMMARY (with Cross-Validation):")
    anomaly_detection = metrics.get('anomaly_rates', {})
    score_stats = metrics.get('score_stats', {})
    diversity_stats = metrics.get('ensemble_diversity', {})
    cv_stability = metrics.get('cv_stability', {})
    cv_performance = metrics.get('cv_performance', {})
    
    print(f"  Dataset Size: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    print(f"  Ensemble: {len(detector.models)} detectors")
    print(f"  CV Folds: {cv_folds}")
    print(f"  Effective Contributors: {len([w for w in detector.ensemble_weights.values() if w > 0.05])} detectors")
    
    print(f"\n CROSS-VALIDATION PERFORMANCE:")
    print(f"  Overall CV Mean: {cv_stability.get('overall_cv_mean', 0):.4f}")
    print(f"  Overall CV Std: {cv_stability.get('overall_cv_std', 0):.4f}")
    print(f"  Detector Consistency: {cv_stability.get('detector_consistency', 0):.4f}")
    print(f"  Best Performing Detector: {cv_stability.get('best_performing_detector', 'N/A')}")
    print(f"  Most Stable Detector: {cv_stability.get('most_stable_detector', 'N/A')}")
    
    print(f"\n ANOMALY DETECTION PERFORMANCE:")
    for threshold in [90, 95, 97, 99]:
        rate = anomaly_detection.get(f'{threshold}th_percentile', 0)
        separation = anomaly_detection.get(f'{threshold}th_separation', 0)
        print(f"  {threshold}th percentile: {rate*100:.2f}% detection rate, {separation:.3f} separation score")
    
    print(f"\nSCORE DISTRIBUTION QUALITY:")
    print(f"  Mean: {score_stats.get('mean', 0):.4f}")
    print(f"  Std: {score_stats.get('std', 0):.4f}")
    print(f"  Skewness: {score_stats.get('skewness', 0):.3f}")
    print(f"  Kurtosis: {score_stats.get('kurtosis', 0):.3f}")
    print(f"  IQR: {score_stats.get('iqr', 0):.4f}")
    
    print(f"\n ENSEMBLE QUALITY:")
    print(f"  Diversity Score: {diversity_stats.get('diversity_score', 0):.3f}/1.0")
    print(f"  Average Correlation: {diversity_stats.get('avg_correlation', 0):.3f}")
    print(f"  Correlation Std: {diversity_stats.get('correlation_std', 0):.3f}")
    print(f"  Pairwise Disagreement: {diversity_stats.get('pairwise_disagreement', 0):.3f}")
    print(f"  Max Individual Weight: {max(detector.ensemble_weights.values()) if detector.ensemble_weights else 0:.3f}")
    
    print(f"\nTOP PERFORMING DETECTORS (CV-based):")
    if cv_performance:
        top_cv_detectors = sorted([(name, perf['mean_score'], perf['stability']) 
                                    for name, perf in cv_performance.items()], 
                                    key=lambda x: x[1], reverse=True)[:5]
        for i, (name, cv_score, stability) in enumerate(top_cv_detectors, 1):
            weight = detector.ensemble_weights.get(name, 0)
            print(f"  {i}. {name.replace('_', ' ').title()}: CV={cv_score:.3f}, Stability={stability:.3f}, Weight={weight:.3f}")
    else:
        print("  No CV performance data available")
    
    print(f"\nOPTIMISED HYPERPARAMETERS:")
    for param, value in best_params.items():
        print(f"  {param.replace('_', ' ').title()}: {value}")
    
    print(f"\nTOP FEATURES FOR TOXICITY DETECTION:")
    if hasattr(detector, 'feature_importance') and detector.feature_importance:
        top_features = sorted(detector.feature_importance.items(), key=lambda x: x[1], reverse=True)[:8]
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"  {i}. {feature.replace('_', ' ').title()}: {importance:.4f}")
    
    print(f"\nCLUSTERING QUALITY ASSESSMENT:")
    clustering_quality = metrics.get('clustering_quality', {})
    if clustering_quality:
        for detector_name, score in sorted(clustering_quality.items(), key=lambda x: x[1], reverse=True):
            if 'silhouette' in detector_name:
                print(f"  {detector_name.replace('_', ' ').title()}: {score:.3f}")
    
    print(f"\nCROSS-VALIDATION INSIGHTS:")
    if cv_performance:
        most_consistent = max(cv_performance.items(), key=lambda x: x[1]['consistency'])[0]
        highest_cv = max([p['mean_score'] for p in cv_performance.values()])
        avg_stability = np.mean([p['stability'] for p in cv_performance.values()])
        
        print(f"  Most Consistent Detector: {most_consistent}")
        print(f"  Highest CV Score: {highest_cv:.3f}")
        print(f"  Average CV Stability: {avg_stability:.3f}")
    else:
        print("  No CV performance insights available")
    
    print(f"\nMODEL OUTPUTS:")
    print(f" - Model with CV: {model_path}")
    print(f" - CV Visualisations: toxicity_plots_cv/")
    print(f" - Training Timestamp: {detector.performance_metrics.get('timestamp', 'N/A')}")
    print(f" - Superior toxicity detection with {len(detector.models)} CV-validated ensemble detectors")
    print(f" - Feature engineering with {len(detector.feature_selector)} optimised features")
    print(f" - Cross-validated anomaly separation with avg CV score: {cv_stability.get('overall_cv_mean', 0):.3f}")
    print(f" - High ensemble diversity score: {diversity_stats.get('diversity_score', 0):.3f}")
    print(f" - CV-validated model stability: {cv_stability.get('overall_cv_std', 0):.3f}")
    
    # Generate enhanced analysis report
    create_summary_report(detector, ensemble_scores, metrics)
    
    return detector, ensemble_scores, metrics

def predict_toxicity_scores(model_path, new_data_path=None, orders_df=None, lob_df=None, trades_df=None):
    """Predict toxicity scores for new market data"""
    
    # Load trained model
    model_package = joblib.load(model_path)
    
    models = model_package['models']
    scalers = model_package['scalers']
    feature_selector = model_package['feature_selector']
    ensemble_weights = model_package['ensemble_weights']
    
    print(f"Loaded model with {len(models)} detectors")
    
    # Prepare new data
    if new_data_path:
        feature_engineer = MarketDataFeatureEngineer()
        orders_df, lob_df, trades_df = feature_engineer.load_market_data(new_data_path)
        features_df = feature_engineer.extract_features(orders_df, lob_df, trades_df)
    elif orders_df is not None:
        feature_engineer = MarketDataFeatureEngineer()
        if lob_df is None:
            lob_df = pd.DataFrame()
        if trades_df is None:
            trades_df = pd.DataFrame()
        features_df = feature_engineer.extract_features(orders_df, lob_df, trades_df)
    else:
        raise ValueError("Either new_data_path or orders_df must be provided")
    
    # Select features and scale
    selected_features = features_df[feature_selector].fillna(0).replace([np.inf, -np.inf], 0)
    X_scaled = scalers['main'].transform(selected_features)
    
    # Generate predictions from each detector
    individual_scores = {}
    
    for name, model in models.items():
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
            else:
                continue
            
            # Normalise scores
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
            
        except Exception as e:
            print(f"Error with detector {name}: {e}")
            continue
    
    # Calculate ensemble scores
    ensemble_scores = np.zeros(len(X_scaled))
    for name, scores in individual_scores.items():
        weight = ensemble_weights.get(name, 0)
        ensemble_scores += weight * scores
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'toxicity_score': ensemble_scores,
        'toxicity_percentile': pd.Series(ensemble_scores).rank(pct=True),
        'is_toxic_95th': ensemble_scores > np.percentile(ensemble_scores, 95),
        'is_toxic_99th': ensemble_scores > np.percentile(ensemble_scores, 99)
    })
    
    # Add individual detector scores
    for name, scores in individual_scores.items():
        results_df[f'{name}_score'] = scores
    
    print(f"Generated toxicity scores for {len(results_df)} orders")
    print(f"Mean toxicity score: {ensemble_scores.mean():.3f}")
    print(f"95th percentile threshold: {np.percentile(ensemble_scores, 95):.3f}")
    print(f"99th percentile threshold: {np.percentile(ensemble_scores, 99):.3f}")
    
    return results_df

def analyze_toxic_patterns(results_df, features_df, top_n=100):
    """Analyze patterns in the most toxic orders"""
    
    # Get top toxic orders
    top_toxic = results_df.nlargest(top_n, 'toxicity_score')
    toxic_indices = top_toxic.index
    
    print(f"\nAnalyzing top {top_n} most toxic orders:")
    print(f"Average toxicity score: {top_toxic['toxicity_score'].mean():.3f}")
    print(f"Score range: {top_toxic['toxicity_score'].min():.3f} - {top_toxic['toxicity_score'].max():.3f}")
    
    # Feature analysis for toxic orders
    toxic_features = features_df.loc[toxic_indices]
    normal_features = features_df.drop(toxic_indices)
    
    print(f"\nToxic Order Characteristics:")
    
    # Size characteristics
    if 'order_size' in toxic_features.columns:
        toxic_size_mean = toxic_features['order_size'].mean()
        normal_size_mean = normal_features['order_size'].mean()
        size_ratio = toxic_size_mean / normal_size_mean
        
        print(f"  Average order size: {toxic_size_mean:.0f} vs {normal_size_mean:.0f} (ratio: {size_ratio:.2f}x)")
        print(f"  Large orders: {toxic_features['large_order'].mean()*100:.1f}% vs {normal_features['large_order'].mean()*100:.1f}%")
    
    # Timing patterns
    if 'arrival_rate' in toxic_features.columns:
        toxic_arrival = toxic_features['arrival_rate'].mean()
        normal_arrival = normal_features['arrival_rate'].mean()
        arrival_ratio = toxic_arrival / normal_arrival
        
        print(f"  Arrival rate: {toxic_arrival:.3f} vs {normal_arrival:.3f} (ratio: {arrival_ratio:.2f}x)")
    
    # Market impact
    if 'impact_per_size' in toxic_features.columns:
        toxic_impact = toxic_features['impact_per_size'].mean()
        normal_impact = normal_features['impact_per_size'].mean()
        
        print(f"  Price impact per size: {toxic_impact:.6f} vs {normal_impact:.6f}")
    
    # Manipulation patterns
    manipulation_features = [col for col in toxic_features.columns if any(pattern in col.lower() 
                            for pattern in ['layering', 'stuffing', 'momentum_ignition', 'wash_trading'])]
    
    if manipulation_features:
        print(f"\nManipulation Pattern Detection:")
        for feature in manipulation_features:
            toxic_rate = toxic_features[feature].mean()
            normal_rate = normal_features[feature].mean()
            if toxic_rate > normal_rate * 1.5:
                print(f"  {feature.replace('_', ' ').title()}: {toxic_rate*100:.1f}% vs {normal_rate*100:.1f}%")
    
    return toxic_features

if __name__ == "__main__":
    try:
        # Check data directory
        data_dir = "market_data"
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory '{data_dir}' not found.")
        
        # List available data files
        order_files = glob.glob(f"{data_dir}/orders_*.csv")
        lob_files = glob.glob(f"{data_dir}/lob_snapshots_*.csv")
        trade_files = glob.glob(f"{data_dir}/trades_*.csv")
        
        print("Starting toxicity detection model training with cross-validation...")
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
            raise FileNotFoundError("No order files found. Expected files like 'orders_20250621_015122.csv'")
        
        # Train the model with cross-validation
        detector, scores, metrics = main_training_pipeline_with_cv(
            data_dir=data_dir,
            n_trials=40,
            cv_folds=5
        )
        
        print("\n" + "="*80)
        print("SUCCESS: Cross-Validation Training completed!")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"File Error: {e}")
    
    except Exception as e:
        print(f"Training Error: {e}")
        import traceback
        traceback.print_exc()