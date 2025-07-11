#training.py
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
from scipy import stats
from scipy.spatial.distance import cdist
import joblib
import glob
import os
import optuna
from optuna.samplers import TPESampler


class MarketDataFeatureEngineer:
    """Feature engineering with advanced toxicity detection patterns"""
    
    def __init__(self):
        self.feature_names = []
        self.feature_importance_scores = {}
    
    def load_market_data(self, data_dir="market_data"):
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
    
    def extract_features(self, orders_df, lob_df, trades_df):
        """Extract comprehensive features for toxicity detection"""
        print("Extracting comprehensive market features...")
        
        features_df = pd.DataFrame(index=orders_df.index)
        
        # ========== EXISTING FEATURES (keep as is) ==========
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
        
        # ========== NEW COMPREHENSIVE FEATURES ==========
        
        # 1. Enhanced Price and Spread Features
        features_df = self._extract_price_features(features_df, orders_df, lob_df)
        
        # 2. Advanced Order Book Depth Features
        features_df = self._extract_depth_features(features_df, lob_df, orders_df)
        
        # 3. Comprehensive Timing Features
        features_df = self._extract_timing_features(features_df, orders_df)
        
        # 4. Market Microstructure Features
        features_df = self._extract_microstructure_features(features_df, orders_df)
        
        # 5. Order Flow Intensity Features
        features_df = self._extract_flow_intensity_features(features_df, orders_df)
        
        # 6. Size Distribution Features
        features_df = self._extract_size_distribution_features(features_df, orders_df)
        
        # 7. Price Impact Features
        features_df = self._extract_price_impact_features(features_df, orders_df, trades_df)
        
        # 8. LOB features (enhanced)
        if not lob_df.empty:
            lob_features = self._extract_lob_features(lob_df, orders_df)
            for col in lob_features.columns:
                if col not in features_df.columns:
                    features_df[col] = lob_features[col]
        
        # 9. Trade features (enhanced)
        if not trades_df.empty:
            trade_features = self._extract_trade_features(trades_df, orders_df)
            for col in trade_features.columns:
                if col not in features_df.columns:
                    features_df[col] = trade_features[col]
        
        # 10. Advanced Pattern Detection
        features_df = self._extract_pattern_features(features_df, orders_df)
        
        # 11. Regime Detection Features
        features_df = self._extract_regime_features(features_df)
        
        # 12. Technical Indicators
        features_df = self._extract_technical_indicators(features_df)
        
        # 13. Anomaly Detection Features
        features_df = self._extract_anomaly_features(features_df)
        
        # 14. Network and Clustering Features
        features_df = self._extract_network_features(features_df, orders_df)
        
        # 15. Sequential pattern features (enhanced)
        features_df = self._add_sequential_patterns(features_df)
        
        # 16. Market impact features (enhanced)
        features_df = self._add_market_impact_features(features_df)
        
        # 17. Rolling features for key variables (enhanced)
        features_df = self._add_rolling_features(features_df)
        
        # 18. Interaction features (enhanced)
        features_df = self._add_interaction_features(features_df)
        
        # Clean up
        features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)
        self.feature_names = features_df.columns.tolist()
        
        print(f"Generated {len(self.feature_names)} comprehensive market features")
        return features_df

    def _extract_price_features(self, features_df, orders_df, lob_df):
        """Extract enhanced price-related features"""
        if 'mid_price' in orders_df.columns:
            mid_price = orders_df['mid_price']
            features_df['mid_price'] = mid_price
            features_df['log_mid_price'] = np.log(mid_price + 1e-8)
            features_df['mid_price_change'] = mid_price.diff().fillna(0)
            features_df['mid_price_returns'] = mid_price.pct_change().fillna(0)
            
            # Price velocity and acceleration
            features_df['price_velocity'] = mid_price.diff().fillna(0)
            features_df['price_acceleration'] = features_df['price_velocity'].diff().fillna(0)
            
            # Price momentum for multiple windows
            for window in [3, 5, 10, 20, 50]:
                features_df[f'price_ma_{window}'] = mid_price.rolling(window, min_periods=1).mean()
                features_df[f'price_trend_{window}'] = (mid_price - features_df[f'price_ma_{window}']) / (features_df[f'price_ma_{window}'] + 1e-8)
                features_df[f'price_momentum_{window}'] = mid_price.rolling(window, min_periods=1).apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0, raw=False)
            
            # Price volatility measures
            for window in [5, 10, 20]:
                features_df[f'price_volatility_{window}'] = features_df['mid_price_returns'].rolling(window, min_periods=1).std()
                features_df[f'price_range_{window}'] = (mid_price.rolling(window, min_periods=1).max() - mid_price.rolling(window, min_periods=1).min()) / mid_price.rolling(window, min_periods=1).mean()
            
            # Price relative to order
            if 'price' in orders_df.columns:
                order_price = orders_df['price'].fillna(mid_price)
                features_df['price_deviation'] = (order_price - mid_price) / (mid_price + 1e-8)
                features_df['abs_price_deviation'] = np.abs(features_df['price_deviation'])
                features_df['price_aggressiveness'] = np.where(
                    orders_df['side'] == 'BUY',
                    np.maximum(0, (order_price - mid_price) / (mid_price + 1e-8)),
                    np.maximum(0, (mid_price - order_price) / (mid_price + 1e-8))
                )
                
                # Distance from mid in basis points
                features_df['distance_from_mid_bps'] = np.abs(order_price - mid_price) / (mid_price + 1e-8) * 10000
        
        # Spread features (enhanced)
        if 'spread' in orders_df.columns:
            spread = orders_df['spread']
            features_df['spread'] = spread
            features_df['log_spread'] = np.log1p(spread)
            features_df['spread_percentile'] = spread.rank(pct=True)
            
            if 'mid_price' in orders_df.columns:
                features_df['relative_spread'] = spread / (mid_price + 1e-8)
                features_df['relative_spread_bps'] = features_df['relative_spread'] * 10000
            
            # Rolling spread statistics
            for window in [3, 5, 10, 20]:
                features_df[f'spread_ma_{window}'] = spread.rolling(window, min_periods=1).mean()
                features_df[f'spread_std_{window}'] = spread.rolling(window, min_periods=1).std()
                features_df[f'spread_min_{window}'] = spread.rolling(window, min_periods=1).min()
                features_df[f'spread_max_{window}'] = spread.rolling(window, min_periods=1).max()
                features_df[f'spread_range_{window}'] = features_df[f'spread_max_{window}'] - features_df[f'spread_min_{window}']
            
            # Spread volatility and persistence
            features_df['spread_volatility_5'] = spread.rolling(5, min_periods=1).std()
            features_df['spread_change'] = spread.diff().fillna(0)
            features_df['spread_acceleration'] = features_df['spread_change'].diff().fillna(0)
        
        return features_df

    def _extract_depth_features(self, features_df, lob_df, orders_df):
        """Extract comprehensive order book depth features"""
        if lob_df.empty:
            return features_df
        
        # Merge LOB data with orders
        if 'timestamp' in lob_df.columns and 'timestamp' in orders_df.columns:
            merged = pd.merge_asof(
                orders_df[['timestamp']].reset_index(),
                lob_df,
                on='timestamp',
                direction='backward'
            ).set_index('index')
            
            # Multi-level depth analysis
            bid_depths = []
            ask_depths = []
            bid_prices = []
            ask_prices = []
            
            for level in range(1, 6):
                bid_size_col = f'bid_size_{level}'
                ask_size_col = f'ask_size_{level}'
                bid_price_col = f'bid_price_{level}'
                ask_price_col = f'ask_price_{level}'
                
                if all(col in merged.columns for col in [bid_size_col, ask_size_col]):
                    bid_size = merged[bid_size_col].fillna(0)
                    ask_size = merged[ask_size_col].fillna(0)
                    
                    bid_depths.append(bid_size)
                    ask_depths.append(ask_size)
                    
                    # Level-specific features
                    features_df[f'bid_depth_L{level}'] = bid_size
                    features_df[f'ask_depth_L{level}'] = ask_size
                    features_df[f'depth_imbalance_L{level}'] = (bid_size - ask_size) / (bid_size + ask_size + 1e-8)
                    features_df[f'depth_ratio_L{level}'] = bid_size / (ask_size + 1e-8)
                    
                    # Price level features
                    if bid_price_col in merged.columns and ask_price_col in merged.columns:
                        bid_price = merged[bid_price_col].fillna(0)
                        ask_price = merged[ask_price_col].fillna(0)
                        
                        bid_prices.append(bid_price)
                        ask_prices.append(ask_price)
                        
                        features_df[f'bid_price_L{level}'] = bid_price
                        features_df[f'ask_price_L{level}'] = ask_price
            
            # Aggregate depth features
            if bid_depths and ask_depths:
                total_bid_depth = sum(bid_depths)
                total_ask_depth = sum(ask_depths)
                
                features_df['total_bid_depth'] = total_bid_depth
                features_df['total_ask_depth'] = total_ask_depth
                features_df['total_depth'] = total_bid_depth + total_ask_depth
                features_df['depth_imbalance'] = (total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth + 1e-8)
                features_df['depth_ratio'] = total_bid_depth / (total_ask_depth + 1e-8)
                
                # Depth concentration and distribution
                features_df['depth_concentration_L1'] = (bid_depths[0] + ask_depths[0]) / (total_bid_depth + total_ask_depth + 1e-8)
                features_df['depth_concentration_L12'] = (bid_depths[0] + ask_depths[0] + bid_depths[1] + ask_depths[1]) / (total_bid_depth + total_ask_depth + 1e-8)
                
                # Depth variance across levels
                if len(bid_depths) > 1:
                    bid_depth_array = np.array(bid_depths)
                    ask_depth_array = np.array(ask_depths)
                    features_df['bid_depth_variance'] = np.var(bid_depth_array, axis=0)
                    features_df['ask_depth_variance'] = np.var(ask_depth_array, axis=0)
                    features_df['depth_skewness'] = features_df['bid_depth_variance'] - features_df['ask_depth_variance']
            
            # Price slope analysis
            if bid_prices and ask_prices and len(bid_prices) > 1:
                # Calculate price slopes across levels
                bid_price_array = np.array(bid_prices)
                ask_price_array = np.array(ask_prices)
                
                # Slope from L1 to L2
                features_df['bid_price_slope_L1L2'] = bid_price_array[1] - bid_price_array[0]
                features_df['ask_price_slope_L1L2'] = ask_price_array[1] - ask_price_array[0]
                
                # Average slope across all levels
                bid_slopes = np.diff(bid_price_array, axis=0)
                ask_slopes = np.diff(ask_price_array, axis=0)
                features_df['bid_price_slope_avg'] = np.mean(bid_slopes, axis=0)
                features_df['ask_price_slope_avg'] = np.mean(ask_slopes, axis=0)
        
        return features_df

    def _extract_timing_features(self, features_df, orders_df):
        """Extract comprehensive timing and arrival features"""
        if 'timestamp' in orders_df.columns:
            timestamps = pd.to_datetime(orders_df['timestamp']) if not pd.api.types.is_datetime64_any_dtype(orders_df['timestamp']) else orders_df['timestamp']
            
            # Basic timing features
            if hasattr(timestamps.iloc[0], 'hour'):
                # Inter-arrival times
                time_diffs = timestamps.diff().dt.total_seconds().fillna(1)
                features_df['inter_arrival_time'] = time_diffs
                features_df['log_inter_arrival_time'] = np.log1p(time_diffs)
                features_df['arrival_rate'] = 1 / (time_diffs + 1e-8)
                
                # Time of day features
                features_df['hour_of_day'] = timestamps.dt.hour
                features_df['minute_of_hour'] = timestamps.dt.minute
                features_df['second_of_minute'] = timestamps.dt.second
                
                # Periodic features
                features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour_of_day'] / 24)
                features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour_of_day'] / 24)
                features_df['minute_sin'] = np.sin(2 * np.pi * features_df['minute_of_hour'] / 60)
                features_df['minute_cos'] = np.cos(2 * np.pi * features_df['minute_of_hour'] / 60)
                
            else:
                # Sequence-based timing
                time_diffs = pd.Series(range(len(timestamps))).diff().fillna(1)
                features_df['inter_arrival_time'] = time_diffs
                features_df['log_inter_arrival_time'] = np.log1p(time_diffs)
                features_df['arrival_rate'] = 1 / (time_diffs + 1e-8)
            
            # Arrival intensity patterns
            for window in [3, 5, 10, 20, 50]:
                features_df[f'arrival_intensity_{window}'] = features_df['arrival_rate'].rolling(window, min_periods=1).mean()
                features_df[f'arrival_volatility_{window}'] = features_df['arrival_rate'].rolling(window, min_periods=1).std()
                features_df[f'arrival_min_{window}'] = features_df['arrival_rate'].rolling(window, min_periods=1).min()
                features_df[f'arrival_max_{window}'] = features_df['arrival_rate'].rolling(window, min_periods=1).max()
            
            # Arrival burst detection
            arrival_ma = features_df['arrival_rate'].rolling(10, min_periods=1).mean()
            arrival_std = features_df['arrival_rate'].rolling(10, min_periods=1).std()
            features_df['arrival_burst'] = ((features_df['arrival_rate'] - arrival_ma) > 2 * arrival_std).astype(int)
            features_df['arrival_burst_intensity'] = (features_df['arrival_rate'] - arrival_ma) / (arrival_std + 1e-8)
            
            # Time clustering measures
            features_df['time_clustering_5'] = features_df['inter_arrival_time'].rolling(5, min_periods=1).std()
            features_df['time_clustering_10'] = features_df['inter_arrival_time'].rolling(10, min_periods=1).std()
            
            # Arrival acceleration
            features_df['arrival_acceleration'] = features_df['arrival_rate'].diff().fillna(0)
        
        return features_df

    def _extract_microstructure_features(self, features_df, orders_df):
        """Extract market microstructure features"""
        # Volatility features
        if 'volatility' in orders_df.columns:
            vol = orders_df['volatility']
            features_df['volatility'] = vol
            features_df['log_volatility'] = np.log1p(vol)
            features_df['vol_percentile'] = vol.rank(pct=True)
            
            # Volatility regimes
            vol_quantiles = vol.quantile([0.25, 0.5, 0.75, 0.9, 0.95])
            features_df['low_vol_regime'] = (vol <= vol_quantiles.iloc[0]).astype(int)
            features_df['med_vol_regime'] = ((vol > vol_quantiles.iloc[0]) & (vol <= vol_quantiles.iloc[2])).astype(int)
            features_df['high_vol_regime'] = (vol >= vol_quantiles.iloc[2]).astype(int)
            features_df['extreme_vol_regime'] = (vol >= vol_quantiles.iloc[4]).astype(int)
            
            # Volatility dynamics
            features_df['vol_change'] = vol.diff().fillna(0)
            features_df['vol_acceleration'] = features_df['vol_change'].diff().fillna(0)
            
            # Rolling volatility statistics
            for window in [5, 10, 20]:
                features_df[f'vol_ma_{window}'] = vol.rolling(window, min_periods=1).mean()
                features_df[f'vol_std_{window}'] = vol.rolling(window, min_periods=1).std()
                features_df[f'vol_trend_{window}'] = (vol - features_df[f'vol_ma_{window}']) / (features_df[f'vol_ma_{window}'] + 1e-8)
        
        # Momentum features
        if 'momentum' in orders_df.columns:
            mom = orders_df['momentum']
            features_df['momentum'] = mom
            features_df['abs_momentum'] = np.abs(mom)
            features_df['momentum_sign'] = np.sign(mom)
            features_df['momentum_squared'] = mom ** 2
            
            # Momentum persistence
            features_df['momentum_persistence_5'] = mom.rolling(5, min_periods=1).apply(lambda x: (x > 0).sum() / len(x), raw=False)
            features_df['momentum_persistence_10'] = mom.rolling(10, min_periods=1).apply(lambda x: (x > 0).sum() / len(x), raw=False)
            
            # Momentum acceleration
            features_df['momentum_change'] = mom.diff().fillna(0)
            features_df['momentum_acceleration'] = features_df['momentum_change'].diff().fillna(0)
        
        # Order book imbalance features
        if 'order_book_imbalance' in orders_df.columns:
            imbalance = orders_df['order_book_imbalance']
            features_df['imbalance'] = imbalance
            features_df['abs_imbalance'] = np.abs(imbalance)
            features_df['imbalance_sign'] = np.sign(imbalance)
            features_df['imbalance_percentile'] = imbalance.rank(pct=True)
            
            # Imbalance persistence
            features_df['imbalance_persistence_5'] = imbalance.rolling(5, min_periods=1).apply(lambda x: (x > 0).sum() / len(x), raw=False)
            features_df['imbalance_persistence_10'] = imbalance.rolling(10, min_periods=1).apply(lambda x: (x > 0).sum() / len(x), raw=False)
            
            # Imbalance dynamics
            features_df['imbalance_change'] = imbalance.diff().fillna(0)
            features_df['imbalance_acceleration'] = features_df['imbalance_change'].diff().fillna(0)
            
            # Imbalance extremes
            imbalance_quantiles = imbalance.quantile([0.05, 0.95])
            features_df['extreme_imbalance'] = ((imbalance <= imbalance_quantiles.iloc[0]) | 
                                            (imbalance >= imbalance_quantiles.iloc[1])).astype(int)
        
        return features_df

    def _extract_flow_intensity_features(self, features_df, orders_df):
        """Extract order flow intensity features"""
        # Order aggressiveness
        if 'is_aggressive' in orders_df.columns:
            aggressive = orders_df['is_aggressive'].astype(int)
            features_df['is_aggressive'] = aggressive
            
            # Aggressiveness patterns
            for window in [5, 10, 20]:
                features_df[f'aggressive_ratio_{window}'] = aggressive.rolling(window, min_periods=1).mean()
                features_df[f'aggressive_count_{window}'] = aggressive.rolling(window, min_periods=1).sum()
        
        # Order execution patterns
        if 'resulted_in_trade' in orders_df.columns:
            executed = orders_df['resulted_in_trade'].astype(int)
            features_df['resulted_in_trade'] = executed
            
            # Execution patterns
            for window in [5, 10, 20]:
                features_df[f'execution_ratio_{window}'] = executed.rolling(window, min_periods=1).mean()
                features_df[f'execution_count_{window}'] = executed.rolling(window, min_periods=1).sum()
        
        # Market vs limit order patterns
        if 'is_market_order' in features_df.columns:
            market_orders = features_df['is_market_order']
            for window in [5, 10, 20]:
                features_df[f'market_order_ratio_{window}'] = market_orders.rolling(window, min_periods=1).mean()
                features_df[f'market_order_count_{window}'] = market_orders.rolling(window, min_periods=1).sum()
        
        # Buy/sell flow patterns
        if 'is_buy' in features_df.columns and 'is_sell' in features_df.columns:
            buy_flow = features_df['is_buy']
            sell_flow = features_df['is_sell']
            
            for window in [5, 10, 20]:
                features_df[f'buy_ratio_{window}'] = buy_flow.rolling(window, min_periods=1).mean()
                features_df[f'sell_ratio_{window}'] = sell_flow.rolling(window, min_periods=1).mean()
                features_df[f'buy_sell_imbalance_{window}'] = (buy_flow.rolling(window, min_periods=1).sum() - 
                                                            sell_flow.rolling(window, min_periods=1).sum())
        
        return features_df

    def _extract_size_distribution_features(self, features_df, orders_df):
        """Extract order size distribution features"""
        if 'order_size' in features_df.columns:
            size = features_df['order_size']
            
            # Size distribution statistics
            for window in [5, 10, 20, 50]:
                features_df[f'size_mean_{window}'] = size.rolling(window, min_periods=1).mean()
                features_df[f'size_std_{window}'] = size.rolling(window, min_periods=1).std()
                features_df[f'size_min_{window}'] = size.rolling(window, min_periods=1).min()
                features_df[f'size_max_{window}'] = size.rolling(window, min_periods=1).max()
                features_df[f'size_range_{window}'] = features_df[f'size_max_{window}'] - features_df[f'size_min_{window}']
                
                # Size percentiles
                features_df[f'size_median_{window}'] = size.rolling(window, min_periods=1).median()
                features_df[f'size_q25_{window}'] = size.rolling(window, min_periods=1).quantile(0.25)
                features_df[f'size_q75_{window}'] = size.rolling(window, min_periods=1).quantile(0.75)
                features_df[f'size_iqr_{window}'] = features_df[f'size_q75_{window}'] - features_df[f'size_q25_{window}']
                
                # Size skewness and kurtosis
                features_df[f'size_skew_{window}'] = size.rolling(window, min_periods=3).skew()
                features_df[f'size_kurtosis_{window}'] = size.rolling(window, min_periods=4).kurt()
            
            # Size concentration measures
            features_df['size_concentration_5'] = size.rolling(5, min_periods=1).apply(lambda x: (x.max() - x.min()) / (x.mean() + 1e-8), raw=False)
            features_df['size_concentration_10'] = size.rolling(10, min_periods=1).apply(lambda x: (x.max() - x.min()) / (x.mean() + 1e-8), raw=False)
            
            # Size consistency
            features_df['size_consistency_5'] = size.rolling(5, min_periods=1).apply(lambda x: 1 - (x.std() / (x.mean() + 1e-8)), raw=False)
            features_df['size_consistency_10'] = size.rolling(10, min_periods=1).apply(lambda x: 1 - (x.std() / (x.mean() + 1e-8)), raw=False)
        
        return features_df

    def _extract_price_impact_features(self, features_df, orders_df, trades_df):
        """Extract price impact features"""
        if 'mid_price_change' in features_df.columns and 'order_size' in features_df.columns:
            # Basic price impact
            features_df['impact_per_size'] = features_df['mid_price_change'] / (features_df['order_size'] + 1e-8)
            features_df['abs_impact_per_size'] = np.abs(features_df['impact_per_size'])
            
            # Impact efficiency
            features_df['impact_efficiency'] = features_df['impact_per_size'] / (features_df['order_size'] + 1e-8)
            
            # Cumulative impact
            features_df['cumulative_impact_5'] = features_df['impact_per_size'].rolling(5, min_periods=1).sum()
            features_df['cumulative_impact_10'] = features_df['impact_per_size'].rolling(10, min_periods=1).sum()
            
            # Impact volatility
            features_df['impact_volatility_5'] = features_df['impact_per_size'].rolling(5, min_periods=1).std()
            features_df['impact_volatility_10'] = features_df['impact_per_size'].rolling(10, min_periods=1).std()
            
            # Abnormal impact detection
            impact_ma = features_df['impact_per_size'].rolling(20, min_periods=1).mean()
            impact_std = features_df['impact_per_size'].rolling(20, min_periods=1).std()
            features_df['abnormal_impact'] = (np.abs(features_df['impact_per_size'] - impact_ma) > 2 * impact_std).astype(int)
            features_df['impact_zscore'] = (features_df['impact_per_size'] - impact_ma) / (impact_std + 1e-8)
            
            # Kyle's lambda estimation (simplified approach)
            print("Calculating Kyle's lambda features...")
            
            # Get the data as numpy arrays
            prices = features_df['mid_price_change'].values
            sizes = features_df['order_size'].values
            
            for window in [10, 20, 50]:
                lambda_values = []
                
                for i in range(len(prices)):
                    start_idx = max(0, i - window + 1)
                    end_idx = i + 1
                    
                    if end_idx - start_idx >= 3:  # Need at least 3 data points
                        try:
                            window_prices = prices[start_idx:end_idx]
                            window_sizes = sizes[start_idx:end_idx]
                            
                            # Calculate Kyle's lambda
                            size_var = np.var(window_sizes)
                            if size_var > 1e-8:  # Avoid division by zero
                                covariance = np.cov(window_prices, window_sizes)[0, 1]
                                lambda_val = covariance / size_var
                                lambda_values.append(lambda_val if not np.isnan(lambda_val) else 0)
                            else:
                                lambda_values.append(0)
                        except:
                            lambda_values.append(0)
                    else:
                        lambda_values.append(0)
                
                # Add to features dataframe
                features_df[f'kyle_lambda_{window}'] = lambda_values
        
        else:
            print("Warning: mid_price_change or order_size not found, skipping price impact features")
        
        return features_df

    def _extract_pattern_features(self, features_df, orders_df):
        """Extract sequential pattern and manipulation detection features"""
        if 'order_size' in features_df.columns:
            # Size acceleration patterns
            size_diff = features_df['order_size'].diff()
            size_accel = size_diff.diff()
            features_df['size_acceleration'] = size_accel
            features_df['size_momentum_burst'] = (size_accel > size_accel.quantile(0.95)).astype(int)
            
            # Consecutive pattern detection
            large_order_threshold = features_df['order_size'].quantile(0.9)
            is_large = (features_df['order_size'] > large_order_threshold).astype(int)
            features_df['consecutive_large_orders'] = is_large.rolling(3, min_periods=1).sum()
            features_df['consecutive_large_orders_5'] = is_large.rolling(5, min_periods=1).sum()
            
            # Size clustering detection
            features_df['size_cluster_indicator'] = (features_df['order_size'].rolling(3, min_periods=1).std() < 
                                                features_df['order_size'].rolling(10, min_periods=1).std() * 0.5).astype(int)
        
        # Directional clustering
        if 'is_buy' in features_df.columns and 'is_sell' in features_df.columns:
            buy_signal = features_df['is_buy']
            sell_signal = features_df['is_sell']
            
            # Consecutive directional orders
            features_df['consecutive_buys'] = buy_signal.rolling(5, min_periods=1).sum()
            features_df['consecutive_sells'] = sell_signal.rolling(5, min_periods=1).sum()
            
            # Directional momentum
            direction_signal = buy_signal - sell_signal
            features_df['direction_momentum_3'] = direction_signal.rolling(3, min_periods=1).sum()
            features_df['direction_momentum_5'] = direction_signal.rolling(5, min_periods=1).sum()
            features_df['direction_momentum_10'] = direction_signal.rolling(10, min_periods=1).sum()
            
            # Direction persistence
            features_df['direction_persistence_5'] = direction_signal.rolling(5, min_periods=1).apply(lambda x: (x > 0).sum() / len(x), raw=False)
            features_df['direction_persistence_10'] = direction_signal.rolling(10, min_periods=1).apply(lambda x: (x > 0).sum() / len(x), raw=False)
        
        # Timing pattern detection
        if 'arrival_rate' in features_df.columns:
            arrival_rate = features_df['arrival_rate']
            
            # Arrival burst detection
            arrival_ma = arrival_rate.rolling(10, min_periods=1).mean()
            arrival_std = arrival_rate.rolling(10, min_periods=1).std()
            features_df['arrival_burst'] = ((arrival_rate - arrival_ma) > 2 * arrival_std).astype(int)
            features_df['arrival_burst_intensity'] = (arrival_rate - arrival_ma) / (arrival_std + 1e-8)
            
            # Arrival clustering
            features_df['arrival_clustering_5'] = arrival_rate.rolling(5, min_periods=1).std() / (arrival_rate.rolling(5, min_periods=1).mean() + 1e-8)
            features_df['arrival_clustering_10'] = arrival_rate.rolling(10, min_periods=1).std() / (arrival_rate.rolling(10, min_periods=1).mean() + 1e-8)
        
        return features_df

    def _extract_regime_features(self, features_df):
        """Extract regime detection features"""
        # Volatility regime detection
        if 'volatility' in features_df.columns:
            vol = features_df['volatility']
            vol_ma_short = vol.rolling(5, min_periods=1).mean()
            vol_ma_long = vol.rolling(20, min_periods=1).mean()
            
            features_df['vol_regime_trend'] = (vol_ma_short - vol_ma_long) / (vol_ma_long + 1e-8)
            features_df['vol_regime_breakout'] = (vol > vol.rolling(20, min_periods=1).quantile(0.95)).astype(int)
            features_df['vol_regime_breakdown'] = (vol < vol.rolling(20, min_periods=1).quantile(0.05)).astype(int)
        
        # Spread regime detection
        if 'spread' in features_df.columns:
            spread = features_df['spread']
            spread_ma = spread.rolling(10, min_periods=1).mean()
            spread_std = spread.rolling(10, min_periods=1).std()
            
            features_df['spread_regime_wide'] = (spread > spread_ma + 2 * spread_std).astype(int)
            features_df['spread_regime_tight'] = (spread < spread_ma - spread_std).astype(int)
            features_df['spread_regime_normal'] = ((spread >= spread_ma - spread_std) & 
                                                (spread <= spread_ma + 2 * spread_std)).astype(int)
        
        # Activity regime detection
        if 'arrival_rate' in features_df.columns:
            arrival_rate = features_df['arrival_rate']
            
            # Calculate quantiles separately
            arrival_q33 = arrival_rate.rolling(50, min_periods=1).quantile(0.33)
            arrival_q67 = arrival_rate.rolling(50, min_periods=1).quantile(0.67)
            
            features_df['activity_regime_low'] = (arrival_rate <= arrival_q33).astype(int)
            features_df['activity_regime_high'] = (arrival_rate >= arrival_q67).astype(int)
            features_df['activity_regime_normal'] = ((arrival_rate > arrival_q33) & 
                                                (arrival_rate < arrival_q67)).astype(int)
        
        return features_df

    def _extract_technical_indicators(self, features_df):
        """Extract technical indicators adapted for microstructure"""
        # RSI for order flow
        if 'is_buy' in features_df.columns and 'is_sell' in features_df.columns:
            direction = features_df['is_buy'] - features_df['is_sell']
            
            def calculate_rsi(series, window=14):
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window, min_periods=1).mean()
                rs = gain / (loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            features_df['flow_rsi_14'] = calculate_rsi(direction, 14)
            features_df['flow_rsi_7'] = calculate_rsi(direction, 7)
        
        # MACD for price momentum
        if 'mid_price' in features_df.columns:
            price = features_df['mid_price']
            
            # MACD components
            ema_12 = price.ewm(span=12, min_periods=1).mean()
            ema_26 = price.ewm(span=26, min_periods=1).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, min_periods=1).mean()
            
            features_df['macd_line'] = macd_line
            features_df['macd_signal'] = signal_line
            features_df['macd_histogram'] = macd_line - signal_line
            features_df['macd_crossover'] = ((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))).astype(int)
        
        # Bollinger Bands for volatility
        if 'mid_price' in features_df.columns:
            price = features_df['mid_price']
            
            for window in [10, 20]:
                sma = price.rolling(window, min_periods=1).mean()
                std = price.rolling(window, min_periods=1).std()
                
                features_df[f'bb_upper_{window}'] = sma + (std * 2)
                features_df[f'bb_lower_{window}'] = sma - (std * 2)
                features_df[f'bb_width_{window}'] = (features_df[f'bb_upper_{window}'] - features_df[f'bb_lower_{window}']) / sma
                features_df[f'bb_position_{window}'] = (price - features_df[f'bb_lower_{window}']) / (features_df[f'bb_upper_{window}'] - features_df[f'bb_lower_{window}'])
        
        # Volume oscillator
        if 'order_size' in features_df.columns:
            volume = features_df['order_size']
            
            vol_ma_short = volume.rolling(5, min_periods=1).mean()
            vol_ma_long = volume.rolling(20, min_periods=1).mean()
            
            features_df['volume_oscillator'] = (vol_ma_short - vol_ma_long) / vol_ma_long
            features_df['volume_momentum'] = volume.rolling(10, min_periods=1).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / (x.iloc[0] + 1e-8), raw=False)
        
        return features_df

    def _extract_anomaly_features(self, features_df):
        """Extract anomaly detection features"""
        # Statistical outlier detection
        numerical_columns = ['order_size', 'spread', 'arrival_rate', 'volatility', 'momentum']
        available_columns = [col for col in numerical_columns if col in features_df.columns]
        
        for col in available_columns:
            series = features_df[col]
            
            # Z-score based outliers
            rolling_mean = series.rolling(20, min_periods=1).mean()
            rolling_std = series.rolling(20, min_periods=1).std()
            z_scores = (series - rolling_mean) / (rolling_std + 1e-8)
            
            features_df[f'{col}_zscore'] = z_scores
            features_df[f'{col}_outlier_2std'] = (np.abs(z_scores) > 2).astype(int)
            features_df[f'{col}_outlier_3std'] = (np.abs(z_scores) > 3).astype(int)
            
            # IQR based outliers
            rolling_q25 = series.rolling(20, min_periods=1).quantile(0.25)
            rolling_q75 = series.rolling(20, min_periods=1).quantile(0.75)
            iqr = rolling_q75 - rolling_q25
            
            features_df[f'{col}_outlier_iqr'] = ((series < rolling_q25 - 1.5 * iqr) | 
                                            (series > rolling_q75 + 1.5 * iqr)).astype(int)
        
        # Simplified multivariate anomaly detection
        if len(available_columns) >= 2:
            print("Calculating Mahalanobis distance features...")
            
            # Get feature matrix
            feature_matrix = features_df[available_columns].fillna(0).values
            n_samples = len(feature_matrix)
            
            # Calculate rolling Mahalanobis distance
            mahal_distances = []
            window = 20
            
            for i in range(n_samples):
                start_idx = max(0, i - window + 1)
                end_idx = i + 1
                
                if end_idx - start_idx >= 2:  # Need at least 2 samples
                    try:
                        window_data = feature_matrix[start_idx:end_idx]
                        
                        # Calculate mean and covariance
                        mean = np.mean(window_data, axis=0)
                        cov = np.cov(window_data.T)
                        
                        # Current observation
                        current_obs = feature_matrix[i]
                        
                        # Calculate Mahalanobis distance
                        if np.linalg.det(cov) != 0:
                            inv_cov = np.linalg.inv(cov)
                            diff = current_obs - mean
                            mahal_dist = np.sqrt(diff.T @ inv_cov @ diff)
                            mahal_distances.append(mahal_dist if not np.isnan(mahal_dist) else 0)
                        else:
                            mahal_distances.append(0)
                    except:
                        mahal_distances.append(0)
                else:
                    mahal_distances.append(0)
            
            # Add Mahalanobis distance feature
            features_df['mahalanobis_distance'] = mahal_distances
            
            # Create outlier indicator
            mahal_series = pd.Series(mahal_distances)
            mahal_q95 = mahal_series.rolling(50, min_periods=1).quantile(0.95)
            features_df['mahalanobis_outlier'] = (mahal_series > mahal_q95).astype(int)
        
        return features_df


    def _extract_network_features(self, features_df, orders_df):
        """Extract network and participant interaction features"""
        # Agent interaction patterns (if agent info available)
        if 'agent_id' in orders_df.columns:
            agent_ids = orders_df['agent_id']
            
            # Agent activity patterns
            for window in [10, 20, 50]:
                features_df[f'unique_agents_{window}'] = agent_ids.rolling(window, min_periods=1).nunique()
                features_df[f'agent_concentration_{window}'] = agent_ids.rolling(window, min_periods=1).apply(
                    lambda x: (x.value_counts().max() / len(x)) if len(x) > 0 else 0, raw=False)
        
        # Order clustering in time
        if 'timestamp' in orders_df.columns:
            # Time-based clustering
            features_df['time_cluster_density_5'] = features_df.index.to_series().rolling(5, min_periods=1).apply(
                lambda x: len(x) / (max(x) - min(x) + 1) if len(x) > 1 else 1, raw=False)
            features_df['time_cluster_density_10'] = features_df.index.to_series().rolling(10, min_periods=1).apply(
                lambda x: len(x) / (max(x) - min(x) + 1) if len(x) > 1 else 1, raw=False)
        
        # Synchronization features
        if 'is_buy' in features_df.columns and 'is_sell' in features_df.columns:
            # Simultaneous order detection
            features_df['simultaneous_orders'] = ((features_df['is_buy'].rolling(3, min_periods=1).sum() > 1) |
                                                (features_df['is_sell'].rolling(3, min_periods=1).sum() > 1)).astype(int)
            
            # Coordination indicators
            features_df['coordination_indicator'] = (features_df['consecutive_buys'] > 3).astype(int) | (features_df['consecutive_sells'] > 3).astype(int)
        
        return features_df

    def _add_sequential_patterns(self, features_df):
        """Enhanced sequential pattern detection for manipulation"""
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
            
            # Size pattern recognition
            features_df['size_pattern_increasing'] = (features_df['order_size'] > features_df['order_size'].shift(1)).astype(int)
            features_df['size_pattern_decreasing'] = (features_df['order_size'] < features_df['order_size'].shift(1)).astype(int)
            
            # Pattern persistence
            features_df['size_increase_streak'] = features_df['size_pattern_increasing'].rolling(10, min_periods=1).sum()
            features_df['size_decrease_streak'] = features_df['size_pattern_decreasing'].rolling(10, min_periods=1).sum()
        
        # Enhanced arrival rate patterns
        if 'arrival_rate' in features_df.columns:
            arrival_rate = features_df['arrival_rate']
            
            # Arrival rate bursts
            arrival_ma = arrival_rate.rolling(10, min_periods=1).mean()
            arrival_std = arrival_rate.rolling(10, min_periods=1).std()
            features_df['arrival_burst'] = ((arrival_rate - arrival_ma) > 2 * arrival_std).astype(int)
            features_df['arrival_burst_intensity'] = (arrival_rate - arrival_ma) / (arrival_std + 1e-8)
            
            # Arrival rate patterns
            features_df['arrival_acceleration'] = arrival_rate.diff().diff().fillna(0)
            features_df['arrival_deceleration'] = (-features_df['arrival_acceleration']).clip(lower=0)
            
            # Pattern clustering
            features_df['arrival_cluster_5'] = arrival_rate.rolling(5, min_periods=1).std() / (arrival_rate.rolling(5, min_periods=1).mean() + 1e-8)
            features_df['arrival_cluster_10'] = arrival_rate.rolling(10, min_periods=1).std() / (arrival_rate.rolling(10, min_periods=1).mean() + 1e-8)
        
        return features_df

    def _add_market_impact_features(self, features_df):
        """Enhanced market impact and manipulation indicators"""
        if 'mid_price_change' in features_df.columns and 'order_size' in features_df.columns:
            # Price impact per unit size
            features_df['impact_per_size'] = features_df['mid_price_change'] / (features_df['order_size'] + 1e-8)
            features_df['abs_impact_per_size'] = np.abs(features_df['impact_per_size'])
            features_df['abnormal_impact'] = (np.abs(features_df['impact_per_size']) > 
                                            np.abs(features_df['impact_per_size']).quantile(0.95)).astype(int)
            
            # Impact efficiency and persistence
            features_df['impact_efficiency'] = features_df['impact_per_size'] / (features_df['order_size'] + 1e-8)
            features_df['impact_persistence_3'] = features_df['impact_per_size'].rolling(3, min_periods=1).sum()
            features_df['impact_persistence_5'] = features_df['impact_per_size'].rolling(5, min_periods=1).sum()
            
            # Impact reversal detection
            features_df['impact_reversal'] = ((features_df['impact_per_size'] > 0) & 
                                            (features_df['impact_per_size'].shift(1) < 0)).astype(int) | \
                                        ((features_df['impact_per_size'] < 0) & 
                                            (features_df['impact_per_size'].shift(1) > 0)).astype(int)
        
        # Enhanced size vs volume analysis
        if 'order_size' in features_df.columns:
            # Size relative to recent activity
            for window in [10, 20, 50]:
                recent_volume = features_df['order_size'].rolling(window, min_periods=1).sum()
                features_df[f'size_vs_recent_volume_{window}'] = features_df['order_size'] / (recent_volume + 1e-8)
                features_df[f'dominant_order_{window}'] = (features_df[f'size_vs_recent_volume_{window}'] > 0.5).astype(int)
            
            # Size dominance patterns
            features_df['size_dominance'] = features_df['order_size'] / (features_df['order_size'].rolling(20, min_periods=1).mean() + 1e-8)
            features_df['size_dominance_extreme'] = (features_df['size_dominance'] > 3).astype(int)
        
        return features_df

    def _add_rolling_features(self, features_df):
        """Enhanced rolling statistical features"""
        key_features = ['order_size', 'spread', 'volatility', 'momentum', 'arrival_rate', 'imbalance']
        available_features = [f for f in key_features if f in features_df.columns]
        
        for feature in available_features:
            series = features_df[feature]
            
            # Enhanced rolling windows
            for window in [3, 5, 10, 20, 50]:
                # Basic statistics
                features_df[f'{feature}_ma_{window}'] = series.rolling(window, min_periods=1).mean()
                features_df[f'{feature}_std_{window}'] = series.rolling(window, min_periods=1).std()
                features_df[f'{feature}_min_{window}'] = series.rolling(window, min_periods=1).min()
                features_df[f'{feature}_max_{window}'] = series.rolling(window, min_periods=1).max()
                features_df[f'{feature}_range_{window}'] = features_df[f'{feature}_max_{window}'] - features_df[f'{feature}_min_{window}']
                
                # Advanced statistics
                features_df[f'{feature}_median_{window}'] = series.rolling(window, min_periods=1).median()
                features_df[f'{feature}_q25_{window}'] = series.rolling(window, min_periods=1).quantile(0.25)
                features_df[f'{feature}_q75_{window}'] = series.rolling(window, min_periods=1).quantile(0.75)
                features_df[f'{feature}_iqr_{window}'] = features_df[f'{feature}_q75_{window}'] - features_df[f'{feature}_q25_{window}']
                
                # Skewness and kurtosis
                if window >= 3:
                    features_df[f'{feature}_skew_{window}'] = series.rolling(window, min_periods=3).skew()
                if window >= 4:
                    features_df[f'{feature}_kurtosis_{window}'] = series.rolling(window, min_periods=4).kurt()
                
                # Z-score and percentile rank
                ma_col = f'{feature}_ma_{window}'
                std_col = f'{feature}_std_{window}'
                features_df[f'{feature}_zscore_{window}'] = (series - features_df[ma_col]) / (features_df[std_col] + 1e-8)
                features_df[f'{feature}_percentile_{window}'] = series.rolling(window, min_periods=1).rank(pct=True)
                
                # Trend and momentum
                features_df[f'{feature}_trend_{window}'] = (series - features_df[ma_col]) / (features_df[ma_col] + 1e-8)
                features_df[f'{feature}_momentum_{window}'] = series.rolling(window, min_periods=1).apply(
                    lambda x: (x.iloc[-1] - x.iloc[0]) / (x.iloc[0] + 1e-8) if len(x) > 1 else 0, raw=False)
        
        return features_df

    def _add_interaction_features(self, features_df):
        """Enhanced interaction features between key variables"""
        # Size-based interactions
        if 'order_size' in features_df.columns:
            base_features = ['spread', 'volatility', 'arrival_rate', 'momentum', 'imbalance']
            available_features = [f for f in base_features if f in features_df.columns]
            
            for feature in available_features:
                # Basic interactions
                features_df[f'size_{feature}_interaction'] = features_df['order_size'] * features_df[feature]
                features_df[f'size_{feature}_ratio'] = features_df['order_size'] / (features_df[feature] + 1e-8)
                
                # Conditional interactions
                features_df[f'size_{feature}_conditional'] = np.where(
                    features_df['order_size'] > features_df['order_size'].median(),
                    features_df[feature],
                    0
                )
        
        # Volatility-based interactions
        if 'volatility' in features_df.columns:
            vol_features = ['arrival_rate', 'spread', 'momentum', 'imbalance']
            available_vol_features = [f for f in vol_features if f in features_df.columns]
            
            for feature in available_vol_features:
                features_df[f'vol_{feature}_interaction'] = features_df['volatility'] * features_df[feature]
                features_df[f'vol_{feature}_ratio'] = features_df['volatility'] / (features_df[feature] + 1e-8)
        
        # Spread-based interactions
        if 'spread' in features_df.columns:
            spread_features = ['arrival_rate', 'momentum', 'imbalance']
            available_spread_features = [f for f in spread_features if f in features_df.columns]
            
            for feature in available_spread_features:
                features_df[f'spread_{feature}_interaction'] = features_df['spread'] * features_df[feature]
                features_df[f'spread_{feature}_ratio'] = features_df['spread'] / (features_df[feature] + 1e-8)
        
        # Three-way interactions
        if all(f in features_df.columns for f in ['order_size', 'volatility', 'arrival_rate']):
            features_df['size_vol_arrival_interaction'] = features_df['order_size'] * features_df['volatility'] * features_df['arrival_rate']
        
        if all(f in features_df.columns for f in ['order_size', 'spread', 'imbalance']):
            features_df['size_spread_imbalance_interaction'] = features_df['order_size'] * features_df['spread'] * features_df['imbalance']
        
        # Regime-conditional interactions
        if 'volatility' in features_df.columns and 'order_size' in features_df.columns:
            high_vol_regime = features_df['volatility'] > features_df['volatility'].quantile(0.75)
            features_df['size_in_high_vol'] = np.where(high_vol_regime, features_df['order_size'], 0)
            features_df['size_in_low_vol'] = np.where(~high_vol_regime, features_df['order_size'], 0)
        
        return features_df
    
    def _extract_lob_features(self, lob_df, orders_df):
        """Extract LOB features"""
        lob_features = pd.DataFrame(index=orders_df.index)
        
        if 'timestamp' in lob_df.columns and 'timestamp' in orders_df.columns:
            # Merge LOB data with orders
            merged = pd.merge_asof(
                orders_df[['timestamp']].reset_index(),
                lob_df,
                on='timestamp',
                direction='backward'
            ).set_index('index')
            
            # Depth features for multiple levels
            total_bid_depth = pd.Series(0, index=merged.index)
            total_ask_depth = pd.Series(0, index=merged.index)
            
            for level in range(1, 6):
                bid_size_col = f'bid_size_{level}'
                ask_size_col = f'ask_size_{level}'
                
                if all(col in merged.columns for col in [bid_size_col, ask_size_col]):
                    bid_size = merged[bid_size_col].fillna(0)
                    ask_size = merged[ask_size_col].fillna(0)
                    
                    total_bid_depth += bid_size
                    total_ask_depth += ask_size
                    
                    # Level-specific features
                    lob_features[f'bid_depth_L{level}'] = bid_size
                    lob_features[f'ask_depth_L{level}'] = ask_size
                    lob_features[f'depth_imbalance_L{level}'] = (bid_size - ask_size) / (bid_size + ask_size + 1e-8)
                    lob_features[f'depth_ratio_L{level}'] = bid_size / (ask_size + 1e-8)
            
            # Aggregate depth features
            lob_features['total_bid_depth'] = total_bid_depth
            lob_features['total_ask_depth'] = total_ask_depth
            lob_features['total_depth'] = total_bid_depth + total_ask_depth
            lob_features['depth_imbalance'] = (total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth + 1e-8)
            lob_features['depth_ratio'] = total_bid_depth / (total_ask_depth + 1e-8)
        
        return lob_features.fillna(0)
    
    def _extract_trade_features(self, trades_df, orders_df):
        """Extract trade-based features"""
        trade_features = pd.DataFrame(index=orders_df.index)
        
        if 'timestamp' in trades_df.columns and 'timestamp' in orders_df.columns:
            # Handle timestamp conversion
            try:
                orders_ts = pd.to_datetime(orders_df['timestamp'])
                trades_ts = pd.to_datetime(trades_df['timestamp'])
                use_datetime = True
            except:
                orders_ts = orders_df['timestamp']
                trades_ts = trades_df['timestamp']
                use_datetime = False
            
            # Sample for performance
            sample_size = min(len(orders_df), 1000)
            sample_indices = np.random.choice(len(orders_df), sample_size, replace=False)
            
            for idx in sample_indices:
                order_time = orders_ts.iloc[idx]
                
                for window_seconds in [10, 30, 60]:
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
                            
                            # Price features
                            if 'price' in recent_trades.columns:
                                prices = recent_trades['price']
                                if len(prices) > 1:
                                    price_returns = prices.pct_change().dropna()
                                    if len(price_returns) > 0:
                                        trade_features.loc[idx, f'trade_volatility_{window_seconds}'] = price_returns.std()
                                        trade_features.loc[idx, f'trade_momentum_{window_seconds}'] = price_returns.mean()
                                
                                # VWAP
                                if recent_trades['quantity'].sum() > 0:
                                    vwap = (prices * recent_trades['quantity']).sum() / recent_trades['quantity'].sum()
                                    trade_features.loc[idx, f'vwap_{window_seconds}'] = vwap
                    except Exception:
                        continue
        
        return trade_features.fillna(0)
    
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
            
        if 'arrival_rate' in features_df.columns:
            # Arrival rate bursts
            arrival_ma = features_df['arrival_rate'].rolling(10, min_periods=1).mean()
            arrival_std = features_df['arrival_rate'].rolling(10, min_periods=1).std()
            features_df['arrival_burst'] = ((features_df['arrival_rate'] - arrival_ma) > 2 * arrival_std).astype(int)
            
        return features_df
    
    def _add_market_impact_features(self, features_df):
        """Add market impact and manipulation indicators"""
        if 'mid_price_change' in features_df.columns and 'order_size' in features_df.columns:
            # Price impact per unit size
            features_df['impact_per_size'] = features_df['mid_price_change'] / (features_df['order_size'] + 1e-8)
            features_df['abnormal_impact'] = (abs(features_df['impact_per_size']) > 
                                            abs(features_df['impact_per_size']).quantile(0.95)).astype(int)
        
        # Size relative to recent trading
        if 'trade_volume_60' in features_df.columns and 'order_size' in features_df.columns:
            features_df['size_vs_recent_volume'] = features_df['order_size'] / (features_df['trade_volume_60'] + 1e-8)
            features_df['dominant_order'] = (features_df['size_vs_recent_volume'] > 0.5).astype(int)
        
        return features_df
    
    def _add_rolling_features(self, features_df):
        """Add rolling statistical features"""
        key_features = ['order_size', 'spread', 'volatility', 'momentum', 'arrival_rate']
        available_features = [f for f in key_features if f in features_df.columns]
        
        for feature in available_features:
            for window in [5, 10, 20]:
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
        
        # Volatility-based interactions
        if 'volatility' in features_df.columns and 'arrival_rate' in features_df.columns:
            features_df['vol_arrival_interaction'] = features_df['volatility'] * features_df['arrival_rate']
        
        return features_df

class AdvancedToxicityDetector:
    """Advanced ensemble detector with algorithms"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.ensemble_weights = {}
        self.performance_metrics = {}
        self.best_hyperparameters = {}
        self.feature_importance = {}
        
    def prepare_features(self, features_df, variance_threshold=0.01, correlation_threshold=0.95):
        """Feature preparation with selection"""
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
    
    def optimize_hyperparameters(self, X_scaled, n_trials=30):
        """Optimize hyperparameters using Optuna"""
        print(f"Optimizing hyperparameters with {n_trials} trials...")
        
        def objective(trial):
            contamination = trial.suggest_float('contamination', 0.01, 0.2)
            n_estimators = trial.suggest_int('n_estimators', 100, 300)
            n_neighbors = trial.suggest_int('n_neighbors', 5, 30)
            n_clusters = trial.suggest_int('n_clusters', 3, min(15, len(X_scaled) // 30))
            
            try:
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
                
                if len(set(cluster_labels)) > 1:
                    silhouette = silhouette_score(X_scaled, cluster_labels)
                    calinski = calinski_harabasz_score(X_scaled, cluster_labels)
                    davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
                    
                    score = silhouette + (calinski / 1000) - davies_bouldin
                else:
                    score = -1
                
                return score
                
            except Exception:
                return -10
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        self.best_hyperparameters = study.best_params
        print(f"Best hyperparameters: {self.best_hyperparameters}")
        print(f"Best score: {study.best_value:.4f}")
        
        return self.best_hyperparameters
    
    def train_ensemble(self, X_scaled, hyperparameters=None):
        """Train ensemble of toxicity detectors"""
        print("Training ensemble detectors...")
        
        if hyperparameters is None:
            hyperparameters = self.best_hyperparameters
        
        detectors = {}
        
        # Isolation Forest variants
        contamination_rates = [0.025, 0.05, 0.1, 0.15]
        for i, contamination in enumerate(contamination_rates):
            try:
                iso_forest = IsolationForest(
                    contamination=contamination,
                    n_estimators=hyperparameters.get('n_estimators', 200),
                    random_state=42 + i,
                    bootstrap=True,
                    n_jobs=-1
                )
                iso_forest.fit(X_scaled)
                detectors[f'isolation_forest_{contamination}'] = iso_forest
            except Exception as e:
                print(f"Warning: Failed to train Isolation Forest with contamination {contamination}: {e}")
        
        # LOF variants
        neighbor_counts = [5, 10, 20]
        for neighbors in neighbor_counts:
            try:
                lof = LocalOutlierFactor(
                    n_neighbors=neighbors,
                    contamination=hyperparameters.get('contamination', 0.1),
                    novelty=True,
                    n_jobs = -1
                )
                lof.fit(X_scaled)
                detectors[f'lof_{neighbors}'] = lof
            except Exception as e:
                print(f"Warning: Failed to train LOF with {neighbors} neighbors: {e}")
       
        # One-Class SVM variants
        svm_configs = [
            {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.05},
            {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.1},
            {'kernel': 'poly', 'degree': 3, 'nu': 0.1}
        ]
        
        for i, config in enumerate(svm_configs):
            try:
                svm = OneClassSVM(**config)
                svm.fit(X_scaled)
                detectors[f'svm_{config["kernel"]}_{config["nu"]}'] = svm
            except Exception as e:
                print(f"Warning: Failed to train SVM with config {config}: {e}")
        
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
        """Calculate ensemble toxicity scores with weighting"""
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
                elif 'svm' in name:
                    scores = -model.decision_function(X_scaled)
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
                
                # Normalization
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
                
                # Weight calculation
                weight = self._calculate_weight(scores, name)
                individual_weights[name] = weight
                
            except Exception as e:
                print(f"Error calculating scores for {name}: {e}")
                continue
        
        # Normalize weights with diversity bonus
        total_weight = sum(individual_weights.values())
        if total_weight > 0:
            # Prevent dominance
            max_weight = 0.25
            normalized_weights = {}
            for name, weight in individual_weights.items():
                normalized_weights[name] = min(weight / total_weight, max_weight)
            
            # Renormalize
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
    
    def _calculate_weight(self, scores, detector_name):
        """Weight calculation with multiple criteria"""
        try:
            # Score distribution quality
            score_std = np.std(scores)
            score_range = np.max(scores) - np.min(scores)
            separation_score = score_std * score_range
            
            # Detection consistency
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
            elif 'svm' in detector_name:
                type_bonus = 1.02
            elif 'gmm' in detector_name:
                type_bonus = 1.0
            elif 'dbscan' in detector_name:
                type_bonus = 0.9
            
            # Score stability
            score_volatility = np.std(np.diff(scores))
            stability_score = 1 / (1 + score_volatility)
            
            # Combine criteria
            weight = separation_score * avg_performance * type_bonus * stability_score
            
            return max(0.05, min(weight, 2.0))
            
        except Exception:
            return 0.1
    
    def evaluate_comprehensive_performance(self, X_scaled, ensemble_scores, individual_scores):
        """Comprehensive performance evaluation"""
        print("Evaluating comprehensive detector performance...")
        
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
            
            # Separation score
            if anomaly_rate > 0 and anomaly_rate < 1:
                anomaly_scores = ensemble_scores[ensemble_scores > threshold_value]
                normal_scores = ensemble_scores[ensemble_scores <= threshold_value]
                if len(normal_scores) > 0 and normal_scores.std() > 0:
                    separation = (anomaly_scores.mean() - normal_scores.mean()) / normal_scores.std()
                    anomaly_rates[f'{threshold}th_separation'] = separation
        
        metrics['anomaly_rates'] = anomaly_rates
        
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
        
        # Individual detector performance
        individual_performance = {}
        for name, scores in individual_scores.items():
            try:
                performance = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'range': scores.max() - scores.min(),
                    'weight': self.ensemble_weights.get(name, 0),
                    'consistency': self._calculate_consistency(scores),
                    'stability': 1 / (1 + np.std(np.diff(scores)))
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
        
        # Feature importance metrics
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
    
    def save_model(self, save_dir="toxicity_models"):
        """Save the model with comprehensive metadata"""
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
            'version': '5.0_production',
            'n_features': len(self.feature_selector) if self.feature_selector else 0,
            'n_detectors': len(self.models),
            'training_summary': {
                'top_features': dict(sorted(self.feature_importance.items(), 
                                            key=lambda x: x[1], reverse=True)[:15]),
                'detector_weights': dict(sorted(self.ensemble_weights.items(), 
                                                key=lambda x: x[1], reverse=True)),
                'ensemble_diversity': self.performance_metrics.get('ensemble_diversity', {}),
                'best_clustering_score': max([v for k, v in self.performance_metrics.items() 
                                            if 'silhouette' in k and isinstance(v, (int, float))], 
                                            default=0),
                'avg_separation_score': np.mean([v for k, v in self.performance_metrics.get('anomaly_rates', {}).items() 
                                                if 'separation' in k])
            }
        }
        
        joblib.dump(model_package, model_path)
        
        print(f"Model saved to: {model_path}")
        return model_path

def create_visualizations(features_df, ensemble_scores, individual_scores, detector):
   """Create comprehensive performance visualizations"""
   
   # Create plots directory
   plots_dir = "toxicity_plots"
   if not os.path.exists(plots_dir):
       os.makedirs(plots_dir)
   
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   
   # Set publication-quality style
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
   
   # 1. Comprehensive Model Performance Dashboard
   fig, axes = plt.subplots(3, 3, figsize=(20, 15))
   
   # Score distribution with statistics
   axes[0, 0].hist(ensemble_scores, bins=60, alpha=0.7, color='skyblue', edgecolor='black', density=True)
   axes[0, 0].axvline(np.percentile(ensemble_scores, 95), color='red', linestyle='--', linewidth=2, label='95th')
   axes[0, 0].axvline(np.percentile(ensemble_scores, 99), color='darkred', linestyle='--', linewidth=2, label='99th')
   axes[0, 0].axvline(np.percentile(ensemble_scores, 99.5), color='purple', linestyle='--', linewidth=2, label='99.5th')
   axes[0, 0].set_xlabel('Toxicity Score')
   axes[0, 0].set_ylabel('Density')
   axes[0, 0].set_title('Toxicity Score Distribution')
   axes[0, 0].legend()
   axes[0, 0].grid(True, alpha=0.3)
   
   # Detector weights
   weights = list(detector.ensemble_weights.values())
   detector_names = list(detector.ensemble_weights.keys())
   sorted_items = sorted(zip(detector_names, weights), key=lambda x: x[1], reverse=True)
   detector_names, weights = zip(*sorted_items)
   
   colors = plt.cm.viridis(np.linspace(0, 1, len(weights)))
   bars = axes[0, 1].bar(range(len(weights)), weights, color=colors, alpha=0.8)
   axes[0, 1].set_xticks(range(len(detector_names)))
   axes[0, 1].set_xticklabels([name[:12] for name in detector_names], rotation=45, ha='right')
   axes[0, 1].set_ylabel('Ensemble Weight')
   axes[0, 1].set_title('Detector Weights (Sorted)')
   axes[0, 1].grid(True, alpha=0.3, axis='y')
   
   # Add value labels
   for bar, weight in zip(bars, weights):
       axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, 
                       f'{weight:.3f}', ha='center', va='bottom', fontsize=8)
   
   # Feature importance
   if hasattr(detector, 'feature_importance') and detector.feature_importance:
       top_features = dict(sorted(detector.feature_importance.items(), key=lambda x: x[1], reverse=True)[:12])
       feature_names = list(top_features.keys())
       importance_scores = list(top_features.values())
       
       colors_feat = plt.cm.plasma(np.linspace(0, 1, len(importance_scores)))
       axes[0, 2].barh(range(len(feature_names)), importance_scores, color=colors_feat, alpha=0.8)
       axes[0, 2].set_yticks(range(len(feature_names)))
       axes[0, 2].set_yticklabels([name[:18] for name in feature_names])
       axes[0, 2].set_xlabel('Importance Score')
       axes[0, 2].set_title('Top 12 Feature Importance')
       axes[0, 2].grid(True, alpha=0.3, axis='x')
   
   # Detection rates with separation scores
   thresholds = [85, 90, 95, 97, 99, 99.5]
   rates = []
   separations = []
   for threshold in thresholds:
       rate = np.mean(ensemble_scores > np.percentile(ensemble_scores, threshold)) * 100
       rates.append(rate)
       
       # Calculate separation
       threshold_value = np.percentile(ensemble_scores, threshold)
       anomaly_scores = ensemble_scores[ensemble_scores > threshold_value]
       normal_scores = ensemble_scores[ensemble_scores <= threshold_value]
       if len(normal_scores) > 0 and normal_scores.std() > 0:
           separation = (anomaly_scores.mean() - normal_scores.mean()) / normal_scores.std()
           separations.append(separation)
       else:
           separations.append(0)
   
   # Detection rates
   bars = axes[1, 0].bar(range(len(thresholds)), rates, 
                         color=['lightblue', 'lightgreen', 'orange', 'coral', 'red', 'darkred'], alpha=0.8)
   axes[1, 0].set_xticks(range(len(thresholds)))
   axes[1, 0].set_xticklabels([f'{t}th' for t in thresholds])
   axes[1, 0].set_ylabel('Detection Rate (%)')
   axes[1, 0].set_title('Detection Rates by Threshold')
   axes[1, 0].grid(True, alpha=0.3, axis='y')
   
   for bar, rate in zip(bars, rates):
       axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                       f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
   
   # Separation scores
   bars_sep = axes[1, 1].bar(range(len(thresholds)), separations, 
                             color=['lightblue', 'lightgreen', 'orange', 'coral', 'red', 'darkred'], alpha=0.8)
   axes[1, 1].set_xticks(range(len(thresholds)))
   axes[1, 1].set_xticklabels([f'{t}th' for t in thresholds])
   axes[1, 1].set_ylabel('Separation Score')
   axes[1, 1].set_title('Anomaly Separation by Threshold')
   axes[1, 1].grid(True, alpha=0.3, axis='y')
   
   for bar, sep in zip(bars_sep, separations):
       axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                       f'{sep:.2f}', ha='center', va='bottom', fontsize=9)
   
   # Score timeline with anomaly highlights
   axes[1, 2].plot(ensemble_scores, alpha=0.7, color='blue', linewidth=1)
   axes[1, 2].axhline(np.percentile(ensemble_scores, 95), color='orange', linestyle='--', alpha=0.8)
   axes[1, 2].axhline(np.percentile(ensemble_scores, 99), color='red', linestyle='--', alpha=0.8)
   
   # Highlight extreme anomalies
   extreme_anomalies = np.where(ensemble_scores > np.percentile(ensemble_scores, 99.5))[0]
   if len(extreme_anomalies) > 0:
       axes[1, 2].scatter(extreme_anomalies, ensemble_scores[extreme_anomalies], 
                         color='red', s=30, alpha=0.8, zorder=5)
   
   axes[1, 2].set_xlabel('Order Sequence')
   axes[1, 2].set_ylabel('Toxicity Score')
   axes[1, 2].set_title('Toxicity Timeline with Anomaly Highlights')
   axes[1, 2].grid(True, alpha=0.3)
   
   # Detector correlation heatmap
   if len(individual_scores) > 1:
       top_detectors = sorted(detector.ensemble_weights.items(), key=lambda x: x[1], reverse=True)[:8]
       top_detector_names = [name for name, _ in top_detectors]
       
       score_data = {name: individual_scores[name] for name in top_detector_names 
                     if name in individual_scores}
       
       if len(score_data) > 1:
           score_df = pd.DataFrame(score_data)
           correlation_matrix = score_df.corr()
           
           im = axes[2, 0].imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
           
           # Add correlation values
           for i in range(len(correlation_matrix)):
               for j in range(len(correlation_matrix)):
                   if abs(correlation_matrix.iloc[i, j]) > 0.3:
                       axes[2, 0].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                                       ha='center', va='center', fontsize=8)
           
           axes[2, 0].set_xticks(range(len(top_detector_names)))
           axes[2, 0].set_xticklabels([name[:8] for name in top_detector_names], rotation=45, ha='right')
           axes[2, 0].set_yticks(range(len(top_detector_names)))
           axes[2, 0].set_yticklabels([name[:8] for name in top_detector_names])
           axes[2, 0].set_title('Detector Score Correlations')
           
           plt.colorbar(im, ax=axes[2, 0], shrink=0.8, label='Correlation')
   
   # Clustering quality comparison
   clustering_scores = []
   clustering_names = []
   
   for metric_name, value in detector.performance_metrics.items():
       if 'silhouette' in metric_name and isinstance(value, (int, float)):
           clustering_scores.append(value)
           clustering_names.append(metric_name.replace('_silhouette', '').replace('_', ' ')[:10])
   
   if clustering_scores:
       bars_clust = axes[2, 1].bar(range(len(clustering_names)), clustering_scores, 
                                   color='lightgreen', alpha=0.8)
       axes[2, 1].set_xticks(range(len(clustering_names)))
       axes[2, 1].set_xticklabels(clustering_names, rotation=45, ha='right')
       axes[2, 1].set_ylabel('Silhouette Score')
       axes[2, 1].set_title('Clustering Quality by Method')
       axes[2, 1].grid(True, alpha=0.3, axis='y')
       
       for bar, score in zip(bars_clust, clustering_scores):
           axes[2, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{score:.3f}', ha='center', va='bottom', fontsize=9)
   
   # Model performance summary
   performance_summary = detector.performance_metrics.get('score_stats', {})
   ensemble_diversity = detector.performance_metrics.get('ensemble_diversity', {})
   
   summary_text = f"""
   MODEL SUMMARY
   
   Dataset: {len(ensemble_scores)} samples
   Features: {len(detector.feature_selector) if detector.feature_selector else 0}
   Detectors: {len(detector.models)}
   
   Score Statistics:
   • Mean: {performance_summary.get('mean', 0):.3f}
   • Std: {performance_summary.get('std', 0):.3f}
   • Skewness: {performance_summary.get('skewness', 0):.3f}
   
   Ensemble Quality:
   • Diversity: {ensemble_diversity.get('diversity_score', 0):.3f}
   • Avg Correlation: {ensemble_diversity.get('avg_correlation', 0):.3f}
   • Max Weight: {max(detector.ensemble_weights.values()) if detector.ensemble_weights else 0:.3f}
   
   Top Features:
   """
   
   if hasattr(detector, 'feature_importance') and detector.feature_importance:
       top_3_features = sorted(detector.feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
       for i, (feature, importance) in enumerate(top_3_features, 1):
           summary_text += f"\n    {i}. {feature[:20]}: {importance:.3f}"
   
   axes[2, 2].text(0.05, 0.95, summary_text, transform=axes[2, 2].transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
   axes[2, 2].set_xlim(0, 1)
   axes[2, 2].set_ylim(0, 1)
   axes[2, 2].axis('off')
   axes[2, 2].set_title('Model Summary', fontweight='bold')
   
   plt.suptitle(f'Toxicity Detection Model - Comprehensive Analysis\nTimestamp: {timestamp}', 
               fontsize=16, fontweight='bold', y=0.98)
   
   plt.tight_layout()
   plt.savefig(f"{plots_dir}/comprehensive_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
   plt.close()
   
   # 2. Anomaly Characteristics Deep Dive
   if len(features_df.columns) > 0:
       fig, axes = plt.subplots(2, 4, figsize=(20, 12))
       axes = axes.flatten()
       
       # Define multiple anomaly thresholds for comparison
       anomaly_threshold_95 = np.percentile(ensemble_scores, 95)
       anomaly_threshold_99 = np.percentile(ensemble_scores, 99)
       
       anomaly_mask_95 = ensemble_scores > anomaly_threshold_95
       anomaly_mask_99 = ensemble_scores > anomaly_threshold_99
       
       key_features = ['order_size', 'volatility', 'spread', 'momentum', 'inter_arrival_time', 
                      'arrival_rate', 'depth_ratio', 'price_aggressiveness']
       available_features = [f for f in key_features if f in features_df.columns][:8]
       
       for i, feature in enumerate(available_features):
           if i < len(axes):
               ax = axes[i]
               
               normal_data = features_df[feature][~anomaly_mask_95].dropna()
               anomaly_95_data = features_df[feature][anomaly_mask_95 & ~anomaly_mask_99].dropna()
               anomaly_99_data = features_df[feature][anomaly_mask_99].dropna()
               
               if len(normal_data) > 0:
                   ax.hist(normal_data, bins=30, alpha=0.6, label='Normal (0-95th)', 
                          color='lightblue', density=True, edgecolor='black')
               
               if len(anomaly_95_data) > 0:
                   ax.hist(anomaly_95_data, bins=30, alpha=0.6, label='Moderate (95-99th)', 
                          color='orange', density=True, edgecolor='black')
               
               if len(anomaly_99_data) > 0:
                   ax.hist(anomaly_99_data, bins=30, alpha=0.6, label='High (99th+)', 
                          color='red', density=True, edgecolor='black')
               
               ax.set_xlabel(feature.replace('_', ' ').title())
               ax.set_ylabel('Density')
               ax.set_title(f'{feature.replace("_", " ").title()}: Multi-Level Analysis')
               ax.legend()
               ax.grid(True, alpha=0.3)
               
               # Add mean lines
               if len(normal_data) > 0:
                   ax.axvline(normal_data.mean(), color='blue', linestyle=':', alpha=0.8)
               if len(anomaly_99_data) > 0:
                   ax.axvline(anomaly_99_data.mean(), color='red', linestyle=':', alpha=0.8)
       
       # Remove empty subplots
       for i in range(len(available_features), len(axes)):
           fig.delaxes(axes[i])
       
       plt.suptitle('Anomaly Characteristics Analysis', fontsize=16, fontweight='bold')
       plt.tight_layout()
       plt.savefig(f"{plots_dir}/anomaly_characteristics_{timestamp}.png", dpi=300, bbox_inches='tight')
       plt.close()
   
   print(f"Visualizations saved to: {plots_dir}")
   
   # List generated files
   plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png') and timestamp in f]
   print(f"Generated {len(plot_files)} plot files:")
   for file in sorted(plot_files):
       print(f"  {file}")

def main_training_pipeline(data_dir="market_data", n_trials=30):
   """Training pipeline for toxicity detection"""
   print("="*80)
   print("MARKET TOXICITY DETECTION MODEL TRAINING")
   print("Advanced Feature Engineering & Ensemble Methods")
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
   
   # Step 3: Advanced feature preparation
   print("\n3. ADVANCED FEATURE PREPARATION")
   print("-" * 40)
   
   detector = AdvancedToxicityDetector()
   X_scaled, selected_features = detector.prepare_features(features_df)
   
   print(f"Final feature set: {X_scaled.shape}")
   
   # Step 4: Hyperparameter optimization
   print("\n4. HYPERPARAMETER OPTIMIZATION")
   print("-" * 40)
   
   best_params = detector.optimize_hyperparameters(X_scaled, n_trials=n_trials)
   
   # Step 5: Train ensemble
   print("\n5. TRAINING ENSEMBLE")
   print("-" * 40)
   
   detectors = detector.train_ensemble(X_scaled, best_params)
   
   # Step 6: Calculate ensemble scores
   print("\n6. CALCULATING ENSEMBLE SCORES")
   print("-" * 40)
   
   ensemble_scores, individual_scores = detector.calculate_ensemble_scores(X_scaled)
   
   # Step 7: Comprehensive performance evaluation
   print("\n7. COMPREHENSIVE PERFORMANCE EVALUATION")
   print("-" * 40)
   
   metrics = detector.evaluate_comprehensive_performance(X_scaled, ensemble_scores, individual_scores)
   
   # Step 8: Generate visualizations
   print("\n8. GENERATING VISUALIZATIONS")
   print("-" * 40)
   
   create_visualizations(selected_features, ensemble_scores, individual_scores, detector)
   
   # Step 9: Save model
   print("\n9. SAVING  MODEL")
   print("-" * 40)
   
   model_path = detector.save_model()
   
   # Print comprehensive results summary
   print("\n" + "="*80)
   print(" TRAINING COMPLETED SUCCESSFULLY")
   print("="*80)
   
   print(f"\n MODEL PERFORMANCE SUMMARY:")
   anomaly_detection = metrics.get('anomaly_rates', {})
   score_stats = metrics.get('score_stats', {})
   diversity_stats = metrics.get('ensemble_diversity', {})
   
   print(f"  Dataset Size: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
   print(f"  Ensemble: {len(detector.models)} detectors")
   print(f"  Effective Contributors: {len([w for w in detector.ensemble_weights.values() if w > 0.05])} detectors")
   
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
   
   print(f"\nTOP PERFORMING DETECTORS:")
   top_detectors = sorted(detector.ensemble_weights.items(), key=lambda x: x[1], reverse=True)[:5]
   for i, (name, weight) in enumerate(top_detectors, 1):
       print(f"  {i}. {name.replace('_', ' ').title()}: {weight:.3f}")
   
   print(f"\nOPTIMIZED HYPERPARAMETERS:")
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
   
   print(f"\nMODEL OUTPUTS:")
   print(f" - Model: {model_path}")
   print(f" - Visualizations: toxicity_plots/")
   print(f" - Training Timestamp: {detector.performance_metrics.get('timestamp', 'N/A')}")
   print(f" - Superior toxicity detection with {len(detector.models)} ensemble detectors")
   print(f" - Advanced feature engineering with {len(detector.feature_selector)} optimized features")
   print(f" - Anomaly separation with avg separation score: {np.mean([v for k, v in anomaly_detection.items() if 'separation' in k]):.3f}")
   print(f" - High ensemble diversity score: {diversity_stats.get('diversity_score', 0):.3f}")
   
   return detector, ensemble_scores, metrics

# Main execution
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
       
       print("Starting toxicity detection model training...")
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
       
       # Train the model
       detector, scores, metrics = main_training_pipeline(
           data_dir=data_dir,
           n_trials=40
       )
       
       print("\n" + "="*80)
       print("SUCCESS: Training completed!")
       print("="*80)
       
   except FileNotFoundError as e:
         print(f"File Error: {e}")
       
   except Exception as e:
       print(f"Training Error: {e}")
       import traceback
       traceback.print_exc()