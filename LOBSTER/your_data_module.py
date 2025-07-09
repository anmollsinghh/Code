"""
LOBSTER Data Loading and Feature Extraction Module
Integrates with the hyperparameter-tuned toxicity detection pipeline
Handles real LOBSTER market data files for production use
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from numba import jit, njit
import os
from datetime import datetime
import time

# Optimised numba functions for speed
@njit
def fast_rolling_mean(arr, window):
    """Fast rolling mean calculation"""
    n = len(arr)
    result = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        result[i] = np.mean(arr[start:i+1])
    return result

@njit
def fast_rolling_std(arr, window):
    """Fast rolling standard deviation"""
    n = len(arr)
    result = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        if i - start + 1 > 1:
            result[i] = np.std(arr[start:i+1])
        else:
            result[i] = 0.0
    return result

@njit
def fast_pct_change(arr, periods=1):
    """Fast percentage change calculation"""
    n = len(arr)
    result = np.zeros(n)
    for i in range(periods, n):
        if arr[i-periods] != 0:
            result[i] = (arr[i] - arr[i-periods]) / arr[i-periods]
    return result

@njit
def fast_zscore(arr, window):
    """Fast rolling z-score calculation"""
    n = len(arr)
    result = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        window_data = arr[start:i+1]
        if len(window_data) > 1:
            mean_val = np.mean(window_data)
            std_val = np.std(window_data)
            if std_val > 1e-8:
                result[i] = (arr[i] - mean_val) / std_val
    return result

class LOBSTERDataLoader:
    """LOBSTER data loading with optimized processing for large files"""
    
    def __init__(self, chunk_size=10000, max_rows=None):
        self.chunk_size = chunk_size
        self.max_rows = max_rows
        
    def load_orderbook_data(self, orderbook_file):
        """Load LOBSTER orderbook data efficiently"""
        print(f"Loading orderbook data from {orderbook_file}...")
        
        try:
            # Check file size and estimate rows
            file_size = os.path.getsize(orderbook_file)
            print(f"Orderbook file size: {file_size / (1024*1024):.1f} MB")
            
            # Load with chunking for large files
            if file_size > 100 * 1024 * 1024:  # >100MB
                print("Large file detected, using chunked loading...")
                chunks = []
                
                chunk_iter = pd.read_csv(
                    orderbook_file, 
                    chunksize=self.chunk_size, 
                    header=None, 
                    low_memory=False,
                    nrows=self.max_rows
                )
                
                for i, chunk in enumerate(chunk_iter):
                    # Check if first chunk has headers
                    if i == 0:
                        first_row = chunk.iloc[0].astype(str)
                        if any('Price' in str(val) or 'Size' in str(val) for val in first_row.values[:8]):
                            chunk = chunk.iloc[1:]
                    
                    # Convert to numeric with error handling
                    chunk = chunk.apply(pd.to_numeric, errors='coerce').fillna(0)
                    chunks.append(chunk)
                    
                    if i % 50 == 0:
                        print(f"  Processed {(i+1) * self.chunk_size:,} rows...")
                    
                    # Stop if we have enough data
                    if self.max_rows and len(chunks) * self.chunk_size >= self.max_rows:
                        break
                
                lob_df = pd.concat(chunks, ignore_index=True)
            else:
                # Small file, load directly
                lob_df = pd.read_csv(orderbook_file, header=None, low_memory=False, nrows=self.max_rows)
                
                # Check for headers
                first_row = lob_df.iloc[0].astype(str)
                if any('Price' in str(val) or 'Size' in str(val) for val in first_row.values[:8]):
                    lob_df = lob_df.iloc[1:]
                
                # Convert to numeric
                lob_df = lob_df.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Create proper column names (40 columns for 10 levels)
            lob_columns = []
            for i in range(1, 11):
                lob_columns.extend([
                    f'Ask_Price_{i}', f'Ask_Size_{i}', 
                    f'Bid_Price_{i}', f'Bid_Size_{i}'
                ])
            
            # Ensure we have the right number of columns
            n_cols = min(len(lob_df.columns), len(lob_columns))
            lob_df = lob_df.iloc[:, :n_cols]
            lob_df.columns = lob_columns[:n_cols]
            
            # Add timestamp
            lob_df['timestamp'] = np.arange(len(lob_df), dtype='int32')
            
            print(f"Loaded {len(lob_df):,} LOB snapshots with {n_cols} price/size columns")
            
            return lob_df
            
        except Exception as e:
            print(f"Error loading orderbook data: {e}")
            raise
    
    def load_message_data(self, message_file):
        """Load LOBSTER message data efficiently"""
        print(f"Loading message data from {message_file}...")
        
        try:
            # Define expected columns and dtypes
            expected_columns = ['Time', 'Type', 'Order ID', 'Size', 'Price', 'Direction']
            dtypes = {
                'Time': 'int64',
                'Type': 'int8', 
                'Order ID': 'int32',
                'Size': 'int32',
                'Price': 'float32',
                'Direction': 'int8'
            }
            
            # Load data
            orders_df = pd.read_csv(
                message_file, 
                low_memory=False,
                nrows=self.max_rows
            )
            
            print(f"Loaded {len(orders_df):,} order messages")
            print(f"Columns: {list(orders_df.columns)}")
            
            # Ensure proper data types
            for col in orders_df.columns:
                if col in dtypes:
                    orders_df[col] = pd.to_numeric(orders_df[col], errors='coerce').fillna(0)
                    if dtypes[col] in ['int8', 'int32', 'int64']:
                        orders_df[col] = orders_df[col].astype(dtypes[col])
                    else:
                        orders_df[col] = orders_df[col].astype(dtypes[col])
            
            # Create timestamp
            orders_df['timestamp'] = np.arange(len(orders_df), dtype='int32')
            
            # Standardize column names
            column_mapping = {
                'Time': 'time',
                'Type': 'type',
                'Order ID': 'order_id', 
                'Size': 'quantity',
                'Price': 'price',
                'Direction': 'direction'
            }
            
            orders_df.rename(columns=column_mapping, inplace=True)
            
            # Create order features
            orders_df['order_type'] = np.where(
                orders_df['type'].isin([1, 2, 3]), 'LIMIT', 'MARKET'
            )
            orders_df['side'] = np.where(
                orders_df['direction'] == 1, 'BUY', 'SELL'
            )
            
            print(f"Message data processed: {len(orders_df):,} orders")
            
            return orders_df
            
        except Exception as e:
            print(f"Error loading message data: {e}")
            raise


class LOBSTERFeatureExtractor:
    """Extract comprehensive features from LOBSTER data for toxicity detection"""
    
    def __init__(self):
        self.feature_names = []
    
    def preprocess_data(self, orders_df, lob_df):
        """Preprocess and align LOBSTER data"""
        print("Preprocessing LOBSTER data...")
        
        # Calculate basic LOB features
        if len(lob_df) > 0 and 'Ask_Price_1' in lob_df.columns and 'Bid_Price_1' in lob_df.columns:
            # Mid price and spread
            valid_mask = (lob_df['Ask_Price_1'] > 0) & (lob_df['Bid_Price_1'] > 0)
            
            lob_df['mid_price'] = 0.0
            lob_df['spread'] = 0.0
            
            if valid_mask.any():
                lob_df.loc[valid_mask, 'mid_price'] = (
                    lob_df.loc[valid_mask, 'Ask_Price_1'] + lob_df.loc[valid_mask, 'Bid_Price_1']
                ) / 2
                lob_df.loc[valid_mask, 'spread'] = (
                    lob_df.loc[valid_mask, 'Ask_Price_1'] - lob_df.loc[valid_mask, 'Bid_Price_1']
                )
            
            # Order book imbalance (first 5 levels)
            bid_cols = [f'Bid_Size_{i}' for i in range(1, 6) if f'Bid_Size_{i}' in lob_df.columns]
            ask_cols = [f'Ask_Size_{i}' for i in range(1, 6) if f'Ask_Size_{i}' in lob_df.columns]
            
            if bid_cols and ask_cols:
                total_bid = lob_df[bid_cols].sum(axis=1)
                total_ask = lob_df[ask_cols].sum(axis=1)
                total_volume = total_bid + total_ask
                
                lob_df['order_book_imbalance'] = np.where(
                    total_volume > 0,
                    (total_bid - total_ask) / total_volume,
                    0
                )
            else:
                lob_df['order_book_imbalance'] = 0.0
        
        # Align LOB features with orders using efficient merge
        print("Aligning LOB features with orders...")
        
        if len(lob_df) > 0 and len(orders_df) > 0:
            # Use searchsorted for fast alignment
            lob_times = lob_df['timestamp'].values
            order_times = orders_df['timestamp'].values
            
            # Find closest LOB timestamp for each order
            indices = np.searchsorted(lob_times, order_times, side='right') - 1
            indices = np.clip(indices, 0, len(lob_df) - 1)
            
            # Add LOB features to orders
            orders_df['mid_price'] = lob_df['mid_price'].iloc[indices].values
            orders_df['spread'] = lob_df['spread'].iloc[indices].values
            orders_df['order_book_imbalance'] = lob_df['order_book_imbalance'].iloc[indices].values
        else:
            # Default values if no LOB data
            orders_df['mid_price'] = 0.0
            orders_df['spread'] = 0.0
            orders_df['order_book_imbalance'] = 0.0
        
        # Calculate volatility and momentum
        if 'mid_price' in orders_df.columns:
            mid_price = orders_df['mid_price'].values
            valid_mask = mid_price > 0
            
            orders_df['volatility'] = 0.0
            orders_df['momentum'] = 0.0
            
            if valid_mask.any() and len(mid_price) > 20:
                vol_values = fast_rolling_std(mid_price, 20)
                mom_values = fast_pct_change(mid_price, 5)
                
                orders_df['volatility'] = vol_values
                orders_df['momentum'] = mom_values
        
        print(f"Preprocessing completed: {len(orders_df):,} orders with LOB features")
        
        return orders_df, lob_df
    
    def extract_comprehensive_features(self, orders_df, lob_df):
        """Extract comprehensive features optimized for the trained model"""
        print("Extracting comprehensive features for toxicity detection...")
        
        features_df = pd.DataFrame(index=orders_df.index)
        
        # 1. Basic Order Features
        print("  Computing basic order features...")
        features_df['order_size'] = orders_df['quantity'].astype('float32')
        features_df['log_order_size'] = np.log1p(orders_df['quantity']).astype('float32')
        features_df['sqrt_order_size'] = np.sqrt(orders_df['quantity']).astype('float32')
        
        # Order size regimes (quantile-based)
        if len(orders_df) > 0:
            try:
                size_quantiles = np.percentile(orders_df['quantity'], [80, 90, 95, 99])
                features_df['large_order'] = (orders_df['quantity'] >= size_quantiles[0]).astype('int8')
                features_df['very_large_order'] = (orders_df['quantity'] >= size_quantiles[1]).astype('int8')
                features_df['extreme_order'] = (orders_df['quantity'] >= size_quantiles[2]).astype('int8')
                features_df['massive_order'] = (orders_df['quantity'] >= size_quantiles[3]).astype('int8')
            except:
                features_df['large_order'] = 0
                features_df['very_large_order'] = 0
                features_df['extreme_order'] = 0
                features_df['massive_order'] = 0
        
        # Order type features
        features_df['is_market_order'] = (orders_df['order_type'] == 'MARKET').astype('int8')
        features_df['is_limit_order'] = (orders_df['order_type'] == 'LIMIT').astype('int8')
        features_df['is_buy'] = (orders_df['side'] == 'BUY').astype('int8')
        features_df['is_sell'] = (orders_df['side'] == 'SELL').astype('int8')
        
        # 2. Price and Spread Features
        print("  Computing price and spread features...")
        if 'mid_price' in orders_df.columns:
            mid_price = orders_df['mid_price'].values
            features_df['mid_price'] = mid_price.astype('float32')
            
            if len(mid_price) > 1 and np.any(mid_price > 0):
                features_df['log_mid_price'] = np.log1p(mid_price).astype('float32')
                features_df['mid_price_change'] = fast_pct_change(mid_price, 1).astype('float32')
                features_df['mid_price_returns'] = fast_pct_change(mid_price, 1).astype('float32')
                
                # Rolling price features
                for window in [5, 10, 20]:
                    if len(mid_price) >= window:
                        features_df[f'price_ma_{window}'] = fast_rolling_mean(mid_price, window).astype('float32')
                        features_df[f'price_std_{window}'] = fast_rolling_std(mid_price, window).astype('float32')
                        
                        # Price deviation from moving average
                        ma_values = fast_rolling_mean(mid_price, window)
                        features_df[f'price_deviation_{window}'] = ((mid_price - ma_values) / (ma_values + 1e-8)).astype('float32')
            else:
                # Default values for invalid prices
                features_df['log_mid_price'] = 0.0
                features_df['mid_price_change'] = 0.0
                features_df['mid_price_returns'] = 0.0
        
        # Spread features
        if 'spread' in orders_df.columns:
            spread = orders_df['spread'].values
            features_df['spread'] = spread.astype('float32')
            features_df['log_spread'] = np.log1p(spread).astype('float32')
            
            if 'mid_price' in orders_df.columns and np.any(orders_df['mid_price'] > 0):
                features_df['relative_spread'] = (spread / (orders_df['mid_price'] + 1e-8)).astype('float32')
                features_df['spread_bps'] = (spread / (orders_df['mid_price'] + 1e-8) * 10000).astype('float32')
        
        # 3. Timing Features
        print("  Computing timing features...")
        if len(orders_df) > 1:
            # Inter-arrival times
            time_diffs = np.diff(orders_df['timestamp'].values, prepend=orders_df['timestamp'].iloc[0])
            time_diffs = np.maximum(time_diffs, 1)  # Avoid zero
            
            features_df['inter_arrival_time'] = time_diffs.astype('float32')
            features_df['log_inter_arrival_time'] = np.log1p(time_diffs).astype('float32')
            features_df['arrival_rate'] = (1 / time_diffs).astype('float32')
            
            # Rolling arrival patterns
            arrival_rate = features_df['arrival_rate'].values
            for window in [10, 20, 50]:
                if len(arrival_rate) >= window:
                    features_df[f'arrival_ma_{window}'] = fast_rolling_mean(arrival_rate, window).astype('float32')
                    features_df[f'arrival_std_{window}'] = fast_rolling_std(arrival_rate, window).astype('float32')
                    
                    # Arrival rate bursts
                    ma_vals = fast_rolling_mean(arrival_rate, window)
                    std_vals = fast_rolling_std(arrival_rate, window)
                    features_df[f'arrival_burst_{window}'] = ((arrival_rate - ma_vals) / (std_vals + 1e-8)).astype('float32')
        
        # 4. Market Microstructure Features
        print("  Computing microstructure features...")
        
        # Volatility features
        if 'volatility' in orders_df.columns:
            vol = orders_df['volatility'].values
            features_df['volatility'] = vol.astype('float32')
            features_df['log_volatility'] = np.log1p(vol).astype('float32')
            features_df['volatility_percentile'] = pd.Series(vol).rank(pct=True).astype('float32')
            
            # Volatility regimes
            if len(vol) > 0 and np.any(vol > 0):
                vol_quantiles = np.percentile(vol[vol > 0], [33, 67, 90])
                features_df['low_vol_regime'] = (vol <= vol_quantiles[0]).astype('int8')
                features_df['high_vol_regime'] = (vol >= vol_quantiles[1]).astype('int8')
                features_df['extreme_vol_regime'] = (vol >= vol_quantiles[2]).astype('int8')
        
        # Momentum features
        if 'momentum' in orders_df.columns:
            mom = orders_df['momentum'].values
            features_df['momentum'] = mom.astype('float32')
            features_df['abs_momentum'] = np.abs(mom).astype('float32')
            features_df['momentum_sign'] = np.sign(mom).astype('float32')
            features_df['momentum_squared'] = (mom ** 2).astype('float32')
        
        # Order book imbalance features
        if 'order_book_imbalance' in orders_df.columns:
            imb = orders_df['order_book_imbalance'].values
            features_df['imbalance'] = imb.astype('float32')
            features_df['abs_imbalance'] = np.abs(imb).astype('float32')
            features_df['imbalance_sign'] = np.sign(imb).astype('float32')
            features_df['imbalance_percentile'] = pd.Series(imb).rank(pct=True).astype('float32')
        
        # 5. Enhanced LOB Features
        print("  Computing LOB depth features...")
        if not lob_df.empty:
            features_df = self._add_lob_depth_features(features_df, lob_df, orders_df)
        
        # 6. Sequential Pattern Features
        print("  Computing sequential patterns...")
        features_df = self._add_sequential_patterns(features_df)
        
        # 7. Rolling Statistical Features
        print("  Computing rolling statistics...")
        features_df = self._add_rolling_statistics(features_df)
        
        # 8. Interaction Features
        print("  Computing interaction features...")
        features_df = self._add_interaction_features(features_df)
        
        # Clean up features
        features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Convert to optimal dtypes
        for col in features_df.select_dtypes(include=['float64']).columns:
            features_df[col] = features_df[col].astype('float32')
        
        self.feature_names = features_df.columns.tolist()
        
        print(f"Feature extraction completed: {len(self.feature_names)} features generated")
        print(f"Feature matrix shape: {features_df.shape}")
        
        return features_df
    
    def _add_lob_depth_features(self, features_df, lob_df, orders_df):
        """Add LOB depth features"""
        try:
            # Efficient merge for LOB features
            lob_times = lob_df['timestamp'].values
            order_times = orders_df['timestamp'].values
            indices = np.searchsorted(lob_times, order_times, side='right') - 1
            indices = np.clip(indices, 0, len(lob_df) - 1)
            
            # Extract depth features for first 3 levels
            for level in range(1, 4):
                bid_size_col = f'Bid_Size_{level}'
                ask_size_col = f'Ask_Size_{level}'
                bid_price_col = f'Bid_Price_{level}'
                ask_price_col = f'Ask_Price_{level}'
                
                if all(col in lob_df.columns for col in [bid_size_col, ask_size_col]):
                    bid_sizes = lob_df[bid_size_col].iloc[indices].values
                    ask_sizes = lob_df[ask_size_col].iloc[indices].values
                    
                    features_df[f'bid_depth_L{level}'] = bid_sizes.astype('float32')
                    features_df[f'ask_depth_L{level}'] = ask_sizes.astype('float32')
                    
                    # Depth imbalance
                    total_size = bid_sizes + ask_sizes + 1e-8
                    features_df[f'depth_imbalance_L{level}'] = ((bid_sizes - ask_sizes) / total_size).astype('float32')
                    
                    # Depth ratio
                    features_df[f'depth_ratio_L{level}'] = (bid_sizes / (ask_sizes + 1e-8)).astype('float32')
                
                # Price level spreads
                if all(col in lob_df.columns for col in [bid_price_col, ask_price_col]):
                    bid_prices = lob_df[bid_price_col].iloc[indices].values
                    ask_prices = lob_df[ask_price_col].iloc[indices].values
                    
                    valid_prices = (bid_prices > 0) & (ask_prices > 0)
                    features_df[f'spread_L{level}'] = 0.0
                    if valid_prices.any():
                        spreads = np.zeros(len(bid_prices))
                        spreads[valid_prices] = ask_prices[valid_prices] - bid_prices[valid_prices]
                        features_df[f'spread_L{level}'] = spreads.astype('float32')
        
        except Exception as e:
            print(f"Warning: Error adding LOB features: {e}")
        
        return features_df
    
    def _add_sequential_patterns(self, features_df):
        """Add sequential pattern features"""
        try:
            if 'order_size' in features_df.columns and len(features_df) > 2:
                order_size = features_df['order_size'].values
                
                # Size acceleration
                size_diff = np.diff(order_size, prepend=order_size[0])
                size_accel = np.diff(size_diff, prepend=size_diff[0])
                features_df['size_acceleration'] = size_accel.astype('float32')
                
                # Consecutive large orders
                if len(order_size) > 0:
                    large_threshold = np.percentile(order_size, 90)
                    is_large = (order_size > large_threshold).astype('int8')
                    
                    # Rolling sum for consecutive pattern
                    consecutive = np.zeros_like(is_large, dtype='float32')
                    for i in range(len(is_large)):
                        start = max(0, i-2)
                        consecutive[i] = np.sum(is_large[start:i+1])
                    
                    features_df['consecutive_large_orders'] = consecutive
                
                # Order size momentum
                if len(order_size) > 5:
                    size_momentum = fast_pct_change(order_size, 5)
                    features_df['size_momentum'] = size_momentum.astype('float32')
        
        except Exception as e:
            print(f"Warning: Error adding sequential patterns: {e}")
        
        return features_df
    
    def _add_rolling_statistics(self, features_df):
        """Add rolling statistical features"""
        try:
            key_features = ['order_size', 'spread', 'volatility', 'arrival_rate']
            available_features = [f for f in key_features if f in features_df.columns]
            
            for feature in available_features[:3]:  # Limit for performance
                values = features_df[feature].values
                
                if len(values) > 10:
                    for window in [10, 20]:
                        if len(values) >= window:
                            # Basic rolling stats
                            ma_values = fast_rolling_mean(values, window)
                            std_values = fast_rolling_std(values, window)
                            
                            features_df[f'{feature}_ma_{window}'] = ma_values.astype('float32')
                            features_df[f'{feature}_std_{window}'] = std_values.astype('float32')
                            
                            # Z-score (standardized values)
                            zscore_values = fast_zscore(values, window)
                            features_df[f'{feature}_zscore_{window}'] = zscore_values.astype('float32')
                            
                            # Rolling min/max
                            rolling_min = np.zeros_like(values)
                            rolling_max = np.zeros_like(values)
                            
                            for i in range(len(values)):
                                start = max(0, i - window + 1)
                                rolling_min[i] = np.min(values[start:i+1])
                                rolling_max[i] = np.max(values[start:i+1])
                            
                            features_df[f'{feature}_min_{window}'] = rolling_min.astype('float32')
                            features_df[f'{feature}_max_{window}'] = rolling_max.astype('float32')
        
        except Exception as e:
            print(f"Warning: Error adding rolling statistics: {e}")
        
        return features_df
    
    def _add_interaction_features(self, features_df):
        """Add interaction features between key variables"""
        try:
            # Size-based interactions
            if 'order_size' in features_df.columns:
                if 'spread' in features_df.columns:
                    features_df['size_spread_interaction'] = (
                        features_df['order_size'] * features_df['spread']
                    ).astype('float32')
                
                if 'volatility' in features_df.columns:
                    features_df['size_volatility_interaction'] = (
                        features_df['order_size'] * features_df['volatility']
                    ).astype('float32')
                
                if 'arrival_rate' in features_df.columns:
                    features_df['size_arrival_interaction'] = (
                        features_df['order_size'] * features_df['arrival_rate']
                    ).astype('float32')
            
            # Market state interactions
            if 'volatility' in features_df.columns and 'imbalance' in features_df.columns:
                features_df['volatility_imbalance_interaction'] = (
                    features_df['volatility'] * features_df['abs_imbalance']
                ).astype('float32')
            
            if 'spread' in features_df.columns and 'arrival_rate' in features_df.columns:
                features_df['spread_arrival_interaction'] = (
                    features_df['spread'] * features_df['arrival_rate']
                ).astype('float32')
        
        except Exception as e:
            print(f"Warning: Error adding interaction features: {e}")
        
        return features_df


def load_and_extract_features(orderbook_file, message_file, max_rows=None):
    """
    Main function to load LOBSTER data and extract features for toxicity detection
    
    Args:
        orderbook_file (str): Path to LOBSTER orderbook file
        message_file (str): Path to LOBSTER message file  
        max_rows (int, optional): Maximum rows to process (for testing)
    
    Returns:
        numpy.ndarray: Feature matrix ready for toxicity detection model
        pandas.DataFrame: Feature dataframe with column names
        dict: Data information and statistics
    """
    print("="*60)
    print("LOBSTER DATA LOADING AND FEATURE EXTRACTION")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Initialize data loader
        data_loader = LOBSTERDataLoader(max_rows=max_rows)
        
        # Load raw data
        print("\n1. LOADING RAW LOBSTER DATA")
        print("-" * 30)
        
        orders_df = data_loader.load_message_data(message_file)
        lob_df = data_loader.load_orderbook_data(orderbook_file)
        
        # Initialize feature extractor
        feature_extractor = LOBSTERFeatureExtractor()
        
        # Preprocess data
        print("\n2. PREPROCESSING DATA")
        print("-" * 30)
        
        orders_df, lob_df = feature_extractor.preprocess_data(orders_df, lob_df)
        
        # Extract features
        print("\n3. EXTRACTING FEATURES")
        print("-" * 30)
        
        features_df = feature_extractor.extract_comprehensive_features(orders_df, lob_df)
        
        # Convert to numpy array for model
        X = features_df.values.astype('float32')
        
        # Data information
        data_info = {
            'n_orders': len(orders_df),
            'n_lob_snapshots': len(lob_df),
            'n_features': len(features_df.columns),
            'feature_names': feature_extractor.feature_names,
            'data_shape': X.shape,
            'processing_time': time.time() - start_time,
            'date_range': {
                'start_timestamp': int(orders_df['timestamp'].min()) if len(orders_df) > 0 else 0,
                'end_timestamp': int(orders_df['timestamp'].max()) if len(orders_df) > 0 else 0
            },
            'order_statistics': {
                'total_orders': len(orders_df),
                'buy_orders': int((orders_df['side'] == 'BUY').sum()) if len(orders_df) > 0 else 0,
                'sell_orders': int((orders_df['side'] == 'SELL').sum()) if len(orders_df) > 0 else 0,
                'market_orders': int((orders_df['order_type'] == 'MARKET').sum()) if len(orders_df) > 0 else 0,
                'limit_orders': int((orders_df['order_type'] == 'LIMIT').sum()) if len(orders_df) > 0 else 0,
                'avg_order_size': float(orders_df['quantity'].mean()) if len(orders_df) > 0 else 0,
                'max_order_size': int(orders_df['quantity'].max()) if len(orders_df) > 0 else 0
            },
            'market_statistics': {
                'avg_mid_price': float(orders_df['mid_price'].mean()) if 'mid_price' in orders_df.columns and len(orders_df) > 0 else 0,
                'avg_spread': float(orders_df['spread'].mean()) if 'spread' in orders_df.columns and len(orders_df) > 0 else 0,
                'avg_volatility': float(orders_df['volatility'].mean()) if 'volatility' in orders_df.columns and len(orders_df) > 0 else 0,
                'avg_imbalance': float(orders_df['order_book_imbalance'].mean()) if 'order_book_imbalance' in orders_df.columns and len(orders_df) > 0 else 0
            }
        }
        
        print(f"\n{'='*60}")
        print("FEATURE EXTRACTION COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"✓ Orders processed: {data_info['n_orders']:,}")
        print(f"✓ LOB snapshots: {data_info['n_lob_snapshots']:,}")
        print(f"✓ Features extracted: {data_info['n_features']}")
        print(f"✓ Feature matrix shape: {data_info['data_shape']}")
        print(f"✓ Processing time: {data_info['processing_time']:.2f} seconds")
        print(f"✓ Processing rate: {data_info['n_orders'] / data_info['processing_time']:.0f} orders/second")
        print(f"{'='*60}")
        
        return X, features_df, data_info
        
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_and_extract_features_batch(orderbook_file, message_file, batch_size=50000):
    """
    Load and extract features in batches for very large datasets
    
    Args:
        orderbook_file (str): Path to LOBSTER orderbook file
        message_file (str): Path to LOBSTER message file
        batch_size (int): Number of orders to process in each batch
    
    Returns:
        generator: Yields (X_batch, features_df_batch, batch_info) for each batch
    """
    print("="*60)
    print("LOBSTER BATCH FEATURE EXTRACTION")
    print("="*60)
    
    # Get total number of orders
    total_orders = sum(1 for _ in open(message_file)) - 1  # Subtract header
    n_batches = (total_orders + batch_size - 1) // batch_size
    
    print(f"Total orders: {total_orders:,}")
    print(f"Batch size: {batch_size:,}")
    print(f"Number of batches: {n_batches}")
    
    for batch_idx in range(n_batches):
        print(f"\nProcessing batch {batch_idx + 1}/{n_batches}")
        
        start_row = batch_idx * batch_size
        end_row = min((batch_idx + 1) * batch_size, total_orders)
        
        try:
            # Load batch data
            data_loader = LOBSTERDataLoader(max_rows=end_row - start_row)
            
            # Load message data for this batch
            orders_batch = pd.read_csv(
                message_file,
                skiprows=range(1, start_row + 1) if start_row > 0 else None,
                nrows=end_row - start_row,
                low_memory=False
            )
            
            # Load corresponding LOB data
            lob_batch = pd.read_csv(
                orderbook_file,
                skiprows=range(0, start_row) if start_row > 0 else None,
                nrows=end_row - start_row,
                header=None,
                low_memory=False
            )
            
            # Process batch
            orders_batch['timestamp'] = range(start_row, end_row)
            
            # Rename columns
            column_mapping = {
                'Time': 'time',
                'Type': 'type',
                'Order ID': 'order_id',
                'Size': 'quantity',
                'Price': 'price',
                'Direction': 'direction'
            }
            orders_batch.rename(columns=column_mapping, inplace=True)
            
            # Create order features
            orders_batch['order_type'] = np.where(orders_batch['type'].isin([1, 2, 3]), 'LIMIT', 'MARKET')
            orders_batch['side'] = np.where(orders_batch['direction'] == 1, 'BUY', 'SELL')
            
            # LOB columns
            lob_columns = []
            for i in range(1, 11):
                lob_columns.extend([f'Ask_Price_{i}', f'Ask_Size_{i}', f'Bid_Price_{i}', f'Bid_Size_{i}'])
            lob_batch.columns = lob_columns[:len(lob_batch.columns)]
            lob_batch['timestamp'] = range(len(lob_batch))
            
            # Extract features
            feature_extractor = LOBSTERFeatureExtractor()
            orders_batch, lob_batch = feature_extractor.preprocess_data(orders_batch, lob_batch)
            features_batch = feature_extractor.extract_comprehensive_features(orders_batch, lob_batch)
            
            X_batch = features_batch.values.astype('float32')
            
            batch_info = {
                'batch_idx': batch_idx,
                'start_row': start_row,
                'end_row': end_row,
                'n_orders': len(orders_batch),
                'n_features': len(features_batch.columns),
                'feature_names': feature_extractor.feature_names
            }
            
            yield X_batch, features_batch, batch_info
            
        except Exception as e:
            print(f"Error processing batch {batch_idx + 1}: {e}")
            continue


def create_feature_summary(features_df, data_info):
    """Create a comprehensive feature summary report"""
    
    print("\n" + "="*60)
    print("FEATURE SUMMARY REPORT")
    print("="*60)
    
    # Basic statistics
    print(f"\n📊 BASIC STATISTICS")
    print("-" * 30)
    print(f"Total features: {len(features_df.columns)}")
    print(f"Total samples: {len(features_df)}")
    print(f"Feature matrix size: {features_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Feature categories
    print(f"\n📋 FEATURE CATEGORIES")
    print("-" * 30)
    
    categories = {
        'Basic Order': ['order_size', 'log_order_size', 'sqrt_order_size', 'large_order', 'very_large_order'],
        'Order Type': ['is_market_order', 'is_limit_order', 'is_buy', 'is_sell'],
        'Price & Spread': ['mid_price', 'spread', 'relative_spread', 'price_ma_', 'price_std_'],
        'Timing': ['inter_arrival_time', 'arrival_rate', 'arrival_ma_', 'arrival_std_'],
        'Microstructure': ['volatility', 'momentum', 'imbalance'],
        'LOB Depth': ['bid_depth_L', 'ask_depth_L', 'depth_imbalance_L'],
        'Sequential': ['size_acceleration', 'consecutive_large_orders', 'size_momentum'],
        'Rolling Stats': ['_ma_', '_std_', '_zscore_', '_min_', '_max_'],
        'Interactions': ['_interaction']
    }
    
    for category, patterns in categories.items():
        matching_features = []
        for pattern in patterns:
            matching_features.extend([col for col in features_df.columns if pattern in col])
        
        matching_features = list(set(matching_features))  # Remove duplicates
        print(f"{category:15s}: {len(matching_features):3d} features")
    
    # Feature statistics
    print(f"\n📈 FEATURE STATISTICS")
    print("-" * 30)
    
    numeric_features = features_df.select_dtypes(include=[np.number])
    
    print(f"Mean feature value: {numeric_features.mean().mean():.4f}")
    print(f"Std feature value:  {numeric_features.std().mean():.4f}")
    print(f"Zero features:      {(numeric_features == 0).all().sum()}")
    print(f"Constant features:  {(numeric_features.std() == 0).sum()}")
    
    # Top features by variance
    print(f"\n🔝 TOP FEATURES BY VARIANCE")
    print("-" * 30)
    
    feature_vars = numeric_features.var().sort_values(ascending=False)
    for i, (feature, variance) in enumerate(feature_vars.head(10).items()):
        print(f"{i+1:2d}. {feature:30s}: {variance:.6f}")
    
    # Data quality checks
    print(f"\n✅ DATA QUALITY CHECKS")
    print("-" * 30)
    
    # Check for missing values
    missing_values = features_df.isnull().sum().sum()
    print(f"Missing values: {missing_values:,}")
    
    # Check for infinite values
    inf_values = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
    print(f"Infinite values: {inf_values:,}")
    
    # Check for extreme values
    extreme_values = (np.abs(features_df.select_dtypes(include=[np.number])) > 1e6).sum().sum()
    print(f"Extreme values (>1e6): {extreme_values:,}")
    
    # Memory usage by data type
    print(f"\n💾 MEMORY USAGE BY DATA TYPE")
    print("-" * 30)
    
    memory_by_dtype = features_df.memory_usage(deep=True).groupby(features_df.dtypes).sum()
    for dtype, memory in memory_by_dtype.items():
        print(f"{str(dtype):15s}: {memory / 1024**2:6.1f} MB")
    
    print(f"\n{'='*60}")
    
    return {
        'n_features': len(features_df.columns),
        'n_samples': len(features_df),
        'memory_usage_mb': features_df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': missing_values,
        'infinite_values': inf_values,
        'extreme_values': extreme_values,
        'top_features_by_variance': feature_vars.head(20).to_dict(),
        'feature_categories': categories
    }


def validate_features_for_model(features_df, min_variance=1e-8, max_correlation=0.95):
    """Validate features for machine learning model compatibility"""
    
    print("\n" + "="*60)
    print("FEATURE VALIDATION FOR ML MODEL")
    print("="*60)
    
    # Get numeric features
    numeric_features = features_df.select_dtypes(include=[np.number])
    
    # Check for low variance features
    print("\n🔍 CHECKING FEATURE VARIANCE")
    print("-" * 30)
    
    feature_vars = numeric_features.var()
    low_variance_features = feature_vars[feature_vars < min_variance].index.tolist()
    
    print(f"Features with variance < {min_variance}: {len(low_variance_features)}")
    if low_variance_features:
        print("Low variance features:")
        for feature in low_variance_features[:10]:  # Show first 10
            print(f"  - {feature}: {feature_vars[feature]:.2e}")
    
    # Check for highly correlated features
    print(f"\n🔗 CHECKING FEATURE CORRELATIONS")
    print("-" * 30)
    
    # Sample for correlation calculation if too many features
    if len(numeric_features.columns) > 100:
        sample_size = min(5000, len(numeric_features))
        sample_features = numeric_features.sample(n=sample_size, random_state=42)
        corr_matrix = sample_features.corr().abs()
        print(f"Using sample of {sample_size} rows for correlation calculation")
    else:
        corr_matrix = numeric_features.corr().abs()
    
    # Find highly correlated pairs
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = []
    
    for column in upper_triangle.columns:
        correlated_features = upper_triangle.index[upper_triangle[column] > max_correlation].tolist()
        for corr_feature in correlated_features:
            high_corr_pairs.append((column, corr_feature, upper_triangle.loc[corr_feature, column]))
    
    print(f"Feature pairs with correlation > {max_correlation}: {len(high_corr_pairs)}")
    if high_corr_pairs:
        print("Top highly correlated pairs:")
        for i, (feat1, feat2, corr) in enumerate(sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:10]):
            print(f"  {i+1}. {feat1} <-> {feat2}: {corr:.4f}")
    
    # Check for infinite and NaN values
    print(f"\n🚨 CHECKING FOR PROBLEMATIC VALUES")
    print("-" * 30)
    
    nan_counts = features_df.isnull().sum()
    inf_counts = np.isinf(features_df.select_dtypes(include=[np.number])).sum()
    
    problematic_features = []
    if nan_counts.sum() > 0:
        nan_features = nan_counts[nan_counts > 0]
        problematic_features.extend(nan_features.index.tolist())
        print(f"Features with NaN values: {len(nan_features)}")
    
    if inf_counts.sum() > 0:
        inf_features = inf_counts[inf_counts > 0]
        problematic_features.extend(inf_features.index.tolist())
        print(f"Features with infinite values: {len(inf_features)}")
    
    # Data type validation
    print(f"\n📊 DATA TYPE VALIDATION")
    print("-" * 30)
    
    dtype_counts = features_df.dtypes.value_counts()
    print("Data types distribution:")
    for dtype, count in dtype_counts.items():
        print(f"  {str(dtype):15s}: {count:3d} features")
    
    # Feature scaling check
    print(f"\n📏 FEATURE SCALING CHECK")
    print("-" * 30)
    
    feature_ranges = numeric_features.max() - numeric_features.min()
    large_range_features = feature_ranges[feature_ranges > 1000].index.tolist()
    
    print(f"Features with range > 1000: {len(large_range_features)}")
    if large_range_features:
        print("Large range features (may need scaling):")
        for feature in large_range_features[:5]:
            print(f"  - {feature}: {feature_ranges[feature]:.2f}")
    
    # Model readiness summary
    print(f"\n✅ MODEL READINESS SUMMARY")
    print("-" * 30)
    
    total_features = len(features_df.columns)
    usable_features = total_features - len(low_variance_features) - len(set(problematic_features))
    
    print(f"Total features: {total_features}")
    print(f"Low variance features: {len(low_variance_features)}")
    print(f"Problematic features: {len(set(problematic_features))}")
    print(f"Usable features: {usable_features}")
    print(f"Feature matrix ready: {'✓' if usable_features > 10 else '✗'}")
    
    validation_results = {
        'total_features': total_features,
        'usable_features': usable_features,
        'low_variance_features': low_variance_features,
        'high_correlation_pairs': high_corr_pairs,
        'problematic_features': list(set(problematic_features)),
        'large_range_features': large_range_features,
        'ready_for_model': usable_features > 10
    }
    
    return validation_results


# Example usage and testing functions
def test_feature_extraction(orderbook_file="AMZN_Orderbook.csv", message_file="AMZN_Order_Message.csv"):
    """Test the feature extraction pipeline with your LOBSTER data"""
    
    print("="*80)
    print("TESTING LOBSTER FEATURE EXTRACTION PIPELINE")
    print("="*80)
    
    try:
        # Test with limited data first
        print("\n🧪 TESTING WITH LIMITED DATA (1000 rows)")
        X_test, features_test, data_info_test = load_and_extract_features(
            orderbook_file, message_file, max_rows=1000
        )
        
        print(f"\n✅ TEST RESULTS:")
        print(f"  Feature matrix shape: {X_test.shape}")
        print(f"  Data types: {features_test.dtypes.value_counts().to_dict()}")
        print(f"  Memory usage: {features_test.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Create feature summary
        summary = create_feature_summary(features_test, data_info_test)
        
        # Validate features
        validation = validate_features_for_model(features_test)
        
        if validation['ready_for_model']:
            print(f"\n🎉 SUCCESS: Features are ready for toxicity detection model!")
            print(f"   Recommended to proceed with full dataset")
        else:
            print(f"\n⚠️  WARNING: Feature validation found issues")
            print(f"   Please review the validation results above")
        
        return X_test, features_test, data_info_test, summary, validation
        
    except Exception as e:
        print(f"\n❌ ERROR: Feature extraction test failed")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


if __name__ == "__main__":
    # Run the test
    print("Running LOBSTER feature extraction test...")
    
    # Test the pipeline
    test_results = test_feature_extraction()
    
    if test_results[0] is not None:
        X, features_df, data_info, summary, validation = test_results
        
        print(f"\n📄 FINAL SUMMARY:")
        print(f"{'='*50}")
        print(f"✓ Successfully extracted {data_info['n_features']} features")
        print(f"✓ From {data_info['n_orders']:,} orders")
        print(f"✓ Processing rate: {data_info['n_orders'] / data_info['processing_time']:.0f} orders/sec")
        print(f"✓ Feature matrix ready for ML: {validation['ready_for_model']}")
        print(f"{'='*50}")
        
        # Save a sample of features for inspection
        features_sample = features_df.head(100)
        features_sample.to_csv('lobster_features_sample.csv', index=False)
        print(f"✓ Sample features saved to: lobster_features_sample.csv")
        
        # Save feature info
        import json
        with open('lobster_feature_info.json', 'w') as f:
            json.dump(data_info, f, indent=2, default=str)
        print(f"✓ Feature info saved to: lobster_feature_info.json")
        
    else:
        print("\n❌ Test failed - please check your LOBSTER data files")