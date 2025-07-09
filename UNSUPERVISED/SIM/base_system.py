"""
Enhanced Market Toxicity Detection System
Production-ready implementation with improved architecture, performance, and robustness
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from contextlib import contextmanager
import joblib
import yaml
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Scientific computing
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.svm import OneClassSVM
from sklearn.model_selection import cross_val_score, KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from scipy import stats
import optuna
from optuna.samplers import TPESampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('toxicity_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelConfig:
    """Configuration for toxicity detection model"""
    # Feature selection
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    max_features: int = 200
    
    # Model parameters
    contamination_rates: List[float] = field(default_factory=lambda: [0.025, 0.05, 0.1, 0.15])
    neighbor_counts: List[int] = field(default_factory=lambda: [5, 10, 20])
    cluster_counts: List[int] = field(default_factory=lambda: [5, 8, 12])
    n_estimators: int = 200
    
    # Optimization
    n_trials: int = 50
    cv_folds: int = 5
    scoring: str = 'roc_auc'
    
    # Performance
    chunk_size: int = 10000
    n_jobs: int = -1
    use_multiprocessing: bool = True
    
    # Paths
    data_dir: str = "enhanced_market_data"
    model_dir: str = "models"
    plots_dir: str = "plots"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ModelConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

class ToxicityDetectionError(Exception):
    """Base exception for toxicity detection errors"""
    pass

class DataValidationError(ToxicityDetectionError):
    """Raised when data validation fails"""
    pass

class ModelTrainingError(ToxicityDetectionError):
    """Raised when model training fails"""
    pass

@contextmanager
def error_handler(operation_name: str):
    """Context manager for consistent error handling"""
    try:
        logger.info(f"Starting {operation_name}")
        yield
        logger.info(f"Completed {operation_name}")
    except Exception as e:
        logger.error(f"Error in {operation_name}: {str(e)}")
        raise

class DataValidator:
    """Validates input data for toxicity detection"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str], 
                          min_rows: int = 1, name: str = "DataFrame") -> None:
        """Validate DataFrame structure and content"""
        if df is None:
            raise DataValidationError(f"{name} is None")
        
        if df.empty:
            raise DataValidationError(f"{name} is empty")
        
        if len(df) < min_rows:
            raise DataValidationError(f"{name} has {len(df)} rows, minimum required: {min_rows}")
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise DataValidationError(f"{name} missing required columns: {missing_cols}")
        
        # Check for data quality issues
        null_counts = df[required_columns].isnull().sum()
        high_null_cols = null_counts[null_counts > len(df) * 0.5].index.tolist()
        if high_null_cols:
            logger.warning(f"{name} has high null rates in columns: {high_null_cols}")
    
    @staticmethod
    def validate_numeric_columns(df: pd.DataFrame, columns: List[str]) -> None:
        """Validate that specified columns are numeric"""
        for col in columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise DataValidationError(f"Column {col} is not numeric")
    
    @staticmethod
    def validate_timestamp_column(df: pd.DataFrame, timestamp_col: str) -> None:
        """Validate timestamp column"""
        if timestamp_col not in df.columns:
            raise DataValidationError(f"Timestamp column {timestamp_col} not found")
        
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            try:
                pd.to_datetime(df[timestamp_col])
            except:
                raise DataValidationError(f"Cannot convert {timestamp_col} to datetime")

class FeatureExtractor(ABC):
    """Base class for feature extractors"""
    
    @abstractmethod
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from data"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        pass

class OrderFeatureExtractor(FeatureExtractor):
    """Extract features from order data"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Extract order-based features"""
        with error_handler("order feature extraction"):
            DataValidator.validate_dataframe(
                orders_df, 
                required_columns=['quantity', 'timestamp'], 
                name="orders_df"
            )
            
            features = pd.DataFrame(index=orders_df.index)
            
            # Basic order characteristics
            features['order_size'] = orders_df['quantity']
            features['log_order_size'] = np.log1p(orders_df['quantity'])
            features['sqrt_order_size'] = np.sqrt(orders_df['quantity'])
            features['order_size_zscore'] = self._calculate_zscore(orders_df['quantity'])
            features['order_size_percentile'] = orders_df['quantity'].rank(pct=True)
            
            # Order type features
            if 'order_type' in orders_df.columns:
                features['is_market_order'] = (orders_df['order_type'] == 'MARKET').astype(int)
                features['is_limit_order'] = (orders_df['order_type'] == 'LIMIT').astype(int)
            
            if 'side' in orders_df.columns:
                features['is_buy'] = (orders_df['side'] == 'BUY').astype(int)
                features['is_sell'] = (orders_df['side'] == 'SELL').astype(int)
            
            # Size regime features
            size_quantiles = orders_df['quantity'].quantile([0.8, 0.9, 0.95, 0.99])
            features['large_order'] = (orders_df['quantity'] >= size_quantiles.iloc[0]).astype(int)
            features['very_large_order'] = (orders_df['quantity'] >= size_quantiles.iloc[1]).astype(int)
            features['extreme_order'] = (orders_df['quantity'] >= size_quantiles.iloc[2]).astype(int)
            features['massive_order'] = (orders_df['quantity'] >= size_quantiles.iloc[3]).astype(int)
            
            # Price features
            if 'price' in orders_df.columns and 'mid_price' in orders_df.columns:
                features = self._add_price_features(features, orders_df)
            
            # Timing features
            if 'timestamp' in orders_df.columns:
                features = self._add_timing_features(features, orders_df)
            
            # Market microstructure features
            for col in ['volatility', 'momentum', 'spread', 'order_book_imbalance']:
                if col in orders_df.columns:
                    features = self._add_microstructure_features(features, orders_df, col)
            
            self.feature_names = features.columns.tolist()
            return features.fillna(0).replace([np.inf, -np.inf], 0)
    
    def _calculate_zscore(self, series: pd.Series) -> pd.Series:
        """Calculate z-score with robust statistics"""
        mean = series.mean()
        std = series.std()
        return (series - mean) / (std + 1e-8) if std > 0 else pd.Series(0, index=series.index)
    
    def _add_price_features(self, features: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Add price-related features"""
        mid_price = orders_df['mid_price']
        order_price = orders_df['price']
        
        features['price_deviation'] = (order_price - mid_price) / (mid_price + 1e-8)
        features['abs_price_deviation'] = np.abs(features['price_deviation'])
        
        # Price aggressiveness
        features['price_aggressiveness'] = np.where(
            orders_df.get('side', 'BUY') == 'BUY',
            np.maximum(0, features['price_deviation']),
            np.maximum(0, -features['price_deviation'])
        )
        
        # Price momentum
        features['mid_price_returns'] = mid_price.pct_change().fillna(0)
        features['price_volatility'] = features['mid_price_returns'].rolling(20, min_periods=1).std()
        
        return features
    
    def _add_timing_features(self, features: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Add timing-related features"""
        try:
            timestamps = pd.to_datetime(orders_df['timestamp'])
            time_diffs = timestamps.diff().dt.total_seconds().fillna(1)
        except:
            # Fallback for non-datetime timestamps
            time_diffs = orders_df['timestamp'].diff().fillna(1)
        
        features['inter_arrival_time'] = time_diffs
        features['log_inter_arrival_time'] = np.log1p(time_diffs)
        features['arrival_rate'] = 1 / (time_diffs + 1e-8)
        
        # Arrival patterns
        for window in [5, 10, 20]:
            features[f'arrival_rate_ma_{window}'] = features['arrival_rate'].rolling(window, min_periods=1).mean()
            features[f'arrival_rate_std_{window}'] = features['arrival_rate'].rolling(window, min_periods=1).std()
        
        return features
    
    def _add_microstructure_features(self, features: pd.DataFrame, orders_df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add microstructure features"""
        values = orders_df[col]
        
        features[col] = values
        features[f'log_{col}'] = np.log1p(np.abs(values))
        features[f'abs_{col}'] = np.abs(values)
        features[f'{col}_percentile'] = values.rank(pct=True)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features[f'{col}_ma_{window}'] = values.rolling(window, min_periods=1).mean()
            features[f'{col}_std_{window}'] = values.rolling(window, min_periods=1).std()
        
        return features
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names

class LOBFeatureExtractor(FeatureExtractor):
    """Extract features from limit order book data"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract(self, lob_df: pd.DataFrame) -> pd.DataFrame:
        """Extract LOB-based features"""
        if lob_df.empty:
            return pd.DataFrame()
        
        with error_handler("LOB feature extraction"):
            features = pd.DataFrame(index=lob_df.index)
            
            # Depth features for multiple levels
            total_bid_depth = pd.Series(0, index=lob_df.index)
            total_ask_depth = pd.Series(0, index=lob_df.index)
            
            for level in range(1, 6):
                bid_col = f'bid_size_{level}'
                ask_col = f'ask_size_{level}'
                
                if bid_col in lob_df.columns and ask_col in lob_df.columns:
                    bid_size = lob_df[bid_col].fillna(0)
                    ask_size = lob_df[ask_col].fillna(0)
                    
                    total_bid_depth += bid_size
                    total_ask_depth += ask_size
                    
                    # Level-specific features
                    features[f'bid_depth_L{level}'] = bid_size
                    features[f'ask_depth_L{level}'] = ask_size
                    features[f'depth_imbalance_L{level}'] = (bid_size - ask_size) / (bid_size + ask_size + 1e-8)
                    features[f'depth_ratio_L{level}'] = bid_size / (ask_size + 1e-8)
            
            # Aggregate features
            features['total_bid_depth'] = total_bid_depth
            features['total_ask_depth'] = total_ask_depth
            features['total_depth'] = total_bid_depth + total_ask_depth
            features['depth_imbalance'] = (total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth + 1e-8)
            features['depth_ratio'] = total_bid_depth / (total_ask_depth + 1e-8)
            
            # Spread features
            if 'bid_price_1' in lob_df.columns and 'ask_price_1' in lob_df.columns:
                spread = lob_df['ask_price_1'] - lob_df['bid_price_1']
                mid_price = (lob_df['bid_price_1'] + lob_df['ask_price_1']) / 2
                
                features['spread'] = spread
                features['relative_spread'] = spread / (mid_price + 1e-8)
                features['log_spread'] = np.log1p(spread)
            
            self.feature_names = features.columns.tolist()
            return features.fillna(0).replace([np.inf, -np.inf], 0)
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names

class TradeFeatureExtractor(FeatureExtractor):
    """Extract features from trade data"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Extract trade-based features"""
        if trades_df.empty:
            return pd.DataFrame()
        
        with error_handler("trade feature extraction"):
            features = pd.DataFrame(index=trades_df.index)
            
            # Basic trade features
            if 'quantity' in trades_df.columns:
                features['trade_size'] = trades_df['quantity']
                features['log_trade_size'] = np.log1p(trades_df['quantity'])
                features['trade_size_zscore'] = self._calculate_zscore(trades_df['quantity'])
            
            # Price features
            if 'price' in trades_df.columns:
                features['trade_price'] = trades_df['price']
                features['price_returns'] = trades_df['price'].pct_change().fillna(0)
                features['price_volatility'] = features['price_returns'].rolling(20, min_periods=1).std()
            
            # Volume features
            if 'quantity' in trades_df.columns:
                for window in [10, 30, 60]:
                    features[f'volume_sum_{window}'] = trades_df['quantity'].rolling(window, min_periods=1).sum()
                    features[f'volume_mean_{window}'] = trades_df['quantity'].rolling(window, min_periods=1).mean()
                    features[f'volume_std_{window}'] = trades_df['quantity'].rolling(window, min_periods=1).std()
            
            # VWAP features
            if 'price' in trades_df.columns and 'quantity' in trades_df.columns:
                for window in [10, 30, 60]:
                    volume_sum = trades_df['quantity'].rolling(window, min_periods=1).sum()
                    vwap = (trades_df['price'] * trades_df['quantity']).rolling(window, min_periods=1).sum() / (volume_sum + 1e-8)
                    features[f'vwap_{window}'] = vwap
                    if window == 30:  # Use 30-period VWAP as reference
                        features['price_vs_vwap'] = (trades_df['price'] - vwap) / (vwap + 1e-8)
            
            self.feature_names = features.columns.tolist()
            return features.fillna(0).replace([np.inf, -np.inf], 0)
    
    def _calculate_zscore(self, series: pd.Series) -> pd.Series:
        """Calculate z-score with robust statistics"""
        mean = series.mean()
        std = series.std()
        return (series - mean) / (std + 1e-8) if std > 0 else pd.Series(0, index=series.index)
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names

class SequentialFeatureExtractor(FeatureExtractor):
    """Extract sequential pattern features"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extract sequential pattern features"""
        with error_handler("sequential pattern extraction"):
            sequential_features = pd.DataFrame(index=features_df.index)
            
            # Size acceleration patterns
            if 'order_size' in features_df.columns:
                size_diff = features_df['order_size'].diff()
                size_accel = size_diff.diff()
                sequential_features['size_acceleration'] = size_accel
                sequential_features['size_momentum_burst'] = (size_accel > size_accel.quantile(0.95)).astype(int)
                
                # Consecutive large orders
                large_threshold = features_df['order_size'].quantile(0.9)
                is_large = (features_df['order_size'] > large_threshold).astype(int)
                sequential_features['consecutive_large_orders'] = is_large.rolling(3, min_periods=1).sum()
            
            # Arrival rate patterns
            if 'arrival_rate' in features_df.columns:
                arrival_ma = features_df['arrival_rate'].rolling(10, min_periods=1).mean()
                arrival_std = features_df['arrival_rate'].rolling(10, min_periods=1).std()
                sequential_features['arrival_burst'] = ((features_df['arrival_rate'] - arrival_ma) > 2 * arrival_std).astype(int)
            
            # Market impact patterns
            if 'price_returns' in features_df.columns and 'order_size' in features_df.columns:
                sequential_features['impact_per_size'] = features_df['price_returns'] / (features_df['order_size'] + 1e-8)
                sequential_features['abnormal_impact'] = (
                    np.abs(sequential_features['impact_per_size']) > 
                    np.abs(sequential_features['impact_per_size']).quantile(0.95)
                ).astype(int)
            
            self.feature_names = sequential_features.columns.tolist()
            return sequential_features.fillna(0).replace([np.inf, -np.inf], 0)
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names

class FeatureEngineeringPipeline:
    """Pipeline for feature engineering"""
    
    def __init__(self, extractors: List[FeatureExtractor]):
        self.extractors = extractors
        self.feature_names = []
    
    def fit_transform(self, orders_df: pd.DataFrame, lob_df: pd.DataFrame = None, 
                     trades_df: pd.DataFrame = None) -> pd.DataFrame:
        """Apply feature extractors and combine results"""
        with error_handler("feature engineering pipeline"):
            all_features = []
            
            # Apply each extractor
            for extractor in self.extractors:
                if isinstance(extractor, OrderFeatureExtractor):
                    features = extractor.extract(orders_df)
                elif isinstance(extractor, LOBFeatureExtractor) and lob_df is not None:
                    features = extractor.extract(lob_df)
                elif isinstance(extractor, TradeFeatureExtractor) and trades_df is not None:
                    features = extractor.extract(trades_df)
                elif isinstance(extractor, SequentialFeatureExtractor):
                    # Sequential features need the combined feature set
                    if all_features:
                        combined = pd.concat(all_features, axis=1)
                        features = extractor.extract(combined)
                    else:
                        continue
                else:
                    continue
                
                if not features.empty:
                    all_features.append(features)
            
            # Combine all features
            if all_features:
                result = pd.concat(all_features, axis=1)
                # Remove duplicate columns
                result = result.loc[:, ~result.columns.duplicated()]
                self.feature_names = result.columns.tolist()
                return result
            else:
                return pd.DataFrame()
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names

class FeatureSelector:
    """Advanced feature selection"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.selected_features = []
        self.selector_pipeline = None
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Apply feature selection pipeline"""
        with error_handler("feature selection"):
            # Handle missing values and infinite values first
            X_clean = self._clean_data(X)
            
            # Start with cleaned data
            current_X = X_clean.copy()
            
            # Step 1: Remove low variance features
            var_selector = VarianceThreshold(threshold=self.config.variance_threshold)
            current_X = pd.DataFrame(
                var_selector.fit_transform(current_X),
                index=current_X.index,
                columns=current_X.columns[var_selector.get_support()]
            )
            logger.info(f"Variance threshold: {current_X.shape[1]} features retained")
            
            # Step 2: Remove highly correlated features
            current_X = self._remove_correlated_features(current_X)
            logger.info(f"Correlation filtering: {current_X.shape[1]} features retained")
            
            # Step 3: Select best features if too many remain
            if current_X.shape[1] > self.config.max_features:
                if y is not None:
                    selector = SelectKBest(f_classif, k=self.config.max_features)
                    current_X = pd.DataFrame(
                        selector.fit_transform(current_X, y),
                        index=current_X.index,
                        columns=current_X.columns[selector.get_support()]
                    )
                else:
                    # Use variance-based selection
                    variances = current_X.var()
                    top_features = variances.nlargest(self.config.max_features).index
                    current_X = current_X[top_features]
                
                logger.info(f"Final selection: {current_X.shape[1]} features retained")
            
            self.selected_features = current_X.columns.tolist()
            return current_X
    
    def _clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing and infinite values"""
        logger.info("Cleaning data for feature selection...")
        
        # Handle missing values
        X_clean = X.copy()
        
        # Replace infinite values with NaN first
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # Handle missing values column by column
        for col in X_clean.columns:
            if X_clean[col].isnull().any():
                if X_clean[col].dtype in ['int64', 'float64']:
                    # Use median for numeric columns
                    median_val = X_clean[col].median()
                    if pd.isna(median_val):
                        X_clean[col] = X_clean[col].fillna(0)
                    else:
                        X_clean[col] = X_clean[col].fillna(median_val)
                else:
                    # Use mode for categorical columns
                    mode_val = X_clean[col].mode()
                    if len(mode_val) > 0:
                        X_clean[col] = X_clean[col].fillna(mode_val.iloc[0])
                    else:
                        X_clean[col] = X_clean[col].fillna(0)
        
        # Final check and cleanup
        X_clean = X_clean.fillna(0)
        
        # Verify no missing or infinite values remain
        if X_clean.isnull().any().any():
            logger.error("Data still contains missing values after cleaning")
            raise ValueError("Unable to clean all missing values")
        
        if np.any(np.isinf(X_clean.values)):
            logger.error("Data still contains infinite values after cleaning")
            raise ValueError("Unable to clean all infinite values")
        
        logger.info(f"Data cleaned successfully: {X_clean.shape}")
        return X_clean
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features"""
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        features_to_drop = set()
        for column in upper_triangle.columns:
            correlated_features = upper_triangle.index[
                upper_triangle[column] > self.config.correlation_threshold
            ].tolist()
            
            if correlated_features:
                # Keep feature with highest variance
                variances = {feat: X[feat].var() for feat in correlated_features + [column]}
                features_to_keep = max(variances, key=variances.get)
                features_to_drop.update([f for f in correlated_features if f != features_to_keep])
        
        return X.drop(columns=list(features_to_drop))
    
    def get_selected_features(self) -> List[str]:
        return self.selected_features

class AnomalyDetector(ABC):
    """Base class for anomaly detectors"""
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'AnomalyDetector':
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        pass

class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest wrapper"""
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100, **kwargs):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            **kwargs
        )
    
    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        self.model.fit(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)

class LOFDetector(AnomalyDetector):
    """Local Outlier Factor wrapper"""
    
    def __init__(self, n_neighbors: int = 20, contamination: float = 0.1, **kwargs):
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            **kwargs
        )
    
    def fit(self, X: np.ndarray) -> 'LOFDetector':
        self.model.fit(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)

class SVMDetector(AnomalyDetector):
    """SVM detector wrapper for consistent interface"""
    def __init__(self, model):
        self.model = model
    
    def fit(self, X):
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def decision_function(self, X):
        return -self.model.decision_function(X)

class KMeansDetector(AnomalyDetector):
    """K-means detector wrapper for consistent interface"""
    def __init__(self, kmeans, threshold):
        self.kmeans = kmeans
        self.threshold = threshold
    
    def fit(self, X):
        return self
    
    def predict(self, X):
        distances = np.min(self.kmeans.transform(X), axis=1)
        return np.where(distances > self.threshold, -1, 1)
    
    def decision_function(self, X):
        return np.min(self.kmeans.transform(X), axis=1)

class EnsembleAnomalyDetector:
    """Ensemble of anomaly detectors with adaptive weighting"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.detectors = {}
        self.weights = {}
        self.scaler = None
        self.performance_metrics = {}
    
    def fit(self, X: pd.DataFrame) -> 'EnsembleAnomalyDetector':
        """Train ensemble of detectors"""
        with error_handler("ensemble training"):
            # Handle missing values first
            X_clean = self._handle_missing_values(X)
            
            # Scale features
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X_clean)
            
            # Additional validation
            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                raise ValueError("Data still contains NaN or infinite values after preprocessing")
            
            # Train different detector types
            self._train_isolation_forests(X_scaled)
            self._train_lof_detectors(X_scaled)
            self._train_svm_detectors(X_scaled)
            self._train_clustering_detectors(X_scaled)
            
            # Calculate adaptive weights
            self._calculate_adaptive_weights(X_scaled)
            
            logger.info(f"Trained {len(self.detectors)} detectors")
            return self
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data"""
        logger.info("Handling missing values...")
        
        # Check initial missing values
        missing_counts = X.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            logger.warning(f"Found {total_missing} missing values across {len(missing_counts[missing_counts > 0])} columns")
            
            # Strategy 1: Remove columns with too many missing values (>50%)
            high_missing_cols = missing_counts[missing_counts > len(X) * 0.5].index.tolist()
            if high_missing_cols:
                logger.info(f"Removing columns with >50% missing values: {high_missing_cols}")
                X = X.drop(columns=high_missing_cols)
            
            # Strategy 2: Fill remaining missing values
            for col in X.columns:
                if X[col].isnull().any():
                    if X[col].dtype in ['int64', 'float64']:
                        # Use median for numeric columns
                        median_val = X[col].median()
                        if pd.isna(median_val):
                            # If median is also NaN, use 0
                            X[col] = X[col].fillna(0)
                        else:
                            X[col] = X[col].fillna(median_val)
                    else:
                        # Use mode for categorical columns
                        mode_val = X[col].mode()
                        if len(mode_val) > 0:
                            X[col] = X[col].fillna(mode_val.iloc[0])
                        else:
                            X[col] = X[col].fillna(0)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill any remaining NaN values with 0
        X = X.fillna(0)
        
        # Final validation
        remaining_missing = X.isnull().sum().sum()
        if remaining_missing > 0:
            logger.error(f"Still have {remaining_missing} missing values after preprocessing")
            raise ValueError(f"Unable to handle all missing values. {remaining_missing} remain.")
        
        # Check for infinite values
        if np.any(np.isinf(X.values)):
            logger.error("Data contains infinite values")
            raise ValueError("Data contains infinite values after preprocessing")
        
        logger.info(f"Successfully cleaned data: {X.shape[0]} samples, {X.shape[1]} features")
        return X
    def _train_isolation_forests(self, X: np.ndarray):
        """Train isolation forest variants"""
        for i, contamination in enumerate(self.config.contamination_rates):
            try:
                detector = IsolationForestDetector(
                    contamination=contamination,
                    n_estimators=self.config.n_estimators,
                    random_state=42 + i,
                    n_jobs=self.config.n_jobs
                )
                detector.fit(X)
                self.detectors[f'isolation_forest_{contamination}'] = detector
                logger.info(f"Successfully trained isolation forest with contamination={contamination}")
            except Exception as e:
                logger.warning(f"Failed to train isolation forest {contamination}: {e}")
    
    def _train_lof_detectors(self, X: np.ndarray):
        """Train LOF variants"""
        for neighbors in self.config.neighbor_counts:
            try:
                detector = LOFDetector(
                    n_neighbors=neighbors,
                    contamination=0.1,
                    n_jobs=self.config.n_jobs
                )
                detector.fit(X)
                self.detectors[f'lof_{neighbors}'] = detector
                logger.info(f"Successfully trained LOF with {neighbors} neighbors")
            except Exception as e:
                logger.warning(f"Failed to train LOF {neighbors}: {e}")
    
    def _train_svm_detectors(self, X: np.ndarray):
        """Train One-Class SVM variants"""
        svm_configs = [
            {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.05},
            {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.1},
        ]
        
        for config in svm_configs:
            try:
                model = OneClassSVM(**config)
                model.fit(X)
                
                detector = SVMDetector(model)
                self.detectors[f'svm_{config["kernel"]}_{config["nu"]}'] = detector
                logger.info(f"Successfully trained SVM with config {config}")
            except Exception as e:
                logger.warning(f"Failed to train SVM {config}: {e}")
    
    def _train_clustering_detectors(self, X: np.ndarray):
        """Train clustering-based detectors"""
        for n_clusters in self.config.cluster_counts:
            try:
                n_clust = min(n_clusters, max(2, len(X) // 50))
                if n_clust < 2:
                    continue
                
                kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                distances = np.min(kmeans.transform(X), axis=1)
                
                threshold = np.percentile(distances, 95)
                detector = KMeansDetector(kmeans, threshold)
                self.detectors[f'kmeans_{n_clust}'] = detector
                logger.info(f"Successfully trained K-means with {n_clust} clusters")
            except Exception as e:
                logger.warning(f"Failed to train K-means {n_clusters}: {e}")
    
    def _calculate_adaptive_weights(self, X: np.ndarray):
        """Calculate adaptive weights for ensemble"""
        individual_scores = {}
        
        # Get scores from each detector
        for name, detector in self.detectors.items():
            try:
                scores = detector.decision_function(X)
                # Normalize scores
                scores = (scores - np.mean(scores)) / (np.std(scores) + 1e-8)
                individual_scores[name] = scores
            except Exception as e:
                logger.warning(f"Failed to get scores from {name}: {e}")
                continue
        
        # Calculate weights based on score quality
        weights = {}
        for name, scores in individual_scores.items():
            # Score distribution quality
            score_std = np.std(scores)
            score_range = np.max(scores) - np.min(scores)
            separation_score = score_std * score_range
            
            # Detection consistency across thresholds
            consistency_scores = []
            for pct in [90, 95, 99]:
                threshold = np.percentile(scores, pct)
                detection_rate = np.mean(scores > threshold)
                expected_rate = (100 - pct) / 100
                if expected_rate > 0:
                    consistency = 1 - abs(detection_rate - expected_rate) / expected_rate
                    consistency_scores.append(max(0, consistency))
            
            avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
            
            # Detector type preference
            type_bonus = 1.0
            if 'isolation_forest' in name:
                type_bonus = 1.1
            elif 'lof' in name:
                type_bonus = 1.05
            elif 'svm' in name:
                type_bonus = 1.02
            
            # Combine criteria
            weight = separation_score * avg_consistency * type_bonus
            weights[name] = max(0.05, min(weight, 2.0))
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            self.weights = {name: 1.0 / len(self.detectors) for name in self.detectors.keys()}
        
        logger.info(f"Detector weights: {self.weights}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomaly probabilities"""
        if self.scaler is None:
            raise ModelTrainingError("Model not fitted")
        
        # Handle missing values before scaling
        X_clean = self._handle_missing_values(X)
        
        # Validate that we have the expected features
        expected_features = self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else list(range(X_clean.shape[1]))
        
        if hasattr(self.scaler, 'feature_names_in_'):
            # Ensure all expected features are present
            for feature in expected_features:
                if feature not in X_clean.columns:
                    X_clean[feature] = 0
                    logger.warning(f"Added missing feature '{feature}' with zeros for prediction")
            
            # Select features in the same order as training
            X_clean = X_clean[expected_features]
        
        X_scaled = self.scaler.transform(X_clean)
        
        # Validate scaled data
        if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
            raise ValueError("Scaled data contains NaN or infinite values")
        
        ensemble_scores = np.zeros(len(X_scaled))
        
        for name, detector in self.detectors.items():
            try:
                scores = detector.decision_function(X_scaled)
                # Normalize scores
                scores = (scores - np.mean(scores)) / (np.std(scores) + 1e-8)
                weight = self.weights.get(name, 0)
                ensemble_scores += weight * scores
            except Exception as e:
                logger.warning(f"Failed to get scores from {name}: {e}")
                continue
        
        # Convert to probabilities
        probabilities = 1 / (1 + np.exp(-ensemble_scores))
        return probabilities
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.95) -> np.ndarray:
        """Predict anomalies"""
        probabilities = self.predict_proba(X)
        threshold_value = np.percentile(probabilities, threshold * 100)
        return (probabilities > threshold_value).astype(int)

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.study = None
        self.best_params = {}
    
    def optimize(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Optimize hyperparameters"""
        with error_handler("hyperparameter optimization"):
            def objective(trial):
                # Suggest hyperparameters
                contamination = trial.suggest_float('contamination', 0.01, 0.2)
                n_estimators = trial.suggest_int('n_estimators', 100, 300)
                n_neighbors = trial.suggest_int('n_neighbors', 5, 30)
                
                try:
                    # Create temporary config
                    temp_config = ModelConfig(
                        contamination_rates=[contamination],
                        n_estimators=n_estimators,
                        neighbor_counts=[n_neighbors]
                    )
                    
                    # Train ensemble
                    detector = EnsembleAnomalyDetector(temp_config)
                    detector.fit(X)
                    
                    # Check if any detectors were trained
                    if len(detector.detectors) == 0:
                        logger.warning("No detectors were successfully trained")
                        return -10
                    
                    # Evaluate using silhouette score
                    probabilities = detector.predict_proba(X)
                    
                    # Use k-means to evaluate clustering quality
                    n_clusters = min(5, max(2, len(X) // 100))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = kmeans.fit_predict(X)
                    
                    if len(set(labels)) > 1:
                        score = silhouette_score(X, labels)
                    else:
                        score = -1
                    
                    return score
                    
                except Exception:
                    return -10
            
            # Run optimization
            self.study = optuna.create_study(direction='maximize', sampler=TPESampler())
            self.study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=True)
            
            self.best_params = self.study.best_params
            logger.info(f"Best hyperparameters: {self.best_params}")
            
            return self.best_params

class ModelMonitor:
    """Monitor model performance and detect drift"""
    
    def __init__(self, model: EnsembleAnomalyDetector, reference_data: pd.DataFrame):
        self.model = model
        self.reference_data = reference_data
        self.reference_scores = model.predict_proba(reference_data)
        self.drift_threshold = 0.05  # p-value threshold for drift detection
    
    def detect_drift(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift using statistical tests"""
        from scipy.stats import ks_2samp
        
        drift_results = {}
        
        # Feature-level drift detection
        feature_drift = {}
        for column in new_data.columns:
            if column in self.reference_data.columns:
                try:
                    stat, p_value = ks_2samp(
                        self.reference_data[column].dropna(),
                        new_data[column].dropna()
                    )
                    feature_drift[column] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'drift_detected': p_value < self.drift_threshold
                    }
                except Exception as e:
                    logger.warning(f"Failed to test drift for {column}: {e}")
        
        # Score-level drift detection
        try:
            new_scores = self.model.predict_proba(new_data)
            score_stat, score_p_value = ks_2samp(self.reference_scores, new_scores)
            
            score_drift = {
                'statistic': score_stat,
                'p_value': score_p_value,
                'drift_detected': score_p_value < self.drift_threshold
            }
        except Exception as e:
            logger.warning(f"Failed to test score drift: {e}")
            score_drift = {'drift_detected': False}
        
        drift_results = {
            'feature_drift': feature_drift,
            'score_drift': score_drift,
            'overall_drift': any(
                fd.get('drift_detected', False) for fd in feature_drift.values()
            ) or score_drift.get('drift_detected', False)
        }
        
        return drift_results
    
    def performance_summary(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate performance summary"""
        new_scores = self.model.predict_proba(new_data)
        
        summary = {
            'data_size': len(new_data),
            'score_statistics': {
                'mean': np.mean(new_scores),
                'std': np.std(new_scores),
                'min': np.min(new_scores),
                'max': np.max(new_scores),
                'percentiles': {
                    '95': np.percentile(new_scores, 95),
                    '99': np.percentile(new_scores, 99),
                    '99.5': np.percentile(new_scores, 99.5)
                }
            },
            'anomaly_rates': {
                f'{pct}th_percentile': np.mean(new_scores > np.percentile(new_scores, pct))
                for pct in [90, 95, 99]
            }
        }
        
        return summary

class ToxicityDetectionSystem:
    """Main system for market toxicity detection"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.feature_pipeline = None
        self.feature_selector = None
        self.model = None
        self.monitor = None
        self.training_data = None
        self.selected_feature_names = []
        self.is_fitted = False
    
    def setup_pipeline(self):
        """Setup feature engineering pipeline"""
        extractors = [
            OrderFeatureExtractor(),
            LOBFeatureExtractor(),
            TradeFeatureExtractor(),
            SequentialFeatureExtractor()
        ]
        
        self.feature_pipeline = FeatureEngineeringPipeline(extractors)
        self.feature_selector = FeatureSelector(self.config)
    
    def load_data(self, data_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load market data"""
        if data_dir is None:
            data_dir = self.config.data_dir
        
        with error_handler("data loading"):
            # Find data files
            data_path = Path(data_dir)
            if not data_path.exists():
                raise DataValidationError(f"Data directory {data_dir} does not exist")
            
            order_files = list(data_path.glob("orders_*.csv"))
            lob_files = list(data_path.glob("lob_snapshots_*.csv"))
            trade_files = list(data_path.glob("trades_*.csv"))
            
            if not order_files:
                raise DataValidationError(f"No order files found in {data_dir}")
            
            # Load most recent files
            latest_order_file = max(order_files, key=lambda p: p.stat().st_mtime)
            orders_df = pd.read_csv(latest_order_file)
            logger.info(f"Loaded {len(orders_df)} orders from {latest_order_file.name}")
            
            # Load LOB data if available
            lob_df = pd.DataFrame()
            if lob_files:
                latest_lob_file = max(lob_files, key=lambda p: p.stat().st_mtime)
                lob_df = pd.read_csv(latest_lob_file)
                logger.info(f"Loaded {len(lob_df)} LOB snapshots from {latest_lob_file.name}")
            
            # Load trade data if available
            trades_df = pd.DataFrame()
            if trade_files:
                latest_trade_file = max(trade_files, key=lambda p: p.stat().st_mtime)
                trades_df = pd.read_csv(latest_trade_file)
                logger.info(f"Loaded {len(trades_df)} trades from {latest_trade_file.name}")
            
            # Validate data
            DataValidator.validate_dataframe(orders_df, ['quantity', 'timestamp'], name="orders")
            DataValidator.validate_numeric_columns(orders_df, ['quantity'])
            DataValidator.validate_timestamp_column(orders_df, 'timestamp')
            
            return orders_df, lob_df, trades_df
    
    def train(self, orders_df: pd.DataFrame, lob_df: pd.DataFrame = None, 
              trades_df: pd.DataFrame = None, optimize_hyperparams: bool = True):
        """Train the toxicity detection model"""
        with error_handler("model training"):
            # Setup pipeline if not already done
            if self.feature_pipeline is None:
                self.setup_pipeline()
            
            # Extract features
            logger.info("Extracting features...")
            features_df = self.feature_pipeline.fit_transform(orders_df, lob_df, trades_df)
            
            if features_df.empty:
                raise ModelTrainingError("No features extracted")
            
            logger.info(f"Extracted {len(features_df.columns)} features")
            
            # Select features
            logger.info("Selecting features...")
            selected_features = self.feature_selector.fit_transform(features_df)
            logger.info(f"Selected {len(selected_features.columns)} features")
            
            # Store feature names for consistency
            self.selected_feature_names = selected_features.columns.tolist()
            
            # Optimize hyperparameters if requested
            if optimize_hyperparams:
                logger.info("Optimizing hyperparameters...")
                optimizer = HyperparameterOptimizer(self.config)
                best_params = optimizer.optimize(selected_features)
                
                # Update config with best parameters
                self.config.contamination_rates = [best_params.get('contamination', 0.1)]
                self.config.n_estimators = best_params.get('n_estimators', 200)
                self.config.neighbor_counts = [best_params.get('n_neighbors', 20)]
            
            # Train ensemble model
            logger.info("Training ensemble model...")
            self.model = EnsembleAnomalyDetector(self.config)
            self.model.fit(selected_features)
            
            # Setup monitoring
            self.monitor = ModelMonitor(self.model, selected_features)
            
            # Store training data reference
            self.training_data = {
                'features': selected_features,
                'feature_names': self.selected_feature_names,
                'orders_shape': orders_df.shape,
                'lob_shape': lob_df.shape if lob_df is not None else (0, 0),
                'trades_shape': trades_df.shape if trades_df is not None else (0, 0)
            }
            
            self.is_fitted = True
            logger.info("Model training completed successfully")
    
    def predict(self, orders_df: pd.DataFrame, lob_df: pd.DataFrame = None, 
                trades_df: pd.DataFrame = None, threshold: float = 0.95) -> Dict[str, Any]:
        """Predict toxicity scores"""
        if not self.is_fitted:
            raise ModelTrainingError("Model not fitted. Call train() first.")
        
        with error_handler("prediction"):
            # Extract features using the same pipeline
            features_df = self.feature_pipeline.fit_transform(orders_df, lob_df, trades_df)
            
            # Ensure we have the same features as training
            training_features = self.selected_feature_names
            
            # Add missing features with zeros
            for feature in training_features:
                if feature not in features_df.columns:
                    features_df[feature] = 0
                    logger.warning(f"Added missing feature '{feature}' with zeros")
            
            # Select only the features used in training, in the same order
            selected_features = features_df[training_features]
            
            # Verify we have the right shape
            if selected_features.shape[1] != len(training_features):
                raise ValueError(f"Feature count mismatch: expected {len(training_features)}, got {selected_features.shape[1]}")
            
            # Get predictions
            probabilities = self.model.predict_proba(selected_features)
            predictions = self.model.predict(selected_features, threshold)
            
            # Detect drift
            drift_results = self.monitor.detect_drift(selected_features)
            
            # Performance summary
            performance = self.monitor.performance_summary(selected_features)
            
            results = {
                'probabilities': probabilities,
                'predictions': predictions,
                'anomaly_indices': np.where(predictions == 1)[0],
                'drift_detected': drift_results['overall_drift'],
                'drift_details': drift_results,
                'performance': performance,
                'threshold_used': threshold,
                'n_features_used': selected_features.shape[1],
                'feature_names': training_features
            }
            
            return results
    
    def save_model(self, filepath: str = None):
        """Save trained model"""
        if not self.is_fitted:
            raise ModelTrainingError("Model not fitted. Call train() first.")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{self.config.model_dir}/toxicity_model_{timestamp}.joblib"
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_package = {
            'model': self.model,
            'feature_pipeline': self.feature_pipeline,
            'feature_selector': self.feature_selector,
            'selected_feature_names': self.selected_feature_names,
            'config': self.config,
            'training_data': self.training_data,
            'version': '2.0_production',
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_package, filepath)
        logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    def load_model(self, filepath: str):
        """Load trained model"""
        with error_handler("model loading"):
            model_package = joblib.load(filepath)
            
            self.model = model_package['model']
            self.feature_pipeline = model_package['feature_pipeline']
            self.feature_selector = model_package['feature_selector']
            self.selected_feature_names = model_package.get('selected_feature_names', [])
            self.config = model_package['config']
            self.training_data = model_package['training_data']
            
            # Recreate monitor
            if self.training_data and 'features' in self.training_data:
                self.monitor = ModelMonitor(self.model, self.training_data['features'])
            
            self.is_fitted = True
            logger.info(f"Model loaded from {filepath}")

class Visualizer:
    """Create visualizations for model performance"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.plots_dir = Path(config.plots_dir)
        self.plots_dir.mkdir(exist_ok=True)
    
    def plot_model_performance(self, results: Dict[str, Any], 
                              features_df: pd.DataFrame = None,
                              save: bool = True) -> None:
        """Create comprehensive performance visualization"""
        # Set basic plot parameters
        plt.rcParams.update({
            'figure.figsize': (18, 12),
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.fontsize': 10,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        probabilities = results['probabilities']
        predictions = results['predictions']
        
        # Score distribution
        axes[0, 0].hist(probabilities, bins=50, alpha=0.7, color='skyblue', density=True)
        axes[0, 0].axvline(np.percentile(probabilities, 95), color='red', linestyle='--', label='95th percentile')
        axes[0, 0].axvline(np.percentile(probabilities, 99), color='darkred', linestyle='--', label='99th percentile')
        axes[0, 0].set_xlabel('Toxicity Score')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Toxicity Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Time series plot
        axes[0, 1].plot(probabilities, alpha=0.7, color='blue', linewidth=1)
        axes[0, 1].axhline(np.percentile(probabilities, 95), color='orange', linestyle='--', alpha=0.8)
        axes[0, 1].axhline(np.percentile(probabilities, 99), color='red', linestyle='--', alpha=0.8)
        
        # Highlight anomalies
        anomaly_indices = results['anomaly_indices']
        if len(anomaly_indices) > 0:
            axes[0, 1].scatter(anomaly_indices, probabilities[anomaly_indices], 
                              color='red', s=20, alpha=0.8, zorder=5)
        
        axes[0, 1].set_xlabel('Order Sequence')
        axes[0, 1].set_ylabel('Toxicity Score')
        axes[0, 1].set_title('Toxicity Timeline')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Anomaly rate by threshold
        thresholds = [85, 90, 95, 97, 99, 99.5]
        rates = []
        for threshold in thresholds:
            rate = np.mean(probabilities > np.percentile(probabilities, threshold)) * 100
            rates.append(rate)
        
        bars = axes[0, 2].bar(range(len(thresholds)), rates, 
                             color=['lightblue', 'lightgreen', 'orange', 'coral', 'red', 'darkred'])
        axes[0, 2].set_xticks(range(len(thresholds)))
        axes[0, 2].set_xticklabels([f'{t}th' for t in thresholds])
        axes[0, 2].set_ylabel('Detection Rate (%)')
        axes[0, 2].set_title('Detection Rates by Threshold')
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           f'{rate:.1f}%', ha='center', va='bottom')
        
        # Performance metrics
        performance = results['performance']
        score_stats = performance['score_statistics']
        
        metrics_text = f"""
        Performance Summary
        
        Data Size: {performance['data_size']} samples
        
        Score Statistics:
        • Mean: {score_stats['mean']:.3f}
        • Std: {score_stats['std']:.3f}
        • Min: {score_stats['min']:.3f}
        • Max: {score_stats['max']:.3f}
        
        Percentiles:
        • 95th: {score_stats['percentiles']['95']:.3f}
        • 99th: {score_stats['percentiles']['99']:.3f}
        • 99.5th: {score_stats['percentiles']['99.5']:.3f}
        
        Anomaly Rates:
        • 90th: {performance['anomaly_rates']['90th_percentile']:.3f}
        • 95th: {performance['anomaly_rates']['95th_percentile']:.3f}
        • 99th: {performance['anomaly_rates']['99th_percentile']:.3f}
        
        Drift Status: {'DETECTED' if results['drift_detected'] else 'OK'}
        """
        
        axes[1, 0].text(0.05, 0.95, metrics_text, transform=axes[1, 0].transAxes,
                       verticalalignment='top', fontfamily='monospace', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Model Performance Summary')
        
        # Feature importance (if available)
        if features_df is not None and len(features_df.columns) > 0:
            # Calculate feature importance based on correlation with scores
            feature_importance = {}
            for col in features_df.columns:
                try:
                    corr = np.corrcoef(features_df[col], probabilities)[0, 1]
                    feature_importance[col] = abs(corr) if not np.isnan(corr) else 0
                except:
                    feature_importance[col] = 0
            
            # Plot top features
            top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
            feature_names = list(top_features.keys())
            importance_values = list(top_features.values())
            
            axes[1, 1].barh(range(len(feature_names)), importance_values, color='lightcoral')
            axes[1, 1].set_yticks(range(len(feature_names)))
            axes[1, 1].set_yticklabels([name[:15] for name in feature_names])
            axes[1, 1].set_xlabel('Importance Score')
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        # Drift detection results
        drift_details = results['drift_details']
        if 'feature_drift' in drift_details:
            drift_features = []
            drift_p_values = []
            
            for feature, drift_info in drift_details['feature_drift'].items():
                if drift_info.get('drift_detected', False):
                    drift_features.append(feature[:15])
                    drift_p_values.append(drift_info['p_value'])
            
            if drift_features:
                axes[1, 2].barh(range(len(drift_features)), drift_p_values, color='orange')
                axes[1, 2].set_yticks(range(len(drift_features)))
                axes[1, 2].set_yticklabels(drift_features)
                axes[1, 2].set_xlabel('P-value')
                axes[1, 2].set_title('Features with Detected Drift')
                axes[1, 2].axvline(0.05, color='red', linestyle='--', alpha=0.7)
                axes[1, 2].grid(True, alpha=0.3, axis='x')
            else:
                axes[1, 2].text(0.5, 0.5, 'No Drift Detected', transform=axes[1, 2].transAxes,
                               ha='center', va='center', fontsize=16, 
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.plots_dir / f"model_performance_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plot saved to {filename}")
        
        plt.show()

def main():
    """Main execution function"""
    # Load configuration
    config = ModelConfig()
    
    # Initialize system
    system = ToxicityDetectionSystem(config)
    
    try:
        # Load data
        logger.info("Loading market data...")
        orders_df, lob_df, trades_df = system.load_data()
        
        # Train model
        logger.info("Training toxicity detection model...")
        system.train(orders_df, lob_df, trades_df, optimize_hyperparams=True)
        
        # Make predictions
        logger.info("Generating predictions...")
        results = system.predict(orders_df, lob_df, trades_df)
        
        # Create visualizations
        logger.info("Creating visualizations...")
        visualizer = Visualizer(config)
        visualizer.plot_model_performance(results, system.training_data['features'])
        
        # Save model
        model_path = system.save_model()
        
        # Print results summary
        logger.info("=== TOXICITY DETECTION RESULTS ===")
        logger.info(f"Total samples: {len(results['probabilities'])}")
        logger.info(f"Anomalies detected: {len(results['anomaly_indices'])}")
        logger.info(f"Anomaly rate: {len(results['anomaly_indices']) / len(results['probabilities']) * 100:.2f}%")
        logger.info(f"Drift detected: {results['drift_detected']}")
        logger.info(f"Model saved to: {model_path}")
        
        return system, results
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    system, results = main()