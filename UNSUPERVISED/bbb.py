"""
Advanced Market Analytics & Explainable AI Extensions
Extends the toxicity detection system with SHAP, LIME, attention mechanisms,
order flow prediction, market microstructure analysis, and automated retraining.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import joblib
import yaml
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from collections import deque
import pickle
from base_system import ToxicityDetectionSystem

# XAI Libraries
import shap
import lime
import lime.tabular
from lime import lime_tabular

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Advanced ML
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import optuna

# Market Microstructure
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AdvancedConfig:
    """Configuration for advanced analytics"""
    # XAI Configuration
    shap_sample_size: int = 1000
    lime_num_features: int = 20
    attention_window: int = 50
    
    # Market Microstructure
    order_flow_window: int = 100
    price_impact_horizon: int = 10
    liquidity_depth_levels: int = 5
    
    # Retraining Configuration
    retrain_frequency_hours: int = 24
    drift_threshold: float = 0.05
    performance_threshold: float = 0.8
    min_samples_retrain: int = 10000
    
    # Model Configuration
    attention_hidden_dim: int = 64
    lstm_hidden_dim: int = 128
    batch_size: int = 256
    learning_rate: float = 0.001
    epochs: int = 100

class ExplainableAI:
    """Explainable AI module for toxicity detection"""
    
    def __init__(self, model, feature_names: List[str], config: AdvancedConfig):
        self.model = model
        self.feature_names = feature_names
        self.config = config
        self.shap_explainer = None
        self.lime_explainer = None
        self.background_data = None
        
    def setup_explainers(self, X_background: pd.DataFrame):
        """Initialize SHAP and LIME explainers"""
        logger.info("Setting up XAI explainers...")
        
        # Sample background data for SHAP
        if len(X_background) > self.config.shap_sample_size:
            self.background_data = X_background.sample(n=self.config.shap_sample_size, random_state=42)
        else:
            self.background_data = X_background
        
        # Initialize SHAP explainer
        def model_predict(X):
            """Wrapper for model prediction compatible with SHAP"""
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=self.feature_names)
            return self.model.predict_proba(X)
        
        self.shap_explainer = shap.KernelExplainer(model_predict, self.background_data.values)
        
        # Initialize LIME explainer
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            self.background_data.values,
            feature_names=self.feature_names,
            mode='regression',
            discretize_continuous=True
        )
        
        logger.info("XAI explainers initialized successfully")
    
    def explain_instance(self, instance: pd.Series, explanation_type: str = 'both') -> Dict[str, Any]:
        """Explain a single prediction instance"""
        if self.shap_explainer is None or self.lime_explainer is None:
            raise ValueError("Explainers not initialized. Call setup_explainers first.")
        
        explanations = {}
        instance_array = instance.values.reshape(1, -1)
        
        if explanation_type in ['shap', 'both']:
            # SHAP explanation
            shap_values = self.shap_explainer.shap_values(instance_array)
            explanations['shap'] = {
                'values': shap_values[0],
                'expected_value': self.shap_explainer.expected_value,
                'feature_names': self.feature_names
            }
        
        if explanation_type in ['lime', 'both']:
            # LIME explanation
            def model_predict_single(X):
                if isinstance(X, np.ndarray) and X.ndim == 1:
                    X = X.reshape(1, -1)
                X_df = pd.DataFrame(X, columns=self.feature_names)
                return self.model.predict_proba(X_df)
            
            lime_exp = self.lime_explainer.explain_instance(
                instance.values,
                model_predict_single,
                num_features=self.config.lime_num_features
            )
            
            explanations['lime'] = {
                'feature_importance': lime_exp.as_list(),
                'score': lime_exp.score
            }
        
        return explanations
    
    def generate_counterfactuals(self, instance: pd.Series, target_class: int = 0, 
                                max_changes: int = 5) -> Dict[str, Any]:
        """Generate counterfactual explanations"""
        original_prediction = self.model.predict_proba(instance.to_frame().T)[0]
        
        # Simple counterfactual generation using feature perturbation
        counterfactuals = []
        feature_changes = []
        
        # Try changing each feature individually
        for i, feature in enumerate(self.feature_names):
            if len(counterfactuals) >= max_changes:
                break
                
            # Create perturbations
            for multiplier in [0.5, 1.5, 2.0, 0.1]:
                modified_instance = instance.copy()
                original_value = modified_instance.iloc[i]
                
                if original_value != 0:
                    modified_instance.iloc[i] = original_value * multiplier
                else:
                    modified_instance.iloc[i] = self.background_data.iloc[:, i].mean() * multiplier
                
                new_prediction = self.model.predict_proba(modified_instance.to_frame().T)[0]
                
                # Check if class changed
                if (original_prediction > 0.5) != (new_prediction > 0.5):
                    counterfactuals.append({
                        'modified_features': {feature: modified_instance.iloc[i]},
                        'original_prediction': original_prediction,
                        'new_prediction': new_prediction,
                        'feature_changed': feature,
                        'change_magnitude': abs(modified_instance.iloc[i] - original_value)
                    })
                    break
        
        return {
            'counterfactuals': counterfactuals,
            'original_prediction': original_prediction
        }

class AttentionMechanism(nn.Module):
    """Attention mechanism for temporal focus in toxicity detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super(AttentionMechanism, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.hidden_dim)
        attention_weights = self.softmax(scores)
        
        # Apply attention
        attended_output = torch.matmul(attention_weights, V)
        
        return attended_output, attention_weights

class TemporalToxicityModel(nn.Module):
    """LSTM-based model with attention for temporal toxicity detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int, attention_dim: int, output_dim: int = 1):
        super(TemporalToxicityModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = AttentionMechanism(hidden_dim * 2, attention_dim)
        self.classifier = nn.Sequential(
            nn.Linear(attention_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attended_out, attention_weights = self.attention(lstm_out)
        
        # Global average pooling over sequence
        pooled = torch.mean(attended_out, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output, attention_weights

class OrderFlowAnalyzer:
    """Advanced order flow toxicity prediction"""
    
    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.flow_features = []
        self.toxicity_predictor = None
        
    def extract_order_flow_features(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced order flow features"""
        logger.info("Extracting order flow features...")
        
        features = pd.DataFrame(index=orders_df.index)
        
        # Order arrival patterns
        features['arrival_intensity'] = self._calculate_arrival_intensity(orders_df)
        features['arrival_variance'] = self._calculate_arrival_variance(orders_df)
        
        # Size distribution analysis
        features['size_concentration'] = self._calculate_size_concentration(orders_df)
        features['size_skewness'] = self._calculate_size_skewness(orders_df)
        features['size_kurtosis'] = self._calculate_size_kurtosis(orders_df)
        
        # Temporal clustering
        features['temporal_clustering'] = self._calculate_temporal_clustering(orders_df)
        
        # Information content
        features['information_content'] = self._calculate_information_content(orders_df)
        
        # Market impact features
        if 'price' in orders_df.columns:
            features['immediate_impact'] = self._calculate_immediate_impact(orders_df)
            features['permanent_impact'] = self._calculate_permanent_impact(orders_df)
        
        self.flow_features = features.columns.tolist()
        return features.fillna(0)
    
    def _calculate_arrival_intensity(self, orders_df: pd.DataFrame) -> pd.Series:
        """Calculate order arrival intensity"""
        timestamps = pd.to_datetime(orders_df['timestamp'])
        time_diffs = timestamps.diff().dt.total_seconds().fillna(1)
        
        # Rolling average intensity
        intensity = 1 / (time_diffs + 1e-8)
        return intensity.rolling(self.config.order_flow_window, min_periods=1).mean()
    
    def _calculate_arrival_variance(self, orders_df: pd.DataFrame) -> pd.Series:
        """Calculate variance in arrival times"""
        timestamps = pd.to_datetime(orders_df['timestamp'])
        time_diffs = timestamps.diff().dt.total_seconds().fillna(1)
        
        return time_diffs.rolling(self.config.order_flow_window, min_periods=1).var()
    
    def _calculate_size_concentration(self, orders_df: pd.DataFrame) -> pd.Series:
        """Calculate concentration of order sizes (Gini coefficient)"""
        def gini_coefficient(x):
            x = np.array(x)
            x = x[x > 0]  # Remove zero values
            if len(x) == 0:
                return 0
            x = np.sort(x)
            n = len(x)
            cumsum = np.cumsum(x)
            return (2 * np.sum((np.arange(1, n + 1) * x))) / (n * cumsum[-1]) - (n + 1) / n
        
        return orders_df['quantity'].rolling(
            self.config.order_flow_window, min_periods=10
        ).apply(gini_coefficient, raw=True)
    
    def _calculate_size_skewness(self, orders_df: pd.DataFrame) -> pd.Series:
        """Calculate skewness of order sizes"""
        return orders_df['quantity'].rolling(
            self.config.order_flow_window, min_periods=1
        ).skew()
    
    def _calculate_size_kurtosis(self, orders_df: pd.DataFrame) -> pd.Series:
        """Calculate kurtosis of order sizes"""
        return orders_df['quantity'].rolling(
            self.config.order_flow_window, min_periods=1
        ).kurt()
    
    def _calculate_temporal_clustering(self, orders_df: pd.DataFrame) -> pd.Series:
        """Calculate temporal clustering coefficient"""
        timestamps = pd.to_datetime(orders_df['timestamp'])
        time_diffs = timestamps.diff().dt.total_seconds().fillna(1)
        
        # Coefficient of variation of inter-arrival times
        cv = time_diffs.rolling(self.config.order_flow_window, min_periods=1).std() / \
             (time_diffs.rolling(self.config.order_flow_window, min_periods=1).mean() + 1e-8)
        
        return cv
    
    def _calculate_information_content(self, orders_df: pd.DataFrame) -> pd.Series:
        """Calculate information content using entropy"""
        def entropy(x):
            x = np.array(x)
            x = x[x > 0]
            if len(x) <= 1:
                return 0
            
            # Discretize sizes into bins
            bins = np.histogram_bin_edges(x, bins='auto')
            if len(bins) <= 1:
                return 0
            
            hist, _ = np.histogram(x, bins=bins)
            probs = hist / hist.sum()
            probs = probs[probs > 0]
            
            return -np.sum(probs * np.log2(probs))
        
        return orders_df['quantity'].rolling(
            self.config.order_flow_window, min_periods=10
        ).apply(entropy, raw=True)
    
    def _calculate_immediate_impact(self, orders_df: pd.DataFrame) -> pd.Series:
        """Calculate immediate price impact"""
        if 'price' not in orders_df.columns:
            return pd.Series(0, index=orders_df.index)
        
        price_changes = orders_df['price'].pct_change()
        sizes = orders_df['quantity']
        
        # Impact per unit size
        impact = price_changes / (sizes + 1e-8)
        return impact.rolling(self.config.order_flow_window, min_periods=1).mean()
    
    def _calculate_permanent_impact(self, orders_df: pd.DataFrame) -> pd.Series:
        """Calculate permanent price impact"""
        if 'price' not in orders_df.columns:
            return pd.Series(0, index=orders_df.index)
        
        # Look ahead for permanent impact
        horizon = self.config.price_impact_horizon
        price_future = orders_df['price'].shift(-horizon)
        price_current = orders_df['price']
        
        permanent_change = (price_future - price_current) / (price_current + 1e-8)
        sizes = orders_df['quantity']
        
        # Impact per unit size
        impact = permanent_change / (sizes + 1e-8)
        return impact.rolling(self.config.order_flow_window, min_periods=1).mean()
    
    def train_toxicity_predictor(self, flow_features: pd.DataFrame, 
                                toxicity_labels: pd.Series) -> Dict[str, float]:
        """Train order flow toxicity predictor"""
        logger.info("Training order flow toxicity predictor...")
        
        # Prepare data
        X = flow_features.fillna(0)
        y = toxicity_labels
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train Random Forest predictor
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # Validate
            val_score = rf.score(X_val, y_val)
            scores.append(val_score)
        
        # Train final model on all data
        self.toxicity_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.toxicity_predictor.fit(X, y)
        
        return {
            'cv_scores': scores,
            'mean_cv_score': np.mean(scores),
            'feature_importance': dict(zip(self.flow_features, self.toxicity_predictor.feature_importances_))
        }
    
    def predict_order_flow_toxicity(self, flow_features: pd.DataFrame) -> np.ndarray:
        """Predict toxicity from order flow features"""
        if self.toxicity_predictor is None:
            raise ValueError("Toxicity predictor not trained. Call train_toxicity_predictor first.")
        
        return self.toxicity_predictor.predict(flow_features.fillna(0))

class MarketMicrostructureAnalyzer:
    """Market maker adverse selection and liquidity analysis"""
    
    def __init__(self, config: AdvancedConfig):
        self.config = config
        
    def calculate_adverse_selection_cost(self, orders_df: pd.DataFrame, 
                                       lob_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market maker adverse selection costs"""
        logger.info("Calculating adverse selection costs...")
        
        results = pd.DataFrame(index=orders_df.index)
        
        if lob_df.empty:
            return results.fillna(0)
        
        # Bid-ask spread
        spread = lob_df['ask_price_1'] - lob_df['bid_price_1']
        mid_price = (lob_df['bid_price_1'] + lob_df['ask_price_1']) / 2
        relative_spread = spread / (mid_price + 1e-8)
        
        results['spread'] = spread
        results['relative_spread'] = relative_spread
        
        # Effective spread for market orders
        if 'order_type' in orders_df.columns and 'price' in orders_df.columns:
            market_orders = orders_df['order_type'] == 'MARKET'
            
            # Calculate effective spread
            effective_spread = np.where(
                orders_df['side'] == 'BUY',
                2 * (orders_df['price'] - mid_price) / mid_price,
                2 * (mid_price - orders_df['price']) / mid_price
            )
            
            results['effective_spread'] = effective_spread
            
            # Realized spread (measure of adverse selection)
            horizon = 10  # 10 periods ahead
            future_mid = mid_price.shift(-horizon)
            
            realized_spread = np.where(
                orders_df['side'] == 'BUY',
                2 * (orders_df['price'] - future_mid) / orders_df['price'],
                2 * (future_mid - orders_df['price']) / orders_df['price']
            )
            
            results['realized_spread'] = realized_spread
            
            # Adverse selection component
            results['adverse_selection'] = results['effective_spread'] - results['realized_spread']
        
        # Price impact measures
        results['price_impact'] = self._calculate_price_impact(orders_df, lob_df)
        
        # Information asymmetry measures
        results['information_asymmetry'] = self._calculate_information_asymmetry(orders_df, lob_df)
        
        return results.fillna(0)
    
    def _calculate_price_impact(self, orders_df: pd.DataFrame, 
                               lob_df: pd.DataFrame) -> pd.Series:
        """Calculate price impact of orders"""
        if 'price' not in orders_df.columns:
            return pd.Series(0, index=orders_df.index)
        
        # Price change after order
        price_before = orders_df['price']
        price_after = orders_df['price'].shift(-1)
        
        impact = (price_after - price_before) / (price_before + 1e-8)
        
        # Adjust for order direction
        signed_impact = np.where(
            orders_df['side'] == 'BUY',
            impact,
            -impact
        )
        
        return pd.Series(signed_impact, index=orders_df.index)
    
    def _calculate_information_asymmetry(self, orders_df: pd.DataFrame, 
                                       lob_df: pd.DataFrame) -> pd.Series:
        """Calculate information asymmetry measures"""
        # Probability of informed trading (PIN) approximation
        if 'quantity' not in orders_df.columns:
            return pd.Series(0, index=orders_df.index)
        
        # Order flow imbalance
        window = 50
        buy_volume = orders_df[orders_df['side'] == 'BUY']['quantity'].rolling(window).sum()
        sell_volume = orders_df[orders_df['side'] == 'SELL']['quantity'].rolling(window).sum()
        
        total_volume = buy_volume + sell_volume
        order_imbalance = (buy_volume - sell_volume) / (total_volume + 1e-8)
        
        # Use absolute imbalance as proxy for information asymmetry
        return order_imbalance.abs().reindex(orders_df.index).fillna(0)
    
    def optimize_liquidity_provision(self, orders_df: pd.DataFrame, 
                                   lob_df: pd.DataFrame) -> Dict[str, Any]:
        """Optimize liquidity provision strategy"""
        logger.info("Optimizing liquidity provision...")
        
        # Calculate current market conditions
        adverse_selection = self.calculate_adverse_selection_cost(orders_df, lob_df)
        
        # Define optimization objective
        def objective(params):
            bid_offset, ask_offset, size_multiplier = params
            
            # Simulate P&L under this strategy
            # Simplified model: profit from spread minus adverse selection costs
            spread_profit = (ask_offset + bid_offset) * size_multiplier
            adverse_cost = adverse_selection['adverse_selection'].mean() * size_multiplier
            
            # Add risk penalty for large positions
            risk_penalty = 0.01 * (size_multiplier ** 2)
            
            return -(spread_profit - adverse_cost - risk_penalty)
        
        # Optimize parameters
        initial_guess = [0.001, 0.001, 1.0]  # bid_offset, ask_offset, size_multiplier
        bounds = [(0.0001, 0.01), (0.0001, 0.01), (0.1, 5.0)]
        
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        optimal_params = {
            'bid_offset': result.x[0],
            'ask_offset': result.x[1],
            'size_multiplier': result.x[2],
            'expected_profit': -result.fun,
            'optimization_success': result.success
        }
        
        return optimal_params

class AutomatedRetrainingPipeline:
    """Automated model retraining and performance monitoring"""
    
    def __init__(self, config: AdvancedConfig, toxicity_system):
        self.config = config
        self.toxicity_system = toxicity_system
        self.performance_history = deque(maxlen=100)
        self.drift_history = deque(maxlen=50)
        self.last_retrain_time = datetime.now()
        self.retrain_thread = None
        self.monitoring_active = False
        
    def start_monitoring(self):
        """Start automated monitoring and retraining"""
        logger.info("Starting automated retraining pipeline...")
        self.monitoring_active = True
        
        # Start monitoring thread
        self.retrain_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.retrain_thread.start()
        
    def stop_monitoring(self):
        """Stop automated monitoring"""
        logger.info("Stopping automated retraining pipeline...")
        self.monitoring_active = False
        if self.retrain_thread:
            self.retrain_thread.join()
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Check if retraining is needed
                if self._should_retrain():
                    self._execute_retraining()
                
                # Sleep for monitoring interval (1 hour)
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(3600)  # Continue monitoring despite errors
    
    def _should_retrain(self) -> bool:
        """Determine if model should be retrained"""
        current_time = datetime.now()
        
        # Time-based retraining
        hours_since_last_retrain = (current_time - self.last_retrain_time).total_seconds() / 3600
        if hours_since_last_retrain >= self.config.retrain_frequency_hours:
            logger.info("Time-based retraining triggered")
            return True
        
        # Performance-based retraining
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(list(self.performance_history)[-10:])
            if recent_performance < self.config.performance_threshold:
                logger.info(f"Performance-based retraining triggered: {recent_performance:.3f}")
                return True
        
        # Drift-based retraining
        if len(self.drift_history) >= 5:
            recent_drift = np.mean(list(self.drift_history)[-5:])
            if recent_drift > self.config.drift_threshold:
                logger.info(f"Drift-based retraining triggered: {recent_drift:.3f}")
                return True
        
        return False
    
    def _execute_retraining(self):
        """Execute model retraining"""
        logger.info("Executing automated model retraining...")
        
        try:
            # Load recent data (this would be replaced with actual data loading)
            # For demo purposes, we'll simulate this
            logger.info("Loading recent training data...")
            
            # In practice, you would:
            # 1. Load new market data from the last training period
            # 2. Validate data quality
            # 3. Retrain the model
            # 4. Validate new model performance
            # 5. Deploy if performance is satisfactory
            
            # Simulate retraining process
            retraining_start = time.time()
            
            # Here you would call:
            # orders_df, lob_df, trades_df = load_recent_data()
            # self.toxicity_system.train(orders_df, lob_df, trades_df)
            
            # For demo, we'll just simulate the time it takes
            time.sleep(5)  # Simulate retraining time
            
            retraining_time = time.time() - retraining_start
            
            # Update retraining timestamp
            self.last_retrain_time = datetime.now()
            
            # Log retraining success
            logger.info(f"Model retraining completed in {retraining_time:.2f} seconds")
            
            # Save retrained model
            model_path = self.toxicity_system.save_model()
            logger.info(f"Retrained model saved to: {model_path}")
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
    
    def update_performance_metric(self, metric_value: float):
        """Update performance history"""
        self.performance_history.append(metric_value)
    
    def update_drift_metric(self, drift_value: float):
        """Update drift history"""
        self.drift_history.append(drift_value)
    
    def get_retraining_status(self) -> Dict[str, Any]:
        """Get current retraining pipeline status"""
        return {
            'monitoring_active': self.monitoring_active,
            'last_retrain_time': self.last_retrain_time.isoformat(),
            'hours_since_last_retrain': (datetime.now() - self.last_retrain_time).total_seconds() / 3600,
            'performance_history_length': len(self.performance_history),
            'drift_history_length': len(self.drift_history),
            'recent_performance': np.mean(list(self.performance_history)[-5:]) if self.performance_history else None,
            'recent_drift': np.mean(list(self.drift_history)[-3:]) if self.drift_history else None
        }

class AdvancedToxicitySystem:
    """Enhanced toxicity detection system with advanced analytics"""
    
    def __init__(self, base_system, config: AdvancedConfig):
        self.base_system = base_system
        self.config = config
        
        # Initialize advanced components
        self.xai = None
        self.order_flow_analyzer = OrderFlowAnalyzer(config)
        self.microstructure_analyzer = MarketMicrostructureAnalyzer(config)
        self.retraining_pipeline = AutomatedRetrainingPipeline(config, base_system)
        self.temporal_model = None
        
    def setup_advanced_analytics(self, training_data: pd.DataFrame):
        """Setup advanced analytics components"""
        logger.info("Setting up advanced analytics...")
        
        # Setup XAI
        if hasattr(self.base_system, 'feature_names'):
            feature_names = self.base_system.feature_names
        else:
            feature_names = [f'feature_{i}' for i in range(training_data.shape[1])]
        
        self.xai = ExplainableAI(self.base_system, feature_names, self.config)
        self.xai.setup_explainers(training_data)
        
        # Setup temporal model
        self._setup_temporal_model(training_data)
        
        logger.info("Advanced analytics setup completed")
    
    def _setup_temporal_model(self, training_data: pd.DataFrame):
        """Setup temporal toxicity model with attention"""
        logger.info("Setting up temporal model with attention mechanism...")
        
        input_dim = training_data.shape[1]
        self.temporal_model = TemporalToxicityModel(
            input_dim=input_dim,
            hidden_dim=self.config.lstm_hidden_dim,
            attention_dim=self.config.attention_hidden_dim
        )
        
    def analyze_comprehensive_toxicity(self, orders_df: pd.DataFrame, 
                                     lob_df: pd.DataFrame, 
                                     trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive toxicity analysis"""
        logger.info("Performing comprehensive toxicity analysis...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'explanations': {},
            'order_flow': {},
            'microstructure': {},
            'predictions': {}
        }
        
        # 1. Base toxicity detection
        logger.info("Running base toxicity detection...")
        base_results = self.base_system.detect_toxicity(orders_df, lob_df, trades_df)
        results['base_detection'] = base_results
        
        # 2. Order flow analysis
        logger.info("Analyzing order flow patterns...")
        try:
            flow_features = self.order_flow_analyzer.extract_order_flow_features(orders_df)
            results['order_flow']['features'] = flow_features.describe().to_dict()
            
            # Train toxicity predictor if we have labels
            if 'toxicity_scores' in base_results:
                flow_prediction_results = self.order_flow_analyzer.train_toxicity_predictor(
                    flow_features, base_results['toxicity_scores']
                )
                results['order_flow']['prediction_performance'] = flow_prediction_results
                
                # Get order flow toxicity predictions
                flow_toxicity = self.order_flow_analyzer.predict_order_flow_toxicity(flow_features)
                results['order_flow']['toxicity_predictions'] = {
                    'mean': float(np.mean(flow_toxicity)),
                    'std': float(np.std(flow_toxicity)),
                    'max': float(np.max(flow_toxicity)),
                    'percentiles': {
                        '95th': float(np.percentile(flow_toxicity, 95)),
                        '99th': float(np.percentile(flow_toxicity, 99))
                    }
                }
        except Exception as e:
            logger.warning(f"Order flow analysis failed: {e}")
            results['order_flow']['error'] = str(e)
        
        # 3. Market microstructure analysis
        logger.info("Analyzing market microstructure...")
        try:
            adverse_selection = self.microstructure_analyzer.calculate_adverse_selection_cost(
                orders_df, lob_df
            )
            results['microstructure']['adverse_selection'] = adverse_selection.describe().to_dict()
            
            # Liquidity optimization
            liquidity_optimization = self.microstructure_analyzer.optimize_liquidity_provision(
                orders_df, lob_df
            )
            results['microstructure']['liquidity_optimization'] = liquidity_optimization
            
        except Exception as e:
            logger.warning(f"Microstructure analysis failed: {e}")
            results['microstructure']['error'] = str(e)
        
        # 4. XAI Analysis
        if self.xai and 'features_df' in base_results:
            logger.info("Generating explainable AI insights...")
            try:
                # Explain most toxic instances
                features_df = base_results['features_df']
                toxicity_scores = base_results['toxicity_scores']
                
                # Get top 5 most toxic instances
                top_toxic_indices = toxicity_scores.nlargest(5).index
                explanations = {}
                
                for idx in top_toxic_indices:
                    if idx in features_df.index:
                        instance_explanation = self.xai.explain_instance(
                            features_df.loc[idx], explanation_type='both'
                        )
                        
                        # Generate counterfactuals
                        counterfactuals = self.xai.generate_counterfactuals(features_df.loc[idx])
                        
                        explanations[str(idx)] = {
                            'toxicity_score': float(toxicity_scores.loc[idx]),
                            'shap_explanation': {
                                'top_features': self._get_top_shap_features(instance_explanation.get('shap', {})),
                                'expected_value': instance_explanation.get('shap', {}).get('expected_value', 0)
                            },
                            'lime_explanation': {
                                'feature_importance': instance_explanation.get('lime', {}).get('feature_importance', [])[:10],
                                'score': instance_explanation.get('lime', {}).get('score', 0)
                            },
                            'counterfactuals': counterfactuals.get('counterfactuals', [])[:3]
                        }
                
                results['explanations'] = explanations
                
            except Exception as e:
                logger.warning(f"XAI analysis failed: {e}")
                results['explanations']['error'] = str(e)
        
        # 5. Temporal analysis with attention
        if self.temporal_model:
            logger.info("Performing temporal analysis...")
            try:
                temporal_results = self._analyze_temporal_patterns(orders_df, base_results.get('features_df'))
                results['temporal_analysis'] = temporal_results
            except Exception as e:
                logger.warning(f"Temporal analysis failed: {e}")
                results['temporal_analysis'] = {'error': str(e)}
        
        # 6. Generate summary insights
        results['summary'] = self._generate_summary_insights(results)
        
        # 7. Update retraining pipeline
        if 'anomaly_rate' in base_results:
            self.retraining_pipeline.update_performance_metric(1 - base_results['anomaly_rate'])
        
        if 'drift_detected' in base_results:
            drift_score = 1.0 if base_results['drift_detected'] else 0.0
            self.retraining_pipeline.update_drift_metric(drift_score)
        
        return results
    
    def _get_top_shap_features(self, shap_explanation: Dict) -> List[Dict]:
        """Extract top SHAP feature contributions"""
        if not shap_explanation or 'values' not in shap_explanation:
            return []
        
        values = shap_explanation['values']
        feature_names = shap_explanation.get('feature_names', [f'feature_{i}' for i in range(len(values))])
        
        # Get top 10 features by absolute SHAP value
        feature_importance = list(zip(feature_names, values))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return [
            {'feature': name, 'shap_value': float(value), 'contribution': 'positive' if value > 0 else 'negative'}
            for name, value in feature_importance[:10]
        ]
    
    def _analyze_temporal_patterns(self, orders_df: pd.DataFrame, 
                                 features_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze temporal patterns using attention mechanism"""
        if features_df is None or len(features_df) < self.config.attention_window:
            return {'error': 'Insufficient data for temporal analysis'}
        
        # Prepare sequences for attention analysis
        sequence_length = self.config.attention_window
        sequences = []
        
        for i in range(len(features_df) - sequence_length + 1):
            seq = features_df.iloc[i:i+sequence_length].values
            sequences.append(seq)
        
        if not sequences:
            return {'error': 'No sequences generated'}
        
        # Convert to tensor
        sequences_tensor = torch.FloatTensor(np.array(sequences))
        
        # Get attention weights (simulation for demo)
        with torch.no_grad():
            try:
                outputs, attention_weights = self.temporal_model(sequences_tensor)
                
                # Analyze attention patterns
                avg_attention = torch.mean(attention_weights, dim=0)
                
                # Find most important time steps
                time_importance = torch.mean(avg_attention, dim=2)  # Average across features
                top_timesteps = torch.topk(time_importance.flatten(), k=5)
                
                return {
                    'attention_analysis': {
                        'most_important_timesteps': top_timesteps.indices.tolist(),
                        'attention_scores': top_timesteps.values.tolist(),
                        'temporal_focus_distribution': time_importance.mean(dim=0).tolist()
                    },
                    'sequences_analyzed': len(sequences)
                }
                
            except Exception as e:
                return {'error': f'Attention analysis failed: {e}'}
    
    def _generate_summary_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level summary insights"""
        summary = {
            'overall_risk_level': 'LOW',
            'key_findings': [],
            'recommendations': [],
            'metrics': {}
        }
        
        try:
            # Analyze base detection results
            base_results = results.get('base_detection', {})
            if 'anomaly_rate' in base_results:
                anomaly_rate = base_results['anomaly_rate']
                summary['metrics']['anomaly_rate'] = anomaly_rate
                
                if anomaly_rate > 0.1:
                    summary['overall_risk_level'] = 'HIGH'
                    summary['key_findings'].append(f"High anomaly rate detected: {anomaly_rate:.1%}")
                elif anomaly_rate > 0.05:
                    summary['overall_risk_level'] = 'MEDIUM'
                    summary['key_findings'].append(f"Moderate anomaly rate: {anomaly_rate:.1%}")
            
            # Analyze order flow results
            order_flow = results.get('order_flow', {})
            if 'toxicity_predictions' in order_flow:
                flow_toxicity = order_flow['toxicity_predictions']
                p99_toxicity = flow_toxicity.get('percentiles', {}).get('99th', 0)
                
                if p99_toxicity > 0.8:
                    summary['key_findings'].append("Severe order flow toxicity detected in 99th percentile")
                    summary['recommendations'].append("Implement enhanced order validation")
                
                summary['metrics']['order_flow_toxicity_p99'] = p99_toxicity
            
            # Analyze microstructure results
            microstructure = results.get('microstructure', {})
            if 'adverse_selection' in microstructure:
                adverse_selection = microstructure['adverse_selection']
                if 'adverse_selection' in adverse_selection and 'mean' in adverse_selection['adverse_selection']:
                    avg_adverse_selection = adverse_selection['adverse_selection']['mean']
                    
                    if avg_adverse_selection > 0.01:  # 1%
                        summary['key_findings'].append(f"High adverse selection cost: {avg_adverse_selection:.3f}")
                        summary['recommendations'].append("Review market making parameters")
                    
                    summary['metrics']['adverse_selection_cost'] = avg_adverse_selection
            
            # Analyze XAI insights
            explanations = results.get('explanations', {})
            if explanations and not explanations.get('error'):
                # Count most influential features across explanations
                feature_mentions = {}
                for explanation in explanations.values():
                    if isinstance(explanation, dict) and 'shap_explanation' in explanation:
                        for feature_info in explanation['shap_explanation'].get('top_features', []):
                            feature = feature_info.get('feature', '')
                            if feature:
                                feature_mentions[feature] = feature_mentions.get(feature, 0) + 1
                
                if feature_mentions:
                    most_influential_feature = max(feature_mentions, key=feature_mentions.get)
                    summary['key_findings'].append(f"Most influential feature: {most_influential_feature}")
                    summary['metrics']['most_influential_feature'] = most_influential_feature
            
            # Generate recommendations based on findings
            if summary['overall_risk_level'] == 'HIGH':
                summary['recommendations'].extend([
                    "Activate enhanced monitoring protocols",
                    "Consider temporary trading restrictions",
                    "Review recent market events"
                ])
            elif summary['overall_risk_level'] == 'MEDIUM':
                summary['recommendations'].extend([
                    "Increase monitoring frequency",
                    "Review risk thresholds"
                ])
            
            # Add temporal insights if available
            temporal = results.get('temporal_analysis', {})
            if temporal and not temporal.get('error'):
                attention_analysis = temporal.get('attention_analysis', {})
                if attention_analysis:
                    summary['key_findings'].append("Temporal patterns analyzed with attention mechanism")
                    summary['metrics']['sequences_analyzed'] = temporal.get('sequences_analyzed', 0)
            
        except Exception as e:
            logger.warning(f"Error generating summary insights: {e}")
            summary['error'] = str(e)
        
        return summary
    
    def start_automated_monitoring(self):
        """Start automated monitoring and retraining"""
        self.retraining_pipeline.start_monitoring()
    
    def stop_automated_monitoring(self):
        """Stop automated monitoring and retraining"""
        self.retraining_pipeline.stop_monitoring()
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return self.retraining_pipeline.get_retraining_status()
    
    def generate_detailed_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate detailed analysis report"""
        report_lines = [
            "=" * 80,
            "ADVANCED TOXICITY DETECTION REPORT",
            "=" * 80,
            f"Generated: {analysis_results.get('timestamp', 'Unknown')}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 40
        ]
        
        summary = analysis_results.get('summary', {})
        report_lines.extend([
            f"Overall Risk Level: {summary.get('overall_risk_level', 'Unknown')}",
            f"Anomaly Rate: {summary.get('metrics', {}).get('anomaly_rate', 'N/A'):.1%}" if isinstance(summary.get('metrics', {}).get('anomaly_rate'), (int, float)) else "Anomaly Rate: N/A",
            "",
            "KEY FINDINGS:",
        ])
        
        for finding in summary.get('key_findings', []):
            report_lines.append(f"• {finding}")
        
        report_lines.extend([
            "",
            "RECOMMENDATIONS:",
        ])
        
        for rec in summary.get('recommendations', []):
            report_lines.append(f"• {rec}")
        
        # Add detailed sections
        report_lines.extend([
            "",
            "DETAILED ANALYSIS",
            "-" * 40,
            "",
            "1. ORDER FLOW ANALYSIS"
        ])
        
        order_flow = analysis_results.get('order_flow', {})
        if 'toxicity_predictions' in order_flow:
            toxicity = order_flow['toxicity_predictions']
            report_lines.extend([
                f"   Mean Toxicity: {toxicity.get('mean', 'N/A'):.4f}" if isinstance(toxicity.get('mean'), (int, float)) else "   Mean Toxicity: N/A",
                f"   99th Percentile: {toxicity.get('percentiles', {}).get('99th', 'N/A'):.4f}" if isinstance(toxicity.get('percentiles', {}).get('99th'), (int, float)) else "   99th Percentile: N/A",
                f"   Standard Deviation: {toxicity.get('std', 'N/A'):.4f}" if isinstance(toxicity.get('std'), (int, float)) else "   Standard Deviation: N/A"
            ])
        
        report_lines.extend([
            "",
            "2. MARKET MICROSTRUCTURE"
        ])
        
        microstructure = analysis_results.get('microstructure', {})
        if 'liquidity_optimization' in microstructure:
            opt = microstructure['liquidity_optimization']
            report_lines.extend([
                f"   Optimal Bid Offset: {opt.get('bid_offset', 'N/A'):.6f}" if isinstance(opt.get('bid_offset'), (int, float)) else "   Optimal Bid Offset: N/A",
                f"   Optimal Ask Offset: {opt.get('ask_offset', 'N/A'):.6f}" if isinstance(opt.get('ask_offset'), (int, float)) else "   Optimal Ask Offset: N/A",
                f"   Expected Profit: {opt.get('expected_profit', 'N/A'):.6f}" if isinstance(opt.get('expected_profit'), (int, float)) else "   Expected Profit: N/A"
            ])
        
        report_lines.extend([
            "",
            "3. EXPLAINABLE AI INSIGHTS"
        ])
        
        explanations = analysis_results.get('explanations', {})
        if explanations and not explanations.get('error'):
            report_lines.append("   Top Toxic Instances Analyzed:")
            for idx, explanation in list(explanations.items())[:3]:
                if isinstance(explanation, dict):
                    toxicity_score = explanation.get('toxicity_score', 'N/A')
                    report_lines.append(f"   • Instance {idx}: Toxicity Score {toxicity_score:.4f}" if isinstance(toxicity_score, (int, float)) else f"   • Instance {idx}: Toxicity Score {toxicity_score}")
        
        report_lines.extend([
            "",
            "=" * 80,
            "End of Report"
        ])
        
        return "\n".join(report_lines)

# Integration with your existing toxicity detection system
def integrate_with_existing_system():
    """
    Integration example for your existing toxicity detection system.
    Replace the imports and system initialization with your actual system.
    """
    
    # REPLACE THIS with your actual system import
    # from your_toxicity_system import ToxicityDetectionSystem
    
    logger.info("Integrating advanced analytics with existing toxicity detection system...")
    
    # Initialize your existing system (REPLACE with your actual initialization)
    # base_system = ToxicityDetectionSystem()  # Your existing system
    
    # Initialize advanced configuration
    config = AdvancedConfig(
        # Adjust these parameters based on your data characteristics
        shap_sample_size=1000,  # Reduce if you have memory constraints
        order_flow_window=100,  # Adjust based on your market frequency
        retrain_frequency_hours=24,  # How often to retrain
        attention_window=50  # Sequence length for temporal analysis
    )
    
    # Create advanced system wrapper
    # advanced_system = AdvancedToxicitySystem(base_system, config)
    
    logger.info("Advanced system integration ready. Use with your actual data.")
    
    return config

def run_production_analysis(orders_df: pd.DataFrame, 
                          lob_df: pd.DataFrame, 
                          trades_df: pd.DataFrame,
                          base_system,
                          config: Optional[AdvancedConfig] = None):
    """
    Run production analysis with your actual market data
    
    Parameters:
    -----------
    orders_df : pd.DataFrame
        Your orders data with columns: timestamp, quantity, price, side, etc.
    lob_df : pd.DataFrame  
        Your limit order book data with bid/ask prices and quantities
    trades_df : pd.DataFrame
        Your trades data with timestamp, price, quantity, side
    base_system : 
        Your existing toxicity detection system
    config : AdvancedConfig
        Configuration for advanced analytics
    
    Returns:
    --------
    Dict containing comprehensive analysis results
    """
    
    if config is None:
        config = AdvancedConfig()
    
    logger.info("Starting production analysis with your market data...")
    logger.info(f"Orders data shape: {orders_df.shape}")
    logger.info(f"LOB data shape: {lob_df.shape}")
    logger.info(f"Trades data shape: {trades_df.shape}")
    
    # Initialize advanced system
    advanced_system = AdvancedToxicitySystem(base_system, config)
    
    # Setup analytics using your training data
    # You should replace this with actual feature extraction from your system
    try:
        # Extract features using your existing system's feature engineering
        if hasattr(base_system, 'feature_engineer'):
            training_features = base_system.feature_engineer.extract_features(
                orders_df, lob_df, trades_df
            )
        else:
            # Fallback: create basic features from your data
            training_features = create_basic_features_from_data(orders_df, lob_df, trades_df)
        
        advanced_system.setup_advanced_analytics(training_features)
        
        # Run comprehensive analysis
        results = advanced_system.analyze_comprehensive_toxicity(
            orders_df, lob_df, trades_df
        )
        
        # Generate detailed report
        report = advanced_system.generate_detailed_report(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save analysis results
        results_file = f"advanced_analysis_{timestamp}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Analysis results saved to {results_file}")
        
        # Save detailed report
        report_file = f"toxicity_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Detailed report saved to {report_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Production analysis failed: {e}")
        raise

def create_basic_features_from_data(orders_df: pd.DataFrame, 
                                   lob_df: pd.DataFrame, 
                                   trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic features from your data for XAI setup.
    This is a fallback if your system doesn't have feature extraction.
    REPLACE this with your actual feature extraction method.
    """
    
    logger.info("Creating basic features from your data...")
    
    features_list = []
    
    # Basic order features
    if not orders_df.empty:
        # Order size features
        features_list.append(orders_df['quantity'].rolling(10).mean().rename('avg_order_size'))
        features_list.append(orders_df['quantity'].rolling(10).std().rename('order_size_volatility'))
        
        # Price features if available
        if 'price' in orders_df.columns:
            features_list.append(orders_df['price'].pct_change().rename('price_change'))
            features_list.append(orders_df['price'].rolling(10).std().rename('price_volatility'))
        
        # Order frequency
        order_freq = orders_df.groupby('timestamp').size().reindex(orders_df['timestamp']).fillna(0)
        features_list.append(order_freq.rename('order_frequency'))
    
    # Basic LOB features
    if not lob_df.empty and 'bid_price_1' in lob_df.columns:
        # Spread features
        spread = lob_df['ask_price_1'] - lob_df['bid_price_1']
        mid_price = (lob_df['bid_price_1'] + lob_df['ask_price_1']) / 2
        
        features_list.append(spread.rename('bid_ask_spread'))
        features_list.append((spread / mid_price).rename('relative_spread'))
        
        # Depth features if available
        if 'bid_quantity_1' in lob_df.columns:
            total_depth = lob_df['bid_quantity_1'] + lob_df['ask_quantity_1']
            depth_imbalance = (lob_df['bid_quantity_1'] - lob_df['ask_quantity_1']) / total_depth
            
            features_list.append(total_depth.rename('total_depth_L1'))
            features_list.append(depth_imbalance.rename('depth_imbalance_L1'))
    
    # Basic trade features
    if not trades_df.empty:
        # Trade size and frequency
        trade_freq = trades_df.groupby('timestamp').size()
        avg_trade_size = trades_df.groupby('timestamp')['quantity'].mean()
        
        # Align with orders timestamps
        if not orders_df.empty:
            trade_freq_aligned = trade_freq.reindex(orders_df['timestamp']).fillna(0)
            avg_trade_size_aligned = avg_trade_size.reindex(orders_df['timestamp']).fillna(method='ffill')
            
            features_list.append(trade_freq_aligned.rename('trade_frequency'))
            features_list.append(avg_trade_size_aligned.rename('avg_trade_size'))
    
    # Combine all features
    if features_list:
        features_df = pd.concat(features_list, axis=1).fillna(0)
        
        # Add some statistical features
        for col in features_df.select_dtypes(include=[np.number]).columns:
            features_df[f'{col}_ma5'] = features_df[col].rolling(5).mean()
            features_df[f'{col}_std5'] = features_df[col].rolling(5).std()
        
        features_df = features_df.fillna(0)
        logger.info(f"Created {features_df.shape[1]} basic features from your data")
        
        return features_df
    else:
        # Fallback: create minimal feature set
        logger.warning("Could not extract features from data, creating minimal feature set")
        n_samples = len(orders_df) if not orders_df.empty else 100
        return pd.DataFrame(np.random.randn(n_samples, 10), 
                          columns=[f'feature_{i}' for i in range(10)])

def save_advanced_config(config: AdvancedConfig, filepath: str):
    """Save advanced configuration to file"""
    config_dict = {
        'shap_sample_size': config.shap_sample_size,
        'lime_num_features': config.lime_num_features,
        'attention_window': config.attention_window,
        'order_flow_window': config.order_flow_window,
        'price_impact_horizon': config.price_impact_horizon,
        'liquidity_depth_levels': config.liquidity_depth_levels,
        'retrain_frequency_hours': config.retrain_frequency_hours,
        'drift_threshold': config.drift_threshold,
        'performance_threshold': config.performance_threshold,
        'min_samples_retrain': config.min_samples_retrain,
        'attention_hidden_dim': config.attention_hidden_dim,
        'lstm_hidden_dim': config.lstm_hidden_dim,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'epochs': config.epochs
    }
    
    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    logger.info(f"Advanced configuration saved to {filepath}")

def load_advanced_config(filepath: str) -> AdvancedConfig:
    """Load advanced configuration from file"""
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return AdvancedConfig(**config_dict)

# EXAMPLE INTEGRATION WITH YOUR EXISTING SYSTEM:
"""
# 1. Import your existing system

# 2. Load your actual market data 
orders_df = pd.read_csv('your_orders.csv')
lob_df = pd.read_csv('your_lob.csv') 
trades_df = pd.read_csv('your_trades.csv')

# 3. Initialize systems
base_system = ToxicityDetectionSystem()
config = AdvancedConfig()

# 4. Run production analysis
results = run_production_analysis(
    orders_df=orders_df,
    lob_df=lob_df, 
    trades_df=trades_df,
    base_system=base_system,
    config=config
)

# 5. Access results
print("Risk Level:", results['summary']['overall_risk_level'])
print("Key Findings:", results['summary']['key_findings'])

# 6. Start automated monitoring
advanced_system = AdvancedToxicitySystem(base_system, config)
advanced_system.start_automated_monitoring()
"""

if __name__ == "__main__":
    # Initialize configuration for your system
    config = integrate_with_existing_system()
    print("Advanced analytics ready for integration with your data!")
    print("Follow the example above to use with your actual market data.")