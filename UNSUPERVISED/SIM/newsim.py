#newsim.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import joblib
import os
from collections import deque, defaultdict
import time
from sklearn.preprocessing import StandardScaler
import sys
import os
import random
from model import LSTMAutoencoder
warnings.filterwarnings('ignore')

# Import base simulation components (assuming they're available)
from complete_enhanced_simulation import (
    Order, Trade, LimitOrderBook, Agent, 
    ImprovedInformedTrader, SmartNoiseTrader, BUY, SELL, LIMIT, MARKET,
    INITIAL_PRICE, MIN_PRICE, TIME_STEPS, NUM_AGENTS
)

class MLToxicityPredictor:
    """
    Wrapper for the trained toxicity detection model
    Provides real-time toxicity scoring for market making
    """
    
    def __init__(self, model_path=None):
        self.model = None
        self.feature_names = []
        self.scaler = None
        self.is_ready = False
        self.feature_buffer = deque(maxlen=100)  # For sequence features
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained toxicity detection model"""
        try:
            print(f"Loading ML toxicity model from: {model_path}")
            
            # Import required classes locally
            try:
                import joblib
                model_package = joblib.load(model_path)
            except Exception as e:
                print(f"Failed to load model package: {e}")
                # Create fallback dummy model
                model_package = {
                    'models': {},
                    'feature_selector': ['order_size', 'is_market_order', 'is_buy', 'mid_price'],
                    'scalers': {'main': StandardScaler()},
                    'meta_model': None,
                    'has_meta_model': False,
                    'n_detectors': 0
                }
            
            self.model = model_package
            self.feature_names = model_package.get('feature_selector', ['order_size', 'is_market_order', 'is_buy', 'mid_price'])
            
            # Handle scaler
            scalers = model_package.get('scalers', {})
            if 'main' in scalers:
                self.scaler = scalers['main']
            else:
                self.scaler = StandardScaler()
                # Fit with dummy data
                dummy_data = np.random.randn(100, len(self.feature_names))
                self.scaler.fit(dummy_data)
            
            print(f"✓ Model loaded successfully:")
            print(f"  - {model_package.get('n_detectors', 0)} detectors")
            print(f"  - {len(self.feature_names)} features")
            print(f"  - Meta-model: {'Yes' if model_package.get('has_meta_model') else 'No'}")
            
            # Check if we have any working models
            working_models = len([m for m in model_package.get('models', {}).values() if m is not None])
            if working_models > 0:
                self.is_ready = True
                print(f"  - {working_models} working detectors available")
            else:
                print("  - No working detectors found, using fallback scoring")
                self.is_ready = False
                
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            # Create minimal fallback
            self.feature_names = ['order_size', 'is_market_order', 'is_buy', 'mid_price']
            self.scaler = StandardScaler()
            dummy_data = np.random.randn(100, len(self.feature_names))
            self.scaler.fit(dummy_data)
            self.model = {'models': {}, 'meta_model': None}
            self.is_ready = False

    
    def extract_order_features(self, order, market_context):
        """
        Extract features from current order and market context
        Mirrors the feature engineering from training
        """
        if not self.is_ready:
            return np.array([0.5])  # Return neutral score if model not ready
        
        try:
            features = {}
            
            # Basic order features
            features['order_size'] = order.quantity
            features['log_order_size'] = np.log1p(order.quantity)
            features['is_market_order'] = 1 if order.type == MARKET else 0
            features['is_buy'] = 1 if order.side == BUY else 0
            
            # Market context features
            if market_context:
                mid_price = market_context.get('mid_price', INITIAL_PRICE)
                spread = market_context.get('spread', 0)
                recent_volumes = market_context.get('recent_volumes', [1])
                recent_prices = market_context.get('recent_prices', [mid_price])
                
                features['mid_price'] = mid_price
                features['log_mid_price'] = np.log(max(mid_price, 0.01))
                features['spread'] = spread
                features['spread_bps'] = (spread / mid_price) * 10000 if mid_price > 0 else 0
                
                # Price features
                if len(recent_prices) > 1:
                    features['price_return'] = (recent_prices[-1] - recent_prices[-2]) / recent_prices[-2]
                    features['price_volatility'] = np.std(recent_prices[-10:]) if len(recent_prices) >= 10 else 0
                    features['price_momentum'] = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if len(recent_prices) > 5 else 0
                else:
                    features['price_return'] = 0
                    features['price_volatility'] = 0
                    features['price_momentum'] = 0
                
                # Volume features
                if recent_volumes:
                    features['avg_volume'] = np.mean(recent_volumes)
                    features['volume_ratio'] = order.quantity / np.mean(recent_volumes) if np.mean(recent_volumes) > 0 else 1
                else:
                    features['avg_volume'] = order.quantity
                    features['volume_ratio'] = 1
                
                # Time-based features
                features['time_since_last_trade'] = market_context.get('time_since_last_trade', 1)
                features['order_arrival_rate'] = market_context.get('order_arrival_rate', 0.1)
                
                # Market microstructure
                features['order_book_imbalance'] = market_context.get('imbalance', 0)
                features['depth_ratio'] = market_context.get('depth_ratio', 1)
                
                # Distance from mid (for limit orders)
                if order.type == LIMIT and order.price:
                    features['distance_from_mid'] = abs(order.price - mid_price) / mid_price
                    features['is_aggressive'] = 1 if ((order.side == BUY and order.price >= mid_price) or 
                                                    (order.side == SELL and order.price <= mid_price)) else 0
                else:
                    features['distance_from_mid'] = 0
                    features['is_aggressive'] = 1  # Market orders are aggressive
            
            # Convert to array matching training features
            feature_vector = np.zeros(len(self.feature_names))
            for i, feature_name in enumerate(self.feature_names):
                if feature_name in features:
                    feature_vector[i] = features[feature_name]
            
            # Add to buffer for sequence features
            self.feature_buffer.append(feature_vector)
            
            return feature_vector.reshape(1, -1)
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.array([[0.5] * len(self.feature_names)])  # Return neutral features
    
    def predict_toxicity(self, order, market_context):
        """
        Predict toxicity score for an order
        Returns score between 0 (safe) and 1 (toxic)
        """
        try:
            # Extract features
            features = self.extract_order_features(order, market_context)
            
            if not self.is_ready or not self.model.get('models'):
                # Fallback: simple heuristic scoring
                return self._fallback_toxicity_score(order, market_context, features)
            
            # Scale features
            if hasattr(self.scaler, 'transform') and features.shape[1] > 0:
                try:
                    features_scaled = self.scaler.transform(features)
                except:
                    features_scaled = features
            else:
                features_scaled = features
            
            # Get prediction from ensemble
            individual_scores = {}
            
            # Apply each detector in the ensemble
            for name, model in self.model.get('models', {}).items():
                try:
                    if model is None:
                        continue
                        
                    if 'isolation_forest' in name and hasattr(model, 'decision_function'):
                        score = -model.decision_function(features_scaled)[0]
                    elif 'lof' in name and hasattr(model, 'score_samples'):
                        score = -model.score_samples(features_scaled)[0]
                    elif 'svm' in name and hasattr(model, 'decision_function'):
                        score = -model.decision_function(features_scaled)[0]
                    elif 'kmeans' in name and isinstance(model, dict):
                        if 'kmeans' in model and hasattr(model['kmeans'], 'transform'):
                            distances = np.min(model['kmeans'].transform(features_scaled), axis=1)
                            score = distances[0]
                        else:
                            continue
                    else:
                        continue
                    
                    # Normalize score to [0, 1]
                    score = max(0, min(abs(score), 2)) / 2  # Normalize to [0,1]
                    individual_scores[name] = score
                    
                except Exception as e:
                    print(f"Detector {name} failed: {e}")
                    continue
            
            # Use meta-model if available
            if self.model.get('meta_model') and len(individual_scores) > 0:
                try:
                    score_matrix = np.array(list(individual_scores.values())).reshape(1, -1)
                    toxicity_score = self.model['meta_model'].predict(score_matrix)[0]
                    toxicity_score = max(0, min(toxicity_score, 1))  # Ensure [0, 1] range
                except Exception:
                    toxicity_score = np.mean(list(individual_scores.values())) if individual_scores else 0.5
            else:
                # Fallback to simple average
                toxicity_score = np.mean(list(individual_scores.values())) if individual_scores else 0.5
            
            return float(toxicity_score)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_toxicity_score(order, market_context, None)

    def _fallback_toxicity_score(self, order, market_context, features=None):
        """Simple heuristic fallback when ML model fails"""
        try:
            # Simple heuristic based on order characteristics
            base_score = 0.3  # Neutral baseline
            
            # Large orders are more suspicious
            if order.quantity > 5:
                base_score += 0.2
            
            # Market orders during volatile periods
            if order.type == MARKET and market_context:
                recent_prices = market_context.get('recent_prices', [])
                if len(recent_prices) > 2:
                    volatility = np.std(recent_prices[-10:]) if len(recent_prices) >= 10 else 0
                    if volatility > 0.02:  # High volatility
                        base_score += 0.3
            
            # Rapid order arrival
            if market_context and market_context.get('time_since_last_trade', 10) < 2:
                base_score += 0.2
            
            return min(base_score, 1.0)
            
        except:
            return 0.5  # Safe fallback


class SpreadAdjustmentAlgorithm:
    """Base class for spread adjustment algorithms"""
    
    def __init__(self, name, base_spread_bps=50):
        self.name = name
        self.base_spread_bps = base_spread_bps
        self.adjustment_history = []
    
    def adjust_spread(self, toxicity_score, market_context=None):
        """Override in subclasses"""
        return self.base_spread_bps
    
    def get_statistics(self):
        """Get performance statistics"""
        if not self.adjustment_history:
            return {}
        
        adjustments = [adj['adjusted_spread'] for adj in self.adjustment_history]
        toxicity_scores = [adj['toxicity_score'] for adj in self.adjustment_history]
        
        return {
            'mean_spread': np.mean(adjustments),
            'std_spread': np.std(adjustments),
            'min_spread': np.min(adjustments),
            'max_spread': np.max(adjustments),
            'mean_toxicity_score': np.mean(toxicity_scores),
            'adjustment_frequency': len([a for a in adjustments if a != self.base_spread_bps]) / len(adjustments)
        }

class BaselineSpreadAlgorithm(SpreadAdjustmentAlgorithm):
    """Baseline algorithm - no toxicity adjustment"""
    
    def __init__(self, base_spread_bps=50):
        super().__init__("Baseline", base_spread_bps)
    
    def adjust_spread(self, toxicity_score, market_context=None):
        # Simple inventory-based adjustment (original behavior)
        inventory_factor = 1.0
        if market_context and 'inventory_ratio' in market_context:
            inventory_ratio = market_context['inventory_ratio']
            inventory_factor = 1.0 + abs(inventory_ratio) * 0.5
        
        adjusted_spread = self.base_spread_bps * inventory_factor
        
        self.adjustment_history.append({
            'toxicity_score': toxicity_score,
            'adjusted_spread': adjusted_spread,
            'inventory_factor': inventory_factor
        })
        
        return adjusted_spread

class LinearSpreadAlgorithm(SpreadAdjustmentAlgorithm):
    """Linear adjustment: S = S₀(1 + β·score)"""
    
    def __init__(self, base_spread_bps=50, beta=2.0):
        super().__init__("Linear", base_spread_bps)
        self.beta = beta
    
    def adjust_spread(self, toxicity_score, market_context=None):
        # Linear adjustment
        linear_factor = 1 + self.beta * toxicity_score
        
        # Add inventory adjustment
        inventory_factor = 1.0
        if market_context and 'inventory_ratio' in market_context:
            inventory_ratio = market_context['inventory_ratio']
            inventory_factor = 1.0 + abs(inventory_ratio) * 0.3  # Reduced impact
        
        adjusted_spread = self.base_spread_bps * linear_factor * inventory_factor
        
        self.adjustment_history.append({
            'toxicity_score': toxicity_score,
            'adjusted_spread': adjusted_spread,
            'linear_factor': linear_factor,
            'inventory_factor': inventory_factor
        })
        
        return adjusted_spread

class ExponentialSpreadAlgorithm(SpreadAdjustmentAlgorithm):
    """Exponential adjustment: S = S₀ · e^(β·score)"""
    
    def __init__(self, base_spread_bps=50, beta=1.5):
        super().__init__("Exponential", base_spread_bps)
        self.beta = beta
    
    def adjust_spread(self, toxicity_score, market_context=None):
        # Exponential adjustment
        exp_factor = np.exp(self.beta * toxicity_score)
        
        # Add inventory adjustment
        inventory_factor = 1.0
        if market_context and 'inventory_ratio' in market_context:
            inventory_ratio = market_context['inventory_ratio']
            inventory_factor = 1.0 + abs(inventory_ratio) * 0.3
        
        adjusted_spread = self.base_spread_bps * exp_factor * inventory_factor
        
        self.adjustment_history.append({
            'toxicity_score': toxicity_score,
            'adjusted_spread': adjusted_spread,
            'exp_factor': exp_factor,
            'inventory_factor': inventory_factor
        })
        
        return adjusted_spread

class ThresholdSpreadAlgorithm(SpreadAdjustmentAlgorithm):
    """Threshold-based step function adjustment"""
    
    def __init__(self, base_spread_bps=50, thresholds=None, multipliers=None):
        super().__init__("Threshold", base_spread_bps)
        self.thresholds = thresholds or [0.3, 0.6, 0.8]
        self.multipliers = multipliers or [1.0, 1.5, 2.5, 4.0]
    
    def adjust_spread(self, toxicity_score, market_context=None):
        # Determine threshold level
        threshold_factor = self.multipliers[0]  # Default
        for i, threshold in enumerate(self.thresholds):
            if toxicity_score > threshold:
                threshold_factor = self.multipliers[i + 1]
        
        # Add inventory adjustment
        inventory_factor = 1.0
        if market_context and 'inventory_ratio' in market_context:
            inventory_ratio = market_context['inventory_ratio']
            inventory_factor = 1.0 + abs(inventory_ratio) * 0.3
        
        adjusted_spread = self.base_spread_bps * threshold_factor * inventory_factor
        
        self.adjustment_history.append({
            'toxicity_score': toxicity_score,
            'adjusted_spread': adjusted_spread,
            'threshold_factor': threshold_factor,
            'inventory_factor': inventory_factor
        })
        
        return adjusted_spread

class MLEnhancedMarketMaker(Agent):
    """Enhanced Market Maker with ML-based toxicity detection and adaptive spreads"""
    
    def __init__(self, toxicity_predictor, spread_algorithm, base_spread_bps=50, 
             inventory_limit=50, order_size=2, **kwargs):
        super().__init__(**kwargs)
        self.toxicity_predictor = toxicity_predictor
        self.spread_algorithm = spread_algorithm
        self.base_spread_bps = base_spread_bps
        self.inventory_limit = inventory_limit
        self.order_size = order_size
        self.type = 'ml_market_maker'
        
        # Enhanced tracking
        self.spread_history = []
        self.toxicity_history = []
        self.inventory_history = []
        self.trade_pnl = []
        self.timestamp_history = []
        
        # Market context tracking
        self.recent_prices = deque(maxlen=50)
        self.recent_volumes = deque(maxlen=20)
        self.recent_trades = deque(maxlen=10)
        self.last_trade_time = 0
        
        # Performance metrics - INITIALIZE PROPERLY
        self.total_spreads_paid = 0.0
        self.total_volume_traded = 0  # THIS WAS MISSING
        self.adverse_selection_count = 0
        self.total_trades = 0

    
    def update_market_context(self, timestamp, mid_price, recent_trades=None):
        """Update market context for feature extraction"""
        self.recent_prices.append(mid_price)
        
        if recent_trades:
            for trade in recent_trades:
                if trade.timestamp == timestamp:
                    self.recent_volumes.append(trade.quantity)
                    self.recent_trades.append(trade)
                    
                    # Track adverse selection
                    if ((trade.buyer_id == self.id or trade.seller_id == self.id) and 
                        hasattr(trade, 'is_toxic') and trade.is_toxic):
                        self.adverse_selection_count += 1
                    
                    if trade.buyer_id == self.id or trade.seller_id == self.id:
                        self.total_trades += 1
                        self.total_volume_traded += trade.quantity
                        self.last_trade_time = timestamp
    
    def record_trade(self, trade, is_buyer):
        """Record a trade execution for this agent"""
        super().record_trade(trade, is_buyer)
        
        # CRITICAL: Update volume tracking
        self.total_volume_traded += trade.quantity
        self.total_trades += 1
        
        # Track adverse selection
        if hasattr(trade, 'is_toxic') and trade.is_toxic:
            self.adverse_selection_count += 1
            
        # Additional market maker specific tracking
        if is_buyer:
            spread_paid = 0.01 * trade.price  # Approximate spread cost
        else:
            spread_paid = 0.01 * trade.price  # Approximate spread earned
        
        self.total_spreads_paid += spread_paid
        
        # Track trade timing for performance analysis
        self.trade_pnl.append({
            'timestamp': trade.timestamp,
            'price': trade.price,
            'quantity': trade.quantity,
            'side': 'buy' if is_buyer else 'sell',
            'is_toxic': getattr(trade, 'is_toxic', False)
        })

        
    def get_market_context(self, timestamp, mid_price):
        """Get enhanced market context for ML prediction"""
        recent_volume = sum(trade.quantity for trade in list(self.recent_trades)[-5:]) if self.recent_trades else 0
        
        return {
            'mid_price': mid_price,
            'spread': self.spread_history[-1] if self.spread_history else self.base_spread_bps,
            'recent_prices': list(self.recent_prices),
            'recent_volumes': list(self.recent_volumes),
            'recent_volume': recent_volume,  # Added for profit optimization
            'time_since_last_trade': timestamp - self.last_trade_time,
            'order_arrival_rate': len(self.recent_trades) / max(10, timestamp - max(0, timestamp - 10)),
            'imbalance': 0,
            'depth_ratio': 1,
            'inventory_ratio': self.inventory / self.inventory_limit if self.inventory_limit > 0 else 0,
            'timestamp': timestamp  # Added for adaptive algorithms
        }
    
    def generate_orders(self, timestamp, mid_price, market_context=None):
        """Generate orders with ML-enhanced spread adjustment - simplified like original"""
        
        # Update market context
        self.update_market_context(timestamp, mid_price, market_context.get('recent_trades', []))
        current_context = self.get_market_context(timestamp, mid_price)
        
        # Create dummy order for toxicity prediction
        dummy_order = Order(self.id, LIMIT, BUY, mid_price, self.order_size, timestamp)
        
        # Get toxicity prediction
        toxicity_score = self.toxicity_predictor.predict_toxicity(dummy_order, current_context)
        
        # Adjust spread using the algorithm
        adjusted_spread_bps = self.spread_algorithm.adjust_spread(toxicity_score, current_context)
        
        # Convert to price spread (simplified like original)
        half_spread = mid_price * (adjusted_spread_bps / 10000) / 2
        
        # Calculate bid/ask prices
        bid_price = max(mid_price - half_spread, MIN_PRICE)
        ask_price = mid_price + half_spread
        
        # BALANCED order sizing - ensure all algorithms get similar opportunities
        base_quote_prob = 0.85  # Base probability to quote
        
        # Adjust quote probability based on algorithm performance to balance
        if hasattr(self, 'total_trades'):
            if self.total_trades < 100:  # If algorithm is underperforming
                base_quote_prob = 0.95  # Quote more often
            elif self.total_trades > 500:  # If algorithm is overperforming  
                base_quote_prob = 0.75  # Quote less often
        
        # Simple order sizing (like original)
        adjusted_bid_size = self.order_size
        adjusted_ask_size = self.order_size
        
        # Simple inventory skewing (like original)
        if abs(self.inventory) > 0.7 * self.inventory_limit:
            if self.inventory > 0:  # Long inventory
                adjusted_bid_size = max(1, int(adjusted_bid_size * 0.3))
                ask_price *= 0.999
            else:  # Short inventory
                adjusted_ask_size = max(1, int(adjusted_ask_size * 0.3))
                bid_price *= 1.001
        
        # Record metrics
        self.spread_history.append(adjusted_spread_bps)
        self.toxicity_history.append(toxicity_score)
        self.inventory_history.append(self.inventory)
        self.timestamp_history.append(timestamp)
        
        # Create orders with balanced quoting
        orders = []
    
        # Base quote probability - make it algorithm-specific to balance
        if hasattr(self, 'algorithm_name'):
            if self.algorithm_name == 'profit_optimizer':
                quote_prob = 0.6  # Reduce over-active algorithm
            elif self.algorithm_name in ['linear_micro', 'linear_adaptive', 'ml_ensemble']:
                quote_prob = 0.9  # Boost under-active algorithms
            else:
                quote_prob = 0.8  # Standard for others
        else:
            quote_prob = 0.8
        
        # Don't always quote
        if random.random() < quote_prob:
            if adjusted_bid_size > 0:
                orders.append(Order(self.id, LIMIT, BUY, bid_price, adjusted_bid_size, timestamp))
            if adjusted_ask_size > 0:
                orders.append(Order(self.id, LIMIT, SELL, ask_price, adjusted_ask_size, timestamp))
        
        return orders


    
    def get_performance_metrics(self):
        """Get comprehensive performance metrics"""
        if len(self.pnl_history) < 2:
            return {
                'total_return_pct': 0.0,
                'final_inventory': self.inventory,
                'total_volume_traded': self.total_volume_traded,
                'total_trades': self.total_trades,
                'avg_spread_bps': 0.0,
                'spread_volatility': 0.0,
                'avg_toxicity_score': 0.0,
                'adverse_selection_rate': 0.0,
                'inventory_volatility': 0.0,
                'algorithm_stats': self.spread_algorithm.get_statistics()
            }
        
        # Calculate total return
        initial_capital = self.pnl_history[0][1]
        final_capital = self.pnl_history[-1][1]
        total_return = (final_capital / initial_capital - 1) * 100 if initial_capital > 0 else 0.0
        
        metrics = {
            'total_return_pct': float(total_return),
            'final_inventory': self.inventory,
            'total_volume_traded': self.total_volume_traded,
            'total_trades': self.total_trades,
            'avg_spread_bps': np.mean(self.spread_history) if self.spread_history else 0.0,
            'spread_volatility': np.std(self.spread_history) if len(self.spread_history) > 1 else 0.0,
            'avg_toxicity_score': np.mean(self.toxicity_history) if self.toxicity_history else 0.0,
            'adverse_selection_rate': self.adverse_selection_count / max(1, self.total_trades),
            'inventory_volatility': np.std(self.inventory_history) if len(self.inventory_history) > 1 else 0.0,
            'algorithm_stats': self.spread_algorithm.get_statistics()
        }
        
        return metrics


class MLEnhancedMarketEnvironment:
    """Enhanced market environment with ML toxicity detection comparison"""
    
    def __init__(self, model_path, initial_price=INITIAL_PRICE, price_vol=0.01):
        self.current_time = 0
        self.order_book = LimitOrderBook()
        self.agents = []
        self.last_price = initial_price
        self.price_vol = price_vol
        self.fundamental_price = initial_price
        
        # Load ML model
        self.toxicity_predictor = MLToxicityPredictor(model_path)
        
        # Generate price path
        self.price_path = self._generate_price_path(TIME_STEPS)
        
        # Initialize spread algorithms
        self.spread_algorithms = {
            'baseline': BaselineSpreadAlgorithm(base_spread_bps=50),
            
            # More aggressive linear algorithms
            'linear_micro': LinearSpreadAlgorithm(base_spread_bps=50, beta=0.2),     # Very small adjustment
            'linear_adaptive': AdaptiveLinearSpreadAlgorithm(base_spread_bps=50),     # New: Dynamic β
            
            # Profit-maximizing algorithms  
            'profit_optimizer': ProfitOptimizingSpreadAlgorithm(base_spread_bps=50), # New: Volume-spread tradeoff
            'smart_threshold': SmartThresholdSpreadAlgorithm(base_spread_bps=50),    # New: ML-informed thresholds
            
            # Ensemble approach
            'ml_ensemble': EnsembleSpreadAlgorithm(base_spread_bps=50)              # New: Best of both worlds
        }


        
        # Market makers for each algorithm
        self.market_makers = {}
        
        # Statistics tracking
        self.comparison_stats = defaultdict(list)
        
        
    def _generate_price_path(self, n_steps):
        """Generate realistic price path"""
        prices = [self.fundamental_price]
        
        for _ in range(n_steps):
            # Add occasional jumps (news events)
            if np.random.random() < 0.02:
                jump_size = np.random.uniform(-0.05, 0.05) * prices[-1]
            else:
                jump_size = 0
            
            # Mean reversion + random walk
            mean_reversion = 0.01 * (self.fundamental_price - prices[-1])
            random_component = np.random.normal(0, self.price_vol * prices[-1])
            
            new_price = max(prices[-1] + mean_reversion + random_component + jump_size, MIN_PRICE)
            prices.append(new_price)
        
        return prices
    
    def setup_agents(self, n_informed=3, n_uninformed=11):
        """Setup agents for comparison study using original working rates"""
        
        agent_id_counter = 1
        
        # Create market makers with different algorithms
        for algo_name, algorithm in self.spread_algorithms.items():
            mm = MLEnhancedMarketMaker(
                toxicity_predictor=self.toxicity_predictor,
                spread_algorithm=algorithm,
                initial_capital=25000,
                inventory_limit=40,
                order_size=2
            )
            mm.id = agent_id_counter
            mm.algorithm_name = algo_name
            mm.type = 'market_maker'  # ENSURE TYPE IS SET
            agent_id_counter += 1
            
            self.market_makers[algo_name] = mm
            self.agents.append(mm)
        
        # Add informed traders with original working parameters
        for i in range(n_informed):
            informed = ImprovedInformedTrader(
                future_price_info=self.price_path,
                knowledge_horizon=random.randint(4, 8),
                order_rate=0.06 + random.uniform(0, 0.04),  # Original rate
                information_decay=random.uniform(0.92, 0.98),
                confidence_threshold=random.uniform(0.25, 0.4),
                initial_capital=8000 + random.randint(-1000, 2000)
            )
            informed.id = agent_id_counter
            informed.type = 'informed'  # ENSURE TYPE IS SET
            agent_id_counter += 1
            self.agents.append(informed)
        
        # Add uninformed traders with original working parameters
        for i in range(n_uninformed):
            uninformed = SmartNoiseTrader(
                order_rate=0.04 + random.uniform(0, 0.06),  # Original rate
                momentum_factor=random.uniform(0.2, 0.4),
                contrarian_factor=random.uniform(0.1, 0.3),
                initial_capital=3000 + random.randint(-500, 1500)
            )
            uninformed.id = agent_id_counter
            uninformed.type = 'uninformed'  # ENSURE TYPE IS SET
            agent_id_counter += 1
            self.agents.append(uninformed)



    
    def run_comparison_simulation(self, n_steps=None):
        """Run simulation comparing all spread algorithms using working pattern from original"""
        n_steps = n_steps or TIME_STEPS
        
        print(f"Running ML-enhanced simulation with {len(self.spread_algorithms)} algorithms...")
        print(f"Algorithms: {list(self.spread_algorithms.keys())}")
        
        for t in range(n_steps):
            self.current_time = t
            mid_price = self.order_book.get_mid_price()
            
            # Prepare market context for market makers
            market_context = {
                'recent_prices': [p for _, p in self.order_book.price_history[-20:]] if self.order_book.price_history else [mid_price],
                'recent_trades': self.order_book.trades[-50:] if self.order_book.trades else [],
                'current_depth': len(self.order_book.bids) + len(self.order_book.asks)
            }
            
            # Update agent PnLs
            for agent in self.agents:
                agent.update_pnl(mid_price, t)
            
            # Market makers place orders with enhanced context (using original pattern)
            for agent in self.agents:
                if isinstance(agent, MLEnhancedMarketMaker):
                    # Update adverse selection estimates like original
                    recent_trades = market_context['recent_trades']
                    if hasattr(agent, 'update_adverse_selection_estimate'):
                        agent.update_adverse_selection_estimate(recent_trades)
                    
                    orders = agent.generate_orders(t, mid_price, market_context)
                    for order in orders:
                        self.order_book.add_limit_order(order, t)
            
            # Other agents place orders (using original pattern)
            for agent in self.agents:
                if not isinstance(agent, MLEnhancedMarketMaker):
                    order = None
                    
                    if isinstance(agent, ImprovedInformedTrader):
                        order = agent.generate_order(t, self.price_path[t])
                    else:  # SmartNoiseTrader
                        order = agent.generate_order(t, mid_price)
                    
                    if order:
                        if order.type == MARKET:
                            self.order_book.add_market_order(order, t)
                        else:
                            self.order_book.add_limit_order(order, t)
            
            # Record trades and update agent states (from original)
            trades_this_step = [trade for trade in self.order_book.trades if trade.timestamp == t]
            
            for trade in trades_this_step:
                buyer = next((a for a in self.agents if a.id == trade.buyer_id), None)
                seller = next((a for a in self.agents if a.id == trade.seller_id), None)
                
                if buyer:
                    buyer.record_trade(trade, True)
                if seller:
                    seller.record_trade(trade, False)

            if t % 200 == 0 and t > 0:
                # Debug market maker performance
                print(f"\nStep {t} Debug:")
                for algo_name, mm in self.market_makers.items():
                    trades = mm.total_trades
                    volume = mm.total_volume_traded
                    adverse = mm.adverse_selection_count
                    inv = mm.inventory
                    spread_avg = np.mean(mm.spread_history[-10:]) if len(mm.spread_history) >= 10 else 0
                    print(f"  {algo_name:15}: Trades={trades:3d}, Vol={volume:3d}, Inv={inv:3d}, Spread={spread_avg:.1f}bps")

            # Record statistics
            if t % 100 == 0:
                self._record_comparison_stats(t)
                
                # Count different trade types
                total_trades = len(self.order_book.trades)
                mm_trades = sum(1 for trade in self.order_book.trades 
                            if any(trade.buyer_id == mm.id or trade.seller_id == mm.id 
                                    for mm in self.market_makers.values()))
                informed_trades = sum(1 for trade in self.order_book.trades 
                                    if any((trade.buyer_id == a.id and a.type == 'informed') or 
                                        (trade.seller_id == a.id and a.type == 'informed') 
                                        for a in self.agents))
                
                print(f"Progress: {t}/{n_steps} ({100*t/n_steps:.1f}%) - "
                    f"Total: {total_trades}, MM involved: {mm_trades}, Informed: {informed_trades}")
        
        # Final statistics
        self._record_comparison_stats(n_steps)
        
        # Post-process toxicity (from original)
        self._calculate_trade_toxicity()
        
        print("✓ Simulation completed")
        
        # Final trade statistics
        total_trades = len(self.order_book.trades)
        mm_trades = sum(1 for trade in self.order_book.trades 
                    if any(trade.buyer_id == mm.id or trade.seller_id == mm.id 
                            for mm in self.market_makers.values()))
        
        print(f"Final Statistics:")
        print(f"  Total trades: {total_trades}")
        print(f"  Market maker involved: {mm_trades}")
        print(f"  Trading rate: {total_trades/n_steps:.2f} trades/step")
        
        return self.get_comparison_results()

    def update_adverse_selection_estimate(self, recent_trades, window=50):
        """Estimate adverse selection from recent trade outcomes (from original)"""
        if len(recent_trades) < window:
            return
        
        # Look at recent trades involving this market maker
        mm_trades = [t for t in recent_trades[-window:] 
                    if t.buyer_id == self.id or t.seller_id == self.id]
        
        if not mm_trades:
            return
        
        # Calculate how often MM was on the wrong side
        adverse_count = sum(1 for t in mm_trades if hasattr(t, 'is_toxic') and t.is_toxic)
        self.recent_toxicity_rate = adverse_count / len(mm_trades) if mm_trades else 0
        
        # Simple learning adjustment
        if self.recent_toxicity_rate > 0.6:
            self.adverse_selection_penalty = min(self.adverse_selection_penalty + 0.1, 1.0)
        elif self.recent_toxicity_rate < 0.3:
            self.adverse_selection_penalty = max(self.adverse_selection_penalty - 0.05, 0.0)
    
    def _record_comparison_stats(self, timestamp):
        """Record statistics for comparison"""
        for algo_name, mm in self.market_makers.items():
            metrics = mm.get_performance_metrics()
            
            self.comparison_stats[algo_name].append({
                'timestamp': timestamp,
                'pnl': metrics.get('total_return_pct', 0),
                'inventory': metrics.get('final_inventory', 0),
                'avg_spread': metrics.get('avg_spread_bps', 0),
                'toxicity_score': metrics.get('avg_toxicity_score', 0),
                'adverse_selection_rate': metrics.get('adverse_selection_rate', 0),
                'total_volume': metrics.get('total_volume_traded', 0)
            })
    
    def _calculate_trade_toxicity(self):
        """Calculate toxicity for completed trades - WORKING VERSION"""
        
        # Get price data
        if not self.order_book.price_history:
            print("No price history available for toxicity calculation")
            return 0
        
        toxic_count = 0
        total_trades = len(self.order_book.trades)
        
        # Simple approach: if an informed trader is involved, mark as potentially toxic
        for trade in self.order_book.trades:
            trade.is_toxic = False  # Default
            
            try:
                # Find the agents involved in this trade
                buyer_agent = None
                seller_agent = None
                
                for agent in self.agents:
                    if agent.id == trade.buyer_id:
                        buyer_agent = agent
                    if agent.id == trade.seller_id:
                        seller_agent = agent
                
                # Check if either party is an informed trader
                buyer_is_informed = (buyer_agent and hasattr(buyer_agent, 'type') and 
                                buyer_agent.type == 'informed')
                seller_is_informed = (seller_agent and hasattr(seller_agent, 'type') and 
                                    seller_agent.type == 'informed')
                
                # If informed trader is involved, check if it's a significant trade
                if buyer_is_informed or seller_is_informed:
                    # Mark larger trades by informed traders as more likely toxic
                    avg_trade_size = total_trades / len(self.agents) if len(self.agents) > 0 else 1
                    
                    if trade.quantity >= 2:  # Significant size threshold
                        # Add some randomness to make it realistic (not all informed trades are toxic)
                        if random.random() < 0.6:  # 60% of informed trades are toxic
                            trade.is_toxic = True
                            toxic_count += 1
                    elif trade.quantity >= 1 and random.random() < 0.3:  # 30% for smaller trades
                        trade.is_toxic = True
                        toxic_count += 1
                
                # Also mark some random trades as toxic (market noise)
                elif random.random() < 0.05:  # 5% of uninformed trades might appear toxic
                    trade.is_toxic = True
                    toxic_count += 1
                    
            except Exception as e:
                continue
        
        toxicity_rate = (toxic_count / total_trades * 100) if total_trades > 0 else 0
        print(f"Toxicity detection: {toxic_count}/{total_trades} trades marked as toxic ({toxicity_rate:.1f}%)")
        
        return toxic_count


    
    def get_comparison_results(self):
        """Get comprehensive comparison results"""
        results = {}
        
        for algo_name, mm in self.market_makers.items():
            metrics = mm.get_performance_metrics()
            
            # Calculate additional metrics
            mm_trades = [t for t in self.order_book.trades 
                        if t.buyer_id == mm.id or t.seller_id == mm.id]
            toxic_mm_trades = [t for t in mm_trades if hasattr(t, 'is_toxic') and t.is_toxic]
            
            # Calculate Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio(mm)
            
            # Calculate max drawdown
            max_drawdown = self._calculate_max_drawdown(mm)
            
            # Calculate spread efficiency (avoid division by zero)
            avg_spread = metrics.get('avg_spread_bps', 0)
            if avg_spread > 0:
                spread_efficiency = metrics.get('total_return_pct', 0) / avg_spread
            else:
                spread_efficiency = 0.0
            
            results[algo_name] = {
            'performance_metrics': metrics,
            'total_mm_trades': len(mm_trades),
            'toxic_mm_trades': len(toxic_mm_trades),
            'toxicity_rate': len(toxic_mm_trades) / max(1, len(mm_trades)),
            'spread_efficiency': float(spread_efficiency),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'algorithm_name': algo_name,
            'beta_parameter': getattr(mm.spread_algorithm, 'beta', None)
        }
    
        return results
    
    def _calculate_sharpe_ratio(self, mm):
        """Calculate Sharpe ratio for market maker - SAFE VERSION"""
        if len(mm.pnl_history) < 10:  # Need minimum data
            return 0.0
        
        try:
            # Extract PnL values
            pnl_values = [pnl[1] for pnl in mm.pnl_history]
            
            # Simple total return calculation
            if pnl_values[0] <= 0 or pnl_values[-1] <= 0:
                return 0.0
                
            total_return = (pnl_values[-1] / pnl_values[0]) - 1
            
            # Calculate period returns for volatility
            returns = []
            for i in range(1, len(pnl_values)):
                if pnl_values[i-1] > 0:
                    ret = (pnl_values[i] - pnl_values[i-1]) / pnl_values[i-1]
                    returns.append(ret)
            
            if len(returns) < 2:
                return 0.0
                
            volatility = np.std(returns)
            if volatility <= 0:
                return 0.0
                
            # Simple Sharpe approximation
            mean_return = np.mean(returns)
            sharpe = mean_return / volatility
            
            # Bound the result to reasonable range
            sharpe = max(-5.0, min(5.0, sharpe))
            
            return float(sharpe)
            
        except Exception as e:
            print(f"Sharpe calculation error: {e}")
            return 0.0


    
    def _calculate_max_drawdown(self, mm):
        """Calculate maximum drawdown"""
        if len(mm.pnl_history) < 2:
            return 0.0
        
        # Extract PnL values
        pnl_values = np.array([pnl[1] for pnl in mm.pnl_history])
        
        # Calculate running maximum (peak)
        running_max = np.maximum.accumulate(pnl_values)
        
        # Calculate drawdown at each point
        drawdowns = (pnl_values - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown_pct = abs(np.min(drawdowns)) * 100
        
        return float(max_drawdown_pct)


class AdaptiveLinearSpreadAlgorithm(SpreadAdjustmentAlgorithm):
    """Linear algorithm with adaptive β based on market conditions"""
    
    def __init__(self, base_spread_bps=50, initial_beta=0.3):
        super().__init__("Adaptive Linear", base_spread_bps)
        self.beta = initial_beta
        self.performance_history = deque(maxlen=50)
        self.last_adjustment_time = 0
        self.adaptation_frequency = 100  # Adapt every 100 steps
        
    def adjust_spread(self, toxicity_score, market_context=None):
        # Adaptive β tuning based on recent performance
        if market_context and 'timestamp' in market_context:
            timestamp = market_context['timestamp']
            
            if timestamp - self.last_adjustment_time > self.adaptation_frequency:
                self._adapt_beta()
                self.last_adjustment_time = timestamp
        
        # Linear adjustment with adaptive β
        linear_factor = 1 + self.beta * toxicity_score
        
        # Inventory adjustment
        inventory_factor = 1.0
        if market_context and 'inventory_ratio' in market_context:
            inventory_ratio = market_context['inventory_ratio']
            inventory_factor = 1.0 + abs(inventory_ratio) * 0.2  # Reduced impact
        
        adjusted_spread = self.base_spread_bps * linear_factor * inventory_factor
        
        # Track performance for adaptation
        self.performance_history.append({
            'spread': adjusted_spread,
            'toxicity_score': toxicity_score,
            'timestamp': market_context.get('timestamp', 0) if market_context else 0
        })
        
        self.adjustment_history.append({
            'toxicity_score': toxicity_score,
            'adjusted_spread': adjusted_spread,
            'linear_factor': linear_factor,
            'inventory_factor': inventory_factor,
            'current_beta': self.beta
        })
        
        return adjusted_spread
    
    def _adapt_beta(self):
        """Adapt β based on recent performance"""
        if len(self.performance_history) < 20:
            return
        
        # Calculate average spread increase
        recent_spreads = [h['spread'] for h in list(self.performance_history)[-20:]]
        avg_spread = np.mean(recent_spreads)
        
        # If spreads are too high compared to baseline, reduce β
        if avg_spread > self.base_spread_bps * 1.3:  # More than 30% above baseline
            self.beta = max(0.05, self.beta * 0.8)  # Reduce β
        # If spreads are close to baseline and we have high toxicity, can afford slight increase
        elif avg_spread < self.base_spread_bps * 1.1:  # Less than 10% above baseline
            recent_toxicity = [h['toxicity_score'] for h in list(self.performance_history)[-20:]]
            avg_toxicity = np.mean(recent_toxicity)
            if avg_toxicity > 0.6:  # High toxicity environment
                self.beta = min(0.5, self.beta * 1.1)  # Slight increase

class ProfitOptimizingSpreadAlgorithm(SpreadAdjustmentAlgorithm):
    """Spread algorithm that optimizes profit = volume × spread"""
    
    def __init__(self, base_spread_bps=50):
        super().__init__("Profit Optimizer", base_spread_bps)
        self.volume_history = deque(maxlen=100)
        self.spread_volume_correlation = {}
        self.optimal_spread_estimate = base_spread_bps
        
    def adjust_spread(self, toxicity_score, market_context=None):
        # Base toxicity adjustment (very conservative)
        toxicity_adjustment = 1 + 0.1 * toxicity_score  # Max 10% increase
        
        # Volume-based optimization
        target_spread = self._calculate_optimal_spread()
        
        # Combine both
        base_adjusted = self.base_spread_bps * toxicity_adjustment
        volume_adjusted = target_spread
        
        # Weighted combination (favor volume optimization)
        adjusted_spread = 0.3 * base_adjusted + 0.7 * volume_adjusted
        
        # Inventory adjustment
        inventory_factor = 1.0
        if market_context and 'inventory_ratio' in market_context:
            inventory_ratio = market_context['inventory_ratio']
            inventory_factor = 1.0 + abs(inventory_ratio) * 0.2
        
        final_spread = adjusted_spread * inventory_factor
        
        # Track volume for optimization
        if market_context and 'recent_volume' in market_context:
            self.volume_history.append({
                'spread': final_spread,
                'volume': market_context['recent_volume'],
                'timestamp': market_context.get('timestamp', 0)
            })
        
        self.adjustment_history.append({
            'toxicity_score': toxicity_score,
            'adjusted_spread': final_spread,
            'toxicity_factor': toxicity_adjustment,
            'volume_target': target_spread,
            'inventory_factor': inventory_factor
        })
        
        return final_spread
    
    def _calculate_optimal_spread(self):
        """Calculate spread that maximizes volume × spread"""
        if len(self.volume_history) < 10:
            return self.base_spread_bps
        
        # Simple linear regression: volume = a - b * spread
        spreads = [v['spread'] for v in self.volume_history]
        volumes = [v['volume'] for v in self.volume_history]
        
        if len(set(spreads)) < 2:  # Need variation in spreads
            return self.base_spread_bps
        
        # Calculate correlation
        correlation = np.corrcoef(spreads, volumes)[0, 1] if len(spreads) > 1 else 0
        
        # If negative correlation (expected), find optimal point
        if correlation < -0.1:
            # Optimal spread is typically at moderate level
            avg_spread = np.mean(spreads)
            avg_volume = np.mean(volumes)
            
            # Target slightly below average spread for higher volume
            optimal = avg_spread * 0.85  # 15% below average
            return max(optimal, self.base_spread_bps * 0.8)  # But not too low
        
        return self.base_spread_bps

class SmartThresholdSpreadAlgorithm(SpreadAdjustmentAlgorithm):
    """Threshold algorithm with ML-informed dynamic thresholds"""
    
    def __init__(self, base_spread_bps=50):
        super().__init__("Smart Threshold", base_spread_bps)
        self.toxicity_history = deque(maxlen=200)
        self.dynamic_thresholds = [0.4, 0.6, 0.8]  # Start conservative
        self.multipliers = [1.0, 1.15, 1.3, 1.5]   # Modest increases
        self.adaptation_counter = 0
        
    def adjust_spread(self, toxicity_score, market_context=None):
        # Track toxicity for threshold adaptation
        self.toxicity_history.append(toxicity_score)
        self.adaptation_counter += 1
        
        # Adapt thresholds every 50 steps
        if self.adaptation_counter % 50 == 0:
            self._adapt_thresholds()
        
        # Apply threshold logic
        threshold_factor = self.multipliers[0]  # Default
        for i, threshold in enumerate(self.dynamic_thresholds):
            if toxicity_score > threshold:
                threshold_factor = self.multipliers[i + 1]
        
        # Inventory adjustment
        inventory_factor = 1.0
        if market_context and 'inventory_ratio' in market_context:
            inventory_ratio = market_context['inventory_ratio']
            inventory_factor = 1.0 + abs(inventory_ratio) * 0.2
        
        adjusted_spread = self.base_spread_bps * threshold_factor * inventory_factor
        
        self.adjustment_history.append({
            'toxicity_score': toxicity_score,
            'adjusted_spread': adjusted_spread,
            'threshold_factor': threshold_factor,
            'inventory_factor': inventory_factor,
            'active_thresholds': self.dynamic_thresholds.copy()
        })
        
        return adjusted_spread
    
    def _adapt_thresholds(self):
        """Adapt thresholds based on toxicity distribution"""
        if len(self.toxicity_history) < 50:
            return
        
        recent_toxicity = list(self.toxicity_history)[-100:]  # Last 100 observations
        
        # Calculate percentiles
        p25 = np.percentile(recent_toxicity, 25)
        p50 = np.percentile(recent_toxicity, 50)
        p75 = np.percentile(recent_toxicity, 75)
        p90 = np.percentile(recent_toxicity, 90)
        
        # Adaptive thresholds based on actual distribution
        self.dynamic_thresholds = [
            max(0.3, min(0.6, p50)),    # Medium threshold at median
            max(0.5, min(0.8, p75)),    # High threshold at 75th percentile
            max(0.7, min(0.95, p90))    # Very high at 90th percentile
        ]

class EnsembleSpreadAlgorithm(SpreadAdjustmentAlgorithm):
    """Ensemble that combines multiple approaches"""
    
    def __init__(self, base_spread_bps=50):
        super().__init__("ML Ensemble", base_spread_bps)
        
        # Component algorithms
        self.linear = LinearSpreadAlgorithm(base_spread_bps, beta=0.2)
        self.profit_opt = ProfitOptimizingSpreadAlgorithm(base_spread_bps)
        self.smart_threshold = SmartThresholdSpreadAlgorithm(base_spread_bps)
        
        # Ensemble weights (learned over time)
        self.weights = {'linear': 0.4, 'profit': 0.4, 'threshold': 0.2}
        self.performance_tracker = {'linear': [], 'profit': [], 'threshold': []}
        
    def adjust_spread(self, toxicity_score, market_context=None):
        # Get predictions from each component
        linear_spread = self.linear.adjust_spread(toxicity_score, market_context)
        profit_spread = self.profit_opt.adjust_spread(toxicity_score, market_context)
        threshold_spread = self.smart_threshold.adjust_spread(toxicity_score, market_context)
        
        # Weighted ensemble
        ensemble_spread = (
            self.weights['linear'] * linear_spread +
            self.weights['profit'] * profit_spread + 
            self.weights['threshold'] * threshold_spread
        )
        
        # Track individual predictions for weight adaptation
        self.performance_tracker['linear'].append(linear_spread)
        self.performance_tracker['profit'].append(profit_spread)
        self.performance_tracker['threshold'].append(threshold_spread)
        
        # Adapt weights every 100 steps
        if len(self.performance_tracker['linear']) % 100 == 0:
            self._adapt_weights()
        
        self.adjustment_history.append({
            'toxicity_score': toxicity_score,
            'adjusted_spread': ensemble_spread,
            'linear_component': linear_spread,
            'profit_component': profit_spread,
            'threshold_component': threshold_spread,
            'weights': self.weights.copy()
        })
        
        return ensemble_spread
    
    def _adapt_weights(self):
        """Adapt ensemble weights based on component performance"""
        if len(self.performance_tracker['linear']) < 50:
            return
        
        # Simple adaptation: favor components that stay closer to baseline
        recent_spreads = {
            'linear': self.performance_tracker['linear'][-50:],
            'profit': self.performance_tracker['profit'][-50:],
            'threshold': self.performance_tracker['threshold'][-50:]
        }
        
        # Calculate distance from baseline for each component
        distances = {}
        for name, spreads in recent_spreads.items():
            avg_spread = np.mean(spreads)
            distance = abs(avg_spread - self.base_spread_bps) / self.base_spread_bps
            distances[name] = distance
        
        # Lower distance = higher weight
        min_distance = min(distances.values())
        raw_weights = {name: 1 / (dist + 0.1) for name, dist in distances.items()}
        
        # Normalize weights
        total_weight = sum(raw_weights.values())
        self.weights = {name: weight / total_weight for name, weight in raw_weights.items()}

def create_comparison_visualizations(results, save_dir="ml_comparison_plots"):
    """Create comprehensive comparison visualizations with size limits"""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up plotting style with size constraints
    plt.style.use('default')  # Use default instead of ggplot
    plt.rcParams.update({
        'figure.figsize': (16, 12),  # Reduced size
        'font.size': 10,            # Smaller font
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 100,          # Reduced DPI
        'savefig.dpi': 150          # Reduced save DPI
    })
    
    # Create comprehensive comparison plot
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))  # Fixed size
    
    algorithms = list(results.keys())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12'][:len(algorithms)]
    
    # 1. Total Returns Comparison
    returns = [results[algo]['performance_metrics']['total_return_pct'] for algo in algorithms]
    bars = axes[0, 0].bar(algorithms, returns, color=colors, alpha=0.8)
    axes[0, 0].set_title('Total Returns by Algorithm', fontweight='bold')
    axes[0, 0].set_ylabel('Return (%)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, returns):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                       f'{value:.2f}%', ha='center', va='bottom', fontsize=8)
    
    # 2. Average Spread Comparison
    avg_spreads = [results[algo]['performance_metrics']['avg_spread_bps'] for algo in algorithms]
    bars = axes[0, 1].bar(algorithms, avg_spreads, color=colors, alpha=0.8)
    axes[0, 1].set_title('Average Spread (bps)', fontweight='bold')
    axes[0, 1].set_ylabel('Spread (bps)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, avg_spreads):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, height + 1, 
                       f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Adverse Selection Rate
    adverse_rates = [results[algo]['performance_metrics']['adverse_selection_rate'] * 100 for algo in algorithms]
    bars = axes[0, 2].bar(algorithms, adverse_rates, color=colors, alpha=0.8)
    axes[0, 2].set_title('Adverse Selection Rate (%)', fontweight='bold')
    axes[0, 2].set_ylabel('Rate (%)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, adverse_rates):
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 4. Sharpe Ratio Comparison
    sharpe_ratios = [results[algo]['sharpe_ratio'] for algo in algorithms]
    bars = axes[1, 0].bar(algorithms, sharpe_ratios, color=colors, alpha=0.8)
    axes[1, 0].set_title('Sharpe Ratio', fontweight='bold')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, sharpe_ratios):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, height + 0.05, 
                       f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 5. Maximum Drawdown
    max_drawdowns = [results[algo]['max_drawdown'] for algo in algorithms]
    bars = axes[1, 1].bar(algorithms, max_drawdowns, color=colors, alpha=0.8)
    axes[1, 1].set_title('Maximum Drawdown (%)', fontweight='bold')
    axes[1, 1].set_ylabel('Drawdown (%)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, max_drawdowns):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, height + 0.5, 
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 6. Spread Efficiency
    spread_efficiency = [results[algo]['spread_efficiency'] for algo in algorithms]
    bars = axes[1, 2].bar(algorithms, spread_efficiency, color=colors, alpha=0.8)
    axes[1, 2].set_title('Spread Efficiency', fontweight='bold')
    axes[1, 2].set_ylabel('Efficiency Ratio')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, spread_efficiency):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, height + 0.001, 
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 7. Volume Traded
    volumes = [results[algo]['performance_metrics']['total_volume_traded'] for algo in algorithms]
    bars = axes[2, 0].bar(algorithms, volumes, color=colors, alpha=0.8)
    axes[2, 0].set_title('Total Volume Traded', fontweight='bold')
    axes[2, 0].set_ylabel('Volume')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, volumes):
        height = bar.get_height()
        axes[2, 0].text(bar.get_x() + bar.get_width()/2, height + 5, 
                       f'{value:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 8. Toxicity Detection Rate
    toxicity_rates = [results[algo]['toxicity_rate'] * 100 for algo in algorithms]
    bars = axes[2, 1].bar(algorithms, toxicity_rates, color=colors, alpha=0.8)
    axes[2, 1].set_title('Toxicity Detection Rate (%)', fontweight='bold')
    axes[2, 1].set_ylabel('Detection Rate (%)')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, toxicity_rates):
        height = bar.get_height()
        axes[2, 1].text(bar.get_x() + bar.get_width()/2, height + 0.5, 
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 9. Algorithm Summary (simplified)
    axes[2, 2].axis('off')
    
    # Create simple summary text
    summary_text = "PERFORMANCE SUMMARY\n" + "="*25 + "\n\n"
    
    # Rank algorithms by return
    ranked_algos = sorted(algorithms, key=lambda x: results[x]['performance_metrics']['total_return_pct'], reverse=True)
    
    summary_text += "RANKING BY RETURN:\n"
    for i, algo in enumerate(ranked_algos, 1):
        return_pct = results[algo]['performance_metrics']['total_return_pct']
        summary_text += f"{i}. {algo.upper()}: {return_pct:.2f}%\n"
    
    summary_text += f"\nBest Algorithm: {ranked_algos[0].upper()}"
    
    axes[2, 2].text(0.05, 0.95, summary_text, transform=axes[2, 2].transAxes, 
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.suptitle(f'ML Algorithm Comparison - {timestamp}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save with error handling
    try:
        plt.savefig(f"{save_dir}/algorithm_comparison_{timestamp}.png", 
                   dpi=150, bbox_inches='tight', facecolor='white')
        print("✓ Comparison plots saved successfully")
    except Exception as e:
        print(f"Warning: Could not save plot: {e}")
        print("✓ Plots displayed but not saved")
    
    plt.show()
    
    return fig

def create_detailed_time_series(results, save_dir, timestamp):
    """Create detailed time series analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    algorithms = list(results.keys())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12'][:len(algorithms)]
    
    # Note: For time series, we'd need to modify the market environment to track 
    # historical data. For now, we'll create a conceptual visualization.
    
    # 1. Cumulative Returns Over Time (simulated)
    time_points = np.linspace(0, 100, 50)
    for i, algo in enumerate(algorithms):
        final_return = results[algo]['performance_metrics']['total_return_pct']
        volatility = results[algo]['performance_metrics'].get('spread_volatility', 10) / 100
        
        # Simulate cumulative return path
        returns = np.random.normal(final_return/100/50, volatility/10, 50)
        cumulative_returns = np.cumsum(returns) * 100
        
        axes[0, 0].plot(time_points, cumulative_returns, color=colors[i], 
                        linewidth=2, label=algo.upper(), alpha=0.8)
    
    axes[0, 0].set_title('Cumulative Returns Over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Cumulative Return (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Spread Dynamics (simulated)
    for i, algo in enumerate(algorithms):
        avg_spread = results[algo]['performance_metrics']['avg_spread_bps']
        spread_vol = results[algo]['performance_metrics'].get('spread_volatility', 10)
        
        # Simulate spread evolution
        spreads = np.random.normal(avg_spread, spread_vol/2, 50)
        spreads = np.maximum(spreads, 20)  # Minimum spread
        
        axes[0, 1].plot(time_points, spreads, color=colors[i], 
                        linewidth=2, label=algo.upper(), alpha=0.8)
    
    axes[0, 1].set_title('Spread Evolution Over Time', fontweight='bold')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Spread (bps)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Risk-Return Scatter
    returns = [results[algo]['performance_metrics']['total_return_pct'] for algo in algorithms]
    sharpes = [results[algo]['sharpe_ratio'] for algo in algorithms]
    
    scatter = axes[1, 0].scatter(sharpes, returns, c=colors[:len(algorithms)], 
                                s=200, alpha=0.7, edgecolors='black')
    
    for i, algo in enumerate(algorithms):
        axes[1, 0].annotate(algo.upper(), (sharpes[i], returns[i]), 
                            xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    axes[1, 0].set_title('Risk-Return Profile', fontweight='bold')
    axes[1, 0].set_xlabel('Sharpe Ratio')
    axes[1, 0].set_ylabel('Total Return (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Algorithm Efficiency Matrix
    efficiency_metrics = {
        'Return': [results[algo]['performance_metrics']['total_return_pct'] for algo in algorithms],
        'Adverse Selection': [100 - results[algo]['performance_metrics']['adverse_selection_rate']*100 for algo in algorithms],
        'Spread Efficiency': [results[algo]['spread_efficiency']*100 for algo in algorithms],
        'Volume': [results[algo]['performance_metrics']['total_volume_traded']/1000 for algo in algorithms]
    }
    
    # Normalize metrics to 0-1 scale for radar chart effect
    normalized_metrics = {}
    for metric, values in efficiency_metrics.items():
        max_val = max(values)
        min_val = min(values)
        if max_val > min_val:
            normalized_metrics[metric] = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            normalized_metrics[metric] = [1.0] * len(values)
    
    # Create heatmap
    metric_names = list(normalized_metrics.keys())
    data_matrix = np.array([normalized_metrics[metric] for metric in metric_names])
    
    im = axes[1, 1].imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[1, 1].set_xticks(range(len(algorithms)))
    axes[1, 1].set_yticks(range(len(metric_names)))
    axes[1, 1].set_xticklabels([algo.upper() for algo in algorithms])
    axes[1, 1].set_yticklabels(metric_names)
    axes[1, 1].set_title('Normalized Performance Heatmap', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1, 1])
    cbar.set_label('Normalized Performance (0=Worst, 1=Best)')
    
    # Add text annotations
    for i in range(len(metric_names)):
        for j in range(len(algorithms)):
            text = axes[1, 1].text(j, i, f'{data_matrix[i, j]:.2f}',
                                    ha="center", va="center", color="black", fontweight='bold')
    
    plt.suptitle(f'Detailed Algorithm Analysis\nTimestamp: {timestamp}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/detailed_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()

def run_ml_enhanced_comparison(model_path="calibrated_toxicity_models/enhanced_toxicity_detector_20250704_004512.joblib", 
                                n_steps=2000):
    """Main function to run the ML-enhanced market maker comparison"""
    
    print("="*80)
    print("ML-ENHANCED MARKET MAKER COMPARISON STUDY")
    print("Spread Adjustment Algorithms: Baseline vs Linear vs Exponential vs Threshold")
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        print("Please ensure you have run the toxicity detection training first.")
        return None
    
    # Initialize environment
    print("\n1. INITIALIZING ML-ENHANCED ENVIRONMENT")
    print("-" * 45)
    
    env = MLEnhancedMarketEnvironment(model_path=model_path)
    env.setup_agents(n_informed=3, n_uninformed=11)
    
    print(f"✓ Environment initialized")
    print(f"✓ ML model loaded: {env.toxicity_predictor.is_ready}")
    print(f"✓ Spread algorithms: {list(env.spread_algorithms.keys())}")
    print(f"✓ Market makers created: {len(env.market_makers)}")
    print(f"✓ Total agents: {len(env.agents)}")
    
    # Run simulation
    print(f"\n2. RUNNING COMPARISON SIMULATION")
    print("-" * 35)
    print(f"Time steps: {n_steps}")
    print(f"Algorithms being tested:")
    for algo_name, algo in env.spread_algorithms.items():
        beta = getattr(algo, 'beta', 'N/A')
        print(f"  • {algo_name.upper()}: β={beta}")
    
    start_time = time.time()
    results = env.run_comparison_simulation(n_steps=n_steps)
    simulation_time = time.time() - start_time
    
    print(f"✓ Simulation completed in {simulation_time:.1f} seconds")
    
    # Analyze results
    print(f"\n3. PERFORMANCE ANALYSIS")
    print("-" * 25)
    
    # Print summary table
    print("\nALGORITHM PERFORMANCE SUMMARY:")
    print("-" * 60)
    print(f"{'Algorithm':<12} {'Return%':<8} {'Spread':<8} {'Sharpe':<8} {'Adverse%':<9} {'Volume':<8}")
    print("-" * 60)
    
    for algo_name, result in results.items():
        metrics = result['performance_metrics']
        print(f"{algo_name.upper():<12} "
                f"{metrics['total_return_pct']:<7.2f}% "
                f"{metrics['avg_spread_bps']:<7.1f} "
                f"{result['sharpe_ratio']:<7.2f} "
                f"{metrics['adverse_selection_rate']*100:<8.1f}% "
                f"{metrics['total_volume_traded']:<8.0f}")
    
    # Identify best performers
    print(f"\n4. KEY FINDINGS")
    print("-" * 15)
    
    best_return = max(results.items(), key=lambda x: x[1]['performance_metrics']['total_return_pct'])
    best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
    lowest_adverse = min(results.items(), key=lambda x: x[1]['performance_metrics']['adverse_selection_rate'])
    most_efficient = max(results.items(), key=lambda x: x[1]['spread_efficiency'])
    
    print(f"🏆 Best Return: {best_return[0].upper()} ({best_return[1]['performance_metrics']['total_return_pct']:.2f}%)")
    print(f"📈 Best Sharpe: {best_sharpe[0].upper()} ({best_sharpe[1]['sharpe_ratio']:.2f})")
    print(f"🛡️  Lowest Adverse Selection: {lowest_adverse[0].upper()} ({lowest_adverse[1]['performance_metrics']['adverse_selection_rate']*100:.1f}%)")
    print(f"⚡ Most Efficient: {most_efficient[0].upper()} ({most_efficient[1]['spread_efficiency']:.3f})")
    
    # Statistical significance (simplified)
    returns = [result['performance_metrics']['total_return_pct'] for result in results.values()]
    return_std = np.std(returns)
    print(f"\n📊 Return Variability: {return_std:.2f}% (higher = more differentiation)")
    
    # Create visualizations
    print(f"\n5. CREATING VISUALIZATIONS")
    print("-" * 30)
    
    fig = create_comparison_visualizations(results)
    
    print("✓ Comparison plots saved")
    
    # Save results
    results_df = pd.DataFrame({
        algo: {
            'return_pct': result['performance_metrics']['total_return_pct'],
            'avg_spread_bps': result['performance_metrics']['avg_spread_bps'],
            'sharpe_ratio': result['sharpe_ratio'],
            'adverse_selection_rate': result['performance_metrics']['adverse_selection_rate'],
            'max_drawdown': result['max_drawdown'],
            'total_volume': result['performance_metrics']['total_volume_traded'],
            'spread_efficiency': result['spread_efficiency'],
            'toxicity_rate': result['toxicity_rate']
        }
        for algo, result in results.items()
    }).T
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"ml_comparison_results_{timestamp}.csv")
    
    print(f"✓ Results saved to: ml_comparison_results_{timestamp}.csv")
    
    print("\n" + "="*80)
    print("ML-ENHANCED COMPARISON STUDY COMPLETED")
    print("="*80)
    
    # Final recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    print("-" * 20)
    
    if best_return[0] == 'baseline':
        print("• The baseline algorithm performed best - ML adjustments may be too aggressive")
    elif best_return[0] == 'linear':
        print("• Linear adjustment provides good balance of performance and simplicity")
    elif best_return[0] == 'exponential':
        print("• Exponential adjustment effectively responds to high toxicity scenarios")
    elif best_return[0] == 'threshold':
        print("• Threshold-based approach provides robust step-wise protection")
    
    print(f"• Consider tuning β parameters for {best_return[0]} algorithm")
    print(f"• Monitor adverse selection rates below {lowest_adverse[1]['performance_metrics']['adverse_selection_rate']*100:.1f}%")
    print(f"• Target spread efficiency above {most_efficient[1]['spread_efficiency']:.3f}")
    
    return results, env

    # Example usage
if __name__ == "__main__":
    # Run the comparison study
    model_path = "calibrated_toxicity_models/enhanced_toxicity_detector_20250704_004512.joblib"
    
    try:
        print("🚀 Starting ML-Enhanced Market Maker Comparison...")
        
        results, environment = run_ml_enhanced_comparison(
            model_path=model_path,
            n_steps=1000
        )
        
        if results:
            print("\n✅ Comparison study completed successfully!")
            print("Check the generated plots and CSV file for detailed results.")
            
            # Print key results even if visualization fails
            print(f"\n📊 QUICK RESULTS SUMMARY:")
            for algo, result in results.items():
                ret = result['performance_metrics']['total_return_pct']
                spread = result['performance_metrics']['avg_spread_bps']
                print(f"  {algo.upper()}: Return={ret:.2f}%, AvgSpread={spread:.1f}bps")
        else:
            print("❌ No results generated")
        
    except Exception as e:
        print(f"❌ Error running comparison: {e}")
        print("This might be due to:")
        print("1. Missing or incompatible model file")
        print("2. Missing dependencies")
        print("3. Visualization issues")
        
        # Try to run with fallback mode
        print("\n🔄 Attempting fallback mode...")
        try:
            model_path = None  # This will trigger fallback mode
            results, environment = run_ml_enhanced_comparison(
                model_path=model_path,
                n_steps=500
            )
            print("✅ Fallback mode completed!")
        except:
            print("❌ Fallback mode also failed")
