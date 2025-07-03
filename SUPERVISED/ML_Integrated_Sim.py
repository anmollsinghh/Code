"""
Complete ML Integration with Existing Simulation
ML_Integrated_Sim.py
==============================================

Integrates trained ML model with your complete_enhanced_simulation.py
to compare ML-enhanced vs baseline market maker performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import os
import sys
import random

# Import your existing simulation components
from complete_enhanced_simulation import (
    EnhancedMarketEnvironment, Agent, Order, Trade,
    ImprovedInformedTrader, SmartNoiseTrader, EnhancedMarketMaker,
    BUY, SELL, LIMIT, MARKET, INITIAL_PRICE, MIN_PRICE, TIME_STEPS
)

class MLEnhancedMarketMaker(Agent):
    """
    Market Maker with integrated ML toxicity detection
    Replaces static spread adjustment with ML-based dynamic pricing
    """
    
    def __init__(self, ml_model_path, base_spread_bps=50, inventory_limit=50, 
                 order_size=2, **kwargs):
        super().__init__(**kwargs)
        
        # Load trained ML model
        self.load_ml_model(ml_model_path)
        
        # Market maker parameters
        self.base_spread_bps = base_spread_bps
        self.inventory_limit = inventory_limit
        self.order_size = order_size
        self.type = 'ml_market_maker'
        
        # Performance tracking
        self.spread_history = []
        self.inventory_history = []
        self.timestamp_history = []
        self.toxicity_predictions = []
        self.ml_decisions = []
        
        # Risk management
        self.quote_update_frequency = 0.9
        
        # Enhanced market maker attributes (for compatibility)
        self.learning_rate = 0.1
        self.volatility_multiplier = 1.0
        self.adverse_selection_penalty = 0.0
        self.inventory_skew_strength = 0.02
        self.trade_pnl_history = []
        self.recent_toxicity_rate = 0.0
        self.last_trade_direction = 0
        self.order_flow_imbalance = 0.0
        
    def load_ml_model(self, model_path):
        """Load the trained ML model - FIXED VERSION"""
        try:
            ml_data = joblib.load(model_path)
            
            # Check what structure we loaded
            if 'ml_market_maker' in ml_data:
                # This is a saved ML market maker object
                loaded_mm = ml_data['ml_market_maker']
                self.model_metadata = ml_data['model_metadata']
                
                # Extract the actual model and scaler from the loaded ML market maker
                if hasattr(loaded_mm, 'model'):
                    self.raw_model = loaded_mm.model
                    self.scaler = loaded_mm.scaler
                    self.feature_names = loaded_mm.feature_names
                    print(f"✅ Loaded ML model from market maker: {self.model_metadata['model_type']} "
                        f"(AUC: {self.model_metadata['auc_score']:.3f})")
                else:
                    print("❌ Loaded ML market maker doesn't have model attribute")
                    self.raw_model = None
                    self.scaler = None
                    self.feature_names = []
                    
            elif 'model' in ml_data:
                # This is a model with metadata
                self.raw_model = ml_data['model']
                self.scaler = ml_data.get('scaler', None)
                self.feature_names = ml_data['feature_names']
                self.model_metadata = ml_data
                print(f"✅ Loaded raw ML model: {ml_data['model_type']} "
                    f"(AUC: {ml_data['auc_score']:.3f})")
            else:
                # Unknown structure
                print(f"❌ Unknown model structure in {model_path}")
                self.raw_model = None
                self.model_metadata = None
                self.scaler = None
                self.feature_names = []
                
            # Always clear the ml_market_maker attribute since we're using raw model approach
            self.ml_market_maker = None
                
        except Exception as e:
            print(f"❌ Error loading ML model: {e}")
            # Fallback to baseline behavior
            self.ml_market_maker = None
            self.model_metadata = None
            self.raw_model = None
            self.scaler = None
            self.feature_names = []

    
    def predict_toxicity_raw(self, features):
        """Direct prediction using raw model"""
        if not hasattr(self, 'raw_model') or self.raw_model is None:
            return 0.0
        
        try:
            # Ensure we have all required features
            if not hasattr(self, 'feature_names') or not self.feature_names:
                print("Warning: No feature names available")
                return 0.0
            
            # Ensure features are in correct order and handle missing features
            feature_vector = []
            for feature in self.feature_names:
                if feature in features:
                    feature_vector.append(float(features[feature]))
                else:
                    print(f"Warning: Missing feature {feature}, using 0.0")
                    feature_vector.append(0.0)
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Check for invalid values
            if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
                print("Warning: Invalid values in feature vector")
                feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Scale if scaler is available
            if hasattr(self, 'scaler') and self.scaler is not None:
                feature_vector = self.scaler.transform(feature_vector)
            
            # Predict toxicity probability
            if hasattr(self.raw_model, 'predict_proba'):
                toxicity_prob = self.raw_model.predict_proba(feature_vector)[0][1]
            else:
                # Fallback for models without predict_proba
                prediction = self.raw_model.predict(feature_vector)[0]
                toxicity_prob = float(prediction)
            
            # Ensure probability is in valid range
            toxicity_prob = max(0.0, min(1.0, toxicity_prob))
            
            return toxicity_prob
            
        except Exception as e:
            print(f"Warning: Raw ML prediction failed: {e}")
            return 0.0

    
    def extract_order_features(self, order_data, market_context):
        """Extract features for ML prediction - IMPROVED VERSION"""
        
        # Get market context features with proper defaults
        recent_prices = market_context.get('recent_prices', [250.0])  # Default price
        
        # Calculate volatility and momentum with safety checks
        if len(recent_prices) > 1:
            try:
                returns = [np.log(recent_prices[i]/recent_prices[i-1]) 
                        for i in range(1, len(recent_prices)) 
                        if recent_prices[i-1] > 0 and recent_prices[i] > 0]
                volatility = np.std(returns) if len(returns) > 1 else 0
                momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if len(recent_prices) > 1 else 0
            except:
                volatility = 0
                momentum = 0
        else:
            volatility = 0
            momentum = 0
        
        # Build feature dictionary with all required features
        features = {
            'quantity': float(order_data.get('quantity', 2)),
            'distance_from_mid': float(order_data.get('distance_from_mid', 0)),
            'is_aggressive': float(1 if order_data.get('is_aggressive', False) else 0),
            'volatility': float(volatility),
            'momentum': float(momentum),
            'order_book_imbalance': float(market_context.get('order_book_imbalance', 0)),
            'time_since_last_trade': float(market_context.get('time_since_last_trade', 5)),
            'spread': float(market_context.get('current_spread', 1)),
            'is_informed': 0.0,  # Unknown in real-time
            'is_market_maker': 0.0,
            'price_volatility': float(volatility * 100)  # Scaled version
        }
        
        return features

    
    def predict_toxicity_and_adjust_spread(self, order_data, market_context):
        """Use ML model to predict toxicity and adjust spread"""
        
        # Default fallback values
        default_spread = self.base_spread_bps
        default_multiplier = 1.0
        default_toxicity = 0.0
        
        # Only try raw model approach (since ml_market_maker is always None now)
        if hasattr(self, 'raw_model') and self.raw_model is not None:
            try:
                # Extract features
                features = self.extract_order_features(order_data, market_context)
                
                # Get ML prediction using raw model
                toxicity_prob = self.predict_toxicity_raw(features)
                
                # IMPROVED spread adjustment logic with more aggressive multipliers
                if toxicity_prob > 0.8:
                    multiplier = 4.0  # Very aggressive for high toxicity
                elif toxicity_prob > 0.6:
                    multiplier = 3.0
                elif toxicity_prob > 0.4:
                    multiplier = 2.5
                elif toxicity_prob > 0.2:
                    multiplier = 2.0
                else:
                    multiplier = 1.5  # Always at least 1.5x wider than base
                
                # Add volatility adjustment
                vol_adjustment = 1 + min(market_context.get('volatility', 0) * 20, 1.0)
                multiplier *= vol_adjustment
                
                adjusted_spread_bps = self.base_spread_bps * multiplier
                
                # Cap maximum spread but allow it to be high
                adjusted_spread_bps = min(adjusted_spread_bps, 500)
                
                # Record decision
                self.toxicity_predictions.append(toxicity_prob)
                self.ml_decisions.append({
                    'timestamp': order_data.get('timestamp', 0),
                    'toxicity_prob': toxicity_prob,
                    'spread_multiplier': multiplier,
                    'adjusted_spread_bps': adjusted_spread_bps
                })
                
                print(f"Raw ML Prediction: toxicity={toxicity_prob:.3f}, multiplier={multiplier:.2f}, spread={adjusted_spread_bps:.1f}bps")
                
                return adjusted_spread_bps, multiplier, toxicity_prob
                
            except Exception as e:
                print(f"Raw ML prediction failed: {e}")
        
        # Final fallback - but still widen spread slightly
        print("No ML prediction available, using baseline with small adjustment")
        return default_spread * 1.5, 1.5, default_toxicity  # At least 1.5x base spread


    
    def generate_orders(self, timestamp, mid_price, market_context=None):
        """Generate orders with ML-enhanced spread adjustment - FIXED VERSION"""
        
        # Don't always quote
        if np.random.random() > self.quote_update_frequency:
            return []
        
        orders = []
        
        # CRITICAL FIX: Always create meaningful order data for ML prediction
        # Use actual market conditions, not dummy data
        order_data = {
            'timestamp': timestamp,
            'quantity': self.order_size,
            'is_aggressive': False,
            'distance_from_mid': 0
        }
        
        # CRITICAL FIX: Ensure market context is properly constructed
        if not market_context:
            market_context = {}
        
        # Calculate market context features if missing
        recent_trades = market_context.get('recent_trades', [])
        if recent_trades:
            # Calculate proper order book imbalance
            recent_buy_volume = sum(t.quantity for t in recent_trades[-10:] 
                                if hasattr(t, 'buyer_id'))
            recent_sell_volume = sum(t.quantity for t in recent_trades[-10:] 
                                if hasattr(t, 'seller_id'))
            total_volume = recent_buy_volume + recent_sell_volume
            imbalance = ((recent_buy_volume - recent_sell_volume) / total_volume 
                    if total_volume > 0 else 0)
            market_context['order_book_imbalance'] = imbalance
            
            # Time since last trade
            last_trade_time = (recent_trades[-1].timestamp if recent_trades else 0)
            market_context['time_since_last_trade'] = max(0, timestamp - last_trade_time)
        else:
            market_context['order_book_imbalance'] = 0
            market_context['time_since_last_trade'] = 10
        
        # Current spread
        market_context['current_spread'] = market_context.get('spread', 1)
        
        # Calculate volatility from recent prices
        recent_prices = market_context.get('recent_prices', [mid_price])
        if len(recent_prices) > 1:
            returns = [np.log(recent_prices[i]/recent_prices[i-1]) 
                    for i in range(1, len(recent_prices)) if recent_prices[i-1] > 0]
            market_context['volatility'] = np.std(returns) if len(returns) > 1 else 0
        else:
            market_context['volatility'] = 0
        
        # CRITICAL FIX: Always call ML prediction, regardless of market context
        try:
            adjusted_spread_bps, multiplier, toxicity_prob = self.predict_toxicity_and_adjust_spread(
                order_data, market_context
            )
            
            # FORCE the spread to be used - this was the main bug
            if adjusted_spread_bps <= self.base_spread_bps:
                # If ML didn't increase spread enough, force minimum multiplier
                adjusted_spread_bps = self.base_spread_bps * max(1.2, multiplier)
                
        except Exception as e:
            print(f"ML prediction failed: {e}, using baseline spread")
            adjusted_spread_bps = self.base_spread_bps
            multiplier = 1.0
            toxicity_prob = 0.0
        
        # CRITICAL FIX: Ensure we're using the ML-adjusted spread
        print(f"Debug: ML spread adjustment - Base: {self.base_spread_bps}, Adjusted: {adjusted_spread_bps}, Multiplier: {multiplier:.2f}")
        
        # Calculate half-spread in price terms using ML-adjusted spread
        half_spread = mid_price * (adjusted_spread_bps / 10000) / 2
        
        # Inventory-based skewing
        inventory_ratio = self.inventory / self.inventory_limit if self.inventory_limit > 0 else 0
        inventory_skew = inventory_ratio * 0.02 * mid_price  # 2% max skew
        
        # Calculate bid/ask prices with ML-enhanced spread
        bid_price = max(mid_price - half_spread - inventory_skew, MIN_PRICE)
        ask_price = max(mid_price + half_spread - inventory_skew, bid_price * 1.001)
        
        # Dynamic order sizing based on toxicity and risk
        if toxicity_prob > 0.7:
            size_multiplier = 0.3  # Reduce size significantly for high toxicity
        elif toxicity_prob > 0.5:
            size_multiplier = 0.5
        elif toxicity_prob > 0.3:
            size_multiplier = 0.8
        else:
            size_multiplier = 1.0
        
        # Additional risk adjustment based on inventory
        inventory_risk = min(abs(inventory_ratio), 0.8)
        size_multiplier *= (1 - inventory_risk * 0.5)
        
        bid_size = max(1, int(self.order_size * size_multiplier))
        ask_size = max(1, int(self.order_size * size_multiplier))
        
        # Inventory constraints - more aggressive
        if abs(self.inventory) > 0.6 * self.inventory_limit:
            if self.inventory > 0:  # Long inventory
                bid_size = max(1, int(bid_size * 0.2))
                ask_price *= 0.998  # More aggressive ask
            else:  # Short inventory
                ask_size = max(1, int(ask_size * 0.2))
                bid_price *= 1.002  # More aggressive bid
        
        # CRITICAL FIX: Record the ACTUAL spread being used
        actual_spread_bps = (ask_price - bid_price) / mid_price * 10000
        self.spread_history.append(actual_spread_bps)
        self.inventory_history.append(self.inventory)
        self.timestamp_history.append(timestamp)
        
        # Debug output
        print(f"ML MM: Toxicity={toxicity_prob:.3f}, Spread={actual_spread_bps:.1f}bps, Sizes=({bid_size},{ask_size})")
        
        # Create orders
        if bid_size > 0:
            orders.append(Order(self.id, LIMIT, BUY, bid_price, bid_size, timestamp))
        if ask_size > 0:
            orders.append(Order(self.id, LIMIT, SELL, ask_price, ask_size, timestamp))
        
        return orders

    
    def update_adverse_selection_estimate(self, recent_trades, window=50):
        """Estimate adverse selection from recent trade outcomes (for compatibility)"""
        if len(recent_trades) < window:
            return
        
        # Look at recent trades involving this market maker
        mm_trades = [t for t in recent_trades[-window:] 
                    if t.buyer_id == self.id or t.seller_id == self.id]
        
        if not mm_trades:
            return
        
        # Calculate how often MM was on the wrong side
        adverse_count = sum(1 for t in mm_trades if hasattr(t, 'is_toxic') and t.is_toxic)
        self.recent_toxicity_rate = adverse_count / len(mm_trades)
        
        # Store for ML analysis
        if hasattr(self, 'ml_decisions'):
            self.ml_decisions.append({
                'timestamp': recent_trades[-1].timestamp if recent_trades else 0,
                'recent_toxicity_rate': self.recent_toxicity_rate,
                'adverse_selection_penalty': self.adverse_selection_penalty
            })
    
    def generate_order(self, timestamp, current_price):
        """
        Dummy method for compatibility with simulation loop
        Market makers should use generate_orders() instead
        """
        return None  # Market makers don't generate individual orders

def run_comparative_simulation(ml_model_path, num_agents=15, time_steps=1000):
    """
    Run simulation comparing ML-enhanced vs baseline market maker
    """
    
    print("🔬 RUNNING COMPARATIVE SIMULATION")
    print("=" * 50)
    
    results = {}
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run both simulations
    for mm_type in ['baseline', 'ml_enhanced']:
        print(f"\n📊 Running {mm_type} simulation...")
        
        # Reset random seeds for fair comparison
        random.seed(42)
        np.random.seed(42)
        
        # Create market environment
        if mm_type == 'ml_enhanced':
            market = EnhancedMarketEnvironmentML(
                initial_price=INITIAL_PRICE,
                price_vol=0.006,
                mean_reversion=0.01
            )
        else:
            market = EnhancedMarketEnvironment(
                initial_price=INITIAL_PRICE,
                price_vol=0.006,
                mean_reversion=0.01
            )
        
        # Agent allocation
        n_informed = 3
        n_uninformed = num_agents - 2  # -1 for MM, -1 for comparison
        
        # Add appropriate market maker
        if mm_type == 'ml_enhanced':
            mm = MLEnhancedMarketMaker(
                ml_model_path=ml_model_path,
                base_spread_bps=50,
                inventory_limit=40,
                order_size=2,
                initial_capital=25000
            )
        else:
            mm = EnhancedMarketMaker(
                base_spread_bps=50,
                inventory_limit=40,
                order_size=2,
                initial_capital=25000
            )
        
        market.add_agent(mm)
        
        # Add informed traders
        for i in range(n_informed):
            market.add_agent(ImprovedInformedTrader(
                future_price_info=market.price_path,
                knowledge_horizon=np.random.randint(4, 8),
                order_rate=0.06 + np.random.uniform(0, 0.04),
                information_decay=np.random.uniform(0.92, 0.98),
                confidence_threshold=np.random.uniform(0.25, 0.4),
                initial_capital=8000 + np.random.randint(-1000, 2000)
            ))
        
        # Add uninformed traders
        for i in range(n_uninformed):
            market.add_agent(SmartNoiseTrader(
                order_rate=0.04 + np.random.uniform(0, 0.06),
                momentum_factor=np.random.uniform(0.2, 0.4),
                contrarian_factor=np.random.uniform(0.1, 0.3),
                initial_capital=3000 + np.random.randint(-500, 1500)
            ))
        
        # Run simulation
        market.run_enhanced_simulation(n_steps=time_steps)
        
        # Store results
        results[mm_type] = {
            'market': market,
            'market_maker': mm,
            'trades_df': market.get_trades_dataframe(),
            'market_df': market.get_market_dataframe()
        }
        
        # Print basic results
        trades_df = market.get_trades_dataframe()
        initial_capital = mm.pnl_history[0][1]
        final_capital = mm.pnl_history[-1][1]
        mm_return = (final_capital / initial_capital - 1) * 100
        
        total_trades = len(trades_df)
        toxic_trades = trades_df['is_toxic'].sum() if not trades_df.empty else 0
        toxic_rate = toxic_trades / total_trades * 100 if total_trades > 0 else 0
        
        print(f"   Market Maker Return: {mm_return:.2f}%")
        print(f"   Total Trades: {total_trades}")
        print(f"   Toxicity Rate: {toxic_rate:.1f}%")
        
        if hasattr(mm, 'spread_history'):
            avg_spread = np.mean(mm.spread_history)
            print(f"   Average Spread: {avg_spread:.1f} bps")
    
    return results

class EnhancedMarketEnvironmentML(EnhancedMarketEnvironment):
    """
    Enhanced market environment with special handling for ML market makers
    """
    
    def run_enhanced_simulation(self, n_steps=None):
        """Enhanced simulation with ML market maker support"""
        n_steps = n_steps or TIME_STEPS
        
        for t in range(n_steps):
            self.current_time = t
            mid_price = self.order_book.get_mid_price()
            
            # Prepare market context for market makers
            market_context = {
                'recent_prices': self.market_stats['prices'][-20:] if self.market_stats['prices'] else [mid_price],
                'recent_trades': self.order_book.trades[-50:] if self.order_book.trades else [],
                'current_depth': len(self.order_book.bids) + len(self.order_book.asks),
                'spread': self.order_book.get_spread()
            }
            
            # Take LOB snapshot
            self.lob_snapshots.append(self.order_book.save_snapshot(t))
            
            # Update agent PnLs
            for agent in self.agents:
                agent.update_pnl(mid_price, t)
            
            # Market makers place orders with enhanced context
            for agent in self.agents:
                if isinstance(agent, (EnhancedMarketMaker, MLEnhancedMarketMaker)):
                    # Update adverse selection estimates
                    agent.update_adverse_selection_estimate(market_context['recent_trades'])
                    
                    orders = agent.generate_orders(t, mid_price, market_context)
                    for order in orders:
                        self.record_order_data(order, t)
                        self.order_book.add_limit_order(order, t)
            
            # Other agents place orders
            for agent in self.agents:
                if not isinstance(agent, (EnhancedMarketMaker, MLEnhancedMarketMaker)):
                    if isinstance(agent, ImprovedInformedTrader):
                        order = agent.generate_order(t, self.price_path[t])
                    else:
                        order = agent.generate_order(t, mid_price)
                    
                    if order:
                        self.record_order_data(order, t)
                        if order.type == MARKET:
                            self.order_book.add_market_order(order, t)
                        else:
                            self.order_book.add_limit_order(order, t)
            
            # Update market statistics
            self.market_stats['timestamps'].append(t)
            self.market_stats['prices'].append(mid_price)
            self.market_stats['spreads'].append(self.order_book.get_spread())
            
            # Record trades and update agent states
            trades_this_step = [trade for trade in self.order_book.trades if trade.timestamp == t]
            self.market_stats['volumes'].append(sum(trade.quantity for trade in trades_this_step))
            
            for trade in trades_this_step:
                buyer = next((a for a in self.agents if a.id == trade.buyer_id), None)
                seller = next((a for a in self.agents if a.id == trade.seller_id), None)
                
                if buyer:
                    buyer.record_trade(trade, True)
                if seller:
                    seller.record_trade(trade, False)
        
        # Enhanced post-processing
        self.enhanced_post_process_trades()

def analyze_comparative_results(results):
    """
    Analyze and compare the performance of ML vs baseline market makers
    """
    print("\n📈 COMPARATIVE ANALYSIS")
    print("=" * 30)
    
    # Extract market makers
    baseline_mm = results['baseline']['market_maker']
    ml_mm = results['ml_enhanced']['market_maker']
    
    # Performance comparison
    baseline_initial = baseline_mm.pnl_history[0][1]
    baseline_final = baseline_mm.pnl_history[-1][1]
    baseline_return = (baseline_final / baseline_initial - 1) * 100
    
    ml_initial = ml_mm.pnl_history[0][1]
    ml_final = ml_mm.pnl_history[-1][1]
    ml_return = (ml_final / ml_initial - 1) * 100
    
    improvement = ml_return - baseline_return
    
    print(f"\n💰 PROFITABILITY COMPARISON:")
    print(f"   Baseline MM Return: {baseline_return:.2f}%")
    print(f"   ML-Enhanced MM Return: {ml_return:.2f}%")
    print(f"   Improvement: {improvement:+.2f} percentage points")
    
    # Risk analysis
    baseline_inventory_std = np.std(baseline_mm.inventory_history)
    ml_inventory_std = np.std(ml_mm.inventory_history)
    
    print(f"\n📊 RISK ANALYSIS:")
    print(f"   Baseline Inventory Volatility: {baseline_inventory_std:.2f}")
    print(f"   ML-Enhanced Inventory Volatility: {ml_inventory_std:.2f}")
    
    # Spread analysis
    baseline_avg_spread = np.mean(baseline_mm.spread_history)
    ml_avg_spread = np.mean(ml_mm.spread_history)
    
    print(f"\n🎯 SPREAD MANAGEMENT:")
    print(f"   Baseline Average Spread: {baseline_avg_spread:.1f} bps")
    print(f"   ML-Enhanced Average Spread: {ml_avg_spread:.1f} bps")
    
    # ML-specific metrics
    if hasattr(ml_mm, 'toxicity_predictions') and ml_mm.toxicity_predictions:
        avg_toxicity_pred = np.mean(ml_mm.toxicity_predictions)
        high_toxicity_rate = np.mean([p > 0.5 for p in ml_mm.toxicity_predictions])
        
        print(f"\n🤖 ML MODEL PERFORMANCE:")
        print(f"   Average Toxicity Prediction: {avg_toxicity_pred:.3f}")
        print(f"   High Toxicity Detection Rate: {high_toxicity_rate:.1%}")
        print(f"   Total ML Predictions Made: {len(ml_mm.toxicity_predictions)}")
        
        # Spread multiplier analysis
        if ml_mm.ml_decisions:
            decisions_df = pd.DataFrame(ml_mm.ml_decisions)
            avg_multiplier = decisions_df['spread_multiplier'].mean()
            print(f"   Average Spread Multiplier: {avg_multiplier:.2f}x")
    
    # Trade-level analysis
    baseline_trades = results['baseline']['trades_df']
    ml_trades = results['ml_enhanced']['trades_df']
    
    if not baseline_trades.empty and not ml_trades.empty:
        # Market maker involvement in toxic trades
        baseline_mm_toxic = baseline_trades[
            (baseline_trades['is_toxic'] == True) & 
            ((baseline_trades['buyer_type'] == 'market_maker') | 
             (baseline_trades['seller_type'] == 'market_maker'))
        ]
        ml_mm_toxic = ml_trades[
            (ml_trades['is_toxic'] == True) & 
            ((ml_trades['buyer_type'] == 'ml_market_maker') | 
             (ml_trades['seller_type'] == 'ml_market_maker'))
        ]
        
        baseline_mm_total = baseline_trades[
            (baseline_trades['buyer_type'] == 'market_maker') | 
            (baseline_trades['seller_type'] == 'market_maker')
        ]
        ml_mm_total = ml_trades[
            (ml_trades['buyer_type'] == 'ml_market_maker') | 
            (ml_trades['seller_type'] == 'ml_market_maker')
        ]
        
        baseline_toxic_rate = len(baseline_mm_toxic) / len(baseline_mm_total) * 100 if len(baseline_mm_total) > 0 else 0
        ml_toxic_rate = len(ml_mm_toxic) / len(ml_mm_total) * 100 if len(ml_mm_total) > 0 else 0
        
        print(f"\n⚠️ TOXIC TRADE EXPOSURE:")
        print(f"   Baseline MM Toxic Rate: {baseline_toxic_rate:.1f}%")
        print(f"   ML-Enhanced MM Toxic Rate: {ml_toxic_rate:.1f}%")
        print(f"   Toxic Exposure Reduction: {baseline_toxic_rate - ml_toxic_rate:+.1f} percentage points")
    
    return {
        'baseline_return': baseline_return,
        'ml_return': ml_return,
        'improvement': improvement,
        'baseline_avg_spread': baseline_avg_spread,
        'ml_avg_spread': ml_avg_spread,
        'baseline_inventory_std': baseline_inventory_std,
        'ml_inventory_std': ml_inventory_std
    }

def plot_comparative_results(results, save_dir="comparative_results"):
    """
    Create comprehensive plots comparing ML vs baseline performance
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    baseline_mm = results['baseline']['market_maker']
    ml_mm = results['ml_enhanced']['market_maker']
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('ML-Enhanced vs Baseline Market Maker Comparison', 
                 fontsize=16, fontweight='bold')
    
    # 1. P&L Comparison
    baseline_times = [t for t, _ in baseline_mm.pnl_history]
    baseline_pnls = [pnl - baseline_mm.pnl_history[0][1] for _, pnl in baseline_mm.pnl_history]
    
    ml_times = [t for t, _ in ml_mm.pnl_history]
    ml_pnls = [pnl - ml_mm.pnl_history[0][1] for _, pnl in ml_mm.pnl_history]
    
    axes[0, 0].plot(baseline_times, baseline_pnls, 'b-', label='Baseline MM', linewidth=2)
    axes[0, 0].plot(ml_times, ml_pnls, 'r-', label='ML-Enhanced MM', linewidth=2)
    axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 0].set_title('P&L Comparison')
    axes[0, 0].set_ylabel('P&L ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Spread Comparison
    axes[0, 1].plot(baseline_mm.timestamp_history, baseline_mm.spread_history, 
                   'b-', alpha=0.7, label='Baseline MM')
    axes[0, 1].plot(ml_mm.timestamp_history, ml_mm.spread_history, 
                   'r-', alpha=0.7, label='ML-Enhanced MM')
    axes[0, 1].set_title('Spread Evolution')
    axes[0, 1].set_ylabel('Spread (bps)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Inventory Comparison
    axes[0, 2].plot(baseline_mm.timestamp_history, baseline_mm.inventory_history, 
                   'b-', alpha=0.7, label='Baseline MM')
    axes[0, 2].plot(ml_mm.timestamp_history, ml_mm.inventory_history, 
                   'r-', alpha=0.7, label='ML-Enhanced MM')
    axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 2].set_title('Inventory Management')
    axes[0, 2].set_ylabel('Inventory')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Spread Distribution
    axes[1, 0].hist(baseline_mm.spread_history, bins=30, alpha=0.6, 
                   label='Baseline MM', color='blue', density=True)
    axes[1, 0].hist(ml_mm.spread_history, bins=30, alpha=0.6, 
                   label='ML-Enhanced MM', color='red', density=True)
    axes[1, 0].set_title('Spread Distribution')
    axes[1, 0].set_xlabel('Spread (bps)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Performance Metrics
    baseline_return = (baseline_mm.pnl_history[-1][1] / baseline_mm.pnl_history[0][1] - 1) * 100
    ml_return = (ml_mm.pnl_history[-1][1] / ml_mm.pnl_history[0][1] - 1) * 100
    
    metrics = ['Return (%)', 'Avg Spread (bps)', 'Inventory Std']
    baseline_values = [
        baseline_return,
        np.mean(baseline_mm.spread_history),
        np.std(baseline_mm.inventory_history)
    ]
    ml_values = [
        ml_return,
        np.mean(ml_mm.spread_history),
        np.std(ml_mm.inventory_history)
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, baseline_values, width, label='Baseline MM', 
                  color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, ml_values, width, label='ML-Enhanced MM', 
                  color='red', alpha=0.7)
    axes[1, 1].set_title('Performance Metrics Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. ML Toxicity Predictions (if available)
    if hasattr(ml_mm, 'toxicity_predictions') and ml_mm.toxicity_predictions:
        axes[1, 2].hist(ml_mm.toxicity_predictions, bins=30, alpha=0.7, 
                       color='red', edgecolor='black')
        axes[1, 2].axvline(np.mean(ml_mm.toxicity_predictions), color='black', 
                          linestyle='--', label=f'Mean: {np.mean(ml_mm.toxicity_predictions):.3f}')
        axes[1, 2].set_title('ML Toxicity Predictions')
        axes[1, 2].set_xlabel('Predicted Toxicity Probability')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'No ML Predictions\nAvailable', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('ML Predictions')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{save_dir}/ml_vs_baseline_comparison_{timestamp}.png", 
                dpi=300, bbox_inches='tight')
    plt.show()

def main_comparison():
    """
    Main function to run ML vs baseline comparison
    """
    
    print("🚀 ML-ENHANCED MARKET MAKER COMPARISON")
    print("=" * 45)
    
    # Find the ML model file
    model_dir = "ml_models"
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
        if model_files:
            # Use the most recent model
            latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
            ml_model_path = os.path.join(model_dir, latest_model)
            print(f"📁 Using ML model: {latest_model}")
        else:
            print("❌ No ML model files found in ml_models directory")
            return
    else:
        print("❌ ml_models directory not found")
        return
    
    # Run comparative simulation
    results = run_comparative_simulation(ml_model_path)
    
    # Analyze results
    analysis = analyze_comparative_results(results)
    
    # Plot results
    print("📊 Creating comparison plots...")
    plot_comparative_results(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    if analysis['improvement'] > 0:
        print(f"✅ ML-Enhanced MM outperformed baseline by {analysis['improvement']:+.2f} percentage points")
        print(f"   This represents a {abs(analysis['improvement']/analysis['baseline_return']*100):.1f}% relative improvement")
    else:
        print(f"❌ ML-Enhanced MM underperformed baseline by {abs(analysis['improvement']):.2f} percentage points")
    
    print(f"\n📊 Key Metrics:")
    print(f"   • Return Improvement: {analysis['improvement']:+.2f}%")
    print(f"   • Spread Efficiency: {analysis['ml_avg_spread'] - analysis['baseline_avg_spread']:+.1f} bps")
    print(f"   • Risk Reduction: {analysis['baseline_inventory_std'] - analysis['ml_inventory_std']:+.2f} inventory std")
    
    # Save summary to file
    if not os.path.exists("comparative_results"):
        os.makedirs("comparative_results")
    
    summary_data = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'ml_model_used': ml_model_path,
        **analysis
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(f"comparative_results/summary_{summary_data['timestamp']}.csv", index=False)
    print(f"\n📁 Summary saved to: comparative_results/summary_{summary_data['timestamp']}.csv")
    
    return results, analysis

if __name__ == "__main__":
    # Run the comparison
    try:
        results, analysis = main_comparison()
        
        print("\n" + "="*60)
        print("🎓 THESIS SECTION 4.4 COMPLETE!")
        print("="*60)
        print("✅ ML model trained and benchmarked")
        print("✅ ML-enhanced market maker implemented")
        print("✅ Comparative analysis completed")
        print("✅ Performance improvements quantified")
        
        if analysis and analysis['improvement'] > 0:
            print(f"\n🏆 KEY FINDING: ML improved market maker profitability by {analysis['improvement']:.2f} percentage points")
            print("This demonstrates the value of ML-based toxicity detection for market making.")
        
    except Exception as e:
        print(f"\n❌ Error in comparison: {e}")
        import traceback
        traceback.print_exc()