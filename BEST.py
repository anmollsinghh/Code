import numpy as np
import pandas as pd
import heapq
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import os

# Define constants
INITIAL_PRICE = 250.0
MIN_PRICE = 0.01  # Price floor to prevent collapse to zero
TIME_STEPS = 1000
NUM_AGENTS = 15

# Order types
BUY = 1
SELL = -1
LIMIT = 0
MARKET = 1

class Order:
    """Represents an order in the market"""
    order_id = 0
    
    def __init__(self, agent_id, order_type, side, price=None, quantity=1, timestamp=None):
        """
        Initialize an order
        
        Parameters:
        - agent_id: ID of the agent placing the order
        - order_type: LIMIT or MARKET
        - side: BUY or SELL
        - price: Price of the limit order (None for market orders)
        - quantity: Size of the order
        - timestamp: Time the order was placed
        """
        Order.order_id += 1
        self.id = Order.order_id
        self.agent_id = agent_id
        self.type = order_type
        self.side = side
        self.price = price
        self.quantity = quantity
        self.timestamp = timestamp or 0
        self.is_toxic = False  # Will be set after execution if applicable
    
    def __lt__(self, other):
        """Less than comparison for heap ordering - compares by order ID for tie-breaking"""
        return self.id < other.id
    
    def __eq__(self, other):
        """Equality comparison"""
        return self.id == other.id
    
    def __str__(self):
        order_type = "LIMIT" if self.type == LIMIT else "MARKET"
        side = "BUY" if self.side == BUY else "SELL"
        return f"Order {self.id}: {order_type} {side} {self.quantity} @ {self.price if self.price else 'MKT'} (Agent {self.agent_id})"

class Trade:
    """Represents a completed trade"""
    def __init__(self, buy_order, sell_order, price, quantity, timestamp):
        self.buy_order_id = buy_order.id
        self.sell_order_id = sell_order.id
        self.buyer_id = buy_order.agent_id
        self.seller_id = sell_order.agent_id
        self.price = price
        self.quantity = quantity
        self.timestamp = timestamp
        self.is_toxic = False  # Will be set after analysis
    
    def __str__(self):
        return f"Trade: {self.quantity} @ {self.price} (Buyer: {self.buyer_id}, Seller: {self.seller_id})"

class LimitOrderBook:
    """Central limit order book for matching orders"""
    
    def __init__(self):
        # Price-ordered heaps (max heap for bids, min heap for asks)
        self.bids = []  # (-price, timestamp, order)
        self.asks = []  # (price, timestamp, order)
        self.last_price = INITIAL_PRICE
        self.trades = []
        self.order_history = []
        self.price_history = []

    def save_snapshot(self, timestamp):
        """Take a snapshot of the current LOB state for ML features"""
        state = self.get_book_state()
        
        # Prepare bid and ask data
        bid_data = {}
        ask_data = {}
        
        # Save up to 5 levels of depth (or fewer if not available)
        for i, (price, qty) in enumerate(state['bid_levels'][:5]):
            bid_data[f'bid_price_{i+1}'] = price
            bid_data[f'bid_size_{i+1}'] = qty
            
        for i, (price, qty) in enumerate(state['ask_levels'][:5]):
            ask_data[f'ask_price_{i+1}'] = price
            ask_data[f'ask_size_{i+1}'] = qty
            
        # Fill missing levels with NaN
        for i in range(len(state['bid_levels']), 5):
            bid_data[f'bid_price_{i+1}'] = np.nan
            bid_data[f'bid_size_{i+1}'] = np.nan
            
        for i in range(len(state['ask_levels']), 5):
            ask_data[f'ask_price_{i+1}'] = np.nan
            ask_data[f'ask_size_{i+1}'] = np.nan
            
        # Combine all data
        snapshot = {
            'timestamp': timestamp,
            'mid_price': state['mid_price'],
            'spread': state['spread'],
            **bid_data,
            **ask_data,
            'imbalance': self.calculate_imbalance(5)  # Order book imbalance (useful feature)
        }
        
        return snapshot
    
    def calculate_imbalance(self, depth=5):
        """Calculate order book imbalance (bid volume - ask volume) / (bid volume + ask volume)"""
        state = self.get_book_state()
        
        bid_volume = sum(qty for _, qty in state['bid_levels'][:depth])
        ask_volume = sum(qty for _, qty in state['ask_levels'][:depth])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0
            
        return (bid_volume - ask_volume) / total_volume
        
    def add_limit_order(self, order, timestamp):
        """Add a limit order to the book"""
        order.timestamp = timestamp
        self.order_history.append(order)
        
        if order.side == BUY:
            # Check if this buy order can be matched with existing sell orders
            while self.asks and order.quantity > 0:
                best_ask_price, best_ask_time, best_ask = self.asks[0]
                
                if order.price >= best_ask_price:
                    # Match found
                    trade_quantity = min(order.quantity, best_ask.quantity)
                    trade_price = best_ask_price  # Price-time priority - first order sets price
                    
                    # Create trade record
                    trade = Trade(order, best_ask, trade_price, trade_quantity, timestamp)
                    self.trades.append(trade)
                    self.last_price = trade_price
                    self.price_history.append((timestamp, trade_price))
                    
                    # Update quantities
                    order.quantity -= trade_quantity
                    best_ask.quantity -= trade_quantity
                    
                    # Remove or update the ask order
                    heapq.heappop(self.asks)
                    if best_ask.quantity > 0:
                        heapq.heappush(self.asks, (best_ask_price, best_ask_time, best_ask))
                else:
                    break
            
            # If order still has quantity, add to book
            if order.quantity > 0:
                heapq.heappush(self.bids, (-order.price, order.timestamp, order))
        
        elif order.side == SELL:
            # Check if this sell order can be matched with existing buy orders
            while self.bids and order.quantity > 0:
                best_bid_neg_price, best_bid_time, best_bid = self.bids[0]
                best_bid_price = -best_bid_neg_price
                
                if order.price <= best_bid_price:
                    # Match found
                    trade_quantity = min(order.quantity, best_bid.quantity)
                    trade_price = best_bid_price  # Price-time priority - first order sets price
                    
                    # Create trade record
                    trade = Trade(best_bid, order, trade_price, trade_quantity, timestamp)
                    self.trades.append(trade)
                    self.last_price = trade_price
                    self.price_history.append((timestamp, trade_price))
                    
                    # Update quantities
                    order.quantity -= trade_quantity
                    best_bid.quantity -= trade_quantity
                    
                    # Remove or update the bid order
                    heapq.heappop(self.bids)
                    if best_bid.quantity > 0:
                        heapq.heappush(self.bids, (best_bid_neg_price, best_bid_time, best_bid))
                else:
                    break
            
            # If order still has quantity, add to book
            if order.quantity > 0:
                heapq.heappush(self.asks, (order.price, order.timestamp, order))
    
    def add_market_order(self, order, timestamp):
        """Add a market order - immediately matches against the book"""
        order.timestamp = timestamp
        self.order_history.append(order)
        
        if order.side == BUY:
            while self.asks and order.quantity > 0:
                best_ask_price, best_ask_time, best_ask = self.asks[0]
                
                # For market buy, take whatever price is available
                trade_quantity = min(order.quantity, best_ask.quantity)
                trade_price = max(best_ask_price, MIN_PRICE)  # Apply price floor
                
                # Create trade record
                trade = Trade(order, best_ask, trade_price, trade_quantity, timestamp)
                self.trades.append(trade)
                self.last_price = trade_price
                self.price_history.append((timestamp, trade_price))
                
                # Update quantities
                order.quantity -= trade_quantity
                best_ask.quantity -= trade_quantity
                
                # Remove or update the ask order
                heapq.heappop(self.asks)
                if best_ask.quantity > 0:
                    heapq.heappush(self.asks, (best_ask_price, best_ask_time, best_ask))
        
        elif order.side == SELL:
            while self.bids and order.quantity > 0:
                best_bid_neg_price, best_bid_time, best_bid = self.bids[0]
                best_bid_price = -best_bid_neg_price
                
                # For market sell, take whatever price is available
                trade_quantity = min(order.quantity, best_bid.quantity)
                trade_price = max(best_bid_price, MIN_PRICE)  # Apply price floor
                
                # Create trade record
                trade = Trade(best_bid, order, trade_price, trade_quantity, timestamp)
                self.trades.append(trade)
                self.last_price = trade_price
                self.price_history.append((timestamp, trade_price))
                
                # Update quantities
                order.quantity -= trade_quantity
                best_bid.quantity -= trade_quantity
                
                # Remove or update the bid order
                heapq.heappop(self.bids)
                if best_bid.quantity > 0:
                    heapq.heappush(self.bids, (best_bid_neg_price, best_bid_time, best_bid))
    
    def get_mid_price(self):
        """Calculate the mid price from the order book"""
        if not self.bids or not self.asks:
            return self.last_price
        
        best_bid = -self.bids[0][0] if self.bids else MIN_PRICE
        best_ask = self.asks[0][0] if self.asks else best_bid * 1.01
        
        return (best_bid + best_ask) / 2
    
    def get_spread(self):
        """Calculate the current bid-ask spread"""
        if not self.bids or not self.asks:
            return 0.0
        
        best_bid = -self.bids[0][0]
        best_ask = self.asks[0][0]
        
        return best_ask - best_bid

    def get_book_state(self):
        """Get current order book state for features"""
        bid_levels = {}
        ask_levels = {}
        
        # Process bids
        for neg_price, _, order in self.bids:
            price = -neg_price
            if price in bid_levels:
                bid_levels[price] += order.quantity
            else:
                bid_levels[price] = order.quantity
                
        # Process asks
        for price, _, order in self.asks:
            if price in ask_levels:
                ask_levels[price] += order.quantity
            else:
                ask_levels[price] = order.quantity
                
        return {
            'bid_levels': sorted([(p, q) for p, q in bid_levels.items()], reverse=True),
            'ask_levels': sorted([(p, q) for p, q in ask_levels.items()]),
            'mid_price': self.get_mid_price(),
            'spread': self.get_spread()
        }

class Agent:
    """Base class for all market participants"""
    next_id = 0
    
    def __init__(self, initial_capital=10000.0, initial_inventory=0):
        Agent.next_id += 1
        self.id = Agent.next_id
        self.capital = initial_capital
        self.inventory = initial_inventory
        self.trades = []
        self.orders = []
        self.pnl_history = [(0, initial_capital)]
    
    def update_pnl(self, current_price, timestamp):
        """Update agent's mark-to-market P&L"""
        mtm_value = self.capital + self.inventory * current_price
        self.pnl_history.append((timestamp, mtm_value))
        return mtm_value
    
    def record_trade(self, trade, is_buyer):
        """Record a trade execution for this agent"""
        self.trades.append(trade)
        
        # Update capital and inventory
        if is_buyer:
            self.capital -= trade.price * trade.quantity
            self.inventory += trade.quantity
        else:
            self.capital += trade.price * trade.quantity
            self.inventory -= trade.quantity

class ImprovedInformedTrader(Agent):
    """More realistic informed trader with decay and partial information"""
    def __init__(self, future_price_info, knowledge_horizon=10, order_rate=0.08, 
                 information_decay=0.95, confidence_threshold=0.3, **kwargs):
        super().__init__(**kwargs)
        self.future_price_info = future_price_info
        self.knowledge_horizon = knowledge_horizon
        self.order_rate = order_rate
        self.information_decay = information_decay  # Information decays over time
        self.confidence_threshold = confidence_threshold  # Minimum confidence to trade
        self.type = 'informed'
        self.recent_trades = []  # Track recent performance
        self.information_quality = random.uniform(0.6, 0.9)  # Not perfect information
    
    def generate_order(self, timestamp, current_price):
        """Generate orders with realistic information decay and noise"""
        if random.random() > self.order_rate:
            return None
        
        # Look ahead with decaying information quality
        max_horizon = min(timestamp + self.knowledge_horizon, len(self.future_price_info) - 1)
        if max_horizon <= timestamp:
            return None
        
        # Sample multiple future points and add noise
        future_prices = []
        for t in range(timestamp + 1, max_horizon + 1):
            decay_factor = self.information_decay ** (t - timestamp)
            noise = random.gauss(0, 0.02 * current_price * (1 - decay_factor))
            noisy_price = self.future_price_info[t] + noise
            future_prices.append(noisy_price * decay_factor + current_price * (1 - decay_factor))
        
        if not future_prices:
            return None
        
        # Calculate expected price change with uncertainty
        expected_future_price = np.mean(future_prices)
        price_change = expected_future_price - current_price
        confidence = self.information_quality * self.information_decay ** self.knowledge_horizon
        
        # Only trade if confident enough and signal is strong enough
        if confidence < self.confidence_threshold or abs(price_change) < 0.08:
            return None
        
        # Adaptive position sizing based on confidence and recent performance
        base_size = max(1, int(abs(price_change) * 8))
        
        # Reduce size if recent trades were bad
        if len(self.recent_trades) > 5:
            recent_success_rate = sum(self.recent_trades[-5:]) / 5
            if recent_success_rate < 0.3:
                base_size = max(1, int(base_size * 0.5))
        
        size = min(base_size, 8)  # Cap maximum size
        
        if price_change > 0:
            return Order(self.id, MARKET, BUY, None, size, timestamp)
        else:
            return Order(self.id, MARKET, SELL, None, size, timestamp)

class SmartNoiseTrader(Agent):
    """Noise trader with some basic market awareness"""
    def __init__(self, order_rate=0.05, momentum_factor=0.3, contrarian_factor=0.2, **kwargs):
        super().__init__(**kwargs)
        self.order_rate = order_rate
        self.momentum_factor = momentum_factor
        self.contrarian_factor = contrarian_factor
        self.type = 'uninformed'
        self.last_prices = []
        
    def generate_order(self, timestamp, current_price):
        """Generate orders with simple momentum/contrarian behaviour"""
        if random.random() > self.order_rate:
            return None
        
        # Track recent prices
        self.last_prices.append(current_price)
        if len(self.last_prices) > 10:
            self.last_prices.pop(0)
        
        # Pure random trade (50% of the time)
        if random.random() < 0.5 or len(self.last_prices) < 3:
            side = BUY if random.random() > 0.5 else SELL
            size = random.randint(1, 3)
            return Order(self.id, MARKET, side, None, size, timestamp)
        
        # Simple momentum/contrarian behaviour
        recent_change = (self.last_prices[-1] - self.last_prices[-3]) / self.last_prices[-3]
        
        # Momentum trade
        if random.random() < self.momentum_factor and abs(recent_change) > 0.01:
            side = BUY if recent_change > 0 else SELL
            size = random.randint(1, 4)
            return Order(self.id, MARKET, side, None, size, timestamp)
        
        # Contrarian trade
        elif random.random() < self.contrarian_factor and abs(recent_change) > 0.02:
            side = SELL if recent_change > 0 else BUY
            size = random.randint(1, 3)
            return Order(self.id, MARKET, side, None, size, timestamp)
        
        return None

class EnhancedMarketMaker(Agent):
    """Market maker with more sophisticated risk management and spread adjustment"""
    def __init__(self, base_spread_bps=40, inventory_limit=50, order_size=2, 
                 learning_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.base_spread_bps = base_spread_bps
        self.current_spread_bps = base_spread_bps
        self.inventory_limit = inventory_limit
        self.order_size = order_size
        self.type = 'market_maker'
        
        # Enhanced risk management
        self.learning_rate = learning_rate
        self.volatility_multiplier = 1.0
        self.adverse_selection_penalty = 0.0
        self.inventory_skew_strength = 0.02
        
        # Track performance metrics
        self.trade_pnl_history = []
        self.recent_toxicity_rate = 0.0
        self.spread_history = []
        self.inventory_history = []
        self.timestamp_history = []
        
        # Market microstructure features
        self.last_trade_direction = 0  # +1 for buy, -1 for sell
        self.order_flow_imbalance = 0.0
        self.quote_update_frequency = 0.8  # How often to update quotes

    def calculate_volatility_adjustment(self, recent_prices, window=20):
        """Calculate volatility-based spread adjustment"""
        if len(recent_prices) < window:
            return 1.0
        
        returns = [np.log(recent_prices[i] / recent_prices[i-1]) 
                  for i in range(1, min(len(recent_prices), window))]
        
        if len(returns) < 2:
            return 1.0
        
        volatility = np.std(returns) * np.sqrt(252)  # Annualised volatility
        # Scale volatility to a reasonable multiplier (1.0 to 3.0)
        vol_adjustment = 1.0 + min(volatility * 5, 2.0)
        return vol_adjustment

    def update_adverse_selection_estimate(self, recent_trades, window=50):
        """Estimate adverse selection from recent trade outcomes"""
        if len(recent_trades) < window:
            return
        
        # Look at recent trades involving this market maker
        mm_trades = [t for t in recent_trades[-window:] 
                    if t.buyer_id == self.id or t.seller_id == self.id]
        
        if not mm_trades:
            return
        
        # Calculate how often MM was on the wrong side
        adverse_count = sum(1 for t in mm_trades if t.is_toxic)
        self.recent_toxicity_rate = adverse_count / len(mm_trades)
        
        # Adjust adverse selection penalty
        if self.recent_toxicity_rate > 0.6:
            self.adverse_selection_penalty += self.learning_rate * 0.2
        elif self.recent_toxicity_rate < 0.3:
            self.adverse_selection_penalty = max(0, self.adverse_selection_penalty - self.learning_rate * 0.1)

    def generate_orders(self, timestamp, mid_price, market_context=None):
        """Generate orders with enhanced risk management"""
        
        # Don't always quote - skip sometimes based on market conditions
        if random.random() > self.quote_update_frequency:
            return []
        
        orders = []
        
        # Calculate dynamic spread components
        base_half_spread = mid_price * (self.base_spread_bps / 10000) / 2
        
        # 1. Volatility adjustment
        if market_context and 'recent_prices' in market_context:
            vol_adj = self.calculate_volatility_adjustment(market_context['recent_prices'])
            self.volatility_multiplier = vol_adj
        
        # 2. Adverse selection adjustment
        adverse_adj = 1.0 + self.adverse_selection_penalty
        
        # 3. Inventory-based skewing
        inventory_ratio = self.inventory / self.inventory_limit if self.inventory_limit > 0 else 0
        inventory_skew = inventory_ratio * self.inventory_skew_strength * mid_price
        
        # 4. Order flow imbalance adjustment (simple version)
        flow_adjustment = self.order_flow_imbalance * 0.01 * mid_price
        
        # Calculate final spread
        adjusted_half_spread = (base_half_spread * self.volatility_multiplier * adverse_adj)
        
        # Calculate bid/ask with all adjustments
        bid_price = max(
            mid_price - adjusted_half_spread - inventory_skew - flow_adjustment,
            MIN_PRICE
        )
        ask_price = max(
            mid_price + adjusted_half_spread - inventory_skew + flow_adjustment,
            bid_price * 1.001
        )
        
        # Dynamic order sizing based on risk
        risk_factor = min(self.volatility_multiplier * (1 + self.adverse_selection_penalty), 3.0)
        adjusted_bid_size = max(1, int(self.order_size / risk_factor))
        adjusted_ask_size = max(1, int(self.order_size / risk_factor))
        
        # Inventory constraints
        if abs(self.inventory) > 0.7 * self.inventory_limit:
            if self.inventory > 0:  # Long inventory - encourage selling
                adjusted_bid_size = max(1, int(adjusted_bid_size * 0.3))
                ask_price *= 0.999  # Slightly more aggressive ask
            else:  # Short inventory - encourage buying
                adjusted_ask_size = max(1, int(adjusted_ask_size * 0.3))
                bid_price *= 1.001  # Slightly more aggressive bid
        
        # Record spread for analysis
        current_spread_bps = (ask_price - bid_price) / mid_price * 10000
        self.spread_history.append(current_spread_bps)
        self.inventory_history.append(self.inventory)
        self.timestamp_history.append(timestamp)
        
        # Create orders
        if adjusted_bid_size > 0:
            orders.append(Order(self.id, LIMIT, BUY, bid_price, adjusted_bid_size, timestamp))
        if adjusted_ask_size > 0:
            orders.append(Order(self.id, LIMIT, SELL, ask_price, adjusted_ask_size, timestamp))
        
        return orders

class EnhancedMarketEnvironment:
    """Enhanced market environment with better microstructure simulation"""
    
    def __init__(self, initial_price=INITIAL_PRICE, price_vol=0.01, mean_reversion=0.05):
        self.current_time = 0
        self.order_book = LimitOrderBook()
        self.agents = []
        self.last_price = initial_price
        self.price_vol = price_vol
        self.mean_reversion = mean_reversion
        self.fundamental_price = initial_price
        self.lob_snapshots = []
        self.orders_data = []
        
        # Generate fundamental price process (mean-reverting with jumps)
        self.price_path = self._generate_price_path(TIME_STEPS)
        
        # Initialize statistics and analytics
        self.market_stats = {
            'timestamps': [],
            'prices': [],
            'spreads': [],
            'volumes': []
        }
        
        # Enhanced features
        self.market_impact_model = True
        self.latency_simulation = True
        self.order_arrival_times = []
        
    def _generate_price_path(self, n_steps):
        """Generate a price path for the fundamental value"""
        prices = [self.fundamental_price]
        
        for _ in range(n_steps):
            # Occasionally add jumps
            if random.random() < 0.02:  # 2% chance of a jump
                jump_size = random.uniform(-0.05, 0.05) * prices[-1]
            else:
                jump_size = 0
            
            # Mean-reverting component
            mean_reversion_component = self.mean_reversion * (self.fundamental_price - prices[-1])
            
            # Random component
            random_component = random.gauss(0, self.price_vol * prices[-1])
            
            # New price (with floor to prevent negative prices)
            new_price = max(prices[-1] + mean_reversion_component + random_component + jump_size, MIN_PRICE)
            prices.append(new_price)
            
        return prices
    
    def add_agent(self, agent):
        """Add an agent to the simulation"""
        self.agents.append(agent)
        return agent
    
    def calculate_market_impact(self, order_size, current_depth):
        """Simple linear market impact model"""
        if current_depth == 0:
            return 0.001  # 10 bps impact if no depth
        
        impact_factor = min(order_size / current_depth, 0.1)  # Cap at 10%
        return impact_factor * 0.002  # 2 bps per unit of size/depth ratio
    
    def run_enhanced_simulation(self, n_steps=None):
        """Enhanced simulation with market microstructure features"""
        n_steps = n_steps or TIME_STEPS
        
        for t in range(n_steps):
            self.current_time = t
            mid_price = self.order_book.get_mid_price()
            
            # Prepare market context for market makers
            market_context = {
                'recent_prices': self.market_stats['prices'][-20:] if self.market_stats['prices'] else [mid_price],
                'recent_trades': self.order_book.trades[-50:] if self.order_book.trades else [],
                'current_depth': len(self.order_book.bids) + len(self.order_book.asks)
            }
            
            # Take LOB snapshot
            self.lob_snapshots.append(self.order_book.save_snapshot(t))
            
            # Update agent PnLs
            for agent in self.agents:
                agent.update_pnl(mid_price, t)
            
            # Market makers place orders with enhanced context
            for agent in self.agents:
                if isinstance(agent, EnhancedMarketMaker):
                    # Update adverse selection estimates
                    agent.update_adverse_selection_estimate(market_context['recent_trades'])
                    
                    orders = agent.generate_orders(t, mid_price, market_context)
                    for order in orders:
                        self.record_order_data(order, t)
                        self.order_book.add_limit_order(order, t)
            
            # Other agents place orders
            for agent in self.agents:
                if not isinstance(agent, EnhancedMarketMaker):
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

    def enhanced_post_process_trades(self, window_size=10, price_move_threshold=0.0015):
        """Enhanced toxicity labelling with multiple criteria"""
        if not self.order_book.trades:
            return
        
        timestamps = [t for t, _ in self.order_book.price_history]
        prices = [p for _, p in self.order_book.price_history]
        
        for trade in self.order_book.trades:
            try:
                trade_idx = timestamps.index(trade.timestamp)
                
                # Multiple toxicity criteria
                is_toxic = False
                
                # 1. Future price move criterion (existing)
                if trade_idx + window_size < len(prices):
                    future_price = prices[trade_idx + window_size]
                    price_move = (future_price - prices[trade_idx]) / prices[trade_idx]
                    
                    if abs(price_move) > price_move_threshold:
                        buyer_agent = next((a for a in self.agents if a.id == trade.buyer_id), None)
                        seller_agent = next((a for a in self.agents if a.id == trade.seller_id), None)
                        
                        if buyer_agent and buyer_agent.type == 'informed' and price_move > 0:
                            is_toxic = True
                        elif seller_agent and seller_agent.type == 'informed' and price_move < 0:
                            is_toxic = True
                
                # 2. Large trade size criterion
                avg_trade_size = np.mean([t.quantity for t in self.order_book.trades])
                if trade.quantity > 2 * avg_trade_size:
                    buyer_agent = next((a for a in self.agents if a.id == trade.buyer_id), None)
                    seller_agent = next((a for a in self.agents if a.id == trade.seller_id), None)
                    if (buyer_agent and buyer_agent.type == 'informed') or \
                       (seller_agent and seller_agent.type == 'informed'):
                        is_toxic = True
                
                trade.is_toxic = is_toxic
                
            except:
                continue

    def record_order_data(self, order, timestamp):
        """Record data about each order for ML training with enhanced features"""
        # Find the agent type
        agent_type = next((a.type for a in self.agents if a.id == order.agent_id), 'unknown')
        
        # Get current LOB state
        mid_price = self.order_book.get_mid_price()
        spread = self.order_book.get_spread()
        
        # Enhanced features for ML
        recent_prices = self.market_stats['prices'][-10:] if self.market_stats['prices'] else [mid_price]
        
        # Calculate volatility
        if len(recent_prices) > 1:
            returns = [np.log(recent_prices[i] / recent_prices[i-1]) for i in range(1, len(recent_prices))]
            volatility = np.std(returns) if len(returns) > 1 else 0
        else:
            volatility = 0
        
        # Calculate momentum
        if len(recent_prices) >= 3:
            momentum = (recent_prices[-1] - recent_prices[-3]) / recent_prices[-3]
        else:
            momentum = 0
        
        # Order book imbalance
        imbalance = self.order_book.calculate_imbalance()
        
        # Time since last trade
        recent_trades = [t for t in self.order_book.trades if t.timestamp >= timestamp - 10]
        time_since_last_trade = 0 if recent_trades else min(10, timestamp)
        
        # Calculate features
        if order.type == LIMIT and order.price is not None:
            distance_from_mid = (order.price - mid_price) / mid_price
            is_aggressive = (order.side == BUY and order.price >= mid_price) or \
                            (order.side == SELL and order.price <= mid_price)
        else:
            distance_from_mid = np.nan
            is_aggressive = True
        
        # Record enhanced order data
        order_data = {
            'timestamp': timestamp,
            'order_id': order.id,
            'agent_id': order.agent_id,
            'agent_type': agent_type,
            'order_type': 'LIMIT' if order.type == LIMIT else 'MARKET',
            'side': 'BUY' if order.side == BUY else 'SELL',
            'price': order.price if order.price is not None else np.nan,
            'quantity': order.quantity,
            'mid_price': mid_price,
            'spread': spread,
            'distance_from_mid': distance_from_mid,
            'is_aggressive': is_aggressive,
            'volatility': volatility,
            'momentum': momentum,
            'order_book_imbalance': imbalance,
            'time_since_last_trade': time_since_last_trade,
            'resulted_in_trade': False,
            'was_toxic': False
        }
        
        self.orders_data.append(order_data)

    def update_order_toxicity(self):
        """Update order data with trade outcomes and toxicity"""
        # Create mappings from order IDs to trades
        order_to_trade = {}
        for trade in self.order_book.trades:
            buy_order = next((o for o in self.order_book.order_history if o.id == trade.buy_order_id), None)
            sell_order = next((o for o in self.order_book.order_history if o.id == trade.sell_order_id), None)
            
            if buy_order:
                order_to_trade[buy_order.id] = trade
            if sell_order:
                order_to_trade[sell_order.id] = trade
        
        # Update order data
        for i, order_data in enumerate(self.orders_data):
            order_id = order_data['order_id']
            if order_id in order_to_trade:
                trade = order_to_trade[order_id]
                self.orders_data[i]['resulted_in_trade'] = True
                self.orders_data[i]['was_toxic'] = trade.is_toxic
                self.orders_data[i]['trade_price'] = trade.price

    def save_data_to_csv(self, output_dir="enhanced_market_data"):
        """Save all collected data to CSV files for ML training"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Update order toxicity information
        self.update_order_toxicity()
        
        # Save LOB snapshots
        lob_df = pd.DataFrame(self.lob_snapshots)
        lob_df.to_csv(f"{output_dir}/lob_snapshots_{timestamp}.csv", index=False)
        
        # Save enhanced order data
        orders_df = pd.DataFrame(self.orders_data)
        orders_df.to_csv(f"{output_dir}/orders_{timestamp}.csv", index=False)
        
        # Save trades data
        trades_df = self.get_trades_dataframe()
        trades_df.to_csv(f"{output_dir}/trades_{timestamp}.csv", index=False)
        
        # Save market data
        market_df = self.get_market_dataframe()
        market_df.to_csv(f"{output_dir}/market_stats_{timestamp}.csv", index=False)
        
        # Save price path
        price_df = pd.DataFrame({
            'timestamp': range(len(self.price_path)),
            'fundamental_price': self.price_path
        })
        price_df.to_csv(f"{output_dir}/price_path_{timestamp}.csv", index=False)
        
        print(f"Enhanced data saved to {output_dir}/ with timestamp {timestamp}")
        return timestamp

    def get_trades_dataframe(self):
        """Convert trades to a pandas DataFrame"""
        trades_data = []
        
        for trade in self.order_book.trades:
            trades_data.append({
                'timestamp': trade.timestamp,
                'price': trade.price,
                'quantity': trade.quantity,
                'buyer_id': trade.buyer_id,
                'seller_id': trade.seller_id,
                'is_toxic': trade.is_toxic,
                'buyer_type': next((a.type for a in self.agents if a.id == trade.buyer_id), 'unknown'),
                'seller_type': next((a.type for a in self.agents if a.id == trade.seller_id), 'unknown')
            })
        
        return pd.DataFrame(trades_data)
    
    def get_market_dataframe(self):
        """Convert market stats to a pandas DataFrame"""
        return pd.DataFrame({
            'timestamp': self.market_stats['timestamps'],
            'price': self.market_stats['prices'],
            'spread': self.market_stats['spreads'],
            'volume': self.market_stats['volumes']
        })

    def plot_enhanced_results(self, save_dir="enhanced_plots"):
        """Plot enhanced simulation results"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        trades_df = self.get_trades_dataframe()
        market_df = self.get_market_dataframe()
        
        # Create comprehensive plot
        fig, axes = plt.subplots(4, 2, figsize=(18, 20))
        fig.suptitle('Enhanced Market Simulation Results', fontsize=16, fontweight='bold')
        
        # 1. Price evolution with toxic trades
        axes[0,0].plot(market_df['timestamp'], market_df['price'], 'b-', alpha=0.7, label='Mid Price')
        if not trades_df.empty:
            normal_trades = trades_df[trades_df['is_toxic'] == False]
            toxic_trades = trades_df[trades_df['is_toxic'] == True]
            axes[0,0].scatter(normal_trades['timestamp'], normal_trades['price'], 
                            c='green', s=20, alpha=0.6, label='Normal Trades')
            axes[0,0].scatter(toxic_trades['timestamp'], toxic_trades['price'], 
                            c='red', s=30, marker='x', alpha=0.8, label='Toxic Trades')
        axes[0,0].set_title('Price Evolution with Trade Toxicity')
        axes[0,0].set_ylabel('Price')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Enhanced market maker spread dynamics
        mm = next((agent for agent in self.agents if isinstance(agent, EnhancedMarketMaker)), None)
        if mm:
            axes[0,1].plot(mm.timestamp_history, mm.spread_history, 'purple', linewidth=2)
            axes[0,1].axhline(y=mm.base_spread_bps, color='red', linestyle='--', 
                            label=f'Base Spread ({mm.base_spread_bps} bps)')
            axes[0,1].set_title('Enhanced Market Maker Spread Adjustment')
            axes[0,1].set_ylabel('Spread (bps)')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Toxicity rate over time
        if not trades_df.empty:
            window = 50
            rolling_toxicity = []
            timestamps = []
            
            for i in range(window, len(trades_df)):
                window_trades = trades_df.iloc[i-window:i]
                toxicity_rate = window_trades['is_toxic'].mean() * 100
                rolling_toxicity.append(toxicity_rate)
                timestamps.append(window_trades['timestamp'].iloc[-1])
            
            axes[1,0].plot(timestamps, rolling_toxicity, 'red', linewidth=2)
            axes[1,0].set_title(f'Rolling Toxicity Rate (Window: {window} trades)')
            axes[1,0].set_ylabel('Toxicity Rate (%)')
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Market maker inventory and adverse selection
        if mm:
            ax_inv = axes[1,1]
            ax_tox = ax_inv.twinx()
            
            line1 = ax_inv.plot(mm.timestamp_history, mm.inventory_history, 'green', 
                              linewidth=2, label='Inventory')
            ax_inv.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax_inv.set_ylabel('Inventory', color='green')
            ax_inv.tick_params(axis='y', labelcolor='green')
            
            # Plot adverse selection penalty
            toxicity_history = [getattr(mm, 'recent_toxicity_rate', 0)] * len(mm.timestamp_history)
            line2 = ax_tox.plot(mm.timestamp_history, toxicity_history, 'red', 
                              linewidth=2, label='Recent Toxicity Rate')
            ax_tox.set_ylabel('Toxicity Rate', color='red')
            ax_tox.tick_params(axis='y', labelcolor='red')
            
            axes[1,1].set_title('Market Maker Inventory vs Adverse Selection')
            axes[1,1].grid(True, alpha=0.3)
        
        # 5. Agent performance comparison
        agent_returns = {}
        for agent in self.agents:
            if len(agent.pnl_history) > 1:
                initial_pnl = agent.pnl_history[0][1]
                final_pnl = agent.pnl_history[-1][1]
                return_pct = (final_pnl / initial_pnl - 1) * 100
                
                if agent.type not in agent_returns:
                    agent_returns[agent.type] = []
                agent_returns[agent.type].append(return_pct)
        
        if agent_returns:
            agent_types = list(agent_returns.keys())
            agent_means = [np.mean(agent_returns[t]) for t in agent_types]
            agent_stds = [np.std(agent_returns[t]) if len(agent_returns[t]) > 1 else 0 for t in agent_types]
            
            x_pos = range(len(agent_types))
            colors = ['green', 'red', 'blue'][:len(agent_types)]
            axes[2,0].bar(x_pos, agent_means, yerr=agent_stds, capsize=5, color=colors, alpha=0.7)
            axes[2,0].set_title('Enhanced Agent Performance')
            axes[2,0].set_xlabel('Agent Type')
            axes[2,0].set_ylabel('Return (%)')
            axes[2,0].set_xticks(x_pos)
            axes[2,0].set_xticklabels([t.replace('_', ' ').title() for t in agent_types])
            axes[2,0].grid(True, alpha=0.3)
            axes[2,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 6. Trade size distribution by toxicity
        if not trades_df.empty:
            toxic_sizes = trades_df[trades_df['is_toxic'] == True]['quantity']
            normal_sizes = trades_df[trades_df['is_toxic'] == False]['quantity']
            
            axes[2,1].hist(normal_sizes, bins=20, alpha=0.6, label='Normal Trades', color='green')
            axes[2,1].hist(toxic_sizes, bins=20, alpha=0.6, label='Toxic Trades', color='red')
            axes[2,1].set_title('Trade Size Distribution by Toxicity')
            axes[2,1].set_xlabel('Trade Size')
            axes[2,1].set_ylabel('Frequency')
            axes[2,1].legend()
            axes[2,1].grid(True, alpha=0.3)
        
        # 7. Order book dynamics
        axes[3,0].plot(market_df['timestamp'], market_df['spread'], 'purple', alpha=0.7)
        axes[3,0].set_title('Bid-Ask Spread Evolution')
        axes[3,0].set_ylabel('Spread')
        axes[3,0].set_xlabel('Time')
        axes[3,0].grid(True, alpha=0.3)
        
        # 8. Volume profile
        axes[3,1].plot(market_df['timestamp'], market_df['volume'], 'orange', alpha=0.7)
        axes[3,1].fill_between(market_df['timestamp'], 0, market_df['volume'], alpha=0.3, color='orange')
        axes[3,1].set_title('Volume Profile')
        axes[3,1].set_ylabel('Volume')
        axes[3,1].set_xlabel('Time')
        axes[3,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_dir}/enhanced_simulation_results_{timestamp}.png", 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

def run_enhanced_simulation(save_data=True, output_dir="enhanced_market_data"):
    """Run enhanced simulation optimised for ML training"""
    
    # Enhanced market environment
    market = EnhancedMarketEnvironment(
        initial_price=INITIAL_PRICE, 
        price_vol=0.006,  # Moderate volatility
        mean_reversion=0.01
    )
    
    print("Running ENHANCED simulation for ML training")
    
    # Agent allocation: more realistic proportions
    n_market_makers = 1
    n_informed = 3  # 20% informed
    n_uninformed = NUM_AGENTS - n_market_makers - n_informed
    
    print(f"Enhanced Agent allocation:")
    print(f"  Market Makers: {n_market_makers} ({n_market_makers/NUM_AGENTS*100:.1f}%)")
    print(f"  Informed Traders: {n_informed} ({n_informed/NUM_AGENTS*100:.1f}%)")
    print(f"  Uninformed Traders: {n_uninformed} ({n_uninformed/NUM_AGENTS*100:.1f}%)")
    
    # Add enhanced market maker
    market.add_agent(EnhancedMarketMaker(
        base_spread_bps=50,
        inventory_limit=40,
        order_size=2,
        learning_rate=0.1,
        initial_capital=25000
    ))
    
    # Add improved informed traders
    for i in range(n_informed):
        market.add_agent(ImprovedInformedTrader(
            future_price_info=market.price_path,
            knowledge_horizon=random.randint(4, 8),
            order_rate=0.06 + random.uniform(0, 0.04),
            information_decay=random.uniform(0.92, 0.98),
            confidence_threshold=random.uniform(0.25, 0.4),
            initial_capital=8000 + random.randint(-1000, 2000)
        ))
    
    # Add smart noise traders
    for i in range(n_uninformed):
        market.add_agent(SmartNoiseTrader(
            order_rate=0.04 + random.uniform(0, 0.06),
            momentum_factor=random.uniform(0.2, 0.4),
            contrarian_factor=random.uniform(0.1, 0.3),
            initial_capital=3000 + random.randint(-500, 1500)
        ))
    
    print(f"\nRunning enhanced simulation with {NUM_AGENTS} agents for {TIME_STEPS} time steps...")
    market.run_enhanced_simulation()
    
    if save_data:
        timestamp = market.save_data_to_csv(output_dir)
        print(f"Enhanced data saved with timestamp {timestamp}")
    
    # Plot results
    market.plot_enhanced_results()
    
    # Get comprehensive statistics
    trades_df = market.get_trades_dataframe()
    
    total_trades = len(trades_df)
    toxic_trades = trades_df['is_toxic'].sum() if not trades_df.empty else 0
    toxic_pct = toxic_trades / total_trades * 100 if total_trades > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"ENHANCED SIMULATION RESULTS")
    print(f"{'='*60}")
    print(f"Total trades: {total_trades}")
    print(f"Toxic trades: {toxic_trades} ({toxic_pct:.2f}%)")
    
    # Enhanced toxicity analysis
    if not trades_df.empty and toxic_trades > 0:
        toxic_df = trades_df[trades_df['is_toxic'] == True]
        
        print(f"\nENHANCED TOXICITY ANALYSIS:")
        
        # Toxicity by agent type
        for agent_type in ['informed', 'uninformed', 'market_maker']:
            agent_trades = trades_df[
                (trades_df['buyer_type'] == agent_type) | 
                (trades_df['seller_type'] == agent_type)
            ]
            if len(agent_trades) > 0:
                agent_toxic_rate = agent_trades['is_toxic'].mean() * 100
                print(f"  {agent_type.capitalize()} toxicity rate: {agent_toxic_rate:.1f}%")
        
        # Trade size analysis
        avg_trade_size = trades_df['quantity'].mean()
        large_trades = trades_df[trades_df['quantity'] > avg_trade_size * 1.5]
        if len(large_trades) > 0:
            large_trade_toxic_rate = large_trades['is_toxic'].mean() * 100
            print(f"  Large trade toxicity rate: {large_trade_toxic_rate:.1f}%")
    
    # Market maker performance analysis
    mm = next((agent for agent in market.agents if isinstance(agent, EnhancedMarketMaker)), None)
    if mm:
        initial_capital = mm.pnl_history[0][1]
        final_capital = mm.pnl_history[-1][1]
        mm_return = (final_capital / initial_capital - 1) * 100
        
        print(f"\nENHANCED MARKET MAKER ANALYSIS:")
        print(f"  Return: {mm_return:.2f}%")
        print(f"  Base spread: {mm.base_spread_bps} bps")
        print(f"  Average spread: {np.mean(mm.spread_history):.1f} bps")
        print(f"  Volatility multiplier range: {min(mm.volatility_multiplier, 1.0):.2f} - {mm.volatility_multiplier:.2f}")
        print(f"  Adverse selection penalty: {mm.adverse_selection_penalty:.3f}")
        print(f"  Recent toxicity rate: {mm.recent_toxicity_rate:.1f}%")
        print(f"  Final inventory: {mm.inventory}")
    
    # Data quality for ML
    orders_df = pd.DataFrame(market.orders_data)
    if not orders_df.empty:
        print(f"\nML TRAINING DATA QUALITY:")
        print(f"  Total orders recorded: {len(orders_df)}")
        print(f"  Orders that resulted in trades: {orders_df['resulted_in_trade'].sum()}")
        print(f"  Toxic order rate: {orders_df['was_toxic'].mean()*100:.1f}%")
        print(f"  Features available: {list(orders_df.columns)}")
    
    print(f"\n{'='*60}")
    print(f"READY FOR ML IMPLEMENTATION")
    print(f"{'='*60}")
    print("Data files contain enhanced features for:")
    print("• Volatility-adjusted spread modeling")
    print("• Adverse selection detection")
    print("• Order flow analysis")
    print("• Market microstructure patterns")
    print("• Multi-criteria toxicity labeling")
    
    return market

if __name__ == "__main__":
    # Run enhanced simulation
    market = run_enhanced_simulation()
    
    print("\n" + "="*60)
    print("NEXT STEPS FOR ML IMPLEMENTATION:")
    print("="*60)
    print("1. Load the enhanced order data CSV")
    print("2. Train ML model on features: volatility, momentum, imbalance, etc.")
    print("3. Implement real-time toxicity prediction")
    print("4. Replace static spread adjustment with ML-based dynamic adjustment")
    print("5. Backtest enhanced market maker vs baseline")
    print("="*60)