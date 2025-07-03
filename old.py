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
TIME_STEPS = 100
NUM_AGENTS = 100

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

class InformedTrader(Agent):
    """Trader with some information about future price moves"""
    def __init__(self, future_price_info, knowledge_horizon=10, order_rate=0.15, **kwargs):
        super().__init__(**kwargs)
        self.future_price_info = future_price_info  # Time series of future prices
        self.knowledge_horizon = knowledge_horizon  # How far ahead they can see
        self.order_rate = order_rate  # Increased rate for more toxic orders
        self.type = 'informed'
    
    def generate_order(self, timestamp, current_price):
        """Generate orders based on future price knowledge - more aggressively for ML training"""
        # Trade more frequently to generate toxic examples
        if random.random() > self.order_rate:
            return None
            
        # Look ahead to see if prices will rise or fall
        future_horizon = min(timestamp + self.knowledge_horizon, len(self.future_price_info) - 1)
        if future_horizon <= timestamp:
            return None
            
        future_price = self.future_price_info[future_horizon]
        price_change = future_price - current_price
        
        # Lower threshold to trade on smaller signals (generates more toxic trades)
        if abs(price_change) < 0.05:  # Much lower threshold
            return None
            
        # Larger position sizes for more impact
        size = max(1, int(abs(price_change) * 15))  # Increased multiplier
        
        if price_change > 0:
            # Buy if price will rise
            return Order(self.id, MARKET, BUY, None, size, timestamp)
        else:
            # Sell if price will fall
            return Order(self.id, MARKET, SELL, None, size, timestamp)

class UninformedTrader(Agent):
    """Trader who trades randomly without information"""
    def __init__(self, order_rate=0.12, **kwargs):
        super().__init__(**kwargs)
        self.order_rate = order_rate  # Increased slightly to compensate for fewer informed traders
        self.type = 'uninformed'
    
    def generate_order(self, timestamp, current_price):
        """Generate random orders"""
        if random.random() > self.order_rate:
            return None
            
        # Decide on order parameters
        side = BUY if random.random() > 0.5 else SELL
        order_type = MARKET if random.random() > 0.7 else LIMIT
        size = random.randint(1, 5)
        
        if order_type == LIMIT:
            # Set limit price around current price
            price_offset = random.uniform(-0.5, 0.5)
            if side == BUY:
                price = max(current_price - price_offset, MIN_PRICE)
            else:
                price = max(current_price + price_offset, MIN_PRICE)
            return Order(self.id, order_type, side, price, size, timestamp)
        else:
            return Order(self.id, order_type, side, None, size, timestamp)

class MarketMaker(Agent):
    """Agent providing liquidity by placing bid and ask orders"""
    def __init__(self, spread_bps=20, inventory_limit=100, order_size=1, **kwargs):
        super().__init__(**kwargs)
        self.base_spread_bps = spread_bps
        self.current_spread_bps = spread_bps
        self.inventory_limit = inventory_limit
        self.inventory_risk_aversion = 0.005
        self.order_size = order_size
        self.type = 'market_maker'
        self.skew_factor = 0.0
        
        # Track spread history
        self.spread_history = []
        self.inventory_history = []
        self.timestamp_history = []

    def generate_orders(self, timestamp, mid_price):
        """Generate limit orders around the mid price"""
        orders = []
        
        # Adjust spread based on inventory
        inventory_skew = self.inventory / self.inventory_limit if self.inventory_limit != 0 else 0
        self.skew_factor = inventory_skew * self.inventory_risk_aversion

        # Calculate half-spread in absolute terms
        half_spread = mid_price * (self.current_spread_bps / 10000) / 2
        
        # Record current spread settings
        self.spread_history.append(self.current_spread_bps)
        self.inventory_history.append(self.inventory)
        self.timestamp_history.append(timestamp)
        
        # Adjust bid/ask levels based on inventory skew
        bid_price = max(mid_price - half_spread - (mid_price * self.skew_factor), MIN_PRICE)
        ask_price = max(mid_price + half_spread - (mid_price * self.skew_factor), bid_price * 1.001)
        
        # Limit order size based on inventory constraints
        bid_size = self.order_size
        ask_size = self.order_size
        
        # Reduce size if approaching inventory limits
        if self.inventory > 0.8 * self.inventory_limit:
            bid_size = max(1, int(bid_size * (1 - self.inventory / self.inventory_limit)))
        elif self.inventory < -0.8 * self.inventory_limit:
            ask_size = max(1, int(ask_size * (1 + self.inventory / self.inventory_limit)))
        
        # Create orders
        if bid_size > 0:
            orders.append(Order(self.id, LIMIT, BUY, bid_price, bid_size, timestamp))
        if ask_size > 0:
            orders.append(Order(self.id, LIMIT, SELL, ask_price, ask_size, timestamp))
        
        return orders

    def adjust_spread(self, toxicity_score):
        """Adjust spread based on observed toxicity"""
        # For the simulation, we'll just assume a simple adjustment
        self.current_spread_bps = self.base_spread_bps * (1 + toxicity_score)

class MarketEnvironment:
    """Main class for simulating the market"""
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
    
    def plot_order_flow_by_agent_type(self, window_size=20, save_dir="plots"):
        """Plot order flow by agent type over time with stacked areas"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Get trades data
        trades_df = self.get_trades_dataframe()
        if trades_df.empty:
            print("No trades to plot")
            return
            
        # Create time windows for aggregation
        max_time = trades_df['timestamp'].max()
        time_windows = range(0, max_time + window_size, window_size)
        
        # Initialize data structures
        flow_data = {
            'time_window': [],
            'informed_volume': [],
            'uninformed_volume': [],
            'market_maker_volume': [],
            'informed_trades': [],
            'uninformed_trades': [],
            'market_maker_trades': [],
            'toxic_rate': []
        }
        
        # Aggregate data by time windows
        for i, window_start in enumerate(time_windows[:-1]):
            window_end = time_windows[i + 1]
            window_trades = trades_df[
                (trades_df['timestamp'] >= window_start) & 
                (trades_df['timestamp'] < window_end)
            ]
            
            if len(window_trades) == 0:
                # Fill with zeros if no trades in window
                flow_data['time_window'].append(window_start + window_size/2)
                for key in ['informed_volume', 'uninformed_volume', 'market_maker_volume',
                           'informed_trades', 'uninformed_trades', 'market_maker_trades']:
                    flow_data[key].append(0)
                flow_data['toxic_rate'].append(0)
                continue
                
            # Count volume and trades by agent type (considering both buyers and sellers)
            informed_vol = 0
            uninformed_vol = 0
            mm_vol = 0
            informed_trades = 0
            uninformed_trades = 0
            mm_trades = 0
            
            for _, trade in window_trades.iterrows():
                # Count volume for each side
                if trade['buyer_type'] == 'informed':
                    informed_vol += trade['quantity']
                    informed_trades += 0.5  # Half trade attribution
                elif trade['buyer_type'] == 'uninformed':
                    uninformed_vol += trade['quantity']
                    uninformed_trades += 0.5
                elif trade['buyer_type'] == 'market_maker':
                    mm_vol += trade['quantity']
                    mm_trades += 0.5
                    
                if trade['seller_type'] == 'informed':
                    informed_vol += trade['quantity']
                    informed_trades += 0.5
                elif trade['seller_type'] == 'uninformed':
                    uninformed_vol += trade['quantity']
                    uninformed_trades += 0.5
                elif trade['seller_type'] == 'market_maker':
                    mm_vol += trade['quantity']
                    mm_trades += 0.5
            
            # Store aggregated data
            flow_data['time_window'].append(window_start + window_size/2)
            flow_data['informed_volume'].append(informed_vol)
            flow_data['uninformed_volume'].append(uninformed_vol)
            flow_data['market_maker_volume'].append(mm_vol)
            flow_data['informed_trades'].append(informed_trades)
            flow_data['uninformed_trades'].append(uninformed_trades)
            flow_data['market_maker_trades'].append(mm_trades)
            
            # Calculate toxic rate for this window
            toxic_count = window_trades['is_toxic'].sum()
            total_count = len(window_trades)
            toxic_rate = toxic_count / total_count * 100 if total_count > 0 else 0
            flow_data['toxic_rate'].append(toxic_rate)
        
        # Convert to DataFrame for easier plotting
        flow_df = pd.DataFrame(flow_data)
        
        # Create the plot
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        # 1. Order Flow by Volume (Stacked Area)
        total_volume = (flow_df['informed_volume'] + 
                       flow_df['uninformed_volume'] + 
                       flow_df['market_maker_volume'])
        
        # Calculate proportions
        informed_prop = flow_df['informed_volume'] / total_volume * 100
        uninformed_prop = flow_df['uninformed_volume'] / total_volume * 100
        mm_prop = flow_df['market_maker_volume'] / total_volume * 100
        
        # Replace NaN with 0
        informed_prop = informed_prop.fillna(0)
        uninformed_prop = uninformed_prop.fillna(0)
        mm_prop = mm_prop.fillna(0)
        
        axes[0].fill_between(flow_df['time_window'], 0, informed_prop, 
                            alpha=0.7, color='red', label='Informed Traders')
        axes[0].fill_between(flow_df['time_window'], informed_prop, 
                            informed_prop + uninformed_prop,
                            alpha=0.7, color='blue', label='Uninformed Traders')
        axes[0].fill_between(flow_df['time_window'], 
                            informed_prop + uninformed_prop,
                            informed_prop + uninformed_prop + mm_prop,
                            alpha=0.7, color='green', label='Market Makers')
        
        axes[0].set_ylabel('Volume Proportion (%)')
        axes[0].set_title('Order Flow by Agent Type (Volume-Weighted)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 100)
        
        # 2. Trade Count Proportions (Stacked Area)
        total_trades = (flow_df['informed_trades'] + 
                       flow_df['uninformed_trades'] + 
                       flow_df['market_maker_trades'])
        
        informed_trade_prop = flow_df['informed_trades'] / total_trades * 100
        uninformed_trade_prop = flow_df['uninformed_trades'] / total_trades * 100
        mm_trade_prop = flow_df['market_maker_trades'] / total_trades * 100
        
        # Replace NaN with 0
        informed_trade_prop = informed_trade_prop.fillna(0)
        uninformed_trade_prop = uninformed_trade_prop.fillna(0)
        mm_trade_prop = mm_trade_prop.fillna(0)
        
        axes[1].fill_between(flow_df['time_window'], 0, informed_trade_prop,
                            alpha=0.7, color='red', label='Informed Traders')
        axes[1].fill_between(flow_df['time_window'], informed_trade_prop,
                            informed_trade_prop + uninformed_trade_prop,
                            alpha=0.7, color='blue', label='Uninformed Traders')
        axes[1].fill_between(flow_df['time_window'],
                            informed_trade_prop + uninformed_trade_prop,
                            informed_trade_prop + uninformed_trade_prop + mm_trade_prop,
                            alpha=0.7, color='green', label='Market Makers')
        
        axes[1].set_ylabel('Trade Count Proportion (%)')
        axes[1].set_title('Order Flow by Agent Type (Trade Count)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 100)
        
        # 3. Toxicity Rate Over Time
        axes[2].plot(flow_df['time_window'], flow_df['toxic_rate'], 
                    'r-', linewidth=2, label='Toxic Trade Rate')
        axes[2].fill_between(flow_df['time_window'], 0, flow_df['toxic_rate'],
                            alpha=0.3, color='red')
        axes[2].set_ylabel('Toxicity Rate (%)')
        axes[2].set_xlabel('Time')
        axes[2].set_title('Toxic Trade Rate Over Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_dir}/order_flow_by_agent_type_{timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print(f"\nORDER FLOW ANALYSIS:")
        print(f"Average Volume Proportions:")
        print(f"  Informed: {informed_prop.mean():.1f}% ± {informed_prop.std():.1f}%")
        print(f"  Uninformed: {uninformed_prop.mean():.1f}% ± {uninformed_prop.std():.1f}%")
        print(f"  Market Makers: {mm_prop.mean():.1f}% ± {mm_prop.std():.1f}%")
        print(f"Average Toxicity Rate: {flow_df['toxic_rate'].mean():.1f}% ± {flow_df['toxic_rate'].std():.1f}%")
        print(f"Peak Toxicity Rate: {flow_df['toxic_rate'].max():.1f}%")
        
        return flow_df

    def plot_simulation_results(self, save_dir="plots"):
        """Plot key statistics and results from the simulation"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Get data frames
        trades_df = self.get_trades_dataframe()
        market_df = self.get_market_dataframe()
        
        # Create plots
        fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
        
        # Plot prices
        axes[0].plot(market_df['timestamp'], market_df['price'], label='Mid Price')
        axes[0].set_title('Price Evolution')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot trades with toxic trades highlighted
        if not trades_df.empty:
            axes[0].scatter(
                trades_df[trades_df['is_toxic'] == False]['timestamp'],
                trades_df[trades_df['is_toxic'] == False]['price'], 
                color='blue', s=10, alpha=0.5, label='Normal Trade'
            )
            axes[0].scatter(
                trades_df[trades_df['is_toxic'] == True]['timestamp'],
                trades_df[trades_df['is_toxic'] == True]['price'], 
                color='red', s=30, marker='x', label='Toxic Trade'
            )
            axes[0].legend()
    
        # Plot spread
        axes[1].plot(market_df['timestamp'], market_df['spread'], label='Bid-Ask Spread')
        axes[1].set_title('Bid-Ask Spread')
        axes[1].set_ylabel('Spread')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot PnL for each agent type - FIXED to start from 0
        agent_types = set(agent.type for agent in self.agents)
        for agent_type in agent_types:
            agents_of_type = [a for a in self.agents if a.type == agent_type]
            if not agents_of_type:
                continue
                
            # Calculate average PnL for this agent type, starting from 0
            pnl_by_time = defaultdict(list)
            
            for agent in agents_of_type:
                initial_value = agent.pnl_history[0][1]  # Starting portfolio value
                for t, portfolio_value in agent.pnl_history:
                    # Calculate P&L as difference from initial value
                    pnl = portfolio_value - initial_value
                    pnl_by_time[t].append(pnl)
            
            # Calculate average P&L across agents of this type
            avg_pnl = [(t, sum(pnls)/len(pnls)) for t, pnls in sorted(pnl_by_time.items())]
            
            # Plot the P&L starting from 0
            times = [t for t, _ in avg_pnl]
            pnls = [p for _, p in avg_pnl]
            axes[2].plot(times, pnls, label=f'{agent_type.capitalize()}', linewidth=2)
        
        axes[2].set_title('Agent P&L by Type (Starting from Zero)')
        axes[2].set_ylabel('P&L ($)')
        axes[2].set_xlabel('Time')
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)  # Add zero line
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_dir}/simulation_results_{timestamp}.png", 
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_market_maker_analysis(self, save_dir="plots"):
        """Plot detailed market maker behavior analysis"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Find the market maker
        market_maker = next((agent for agent in self.agents if isinstance(agent, MarketMaker)), None)
        
        if not market_maker:
            print("No market maker found in simulation")
            return
        
        # Get data frames
        trades_df = self.get_trades_dataframe()
        market_df = self.get_market_dataframe()
        
        # Create subplot figure
        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
        
        # 1. Market Maker Spread Settings
        axes[0].plot(market_maker.timestamp_history, market_maker.spread_history, 
                    'b-', linewidth=2, label='MM Quoted Spread (bps)')
        axes[0].axhline(y=market_maker.base_spread_bps, color='r', linestyle='--', 
                        label=f'Base Spread ({market_maker.base_spread_bps} bps)')
        axes[0].set_ylabel('Spread (bps)')
        axes[0].set_title('Market Maker Spread Settings Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Market Maker Inventory
        axes[1].plot(market_maker.timestamp_history, market_maker.inventory_history, 
                    'g-', linewidth=2, label='MM Inventory')
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.5)
        axes[1].axhline(y=market_maker.inventory_limit, color='r', linestyle='--', alpha=0.7, 
                        label=f'Inventory Limit (±{market_maker.inventory_limit})')
        axes[1].axhline(y=-market_maker.inventory_limit, color='r', linestyle='--', alpha=0.7)
        axes[1].set_ylabel('Inventory')
        axes[1].set_title('Market Maker Inventory Position')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Observed Bid-Ask Spread (from order book)
        axes[2].plot(market_df['timestamp'], market_df['spread'], 
                    'purple', linewidth=1, alpha=0.7, label='Observed LOB Spread')
        axes[2].set_ylabel('Price')
        axes[2].set_title('Observed Limit Order Book Spread')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. Market Maker PnL - FIXED to start from 0
        mm_pnl_timestamps = [t for t, _ in market_maker.pnl_history]
        mm_pnl_values = [pnl for _, pnl in market_maker.pnl_history]
        
        # Calculate P&L relative to starting value
        initial_value = mm_pnl_values[0]
        mm_pnl_relative = [pnl - initial_value for pnl in mm_pnl_values]
        
        axes[3].plot(mm_pnl_timestamps, mm_pnl_relative, 
                    'orange', linewidth=2, label='MM P&L')
        axes[3].axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=1)
        axes[3].set_ylabel('P&L ($)')
        axes[3].set_xlabel('Time')
        axes[3].set_title('Market Maker Profit & Loss (Starting from Zero)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_dir}/market_maker_analysis_{timestamp}.png", 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics - also fix the return calculation
        print(f"\nMarket Maker Analysis:")
        print(f"Base spread: {market_maker.base_spread_bps} bps")
        print(f"Spread range: {min(market_maker.spread_history):.1f} - {max(market_maker.spread_history):.1f} bps")
        print(f"Average spread: {np.mean(market_maker.spread_history):.1f} bps")
        print(f"Inventory range: {min(market_maker.inventory_history)} - {max(market_maker.inventory_history)}")
        print(f"Final P&L: {mm_pnl_relative[-1]:.2f}")  # Now shows actual P&L, not total portfolio value
        print(f"Total return: {(mm_pnl_relative[-1] / initial_value) * 100:.2f}%")  # Corrected return calculation

    def plot_additional_analytics(self, save_dir="plots"):
        """Plot additional market analytics and save to directory"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        trades_df = self.get_trades_dataframe()
        market_df = self.get_market_dataframe()
        
        if trades_df.empty:
            print("No trades data for additional analytics")
            return
            
        # Create a comprehensive analytics plot
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        fig.suptitle('Comprehensive Market Analytics', fontsize=16, fontweight='bold')
        
        # 1. Trade Size Distribution by Agent Type
        for agent_type in ['informed', 'uninformed', 'market_maker']:
            buyer_trades = trades_df[trades_df['buyer_type'] == agent_type]['quantity']
            seller_trades = trades_df[trades_df['seller_type'] == agent_type]['quantity']
            all_trades = pd.concat([buyer_trades, seller_trades])
            
            if not all_trades.empty:
                axes[0,0].hist(all_trades, bins=20, alpha=0.6, label=f'{agent_type.capitalize()}')
        
        axes[0,0].set_title('Trade Size Distribution by Agent Type')
        axes[0,0].set_xlabel('Trade Size')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Toxicity Rate by Trade Size - FIXED VERSION
        if 'quantity' in trades_df.columns and 'is_toxic' in trades_df.columns:
            # Bin trades by size
            trades_df['size_bin'] = pd.cut(trades_df['quantity'], bins=5)
            toxicity_by_size = trades_df.groupby('size_bin')['is_toxic'].mean() * 100
            
            axes[0,1].bar(range(len(toxicity_by_size)), toxicity_by_size.values)
            axes[0,1].set_title('Toxicity Rate by Trade Size')
            axes[0,1].set_xlabel('Trade Size Bin')
            axes[0,1].set_ylabel('Toxicity Rate (%)')
            axes[0,1].set_xticks(range(len(toxicity_by_size)))
            # Fix the formatting issue here
            bin_labels = []
            for interval in toxicity_by_size.index:
                left = interval.left
                right = interval.right
                bin_labels.append(f'{left:.1f}-{right:.1f}')
            axes[0,1].set_xticklabels(bin_labels, rotation=45)
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Price Impact Analysis
        if len(trades_df) > 1:
            # Calculate price impact (price change after trade)
            trades_df_sorted = trades_df.sort_values('timestamp')
            trades_df_sorted['price_impact'] = trades_df_sorted['price'].diff().shift(-1)
            trades_df_sorted['price_impact_pct'] = (trades_df_sorted['price_impact'] / trades_df_sorted['price']) * 100
            
            # Plot price impact for toxic vs normal trades
            toxic_impact = trades_df_sorted[trades_df_sorted['is_toxic']]['price_impact_pct'].dropna()
            normal_impact = trades_df_sorted[~trades_df_sorted['is_toxic']]['price_impact_pct'].dropna()
            
            axes[1,0].hist(normal_impact, bins=30, alpha=0.6, label='Normal Trades', color='blue')
            axes[1,0].hist(toxic_impact, bins=30, alpha=0.6, label='Toxic Trades', color='red')
            axes[1,0].set_title('Price Impact Distribution')
            axes[1,0].set_xlabel('Price Impact (%)')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Volume Profile
        hourly_volume = trades_df.groupby(trades_df['timestamp'] // 50)['quantity'].sum()
        axes[1,1].plot(hourly_volume.index * 50, hourly_volume.values, 'b-', linewidth=1)
        axes[1,1].fill_between(hourly_volume.index * 50, 0, hourly_volume.values, alpha=0.3)
        axes[1,1].set_title('Volume Profile Over Time')
        axes[1,1].set_xlabel('Time')
        axes[1,1].set_ylabel('Volume')
        axes[1,1].grid(True, alpha=0.3)
        
        # 5. Agent Performance Comparison
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
            agent_stds = [np.std(agent_returns[t]) for t in agent_types]
            
            x_pos = range(len(agent_types))
            axes[2,0].bar(x_pos, agent_means, yerr=agent_stds, capsize=5, 
                            color=['red', 'blue', 'green'][:len(agent_types)])
            axes[2,0].set_title('Average Returns by Agent Type')
            axes[2,0].set_xlabel('Agent Type')
            axes[2,0].set_ylabel('Return (%)')
            axes[2,0].set_xticks(x_pos)
            axes[2,0].set_xticklabels([t.capitalize() for t in agent_types])
            axes[2,0].grid(True, alpha=0.3)
            axes[2,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 6. Spread vs Toxicity Correlation - FIXED VERSION
        if not market_df.empty:
            try:
                # Calculate rolling toxicity rate - handle duplicates properly
                window = 50
                
                # First, aggregate trades by timestamp to handle duplicates
                trades_agg = trades_df.groupby('timestamp').agg({
                    'is_toxic': 'mean',  # Average toxicity rate for this timestamp
                    'quantity': 'sum'    # Total volume for this timestamp
                }).reset_index()
                
                # Now calculate rolling toxicity
                trades_agg = trades_agg.sort_values('timestamp')
                trades_agg['rolling_toxicity'] = trades_agg['is_toxic'].rolling(window=window, min_periods=1).mean() * 100
                
                # Create a mapping from timestamp to rolling toxicity
                toxicity_map = dict(zip(trades_agg['timestamp'], trades_agg['rolling_toxicity']))
                
                # Map to market data
                market_with_toxicity = market_df.copy()
                market_with_toxicity['toxicity_rate'] = market_with_toxicity['timestamp'].map(toxicity_map)
                market_with_toxicity = market_with_toxicity.dropna()
                
                if not market_with_toxicity.empty:
                    scatter = axes[2,1].scatter(market_with_toxicity['toxicity_rate'], 
                                                market_with_toxicity['spread'], 
                                                alpha=0.6, c=market_with_toxicity['timestamp'], 
                                                cmap='viridis', s=20)
                    axes[2,1].set_title('Spread vs Toxicity Rate')
                    axes[2,1].set_xlabel('Toxicity Rate (%)')
                    axes[2,1].set_ylabel('Bid-Ask Spread')
                    axes[2,1].grid(True, alpha=0.3)
                    plt.colorbar(scatter, ax=axes[2,1], label='Time')
            except Exception as e:
                print(f"Warning: Could not create spread vs toxicity plot: {e}")
                axes[2,1].text(0.5, 0.5, 'Spread vs Toxicity\nPlot Failed', 
                            ha='center', va='center', transform=axes[2,1].transAxes)
                axes[2,1].set_title('Spread vs Toxicity Rate (Failed)')
        
        # 7. Informed Trader Activity Heatmap
        if not trades_df.empty:
            # Create time-price grid for informed activity
            time_bins = 20
            price_bins = 15
            
            informed_trades = trades_df[
                (trades_df['buyer_type'] == 'informed') | 
                (trades_df['seller_type'] == 'informed')
            ]
            
            if not informed_trades.empty:
                time_range = informed_trades['timestamp'].max() - informed_trades['timestamp'].min()
                price_range = informed_trades['price'].max() - informed_trades['price'].min()
                
                if time_range > 0 and price_range > 0:
                    # Create 2D histogram
                    h, xedges, yedges = np.histogram2d(
                        informed_trades['timestamp'], 
                        informed_trades['price'],
                        bins=[time_bins, price_bins]
                    )
                    
                    im = axes[3,0].imshow(h.T, origin='lower', aspect='auto', 
                                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                            cmap='Reds')
                    axes[3,0].set_title('Informed Trader Activity Heatmap')
                    axes[3,0].set_xlabel('Time')
                    axes[3,0].set_ylabel('Price')
                    plt.colorbar(im, ax=axes[3,0], label='Trade Count')
        
        # 8. Market Efficiency Analysis (Autocorrelation of Returns)
        if len(market_df) > 1:
            # Calculate returns
            market_df['returns'] = market_df['price'].pct_change()
            market_df = market_df.dropna()
            
            if len(market_df) > 10:
                # Calculate autocorrelation
                lags = range(1, min(21, len(market_df)//2))
                autocorrs = [market_df['returns'].autocorr(lag=lag) for lag in lags]
                
                axes[3,1].bar(lags, autocorrs, alpha=0.7)
                axes[3,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                axes[3,1].set_title('Return Autocorrelation (Market Efficiency)')
                axes[3,1].set_xlabel('Lag')
                axes[3,1].set_ylabel('Autocorrelation')
                axes[3,1].grid(True, alpha=0.3)
                
                # Add significance bands (approximate)
                n = len(market_df)
                sig_level = 1.96 / np.sqrt(n)
                axes[3,1].axhline(y=sig_level, color='red', linestyle='--', alpha=0.5, label='95% Confidence')
                axes[3,1].axhline(y=-sig_level, color='red', linestyle='--', alpha=0.5)
                axes[3,1].legend()
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_dir}/additional_analytics_{timestamp}.png", 
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_all_analytics(self, save_dir="plots"):
        """Generate and save all analysis plots"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        print(f"Generating all plots and saving to '{save_dir}' directory...")
        
        # 1. Main simulation results
        print("📊 Plotting simulation results...")
        self.plot_simulation_results(save_dir)
        
        # 2. Market maker analysis
        print("📈 Plotting market maker analysis...")
        self.plot_market_maker_analysis(save_dir)
        
        # 3. Order flow by agent type (the new one you requested)
        print("🔄 Plotting order flow by agent type...")
        flow_df = self.plot_order_flow_by_agent_type(save_dir=save_dir)
        
        # 4. Additional analytics plot
        print("📋 Plotting additional analytics...")
        self.plot_additional_analytics(save_dir)
        
        print(f"✅ All plots saved to '{save_dir}' directory")
        return flow_df

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
    
    def pre_process_trades(self, window_size=5, price_move_threshold=0.002):
        """Label toxic trades based on future price moves - more sensitive for ML training"""
        # Get price time series from trades
        if not self.order_book.trades:
            return
            
        # Make a list of all timestamps and prices
        timestamps = [t for t, _ in self.order_book.price_history]
        prices = [p for _, p in self.order_book.price_history]
        
        # Label trades as toxic if they occur right before significant price moves
        for trade in self.order_book.trades:
            try:
                # Find the index of this trade's timestamp
                trade_idx = timestamps.index(trade.timestamp)
                
                # Look for future price moves within window (shorter window, smaller threshold)
                if trade_idx + window_size < len(prices):
                    future_price = prices[trade_idx + window_size]
                    price_move = (future_price - prices[trade_idx]) / prices[trade_idx]
                    
                    # More sensitive toxicity detection
                    if abs(price_move) > price_move_threshold:
                        # Check if trade was in direction of future price move
                        buyer_agent = next((a for a in self.agents if a.id == trade.buyer_id), None)
                        seller_agent = next((a for a in self.agents if a.id == trade.seller_id), None)
                        
                        # Mark as toxic if informed trader was involved and price moved in their favor
                        if buyer_agent and buyer_agent.type == 'informed' and price_move > 0:
                            trade.is_toxic = True
                        elif seller_agent and seller_agent.type == 'informed' and price_move < 0:
                            trade.is_toxic = True
            except:
                continue
    
    def run_simulation(self, n_steps=None):
        """Run the market simulation for n_steps and collect data for ML"""
        n_steps = n_steps or TIME_STEPS
        
        for t in range(n_steps):
            self.current_time = t
            mid_price = self.order_book.get_mid_price()
            
            # Take LOB snapshot before any actions
            self.lob_snapshots.append(self.order_book.save_snapshot(t))
            
            # Update agent PnLs
            for agent in self.agents:
                agent.update_pnl(mid_price, t)
            
            # Market makers place orders
            for agent in self.agents:
                if isinstance(agent, MarketMaker):
                    orders = agent.generate_orders(t, mid_price)
                    for order in orders:
                        # Record order data before adding to book
                        self.record_order_data(order, t)
                        self.order_book.add_limit_order(order, t)
            
            # Other agents generate and place orders
            for agent in self.agents:
                if not isinstance(agent, MarketMaker):
                    if isinstance(agent, InformedTrader):
                        order = agent.generate_order(t, self.price_path[t])
                    else:
                        order = agent.generate_order(t, mid_price)
                    
                    if order:
                        # Record order data before adding to book
                        self.record_order_data(order, t)
                        if order.type == MARKET:
                            self.order_book.add_market_order(order, t)
                        else:
                            self.order_book.add_limit_order(order, t)
                            
            for agent in self.agents:
                if isinstance(agent, MarketMaker):
                    # Calculate recent volatility (simple example)
                    if len(self.market_stats['prices']) > 10:
                        recent_prices = self.market_stats['prices'][-10:]
                        returns = [np.log(recent_prices[i]/recent_prices[i-1]) for i in range(1, len(recent_prices))]
                        volatility = np.std(returns) if len(returns) > 1 else 0
                        
                        # Simple toxicity proxy: high volatility = potentially toxic environment
                        toxicity_proxy = min(volatility * 1000, 1.0)  # Scale and cap at 1.0
                        
                        # Adjust spread
                        agent.adjust_spread(toxicity_proxy)

            # Update market statistics
            self.market_stats['timestamps'].append(t)
            self.market_stats['prices'].append(mid_price)
            self.market_stats['spreads'].append(self.order_book.get_spread())
            
            # Record trades for this step
            trades_this_step = [trade for trade in self.order_book.trades if trade.timestamp == t]
            self.market_stats['volumes'].append(sum(trade.quantity for trade in trades_this_step))
            
            # Process trades to update agent states
            for trade in trades_this_step:
                # Find buyer and seller
                buyer = next((a for a in self.agents if a.id == trade.buyer_id), None)
                seller = next((a for a in self.agents if a.id == trade.seller_id), None)
                
                if buyer:
                    buyer.record_trade(trade, True)
                if seller:
                    seller.record_trade(trade, False)
        
        # Post-process trades to identify toxic ones
        self.pre_process_trades()

    def record_order_data(self, order, timestamp):
        """Record data about each order for ML training"""
        # Find the agent type
        agent_type = next((a.type for a in self.agents if a.id == order.agent_id), 'unknown')
        
        # Get current LOB state
        mid_price = self.order_book.get_mid_price()
        spread = self.order_book.get_spread()
        
        # Calculate some basic features
        if order.type == LIMIT and order.price is not None:
            # For limit orders, calculate distance from mid
            distance_from_mid = (order.price - mid_price) / mid_price
            # Is this a passive or aggressive limit?
            is_aggressive = (order.side == BUY and order.price >= mid_price) or \
                            (order.side == SELL and order.price <= mid_price)
        else:
            # For market orders
            distance_from_mid = np.nan
            is_aggressive = True
        
        # Record the order data
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
            # Will be filled in later after trade processing
            'resulted_in_trade': False,
            'was_toxic': False
        }
        
        self.orders_data.append(order_data)

    def update_order_toxicity(self):
        """Update order data with trade outcomes and toxicity"""
        # Create mappings from order IDs to trades
        order_to_trade = {}
        for trade in self.order_book.trades:
            # Find the associated orders
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
                
                # Calculate additional features
                if order_data['side'] == 'BUY':
                    # For buy orders, positive = buyer won, negative = seller won
                    self.orders_data[i]['future_pnl'] = \
                        (self.price_path[min(trade.timestamp + 10, len(self.price_path) - 1)] - trade.price) / trade.price
                else:
                    # For sell orders, positive = seller won, negative = buyer won
                    self.orders_data[i]['future_pnl'] = \
                        (trade.price - self.price_path[min(trade.timestamp + 10, len(self.price_path) - 1)]) / trade.price
    
    def save_data_to_csv(self, output_dir="market_data"):
        """Save all collected data to CSV files for ML training"""
        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate timestamp for the filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Update order toxicity information
        self.update_order_toxicity()
        
        # Save LOB snapshots
        lob_df = pd.DataFrame(self.lob_snapshots)
        lob_df.to_csv(f"{output_dir}/lob_snapshots_{timestamp}.csv", index=False)
        
        # Save order data
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
        
        print(f"All data saved to {output_dir}/ with timestamp {timestamp}")
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

def run_market_simulation(save_data=True, output_dir="market_data", stress_test_mode=False, high_toxicity_mode=True):
   """
   Run a complete market simulation with configurable toxicity levels
   
   Parameters:
   - save_data: Whether to save data for ML training
   - output_dir: Directory to save data files
   - stress_test_mode: If True, use higher informed trader proportion for stress testing
   - high_toxicity_mode: If True, optimize for generating more toxic trades for ML training
   """
   # Initialize market environment with higher volatility for more price moves
   if high_toxicity_mode:
       market = MarketEnvironment(initial_price=INITIAL_PRICE, price_vol=0.008, mean_reversion=0.015)
       print("Running in HIGH TOXICITY mode optimized for ML training")
   else:
       market = MarketEnvironment(initial_price=INITIAL_PRICE, price_vol=0.005, mean_reversion=0.01)
   
   # Calculate agent proportions based on mode
   if stress_test_mode:
       # Stress test mode: Higher proportion of informed traders
       print("Running in STRESS TEST mode with elevated informed trader proportion")
       n_market_makers = max(1, int(NUM_AGENTS * 0.15))    # 15% market makers
       n_informed = int(NUM_AGENTS * 0.35)                 # 35% informed (stress test)
       n_uninformed = NUM_AGENTS - n_market_makers - n_informed  # ~50% uninformed
   elif high_toxicity_mode:
       # Balanced mode for ML training: moderate informed trader proportion
       print("Using MODERATE informed trader proportion for ML training")
       n_market_makers = max(1, int(NUM_AGENTS * 0.15))    # 15% market makers
       n_informed = int(NUM_AGENTS * 0.20)                 # 20% informed (good for ML)
       n_uninformed = NUM_AGENTS - n_market_makers - n_informed  # ~65% uninformed
   else:
       # Realistic proportions based on empirical literature
       print("Running with REALISTIC agent proportions")
       n_market_makers = max(1, int(NUM_AGENTS * 0.15))    # 15% market makers
       n_informed = max(1, int(NUM_AGENTS * 0.10))         # 10% informed (realistic)
       n_uninformed = NUM_AGENTS - n_market_makers - n_informed  # ~75% uninformed
   
   print(f"Agent allocation:")
   print(f"  Market Makers: {n_market_makers} ({n_market_makers/NUM_AGENTS*100:.1f}%)")
   print(f"  Informed Traders: {n_informed} ({n_informed/NUM_AGENTS*100:.1f}%)")
   print(f"  Uninformed Traders: {n_uninformed} ({n_uninformed/NUM_AGENTS*100:.1f}%)")
   
   # Add market makers (with larger capital since they need inventory)
   for i in range(n_market_makers):
       market.add_agent(MarketMaker(
           spread_bps=20 + random.randint(-5, 5),  # Some variation in spreads
           inventory_limit=50 + random.randint(-10, 20), 
           order_size=2 + random.randint(0, 2), 
           initial_capital=15000 + random.randint(-2000, 5000)
       ))
   
   # Add informed traders with different configurations based on mode
   for i in range(n_informed):
       if high_toxicity_mode:
           # More aggressive settings for ML training
           market.add_agent(InformedTrader(
               future_price_info=market.price_path,
               knowledge_horizon=random.randint(3, 8),
               order_rate=0.12 + random.uniform(0, 0.08),  # Higher rate
               initial_capital=5000 + random.randint(-1000, 3000)
           ))
       else:
           # Standard settings
           market.add_agent(InformedTrader(
               future_price_info=market.price_path,
               knowledge_horizon=random.randint(3, 8),
               order_rate=0.03 + random.uniform(0, 0.04),  # Lower rate
               initial_capital=5000 + random.randint(-1000, 3000)
           ))
   
   # Add uninformed/noise traders (majority of the market)
   for i in range(n_uninformed):
       market.add_agent(UninformedTrader(
           order_rate=0.08 + random.uniform(0, 0.08),  # Varied activity levels
           initial_capital=3000 + random.randint(-500, 2000)
       ))
   
   # Run simulation
   print(f"\nRunning simulation with {NUM_AGENTS} agents for {TIME_STEPS} time steps...")
   market.run_simulation()
   
   # Save data for ML training
   if save_data:
       timestamp = market.save_data_to_csv(output_dir)
       print(f"Data saved with timestamp {timestamp}")
   
   # Display results
   market.plot_all_analytics(save_dir="plots")
   
   # Get comprehensive statistics
   trades_df = market.get_trades_dataframe()
   
   total_trades = len(trades_df)
   toxic_trades = trades_df['is_toxic'].sum() if not trades_df.empty else 0
   toxic_pct = toxic_trades / total_trades * 100 if total_trades > 0 else 0
   
   print(f"\n{'='*60}")
   print(f"SIMULATION RESULTS")
   print(f"{'='*60}")
   print(f"Total trades: {total_trades}")
   print(f"Toxic trades: {toxic_trades} ({toxic_pct:.2f}%)")
   
   if high_toxicity_mode:
       if toxic_pct < 10:
           print(f"⚠️  Toxicity rate is {toxic_pct:.1f}% - consider increasing informed trader activity")
       elif toxic_pct > 30:
           print(f"⚠️  Toxicity rate is {toxic_pct:.1f}% - consider reducing informed trader activity")
       else:
           print(f"✅ Good toxicity rate of {toxic_pct:.1f}% for ML training")
   
   # Show who's involved in toxic trades
   if not trades_df.empty and toxic_trades > 0:
       toxic_df = trades_df[trades_df['is_toxic'] == True]
       
       buyer_types = toxic_df['buyer_type'].value_counts()
       seller_types = toxic_df['seller_type'].value_counts()
       
       print(f"\nTOXIC TRADE ANALYSIS:")
       print(f"Buyers in toxic trades:")
       for agent_type, count in buyer_types.items():
           print(f"  {agent_type.capitalize()}: {count} trades ({count/len(toxic_df)*100:.1f}%)")
       
       print(f"Sellers in toxic trades:")
       for agent_type, count in seller_types.items():
           print(f"  {agent_type.capitalize()}: {count} trades ({count/len(toxic_df)*100:.1f}%)")
       
       # Trade statistics by agent type
       print(f"\nTRADE STATISTICS BY AGENT TYPE:")
       for agent_type in set(trades_df['buyer_type']).union(set(trades_df['seller_type'])):
           buys = len(trades_df[trades_df['buyer_type'] == agent_type])
           sells = len(trades_df[trades_df['seller_type'] == agent_type])
           total = buys + sells
           
           toxic_buys = len(toxic_df[toxic_df['buyer_type'] == agent_type])
           toxic_sells = len(toxic_df[toxic_df['seller_type'] == agent_type])
           toxic_total = toxic_buys + toxic_sells
           
           toxicity_rate = toxic_total / total * 100 if total > 0 else 0
           trade_share = total / (len(trades_df) * 2) * 100  # *2 because each trade has buyer and seller
           
           print(f"  {agent_type.capitalize()}: {buys} buys, {sells} sells")
           print(f"    Total participation: {total} sides ({trade_share:.1f}% of all trade sides)")
           print(f"    Toxicity rate: {toxicity_rate:.1f}%")
   
   # Agent performance summary
   print(f"\nAGENT PERFORMANCE SUMMARY:")
   agent_performance = {}
   for agent in market.agents:
       initial_value = agent.pnl_history[0][1]
       final_value = agent.pnl_history[-1][1]
       return_pct = (final_value / initial_value - 1) * 100
       
       if agent.type not in agent_performance:
           agent_performance[agent.type] = []
       agent_performance[agent.type].append(return_pct)
   
   for agent_type, returns in agent_performance.items():
       avg_return = np.mean(returns)
       std_return = np.std(returns)
       print(f"  {agent_type.capitalize()}:")
       print(f"    Average return: {avg_return:.2f}% ± {std_return:.2f}%")
       print(f"    Best performer: {max(returns):.2f}%")
       print(f"    Worst performer: {min(returns):.2f}%")
   
   # Describe the saved data if applicable
   if save_data:
       print(f"\n{'='*60}")
       print(f"SAVED DATA FOR ML TRAINING:")
       print(f"{'='*60}")
       print("1. LOB snapshots: Order book state at each timestep")
       print("2. Orders: All orders with features and toxicity labels") 
       print("3. Trades: All executed trades with toxicity labels")
       print("4. Market stats: Price, spread and volume over time")
       print("5. Price path: Fundamental price process")
       print(f"\nToxicity rate of {toxic_pct:.1f}% should provide sufficient examples")
       print("for training ML models for toxicity prediction.")
   
   return market

if __name__ == "__main__":
   # Run with high toxicity mode for ML training
   market = run_market_simulation(high_toxicity_mode=True)
   
   # Uncomment below to also run different modes for comparison
   # print("\n" + "="*80)
   # print("RUNNING REALISTIC MODE FOR COMPARISON")
   # print("="*80)
   # realistic_market = run_market_simulation(high_toxicity_mode=False, output_dir="market_data_realistic")
   
   # print("\n" + "="*80)
   # print("RUNNING STRESS TEST WITH MAXIMUM INFORMED TRADERS")
   # print("="*80)
   # stress_market = run_market_simulation(stress_test_mode=True, output_dir="market_data_stress")