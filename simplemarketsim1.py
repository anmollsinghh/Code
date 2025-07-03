import numpy as np
import pandas as pd
import heapq
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import os

# Define constants
INITIAL_PRICE = 100.0
MIN_PRICE = 0.01  # Price floor to prevent collapse to zero
TIME_STEPS = 1000
NUM_AGENTS = 50

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


    """Central limit order book for matching orders"""
    def __init__(self):
        # Price-ordered heaps (max heap for bids, min heap for asks)
        self.bids = []  # (-price, timestamp, order)
        self.asks = []  # (price, timestamp, order)
        self.last_price = INITIAL_PRICE
        self.trades = []
        self.order_history = []
        self.price_history = []
        
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
    def __init__(self, future_price_info, knowledge_horizon=10, **kwargs):
        super().__init__(**kwargs)
        self.future_price_info = future_price_info  # Time series of future prices
        self.knowledge_horizon = knowledge_horizon  # How far ahead they can see
        self.type = 'informed'
    
    def generate_order(self, timestamp, current_price):
        """Generate orders based on future price knowledge"""
        # Look ahead to see if prices will rise or fall
        future_horizon = min(timestamp + self.knowledge_horizon, len(self.future_price_info) - 1)
        if future_horizon <= timestamp:
            return None
            
        future_price = self.future_price_info[future_horizon]
        price_change = future_price - current_price
        
        # Generate order with probability proportional to expected profit
        if abs(price_change) < 0.1:  # Ignore small changes
            return None
            
        # Position size is proportional to expected profit
        size = max(1, int(abs(price_change) * 10))
        
        if price_change > 0:
            # Buy if price will rise
            return Order(self.id, MARKET, BUY, None, size, timestamp)
        else:
            # Sell if price will fall
            return Order(self.id, MARKET, SELL, None, size, timestamp)

class UninformedTrader(Agent):
    """Trader who trades randomly without information"""
    def __init__(self, order_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.order_rate = order_rate  # Probability of submitting an order
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
    
    def plot_market_maker_analysis(self):
        """Plot detailed market maker behavior analysis"""
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
        
        # 4. Market Maker PnL
        mm_pnl_timestamps = [t for t, _ in market_maker.pnl_history]
        mm_pnl_values = [pnl for _, pnl in market_maker.pnl_history]
        
        axes[3].plot(mm_pnl_timestamps, mm_pnl_values, 
                    'orange', linewidth=2, label='MM PnL')
        axes[3].axhline(y=market_maker.pnl_history[0][1], color='k', linestyle='--', 
                        alpha=0.5, label='Starting Capital')
        axes[3].set_ylabel('PnL')
        axes[3].set_xlabel('Time')
        axes[3].set_title('Market Maker Profit & Loss')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\nMarket Maker Analysis:")
        print(f"Base spread: {market_maker.base_spread_bps} bps")
        print(f"Spread range: {min(market_maker.spread_history):.1f} - {max(market_maker.spread_history):.1f} bps")
        print(f"Average spread: {np.mean(market_maker.spread_history):.1f} bps")
        print(f"Inventory range: {min(market_maker.inventory_history)} - {max(market_maker.inventory_history)}")
        print(f"Final PnL: {mm_pnl_values[-1]:.2f}")
        print(f"Total return: {((mm_pnl_values[-1] / mm_pnl_values[0]) - 1) * 100:.2f}%")

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
    
    def pre_process_trades(self, window_size=10, price_move_threshold=0.005):
        """Label toxic trades based on future price moves"""
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
                
                # Look for future price moves within window
                if trade_idx + window_size < len(prices):
                    future_price = prices[trade_idx + window_size]
                    price_move = (future_price - prices[trade_idx]) / prices[trade_idx]
                    
                    # Label as toxic if significant move in direction of trade
                    if (trade.buyer_id > trade.seller_id and price_move > price_move_threshold) or \
                       (trade.seller_id > trade.buyer_id and price_move < -price_move_threshold):
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
    
    def plot_simulation_results(self):
        """Plot key statistics and results from the simulation"""
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
        
        # Plot PnL for each agent type
        agent_types = set(agent.type for agent in self.agents)
        for agent_type in agent_types:
            agents_of_type = [a for a in self.agents if a.type == agent_type]
            if not agents_of_type:
                continue
                
            # Calculate average PnL for this agent type
            pnl_by_time = defaultdict(list)
            for agent in agents_of_type:
                for t, pnl in agent.pnl_history:
                    pnl_by_time[t].append(pnl)
            
            avg_pnl = [(t, sum(pnls)/len(pnls)) for t, pnls in sorted(pnl_by_time.items())]
            axes[2].plot([t for t, _ in avg_pnl], [p for _, p in avg_pnl], label=f'{agent_type.capitalize()}')
        
        axes[2].set_title('Agent PnL by Type')
        axes[2].set_ylabel('PnL')
        axes[2].set_xlabel('Time')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()

def run_market_simulation(save_data=True, output_dir="market_data"):
    """Run a complete market simulation and display results"""
    # Initialize market environment
    market = MarketEnvironment(initial_price=INITIAL_PRICE, price_vol=0.005, mean_reversion=0.01)
    
    # Add agents
    # Market maker
    market.add_agent(MarketMaker(spread_bps=20, inventory_limit=50, order_size=3, initial_capital=10000))
    
    # Informed traders (knows about future price moves)
    for _ in range(5):
        market.add_agent(InformedTrader(
            future_price_info=market.price_path,
            knowledge_horizon=random.randint(3, 8),
            initial_capital=5000
        ))
    
    # Uninformed/noise traders
    for _ in range(NUM_AGENTS - 6):  # -6 to account for MM and informed traders
        market.add_agent(UninformedTrader(order_rate=0.1, initial_capital=5000))
    
    # Run simulation
    market.run_simulation()
    
    # Save data for ML training
    if save_data:
        timestamp = market.save_data_to_csv(output_dir)
        print(f"Data saved with timestamp {timestamp}")
    
    # Display results
    market.plot_simulation_results()
    market.plot_market_maker_analysis()
    # Get statistics on toxic trades
    trades_df = market.get_trades_dataframe()
    
    total_trades = len(trades_df)
    toxic_trades = trades_df['is_toxic'].sum()
    toxic_pct = toxic_trades / total_trades * 100 if total_trades > 0 else 0
    
    print(f"Simulation results:")
    print(f"Total trades: {total_trades}")
    print(f"Toxic trades: {toxic_trades} ({toxic_pct:.2f}%)")
    
    # Show who's involved in toxic trades
    if not trades_df.empty:
        toxic_df = trades_df[trades_df['is_toxic'] == True]
        
        buyer_types = toxic_df['buyer_type'].value_counts()
        seller_types = toxic_df['seller_type'].value_counts()
        
        print("\nToxic trade statistics:")
        print("Buyers in toxic trades:")
        for agent_type, count in buyer_types.items():
            print(f"  {agent_type}: {count} trades ({count/len(toxic_df)*100:.2f}%)")
        
        print("Sellers in toxic trades:")
        for agent_type, count in seller_types.items():
            print(f"  {agent_type}: {count} trades ({count/len(toxic_df)*100:.2f}%)")
        
        # More trade details
        print("\nTrade statistics by agent type:")
        for agent_type in set(trades_df['buyer_type']).union(set(trades_df['seller_type'])):
            buys = len(trades_df[trades_df['buyer_type'] == agent_type])
            sells = len(trades_df[trades_df['seller_type'] == agent_type])
            total = buys + sells
            
            toxic_buys = len(toxic_df[toxic_df['buyer_type'] == agent_type])
            toxic_sells = len(toxic_df[toxic_df['seller_type'] == agent_type])
            toxic_total = toxic_buys + toxic_sells
            
            toxic_rate = toxic_total / total * 100 if total > 0 else 0
            
            print(f"  {agent_type.capitalize()}: {buys} buys, {sells} sells, {toxic_rate:.2f}% toxicity rate")
    
    # Describe the saved data if applicable
    if save_data:
        print("\nThe following data has been saved for ML training:")
        print("1. LOB snapshots: Order book state at each timestep")
        print("2. Orders: All orders with features and toxicity labels")
        print("3. Trades: All executed trades with toxicity labels")
        print("4. Market stats: Price, spread and volume over time")
        print("5. Price path: Fundamental price process")
        print("\nYou can use these files to train ML models for toxicity prediction")
    
    return market

if __name__ == "__main__":
    market = run_market_simulation()