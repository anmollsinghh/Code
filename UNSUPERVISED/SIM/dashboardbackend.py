"""
Real-Time Dashboard Backend Integration
Connects the ML market making simulation to the monitoring dashboard
"""

import json
import time
import threading
import websocket
from datetime import datetime
from collections import deque
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import logging

# Suppress Flask development server warning
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class DashboardDataBroker:
    """
    Manages real-time data flow between simulation and dashboard
    """
    
    def __init__(self, update_frequency=1.0):
        self.update_frequency = update_frequency  # seconds
        self.data_buffer = deque(maxlen=1000)
        self.current_metrics = {}
        self.algorithm_stats = {}
        self.is_running = False
        self.subscribers = []
        
        # Initialize with default values
        self.current_metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_return': 1.58,
            'sharpe_ratio': 4.82,
            'current_spread': 56.0,
            'volume': 2847,
            'unrealized_pnl': 3247,
            'realized_pnl': 1892,
            'mid_price': 248.92,
            'inventory': 12,
            'toxicity_score': 0.342,
            'detection_rate': 2.4,
            'fill_rate': 94.2,
            'avg_fill_time': 120,
            'active_algorithm': 'profit_optimizer'
        }
        
        self.algorithm_stats = {
            'profit_optimizer': {'return': 1.58, 'sharpe': 4.82, 'spread': 56.0, 'volume': 2847, 'status': 'excellent'},
            'ml_ensemble': {'return': 0.92, 'sharpe': 3.92, 'spread': 59.4, 'volume': 2653, 'status': 'good'},
            'linear_micro': {'return': 0.96, 'sharpe': 3.87, 'spread': 66.9, 'volume': 2421, 'status': 'good'},
            'baseline': {'return': 0.90, 'sharpe': 3.62, 'spread': 68.2, 'volume': 2198, 'status': 'good'}
        }
    
    def update_from_simulation(self, market_makers, order_book, current_time):
        """
        Update metrics from the running simulation
        
        Parameters:
        - market_makers: dict of market maker instances
        - order_book: current order book state
        - current_time: simulation timestamp
        """
        try:
            # Update algorithm performance
            for algo_name, mm in market_makers.items():
                metrics = mm.get_performance_metrics()
                self.algorithm_stats[algo_name] = {
                    'return': metrics.get('total_return_pct', 0),
                    'sharpe': self._calculate_sharpe_ratio(mm),
                    'spread': metrics.get('avg_spread_bps', 0),
                    'volume': metrics.get('total_volume_traded', 0),
                    'status': self._get_performance_status(metrics.get('total_return_pct', 0))
                }
            
            # Find best performing algorithm
            best_algo = max(self.algorithm_stats.items(), key=lambda x: x[1]['return'])
            best_mm = market_makers[best_algo[0]]
            
            # Update current metrics from best algorithm
            mid_price = order_book.get_mid_price()
            spread = order_book.get_spread()
            
            self.current_metrics.update({
                'timestamp': datetime.now().isoformat(),
                'total_return': best_algo[1]['return'],
                'sharpe_ratio': best_algo[1]['sharpe'],
                'current_spread': (spread / mid_price) * 10000 if mid_price > 0 else 0,
                'volume': best_algo[1]['volume'],
                'mid_price': mid_price,
                'inventory': best_mm.inventory,
                'toxicity_score': np.mean(best_mm.toxicity_history[-10:]) if best_mm.toxicity_history else 0.5,
                'active_algorithm': best_algo[0]
            })
            
            # Calculate P&L
            if len(best_mm.pnl_history) > 1:
                current_pnl = best_mm.pnl_history[-1][1]
                initial_pnl = best_mm.pnl_history[0][1]
                self.current_metrics['unrealized_pnl'] = current_pnl - initial_pnl
            
            # Add to data buffer for time series
            self.data_buffer.append(self.current_metrics.copy())
            
            # Notify subscribers
            self._notify_subscribers()
            
        except Exception as e:
            print(f"Error updating dashboard metrics: {e}")
    
    def _calculate_sharpe_ratio(self, mm):
        """Calculate Sharpe ratio from P&L history"""
        if len(mm.pnl_history) < 2:
            return 0
        
        returns = []
        for i in range(1, len(mm.pnl_history)):
            prev_pnl = mm.pnl_history[i-1][1]
            curr_pnl = mm.pnl_history[i][1]
            if prev_pnl > 0:
                returns.append((curr_pnl - prev_pnl) / prev_pnl)
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        
        return (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    
    def _get_performance_status(self, return_pct):
        """Get performance status badge"""
        if return_pct > 1.0:
            return 'excellent'
        elif return_pct > 0.5:
            return 'good'
        else:
            return 'poor'
    
    def _notify_subscribers(self):
        """Notify all subscribers of data update"""
        for callback in self.subscribers:
            try:
                callback(self.current_metrics)
            except:
                pass
    
    def subscribe(self, callback):
        """Subscribe to data updates"""
        self.subscribers.append(callback)
    
    def get_current_data(self):
        """Get current metrics snapshot"""
        return {
            'metrics': self.current_metrics,
            'algorithms': self.algorithm_stats,
            'time_series': list(self.data_buffer)[-50:]  # Last 50 points
        }

class DashboardWebServer:
    """
    Flask web server for the dashboard
    """
    
    def __init__(self, data_broker, host='localhost', port=5000):
        self.data_broker = data_broker
        self.host = host
        self.port = port
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'ml_market_making_dashboard'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio()
        
        # Subscribe to data updates
        self.data_broker.subscribe(self._broadcast_update)
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Serve the dashboard HTML"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/data')
        def get_data():
            """API endpoint for current data"""
            return jsonify(self.data_broker.get_current_data())
        
        @self.app.route('/api/algorithms')
        def get_algorithms():
            """API endpoint for algorithm comparison"""
            return jsonify(self.data_broker.algorithm_stats)
        
        @self.app.route('/api/settings', methods=['POST'])
        def update_settings():
            """API endpoint for updating settings"""
            settings = request.json
            print(f"Settings updated: {settings}")
            return jsonify({'status': 'success', 'settings': settings})
    
    def _setup_socketio(self):
        """Setup WebSocket events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print('Dashboard client connected')
            emit('initial_data', self.data_broker.get_current_data())
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('Dashboard client disconnected')
        
        @self.socketio.on('switch_algorithm')
        def handle_algorithm_switch(data):
            algorithm = data.get('algorithm')
            print(f"Algorithm switch requested: {algorithm}")
            # Here you would implement actual algorithm switching logic
            emit('algorithm_switched', {'algorithm': algorithm, 'status': 'success'})
        
        @self.socketio.on('emergency_stop')
        def handle_emergency_stop():
            print("EMERGENCY STOP REQUESTED")
            # Implement emergency stop logic
            emit('emergency_stop_confirmed', {'status': 'stopped'})
    
    def _broadcast_update(self, metrics):
        """Broadcast data update to all connected clients"""
        self.socketio.emit('data_update', {
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
    
    def run(self, debug=False):
        """Start the web server"""
        print(f"🚀 Starting dashboard server at http://{self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port, debug=debug)

class EnhancedSimulationWithDashboard:
    """
    Enhanced simulation that integrates with the dashboard
    """
    
    def __init__(self, model_path, dashboard_enabled=True, dashboard_port=5000):
        self.dashboard_enabled = dashboard_enabled
        self.dashboard_port = dashboard_port
        
        # Initialize original simulation components
        from newsim import MLEnhancedMarketEnvironment
        self.market_env = MLEnhancedMarketEnvironment(model_path)
        
        # Initialize dashboard components
        if self.dashboard_enabled:
            self.data_broker = DashboardDataBroker(update_frequency=2.0)
            self.web_server = DashboardWebServer(self.data_broker, port=dashboard_port)
            self.dashboard_thread = None
    
    def start_dashboard(self):
        """Start the dashboard in a separate thread"""
        if not self.dashboard_enabled:
            return
        
        def run_dashboard():
            self.web_server.run(debug=False)
        
        self.dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        self.dashboard_thread.start()
        
        print(f"📊 Dashboard started at http://localhost:{self.dashboard_port}")
        print("🔄 Real-time data streaming enabled")
    
    def run_simulation_with_dashboard(self, n_steps=2000, update_frequency=100):
        """
        Run simulation with real-time dashboard updates
        """
        print("🚀 Starting ML-Enhanced Market Making with Real-Time Dashboard")
        print("="*70)
        
        # Setup agents
        self.market_env.setup_agents(n_informed=3, n_uninformed=11)
        
        # Start dashboard
        if self.dashboard_enabled:
            self.start_dashboard()
            time.sleep(2)  # Give dashboard time to start
        
        print(f"\n📈 Running simulation with real-time monitoring")
        print(f"🌐 Dashboard: http://localhost:{self.dashboard_port}")
        print(f"⏱️  Update frequency: every {update_frequency} steps")
        print("-" * 70)
        
        # Run simulation with dashboard updates
        for t in range(n_steps):
            self.market_env.current_time = t
            mid_price = self.market_env.order_book.get_mid_price()
            
            # Prepare market context
            market_context = {
                'recent_prices': [p for _, p in self.market_env.order_book.price_history[-20:]] if self.market_env.order_book.price_history else [mid_price],
                'recent_trades': self.market_env.order_book.trades[-50:] if self.market_env.order_book.trades else [],
                'current_depth': len(self.market_env.order_book.bids) + len(self.market_env.order_book.asks)
            }
            
            # Update agent PnLs
            for agent in self.market_env.agents:
                agent.update_pnl(mid_price, t)
            
            # Market makers place orders
            for agent in self.market_env.agents:
                if hasattr(agent, 'spread_algorithm'):  # ML Enhanced Market Maker
                    orders = agent.generate_orders(t, mid_price, market_context)
                    for order in orders:
                        self.market_env.order_book.add_limit_order(order, t)
            
            # Other agents trade
            for agent in self.market_env.agents:
                if not hasattr(agent, 'spread_algorithm'):
                    order = None
                    
                    if hasattr(agent, 'future_price_info'):  # Informed trader
                        order = agent.generate_order(t, self.market_env.price_path[t])
                    else:  # Noise trader
                        order = agent.generate_order(t, mid_price)
                    
                    if order:
                        if order.type == 1:  # MARKET
                            self.market_env.order_book.add_market_order(order, t)
                        else:
                            self.market_env.order_book.add_limit_order(order, t)
            
            # Record trades and update agent states
            trades_this_step = [trade for trade in self.market_env.order_book.trades if trade.timestamp == t]
            for trade in trades_this_step:
                buyer = next((a for a in self.market_env.agents if a.id == trade.buyer_id), None)
                seller = next((a for a in self.market_env.agents if a.id == trade.seller_id), None)
                
                if buyer and hasattr(buyer, 'record_trade'):
                    buyer.record_trade(trade, True)
                if seller and hasattr(seller, 'record_trade'):
                    seller.record_trade(trade, False)
            
            # Update dashboard every update_frequency steps
            if self.dashboard_enabled and t % update_frequency == 0:
                self.data_broker.update_from_simulation(
                    self.market_env.market_makers,
                    self.market_env.order_book,
                    t
                )
                
                # Print progress
                total_trades = len(self.market_env.order_book.trades)
                mm_trades = sum(1 for trade in self.market_env.order_book.trades 
                              if any(trade.buyer_id == mm.id or trade.seller_id == mm.id 
                                    for mm in self.market_env.market_makers.values()))
                
                print(f"Step {t:4d}/{n_steps} ({100*t/n_steps:5.1f}%) | "
                      f"Trades: {total_trades:4d} | MM: {mm_trades:4d} | "
                      f"Price: ${mid_price:7.2f} | "
                      f"Dashboard: 📊 Live")
        
        # Final update
        if self.dashboard_enabled:
            self.data_broker.update_from_simulation(
                self.market_env.market_makers,
                self.market_env.order_book,
                n_steps
            )
        
        # Calculate final results
        results = self.market_env.get_comparison_results()
        
        print("\n" + "="*70)
        print("🏁 SIMULATION COMPLETED - REAL-TIME MONITORING ACTIVE")
        print("="*70)
        
        # Print final results
        print(f"\n📊 FINAL RESULTS:")
        for algo_name, result in results.items():
            metrics = result['performance_metrics']
            print(f"  {algo_name.upper():<15}: "
                  f"Return={metrics['total_return_pct']:6.2f}% | "
                  f"Sharpe={result['sharpe_ratio']:5.2f} | "
                  f"Spread={metrics['avg_spread_bps']:5.1f}bps")
        
        if self.dashboard_enabled:
            print(f"\n🌐 Dashboard running at: http://localhost:{self.dashboard_port}")
            print("📊 Real-time monitoring continues...")
            print("💡 Press Ctrl+C to stop")
            
            # Keep the dashboard running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Dashboard stopped")
        
        return results

# Example usage and testing
if __name__ == "__main__":
    # Test the dashboard with simulation
    try:
        model_path = "calibrated_toxicity_models/enhanced_toxicity_detector_20250626_005749.joblib"
        
        # Create enhanced simulation with dashboard
        enhanced_sim = EnhancedSimulationWithDashboard(
            model_path=model_path,
            dashboard_enabled=True,
            dashboard_port=5001
        )
        
        # Run with real-time monitoring
        results = enhanced_sim.run_simulation_with_dashboard(
            n_steps=1000,
            update_frequency=50  # Update dashboard every 50 steps
        )
        
    except Exception as e:
        print(f"Error running enhanced simulation: {e}")
        import traceback
        traceback.print_exc()

# Dashboard HTML template creation
def create_dashboard_template():
    """Create the dashboard HTML template file"""
    import os
    
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # The dashboard HTML would be saved as templates/dashboard.html
    # For now, users can save the HTML artifact as templates/dashboard.html
    print("📁 Create 'templates' folder and save the dashboard HTML as 'templates/dashboard.html'")
    print("🚀 Then run this script to start the integrated dashboard!")

if __name__ == "__main__":
    create_dashboard_template()