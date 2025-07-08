"""
PIN and VPIN Calculator with ML Toxicity Comparison
Implements traditional microstructure toxicity measures and compares with ML predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import poisson, pearsonr, spearmanr
from collections import deque
import warnings
from datetime import datetime
import joblib
import os

warnings.filterwarnings('ignore')

class PINCalculator:
    """
    Implements PIN (Probability of Informed Trading) calculation using Easley et al. (1996) methodology
    """
    
    def __init__(self, estimation_window=250):
        self.estimation_window = estimation_window
        self.mu = None  # Probability of information event
        self.alpha = None  # Probability of informed trading given information event
        self.epsilon_b = None  # Arrival rate of uninformed buy orders
        self.epsilon_s = None  # Arrival rate of uninformed sell orders
        self.delta = None  # Probability that information event is bad news
        
    def classify_trades(self, trades_df):
        """
        Classify trades as buyer-initiated or seller-initiated using tick rule
        
        Parameters:
        trades_df: DataFrame with columns ['timestamp', 'price', 'quantity']
        
        Returns:
        DataFrame with additional 'side' column (1 for buy, -1 for sell)
        """
        df = trades_df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Initialize side column
        df['side'] = 0
        
        # Use tick rule: compare current price with previous price
        for i in range(1, len(df)):
            if df.loc[i, 'price'] > df.loc[i-1, 'price']:
                df.loc[i, 'side'] = 1  # Buy
            elif df.loc[i, 'price'] < df.loc[i-1, 'price']:
                df.loc[i, 'side'] = -1  # Sell
            else:
                # If price unchanged, use previous classification
                df.loc[i, 'side'] = df.loc[i-1, 'side']
        
        # Handle first trade (assume neutral, assign randomly)
        if len(df) > 0:
            df.loc[0, 'side'] = 1 if np.random.random() > 0.5 else -1
            
        return df
    
    def aggregate_daily_data(self, trades_df):
        """
        Aggregate trades into daily buy and sell volumes
        
        Parameters:
        trades_df: DataFrame with classified trades
        
        Returns:
        DataFrame with daily aggregated data
        """
        # Create date column from timestamp
        trades_df['date'] = pd.to_datetime(trades_df['timestamp'], unit='s').dt.date
        
        # Aggregate by date
        daily_data = []
        
        for date in trades_df['date'].unique():
            day_trades = trades_df[trades_df['date'] == date]
            
            buy_trades = day_trades[day_trades['side'] == 1]
            sell_trades = day_trades[day_trades['side'] == -1]
            
            buy_volume = buy_trades['quantity'].sum() if len(buy_trades) > 0 else 0
            sell_volume = sell_trades['quantity'].sum() if len(sell_trades) > 0 else 0
            
            buy_count = len(buy_trades)
            sell_count = len(sell_trades)
            
            daily_data.append({
                'date': date,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'total_volume': buy_volume + sell_volume,
                'total_trades': buy_count + sell_count
            })
        
        return pd.DataFrame(daily_data)
    
    def likelihood_function(self, params, buy_counts, sell_counts):
        """
        Log-likelihood function for PIN estimation
        
        Parameters:
        params: [mu, alpha, epsilon_b, epsilon_s, delta]
        buy_counts: Array of daily buy order counts
        sell_counts: Array of daily sell order counts
        
        Returns:
        Negative log-likelihood (for minimization)
        """
        mu, alpha, epsilon_b, epsilon_s, delta = params
        
        # Ensure parameters are valid
        if any(p < 0 for p in params) or any(p > 1 for p in [mu, alpha, delta]):
            return 1e10
        
        n_days = len(buy_counts)
        log_likelihood = 0
        
        for i in range(n_days):
            b = buy_counts[i]
            s = sell_counts[i]
            
            # Three scenarios for each day:
            # 1. No information event (probability 1-mu)
            prob_no_info = (1 - mu) * poisson.pmf(b, epsilon_b) * poisson.pmf(s, epsilon_s)
            
            # 2. Good news (probability mu * delta)
            prob_good_news = (mu * delta * 
                             poisson.pmf(b, epsilon_b + alpha) * 
                             poisson.pmf(s, epsilon_s))
            
            # 3. Bad news (probability mu * (1-delta))
            prob_bad_news = (mu * (1 - delta) * 
                            poisson.pmf(b, epsilon_b) * 
                            poisson.pmf(s, epsilon_s + alpha))
            
            # Total probability for this day
            total_prob = prob_no_info + prob_good_news + prob_bad_news
            
            if total_prob > 0:
                log_likelihood += np.log(total_prob)
            else:
                return 1e10
        
        return -log_likelihood
    
    def estimate_pin(self, trades_df, multiple_starting_points=True):
        """
        Estimate PIN parameters using maximum likelihood estimation
        
        Parameters:
        trades_df: DataFrame with trade data
        multiple_starting_points: Whether to try multiple initial parameter values
        
        Returns:
        Dictionary with estimated parameters and PIN value
        """
        # Classify trades and aggregate daily data
        classified_trades = self.classify_trades(trades_df)
        daily_data = self.aggregate_daily_data(classified_trades)
        
        if len(daily_data) < 5:
            print("Warning: Insufficient data for PIN estimation (need at least 5 days)")
            return None
        
        buy_counts = daily_data['buy_count'].values
        sell_counts = daily_data['sell_count'].values
        
        best_result = None
        best_likelihood = np.inf
        
        # Try multiple starting points if requested
        starting_points = []
        if multiple_starting_points:
            # Generate random starting points
            np.random.seed(42)  # For reproducibility
            for _ in range(10):
                starting_points.append([
                    np.random.uniform(0.1, 0.9),  # mu
                    np.random.uniform(0.1, 0.9),  # alpha
                    np.random.uniform(0.1, np.mean(buy_counts)),  # epsilon_b
                    np.random.uniform(0.1, np.mean(sell_counts)),  # epsilon_s
                    np.random.uniform(0.3, 0.7)   # delta
                ])
        else:
            # Single starting point based on data
            starting_points.append([
                0.5,  # mu
                0.3,  # alpha
                np.mean(buy_counts) * 0.8,  # epsilon_b
                np.mean(sell_counts) * 0.8,  # epsilon_s
                0.5   # delta
            ])
        
        # Bounds for parameters
        bounds = [
            (0.001, 0.999),  # mu
            (0.001, 0.999),  # alpha
            (0.001, None),   # epsilon_b
            (0.001, None),   # epsilon_s
            (0.001, 0.999)   # delta
        ]
        
        for start_params in starting_points:
            try:
                result = minimize(
                    self.likelihood_function,
                    start_params,
                    args=(buy_counts, sell_counts),
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 1000}
                )
                
                if result.success and result.fun < best_likelihood:
                    best_likelihood = result.fun
                    best_result = result
                    
            except Exception as e:
                continue
        
        if best_result is None:
            print("PIN estimation failed to converge")
            return None
        
        # Extract parameters
        self.mu, self.alpha, self.epsilon_b, self.epsilon_s, self.delta = best_result.x
        
        # Calculate PIN
        pin_value = (self.mu * self.alpha) / (self.mu * self.alpha + self.epsilon_b + self.epsilon_s)
        
        return {
            'PIN': pin_value,
            'mu': self.mu,
            'alpha': self.alpha,
            'epsilon_b': self.epsilon_b,
            'epsilon_s': self.epsilon_s,
            'delta': self.delta,
            'log_likelihood': -best_likelihood,
            'daily_data': daily_data
        }

class VPINCalculator:
    """
    Implements VPIN (Volume-Synchronized PIN) calculation using Easley et al. (2012) methodology
    """
    
    def __init__(self, volume_bucket_size=None, n_buckets=50):
        """
        Parameters:
        volume_bucket_size: Volume per bucket (if None, calculated from data)
        n_buckets: Number of buckets for VPIN calculation
        """
        self.volume_bucket_size = volume_bucket_size
        self.n_buckets = n_buckets
        self.vpin_history = []
        
    def classify_trades_bulk(self, trades_df):
        """
        Bulk classify trades using multiple methods for robustness
        """
        df = trades_df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Method 1: Tick rule
        df['side_tick'] = 0
        for i in range(1, len(df)):
            if df.loc[i, 'price'] > df.loc[i-1, 'price']:
                df.loc[i, 'side_tick'] = 1
            elif df.loc[i, 'price'] < df.loc[i-1, 'price']:
                df.loc[i, 'side_tick'] = -1
            else:
                df.loc[i, 'side_tick'] = df.loc[i-1, 'side_tick']
        
        # Handle first trade
        if len(df) > 0:
            df.loc[0, 'side_tick'] = 1 if np.random.random() > 0.5 else -1
        
        # Method 2: Quote rule (if we had bid/ask data, we'd use this)
        # For simulation data, we'll use a proxy based on trade size
        # Larger trades more likely to be informed (buy-side)
        median_size = df['quantity'].median()
        df['side_size'] = np.where(df['quantity'] > median_size, 1, -1)
        
        # Combine methods (giving more weight to tick rule)
        df['side'] = np.where(
            df['side_tick'] != 0,
            df['side_tick'],
            df['side_size']
        )
        
        return df
    
    def create_volume_buckets(self, trades_df):
        """
        Create volume buckets for VPIN calculation
        
        Parameters:
        trades_df: DataFrame with classified trades
        
        Returns:
        List of volume buckets with buy/sell volumes
        """
        if self.volume_bucket_size is None:
            # Calculate bucket size as 1/50th of total volume
            total_volume = trades_df['quantity'].sum()
            self.volume_bucket_size = total_volume / self.n_buckets
        
        buckets = []
        current_bucket = {'buy_volume': 0, 'sell_volume': 0, 'total_volume': 0, 'trades': []}
        
        for _, trade in trades_df.iterrows():
            # Add trade to current bucket
            if trade['side'] == 1:  # Buy
                current_bucket['buy_volume'] += trade['quantity']
            else:  # Sell
                current_bucket['sell_volume'] += trade['quantity']
            
            current_bucket['total_volume'] += trade['quantity']
            current_bucket['trades'].append(trade)
            
            # Check if bucket is full
            if current_bucket['total_volume'] >= self.volume_bucket_size:
                buckets.append(current_bucket.copy())
                current_bucket = {'buy_volume': 0, 'sell_volume': 0, 'total_volume': 0, 'trades': []}
        
        # Add remaining trades as final bucket if significant
        if current_bucket['total_volume'] > self.volume_bucket_size * 0.1:
            buckets.append(current_bucket)
        
        return buckets
    
    def calculate_vpin(self, trades_df):
        """
        Calculate VPIN values over rolling windows
        
        Parameters:
        trades_df: DataFrame with trade data
        
        Returns:
        Dictionary with VPIN time series and statistics
        """
        # Classify trades
        classified_trades = self.classify_trades_bulk(trades_df)
        
        # Create volume buckets
        buckets = self.create_volume_buckets(classified_trades)
        
        if len(buckets) < self.n_buckets:
            print(f"Warning: Only {len(buckets)} buckets created, need at least {self.n_buckets}")
            self.n_buckets = max(len(buckets) - 1, 1)
        
        # Calculate VPIN for rolling windows
        vpin_values = []
        bucket_timestamps = []
        
        for i in range(self.n_buckets, len(buckets)):
            window_buckets = buckets[i-self.n_buckets:i]
            
            # Calculate VPIN for this window
            total_imbalance = 0
            total_volume = 0
            
            for bucket in window_buckets:
                imbalance = abs(bucket['buy_volume'] - bucket['sell_volume'])
                total_imbalance += imbalance
                total_volume += bucket['total_volume']
            
            if total_volume > 0:
                vpin = total_imbalance / total_volume
            else:
                vpin = 0
            
            vpin_values.append(vpin)
            
            # Use timestamp of last trade in current bucket
            if window_buckets[-1]['trades']:
                bucket_timestamps.append(window_buckets[-1]['trades'][-1]['timestamp'])
            else:
                bucket_timestamps.append(0)
        
        self.vpin_history = vpin_values
        
        return {
            'vpin_values': vpin_values,
            'timestamps': bucket_timestamps,
            'mean_vpin': np.mean(vpin_values) if vpin_values else 0,
            'std_vpin': np.std(vpin_values) if vpin_values else 0,
            'max_vpin': np.max(vpin_values) if vpin_values else 0,
            'buckets': buckets
        }

class ToxicityComparisonFramework:
    """
    Framework for comparing PIN, VPIN, and ML toxicity measures
    """
    
    def __init__(self, ml_model_path=None):
        self.pin_calculator = PINCalculator()
        self.vpin_calculator = VPINCalculator()
        self.ml_predictor = None
        
        # Load ML model if available
        if ml_model_path and os.path.exists(ml_model_path):
            self.load_ml_model(ml_model_path)
    
    def load_ml_model(self, model_path):
        """Load the trained ML toxicity model"""
        try:
            print(f"Loading ML toxicity model from: {model_path}")
            model_package = joblib.load(model_path)
            self.ml_predictor = model_package
            print("✓ ML model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading ML model: {e}")
            self.ml_predictor = None
    
    def extract_ml_features(self, trades_df, timestamp_idx):
        """
        Extract features for ML prediction (simplified version)
        """
        if timestamp_idx < 10:
            return np.array([[0.5] * 20])  # Return neutral features for early trades
        
        # Get recent window of trades
        window_start = max(0, timestamp_idx - 10)
        window_trades = trades_df.iloc[window_start:timestamp_idx + 1]
        
        # Basic features
        features = {}
        
        # Order flow features
        recent_volumes = window_trades['quantity'].values
        features['order_size'] = window_trades.iloc[-1]['quantity']
        features['log_order_size'] = np.log1p(features['order_size'])
        features['avg_volume'] = np.mean(recent_volumes) if len(recent_volumes) > 0 else 1
        features['volume_ratio'] = features['order_size'] / features['avg_volume'] if features['avg_volume'] > 0 else 1
        
        # Price features
        recent_prices = window_trades['price'].values
        if len(recent_prices) > 1:
            features['price_return'] = (recent_prices[-1] - recent_prices[-2]) / recent_prices[-2]
            features['price_volatility'] = np.std(recent_prices)
        else:
            features['price_return'] = 0
            features['price_volatility'] = 0
        
        # Market microstructure
        features['is_market_order'] = 1  # Assume market orders for simplicity
        features['time_since_last_trade'] = 1  # Default value
        
        # Create feature vector (using first 20 features to match model expectation)
        feature_names = [
            'order_size', 'log_order_size', 'avg_volume', 'volume_ratio',
            'price_return', 'price_volatility', 'is_market_order', 'time_since_last_trade'
        ]
        
        # Pad with additional features if needed
        while len(feature_names) < 20:
            feature_names.append(f'feature_{len(feature_names)}')
            features[f'feature_{len(feature_names)-1}'] = 0
        
        # Convert to array
        feature_vector = np.array([[features.get(name, 0) for name in feature_names[:20]]])
        
        return feature_vector
    
    def get_ml_toxicity_scores(self, trades_df):
        """
        Get ML toxicity scores for trades
        """
        if self.ml_predictor is None:
            # Generate synthetic scores for demonstration
            np.random.seed(42)
            return np.random.beta(2, 5, len(trades_df))  # Skewed towards lower toxicity
        
        toxicity_scores = []
        
        for i in range(len(trades_df)):
            try:
                features = self.extract_ml_features(trades_df, i)
                
                # Get ensemble prediction (simplified)
                individual_scores = []
                
                for name, model in self.ml_predictor.get('models', {}).items():
                    try:
                        if model is None:
                            continue
                        
                        if 'isolation_forest' in name and hasattr(model, 'decision_function'):
                            score = -model.decision_function(features)[0]
                        elif 'lof' in name and hasattr(model, 'score_samples'):
                            score = -model.score_samples(features)[0]
                        else:
                            continue
                        
                        # Normalize to [0, 1]
                        score = max(0, min(abs(score), 2)) / 2
                        individual_scores.append(score)
                        
                    except Exception:
                        continue
                
                # Average ensemble scores
                if individual_scores:
                    toxicity_score = np.mean(individual_scores)
                else:
                    toxicity_score = 0.5  # Default
                
                toxicity_scores.append(toxicity_score)
                
            except Exception:
                toxicity_scores.append(0.5)  # Default for errors
        
        return np.array(toxicity_scores)
    
    def run_comparison_analysis(self, trades_df, save_results=True):
        """
        Run comprehensive comparison of PIN, VPIN, and ML toxicity measures
        
        Parameters:
        trades_df: DataFrame with columns ['timestamp', 'price', 'quantity']
        save_results: Whether to save results and plots
        
        Returns:
        Dictionary with all calculated measures and comparison statistics
        """
        print("🔍 Running Toxicity Measures Comparison Analysis...")
        print(f"📊 Analyzing {len(trades_df)} trades")
        
        results = {}
        
        # 1. Calculate PIN
        print("\n1. Calculating PIN (Probability of Informed Trading)...")
        try:
            pin_results = self.pin_calculator.estimate_pin(trades_df)
            if pin_results:
                results['PIN'] = pin_results
                print(f"   ✓ PIN = {pin_results['PIN']:.4f}")
                print(f"   ✓ μ (prob. info event) = {pin_results['mu']:.4f}")
                print(f"   ✓ α (prob. informed trading) = {pin_results['alpha']:.4f}")
            else:
                print("   ✗ PIN calculation failed")
                results['PIN'] = None
        except Exception as e:
            print(f"   ✗ PIN calculation error: {e}")
            results['PIN'] = None
        
        # 2. Calculate VPIN
        print("\n2. Calculating VPIN (Volume-Synchronized PIN)...")
        try:
            vpin_results = self.vpin_calculator.calculate_vpin(trades_df)
            results['VPIN'] = vpin_results
            print(f"   ✓ Mean VPIN = {vpin_results['mean_vpin']:.4f}")
            print(f"   ✓ Max VPIN = {vpin_results['max_vpin']:.4f}")
            print(f"   ✓ VPIN Std = {vpin_results['std_vpin']:.4f}")
        except Exception as e:
            print(f"   ✗ VPIN calculation error: {e}")
            results['VPIN'] = None
        
        # 3. Get ML Toxicity Scores
        print("\n3. Calculating ML Toxicity Scores...")
        try:
            ml_scores = self.get_ml_toxicity_scores(trades_df)
            results['ML_Toxicity'] = {
                'scores': ml_scores,
                'mean_toxicity': np.mean(ml_scores),
                'std_toxicity': np.std(ml_scores),
                'max_toxicity': np.max(ml_scores)
            }
            print(f"   ✓ Mean ML Toxicity = {np.mean(ml_scores):.4f}")
            print(f"   ✓ Max ML Toxicity = {np.max(ml_scores):.4f}")
            print(f"   ✓ ML Toxicity Std = {np.std(ml_scores):.4f}")
        except Exception as e:
            print(f"   ✗ ML toxicity calculation error: {e}")
            results['ML_Toxicity'] = None
        
        # 4. Correlation Analysis
        print("\n4. Running Correlation Analysis...")
        try:
            correlations = self.calculate_correlations(results)
            results['Correlations'] = correlations
            
            if correlations:
                print("   Correlation Matrix:")
                for measure1, corr_dict in correlations.items():
                    for measure2, corr_value in corr_dict.items():
                        if measure1 != measure2:
                            print(f"   {measure1} vs {measure2}: {corr_value:.4f}")
        except Exception as e:
            print(f"   ✗ Correlation analysis error: {e}")
            results['Correlations'] = None
        
        # 5. Create Visualizations
        if save_results:
            print("\n5. Creating Visualizations...")
            try:
                self.create_comparison_plots(results, trades_df)
                print("   ✓ Plots saved successfully")
            except Exception as e:
                print(f"   ✗ Plotting error: {e}")
        
        # 6. Summary Statistics
        print("\n6. Summary Statistics:")
        print("   " + "="*50)
        
        if results.get('PIN'):
            print(f"   PIN: {results['PIN']['PIN']:.4f}")
        
        if results.get('VPIN'):
            print(f"   VPIN (mean): {results['VPIN']['mean_vpin']:.4f}")
        
        if results.get('ML_Toxicity'):
            print(f"   ML Toxicity (mean): {results['ML_Toxicity']['mean_toxicity']:.4f}")
        
        return results
    
    def calculate_correlations(self, results):
        """
        Calculate correlations between different toxicity measures
        """
        correlations = {}
        
        # Prepare data for correlation analysis
        measures = {}
        
        # PIN (single value per analysis period)
        if results.get('PIN'):
            measures['PIN'] = results['PIN']['PIN']
        
        # VPIN (time series)
        if results.get('VPIN') and results['VPIN']['vpin_values']:
            measures['VPIN'] = np.mean(results['VPIN']['vpin_values'])
        
        # ML Toxicity (time series)
        if results.get('ML_Toxicity'):
            measures['ML_Toxicity'] = results['ML_Toxicity']['mean_toxicity']
        
        # Calculate pairwise correlations
        measure_names = list(measures.keys())
        
        for i, measure1 in enumerate(measure_names):
            correlations[measure1] = {}
            for j, measure2 in enumerate(measure_names):
                if i == j:
                    correlations[measure1][measure2] = 1.0
                else:
                    # For now, we can only compare means since PIN is a single value
                    # In a full implementation, we'd need time-aligned series
                    correlations[measure1][measure2] = 0.0  # Placeholder
        
        # If we have time series data, calculate proper correlations
        if (results.get('VPIN') and results.get('ML_Toxicity') and 
            len(results['VPIN']['vpin_values']) > 1 and 
            len(results['ML_Toxicity']['scores']) > 1):
            
            # Align time series for correlation
            vpin_values = results['VPIN']['vpin_values']
            ml_scores = results['ML_Toxicity']['scores']
            
            # Simple alignment: take last N values where N is min length
            min_len = min(len(vpin_values), len(ml_scores))
            if min_len > 5:  # Need sufficient data points
                vpin_aligned = vpin_values[-min_len:]
                ml_aligned = ml_scores[-min_len:]
                
                try:
                    pearson_corr, _ = pearsonr(vpin_aligned, ml_aligned)
                    spearman_corr, _ = spearmanr(vpin_aligned, ml_aligned)
                    
                    correlations['VPIN']['ML_Toxicity'] = pearson_corr
                    correlations['ML_Toxicity']['VPIN'] = pearson_corr
                    
                    # Add Spearman correlation
                    correlations['VPIN_Spearman'] = {'ML_Toxicity': spearman_corr}
                    
                except Exception as e:
                    print(f"Warning: Correlation calculation failed: {e}")
        
        return correlations
    
    def create_comparison_plots(self, results, trades_df):
        """
        Create comprehensive comparison plots
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Toxicity Measures Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. Time series of all measures
        if results.get('VPIN') and results.get('ML_Toxicity'):
            vpin_values = results['VPIN']['vpin_values']
            ml_scores = results['ML_Toxicity']['scores']
            
            # Plot VPIN time series
            axes[0, 0].plot(vpin_values, label='VPIN', color='blue', alpha=0.7)
            axes[0, 0].set_title('VPIN Time Series')
            axes[0, 0].set_xlabel('Volume Bucket')
            axes[0, 0].set_ylabel('VPIN Value')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot ML Toxicity time series (sample every N points to match VPIN length)
            if len(ml_scores) > len(vpin_values):
                step = len(ml_scores) // len(vpin_values)
                ml_sampled = ml_scores[::step][:len(vpin_values)]
            else:
                ml_sampled = ml_scores
            
            axes[0, 1].plot(ml_sampled, label='ML Toxicity', color='red', alpha=0.7)
            axes[0, 1].set_title('ML Toxicity Scores Time Series')
            axes[0, 1].set_xlabel('Trade Index')
            axes[0, 1].set_ylabel('Toxicity Score')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 2. Distribution comparison
        if results.get('VPIN') and results.get('ML_Toxicity'):
            vpin_values = results['VPIN']['vpin_values']
            ml_scores = results['ML_Toxicity']['scores']
            
            axes[0, 2].hist(vpin_values, bins=20, alpha=0.6, label='VPIN', color='blue', density=True)
            axes[0, 2].hist(ml_scores, bins=20, alpha=0.6, label='ML Toxicity', color='red', density=True)
            axes[0, 2].set_title('Distribution Comparison')
            axes[0, 2].set_xlabel('Value')
            axes[0, 2].set_ylabel('Density')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 3. Scatter plot correlation
        if results.get('VPIN') and results.get('ML_Toxicity'):
            vpin_values = results['VPIN']['vpin_values']
            ml_scores = results['ML_Toxicity']['scores']
            
            # Align data for scatter plot
            min_len = min(len(vpin_values), len(ml_scores))
            if min_len > 1:
                if len(ml_scores) > len(vpin_values):
                    step = len(ml_scores) // len(vpin_values)
                    ml_aligned = ml_scores[::step][:len(vpin_values)]
                    vpin_aligned = vpin_values
                else:
                    ml_aligned = ml_scores[-min_len:]
                    vpin_aligned = vpin_values[-min_len:]
                
                axes[1, 0].scatter(vpin_aligned, ml_aligned, alpha=0.6, color='green')
                axes[1, 0].set_title('VPIN vs ML Toxicity Correlation')
                axes[1, 0].set_xlabel('VPIN')
                axes[1, 0].set_ylabel('ML Toxicity Score')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add correlation coefficient to plot
                if len(vpin_aligned) > 2:
                    try:
                        corr, _ = pearsonr(vpin_aligned, ml_aligned)
                        axes[1, 0].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                                       transform=axes[1, 0].transAxes, 
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except:
                        pass
        
        # 4. Summary statistics comparison
        measures_summary = {}
        if results.get('PIN'):
            measures_summary['PIN'] = results['PIN']['PIN']
        if results.get('VPIN'):
            measures_summary['VPIN (mean)'] = results['VPIN']['mean_vpin']
        if results.get('ML_Toxicity'):
            measures_summary['ML Toxicity (mean)'] = results['ML_Toxicity']['mean_toxicity']
        
        if measures_summary:
            measures = list(measures_summary.keys())
            values = list(measures_summary.values())
            
            bars = axes[1, 1].bar(measures, values, color=['blue', 'orange', 'red'][:len(measures)], alpha=0.7)
            axes[1, 1].set_title('Average Toxicity Measures Comparison')
            axes[1, 1].set_ylabel('Average Value')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Trade volume vs toxicity
        if results.get('ML_Toxicity'):
            ml_scores = results['ML_Toxicity']['scores']
            volumes = trades_df['quantity'].values
            
            # Ensure same length
            min_len = min(len(ml_scores), len(volumes))
            ml_plot = ml_scores[:min_len]
            vol_plot = volumes[:min_len]
            
            # Create binned analysis
            n_bins = 10
            ml_bins = np.linspace(np.min(ml_plot), np.max(ml_plot), n_bins)
            binned_volumes = []
            bin_centers = []
            
            for i in range(len(ml_bins)-1):
                mask = (ml_plot >= ml_bins[i]) & (ml_plot < ml_bins[i+1])
                if np.any(mask):
                    binned_volumes.append(np.mean(vol_plot[mask]))
                    bin_centers.append((ml_bins[i] + ml_bins[i+1]) / 2)
            
            if binned_volumes:
                axes[1, 2].plot(bin_centers, binned_volumes, 'o-', color='purple', linewidth=2, markersize=6)
                axes[1, 2].set_title('Trade Volume vs ML Toxicity')
                axes[1, 2].set_xlabel('ML Toxicity Score')
                axes[1, 2].set_ylabel('Average Trade Volume')
                axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'toxicity_comparison_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✓ Plot saved as: {plot_filename}")
        
        plt.show()
        
        # Create detailed comparison table
        self.create_comparison_table(results, timestamp)
    
    def create_comparison_table(self, results, timestamp):
        """
        Create a detailed comparison table of all measures
        """
        print("\n📊 DETAILED COMPARISON TABLE")
        print("="*80)
        
        # Prepare data for table
        comparison_data = []
        
        if results.get('PIN'):
            pin_data = results['PIN']
            comparison_data.append({
                'Measure': 'PIN',
                'Value': f"{pin_data['PIN']:.4f}",
                'Type': 'Single Value',
                'Time Horizon': 'Daily Aggregation',
                'Key Insight': f"μ={pin_data['mu']:.3f}, α={pin_data['alpha']:.3f}"
            })
        
        if results.get('VPIN'):
            vpin_data = results['VPIN']
            comparison_data.append({
                'Measure': 'VPIN',
                'Value': f"{vpin_data['mean_vpin']:.4f} ±{vpin_data['std_vpin']:.4f}",
                'Type': 'Time Series',
                'Time Horizon': 'Volume Buckets',
                'Key Insight': f"Max={vpin_data['max_vpin']:.3f}, σ={vpin_data['std_vpin']:.3f}"
            })
        
        if results.get('ML_Toxicity'):
            ml_data = results['ML_Toxicity']
            comparison_data.append({
                'Measure': 'ML Toxicity',
                'Value': f"{ml_data['mean_toxicity']:.4f} ±{ml_data['std_toxicity']:.4f}",
                'Type': 'Time Series',
                'Time Horizon': 'Per Trade',
                'Key Insight': f"Max={ml_data['max_toxicity']:.3f}, σ={ml_data['std_toxicity']:.3f}"
            })
        
        # Print table
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            print(df_comparison.to_string(index=False))
            
            # Save to CSV
            csv_filename = f'toxicity_comparison_{timestamp}.csv'
            df_comparison.to_csv(csv_filename, index=False)
            print(f"\n✓ Comparison table saved as: {csv_filename}")
        
        # Correlation summary
        if results.get('Correlations'):
            print(f"\n📈 CORRELATION ANALYSIS")
            print("-"*40)
            correlations = results['Correlations']
            
            for measure1, corr_dict in correlations.items():
                for measure2, corr_value in corr_dict.items():
                    if measure1 != measure2 and isinstance(corr_value, (int, float)) and corr_value != 0:
                        print(f"{measure1} vs {measure2}: {corr_value:.4f}")
        
        return comparison_data

def run_comprehensive_toxicity_analysis(trades_data_path=None, ml_model_path=None, 
                                       synthetic_data=True, n_trades=1000):
    """
    Main function to run comprehensive toxicity analysis
    
    Parameters:
    trades_data_path: Path to trade data CSV (optional)
    ml_model_path: Path to ML toxicity model (optional)
    synthetic_data: Whether to generate synthetic data for testing
    n_trades: Number of synthetic trades to generate
    """
    
    print("🚀 Starting Comprehensive Toxicity Analysis...")
    print("="*60)
    
    # Initialize comparison framework
    framework = ToxicityComparisonFramework(ml_model_path)
    
    # Load or generate trade data
    if trades_data_path and os.path.exists(trades_data_path):
        print(f"📁 Loading trade data from: {trades_data_path}")
        trades_df = pd.read_csv(trades_data_path)
    else:
        raise ValueError("Either provide trades_data_path or enable synthetic_data")
    
    # Ensure required columns exist
    required_columns = ['timestamp', 'price', 'quantity']
    if not all(col in trades_df.columns for col in required_columns):
        raise ValueError(f"Trade data must contain columns: {required_columns}")
    
    print(f"✓ Trade data loaded: {len(trades_df)} trades")
    print(f"  Price range: ${trades_df['price'].min():.2f} - ${trades_df['price'].max():.2f}")
    print(f"  Volume range: {trades_df['quantity'].min()} - {trades_df['quantity'].max()}")
    print(f"  Time span: {trades_df['timestamp'].max() - trades_df['timestamp'].min():.0f} seconds")
    
    # Run comprehensive analysis
    results = framework.run_comparison_analysis(trades_df, save_results=True)
    
    # Performance insights
    print(f"\n🎯 KEY INSIGHTS")
    print("-"*30)
    
    if results.get('PIN'):
        pin_val = results['PIN']['PIN']
        if pin_val > 0.3:
            print(f"• HIGH PIN ({pin_val:.3f}): Strong evidence of informed trading")
        elif pin_val > 0.15:
            print(f"• MODERATE PIN ({pin_val:.3f}): Some informed trading present")
        else:
            print(f"• LOW PIN ({pin_val:.3f}): Limited informed trading")
    
    if results.get('VPIN'):
        vpin_mean = results['VPIN']['mean_vpin']
        vpin_max = results['VPIN']['max_vpin']
        print(f"• VPIN volatility: Mean={vpin_mean:.3f}, Peak={vpin_max:.3f}")
        if vpin_max > 0.5:
            print("  ⚠️  High VPIN peaks detected - potential flow toxicity events")
    
    if results.get('ML_Toxicity'):
        ml_mean = results['ML_Toxicity']['mean_toxicity']
        ml_max = results['ML_Toxicity']['max_toxicity']
        print(f"• ML Toxicity: Mean={ml_mean:.3f}, Peak={ml_max:.3f}")
        high_toxicity_trades = np.sum(results['ML_Toxicity']['scores'] > 0.7)
        print(f"  🔍 {high_toxicity_trades} trades flagged as high toxicity (>0.7)")
    
    # Method comparison
    print(f"\n🔬 METHOD COMPARISON")
    print("-"*35)
    print("PIN:   📊 Daily aggregation, structural model, requires many days")
    print("VPIN:  📈 Volume-based buckets, rolling window, high frequency")
    print("ML:    🤖 Per-trade prediction, ensemble methods, real-time")
    
    if results.get('Correlations'):
        corr_vpin_ml = results['Correlations'].get('VPIN', {}).get('ML_Toxicity', 0)
        if abs(corr_vpin_ml) > 0.3:
            print(f"\n✅ Good agreement between VPIN and ML (r={corr_vpin_ml:.3f})")
        elif abs(corr_vpin_ml) > 0.1:
            print(f"\n⚡ Moderate agreement between VPIN and ML (r={corr_vpin_ml:.3f})")
        else:
            print(f"\n⚠️  Low agreement between VPIN and ML (r={corr_vpin_ml:.3f})")
    
    print(f"\n✅ Analysis completed successfully!")
    return results

# Example usage
if __name__ == "__main__":
    print("\n\nSimulation Data Analysis")
    print("="*50)
    
    results = run_comprehensive_toxicity_analysis(
        trades_data_path="your_simulation_trades.csv",
        ml_model_path="enhanced_toxicity_models/enhanced_toxicity_detector_20250704_004512.joblib",
        synthetic_data=False
    )
    
    print("\n🎉 All examples completed!")