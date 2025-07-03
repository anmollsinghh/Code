import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import os

def load_lobster_data(message_file, orderbook_file):
    """
    Load and process LOBSTER data files
    """
    print("Loading LOBSTER data...")
    
    # Load message file
    message_cols = ['Time', 'Type', 'Order_ID', 'Size', 'Price', 'Direction']
    message_df = pd.read_csv(message_file, names=message_cols)
    
    # Load orderbook file (10 levels = 40 columns)
    orderbook_cols = []
    for level in range(1, 11):
        orderbook_cols.extend([f'Ask_Price_{level}', f'Ask_Size_{level}', 
                              f'Bid_Price_{level}', f'Bid_Size_{level}'])
    
    orderbook_df = pd.read_csv(orderbook_file, names=orderbook_cols)
    
    # Convert prices from LOBSTER format (price * 10000) to dollars
    message_df['Price'] = message_df['Price'] / 10000
    for level in range(1, 11):
        orderbook_df[f'Ask_Price_{level}'] = orderbook_df[f'Ask_Price_{level}'] / 10000
        orderbook_df[f'Bid_Price_{level}'] = orderbook_df[f'Bid_Price_{level}'] / 10000
    
    # Calculate mid price and spread
    orderbook_df['Mid_Price'] = (orderbook_df['Ask_Price_1'] + orderbook_df['Bid_Price_1']) / 2
    orderbook_df['Spread'] = orderbook_df['Ask_Price_1'] - orderbook_df['Bid_Price_1']
    orderbook_df['Spread_bps'] = (orderbook_df['Spread'] / orderbook_df['Mid_Price']) * 10000
    
    # Filter out invalid data
    valid_mask = (orderbook_df['Ask_Price_1'] > 0) & (orderbook_df['Bid_Price_1'] > 0)
    orderbook_df = orderbook_df[valid_mask].copy()
    message_df = message_df.iloc[:len(orderbook_df)].copy()
    
    print(f"Loaded {len(message_df)} messages and {len(orderbook_df)} orderbook snapshots")
    return message_df, orderbook_df

def load_simulation_data(data_dir="market_data"):
    """
    Load simulation data for comparison
    """
    print("Loading simulation data...")
    
    files = os.listdir(data_dir)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    # Extract timestamp
    sample_file = csv_files[0]
    parts = sample_file.split('_')
    if len(parts) >= 3:
        timestamp = '_'.join(parts[-2:]).replace('.csv', '')
    else:
        timestamp = parts[-1].replace('.csv', '')
    
    # Load simulation files
    trades_df = pd.read_csv(f"{data_dir}/trades_{timestamp}.csv")
    market_df = pd.read_csv(f"{data_dir}/market_stats_{timestamp}.csv")
    lob_df = pd.read_csv(f"{data_dir}/lob_snapshots_{timestamp}.csv")
    
    # Calculate additional metrics
    market_df['returns'] = market_df['price'].pct_change().dropna()
    market_df['spread_bps'] = (market_df['spread'] / market_df['price']) * 10000
    
    print(f"Loaded simulation: {len(trades_df)} trades, {len(market_df)} market snapshots")
    return trades_df, market_df, lob_df

def calculate_lobster_metrics(message_df, orderbook_df):
    """
    Calculate key metrics from LOBSTER data
    """
    print("Calculating LOBSTER metrics...")
    
    metrics = {}
    
    # 1. Spread statistics
    valid_spreads = orderbook_df['Spread_bps'][orderbook_df['Spread_bps'] > 0]
    metrics['median_spread_bps'] = valid_spreads.median()
    metrics['mean_spread_bps'] = valid_spreads.mean()
    metrics['spread_95th_percentile'] = valid_spreads.quantile(0.95)
    metrics['spread_std'] = valid_spreads.std()
    
    # 2. Return statistics
    returns = orderbook_df['Mid_Price'].pct_change().dropna()
    metrics['return_kurtosis'] = stats.kurtosis(returns, fisher=True)
    metrics['return_skewness'] = stats.skew(returns)
    metrics['return_volatility'] = returns.std()
    
    # 3. Trade size statistics
    trades = message_df[message_df['Type'].isin([4, 5])]  # Executions only
    if len(trades) > 0:
        metrics['mean_trade_size'] = trades['Size'].mean()
        metrics['trade_size_cv'] = trades['Size'].std() / trades['Size'].mean()
        metrics['trade_size_median'] = trades['Size'].median()
    
    # 4. Volume autocorrelation
    if len(trades) > 10:
        volume_series = trades['Size'].reset_index(drop=True)
        if len(volume_series) > 1:
            metrics['volume_autocorr_lag1'] = volume_series.autocorr(lag=1)
    
    # 5. Price impact estimation (simplified Kyle lambda)
    if len(trades) > 50:
        # Calculate price changes around trades
        trade_times = trades.index
        price_changes = []
        trade_sizes = []
        
        for i, idx in enumerate(trade_times[:100]):  # Sample first 100 trades
            if idx < len(orderbook_df) - 5:
                pre_price = orderbook_df['Mid_Price'].iloc[max(0, idx-1)]
                post_price = orderbook_df['Mid_Price'].iloc[min(len(orderbook_df)-1, idx+2)]
                price_change = abs(post_price - pre_price) / pre_price
                
                if price_change > 0:
                    price_changes.append(price_change)
                    trade_sizes.append(trades['Size'].iloc[i])
        
        if len(price_changes) > 10:
            # Simple linear regression for price impact
            price_changes = np.array(price_changes)
            trade_sizes = np.array(trade_sizes)
            
            if trade_sizes.std() > 0:
                correlation = np.corrcoef(trade_sizes, price_changes)[0, 1]
                slope = correlation * (np.std(price_changes) / np.std(trade_sizes))
                metrics['price_impact_slope'] = abs(slope)
    
    # 6. Market depth analysis
    total_depth_bid = orderbook_df[[f'Bid_Size_{i}' for i in range(1, 6)]].sum(axis=1)
    total_depth_ask = orderbook_df[[f'Ask_Size_{i}' for i in range(1, 6)]].sum(axis=1)
    total_depth = total_depth_bid + total_depth_ask
    
    # Correlation between depth and volatility
    if len(returns) == len(total_depth):
        rolling_vol = returns.rolling(window=50).std()
        depth_vol_corr = total_depth.corr(rolling_vol)
        if not np.isnan(depth_vol_corr):
            metrics['depth_volatility_corr'] = depth_vol_corr
    
    return metrics

def calculate_simulation_metrics(trades_df, market_df, lob_df):
    """
    Calculate corresponding metrics from simulation data
    """
    print("Calculating simulation metrics...")
    
    metrics = {}
    
    # 1. Spread statistics
    valid_spreads = market_df['spread_bps'][market_df['spread_bps'] > 0]
    metrics['median_spread_bps'] = valid_spreads.median()
    metrics['mean_spread_bps'] = valid_spreads.mean()
    metrics['spread_95th_percentile'] = valid_spreads.quantile(0.95)
    metrics['spread_std'] = valid_spreads.std()
    
    # 2. Return statistics
    returns = market_df['returns'].dropna()
    metrics['return_kurtosis'] = stats.kurtosis(returns, fisher=True)
    metrics['return_skewness'] = stats.skew(returns)
    metrics['return_volatility'] = returns.std()
    
    # 3. Trade size statistics
    metrics['mean_trade_size'] = trades_df['quantity'].mean()
    metrics['trade_size_cv'] = trades_df['quantity'].std() / trades_df['quantity'].mean()
    metrics['trade_size_median'] = trades_df['quantity'].median()
    
    # 4. Volume autocorrelation
    volume_series = trades_df['quantity']
    if len(volume_series) > 1:
        metrics['volume_autocorr_lag1'] = volume_series.autocorr(lag=1)
    
    # 5. Price impact estimation
    if len(trades_df) > 50:
        # Calculate price changes around trades
        price_changes = trades_df['price'].pct_change().abs().dropna()
        trade_sizes = trades_df['quantity'].iloc[1:]  # Align with price changes
        
        if len(price_changes) > 10 and len(price_changes) == len(trade_sizes):
            if trade_sizes.std() > 0:
                correlation = np.corrcoef(trade_sizes, price_changes)[0, 1]
                slope = correlation * (np.std(price_changes) / np.std(trade_sizes))
                metrics['price_impact_slope'] = abs(slope)
    
    # 6. Market depth analysis (if available)
    if 'bid_size_1' in lob_df.columns and 'ask_size_1' in lob_df.columns:
        # Sum first 5 levels if available
        bid_cols = [col for col in lob_df.columns if 'bid_size' in col][:5]
        ask_cols = [col for col in lob_df.columns if 'ask_size' in col][:5]
        
        total_depth = lob_df[bid_cols + ask_cols].sum(axis=1)
        
        # Correlation with volatility
        rolling_vol = returns.rolling(window=50).std()
        if len(total_depth) == len(rolling_vol):
            depth_vol_corr = total_depth.corr(rolling_vol)
            if not np.isnan(depth_vol_corr):
                metrics['depth_volatility_corr'] = depth_vol_corr
    
    return metrics

def create_comparison_table(lobster_metrics, sim_metrics):
    """
    Create comparison table and calculate differences
    """
    print("Creating comparison table...")
    
    # Common metrics to compare
    common_metrics = [
        'median_spread_bps',
        'return_kurtosis', 
        'trade_size_cv',
        'volume_autocorr_lag1',
        'price_impact_slope'
    ]
    
    comparison_data = []
    
    for metric in common_metrics:
        if metric in lobster_metrics and metric in sim_metrics:
            lobster_val = lobster_metrics[metric]
            sim_val = sim_metrics[metric]
            difference_pct = ((sim_val - lobster_val) / lobster_val) * 100
            
            comparison_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Simulation': f"{sim_val:.3f}",
                'Real Market (LOBSTER)': f"{lobster_val:.3f}",
                'Difference (%)': f"{difference_pct:+.1f}%"
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

def plot_comparison_analysis(lobster_metrics, sim_metrics, message_df, orderbook_df, 
                           trades_df, market_df, save_dir="plots2"):
    """
    Create comprehensive comparison plots
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("Generating comparison plots...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Spread Distribution Comparison
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    lobster_spreads = orderbook_df['Spread_bps'][orderbook_df['Spread_bps'] > 0]
    lobster_spreads = lobster_spreads[lobster_spreads < 50]  # Remove outliers
    plt.hist(lobster_spreads, bins=50, alpha=0.7, label='LOBSTER (Real)', color='blue', density=True)
    
    sim_spreads = market_df['spread_bps'][market_df['spread_bps'] > 0]
    sim_spreads = sim_spreads[sim_spreads < 50]
    plt.hist(sim_spreads, bins=50, alpha=0.7, label='Simulation', color='red', density=True)
    
    plt.xlabel('Spread (basis points)')
    plt.ylabel('Density')
    plt.title('Spread Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Box plot comparison
    data_to_plot = [lobster_spreads.values, sim_spreads.values]
    plt.boxplot(data_to_plot, labels=['LOBSTER', 'Simulation'])
    plt.ylabel('Spread (basis points)')
    plt.title('Spread Distribution Box Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/spread_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Return Distribution Comparison
    plt.figure(figsize=(14, 6))
    
    lobster_returns = orderbook_df['Mid_Price'].pct_change().dropna()
    lobster_returns = lobster_returns[np.abs(lobster_returns) < 0.01]  # Remove outliers
    
    sim_returns = market_df['returns'].dropna()
    sim_returns = sim_returns[np.abs(sim_returns) < 0.01]
    
    plt.subplot(1, 2, 1)
    plt.hist(lobster_returns, bins=50, alpha=0.7, label='LOBSTER (Real)', color='blue', density=True)
    plt.hist(sim_returns, bins=50, alpha=0.7, label='Simulation', color='red', density=True)
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.title('Return Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Q-Q plot
    plt.subplot(1, 2, 2)
    # Sample to same size for fair comparison
    min_len = min(len(lobster_returns), len(sim_returns))
    lobster_sample = np.random.choice(lobster_returns, min_len, replace=False)
    sim_sample = np.random.choice(sim_returns, min_len, replace=False)
    
    plt.scatter(np.sort(lobster_sample), np.sort(sim_sample), alpha=0.5)
    plt.plot([lobster_returns.min(), lobster_returns.max()], 
             [lobster_returns.min(), lobster_returns.max()], 'r-', lw=2)
    plt.xlabel('LOBSTER Returns (quantiles)')
    plt.ylabel('Simulation Returns (quantiles)')
    plt.title('Q-Q Plot: Returns Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/returns_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Trade Size Comparison
    plt.figure(figsize=(12, 6))
    
    lobster_trades = message_df[message_df['Type'].isin([4, 5])]
    
    plt.hist(lobster_trades['Size'], bins=30, alpha=0.7, label='LOBSTER (Real)', 
             color='blue', density=True)
    plt.hist(trades_df['quantity'], bins=30, alpha=0.7, label='Simulation', 
             color='red', density=True)
    
    plt.xlabel('Trade Size')
    plt.ylabel('Density')
    plt.title('Trade Size Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/trade_size_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Metrics Comparison Bar Chart
    plt.figure(figsize=(12, 8))
    
    metrics_comparison = [
        ('Median Spread (bps)', lobster_metrics.get('median_spread_bps', 0), 
         sim_metrics.get('median_spread_bps', 0)),
        ('Return Kurtosis', lobster_metrics.get('return_kurtosis', 0), 
         sim_metrics.get('return_kurtosis', 0)),
        ('Trade Size CV', lobster_metrics.get('trade_size_cv', 0), 
         sim_metrics.get('trade_size_cv', 0)),
        ('Volume Autocorr', lobster_metrics.get('volume_autocorr_lag1', 0), 
         sim_metrics.get('volume_autocorr_lag1', 0))
    ]
    
    x = np.arange(len(metrics_comparison))
    width = 0.35
    
    lobster_vals = [m[1] for m in metrics_comparison]
    sim_vals = [m[2] for m in metrics_comparison]
    
    plt.bar(x - width/2, lobster_vals, width, label='LOBSTER (Real)', color='blue', alpha=0.7)
    plt.bar(x + width/2, sim_vals, width, label='Simulation', color='red', alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Key Metrics Comparison')
    plt.xticks(x, [m[0] for m in metrics_comparison], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/metrics_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {save_dir}/")

def main():
    """
    Main comparison analysis function
    """
    print("="*80)
    print("MARKET SIMULATION vs REAL DATA (LOBSTER) COMPARISON")
    print("="*80)
    
    # File paths - UPDATE THESE TO YOUR ACTUAL FILE PATHS
    message_file = "/Users/as/Downloads/LOBSTER_SampleFile_AMZN_2012-06-21_10/AMZN_2012-06-21_34200000_57600000_message_10.csv"
    orderbook_file = "/Users/as/Downloads/LOBSTER_SampleFile_AMZN_2012-06-21_10/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    
    try:
        # Load data
        message_df, orderbook_df = load_lobster_data(message_file, orderbook_file)
        trades_df, market_df, lob_df = load_simulation_data("market_data")
        
        # Calculate metrics
        lobster_metrics = calculate_lobster_metrics(message_df, orderbook_df)
        sim_metrics = calculate_simulation_metrics(trades_df, market_df, lob_df)
        
        # Create comparison table
        comparison_table = create_comparison_table(lobster_metrics, sim_metrics)
        
        # Display results
        print("\nCOMPARISON TABLE:")
        print("="*60)
        print(comparison_table.to_string(index=False))
        
        print("\nDETAILED LOBSTER METRICS:")
        print("-"*40)
        for key, value in lobster_metrics.items():
            print(f"{key:25}: {value:.6f}")
            
        print("\nDETAILED SIMULATION METRICS:")
        print("-"*40)
        for key, value in sim_metrics.items():
            print(f"{key:25}: {value:.6f}")
        
        # Generate plots
        plot_comparison_analysis(lobster_metrics, sim_metrics, message_df, orderbook_df,
                               trades_df, market_df)
        
        # Save comparison table
        comparison_table.to_csv("lobster_simulation_comparison.csv", index=False)
        print(f"\nComparison table saved to: lobster_simulation_comparison.csv")
        
        # Calculate overall similarity score
        differences = []
        for _, row in comparison_table.iterrows():
            diff_str = row['Difference (%)'].replace('%', '').replace('+', '')
            try:
                diff_val = abs(float(diff_str))
                differences.append(diff_val)
            except:
                continue
        
        if differences:
            avg_difference = np.mean(differences)
            similarity_score = max(0, 100 - avg_difference)
            print(f"\nOVERALL SIMILARITY SCORE: {similarity_score:.1f}%")
            print(f"Average absolute difference: {avg_difference:.1f}%")
            
            if similarity_score >= 80:
                print("✅ EXCELLENT similarity to real market data")
            elif similarity_score >= 60:
                print("✅ GOOD similarity to real market data")
            elif similarity_score >= 40:
                print("⚠️ MODERATE similarity to real market data")
            else:
                print("❌ LOW similarity to real market data")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Please ensure LOBSTER files are in the current directory")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()