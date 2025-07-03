import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

class MicrostructureValidation:
    """Advanced microstructure validation for academic thesis"""
    
    def __init__(self, market_environment):
        self.market = market_environment
        self.trades_df = market_environment.get_trades_dataframe()
        self.market_df = market_environment.get_market_dataframe()
        self.lob_df = pd.DataFrame(market_environment.lob_snapshots)
        
    def calculate_order_flow_imbalance_predictive_power(self, horizon_steps=10):
        """
        Calculate R-squared values for order flow imbalance predicting price movements
        """
        if self.lob_df.empty or len(self.lob_df) < horizon_steps + 1:
            return {'r_squared': 0.0, 'correlation': 0.0, 'sample_size': 0}
        
        # Calculate price changes at specified horizon
        prices = self.lob_df['mid_price'].values
        price_changes = []
        imbalances = []
        
        for i in range(len(prices) - horizon_steps):
            future_price = prices[i + horizon_steps]
            current_price = prices[i]
            price_change = (future_price - current_price) / current_price
            
            current_imbalance = self.lob_df.iloc[i]['imbalance']
            
            if not np.isnan(current_imbalance) and not np.isnan(price_change):
                price_changes.append(price_change)
                imbalances.append(current_imbalance)
        
        if len(price_changes) < 10:
            return {'r_squared': 0.0, 'correlation': 0.0, 'sample_size': len(price_changes)}
        
        # Calculate R-squared
        imbalances = np.array(imbalances).reshape(-1, 1)
        price_changes = np.array(price_changes)
        
        model = LinearRegression()
        model.fit(imbalances, price_changes)
        predictions = model.predict(imbalances)
        r_squared = r2_score(price_changes, predictions)
        
        # Calculate correlation
        correlation, _ = pearsonr(imbalances.flatten(), price_changes)
        
        return {
            'r_squared': r_squared,
            'correlation': correlation,
            'sample_size': len(price_changes),
            'coefficient': model.coef_[0] if len(model.coef_) > 0 else 0
        }
    
    def analyze_asymmetric_trade_impact(self):
        """
        Analyze asymmetric impact of buyer vs seller initiated trades
        """
        if self.trades_df.empty or len(self.trades_df) < 20:
            return {'buy_impact': 0.0, 'sell_impact': 0.0, 'asymmetry_ratio': 1.0}
        
        # Sort trades by timestamp
        trades_sorted = self.trades_df.sort_values('timestamp').copy()
        
        # Calculate price impact for each trade
        price_impacts = []
        trade_directions = []
        
        for i in range(len(trades_sorted) - 1):
            current_price = trades_sorted.iloc[i]['price']
            next_price = trades_sorted.iloc[i + 1]['price'] if i + 1 < len(trades_sorted) else current_price
            
            # Determine trade direction (simplified - market orders are directional)
            # In real analysis, would use Lee-Ready algorithm or similar
            price_impact = (next_price - current_price) / current_price
            
            # Classify as buy or sell based on whether informed trader was buyer
            trade_row = trades_sorted.iloc[i]
            is_buy_initiated = trade_row['buyer_type'] == 'informed' or (
                trade_row['buyer_type'] != 'market_maker' and trade_row['seller_type'] == 'market_maker'
            )
            
            if not np.isnan(price_impact) and abs(price_impact) < 0.1:  # Filter outliers
                price_impacts.append(abs(price_impact))
                trade_directions.append('buy' if is_buy_initiated else 'sell')
        
        if len(price_impacts) < 10:
            return {'buy_impact': 0.0, 'sell_impact': 0.0, 'asymmetry_ratio': 1.0}
        
        # Calculate average impact by direction
        impacts_df = pd.DataFrame({
            'impact': price_impacts,
            'direction': trade_directions
        })
        
        buy_impact = impacts_df[impacts_df['direction'] == 'buy']['impact'].mean()
        sell_impact = impacts_df[impacts_df['direction'] == 'sell']['impact'].mean()
        
        asymmetry_ratio = buy_impact / sell_impact if sell_impact > 0 else 1.0
        
        return {
            'buy_impact': buy_impact,
            'sell_impact': sell_impact,
            'asymmetry_ratio': asymmetry_ratio,
            'buy_count': len(impacts_df[impacts_df['direction'] == 'buy']),
            'sell_count': len(impacts_df[impacts_df['direction'] == 'sell'])
        }
    
    def calculate_temporal_clustering_coefficient(self, window_size=20):
        """
        Calculate temporal clustering coefficient for large trades
        """
        if self.trades_df.empty or len(self.trades_df) < window_size:
            return 0.0
        
        # Define large trades (top quartile)
        quantity_75th = self.trades_df['quantity'].quantile(0.75)
        large_trades = self.trades_df[self.trades_df['quantity'] >= quantity_75th].copy()
        
        if len(large_trades) < 5:
            return 0.0
        
        # Sort by timestamp
        large_trades = large_trades.sort_values('timestamp')
        
        # Calculate clustering coefficient
        clustering_scores = []
        
        for i in range(len(large_trades)):
            current_trade = large_trades.iloc[i]
            current_time = current_trade['timestamp']
            
            # Look at trades in surrounding window
            window_trades = large_trades[
                (large_trades['timestamp'] >= current_time - window_size) &
                (large_trades['timestamp'] <= current_time + window_size) &
                (large_trades['timestamp'] != current_time)
            ]
            
            if len(window_trades) > 0:
                # Count trades in same direction
                same_direction_count = 0
                for _, trade in window_trades.iterrows():
                    # Simplified direction classification
                    current_is_buy = (current_trade['buyer_type'] == 'informed' or 
                                    current_trade['buyer_type'] != 'market_maker')
                    trade_is_buy = (trade['buyer_type'] == 'informed' or 
                                  trade['buyer_type'] != 'market_maker')
                    
                    if current_is_buy == trade_is_buy:
                        same_direction_count += 1
                
                clustering_score = same_direction_count / len(window_trades)
                clustering_scores.append(clustering_score)
        
        return np.mean(clustering_scores) if clustering_scores else 0.0
    
    def generate_correlation_heatmap(self, save_path="microstructure_correlation_heatmap.png"):
        """
        Generate correlation heatmap of key microstructure variables
        """
        # Prepare microstructure variables
        if self.lob_df.empty or self.market_df.empty:
            print("Insufficient data for correlation analysis")
            return None
        
        # Merge relevant data
        micro_data = pd.DataFrame()
        
        # From LOB snapshots
        if len(self.lob_df) > 0:
            micro_data['spread'] = self.lob_df['spread']
            micro_data['imbalance'] = self.lob_df['imbalance']
            micro_data['mid_price'] = self.lob_df['mid_price']
            
            # Calculate depth (sum of first 3 levels)
            depth_cols = [col for col in self.lob_df.columns if 'size_' in col][:6]
            if depth_cols:
                micro_data['depth'] = self.lob_df[depth_cols].sum(axis=1, skipna=True)
        
        # Add market-level variables
        if len(self.market_df) > 0:
            # Align lengths
            min_length = min(len(micro_data), len(self.market_df))
            micro_data = micro_data.iloc[:min_length]
            market_subset = self.market_df.iloc[:min_length]
            
            micro_data['volume'] = market_subset['volume'].values
            
            # Calculate volatility (rolling standard deviation of returns)
            prices = market_subset['price'].values
            returns = np.diff(prices) / prices[:-1]
            volatility = pd.Series(returns).rolling(window=10, min_periods=1).std()
            micro_data['volatility'] = np.concatenate([[np.nan], volatility.values])
        
        # Remove rows with missing data
        micro_data = micro_data.dropna()
        
        if len(micro_data) < 10:
            print("Insufficient clean data for correlation analysis")
            return None
        
        # Calculate correlation matrix
        correlation_matrix = micro_data.corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={'shrink': 0.8})
        
        plt.title('Microstructure Variables Correlation Matrix\n(Simulation Results)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Variables', fontsize=12)
        plt.ylabel('Variables', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation_matrix
    
    def analyze_trade_size_clustering(self):
        """
        Analyze clustering patterns in trade sizes for institutional order splitting
        """
        if self.trades_df.empty or len(self.trades_df) < 50:
            return {'clustering_detected': False, 'avg_sequence_length': 0}
        
        # Sort trades by timestamp
        trades_sorted = self.trades_df.sort_values('timestamp').copy()
        
        # Define large trades (top 25%)
        large_threshold = trades_sorted['quantity'].quantile(0.75)
        
        # Find sequences of trades preceding large trades
        sequence_lengths = []
        
        for i, trade in trades_sorted.iterrows():
            if trade['quantity'] >= large_threshold:
                # Look backwards for preceding smaller trades in same direction
                sequence_length = 0
                
                # Determine direction of large trade
                large_is_buy = (trade['buyer_type'] == 'informed' or 
                              trade['buyer_type'] != 'market_maker')
                
                # Look at previous trades
                for j in range(max(0, i-10), i):
                    prev_trade = trades_sorted.iloc[j]
                    prev_is_buy = (prev_trade['buyer_type'] == 'informed' or 
                                 prev_trade['buyer_type'] != 'market_maker')
                    
                    # If same direction and smaller size
                    if (prev_is_buy == large_is_buy and 
                        prev_trade['quantity'] < trade['quantity']):
                        sequence_length += 1
                    else:
                        break  # Sequence broken
                
                sequence_lengths.append(sequence_length)
        
        avg_sequence_length = np.mean(sequence_lengths) if sequence_lengths else 0
        clustering_detected = avg_sequence_length > 1.5  # Threshold for clustering
        
        return {
            'clustering_detected': clustering_detected,
            'avg_sequence_length': avg_sequence_length,
            'num_large_trades': len(sequence_lengths),
            'max_sequence_length': max(sequence_lengths) if sequence_lengths else 0
        }
    
    def generate_validation_report(self, save_path="validation_report.txt"):
        """
        Generate comprehensive validation report for thesis
        """
        print("Generating microstructure validation analysis...")
        
        # Calculate all metrics
        imbalance_analysis = self.calculate_order_flow_imbalance_predictive_power()
        asymmetric_impact = self.analyze_asymmetric_trade_impact()
        clustering_coeff = self.calculate_temporal_clustering_coefficient()
        trade_clustering = self.analyze_trade_size_clustering()
        correlation_matrix = self.generate_correlation_heatmap()
        
        # Generate report
        report = []
        report.append("MICROSTRUCTURE VALIDATION ANALYSIS")
        report.append("=" * 50)
        report.append("")
        
        # Order flow imbalance analysis
        report.append("1. ORDER FLOW IMBALANCE PREDICTIVE POWER")
        report.append(f"   R-squared (10-step horizon): {imbalance_analysis['r_squared']:.4f}")
        report.append(f"   Correlation coefficient: {imbalance_analysis['correlation']:.4f}")
        report.append(f"   Sample size: {imbalance_analysis['sample_size']}")
        report.append(f"   Empirical benchmark: 0.12-0.18")
        
        # Assess if within empirical range
        if 0.12 <= imbalance_analysis['r_squared'] <= 0.18:
            report.append("   Status: WITHIN EMPIRICAL RANGE")
        else:
            report.append("   Status: Outside empirical range")
        report.append("")
        
        # Asymmetric trade impact
        report.append("2. ASYMMETRIC TRADE IMPACT ANALYSIS")
        report.append(f"   Buy-initiated impact: {asymmetric_impact['buy_impact']:.6f}")
        report.append(f"   Sell-initiated impact: {asymmetric_impact['sell_impact']:.6f}")
        report.append(f"   Asymmetry ratio: {asymmetric_impact['asymmetry_ratio']:.3f}")
        report.append(f"   Buy trades analyzed: {asymmetric_impact['buy_count']}")
        report.append(f"   Sell trades analyzed: {asymmetric_impact['sell_count']}")
        
        if asymmetric_impact['asymmetry_ratio'] > 1.0:
            report.append("   Result: Buy orders show higher impact (realistic)")
        else:
            report.append("   Result: Sell orders show higher impact")
        report.append("")
        
        # Temporal clustering
        report.append("3. TEMPORAL CLUSTERING COEFFICIENT")
        report.append(f"   Clustering coefficient: {clustering_coeff:.3f}")
        report.append(f"   Empirical benchmark: 0.20-0.30")
        
        if 0.20 <= clustering_coeff <= 0.30:
            report.append("   Status: WITHIN EMPIRICAL RANGE")
        else:
            report.append("   Status: Outside empirical range")
        report.append("")
        
        # Trade size clustering
        report.append("4. INSTITUTIONAL ORDER SPLITTING PATTERNS")
        report.append(f"   Average sequence length: {trade_clustering['avg_sequence_length']:.2f}")
        report.append(f"   Clustering detected: {trade_clustering['clustering_detected']}")
        report.append(f"   Large trades analyzed: {trade_clustering['num_large_trades']}")
        report.append(f"   Maximum sequence length: {trade_clustering['max_sequence_length']}")
        report.append("")
        
        # Overall assessment
        report.append("5. VALIDATION SUMMARY")
        validation_score = 0
        total_metrics = 0
        
        # Score each metric
        if 0.12 <= imbalance_analysis['r_squared'] <= 0.18:
            validation_score += 1
        total_metrics += 1
        
        if asymmetric_impact['asymmetry_ratio'] > 1.0:
            validation_score += 1
        total_metrics += 1
        
        if 0.20 <= clustering_coeff <= 0.30:
            validation_score += 1
        total_metrics += 1
        
        validation_percentage = (validation_score / total_metrics) * 100
        report.append(f"   Metrics within empirical ranges: {validation_score}/{total_metrics}")
        report.append(f"   Validation score: {validation_percentage:.1f}%")
        report.append("")
        
        if validation_percentage >= 66.7:
            report.append("   CONCLUSION: Simulation exhibits realistic microstructure properties")
            report.append("   suitable for ML model training with ground truth advantages.")
        else:
            report.append("   CONCLUSION: Simulation may require parameter adjustment for")
            report.append("   improved realism, though controlled environment benefits remain.")
        
        # Save report
        report_text = "\n".join(report)
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print("Validation report saved to:", save_path)
        print("\nKey Results:")
        print(f"Order flow R-squared: {imbalance_analysis['r_squared']:.4f}")
        print(f"Clustering coefficient: {clustering_coeff:.3f}")
        print(f"Overall validation: {validation_percentage:.1f}%")
        
        return {
            'imbalance_analysis': imbalance_analysis,
            'asymmetric_impact': asymmetric_impact,
            'clustering_coefficient': clustering_coeff,
            'trade_clustering': trade_clustering,
            'validation_score': validation_percentage
        }

def run_academic_validation(market_environment):
    """
    Run complete academic validation analysis
    """
    validator = MicrostructureValidation(market_environment)
    results = validator.generate_validation_report()
    
    return validator, results

# Import your market simulation
try:
    from BEST import run_market_simulation, MarketEnvironment
    print("Successfully imported from BEST.py")
except ImportError as e:
    print(f"Error importing from BEST.py: {e}")
    print("Please ensure BEST.py is in the same directory")

# Integration with your existing simulation
def enhanced_simulation_with_validation():
    """
    Run simulation from BEST.py and perform academic validation
    """
    try:
        # Run your existing simulation from BEST.py
        print("Running market simulation from BEST.py...")
        market = run_market_simulation(high_toxicity_mode=True, save_data=False)
        
        # Perform validation analysis
        print("\nPerforming academic microstructure validation...")
        validator, results = run_academic_validation(market)
        
        return market, validator, results
    
    except Exception as e:
        print(f"Error running simulation: {e}")
        return None, None, None

def validate_existing_simulation(market_environment):
    """
    Validate an already-run simulation from BEST.py
    """
    print("Performing academic microstructure validation on existing simulation...")
    validator, results = run_academic_validation(market_environment)
    return validator, results

if __name__ == "__main__":
    # Option 1: Run new simulation with validation
    print("Running enhanced simulation with academic validation...")
    market, validator, results = enhanced_simulation_with_validation()
    
    if market is not None:
        print("\nValidation completed successfully!")
        print(f"Validation score: {results['validation_score']:.1f}%")
    else:
        print("Simulation failed. Please check BEST.py file.")