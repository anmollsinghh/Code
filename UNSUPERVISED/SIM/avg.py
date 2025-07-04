# multi_run_comparison.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import the main comparison function
from newsim import run_ml_enhanced_comparison

def run_multiple_comparisons(n_runs=10, model_path=None, n_steps=1000, save_individual=False):
    """
    Run the ML-enhanced market maker comparison multiple times and aggregate results
    
    Args:
        n_runs: Number of simulation runs
        model_path: Path to the ML model file
        n_steps: Number of time steps per simulation
        save_individual: Whether to save individual run results
    
    Returns:
        dict: Aggregated results with means, standard deviations, and confidence intervals
    """
    
    print("="*80)
    print(f"MULTI-RUN ML MARKET MAKER COMPARISON STUDY")
    print(f"Running {n_runs} simulations with {n_steps} steps each")
    print("="*80)
    
    # Storage for all results
    all_results = defaultdict(lambda: defaultdict(list))
    individual_results = []
    
    # Default model path
    if model_path is None:
        model_path = "calibrated_toxicity_models/enhanced_toxicity_detector_20250704_004512.joblib"
    
    # Run multiple simulations
    for run_num in range(n_runs):
        print(f"\n{'='*20} RUN {run_num + 1}/{n_runs} {'='*20}")
        
        try:
            # Run single comparison
            start_time = time.time()
            results, environment = run_ml_enhanced_comparison(
                model_path=model_path,
                n_steps=n_steps
            )
            run_time = time.time() - start_time
            
            if results:
                # Store results for this run
                run_results = {}
                
                for algo_name, result in results.items():
                    metrics = result['performance_metrics']
                    
                    # Extract key metrics
                    run_data = {
                        'total_return_pct': metrics['total_return_pct'],
                        'avg_spread_bps': metrics['avg_spread_bps'],
                        'sharpe_ratio': result['sharpe_ratio'],
                        'adverse_selection_rate': metrics['adverse_selection_rate'],
                        'max_drawdown': result['max_drawdown'],
                        'total_volume_traded': metrics['total_volume_traded'],
                        'spread_efficiency': result['spread_efficiency'],
                        'toxicity_rate': result['toxicity_rate'],
                        'total_trades': metrics['total_trades'],
                        'spread_volatility': metrics.get('spread_volatility', 0),
                        'inventory_volatility': metrics.get('inventory_volatility', 0),
                        'avg_toxicity_score': metrics.get('avg_toxicity_score', 0)
                    }
                    
                    run_results[algo_name] = run_data
                    
                    # Add to aggregated results
                    for metric, value in run_data.items():
                        all_results[algo_name][metric].append(value)
                
                # Store individual run
                individual_results.append({
                    'run_number': run_num + 1,
                    'run_time': run_time,
                    'results': run_results
                })
                
                # Print progress
                print(f"✓ Run {run_num + 1} completed in {run_time:.1f}s")
                
                # Show quick results
                best_return = max(run_results.items(), key=lambda x: x[1]['total_return_pct'])
                print(f"  Best performer: {best_return[0].upper()} ({best_return[1]['total_return_pct']:.2f}%)")
                
            else:
                print(f"❌ Run {run_num + 1} failed - no results generated")
                
        except Exception as e:
            print(f"❌ Run {run_num + 1} failed with error: {e}")
            continue
    
    # Calculate aggregated statistics
    print(f"\n{'='*20} AGGREGATING RESULTS {'='*20}")
    
    aggregated_results = {}
    
    for algo_name in all_results.keys():
        algo_stats = {}
        
        for metric_name, values in all_results[algo_name].items():
            if values:  # Only if we have data
                values = np.array(values)
                
                algo_stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'count': len(values),
                    'ci_lower': np.percentile(values, 2.5),  # 95% CI
                    'ci_upper': np.percentile(values, 97.5),
                    'all_values': values.tolist()
                }
        
        aggregated_results[algo_name] = algo_stats
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual results if requested
    if save_individual and individual_results:
        individual_df = []
        for run_data in individual_results:
            for algo, metrics in run_data['results'].items():
                row = {'run_number': run_data['run_number'], 'algorithm': algo}
                row.update(metrics)
                individual_df.append(row)
        
        individual_df = pd.DataFrame(individual_df)
        individual_df.to_csv(f"multi_run_individual_results_{timestamp}.csv", index=False)
        print(f"✓ Individual results saved to: multi_run_individual_results_{timestamp}.csv")
    
    # Create and save summary statistics
    summary_stats = create_summary_statistics(aggregated_results, n_runs, timestamp)
    
    # Create visualisations
    create_multi_run_visualizations(aggregated_results, n_runs, timestamp)
    
    print(f"\n{'='*20} STUDY COMPLETED {'='*20}")
    print(f"✓ {len(individual_results)} successful runs out of {n_runs} attempts")
    print(f"✓ Results saved with timestamp: {timestamp}")
    
    return aggregated_results, individual_results, summary_stats

def create_summary_statistics(aggregated_results, n_runs, timestamp):
    """Create and save summary statistics table"""
    
    # Create summary DataFrame
    summary_data = []
    
    for algo_name, algo_stats in aggregated_results.items():
        row = {
            'Algorithm': algo_name.upper(),
            'Runs': algo_stats['total_return_pct']['count'],
            'Mean_Return_%': algo_stats['total_return_pct']['mean'],
            'Std_Return_%': algo_stats['total_return_pct']['std'],
            'Return_CI_Lower': algo_stats['total_return_pct']['ci_lower'],
            'Return_CI_Upper': algo_stats['total_return_pct']['ci_upper'],
            'Mean_Spread_bps': algo_stats['avg_spread_bps']['mean'],
            'Std_Spread_bps': algo_stats['avg_spread_bps']['std'],
            'Mean_Sharpe': algo_stats['sharpe_ratio']['mean'],
            'Std_Sharpe': algo_stats['sharpe_ratio']['std'],
            'Mean_Adverse_Selection_%': algo_stats['adverse_selection_rate']['mean'] * 100,
            'Std_Adverse_Selection_%': algo_stats['adverse_selection_rate']['std'] * 100,
            'Mean_Max_Drawdown_%': algo_stats['max_drawdown']['mean'],
            'Std_Max_Drawdown_%': algo_stats['max_drawdown']['std'],
            'Mean_Spread_Efficiency': algo_stats['spread_efficiency']['mean'],
            'Std_Spread_Efficiency': algo_stats['spread_efficiency']['std'],
            'Mean_Volume': algo_stats['total_volume_traded']['mean'],
            'Std_Volume': algo_stats['total_volume_traded']['std'],
            'Mean_Toxicity_Rate_%': algo_stats['toxicity_rate']['mean'] * 100,
            'Std_Toxicity_Rate_%': algo_stats['toxicity_rate']['std'] * 100
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary statistics
    summary_df.to_csv(f"multi_run_summary_statistics_{timestamp}.csv", index=False)
    
    # Print summary table
    print(f"\n{'='*20} SUMMARY STATISTICS {'='*20}")
    print(f"Based on {n_runs} simulation runs\n")
    
    # Print key metrics table
    print("MEAN PERFORMANCE COMPARISON:")
    print("-" * 80)
    print(f"{'Algorithm':<15} {'Return%':<10} {'±Std':<8} {'Spread':<10} {'Sharpe':<8} {'Adverse%':<10}")
    print("-" * 80)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Algorithm']:<15} "
              f"{row['Mean_Return_%']:<9.2f} "
              f"±{row['Std_Return_%']:<7.2f} "
              f"{row['Mean_Spread_bps']:<9.1f} "
              f"{row['Mean_Sharpe']:<7.2f} "
              f"{row['Mean_Adverse_Selection_%']:<9.1f}")
    
    print("\n95% CONFIDENCE INTERVALS FOR RETURNS:")
    print("-" * 60)
    for _, row in summary_df.iterrows():
        print(f"{row['Algorithm']:<15} [{row['Return_CI_Lower']:<6.2f}%, {row['Return_CI_Upper']:<6.2f}%]")
    
    # Statistical significance test (simplified)
    print(f"\nSTATISTICAL ANALYSIS:")
    print("-" * 25)
    
    # Find best performing algorithm
    best_algo = summary_df.loc[summary_df['Mean_Return_%'].idxmax()]
    print(f"🏆 Best Mean Return: {best_algo['Algorithm']} ({best_algo['Mean_Return_%']:.2f}%)")
    
    # Check if differences are significant (simplified test)
    returns_std = summary_df['Mean_Return_%'].std()
    if returns_std > 0.5:  # If standard deviation of means > 0.5%
        print(f"📊 Significant performance differences detected (σ = {returns_std:.2f}%)")
    else:
        print(f"📊 Performance differences may not be statistically significant (σ = {returns_std:.2f}%)")
    
    # Most consistent performer
    most_consistent = summary_df.loc[summary_df['Std_Return_%'].idxmin()]
    print(f"🎯 Most Consistent: {most_consistent['Algorithm']} (σ = {most_consistent['Std_Return_%']:.2f}%)")
    
    # Best risk-adjusted
    best_sharpe = summary_df.loc[summary_df['Mean_Sharpe'].idxmax()]
    print(f"📈 Best Risk-Adjusted: {best_sharpe['Algorithm']} (Sharpe = {best_sharpe['Mean_Sharpe']:.2f})")
    
    return summary_df

def create_multi_run_visualizations(aggregated_results, n_runs, timestamp):
    """Create comprehensive visualizations for multi-run results"""
    
    # Set up plotting parameters
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': (20, 16),
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 9
    })
    
    algorithms = list(aggregated_results.keys())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'][:len(algorithms)]
    
    # Create comprehensive figure
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    
    # 1. Return Distribution (Box Plot)
    return_data = [aggregated_results[algo]['total_return_pct']['all_values'] for algo in algorithms]
    bp = axes[0, 0].boxplot(return_data, labels=[algo.upper() for algo in algorithms], 
                            patch_artist=True, notch=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[0, 0].set_title(f'Return Distribution Across {n_runs} Runs', fontweight='bold')
    axes[0, 0].set_ylabel('Return (%)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Mean Returns with Error Bars
    means = [aggregated_results[algo]['total_return_pct']['mean'] for algo in algorithms]
    stds = [aggregated_results[algo]['total_return_pct']['std'] for algo in algorithms]
    
    bars = axes[0, 1].bar(algorithms, means, yerr=stds, capsize=5, 
                         color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Mean Returns with Standard Deviation', fontweight='bold')
    axes[0, 1].set_ylabel('Return (%)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, height + std + 0.1,
                       f'{mean:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Sharpe Ratio Comparison
    sharpe_means = [aggregated_results[algo]['sharpe_ratio']['mean'] for algo in algorithms]
    sharpe_stds = [aggregated_results[algo]['sharpe_ratio']['std'] for algo in algorithms]
    
    bars = axes[0, 2].bar(algorithms, sharpe_means, yerr=sharpe_stds, capsize=5,
                         color=colors, alpha=0.7, edgecolor='black')
    axes[0, 2].set_title('Mean Sharpe Ratios', fontweight='bold')
    axes[0, 2].set_ylabel('Sharpe Ratio')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Spread Efficiency vs Return Scatter
    spread_eff_means = [aggregated_results[algo]['spread_efficiency']['mean'] for algo in algorithms]
    return_means = [aggregated_results[algo]['total_return_pct']['mean'] for algo in algorithms]
    
    scatter = axes[1, 0].scatter(spread_eff_means, return_means, c=colors, s=200, alpha=0.7, edgecolors='black')
    
    for i, algo in enumerate(algorithms):
        axes[1, 0].annotate(algo.upper(), (spread_eff_means[i], return_means[i]),
                           xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    axes[1, 0].set_title('Spread Efficiency vs Return', fontweight='bold')
    axes[1, 0].set_xlabel('Spread Efficiency')
    axes[1, 0].set_ylabel('Return (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Adverse Selection Rate
    adverse_means = [aggregated_results[algo]['adverse_selection_rate']['mean'] * 100 for algo in algorithms]
    adverse_stds = [aggregated_results[algo]['adverse_selection_rate']['std'] * 100 for algo in algorithms]
    
    bars = axes[1, 1].bar(algorithms, adverse_means, yerr=adverse_stds, capsize=5,
                         color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Adverse Selection Rate', fontweight='bold')
    axes[1, 1].set_ylabel('Adverse Selection Rate (%)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Risk-Return Scatter with Error Bars
    axes[1, 2].errorbar(sharpe_means, return_means, xerr=sharpe_stds, yerr=stds,
                       fmt='o', capsize=5, capthick=2, markersize=8, alpha=0.7)
    
    for i, (algo, color) in enumerate(zip(algorithms, colors)):
        axes[1, 2].scatter(sharpe_means[i], return_means[i], c=[color], s=150, alpha=0.8, edgecolors='black')
        axes[1, 2].annotate(algo.upper(), (sharpe_means[i], return_means[i]),
                           xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    axes[1, 2].set_title('Risk-Return Profile with Uncertainty', fontweight='bold')
    axes[1, 2].set_xlabel('Sharpe Ratio')
    axes[1, 2].set_ylabel('Return (%)')
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Consistency Analysis (Coefficient of Variation)
    cv_return = [aggregated_results[algo]['total_return_pct']['std'] / 
                abs(aggregated_results[algo]['total_return_pct']['mean']) 
                for algo in algorithms]
    
    bars = axes[2, 0].bar(algorithms, cv_return, color=colors, alpha=0.7, edgecolor='black')
    axes[2, 0].set_title('Return Consistency (Lower = More Consistent)', fontweight='bold')
    axes[2, 0].set_ylabel('Coefficient of Variation')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].tick_params(axis='x', rotation=45)
    
    # 8. Volume Traded
    volume_means = [aggregated_results[algo]['total_volume_traded']['mean'] for algo in algorithms]
    volume_stds = [aggregated_results[algo]['total_volume_traded']['std'] for algo in algorithms]
    
    bars = axes[2, 1].bar(algorithms, volume_means, yerr=volume_stds, capsize=5,
                         color=colors, alpha=0.7, edgecolor='black')
    axes[2, 1].set_title('Mean Volume Traded', fontweight='bold')
    axes[2, 1].set_ylabel('Volume')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].tick_params(axis='x', rotation=45)
    
    # 9. Performance Ranking Summary
    axes[2, 2].axis('off')
    
    # Create ranking based on mean returns
    algo_returns = [(algo, aggregated_results[algo]['total_return_pct']['mean']) 
                   for algo in algorithms]
    ranked_algos = sorted(algo_returns, key=lambda x: x[1], reverse=True)
    
    ranking_text = f"PERFORMANCE RANKING\n"
    ranking_text += f"Based on {n_runs} simulation runs\n"
    ranking_text += "=" * 30 + "\n\n"
    
    for i, (algo, return_val) in enumerate(ranked_algos, 1):
        std_val = aggregated_results[algo]['total_return_pct']['std']
        ranking_text += f"{i}. {algo.upper()}\n"
        ranking_text += f"   Mean: {return_val:.2f}%\n"
        ranking_text += f"   Std:  ±{std_val:.2f}%\n\n"
    
    # Add statistical note
    ranking_text += "STATISTICAL NOTES:\n"
    ranking_text += f"• Sample size: {n_runs} runs\n"
    ranking_text += f"• 95% confidence intervals shown\n"
    ranking_text += f"• Results may vary with different\n"
    ranking_text += f"  random seeds and market conditions"
    
    axes[2, 2].text(0.05, 0.95, ranking_text, transform=axes[2, 2].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(f'Multi-Run ML Algorithm Comparison Results\n'
                f'{n_runs} Simulation Runs - {timestamp}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save plot
    save_dir = "multi_run_plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.savefig(f"{save_dir}/multi_run_comparison_{timestamp}.png", 
               dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Multi-run visualizations saved to: {save_dir}/multi_run_comparison_{timestamp}.png")
    
    plt.show()

def main():
    """Main function to run the multi-run comparison study"""
    
    print("🚀 Starting Multi-Run ML Market Maker Comparison Study...")
    
    # Configuration
    N_RUNS = 10
    N_STEPS = 1000
    MODEL_PATH = "calibrated_toxicity_models/enhanced_toxicity_detector_20250704_004512.joblib"
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Model file not found: {MODEL_PATH}")
        print("Proceeding with fallback mode (no ML model)")
        MODEL_PATH = None
    
    try:
        # Run multiple comparisons
        aggregated_results, individual_results, summary_stats = run_multiple_comparisons(
            n_runs=N_RUNS,
            model_path=MODEL_PATH,
            n_steps=N_STEPS,
            save_individual=True
        )
        
        if aggregated_results:
            print("\n✅ Multi-run comparison study completed successfully!")
            print(f"✅ {len(individual_results)} successful runs completed")
            print("✅ Summary statistics and visualizations generated")
            
            # Print final recommendations
            print(f"\n🎯 FINAL RECOMMENDATIONS:")
            print("-" * 30)
            
            # Best overall performer
            best_algo = max(aggregated_results.items(), 
                          key=lambda x: x[1]['total_return_pct']['mean'])
            
            print(f"🏆 Best Overall Algorithm: {best_algo[0].upper()}")
            print(f"   Mean Return: {best_algo[1]['total_return_pct']['mean']:.2f}%")
            print(f"   Std Deviation: ±{best_algo[1]['total_return_pct']['std']:.2f}%")
            
            # Most consistent
            most_consistent = min(aggregated_results.items(),
                                key=lambda x: x[1]['total_return_pct']['std'])
            
            print(f"🎯 Most Consistent Algorithm: {most_consistent[0].upper()}")
            print(f"   Standard Deviation: ±{most_consistent[1]['total_return_pct']['std']:.2f}%")
            
            # Best risk-adjusted
            best_sharpe = max(aggregated_results.items(),
                            key=lambda x: x[1]['sharpe_ratio']['mean'])
            
            print(f"📈 Best Risk-Adjusted Algorithm: {best_sharpe[0].upper()}")
            print(f"   Mean Sharpe Ratio: {best_sharpe[1]['sharpe_ratio']['mean']:.2f}")
            
        else:
            print("❌ No successful runs completed")
            
    except Exception as e:
        print(f"❌ Multi-run comparison failed: {e}")
        print("Check that all dependencies are installed and model file exists")

if __name__ == "__main__":
    main()