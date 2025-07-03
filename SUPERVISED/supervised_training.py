"""
Supervised Model Benchmarking for Toxicity Detection
supervised_training.py
===================================================

Section 4.4: Benchmarking with Supervised Models
- Ground-truth labelling methodology
- Supervised model selection and training pipeline  
- Evaluation and insights from supervised approach
- ML-enhanced market maker implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve, CalibrationDisplay
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
import joblib
import glob
import os

class ToxicityLabeller:
    """
    4.4.1 Ground-Truth Labelling Methodology
    Implements multiple labelling strategies for toxic trades
    """
    
    def __init__(self):
        self.labelling_methods = ['agent_type', 'price_movement', 'mm_loss', 'combined']
    
    def label_agent_type_method(self, trades_df):
        """
        Method 1: Agent Type Logs
        A trade is toxic if an informed agent is involved
        """
        trades_df['toxic_agent_type'] = (
            (trades_df['buyer_type'] == 'informed') | 
            (trades_df['seller_type'] == 'informed')
        ).astype(int)
        return trades_df
    
    def label_price_movement_method(self, trades_df, orders_df, horizon_ms=10, threshold=0.002):
        """
        Method 2: Future Return Delta Thresholds
        A trade is toxic if price moves > threshold in informed trader's direction within horizon
        """
        trades_df = trades_df.sort_values('timestamp').copy()
        trades_df['toxic_price_movement'] = 0
        
        for idx, trade in trades_df.iterrows():
            # Get future prices within horizon
            future_trades = trades_df[
                (trades_df['timestamp'] > trade['timestamp']) &
                (trades_df['timestamp'] <= trade['timestamp'] + horizon_ms)
            ]
            
            if len(future_trades) > 0:
                future_price = future_trades['price'].iloc[-1]
                price_change = (future_price - trade['price']) / trade['price']
                
                # Check if informed trader benefited
                buyer_informed = trade['buyer_type'] == 'informed'
                seller_informed = trade['seller_type'] == 'informed'
                
                if buyer_informed and price_change > threshold:
                    trades_df.loc[idx, 'toxic_price_movement'] = 1
                elif seller_informed and price_change < -threshold:
                    trades_df.loc[idx, 'toxic_price_movement'] = 1
        
        return trades_df
    
    def label_mm_loss_method(self, trades_df, orders_df):
        """
        Method 3: Post-trade Market Maker Loss
        A trade is toxic if market maker loses money on the position
        """
        # Find market maker trades
        mm_trades = trades_df[
            (trades_df['buyer_type'] == 'market_maker') | 
            (trades_df['seller_type'] == 'market_maker')
        ].copy()
        
        trades_df['toxic_mm_loss'] = 0
        
        for idx, trade in mm_trades.iterrows():
            # Simple proxy: if MM bought and price went down, or MM sold and price went up
            future_trades = trades_df[trades_df['timestamp'] > trade['timestamp']].head(5)
            
            if len(future_trades) > 0:
                future_avg_price = future_trades['price'].mean()
                
                mm_is_buyer = trade['buyer_type'] == 'market_maker'
                mm_is_seller = trade['seller_type'] == 'market_maker'
                
                if mm_is_buyer and future_avg_price < trade['price']:
                    trades_df.loc[idx, 'toxic_mm_loss'] = 1
                elif mm_is_seller and future_avg_price > trade['price']:
                    trades_df.loc[idx, 'toxic_mm_loss'] = 1
        
        return trades_df
    
    def label_combined_method(self, trades_df):
        """
        Method 4: Combined Approach
        A trade is toxic if it meets multiple criteria
        """
        # Ensure required columns exist
        required_cols = ['toxic_agent_type', 'toxic_price_movement', 'toxic_mm_loss']
        for col in required_cols:
            if col not in trades_df.columns:
                trades_df[col] = 0
        
        # Combined: at least 2 out of 3 methods agree
        trades_df['toxic_combined'] = (
            (trades_df['toxic_agent_type'] + 
             trades_df['toxic_price_movement'] + 
             trades_df['toxic_mm_loss']) >= 2
        ).astype(int)
        
        return trades_df
    
    def generate_all_labels(self, trades_df, orders_df):
        """Generate all labelling methods"""
        print("Generating ground-truth labels...")
        
        # Apply all labelling methods
        trades_df = self.label_agent_type_method(trades_df)
        trades_df = self.label_price_movement_method(trades_df, orders_df)
        trades_df = self.label_mm_loss_method(trades_df, orders_df)
        trades_df = self.label_combined_method(trades_df)
        
        # Print summary
        print("\nLabelling Results:")
        for method in ['toxic_agent_type', 'toxic_price_movement', 'toxic_mm_loss', 'toxic_combined']:
            if method in trades_df.columns:
                toxic_rate = trades_df[method].mean() * 100
                print(f"  {method}: {toxic_rate:.1f}% toxic trades")
        
        return trades_df

class SupervisedModelPipeline:
    """
    4.4.2 Supervised Model Selection and Training Pipeline
    Implements RF, XGBoost, and Neural Network with proper evaluation
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.results = {}
    
    def prepare_features(self, orders_df, target_col='was_toxic'):
        """Prepare feature matrix and target vector"""
        print(f"Preparing features with target: {target_col}")
        
        # Select relevant features
        feature_columns = [
            'quantity', 'distance_from_mid', 'is_aggressive',
            'volatility', 'momentum', 'order_book_imbalance',
            'time_since_last_trade', 'spread'
        ]
        
        # Agent type features
        orders_df['is_informed'] = (orders_df['agent_type'] == 'informed').astype(int)
        orders_df['is_market_maker'] = (orders_df['agent_type'] == 'market_maker').astype(int)
        feature_columns.extend(['is_informed', 'is_market_maker'])
        
        # Market regime features
        if 'mid_price' in orders_df.columns:
            orders_df['price_volatility'] = orders_df['mid_price'].rolling(20).std()
            feature_columns.append('price_volatility')
        
        # Select available features
        available_features = [col for col in feature_columns if col in orders_df.columns]
        
        # Remove rows with missing target
        clean_df = orders_df.dropna(subset=[target_col])
        
        X = clean_df[available_features].fillna(0)
        y = clean_df[target_col].astype(int)
        
        self.feature_names = available_features
        
        print(f"Features prepared: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def handle_class_imbalance(self, X, y, method='smote'):
        """Handle class imbalance using various techniques"""
        print(f"Handling class imbalance with method: {method}")
        
        original_dist = y.value_counts()
        print(f"Original distribution: {original_dist.to_dict()}")
        
        if method == 'smote':
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y
        
        new_dist = pd.Series(y_resampled).value_counts()
        print(f"New distribution: {new_dist.to_dict()}")
        
        return X_resampled, y_resampled
    
    def train_random_forest(self, X_train, y_train, class_weight='balanced'):
        """Train Random Forest model"""
        print("Training Random Forest...")
        
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            class_weight=class_weight,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        
        return rf
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        print("Training XGBoost...")
        
        # Calculate scale_pos_weight for imbalanced data
        neg_count = sum(y_train == 0)
        pos_count = sum(y_train == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc'
        )
        
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        return xgb_model
    
    def train_neural_network(self, X_train, y_train):
        """Train Feedforward Neural Network"""
        print("Training Neural Network...")
        
        # Scale features for neural network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['neural_network'] = scaler
        
        nn = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        nn.fit(X_train_scaled, y_train)
        self.models['neural_network'] = nn
        
        return nn
    
    def evaluate_model(self, model, X_test, y_test, model_name, use_scaler=False):
        """Comprehensive model evaluation"""
        print(f"\nEvaluating {model_name}...")
        
        # Prepare test data
        if use_scaler and model_name in self.scalers:
            X_test_processed = self.scalers[model_name].transform(X_test)
        else:
            X_test_processed = X_test
        
        # Predictions
        y_pred = model.predict(X_test_processed)
        y_proba = model.predict_proba(X_test_processed)[:, 1]
        
        # Metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        results = {
            'model': model_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        self.results[model_name] = results
        
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1: {f1:.3f}")
        print(f"  AUC: {auc:.3f}")
        
        return results
    
    def cross_validate_models(self, X, y, cv=5):
        """Perform k-fold cross-validation"""
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            if model_name == 'neural_network':
                # Scale features for neural network
                X_scaled = self.scalers['neural_network'].fit_transform(X)
                scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
            else:
                scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            
            cv_results[model_name] = {
                'mean_auc': scores.mean(),
                'std_auc': scores.std(),
                'scores': scores
            }
            
            print(f"  {model_name}: AUC = {scores.mean():.3f} ± {scores.std():.3f}")
        
        return cv_results
    
    def plot_evaluation_results(self, save_dir="model_evaluation"):
        """Plot comprehensive evaluation results"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Supervised Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Performance comparison
        models = list(self.results.keys())
        metrics = ['precision', 'recall', 'f1', 'auc']
        
        performance_data = {metric: [self.results[model][metric] for model in models] 
                          for metric in metrics}
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            axes[0, 0].bar(x + i*width, performance_data[metric], width, 
                          label=metric.upper(), alpha=0.8)
        
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x + width * 1.5)
        axes[0, 0].set_xticklabels(models, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROC Curves
        for model_name, results in self.results.items():
            fpr, tpr, _ = roc_curve(results['y_test'], results['y_proba'])
            auc = results['auc']
            axes[0, 1].plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", linewidth=2)
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curves
        for model_name, results in self.results.items():
            precision, recall, _ = precision_recall_curve(results['y_test'], results['y_proba'])
            axes[0, 2].plot(recall, precision, label=model_name, linewidth=2)
        
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curves')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Confusion Matrices
        for i, (model_name, results) in enumerate(self.results.items()):
            cm = confusion_matrix(results['y_test'], results['y_pred'])
            
            row = (i // 3) + 1
            col = i % 3
            
            if row < 2:  # Only plot if we have space
                sns.heatmap(cm, annot=True, fmt='d', ax=axes[row, col], 
                           cmap='Blues', cbar=False)
                axes[row, col].set_title(f'Confusion Matrix - {model_name}')
                axes[row, col].set_xlabel('Predicted')
                axes[row, col].set_ylabel('Actual')
        
        # Remove empty subplots
        for i in range(len(self.results), 3):
            row = (i // 3) + 1
            col = i % 3
            if row < 2:
                fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/model_evaluation_results.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_calibration_analysis(self, save_dir="model_evaluation"):
        """Plot calibration curves for reliability analysis"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calibration plots
        for model_name, results in self.results.items():
            fraction_of_positives, mean_predicted_value = calibration_curve(
                results['y_test'], results['y_proba'], n_bins=10
            )
            
            axes[0].plot(mean_predicted_value, fraction_of_positives, 'o-', 
                        label=model_name, linewidth=2, markersize=6)
        
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        axes[0].set_xlabel('Mean Predicted Probability')
        axes[0].set_ylabel('Fraction of Positives')
        axes[0].set_title('Calibration Curves (Reliability)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Distribution of predicted probabilities
        for model_name, results in self.results.items():
            axes[1].hist(results['y_proba'], bins=20, alpha=0.6, 
                        label=model_name, density=True)
        
        axes[1].set_xlabel('Predicted Probability')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Distribution of Predicted Probabilities')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/calibration_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()

class MLEnhancedMarketMaker:
    """
    ML-Enhanced Market Maker with Dynamic Spread Adjustment
    Integrates best performing model for real-time toxicity detection
    """
    
    def __init__(self, best_model, scaler, feature_names, base_spread_bps=50):
        self.model = best_model
        self.scaler = scaler
        self.feature_names = feature_names
        self.base_spread_bps = base_spread_bps
        
        # Performance tracking
        self.predictions = []
        self.spread_adjustments = []
        self.pnl_history = []
        
    def extract_order_features(self, order_data, market_context):
        """Extract features from incoming order and market context"""
        features = {}
        
        # Order features
        features['quantity'] = order_data.get('quantity', 1)
        features['distance_from_mid'] = order_data.get('distance_from_mid', 0)
        features['is_aggressive'] = order_data.get('is_aggressive', 0)
        
        # Market context features
        features['volatility'] = market_context.get('volatility', 0)
        features['momentum'] = market_context.get('momentum', 0)
        features['order_book_imbalance'] = market_context.get('order_book_imbalance', 0)
        features['time_since_last_trade'] = market_context.get('time_since_last_trade', 0)
        features['spread'] = market_context.get('spread', 1)
        
        # Agent type (unknown in real-time, use conservative defaults)
        features['is_informed'] = 0  # Conservative assumption
        features['is_market_maker'] = 0
        
        # Market regime
        features['price_volatility'] = market_context.get('price_volatility', 0)
        
        return features
    
    def predict_toxicity(self, order_features):
        """Predict toxicity probability for incoming order"""
        # Ensure features are in correct order and fill missing
        feature_vector = np.array([
            order_features.get(feature, 0) for feature in self.feature_names
        ]).reshape(1, -1)
        
        # Scale if scaler is available
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector)
        
        # Predict toxicity probability
        toxicity_prob = self.model.predict_proba(feature_vector)[0][1]
        
        return toxicity_prob
    
    def calculate_spread_adjustment(self, toxicity_prob, market_volatility=0):
        """
        Calculate dynamic spread adjustment based on ML prediction
        
        Args:
            toxicity_prob: Predicted toxicity probability [0,1]
            market_volatility: Current market volatility for additional adjustment
        
        Returns:
            adjusted_spread_bps: New spread in basis points
            multiplier: Adjustment multiplier
        """
        # Improved base adjustment from toxicity prediction
        if toxicity_prob > 0.8:
            base_multiplier = 3.0  # Very high toxicity
        elif toxicity_prob > 0.6:
            base_multiplier = 2.5  # High toxicity
        elif toxicity_prob > 0.4:
            base_multiplier = 2.0  # Medium toxicity
        elif toxicity_prob > 0.2:
            base_multiplier = 1.5  # Low-medium toxicity
        else:
            base_multiplier = 1.0  # Low toxicity
        
        # Additional volatility adjustment
        vol_multiplier = 1 + min(market_volatility * 15, 0.8)  # Increased vol sensitivity
        
        # Combined multiplier
        total_multiplier = base_multiplier * vol_multiplier
        
        # Calculate adjusted spread
        adjusted_spread_bps = self.base_spread_bps * total_multiplier
        
        # Cap maximum spread to prevent unrealistic values
        adjusted_spread_bps = min(adjusted_spread_bps, 400)  # Max 400 bps
        
        return adjusted_spread_bps, total_multiplier
    
    def process_order(self, order_data, market_context):
        """Process incoming order and return spread recommendation"""
        # Extract features
        order_features = self.extract_order_features(order_data, market_context)
        
        # Predict toxicity
        toxicity_prob = self.predict_toxicity(order_features)
        
        # Calculate spread adjustment
        adjusted_spread, multiplier = self.calculate_spread_adjustment(
            toxicity_prob, market_context.get('volatility', 0)
        )
        
        # Record decision
        decision = {
            'timestamp': order_data.get('timestamp', 0),
            'toxicity_prob': toxicity_prob,
            'spread_multiplier': multiplier,
            'adjusted_spread_bps': adjusted_spread,
            'base_spread_bps': self.base_spread_bps
        }
        
        self.predictions.append(decision)
        
        return decision
    
    def get_performance_summary(self):
        """Get performance summary of ML-enhanced market maker"""
        if not self.predictions:
            return "No predictions made yet"
        
        df = pd.DataFrame(self.predictions)
        
        summary = {
            'total_predictions': len(df),
            'avg_toxicity_prob': df['toxicity_prob'].mean(),
            'avg_spread_multiplier': df['spread_multiplier'].mean(),
            'avg_adjusted_spread': df['adjusted_spread_bps'].mean(),
            'high_toxicity_rate': (df['toxicity_prob'] > 0.5).mean(),
            'spread_widening_rate': (df['spread_multiplier'] > 1.2).mean()
        }
        
        return summary

def load_simulation_data(data_dir="enhanced_market_data"):
    """Load the most recent simulation data"""
    print("Loading simulation data...")
    
    # Find most recent files
    order_files = glob.glob(f"{data_dir}/orders_*.csv")
    trade_files = glob.glob(f"{data_dir}/trades_*.csv")
    
    if not order_files or not trade_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    latest_order_file = max(order_files, key=os.path.getctime)
    latest_trade_file = max(trade_files, key=os.path.getctime)
    
    orders_df = pd.read_csv(latest_order_file)
    trades_df = pd.read_csv(latest_trade_file)
    
    print(f"Loaded {len(orders_df)} orders and {len(trades_df)} trades")
    
    return orders_df, trades_df

def main_benchmarking_pipeline():
    """
    Main pipeline for supervised model benchmarking
    Implements Section 4.4 of the thesis
    """
    print("="*80)
    print("SUPERVISED MODEL BENCHMARKING FOR TOXICITY DETECTION")
    print("Section 4.4: Ground-Truth Labelling and Model Selection")
    print("="*80)
    
    # Load data
    orders_df, trades_df = load_simulation_data()
    
    # 4.4.1 Ground-Truth Labelling Methodology
    print("\n4.4.1 GROUND-TRUTH LABELLING METHODOLOGY")
    print("-" * 50)
    
    labeller = ToxicityLabeller()
    trades_with_labels = labeller.generate_all_labels(trades_df, orders_df)
    
    # 4.4.2 Supervised Model Selection and Training Pipeline
    print("\n4.4.2 SUPERVISED MODEL SELECTION AND TRAINING PIPELINE")
    print("-" * 60)
    
    # Test different labelling methods
    label_methods = ['was_toxic', 'toxic_agent_type', 'toxic_price_movement', 'toxic_combined']
    best_results = {}
    
    for label_method in label_methods:
        if label_method not in orders_df.columns:
            continue
            
        print(f"\n--- Testing with {label_method} labels ---")
        
        pipeline = SupervisedModelPipeline()
        
        # Prepare features
        X, y = pipeline.prepare_features(orders_df, target_col=label_method)
        
        if len(y.unique()) < 2:
            print(f"Skipping {label_method} - insufficient class diversity")
            continue
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = pipeline.handle_class_imbalance(
            X_train, y_train, method='smote'
        )
        
        # Train models
        pipeline.train_random_forest(X_train_balanced, y_train_balanced)
        pipeline.train_xgboost(X_train_balanced, y_train_balanced)
        pipeline.train_neural_network(X_train_balanced, y_train_balanced)
        
        # Evaluate models
        rf_results = pipeline.evaluate_model(
            pipeline.models['random_forest'], X_test, y_test, 'random_forest'
        )
        xgb_results = pipeline.evaluate_model(
            pipeline.models['xgboost'], X_test, y_test, 'xgboost'
        )
        nn_results = pipeline.evaluate_model(
            pipeline.models['neural_network'], X_test, y_test, 'neural_network', use_scaler=True
        )
        
        # Cross-validation
        cv_results = pipeline.cross_validate_models(X_train_balanced, y_train_balanced)
        
        # Store best results for this labelling method
        best_model_name = max(pipeline.results.keys(), 
                            key=lambda x: pipeline.results[x]['auc'])
        best_results[label_method] = {
            'pipeline': pipeline,
            'best_model': best_model_name,
            'best_auc': pipeline.results[best_model_name]['auc'],
            'cv_results': cv_results
        }
        
        # Plot results
        print(f"\nGenerating evaluation plots for {label_method}...")
        pipeline.plot_evaluation_results(save_dir=f"evaluation_{label_method}")
        pipeline.plot_calibration_analysis(save_dir=f"evaluation_{label_method}")
    
    # Select overall best model
    print("\n" + "="*60)
    print("MODEL SELECTION RESULTS")
    print("="*60)
    
    overall_best = None
    best_auc = 0
    
    for label_method, results in best_results.items():
        print(f"\n{label_method}:")
        print(f"  Best model: {results['best_model']}")
        print(f"  Best AUC: {results['best_auc']:.3f}")
        
        if results['best_auc'] > best_auc:
            best_auc = results['best_auc']
            overall_best = (label_method, results)
    
    if overall_best:
        best_label_method, best_result = overall_best
        best_pipeline = best_result['pipeline']
        best_model_name = best_result['best_model']
        best_model = best_pipeline.models[best_model_name]
        best_scaler = best_pipeline.scalers.get(best_model_name, None)
        
        print(f"\n🏆 OVERALL BEST MODEL:")
        print(f"   Label Method: {best_label_method}")
        print(f"   Model Type: {best_model_name}")
        print(f"   AUC Score: {best_auc:.3f}")
        
        # Save best model
        model_save_dir = "saved_models"
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{model_save_dir}/best_toxicity_model_{timestamp}.joblib"
        
        # Save model and metadata
        model_metadata = {
            'model': best_model,
            'scaler': best_scaler,
            'feature_names': best_pipeline.feature_names,
            'label_method': best_label_method,
            'model_type': best_model_name,
            'auc_score': best_auc,
            'timestamp': timestamp
        }
        
        joblib.dump(model_metadata, model_path)
        print(f"✅ Best model saved to: {model_path}")
        
        # 4.4.3 Insights and Limitations Analysis
        print("\n4.4.3 INSIGHTS AND LIMITATIONS FROM SUPERVISED APPROACH")
        print("-" * 60)
        
        analyze_model_insights_and_limitations(best_results, trades_with_labels)
        
        # Implement ML-Enhanced Market Maker
        print("\nIMPLEMENTING ML-ENHANCED MARKET MAKER")
        print("-" * 45)
        
        ml_market_maker = MLEnhancedMarketMaker(
            best_model=best_model,
            scaler=best_scaler,
            feature_names=best_pipeline.feature_names,
            base_spread_bps=50
        )
        
        # Test ML-enhanced market maker on recent data
        test_ml_market_maker(ml_market_maker, orders_df, trades_df)
        
        # Save complete ML market maker for integration
        save_ml_market_maker_for_simulation(ml_market_maker, model_metadata)
        
        return ml_market_maker, model_metadata
    
    else:
        print("❌ No suitable model found")
        return None, None

def analyze_model_insights_and_limitations(best_results, trades_with_labels):
    """
    4.4.3 Analyze insights and limitations from supervised approach
    """
    print("\n📊 ANALYSIS OF SUPERVISED APPROACH")
    
    # 1. Challenges with Label Quality
    print("\n1. CHALLENGES WITH LABEL QUALITY:")
    
    # Compare different labelling methods
    label_correlations = {}
    label_cols = ['toxic_agent_type', 'toxic_price_movement', 'toxic_mm_loss', 'toxic_combined']
    
    available_labels = [col for col in label_cols if col in trades_with_labels.columns]
    
    if len(available_labels) > 1:
        correlation_matrix = trades_with_labels[available_labels].corr()
        print("\n   Label Method Correlations:")
        print(correlation_matrix.round(3))
        
        # Calculate agreement rates
        for i, label1 in enumerate(available_labels):
            for label2 in available_labels[i+1:]:
                agreement = (trades_with_labels[label1] == trades_with_labels[label2]).mean()
                print(f"   {label1} vs {label2}: {agreement:.1%} agreement")
    
    # 2. Label Noise Analysis
    print("\n   Label Noise Indicators:")
    for label_method, results in best_results.items():
        pipeline = results['pipeline']
        if 'random_forest' in pipeline.models:
            rf_model = pipeline.models['random_forest']
            feature_importance = pd.DataFrame({
                'feature': pipeline.feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n   Top features for {label_method}:")
            print(feature_importance.head(5).to_string(index=False))
    
    # 3. Real-World Inapplicability
    print("\n2. REAL-WORLD INAPPLICABILITY CHALLENGES:")
    print("   ❌ No ground-truth labels available in live trading")
    print("   ❌ Simulation patterns may not generalise to real markets")
    print("   ❌ Non-stationary market regimes cause model degradation")
    print("   ❌ Information leakage from future-looking labels")
    
    # 4. Model Stability Analysis
    print("\n3. MODEL STABILITY ANALYSIS:")
    
    # Check consistency across different data splits
    stability_scores = []
    for label_method, results in best_results.items():
        cv_results = results['cv_results']
        best_model_name = results['best_model']
        
        if best_model_name in cv_results:
            cv_std = cv_results[best_model_name]['std_auc']
            stability_score = 1 - cv_std  # Higher is more stable
            stability_scores.append((label_method, best_model_name, stability_score))
            
            print(f"   {label_method} ({best_model_name}): Stability = {stability_score:.3f}")
    
    # 5. Practical Limitations
    print("\n4. PRACTICAL LIMITATIONS FOR REAL DEPLOYMENT:")
    print("   • Feature availability in real-time (agent type unknown)")
    print("   • Latency requirements for high-frequency trading")
    print("   • Model interpretability for regulatory compliance")
    print("   • Adaptation to changing market microstructure")
    print("   • Risk of overfitting to historical patterns")

def test_ml_market_maker(ml_market_maker, orders_df, trades_df):
    """Test ML-enhanced market maker on historical data"""
    print("\n🤖 TESTING ML-ENHANCED MARKET MAKER")
    print("-" * 45)
    
    # Simulate real-time order processing
    test_orders = orders_df.sample(n=min(1000, len(orders_df)), random_state=42)
    
    total_spread_adjustment = 0
    high_toxicity_count = 0
    decisions = []
    
    for _, order in test_orders.iterrows():
        # Simulate market context (in practice, this comes from live data)
        market_context = {
            'volatility': order.get('volatility', 0),
            'momentum': order.get('momentum', 0),
            'order_book_imbalance': order.get('order_book_imbalance', 0),
            'time_since_last_trade': order.get('time_since_last_trade', 0),
            'spread': order.get('spread', 1),
            'price_volatility': orders_df['volatility'].rolling(20).std().iloc[-1] if len(orders_df) > 20 else 0
        }
        
        # Process order through ML market maker
        decision = ml_market_maker.process_order(order.to_dict(), market_context)
        decisions.append(decision)
        
        total_spread_adjustment += decision['spread_multiplier']
        if decision['toxicity_prob'] > 0.5:
            high_toxicity_count += 1
    
    # Analyze performance
    print(f"\n📈 ML MARKET MAKER PERFORMANCE:")
    print(f"   Orders processed: {len(decisions)}")
    print(f"   High toxicity detected: {high_toxicity_count} ({high_toxicity_count/len(decisions)*100:.1f}%)")
    print(f"   Average spread multiplier: {total_spread_adjustment/len(decisions):.2f}x")
    
    # Get detailed performance summary
    performance = ml_market_maker.get_performance_summary()
    print(f"\n📊 DETAILED PERFORMANCE METRICS:")
    for key, value in performance.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # Plot ML market maker decisions
    plot_ml_market_maker_performance(ml_market_maker, save_dir="ml_market_maker_results")
    
    return decisions

def plot_ml_market_maker_performance(ml_market_maker, save_dir="ml_market_maker_results"):
    """Plot ML market maker performance metrics"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if not ml_market_maker.predictions:
        print("No predictions to plot")
        return
    
    df = pd.DataFrame(ml_market_maker.predictions)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ML-Enhanced Market Maker Performance', fontsize=16, fontweight='bold')
    
    # 1. Toxicity probability distribution
    axes[0, 0].hist(df['toxicity_prob'], bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].axvline(df['toxicity_prob'].mean(), color='black', linestyle='--', 
                      label=f'Mean: {df["toxicity_prob"].mean():.3f}')
    axes[0, 0].set_xlabel('Predicted Toxicity Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Toxicity Predictions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Spread adjustment over time
    axes[0, 1].plot(df.index, df['adjusted_spread_bps'], 'b-', alpha=0.7, label='Adjusted Spread')
    axes[0, 1].axhline(df['base_spread_bps'].iloc[0], color='red', linestyle='--', 
                      label=f'Base Spread ({df["base_spread_bps"].iloc[0]} bps)')
    axes[0, 1].set_xlabel('Order Sequence')
    axes[0, 1].set_ylabel('Spread (bps)')
    axes[0, 1].set_title('Dynamic Spread Adjustment')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Spread multiplier vs toxicity probability
    scatter = axes[1, 0].scatter(df['toxicity_prob'], df['spread_multiplier'], 
                                alpha=0.6, c=df['toxicity_prob'], cmap='Reds')
    axes[1, 0].set_xlabel('Toxicity Probability')
    axes[1, 0].set_ylabel('Spread Multiplier')
    axes[1, 0].set_title('Spread Adjustment vs Toxicity')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0], label='Toxicity Probability')
    
    # 4. Cumulative spread adjustment benefit
    df['cumulative_adjustment'] = (df['adjusted_spread_bps'] - df['base_spread_bps']).cumsum()
    axes[1, 1].plot(df.index, df['cumulative_adjustment'], 'g-', linewidth=2)
    axes[1, 1].fill_between(df.index, 0, df['cumulative_adjustment'], alpha=0.3, color='green')
    axes[1, 1].set_xlabel('Order Sequence')
    axes[1, 1].set_ylabel('Cumulative Spread Benefit (bps)')
    axes[1, 1].set_title('Cumulative Spread Widening Benefit')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{save_dir}/ml_market_maker_performance_{timestamp}.png", 
                dpi=300, bbox_inches='tight')
    plt.show()

def save_ml_market_maker_for_simulation(ml_market_maker, model_metadata, save_dir="ml_models"):
    """Save ML market maker for integration with simulation"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save complete ML market maker
    ml_mm_data = {
        'ml_market_maker': ml_market_maker,
        'model_metadata': model_metadata,
        'timestamp': timestamp,
        'version': '1.0'
    }
    
    save_path = f"{save_dir}/ml_enhanced_market_maker_{timestamp}.joblib"
    joblib.dump(ml_mm_data, save_path)
    
    print(f"\n💾 ML-Enhanced Market Maker saved to: {save_path}")
    print(f"   Model type: {model_metadata['model_type']}")
    print(f"   AUC score: {model_metadata['auc_score']:.3f}")
    print(f"   Features: {len(model_metadata['feature_names'])}")
    
    return save_path

def load_ml_market_maker(model_path):
    """Load saved ML market maker"""
    try:
        ml_mm_data = joblib.load(model_path)
        print(f"✅ ML Market Maker loaded from: {model_path}")
        return ml_mm_data['ml_market_maker'], ml_mm_data['model_metadata']
    except Exception as e:
        print(f"❌ Error loading ML Market Maker: {e}")
        return None, None

if __name__ == "__main__":
    # Run the complete supervised benchmarking pipeline
    print("Starting Supervised Model Benchmarking Pipeline...")
    
    try:
        # Main benchmarking
        ml_market_maker, model_metadata = main_benchmarking_pipeline()
        
        if ml_market_maker and model_metadata:
            print("\n" + "="*80)
            print("SUPERVISED BENCHMARKING COMPLETE!")
            print("="*80)
            print("✅ Multiple supervised models trained and evaluated")
            print("✅ Best model selected and saved")
            print("✅ ML-enhanced market maker implemented and tested")
            print("✅ Performance analysis completed")
            
            print(f"\n🔗 INTEGRATION READY:")
            print(f"   Best model: {model_metadata['model_type']}")
            print(f"   Performance: {model_metadata['auc_score']:.3f} AUC")
            
            print(f"\n📋 NEXT STEPS:")
            print("   1. Integrate ML market maker into enhanced simulation")
            print("   2. Compare ML-enhanced vs baseline market maker performance")
            print("   3. Analyze improvement in profitability and risk management")
            print("   4. Document insights for thesis Section 4.4")
            
        else:
            print("\n❌ Benchmarking failed - check data and try again")
            
    except Exception as e:
        print(f"\n❌ Error in benchmarking pipeline: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)