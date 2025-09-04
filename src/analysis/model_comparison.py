import numpy as np
import pandas as pd
import torch
import sys
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, '/home/degan/Graph_Neural_Network_Mutation_Data/src/core')
import process_data as prc
import utility_functions as uf
import read_specific as rs
import cox_loss as cl
import model as m
import download_study as ds

class SurvivalModelComparison:
    """
    Compare different models for survival prediction using the same data as the GNN.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        self.models = {}
        self.scalers = {}
        
    def prepare_data(self):
        """Prepare the same data used by the GNN model."""
        print("🔄 Preparing data (same as GNN)...")
        
        # Download and process data (same as main.py)
        path, sources, urls = ds.download_study(name='msk_pan_2017')
        data_dict = prc.read_files(path[0])
        
        # Extract same features as GNN
        protein_pos = data_dict['mutation']['protein_pos']      
        var_type = data_dict['mutation']['variant_type_np']   
        aa_sub = data_dict['mutation']['amino_acid']      
        chrom_mut = data_dict['mutation']['chromosome_np']              
        var_class_mut = data_dict['mutation']['var_class_np']
        fs_mut = data_dict['mutation']['frameshift']
        
        chrom_sv = data_dict['sv']['chromosome']             
        var_class_sv = data_dict['sv']['var_class']       
        region_sites = data_dict['sv']['region_sites']
        connection_type = data_dict['sv']['connection_type']
        sv_length = data_dict['sv']['sv_length']
        
        cna = data_dict['cna']['cna']
        cna = np.expand_dims(cna, axis=-1)
        
        # Patient clinical data
        osurv_data = data_dict['os_array']
        clinical_data = data_dict['patient']
        
        # Sample metadata
        sample_meta = data_dict['sample_meta']['metadata']
        sample_embeddings = data_dict['sample_meta']['embeddings']
        
        # Process same as GNN
        var_class_mut_flat = uf.merge_last_two_dims(var_class_mut)
        chrom_mut_flat = uf.merge_last_two_dims(chrom_mut)
        aa_sub_flat = uf.merge_last_two_dims(aa_sub)
        var_type_flat = uf.merge_last_two_dims(var_type)
        chrom_sv_flat = uf.merge_last_two_dims(chrom_sv)
        var_class_sv_flat = uf.merge_last_two_dims(var_class_sv)
        region_sites_flat = uf.merge_last_two_dims(region_sites)
        
        arrays_to_concat = [
            protein_pos, fs_mut, var_class_mut_flat, chrom_mut_flat,
            var_type_flat, chrom_sv_flat, aa_sub_flat, var_class_sv_flat,
            region_sites_flat, sv_length, connection_type, cna
        ]
        
        omics = np.concatenate(arrays_to_concat, axis=2)
        
        # Store data
        self.omics_data = omics
        self.clinical_data = clinical_data
        self.osurv_data = osurv_data
        self.sample_embeddings = sample_embeddings
        self.sample_index = data_dict['sample_index']
        self.gene_index = data_dict['gene_index']
        
        print(f"📊 Data shapes:")
        print(f"   Omics: {omics.shape}")
        print(f"   Clinical: {clinical_data.shape}")
        print(f"   Survival: {osurv_data.shape}")
        print(f"   Sample embeddings: {sample_embeddings.shape}")
        
        return self
    
    def create_features(self):
        """Create flattened features for traditional ML models."""
        print("🔧 Creating features for traditional ML models...")
        
        # Flatten omics data: (samples, genes, features) -> (samples, genes*features)
        n_samples, n_genes, n_features = self.omics_data.shape
        omics_flat = self.omics_data.reshape(n_samples, n_genes * n_features)
        
        # Combine all features
        self.X_full = np.concatenate([
            omics_flat,
            self.clinical_data,
            self.sample_embeddings
        ], axis=1)
        
        # Survival outcomes
        self.y_time = self.osurv_data[:, 0]  # Survival time
        self.y_event = self.osurv_data[:, 1]  # Event indicator
        
        # For regression models, we'll predict log(survival_time)
        # Add small epsilon to avoid log(0)
        self.y_log_time = np.log(self.y_time + 1e-6)
        
        print(f"📊 Feature matrix shape: {self.X_full.shape}")
        print(f"📊 Number of events: {self.y_event.sum()}/{len(self.y_event)} ({100*self.y_event.mean():.1f}%)")
        
        return self
    
    def split_data(self):
        """Create same train/val/test splits as GNN."""
        print("📂 Creating train/val/test splits (same as GNN)...")
        
        # Same split as main.py
        np.random.seed(3)
        sample_idx = list(self.sample_index.values())
        np.random.shuffle(sample_idx)
        
        ntrain = int(0.8 * len(self.sample_index))
        nval = int(0.1 * len(self.sample_index))
        
        self.train_idx = sample_idx[:ntrain]
        self.val_idx = sample_idx[ntrain:ntrain + nval]
        self.test_idx = sample_idx[ntrain + nval:]
        
        # Create splits
        self.X_train = self.X_full[self.train_idx]
        self.X_val = self.X_full[self.val_idx]
        self.X_test = self.X_full[self.test_idx]
        
        self.y_train = self.y_log_time[self.train_idx]
        self.y_val = self.y_log_time[self.val_idx]
        self.y_test = self.y_log_time[self.test_idx]
        
        # Also keep event indicators for C-index calculation
        self.y_train_event = self.y_event[self.train_idx]
        self.y_val_event = self.y_event[self.val_idx]
        self.y_test_event = self.y_event[self.test_idx]
        
        # Keep original times for C-index
        self.y_train_time = self.y_time[self.train_idx]
        self.y_val_time = self.y_time[self.val_idx]
        self.y_test_time = self.y_time[self.test_idx]
        
        print(f"📊 Split sizes:")
        print(f"   Train: {len(self.train_idx)} samples")
        print(f"   Val: {len(self.val_idx)} samples")
        print(f"   Test: {len(self.test_idx)} samples")
        
        return self
    
    def train_linear_models(self):
        """Train linear regression models with regularization."""
        print("🤖 Training Linear Regression models...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_val_scaled = scaler.transform(self.X_val)
        X_test_scaled = scaler.transform(self.X_test)
        
        self.scalers['linear'] = scaler
        
        # Ridge Regression with cross-validation
        ridge_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
        ridge = GridSearchCV(
            Ridge(random_state=self.random_state),
            ridge_params,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        ridge.fit(X_train_scaled, self.y_train)
        
        # ElasticNet with cross-validation
        elastic_params = {
            'alpha': [0.001, 0.01, 0.1, 1.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        elastic = GridSearchCV(
            ElasticNet(random_state=self.random_state, max_iter=2000),
            elastic_params,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        elastic.fit(X_train_scaled, self.y_train)
        
        # Store models
        self.models['ridge'] = ridge
        self.models['elastic'] = elastic
        
        # Evaluate models
        for name, model in [('ridge', ridge), ('elastic', elastic)]:
            # Predictions
            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)
            test_pred = model.predict(X_test_scaled)
            
            # Convert back to original time scale
            train_pred_time = np.exp(train_pred)
            val_pred_time = np.exp(val_pred)
            test_pred_time = np.exp(test_pred)
            
            # Calculate metrics
            train_mse = mean_squared_error(self.y_train, train_pred)
            val_mse = mean_squared_error(self.y_val, val_pred)
            test_mse = mean_squared_error(self.y_test, test_pred)
            
            train_r2 = r2_score(self.y_train, train_pred)
            val_r2 = r2_score(self.y_val, val_pred)
            test_r2 = r2_score(self.y_test, test_pred)
            
            # FIXED: C-index calculation consistent with Cox loss
            # For survival prediction: higher predicted time = lower risk
            # So we use negative predicted time as risk score
            # This matches your Cox loss expectation: higher prediction = higher risk
            train_ci = self._calculate_cindex_survival(self.y_train_time, train_pred_time, self.y_train_event)
            val_ci = self._calculate_cindex_survival(self.y_val_time, val_pred_time, self.y_val_event)
            test_ci = self._calculate_cindex_survival(self.y_test_time, test_pred_time, self.y_test_event)
            
            self.results[name] = {
                'train_mse': train_mse,
                'val_mse': val_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'test_r2': test_r2,
                'train_ci': train_ci,
                'val_ci': val_ci,
                'test_ci': test_ci,
                'best_params': model.best_params_
            }
            
            print(f"   {name.upper()}: Test R² = {test_r2:.4f}, Test C-Index = {test_ci:.4f}")
        
        return self
    
    def train_random_forest(self):
        """Train Random Forest model."""
        print("🌲 Training Random Forest model...")
        
        # Random Forest parameters (same as before)
        rf_params_small = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', None]
        }
        
        rf = GridSearchCV(
            RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
            rf_params_small,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        rf.fit(self.X_train, self.y_train)
        
        self.models['random_forest'] = rf
        
        # Evaluate
        train_pred = rf.predict(self.X_train)
        val_pred = rf.predict(self.X_val)
        test_pred = rf.predict(self.X_test)
        
        # Convert back to original time scale
        train_pred_time = np.exp(train_pred)
        val_pred_time = np.exp(val_pred)
        test_pred_time = np.exp(test_pred)
        
        # Calculate metrics
        train_mse = mean_squared_error(self.y_train, train_pred)
        val_mse = mean_squared_error(self.y_val, val_pred)
        test_mse = mean_squared_error(self.y_test, test_pred)
        
        train_r2 = r2_score(self.y_train, train_pred)
        val_r2 = r2_score(self.y_val, val_pred)
        test_r2 = r2_score(self.y_test, test_pred)
        
        # FIXED: C-index calculation consistent with Cox loss
        train_ci = self._calculate_cindex_survival(self.y_train_time, train_pred_time, self.y_train_event)
        val_ci = self._calculate_cindex_survival(self.y_val_time, val_pred_time, self.y_val_event)
        test_ci = self._calculate_cindex_survival(self.y_test_time, test_pred_time, self.y_test_event)
        
        self.results['random_forest'] = {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'train_ci': train_ci,
            'val_ci': val_ci,
            'test_ci': test_ci,
            'best_params': rf.best_params_
        }
        
        print(f"   RANDOM FOREST: Test R² = {test_r2:.4f}, Test C-Index = {test_ci:.4f}")
        
        return self
    
    def load_gnn_results(self):
        """Load GNN results from the most recent training."""
        print("📊 Loading GNN results...")
        
        results_dir = "/home/degan/Graph_Neural_Network_Mutation_Data/results/training_outputs"
        
        # Find most recent summary file
        summary_files = [f for f in os.listdir(results_dir) if f.startswith('training_summary_')]
        if not summary_files:
            print("   ⚠️ No GNN results found. Please run main.py first.")
            return self
        
        latest_summary = max(summary_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
        summary_path = os.path.join(results_dir, latest_summary)
        
        summary_df = pd.read_csv(summary_path)
        
        # Extract GNN metrics
        gnn_results = {}
        for _, row in summary_df.iterrows():
            metric = row['metric']
            value = row['value']
            gnn_results[metric] = value
        
        # Store in results
        self.results['gnn'] = {
            'test_ci': gnn_results.get('final_test_ci', 0),
            'best_val_ci': gnn_results.get('best_val_ci', 0),
            'best_val_loss': gnn_results.get('best_val_loss', 0),
            'test_loss': gnn_results.get('final_test_loss', 0)
        }
        
        print(f"   GNN: Test C-Index = {self.results['gnn']['test_ci']:.4f}")
        
        return self
    
    def create_comparison_plots(self):
        """Create visualization comparing all models."""
        print("📈 Creating comparison plots...")
        
        # Prepare data for plotting
        models = []
        test_ci_scores = []
        val_ci_scores = []
        test_r2_scores = []
        
        for model_name, results in self.results.items():
            models.append(model_name.replace('_', ' ').title())
            test_ci_scores.append(results.get('test_ci', 0))
            
            if model_name == 'gnn':
                val_ci_scores.append(results.get('best_val_ci', 0))
                test_r2_scores.append(0)  # GNN doesn't have R²
            else:
                val_ci_scores.append(results.get('val_ci', 0))
                test_r2_scores.append(results.get('test_r2', 0))
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # C-Index Comparison
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, test_ci_scores, width, label='Test C-Index', alpha=0.8, color='skyblue')
        ax1.bar(x + width/2, val_ci_scores, width, label='Validation C-Index', alpha=0.8, color='lightcoral')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('C-Index')
        ax1.set_title('🎯 Model Performance: Concordance Index (C-Index)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
        
        # Add value labels on bars
        for i, (test_val, val_val) in enumerate(zip(test_ci_scores, val_ci_scores)):
            ax1.text(i - width/2, test_val + 0.01, f'{test_val:.3f}', ha='center', va='bottom', fontweight='bold')
            ax1.text(i + width/2, val_val + 0.01, f'{val_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # R² Comparison (excluding GNN)
        non_gnn_models = [m for m in models if 'Gnn' not in m]
        non_gnn_r2 = [r2 for m, r2 in zip(models, test_r2_scores) if 'Gnn' not in m]
        
        if non_gnn_r2:
            ax2.bar(non_gnn_models, non_gnn_r2, alpha=0.8, color='lightgreen')
            ax2.set_xlabel('Models')
            ax2.set_ylabel('R² Score')
            ax2.set_title('📊 Model Performance: R² Score')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for i, val in enumerate(non_gnn_r2):
                ax2.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Feature importance for Random Forest
        if 'random_forest' in self.models:
            rf_model = self.models['random_forest'].best_estimator_
            importances = rf_model.feature_importances_
            
            # Get top 20 most important features
            n_features = min(20, len(importances))
            top_indices = np.argsort(importances)[-n_features:]
            top_importances = importances[top_indices]
            
            # Create feature names (simplified)
            n_omics = self.omics_data.shape[1] * self.omics_data.shape[2]
            n_clin = self.clinical_data.shape[1]
            n_embed = self.sample_embeddings.shape[1]
            
            feature_names = []
            for i in top_indices:
                if i < n_omics:
                    feature_names.append(f'Omics_{i}')
                elif i < n_omics + n_clin:
                    feature_names.append(f'Clinical_{i - n_omics}')
                else:
                    feature_names.append(f'Embedding_{i - n_omics - n_clin}')
            
            ax3.barh(range(n_features), top_importances, alpha=0.8, color='orange')
            ax3.set_xlabel('Feature Importance')
            ax3.set_ylabel('Features')
            ax3.set_title('🔍 Random Forest: Top Feature Importances')
            ax3.set_yticks(range(n_features))
            ax3.set_yticklabels(feature_names)
            ax3.grid(True, alpha=0.3)
        
        # Model ranking
        model_ranking = sorted([(name, results.get('test_ci', 0)) for name, results in self.results.items()], 
                              key=lambda x: x[1], reverse=True)
        
        ranking_names, ranking_scores = zip(*model_ranking)
        ranking_names = [name.replace('_', ' ').title() for name in ranking_names]
        
        colors = ['gold', 'silver', '#CD7F32', 'lightblue'][:len(ranking_names)]
        ax4.bar(ranking_names, ranking_scores, alpha=0.8, color=colors)
        ax4.set_xlabel('Models (Ranked by Test C-Index)')
        ax4.set_ylabel('Test C-Index')
        ax4.set_title('🏆 Model Ranking: Test C-Index Performance')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add ranking labels
        for i, (name, score) in enumerate(zip(ranking_names, ranking_scores)):
            ax4.text(i, score + 0.01, f'#{i+1}\n{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'/home/degan/Graph_Neural_Network_Mutation_Data/results/figures/model_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(f'/home/degan/Graph_Neural_Network_Mutation_Data/results/figures/model_comparison.pdf', 
                   bbox_inches='tight')
         
        return self
    
    def save_results(self):
        """Save comparison results to CSV."""
        print("💾 Saving comparison results...")
        
        # Create results DataFrame
        results_data = []
        for model_name, results in self.results.items():
            row = {'model': model_name}
            row.update(results)
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        
        # Save to CSV
        results_path = f'/home/degan/Graph_Neural_Network_Mutation_Data/results/model_comparison.csv'
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results_df.to_csv(results_path, index=False)
        
        print(f"   📄 Results saved to: {results_path}")
        
        return self
    
    def print_summary(self):
        """Print comprehensive summary of results."""
        print("\n" + "="*80)
        print("🎯 MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Sort models by test C-index
        sorted_results = sorted(self.results.items(), key=lambda x: x[1].get('test_ci', 0), reverse=True)
        
        print(f"{'Rank':<6} {'Model':<15} {'Test C-Index':<12} {'Val C-Index':<12} {'Test R²':<10} {'Notes'}")
        print("-" * 80)
        
        for rank, (model_name, results) in enumerate(sorted_results, 1):
            test_ci = results.get('test_ci', 0)
            val_ci = results.get('val_ci', results.get('best_val_ci', 0))
            test_r2 = results.get('test_r2', 0)
            
            # Performance assessment
            if test_ci > 0.7:
                notes = "🌟 Excellent"
            elif test_ci > 0.6:
                notes = "👍 Good"
            elif test_ci > 0.55:
                notes = "⚡ Fair"
            else:
                notes = "🔧 Needs Improvement"
            
            if model_name == 'gnn':
                r2_str = "N/A"
            else:
                r2_str = f"{test_r2:.4f}"
            
            print(f"#{rank:<5} {model_name.replace('_', ' ').title():<15} {test_ci:<12.4f} {val_ci:<12.4f} {r2_str:<10} {notes}")
        
        print("\n" + "="*80)
        
        # Best model
        best_model, best_results = sorted_results[0]
        print(f"🏆 WINNER: {best_model.replace('_', ' ').title()}")
        print(f"   📊 Test C-Index: {best_results.get('test_ci', 0):.4f}")
        
        if best_model != 'gnn':
            print(f"   📈 Test R²: {best_results.get('test_r2', 0):.4f}")
            print(f"   ⚙️ Best Parameters: {best_results.get('best_params', {})}")
        
        print("\n💡 INSIGHTS:")
        print("   • C-Index > 0.5: Better than random")
        print("   • C-Index > 0.6: Good predictive performance")
        print("   • C-Index > 0.7: Excellent predictive performance")
        print("   • Higher R² indicates better fit to survival times")
        
        return self

def main():
    """Run the complete model comparison."""
    print("🚀 Starting Model Comparison for Survival Prediction")
    print("="*60)
    
    # Initialize comparison
    comparison = SurvivalModelComparison(random_state=42)
    
    # Run complete pipeline
    (comparison
     .prepare_data()
     .create_features()
     .split_data()
     .train_linear_models()
     .train_random_forest()
     .load_gnn_results()
     .create_comparison_plots()
     .save_results()
     .print_summary())
    
    print("\n✅ Model comparison completed successfully!")
    print("📁 Check the 'results' directory for outputs.")

if __name__ == "__main__":
    main()