import re
import pandas as pd
import numpy as np
import os
import glob
import json
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class EnhancedModelComparisonTool:
    """
    Enhanced tool to extract key metrics from CNN training outputs and create comprehensive comparison tables.
    """
    
    def __init__(self, results_path: str = None):
        self.models_data = []
        self.results_path = results_path or "/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results"
    
    def load_all_results(self):
        """Load all comprehensive result files from the results directory."""
        if not os.path.exists(self.results_path):
            print(f"Results path {self.results_path} does not exist!")
            return
        
        # Look for comprehensive result JSON files
        result_files = glob.glob(os.path.join(self.results_path, "*_comprehensive_results.json"))
        
        if not result_files:
            print(f"No comprehensive result files found in {self.results_path}")
            return
        
        print(f"Found {len(result_files)} result files:")
        for file in result_files:
            print(f"  - {os.path.basename(file)}")
        
        # Load each file
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Convert JSON data to text format for parsing
                model_data = self._parse_json_data(data)
                self.models_data.append(model_data)
                print(f"✅ Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"❌ Error loading {os.path.basename(file_path)}: {e}")
    
    def _parse_json_data(self, data):
        """Parse JSON data directly into model metrics."""
        result = {
            'Model': data.get('model_name', 'Unknown'),
            'Test_MAE': data.get('test_metrics', {}).get('mae', 0.0),
            'Test_RMSE': data.get('test_metrics', {}).get('rmse', 0.0),
            'Test_R2': data.get('test_metrics', {}).get('r2_score', 0.0),
            'Train_MAE': data.get('train_metrics', {}).get('mae', 0.0),
            'Train_RMSE': data.get('train_metrics', {}).get('rmse', 0.0),
            'Train_R2': data.get('train_metrics', {}).get('r2_score', 0.0),
            'Parameters': data.get('model_complexity', {}).get('total_params', 0),
            'Training_Time_s': data.get('training_history', {}).get('training_time_seconds', 0.0),
            'Training_Time_h': data.get('training_history', {}).get('training_time_seconds', 0.0) / 3600,
            'Model_Size_MB': data.get('model_complexity', {}).get('model_size_mb', 0.0),
            'Epochs': data.get('training_history', {}).get('epochs_trained', 0),
            'Best_Epoch': data.get('training_history', {}).get('epochs_trained', 0),
            'Final_Train_Loss': data.get('training_history', {}).get('final_train_loss', 0.0),
            'Final_Val_Loss': data.get('training_history', {}).get('final_val_loss', 0.0),
            'Best_Val_Loss': data.get('training_history', {}).get('best_val_loss', 0.0),
            'Overfitting_Ratio': 0.0
        }
        
        # Calculate overfitting ratio
        if result['Final_Train_Loss'] > 0:
            result['Overfitting_Ratio'] = result['Final_Val_Loss'] / result['Final_Train_Loss']
        
        return result
    
    def add_model_from_json(self, json_data):
        """Add a model from JSON data directly."""
        model_data = self._parse_json_data(json_data)
        self.models_data.append(model_data)
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create a comprehensive comparison table."""
        if not self.models_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.models_data)
        
        # Round numerical values for better display
        numeric_cols = ['Test_MAE', 'Test_RMSE', 'Test_R2', 'Train_MAE', 'Train_RMSE', 'Train_R2', 
                       'Training_Time_s', 'Training_Time_h', 'Model_Size_MB', 'Final_Train_Loss', 
                       'Final_Val_Loss', 'Best_Val_Loss', 'Overfitting_Ratio']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].round(6)
        
        return df
    
    def create_performance_summary(self) -> pd.DataFrame:
        """Create a focused performance comparison table."""
        if not self.models_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.models_data)
        
        # Select key performance metrics
        summary_cols = ['Model', 'Test_MAE', 'Test_RMSE', 'Test_R2', 'Parameters', 
                       'Training_Time_h', 'Model_Size_MB', 'Epochs', 'Best_Epoch']
        
        summary_df = df[summary_cols].copy()
        
        # Round values
        summary_df['Test_MAE'] = summary_df['Test_MAE'].round(6)
        summary_df['Test_RMSE'] = summary_df['Test_RMSE'].round(6)
        summary_df['Test_R2'] = summary_df['Test_R2'].round(6)
        summary_df['Training_Time_h'] = summary_df['Training_Time_h'].round(4)
        summary_df['Model_Size_MB'] = summary_df['Model_Size_MB'].round(2)
        
        # Add ranking columns
        summary_df['MAE_Rank'] = summary_df['Test_MAE'].rank(ascending=True).astype(int)
        summary_df['RMSE_Rank'] = summary_df['Test_RMSE'].rank(ascending=True).astype(int)
        summary_df['R2_Rank'] = summary_df['Test_R2'].rank(ascending=False).astype(int)
        
        # Sort by overall performance (combination of ranks)
        summary_df['Overall_Rank'] = (summary_df['MAE_Rank'] + summary_df['RMSE_Rank'] + summary_df['R2_Rank']) / 3
        summary_df = summary_df.sort_values('Overall_Rank')
        
        return summary_df
    
    def create_fixed_visualizations(self, save_path: str = None):
        """Create fixed visualization plots for model comparison."""
        if not self.models_data:
            print("No data available for visualization")
            return
        
        df = pd.DataFrame(self.models_data)
        
        # Set up matplotlib backend and style
        plt.style.use('default')  # Use default style instead of seaborn
        
        # Create figure with proper spacing
        fig = plt.figure(figsize=(20, 12))
        
        # Define colors for consistency
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Plot 1: Test Metrics Comparison (2x3 grid, position 1)
        ax1 = plt.subplot(2, 3, 1)
        x = range(len(df))
        width = 0.25
        
        bars1 = ax1.bar([i - width for i in x], df['Test_MAE'], width, 
                       label='MAE', alpha=0.8, color=colors[0])
        bars2 = ax1.bar(x, df['Test_RMSE'], width, 
                       label='RMSE', alpha=0.8, color=colors[1])
        bars3 = ax1.bar([i + width for i in x], df['Test_R2'], width, 
                       label='R²', alpha=0.8, color=colors[2])
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Test Metrics Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Model Complexity
        ax2 = plt.subplot(2, 3, 2)
        bars = ax2.bar(df['Model'], df['Parameters'], color=colors[3], alpha=0.7)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Parameters')
        ax2.set_title('Model Complexity (Parameters)', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 1000:
                label = f'{int(height/1000)}K'
            else:
                label = str(int(height))
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    label, ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Training Time
        ax3 = plt.subplot(2, 3, 3)
        bars = ax3.bar(df['Model'], df['Training_Time_h'], color=colors[4], alpha=0.7)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Training Time (hours)')
        ax3.set_title('Training Time Comparison', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}h', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Model Size
        ax4 = plt.subplot(2, 3, 4)
        bars = ax4.bar(df['Model'], df['Model_Size_MB'], color=colors[5], alpha=0.7)
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Model Size (MB)')
        ax4.set_title('Model Size Comparison', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Overfitting Analysis
        ax5 = plt.subplot(2, 3, 5)
        overfitting_colors = []
        for ratio in df['Overfitting_Ratio']:
            if ratio > 1.1:
                overfitting_colors.append('red')
            elif ratio > 1.05:
                overfitting_colors.append('orange')
            else:
                overfitting_colors.append('green')
        
        bars = ax5.bar(df['Model'], df['Overfitting_Ratio'], color=overfitting_colors, alpha=0.7)
        ax5.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Fit')
        ax5.axhline(y=1.1, color='red', linestyle='--', alpha=0.5, label='Overfitting Threshold')
        ax5.set_xlabel('Models')
        ax5.set_ylabel('Val/Train Loss Ratio')
        ax5.set_title('Overfitting Analysis', fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Performance vs Complexity Scatter
        ax6 = plt.subplot(2, 3, 6)
        scatter = ax6.scatter(df['Parameters'], df['Test_R2'], 
                            s=df['Model_Size_MB']*50, alpha=0.6, 
                            c=df['Training_Time_h'], cmap='viridis')
        ax6.set_xlabel('Parameters')
        ax6.set_ylabel('Test R² Score')
        ax6.set_title('Performance vs Complexity', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # Add model labels
        for i, model in enumerate(df['Model']):
            ax6.annotate(model, (df['Parameters'].iloc[i], df['Test_R2'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('Training Time (hours)')
        
        # Add main title
        fig.suptitle('CNN Model Performance Comparison Dashboard', fontsize=16, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def get_best_model_summary(self) -> Dict:
        """Get summary of the best performing model for each metric."""
        if not self.models_data:
            return {}
        
        df = pd.DataFrame(self.models_data)
        
        # Handle edge cases where all values might be 0
        best_models = {}
        
        if df['Test_MAE'].sum() > 0:
            best_models['Best_MAE'] = df.loc[df['Test_MAE'].idxmin(), 'Model']
            best_models['Best_MAE_Value'] = df['Test_MAE'].min()
        
        if df['Test_RMSE'].sum() > 0:
            best_models['Best_RMSE'] = df.loc[df['Test_RMSE'].idxmin(), 'Model']
            best_models['Best_RMSE_Value'] = df['Test_RMSE'].min()
        
        if df['Test_R2'].sum() > 0:
            best_models['Best_R2'] = df.loc[df['Test_R2'].idxmax(), 'Model']
            best_models['Best_R2_Value'] = df['Test_R2'].max()
        
        if df['Training_Time_h'].sum() > 0:
            best_models['Fastest_Training'] = df.loc[df['Training_Time_h'].idxmin(), 'Model']
            best_models['Fastest_Training_Time'] = df['Training_Time_h'].min()
        
        if df['Model_Size_MB'].sum() > 0:
            best_models['Smallest_Model'] = df.loc[df['Model_Size_MB'].idxmin(), 'Model']
            best_models['Smallest_Size'] = df['Model_Size_MB'].min()
        
        if df['Parameters'].sum() > 0:
            best_models['Fewest_Parameters'] = df.loc[df['Parameters'].idxmin(), 'Model']
            best_models['Fewest_Parameters_Count'] = df['Parameters'].min()
        
        return best_models
    
    def print_comparison(self):
        """Print formatted comparison tables."""
        if not self.models_data:
            print("No models added for comparison.")
            return
        
        print("=" * 100)
        print("🔍 COMPREHENSIVE MODEL PERFORMANCE COMPARISON")
        print("=" * 100)
        
        # Performance summary
        summary_df = self.create_performance_summary()
        print("\n📊 PERFORMANCE SUMMARY (Sorted by Overall Performance):")
        print(summary_df.to_string(index=False))
        
        # Best models summary
        best_models = self.get_best_model_summary()
        if best_models:
            print(f"\n🏆 BEST PERFORMERS:")
            for key, value in best_models.items():
                print(f"{key}: {value}")
        
        # Full comparison table
        full_df = self.create_comparison_table()
        print(f"\n📋 DETAILED COMPARISON:")
        print(full_df.to_string(index=False))
        
        print("\n" + "=" * 100)
    
    def save_results(self, output_dir: str = None):
        """Save comparison results to CSV files and visualizations."""
        if not self.models_data:
            print("No data to save.")
            return
        
        if output_dir is None:
            output_dir = self.results_path
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save CSV files
        performance_df = self.create_performance_summary()
        full_df = self.create_comparison_table()
        
        performance_file = os.path.join(output_dir, 'model_performance_summary.csv')
        full_file = os.path.join(output_dir, 'model_full_comparison.csv')
        viz_file = os.path.join(output_dir, 'model_comparison_visualization.png')
        
        performance_df.to_csv(performance_file, index=False)
        full_df.to_csv(full_file, index=False)
        
        # Create visualizations
        self.create_fixed_visualizations(viz_file)
        
        print(f"\n✅ Results saved to:")
        print(f"  - Performance Summary: {performance_file}")
        print(f"  - Full Comparison: {full_file}")
        print(f"  - Visualization: {viz_file}")

# Example usage with your RCN_GRU data
def test_with_sample_data():
    """Test the tool with your RCN_GRU data."""
    comparison = EnhancedModelComparisonTool()
    
    # Sample data structure matching your JSON
    sample_data = {
        "model_name": "RCN_GRU",
        "test_metrics": {
            "mae": 0.06889387965202332,
            "rmse": 0.18600937724113464,
            "r2_score": 0.9660155117871979
        },
        "train_metrics": {
            "mae": 0.06833122670650482,
            "rmse": 0.18369796872138977,
            "r2_score": 0.9651619905031995
        },
        "model_complexity": {
            "total_params": 187377,
            "model_size_mb": 0.7147865295410156
        },
        "training_history": {
            "epochs_trained": 21,
            "training_time_seconds": 33.40950417518616,
            "final_train_loss": 0.08788900822401047,
            "final_val_loss": 0.07783506065607071,
            "best_val_loss": 0.07328089326620102
        }
    }
    
    # Add sample data
    comparison.add_model_from_json(sample_data)
    
    # For demonstration, let's add a few more sample models
    comparison.add_model_from_json({
        "model_name": "AlexNet",
        "test_metrics": {"mae": 0.075, "rmse": 0.195, "r2_score": 0.960},
        "train_metrics": {"mae": 0.071, "rmse": 0.188, "r2_score": 0.963},
        "model_complexity": {"total_params": 60000000, "model_size_mb": 228.0},
        "training_history": {"epochs_trained": 50, "training_time_seconds": 3600, 
                           "final_train_loss": 0.085, "final_val_loss": 0.092, "best_val_loss": 0.088}
    })
    
    comparison.add_model_from_json({
        "model_name": "VGG19",
        "test_metrics": {"mae": 0.072, "rmse": 0.190, "r2_score": 0.962},
        "train_metrics": {"mae": 0.070, "rmse": 0.185, "r2_score": 0.965},
        "model_complexity": {"total_params": 143000000, "model_size_mb": 549.0},
        "training_history": {"epochs_trained": 45, "training_time_seconds": 7200, 
                           "final_train_loss": 0.082, "final_val_loss": 0.089, "best_val_loss": 0.085}
    })
    
    # Print comparison
    comparison.print_comparison()
    
    # Create visualizations
    comparison.create_fixed_visualizations()

if __name__ == "__main__":
    test_with_sample_data()