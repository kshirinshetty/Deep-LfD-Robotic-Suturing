#!/usr/bin/env python3
"""
Script to run the enhanced model comparison tool for your CNN results.
"""

import sys
import os

# Add the path to your results directory
RESULTS_PATH = "/home/mona/suturing_ws/src/Deep-LfD-Robotic-Suturing/branch/results"

def main():
    """Run the model comparison analysis."""
    
    print("🚀 Starting Enhanced Model Comparison Analysis")
    print("=" * 60)
    
    # Import the comparison tool (assuming you saved it as model_comparison.py)
    from model_comparison import EnhancedModelComparisonTool
    
    # Initialize the comparison tool
    comparison = EnhancedModelComparisonTool(RESULTS_PATH)
    
    # Load all results from the directory
    print("📁 Loading model results...")
    comparison.load_all_results()
    
    if not comparison.models_data:
        print("❌ No model data found. Please check your results directory.")
        return
    
    print(f"✅ Successfully loaded {len(comparison.models_data)} models")
    
    # Print comprehensive comparison
    comparison.print_comparison()
    
    # Save results and create visualizations
    print("\n💾 Saving results and creating visualizations...")
    comparison.save_results()
    
    # Additional analysis
    print("\n🔍 Additional Analysis:")
    
    # Get the best performing model overall
    summary_df = comparison.create_performance_summary()
    best_overall = summary_df.iloc[0]
    
    print(f"\n🏆 OVERALL BEST MODEL: {best_overall['Model']}")
    print(f"   - Test MAE: {best_overall['Test_MAE']:.6f} (Rank: {best_overall['MAE_Rank']})")
    print(f"   - Test RMSE: {best_overall['Test_RMSE']:.6f} (Rank: {best_overall['RMSE_Rank']})")
    print(f"   - Test R²: {best_overall['Test_R2']:.6f} (Rank: {best_overall['R2_Rank']})")
    print(f"   - Parameters: {best_overall['Parameters']:,}")
    print(f"   - Training Time: {best_overall['Training_Time_h']:.2f}h")
    print(f"   - Model Size: {best_overall['Model_Size_MB']:.1f}MB")
    
    # Efficiency analysis
    print(f"\n⚡ EFFICIENCY ANALYSIS:")
    full_df = comparison.create_comparison_table()
    
    # Calculate efficiency metrics
    full_df['Performance_per_Parameter'] = full_df['Test_R2'] / (full_df['Parameters'] / 1000000)  # R² per million parameters
    full_df['Performance_per_MB'] = full_df['Test_R2'] / full_df['Model_Size_MB']  # R² per MB
    full_df['Performance_per_Hour'] = full_df['Test_R2'] / full_df['Training_Time_h']  # R² per training hour
    
    most_efficient_params = full_df.loc[full_df['Performance_per_Parameter'].idxmax(), 'Model']
    most_efficient_size = full_df.loc[full_df['Performance_per_MB'].idxmax(), 'Model']
    most_efficient_time = full_df.loc[full_df['Performance_per_Hour'].idxmax(), 'Model']
    
    print(f"   - Most Parameter Efficient: {most_efficient_params}")
    print(f"   - Most Size Efficient: {most_efficient_size}")
    print(f"   - Most Time Efficient: {most_efficient_time}")
    
    print("\n🎉 Analysis Complete!")
    print("Check the generated CSV files and visualization for detailed results.")

if __name__ == "__main__":
    main()