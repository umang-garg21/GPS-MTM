"""
Create Comprehensive Metrics Table for GPS Trajectory Prediction
================================================================

This script scans all test output folders and collects the TOP 3 KEY METRICS
from each dataset and mask pattern, then creates a comprehensive comparison table.

Metrics collected:
1. Overall Accuracy
2. Recall Range (consistency)
3. Bias Ratio (fairness)
"""

import os
import glob
import pandas as pd
import re
from pathlib import Path

def extract_metrics_from_file(file_path):
    """Extract the 3 key metrics from a top3_metrics file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract metrics using regex
        accuracy_match = re.search(r'1\. Overall Accuracy: ([\d.]+)', content)
        range_match = re.search(r'2\. Recall Range: ([\d.]+)', content)
        bias_match = re.search(r'3\. Bias Ratio: ([\d.]+)x', content)
        
        accuracy = float(accuracy_match.group(1)) if accuracy_match else None
        recall_range = float(range_match.group(1)) if range_match else None
        bias_ratio = float(bias_match.group(1)) if bias_match else None
        
        return accuracy, recall_range, bias_ratio
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None

def scan_output_folders():
    """Scan all test output folders for top3_metrics files"""
    base_path = "/data/home/umang/Trajectory_project/GPS-MTM/outputs"
    results = []
    
    # Look for test folders (not train folders)
    test_folders = [f for f in os.listdir(base_path) if f.startswith('test_')]
    
    for test_folder in test_folders:
        test_path = os.path.join(base_path, test_folder)
        
        # Look for timestamp folders
        for timestamp_folder in os.listdir(test_path):
            timestamp_path = os.path.join(test_path, timestamp_folder)
            
            # Look for test_outputs folder
            test_outputs_path = os.path.join(timestamp_path, "test_outputs")
            if not os.path.exists(test_outputs_path):
                continue
                
            # Look for masking folders
            for masking_folder in os.listdir(test_outputs_path):
                masking_path = os.path.join(test_outputs_path, masking_folder)
                
                # Look for top3_metrics files
                metrics_files = glob.glob(os.path.join(masking_path, "top3_metrics_*.txt"))
                
                for metrics_file in metrics_files:
                    # Extract mask pattern from filename
                    mask_pattern = os.path.basename(metrics_file).replace('top3_metrics_', '').replace('.txt', '')
                    
                    # Extract metrics
                    accuracy, recall_range, bias_ratio = extract_metrics_from_file(metrics_file)
                    
                    if accuracy is not None:
                        # Parse dataset name
                        dataset_parts = test_folder.replace('test_', '').split('_')
                        if len(dataset_parts) >= 3:
                            region = dataset_parts[1]  # atlanta, berlin, etc.
                            category = '_'.join(dataset_parts[2:])  # hunger_outliers, etc.
                            dataset_name = f"{region}_{category}"
                        else:
                            dataset_name = test_folder.replace('test_', '')
                        
                        results.append({
                            'Dataset': dataset_name,
                            'Mask_Pattern': mask_pattern,
                            'Accuracy': accuracy,
                            'Recall_Range': recall_range,
                            'Bias_Ratio': bias_ratio,
                            'Timestamp': timestamp_folder,
                            'File_Path': metrics_file
                        })
    
    return results

def create_summary_table(results):
    """Create a comprehensive summary table"""
    if not results:
        print("No metrics files found!")
        return
    
    df = pd.DataFrame(results)
    
    # Group by dataset and mask pattern, take the most recent timestamp
    df_latest = df.sort_values('Timestamp').groupby(['Dataset', 'Mask_Pattern']).tail(1).reset_index(drop=True)
    
    print("="*100)
    print("📊 COMPREHENSIVE GPS TRAJECTORY PREDICTION METRICS TABLE")
    print("="*100)
    print()
    
    # Create pivot table for better readability
    datasets = df_latest['Dataset'].unique()
    mask_patterns = ['ID', 'FD', 'GOAL', 'RANDOM']
    
    print(f"{'Dataset':<25} {'Mask':<8} {'Accuracy':<10} {'Range':<8} {'Bias':<6} {'Assessment':<15}")
    print("-" * 100)
    
    for dataset in sorted(datasets):
        for mask in mask_patterns:
            row = df_latest[(df_latest['Dataset'] == dataset) & (df_latest['Mask_Pattern'] == mask)]
            if not row.empty:
                acc = row['Accuracy'].iloc[0]
                range_val = row['Recall_Range'].iloc[0]
                bias = row['Bias_Ratio'].iloc[0]
                
                # Assessment
                if acc > 0.97 and range_val < 0.05 and 0.9 <= bias <= 1.1:
                    assessment = "🏆 Excellent"
                elif acc > 0.90 and range_val < 0.15 and 0.8 <= bias <= 1.3:
                    assessment = "✅ Good"
                elif acc > 0.80:
                    assessment = "⚠️  Fair"
                else:
                    assessment = "❌ Poor"
                
                print(f"{dataset:<25} {mask:<8} {acc:<10.3f} {range_val:<8.3f} {bias:<6.2f} {assessment:<15}")
            else:
                print(f"{dataset:<25} {mask:<8} {'N/A':<10} {'N/A':<8} {'N/A':<6} {'No Data':<15}")
        print()  # Empty line between datasets
    
    return df_latest

def create_best_worst_analysis(df):
    """Analyze best and worst performing configurations"""
    print("\n" + "="*60)
    print("🏆 BEST & WORST PERFORMING CONFIGURATIONS")
    print("="*60)
    
    # Best overall accuracy
    best_acc = df.loc[df['Accuracy'].idxmax()]
    print(f"🥇 Best Accuracy: {best_acc['Dataset']} ({best_acc['Mask_Pattern']}) = {best_acc['Accuracy']:.3f}")
    
    # Best consistency (lowest range)
    best_consistency = df.loc[df['Recall_Range'].idxmin()]
    print(f"🎯 Best Consistency: {best_consistency['Dataset']} ({best_consistency['Mask_Pattern']}) = {best_consistency['Recall_Range']:.3f}")
    
    # Best fairness (closest to 1.0)
    df['bias_distance'] = abs(df['Bias_Ratio'] - 1.0)
    best_fairness = df.loc[df['bias_distance'].idxmin()]
    print(f"⚖️  Best Fairness: {best_fairness['Dataset']} ({best_fairness['Mask_Pattern']}) = {best_fairness['Bias_Ratio']:.2f}x")
    
    print("\n" + "-"*60)
    
    # Worst performers
    worst_acc = df.loc[df['Accuracy'].idxmin()]
    print(f"⚠️  Worst Accuracy: {worst_acc['Dataset']} ({worst_acc['Mask_Pattern']}) = {worst_acc['Accuracy']:.3f}")
    
    worst_consistency = df.loc[df['Recall_Range'].idxmax()]
    print(f"📊 Worst Consistency: {worst_consistency['Dataset']} ({worst_consistency['Mask_Pattern']}) = {worst_consistency['Recall_Range']:.3f}")
    
    worst_fairness = df.loc[df['bias_distance'].idxmax()]
    print(f"🚨 Worst Fairness: {worst_fairness['Dataset']} ({worst_fairness['Mask_Pattern']}) = {worst_fairness['Bias_Ratio']:.2f}x")

def create_task_comparison(df):
    """Compare performance across different mask patterns"""
    print("\n" + "="*60)
    print("📈 TASK-WISE PERFORMANCE COMPARISON")
    print("="*60)
    
    task_stats = df.groupby('Mask_Pattern').agg({
        'Accuracy': ['mean', 'std'],
        'Recall_Range': ['mean', 'std'],
        'Bias_Ratio': ['mean', 'std']
    }).round(3)
    
    print(f"{'Task':<8} {'Acc_Mean':<10} {'Acc_Std':<8} {'Range_Mean':<11} {'Range_Std':<10} {'Bias_Mean':<10} {'Bias_Std':<8}")
    print("-" * 80)
    
    for task in ['ID', 'FD', 'GOAL', 'RANDOM']:
        if task in task_stats.index:
            acc_mean = task_stats.loc[task, ('Accuracy', 'mean')]
            acc_std = task_stats.loc[task, ('Accuracy', 'std')]
            range_mean = task_stats.loc[task, ('Recall_Range', 'mean')]
            range_std = task_stats.loc[task, ('Recall_Range', 'std')]
            bias_mean = task_stats.loc[task, ('Bias_Ratio', 'mean')]
            bias_std = task_stats.loc[task, ('Bias_Ratio', 'std')]
            
            print(f"{task:<8} {acc_mean:<10.3f} {acc_std:<8.3f} {range_mean:<11.3f} {range_std:<10.3f} {bias_mean:<10.3f} {bias_std:<8.3f}")

def save_results_to_csv(df):
    """Save results to CSV file"""
    output_file = "/data/home/umang/Trajectory_project/GPS-MTM/outputs/comprehensive_metrics_table.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")

def main():
    """Main function to run the metrics collection and analysis"""
    print("🔍 Scanning for top3_metrics files...")
    results = scan_output_folders()
    
    if not results:
        print("❌ No metrics files found! Make sure you've run the evaluation script first.")
        return
    
    print(f"✅ Found {len(results)} metrics files")
    
    # Create comprehensive table
    df = create_summary_table(results)
    
    if df is not None and not df.empty:
        # Additional analyses
        create_best_worst_analysis(df)
        create_task_comparison(df)
        save_results_to_csv(df)
        
        print(f"\n📋 Summary: Analyzed {len(df)} configurations across {df['Dataset'].nunique()} datasets")
        print("="*100)
    else:
        print("❌ No valid data found in metrics files!")

if __name__ == "__main__":
    main()
