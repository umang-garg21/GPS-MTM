"""
GPS Trajectory Prediction Evaluation Script - TOP 3 KEY METRICS
================================================================

This script evaluates GPS trajectory prediction models using the 3 most important metrics:
1. Overall Accuracy - How often predictions are correct
2. Recall Range - Consistency across different location classes  
3. Bias Ratio - Model fairness (prediction bias vs true distribution)

These 3 metrics provide a complete assessment of model performance, consistency, and fairness.
"""

import pickle
import torch
import os
import glob
from sklearn.metrics import accuracy_score  # Only using accuracy_score for top 3 metrics
# from sklearn.metrics import f1_score, precision_score, recall_score, classification_report  # Commented out - not needed for top 3
import numpy as np

main_folder="/data/home/umang/Trajectory_project/GPS-MTM/outputs/test_geolife/2025-08-29_11-10-58/test_outputs/random_masking_0.15_testing/"

for folder in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder)
    # Skip if it's not a directory (e.g., skip .txt files)
    if not os.path.isdir(folder_path):
        continue

    # Get all batch files in the folder
    print("---------------------MASK PATTERN: {}-----------------------".format(folder))
    pred_files = sorted(glob.glob(f"{main_folder}/{folder}/predictions_batch_*.pkl"))
    batch_files = sorted(glob.glob(f"{main_folder}/{folder}/ground_truth_batch_*.pkl"))
    attention_files = sorted(glob.glob(f"{main_folder}/{folder}/attention_masks_batch_*.pkl"))
    masks_files = sorted(glob.glob(f"{main_folder}/{folder}/masks_batch_*.pkl"))

    ground_truth_list_states = []
    predictions_list_states = []

    for pred_file, batch_file, attention_file, masks_file in zip(pred_files, batch_files, attention_files, masks_files):
        with open(pred_file, 'rb') as f:
            predictions = pickle.load(f)

        with open(batch_file, 'rb') as f:
            batch = pickle.load(f)

        with open(attention_file, 'rb') as f:
            attention_masks = pickle.load(f)

        with open(masks_file, 'rb') as f:
            masks = pickle.load(f)

        for i in range(len(attention_masks)):
            # first zero mask is first time when attention_masks[i] becomes zero
            zero_indices = (attention_masks[i].flatten() == 0).nonzero(as_tuple=True)[0]
            first_zero_mask = zero_indices[0].item() if len(zero_indices) > 0 else attention_masks[i].numel()

            # Get the predictions and ground truth for this sequence
            pred_states = predictions["states"][i, :first_zero_mask, :]
            gt_states = batch["states"][i, :first_zero_mask, :]
            
            # Get the mask for this sequence
            mask_states = masks["states"][:first_zero_mask] == 1
            
            # Apply mask to filter only masked positions
            if mask_states.sum() > 0:  # Only process if there are masked positions
                pred_states_masked = pred_states[mask_states, :]
                gt_states_masked = gt_states[mask_states, :]
                
                # Take argmax
                pred_states_masked = torch.argmax(pred_states_masked, dim=-1)
                gt_states_masked = torch.argmax(gt_states_masked, dim=-1)
                
                ground_truth_list_states.append(gt_states_masked)
                predictions_list_states.append(pred_states_masked)

    # Combine all batches
    if not ground_truth_list_states or not predictions_list_states:
        print(f"No data found for folder {folder}. Skipping...")
        continue
        
    ground_truth_list_states = torch.cat(ground_truth_list_states).flatten()
    predictions_list_states = torch.cat(predictions_list_states).flatten()

    print(f"Total number of ground truth states: {ground_truth_list_states.numel()}")
    print(f"Total number of predictions states: {predictions_list_states.numel()}")

    # Calculate accuracy using individual comparison (from notebook cell 14)
    total_correct = 0
    total_elements = 0
    for pred_states, gt_states in zip(predictions_list_states, ground_truth_list_states):
        correct = (pred_states == gt_states).float()
        total_correct += correct.sum().item()
        total_elements += correct.numel()
    manual_accuracy = total_correct / total_elements if total_elements > 0 else 0
    print(f"Manual Accuracy: {manual_accuracy:.4f}")

    # Convert to numpy for sklearn compatibility
    gt_numpy = ground_truth_list_states.cpu().numpy()
    pred_numpy = predictions_list_states.cpu().numpy()

    # Overall metrics - TOP 3 KEY METRICS
    overall_accuracy = accuracy_score(gt_numpy, pred_numpy)

    # CLASS IMBALANCE ANALYSIS METRICS
    unique_classes = torch.unique(ground_truth_list_states)
    class_counts = [(cls.item(), (ground_truth_list_states == cls).sum().item()) for cls in unique_classes]
    class_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate class distribution metrics
    total_samples = len(ground_truth_list_states)
    class_frequencies = [count/total_samples for _, count in class_counts]
    
    # Imbalance Ratio: ratio of most frequent to least frequent class
    imbalance_ratio = max(class_frequencies) / min(class_frequencies)
    
    # Effective Number of Classes (diversity measure)
    entropy = -sum(freq * np.log(freq) for freq in class_frequencies if freq > 0)
    effective_num_classes = np.exp(entropy)
    
    # Model Bias Metrics
    # 1. Majority Class Bias: How much does the model favor the majority class?
    majority_class = class_counts[0][0]
    majority_predictions = (pred_numpy == majority_class).sum()
    majority_bias = majority_predictions / len(pred_numpy)
    majority_true_freq = class_counts[0][1] / total_samples
    
    # 2. Per-class recall range (shows if model ignores minority classes)
    per_class_recalls = []
    per_class_precisions = []
    per_class_f1s = []
    
    for cls, count in class_counts:
        tp = ((pred_numpy == cls) & (gt_numpy == cls)).sum()
        fp = ((pred_numpy == cls) & (gt_numpy != cls)).sum()
        fn = ((pred_numpy != cls) & (gt_numpy == cls)).sum()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_recalls.append(recall)
        per_class_precisions.append(precision)
        per_class_f1s.append(f1)
    
    # Calculate BIAS RATIO (KEY METRIC #3)
    bias_ratio = majority_bias / majority_true_freq

    # Calculate RECALL RANGE (KEY METRIC #2)
    recall_range = max(per_class_recalls) - min(per_class_recalls)
    
    # ========== TOP 3 KEY METRICS OUTPUT ==========
    print(f"\nTOP 3 KEY METRICS:")
    print(f"1. Overall Accuracy: {overall_accuracy:.4f}")
    print(f"2. Recall Range: {recall_range:.3f} (consistency across classes)")
    print(f"3. Bias Ratio: {bias_ratio:.2f}x (1.0 = unbiased)")
    
    # Performance Assessment
    if overall_accuracy > 0.97:
        acc_status = "Excellent"
    elif overall_accuracy > 0.90:
        acc_status = "Good"
    else:
        acc_status = "Needs Improvement"
        
    if recall_range < 0.05:
        range_status = "Excellent Consistency"
    elif recall_range < 0.15:
        range_status = "Good Consistency"
    else:
        range_status = "Concerning Gaps"
        
    if 0.9 <= bias_ratio <= 1.1:
        bias_status = "Unbiased"
    elif 0.8 <= bias_ratio <= 1.3:
        bias_status = "Acceptable Bias"
    else:
        bias_status = "Problematic Bias"
    
    print(f"\nASSESSMENT:")
    print(f"Accuracy: {acc_status}")
    print(f"Consistency: {range_status}")
    print(f"Fairness: {bias_status}")

    # Save the TOP 3 KEY METRICS to a text file
    report_file = f"{main_folder}/top3_metrics_{folder}.txt"
    with open(report_file, 'w') as f:
        f.write("=== TOP 3 KEY METRICS FOR GPS TRAJECTORY PREDICTION ===\n\n")
        
        f.write(f"1. Overall Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"2. Recall Range: {recall_range:.3f} (consistency across classes)\n")
        f.write(f"3. Bias Ratio: {bias_ratio:.2f}x (1.0 = unbiased)\n\n")
        
        f.write("ASSESSMENT:\n")
        f.write(f"Accuracy: {acc_status}\n")
        f.write(f"Consistency: {range_status}\n")
        f.write(f"Fairness: {bias_status}\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("- Overall Accuracy: How often predictions are correct\n")
        f.write("- Recall Range: Gap between best/worst performing classes (lower is better)\n")
        f.write("- Bias Ratio: Model prediction bias vs true distribution (1.0 is ideal)\n\n")
        
        f.write("THRESHOLDS:\n")
        f.write("Accuracy: >97% Excellent, 90-97% Good, <90% Needs Improvement\n")
        f.write("Range: <0.05 Excellent, 0.05-0.15 Good, >0.15 Concerning\n")
        f.write("Bias: 0.9-1.1 Unbiased, 1.1-1.5 Acceptable, >1.5 Problematic\n")

    print("main folder:", main_folder)
    print(f"\nTop 3 metrics report saved to {report_file}")
    print("\n" + "="*80 + "\n")
