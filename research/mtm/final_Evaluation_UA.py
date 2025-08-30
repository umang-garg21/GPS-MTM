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

main_folder="/data/home/umang/Trajectory_project/GPS-MTM/outputs/test_UA_berlin_social_outliers/2025-08-29_16-24-47/test_outputs/random_masking_0.15_testing"

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
    # macro_f1 = f1_score(gt_numpy, pred_numpy, average='macro')
    # weighted_f1 = f1_score(gt_numpy, pred_numpy, average='weighted')
    # macro_precision = precision_score(gt_numpy, pred_numpy, average='macro')
    # weighted_precision = precision_score(gt_numpy, pred_numpy, average='weighted')
    # macro_recall = recall_score(gt_numpy, pred_numpy, average='macro')
    # weighted_recall = recall_score(gt_numpy, pred_numpy, average='weighted')

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
    print(f"\n🎯 TOP 3 KEY METRICS:")
    print(f"1. Overall Accuracy: {overall_accuracy:.4f}")
    print(f"2. Recall Range: {recall_range:.3f} (consistency across classes)")
    print(f"3. Bias Ratio: {bias_ratio:.2f}x (1.0 = unbiased)")
    
    # Performance Assessment
    if overall_accuracy > 0.97:
        acc_status = "🏆 Excellent"
    elif overall_accuracy > 0.90:
        acc_status = "✅ Good"
    else:
        acc_status = "⚠️  Needs Improvement"
        
    if recall_range < 0.05:
        range_status = "🏆 Excellent Consistency"
    elif recall_range < 0.15:
        range_status = "✅ Good Consistency"
    else:
        range_status = "⚠️  Concerning Gaps"
        
    if 0.9 <= bias_ratio <= 1.1:
        bias_status = "🏆 Unbiased"
    elif 0.8 <= bias_ratio <= 1.3:
        bias_status = "✅ Acceptable Bias"
    else:
        bias_status = "⚠️  Problematic Bias"
    
    print(f"\n📊 ASSESSMENT:")
    print(f"Accuracy: {acc_status}")
    print(f"Consistency: {range_status}")
    print(f"Fairness: {bias_status}")

    # ========== COMMENTED OUT: DETAILED ANALYSIS ==========
    # print("\nOverall Performance Metrics:")
    # print(f"Manual Accuracy: {manual_accuracy:.4f}")
    # print(f"Overall Accuracy: {overall_accuracy:.4f}")
    # print(f"Macro F1 Score: {macro_f1:.4f}")
    # print(f"Weighted F1 Score: {weighted_f1:.4f}")
    
    # print(f"\n🔍 CLASS IMBALANCE ANALYSIS:")
    # print(f"Imbalance Ratio: {imbalance_ratio:.2f}x (most/least frequent)")
    # print(f"Effective Number of Classes: {effective_num_classes:.2f} (out of {len(unique_classes)})")
    # print(f"Dataset Entropy: {entropy:.3f}")
    
    # print(f"\n⚖️  MODEL BIAS ANALYSIS:")
    # print(f"Majority Class Prediction Bias: {majority_bias:.3f} (vs true freq: {majority_true_freq:.3f})")
    # print(f"Bias Ratio: {majority_bias/majority_true_freq:.2f}x")
    
    # print(f"\n📊 PERFORMANCE GAPS:")
    # print(f"Recall Range: {recall_range:.3f} (max-min across classes)")
    # print(f"Precision Range: {precision_range:.3f}")
    # print(f"F1 Range: {f1_range:.3f}")
    
    # print(f"\n🎯 MINORITY vs MAJORITY CLASS PERFORMANCE:")
    # print(f"Majority Classes Avg Recall: {majority_avg_recall:.3f}")
    # print(f"Minority Classes Avg Recall: {minority_avg_recall:.3f}")
    # print(f"Recall Gap: {majority_avg_recall - minority_avg_recall:.3f}")
    
    # print(f"\n📈 DETAILED METRICS:")
    # print(f"Macro Precision: {macro_precision:.4f}")
    # print(f"Weighted Precision: {weighted_precision:.4f}")
    # print(f"Macro Recall: {macro_recall:.4f}")
    # print(f"Weighted Recall: {weighted_recall:.4f}")

    # print(f"\n📋 CLASS-WISE PERFORMANCE (sorted by frequency):")
    # print(f"{'Class':<5} {'Count':<7} {'Freq%':<6} {'Recall':<7} {'Prec':<7} {'F1':<6} {'Status':<12}")
    # print("-" * 65)

    # for i, (cls, count) in enumerate(class_counts):
    #     freq_pct = (count / len(ground_truth_list_states)) * 100
    #     recall = per_class_recalls[i]
    #     precision = per_class_precisions[i]
    #     f1 = per_class_f1s[i]
        
    #     # Add status indicators
    #     if recall < 0.3:
    #         status = "🔴 Poor"
    #     elif recall < 0.6:
    #         status = "🟡 Fair"
    #     else:
    #         status = "🟢 Good"
            
    #     if freq_pct < 5:
    #         status += " (Rare)"
    #     elif freq_pct > 30:
    #         status += " (Freq)"
        
    #     print(f"{cls:<5} {count:<7} {freq_pct:<6.1f} {recall:<7.3f} {precision:<7.3f} {f1:<6.3f} {status:<12}")

    # ========== COMMENTED OUT: TOP-K ANALYSIS ==========
    # Get the most frequently top-k repeated in ground_truth_list_states
    # unique, counts = torch.unique(ground_truth_list_states, return_counts=True)
    # top_1_ground_truth = unique[torch.topk(counts, k=1).indices]
    # top_4_ground_truth = unique[torch.topk(counts, k=4).indices]
    # top_10_ground_truth = unique[torch.topk(counts, k=10).indices]
    # top_20_ground_truth = unique[torch.topk(counts, k=20).indices]

    # print(f"\nTop-K Class Analysis:")
    # print(f"Top 1 most frequent class: {top_1_ground_truth}")
    # print(f"Top 4 most frequent classes: {top_4_ground_truth}")

    # Get the number of correct predictions for classes in top-k ground truth
    # top_k = [top_1_ground_truth, top_4_ground_truth]
    # for i in range(len(top_k)):
    #     filtered_gt = ground_truth_list_states[(ground_truth_list_states.unsqueeze(-1) == top_k[i]).any(dim=-1)]
    #     filtered_pred = predictions_list_states[(ground_truth_list_states.unsqueeze(-1) == top_k[i]).any(dim=-1)]

    #     # Get the number of correct predictions and accuracy
    #     correct_predictions = (filtered_pred == filtered_gt).sum()
    #     accuracy = correct_predictions / filtered_gt.numel()
    #     k_val = "1" if i == 0 else "4"
    #     print(f"Top-{k_val} accuracy: {accuracy.item():.4f} ({correct_predictions.item()}/{filtered_gt.numel()} correct predictions)")

    # print("\nDetailed Classification Report:")
    # print(classification_report(gt_numpy, pred_numpy))

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

    # ========== COMMENTED OUT: DETAILED REPORT GENERATION ==========
    # Save the classification report to a text file in the main folder with folder name in filename
    # report_file = f"{main_folder}/classification_report_{folder}.txt"
    # with open(report_file, 'w') as f:
    #     f.write("Overall Performance Metrics:\n")
    #     f.write(f"Manual Accuracy: {manual_accuracy:.4f}\n")
    #     f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
    #     f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
    #     f.write(f"Weighted F1 Score: {weighted_f1:.4f}\n")
        
    #     f.write(f"\nCLASS IMBALANCE ANALYSIS:\n")
    #     f.write(f"Imbalance Ratio: {imbalance_ratio:.2f}x (most/least frequent)\n")
    #     f.write(f"Effective Number of Classes: {effective_num_classes:.2f} (out of {len(unique_classes)})\n")
    #     f.write(f"Dataset Entropy: {entropy:.3f}\n")
        
    #     f.write(f"\nMODEL BIAS ANALYSIS:\n")
    #     f.write(f"Majority Class Prediction Bias: {majority_bias:.3f} (vs true freq: {majority_true_freq:.3f})\n")
    #     f.write(f"Bias Ratio: {majority_bias/majority_true_freq:.2f}x\n")
        
    #     f.write(f"\nPERFORMANCE GAPS:\n")
    #     f.write(f"Recall Range: {recall_range:.3f} (max-min across classes)\n")
    #     f.write(f"Precision Range: {precision_range:.3f}\n")
    #     f.write(f"F1 Range: {f1_range:.3f}\n")
        
    #     f.write(f"\nMINORITY vs MAJORITY CLASS PERFORMANCE:\n")
    #     f.write(f"Majority Classes Avg Recall: {majority_avg_recall:.3f}\n")
    #     f.write(f"Minority Classes Avg Recall: {minority_avg_recall:.3f}\n")
    #     f.write(f"Recall Gap: {majority_avg_recall - minority_avg_recall:.3f}\n")
        
    #     f.write(f"\nDETAILED METRICS:\n")
    #     f.write(f"Macro Precision: {macro_precision:.4f}\n")
    #     f.write(f"Weighted Precision: {weighted_precision:.4f}\n")
    #     f.write(f"Macro Recall: {macro_recall:.4f}\n")
    #     f.write(f"Weighted Recall: {weighted_recall:.4f}\n\n")

    #     f.write("Class-wise Metrics:\n")
    #     f.write(f"{'Class':<5} {'Count':<7} {'Freq%':<6} {'Recall':<7} {'Prec':<7} {'F1':<6}\n")
    #     f.write("-" * 50 + "\n")
    #     for i, (cls, count) in enumerate(class_counts):
    #         freq_pct = (count / len(ground_truth_list_states)) * 100
    #         recall = per_class_recalls[i]
    #         precision = per_class_precisions[i]
    #         f1 = per_class_f1s[i]
    #         f.write(f"{cls:<5} {count:<7} {freq_pct:<6.1f} {recall:<7.3f} {precision:<7.3f} {f1:<6.3f}\n")

    #     f.write("\nTop-K Class Analysis:\n")
    #     f.write(f"Top 1 most frequent class: {top_1_ground_truth}\n")
    #     f.write(f"Top 4 most frequent classes: {top_4_ground_truth}\n")
    #     for i in range(len(top_k)):
    #         filtered_gt = ground_truth_list_states[(ground_truth_list_states.unsqueeze(-1) == top_k[i]).any(dim=-1)]
    #         filtered_pred = predictions_list_states[(ground_truth_list_states.unsqueeze(-1) == top_k[i]).any(dim=-1)]
    #         correct_predictions = (filtered_pred == filtered_gt).sum()
    #         accuracy = correct_predictions / filtered_gt.numel()
    #         k_val = "1" if i == 0 else "4"
    #         f.write(f"Top-{k_val} accuracy: {accuracy.item():.4f} ({correct_predictions.item()}/{filtered_gt.numel()} correct predictions)\n")
        
    #     f.write("\nDetailed Classification Report:\n")
    #     f.write(classification_report(gt_numpy, pred_numpy))
    
    print("main folder:", main_folder)
    print(f"\nTop 3 metrics report saved to {report_file}")
    print("\n" + "="*80 + "\n")
