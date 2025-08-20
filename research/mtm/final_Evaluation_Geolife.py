import pickle
import torch
import os
import glob
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import numpy as np

main_folder="/data/home/umang/Trajectory_project/GPS-MTM/outputs/test_geolife_10000/2025-08-20_08-19-12/test_outputs/random_masking_0.5_testing"

for folder in os.listdir(main_folder):
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

            ground_truth_states = batch["states"][i, :first_zero_mask, :]
            ground_truth_states = torch.argmax(ground_truth_states, dim=-1)
            ground_truth_list_states.append(ground_truth_states)

            predictions_states = predictions["states"][i, :first_zero_mask, :]
            predictions_states = torch.argmax(predictions_states, dim=-1)
            predictions_list_states.append(predictions_states)

    # Combine all batches
    ground_truth_list_states = torch.cat(ground_truth_list_states).flatten()
    predictions_list_states = torch.cat(predictions_list_states).flatten()

    print(f"Total number of ground truth states: {ground_truth_list_states.numel()}")
    print(f"Total number of predictions states: {predictions_list_states.numel()}")

    # Convert to numpy for sklearn compatibility
    gt_numpy = ground_truth_list_states.cpu().numpy()
    pred_numpy = predictions_list_states.cpu().numpy()

    # Overall metrics
    overall_accuracy = accuracy_score(gt_numpy, pred_numpy)
    macro_f1 = f1_score(gt_numpy, pred_numpy, average='macro')
    weighted_f1 = f1_score(gt_numpy, pred_numpy, average='weighted')
    macro_precision = precision_score(gt_numpy, pred_numpy, average='macro')
    weighted_precision = precision_score(gt_numpy, pred_numpy, average='weighted')
    macro_recall = recall_score(gt_numpy, pred_numpy, average='macro')
    weighted_recall = recall_score(gt_numpy, pred_numpy, average='weighted')

    print("\nOverall Performance Metrics:")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")

    # Class-wise metrics accounting for imbalance
    unique_classes = torch.unique(ground_truth_list_states)
    class_counts = [(cls.item(), (ground_truth_list_states == cls).sum().item()) for cls in unique_classes]
    class_counts.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Class':<5} {'Count':<6} {'Freq%':<6} {'Precision':<9} {'Recall':<6} {'F1':<6}")
    print("-" * 50)

    for cls, count in class_counts:
        freq_pct = (count / len(ground_truth_list_states)) * 100
        
        # Calculate metrics for this specific class
        tp = ((pred_numpy == cls) & (gt_numpy == cls)).sum()
        fp = ((pred_numpy == cls) & (gt_numpy != cls)).sum()
        fn = ((pred_numpy != cls) & (gt_numpy == cls)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{cls:<5} {count:<6} {freq_pct:<6.1f} {precision:<9.4f} {recall:<6.4f} {f1:<6.4f}")

    # Get the most frequently top-k repeated in ground_truth_list_states
    unique, counts = torch.unique(ground_truth_list_states, return_counts=True)
    top_1_ground_truth = unique[torch.topk(counts, k=1).indices]
    top_5_ground_truth = unique[torch.topk(counts, k=5).indices]
    top_10_ground_truth = unique[torch.topk(counts, k=10).indices]
    top_20_ground_truth = unique[torch.topk(counts, k=20).indices]

    print(f"\nTop-K Class Analysis:")
    print(f"Top 1 most frequent class: {top_1_ground_truth}")
    print(f"Top 5 most frequent classes: {top_5_ground_truth}")
    print(f"Top 10 most frequent classes: {top_10_ground_truth}")
    print(f"Top 20 most frequent classes: {top_20_ground_truth}")

    # Get the number of correct predictions for classes in top-k ground truth
    top_k = [top_1_ground_truth, top_5_ground_truth, top_10_ground_truth, top_20_ground_truth]
    for i in range(len(top_k)):
        filtered_gt = ground_truth_list_states[(ground_truth_list_states.unsqueeze(-1) == top_k[i]).any(dim=-1)]
        filtered_pred = predictions_list_states[(ground_truth_list_states.unsqueeze(-1) == top_k[i]).any(dim=-1)]

        # Get the number of correct predictions and accuracy
        correct_predictions = (filtered_pred == filtered_gt).sum()
        accuracy = correct_predictions / filtered_gt.numel()
        k_vals = ["1", "5", "10", "20"]
        print(f"Top-{k_vals[i]} accuracy: {accuracy.item():.4f} ({correct_predictions.item()}/{filtered_gt.numel()} correct predictions)")

    print("\nDetailed Classification Report:")
    print(classification_report(gt_numpy, pred_numpy))

    # Save the classification report to a text file in the main folder with folder name in filename
    report_file = f"{main_folder}/classification_report_{folder}.txt"
    with open(report_file, 'w') as f:
        f.write("Overall Performance Metrics:\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
        f.write(f"Weighted F1 Score: {weighted_f1:.4f}\n")
        f.write(f"Macro Precision: {macro_precision:.4f}\n")
        f.write(f"Weighted Precision: {weighted_precision:.4f}\n")
        f.write(f"Macro Recall: {macro_recall:.4f}\n")
        f.write(f"Weighted Recall: {weighted_recall:.4f}\n\n")

        f.write("Class-wise Metrics:\n")
        f.write(f"{'Class':<5} {'Count':<6} {'Freq%':<6} {'Precision':<9} {'Recall':<6} {'F1':<6}\n")
        f.write("-" * 50 + "\n")
        for cls, count in class_counts:
            freq_pct = (count / len(ground_truth_list_states)) * 100
            
            # Calculate metrics for this specific class
            tp = ((pred_numpy == cls) & (gt_numpy == cls)).sum()
            fp = ((pred_numpy == cls) & (gt_numpy != cls)).sum()
            fn = ((pred_numpy != cls) & (gt_numpy == cls)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            f.write(f"{cls:<5} {count:<6} {freq_pct:<6.1f} {precision:<9.4f} {recall:<6.4f} {f1:<6.4f}\n")

        f.write("\nTop-K Class Analysis:\n")
        f.write(f"Top 1 most frequent class: {top_1_ground_truth}\n")
        f.write(f"Top 5 most frequent classes: {top_5_ground_truth}\n")
        f.write(f"Top 10 most frequent classes: {top_10_ground_truth}\n")
        f.write(f"Top 20 most frequent classes: {top_20_ground_truth}\n")
        for i in range(len(top_k)):
            filtered_gt = ground_truth_list_states[(ground_truth_list_states.unsqueeze(-1) == top_k[i]).any(dim=-1)]
            filtered_pred = predictions_list_states[(ground_truth_list_states.unsqueeze(-1) == top_k[i]).any(dim=-1)]
            correct_predictions = (filtered_pred == filtered_gt).sum()
            accuracy = correct_predictions / filtered_gt.numel()
            k_vals = ["1", "5", "10", "20"]
            f.write(f"Top-{k_vals[i]} accuracy: {accuracy.item():.4f} ({correct_predictions.item()}/{filtered_gt.numel()} correct predictions)\n")
        
        f.write("\nDetailed Classification Report:\n")
        f.write(classification_report(gt_numpy, pred_numpy))
    
    print("main folder:", main_folder)
    print(f"\nClassification report saved to {report_file}")
    print("\n" + "="*80 + "\n")
