import pickle
import torch
import os
import glob

main_folder="/data/home/umang/Trajectory_project/GPS-MTM/outputs/mtm_test/2025-08-18_09-31-02/test_outputs/random_masking_0.8_testing"

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

    # Get the most frequently top-k repeated in ground_truth_list_states
    unique, counts = torch.unique(ground_truth_list_states, return_counts=True)
    top_1_ground_truth = unique[torch.topk(counts, k=1).indices]
    top_5_ground_truth = unique[torch.topk(counts, k=5).indices]
    top_10_ground_truth = unique[torch.topk(counts, k=10).indices]
    top_20_ground_truth = unique[torch.topk(counts, k=20).indices]

    # Get the number of correct predictions for classes in top-k ground truth
    top_k = [top_1_ground_truth, top_5_ground_truth, top_10_ground_truth, top_20_ground_truth]
    for i in range(len(top_k)):
        filtered_gt = ground_truth_list_states[(ground_truth_list_states.unsqueeze(-1) == top_k[i]).any(dim=-1)]
        filtered_pred = predictions_list_states[(ground_truth_list_states.unsqueeze(-1) == top_k[i]).any(dim=-1)]

        # Get the number of correct predictions and accuracy
        correct_predictions = (filtered_pred == filtered_gt).sum()
        accuracy = correct_predictions / filtered_gt.numel()
        print(f"Number of correct predictions in {top_k[i]} ground truth: {correct_predictions.item()}")
        print(f"Accuracy of correct predictions in {top_k[i]} ground truth: {accuracy.item()}")
