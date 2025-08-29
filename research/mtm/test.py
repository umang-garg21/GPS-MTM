# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Main script for testing/evaluating a trained policy on a dataset.
"""
import os
import pprint
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Sequence, Tuple

import hydra
import numpy as np
import torch
import torch.distributed
import torch.multiprocessing
import torch.nn.functional as F
import torch.nn.parallel
import wandb

# Print the Python executable path to verify the environment
print(f"Python executable: {sys.executable}")
import sys

from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataloader import DataLoader

sys.path.append("/data/home/umang/Trajectory_project/GPS-MTM")

from research.logger import WandBLogger, WandBLoggerConfig, logger, stopwatch
from research.mtm.datasets.base import DatasetProtocol
from research.mtm.distributed_utils import DistributedParams, get_distributed_params
from research.mtm.masks import (
    MaskType,
    create_bc_mask,
    create_forward_dynamics_mask,
    create_full_random_masks,
    create_goal_n_reaching_masks,
    create_goal_reaching_masks,
    create_inverse_dynamics_mask,
    create_random_autoregressize_mask,
    create_random_bc_masks,
    create_random_mask,
    create_random_masks,
    create_rcbc_mask,
    maybe_add_rew_to_mask,
)
from research.mtm.models.mtm_model import MTM, make_plots_with_masks
from research.mtm.tokenizers.base import Tokenizer, TokenizerManager
from research.mtm.tokenizers.continuous import ContinuousTokenizer
from research.mtm.utils import (
    get_cfg_hash,
    get_ckpt_path_from_folder,
    get_git_dirty,
    get_git_hash,
    set_seed_everywhere,
)
dir_path = os.path.dirname(os.path.realpath(__file__))


def eval_fd(
    model: MTM,
    env,
    eval_batch,
    tokenizer_manager,
    ratio: int = 1,
) -> Dict[str, Any]:
    """Evaluate the model on the forward dynamics task.
    Args:
        env (gym.Env): env
        eval_batch (Dict[str, torch.Tensor]): eval_batch
        tokenizer_manager (TokenizerManager): tokenizer_manager
    """
    seq_len = eval_batch["actions"].shape[1]
    device = eval_batch["states"].device
    assert seq_len >= 2, "Forward dynamics eval only works for seq_len=2"

    # Given initial state and all actions. Predict future states.
    obs_mask1 = torch.ones(seq_len, device=device)
    obs_mask1[-1] = 0
    actions_mask1 = torch.zeros(seq_len, device=device)
    actions_mask1[-2] = 1
    returns_mask = torch.zeros(seq_len, device=device)
    masks = {
        "states": obs_mask1,
        "actions": actions_mask1,
    }
    attention_masks = (eval_batch.pop("attention_masks", None),)
    encoded_batch = tokenizer_manager.encode(
        eval_batch, attention_masks=attention_masks
    )
    predictions = model.mask_git_forward(encoded_batch, masks, ratio=ratio)
    predicted_next_state = tokenizer_manager.decode(predictions)["states"]

    states = eval_batch["states"]
    next_state = states[:, -1]
    state_error = (next_state - predicted_next_state[:, -1, :]) ** 2
    eval_dict = {}
    eval_dict[f"eval/fd_state_error_r={ratio}"] = torch.mean(state_error).item()
    return eval_dict


def eval_id(
    model: MTM, env, eval_batch, tokenizer_manager, ratio: int = 1
) -> Dict[str, Any]:
    """Evaluate the model on the inverse dynamics task.
    Args:
        env (gym.Env): env
        eval_batch (Dict[str, torch.Tensor]): eval_batch
        tokenizer_manager (TokenizerManager): tokenizer_manager
    """
    seq_len = eval_batch["actions"].shape[1]
    B, T, S = eval_batch["states"].shape
    device = eval_batch["states"].device
    assert seq_len >= 2, "Forward dynamics eval only works for seq_len=2"

    # Given all states. Predict second to last action.
    obs_mask1 = torch.ones(seq_len, device=device)
    actions_mask1 = torch.zeros(seq_len, device=device)
    returns_mask = torch.zeros(seq_len, device=device)
    masks = {
        "states": obs_mask1,
        "actions": actions_mask1,
        "returns": returns_mask,
    }
    attention_masks = eval_batch.pop("attention_masks", None)
    encoded_batch = tokenizer_manager.encode(
        eval_batch, attention_masks=attention_masks
    )
    predictions = model.mask_git_forward(encoded_batch, masks, ratio=ratio)
    predicted_actions = tokenizer_manager.decode(predictions)["actions"]
    predicted_action = predicted_actions[:, -2, :]

    state_error = []
    gt_state_error = []
    action_error = []

    states = eval_batch["states"]
    actions = eval_batch["actions"]
    actions = eval_batch["actions"][:, -2, :]

    # MSE loss on the actions
    action_error = ((predicted_action - actions) ** 2).mean()
    eval_dict = {}
    eval_dict[f"eval/id_action_error_r={ratio}"] = torch.mean(
        torch.tensor(action_error)
    ).item()
    return eval_dict

    for i in range(B):
        # set state to be the second to last state
        env.reset()
        phys_state = np.zeros(S + 2)
        phys_state[2:] = states[i, T - 2].detach().cpu().numpy()
        env.sim.set_state_from_flattened(phys_state.copy())
        env.sim.forward()
        # get the action from the model
        action = predicted_action[i].detach().cpu().numpy()
        action = np.clip(action, -1, 1)

        # get the ground truth action
        gt_action = actions[i, T - 2].detach().cpu().numpy()
        # get the next state
        next_state = states[i, T - 1].detach().cpu().numpy()
        # get the next state from the model
        next_state_model = env.step(action)[0]

        # reset and test groud truth action
        env.reset()
        env.sim.set_state_from_flattened(phys_state.copy())
        env.sim.forward()
        next_state_gt = env.step(gt_action)[0]
        qpos_size = env.sim.data.qpos.shape[0]

        # compute action error
        action_error.append((action - gt_action) ** 2)
        # compute state error
        state_error.append((next_state[:qpos_size] - next_state_model[:qpos_size]) ** 2)
        gt_error = (next_state[:qpos_size] - next_state_gt[:qpos_size]) ** 2

        # if np.sum(gt_error) > 1e-7:
        #     print(gt_error)
        #     import ipdb; ipdb.set_trace();
        #     print("minor")

        gt_state_error.append(gt_error)

    eval_dict = {}
    eval_dict[f"eval/id_state_error_r={ratio}"] = torch.mean(
        torch.tensor(state_error)
    ).item()
    eval_dict[f"eval/id_action_error_r={ratio}"] = torch.mean(
        torch.tensor(action_error)
    ).item()
    eval_dict[f"eval/id_gt_state_error_r={ratio}"] = torch.mean(
        torch.tensor(gt_state_error)
    ).item()
    return eval_dict


def eval_full_id(
    model: MTM, env, eval_batch, tokenizer_manager, ratio: int = 1
) -> Dict[str, Any]:
    """Evaluate the model on the inverse dynamics task.
    Args:
        env (gym.Env): env
        eval_batch (Dict[str, torch.Tensor]): eval_batch
        tokenizer_manager (TokenizerManager): tokenizer_manager
    """
    seq_len = eval_batch["actions"].shape[1]
    B, T, S = eval_batch["states"].shape
    device = eval_batch["states"].device
    assert seq_len >= 2, "Forward dynamics eval only works for seq_len=2"

    # Given all states. Predict ALL actions.
    obs_mask1 = torch.ones(seq_len, device=device)
    actions_mask1 = torch.zeros(seq_len, device=device)
    returns_mask = torch.zeros(seq_len, device=device)
    masks = {
        "states": obs_mask1,
        "actions": actions_mask1,
        "returns": returns_mask,
    }

    attention_masks = eval_batch.pop("attention_masks", None)
    encoded_batch = tokenizer_manager.encode(
        eval_batch, attention_masks=attention_masks
    )
    predictions = model.mask_git_forward(encoded_batch, masks, ratio=ratio)

    predicted_actions = tokenizer_manager.decode(predictions)["actions"]
    actions = eval_batch["actions"]
    action_error = ((predicted_actions - actions) ** 2).mean()

    eval_dict = {}
    eval_dict[f"eval/full_id_action_error_r={ratio}"] = torch.mean(
        torch.tensor(action_error)
    ).item()
    return eval_dict


def create_eval_logs_states_actions_images(
    predict_fn: Callable,
    trajectories: Dict[str, torch.Tensor],
    tokenizer_manager: TokenizerManager,
) -> Dict[str, Any]:
    eval_logs = {}
    assert "states" in trajectories
    assert "actions" in trajectories
    device = trajectories["states"].device
    seq_len = trajectories["states"].shape[1]

    # Given initial state and all actions. Predict future states.
    obs_mask1 = np.ones(seq_len)
    obs_mask1[1:] = 0
    actions_mask1 = np.ones(seq_len)

    obs_mask2 = np.ones(seq_len)
    obs_mask2[1:-1] = 0
    actions_mask2 = np.zeros(seq_len)

    obs_mask3 = np.ones(seq_len)
    obs_mask3[1:-1] = 0
    obs_mask3[::16] = 1
    actions_mask3 = np.zeros(seq_len)

    obs_mask4 = np.ones(seq_len)
    actions_mask4 = np.zeros(seq_len)

    rnd = np.random.RandomState(0)
    obs_mask5 = create_random_mask(seq_len, 0.15, device, rnd).detach().cpu().numpy()
    actions_mask5 = (
        create_random_mask(seq_len, 0.15, device, rnd).detach().cpu().numpy()
    )

    obs_use_mask_list = [
        obs_mask1,
        obs_mask2,
        obs_mask3,
        obs_mask4,
        obs_mask5,
    ]
    actions_use_mask_list = [
        actions_mask1,
        actions_mask2,
        actions_mask3,
        actions_mask4,
        actions_mask5,
    ]
    masks_list = []
    for obs_mask, actions_mask in zip(obs_use_mask_list, actions_use_mask_list):
        masks_list.append(
            {
                "states": torch.from_numpy(np.zeros_like(obs_mask)).to(device),
                "images": torch.from_numpy(obs_mask).to(device),
                "actions": torch.from_numpy(actions_mask).to(device),
            }
        )

    r1 = create_random_mask(seq_len, 0.15, device, rnd).detach().cpu().numpy()
    r2 = create_random_mask(seq_len, 0.15, device, rnd).detach().cpu().numpy()
    r3 = create_random_mask(seq_len, 0.15, device, rnd).detach().cpu().numpy()
    masks_list.append(
        {
            "states": torch.from_numpy(r1).to(device),
            "images": torch.from_numpy(r2).to(device),
            "actions": torch.from_numpy(r3).to(device),
        }
    )

    prefixs = ["f_dynamics", "goal", "goal_32", "inv_dynamics", "random", "random_all"]
    return make_plots_with_masks(
        predict_fn,
        trajectories,
        tokenizer_manager,
        masks_list,
        prefixs,
        max_n_plots=1,
    )


def create_eval_logs_actions_images(
    predict_fn: Callable,
    trajectories: Dict[str, torch.Tensor],
    tokenizer_manager: TokenizerManager,
    rewards: bool = False,
) -> Dict[str, Any]:
    eval_logs = {}
    assert "images" in trajectories
    assert "actions" in trajectories
    device = trajectories["images"].device
    seq_len = trajectories["images"].shape[1]

    # Given initial state and all actions. Predict future states.
    obs_mask1 = np.ones(seq_len)
    obs_mask1[1:] = 0
    actions_mask1 = np.ones(seq_len)

    obs_mask2 = np.ones(seq_len)
    obs_mask2[1:-1] = 0
    actions_mask2 = np.zeros(seq_len)

    obs_mask3 = np.ones(seq_len)
    obs_mask3[1:-1] = 0
    obs_mask3[::16] = 1
    actions_mask3 = np.zeros(seq_len)

    obs_mask4 = np.ones(seq_len)
    actions_mask4 = np.zeros(seq_len)

    rnd = np.random.RandomState(0)
    obs_mask5 = create_random_mask(seq_len, 0.15, device, rnd).detach().cpu().numpy()
    actions_mask5 = (
        create_random_mask(seq_len, 0.15, device, rnd).detach().cpu().numpy()
    )

    obs_use_mask_list = [
        obs_mask1,
        obs_mask2,
        obs_mask3,
        obs_mask4,
        obs_mask5,
    ]
    actions_use_mask_list = [
        actions_mask1,
        actions_mask2,
        actions_mask3,
        actions_mask4,
        actions_mask5,
    ]
    masks_list = []
    for obs_mask, actions_mask in zip(obs_use_mask_list, actions_use_mask_list):
        masks_list.append(
            {
                "images": torch.from_numpy(obs_mask).to(device),
                "actions": torch.from_numpy(actions_mask).to(device),
            }
        )
        if rewards:
            masks_list[-1]["rewards"] = masks_list[-1]["images"].clone()

    prefixs = ["f_dynamics", "goal", "goal_32", "inv_dynamics", "random"]
    return make_plots_with_masks(
        predict_fn,
        trajectories,
        tokenizer_manager,
        masks_list,
        prefixs,
        max_n_plots=2,
    )


def create_eval_logs_states_actions(
    predict_fn: Callable,
    trajectories: Dict[str, torch.Tensor],
    tokenizer_manager: TokenizerManager,
    rewards: bool = False,
) -> Dict[str, Any]:
    eval_logs = {}
    assert "states" in trajectories
    assert "actions" in trajectories
    device = trajectories["states"].device
    seq_len = trajectories["states"].shape[1]

    # Given initial state and all actions. Predict future states.
    obs_mask1 = np.ones(seq_len)
    obs_mask1[1:] = 0
    actions_mask1 = np.ones(seq_len)

    obs_mask2 = np.ones(seq_len)
    obs_mask2[1:-1] = 0
    actions_mask2 = np.zeros(seq_len)

    obs_mask3 = np.ones(seq_len)
    obs_mask3[1:-1] = 0
    obs_mask3[::16] = 1
    actions_mask3 = np.zeros(seq_len)

    obs_mask4 = np.ones(seq_len)
    actions_mask4 = np.zeros(seq_len)

    rnd = np.random.RandomState(0)
    obs_mask5 = create_random_mask(seq_len, 0.15, device, rnd).detach().cpu().numpy()
    actions_mask5 = (
        create_random_mask(seq_len, 0.15, device, rnd).detach().cpu().numpy()
    )

    obs_use_mask_list = [
        obs_mask1,
        obs_mask2,
        obs_mask3,
        obs_mask4,
        obs_mask5,
    ]
    actions_use_mask_list = [
        actions_mask1,
        actions_mask2,
        actions_mask3,
        actions_mask4,
        actions_mask5,
    ]
    masks_list = []
    for obs_mask, actions_mask in zip(obs_use_mask_list, actions_use_mask_list):
        masks_list.append(
            {
                "states": torch.from_numpy(obs_mask).to(device),
                "actions": torch.from_numpy(actions_mask).to(device),
            }
        )
        if rewards:
            masks_list[-1]["rewards"] = masks_list[-1]["states"].clone()
            masks_list[-1]["returns"] = masks_list[-1]["states"].clone()

    prefixs = ["f_dynamics", "goal", "goal_32", "inv_dynamics", "random"]
    return make_plots_with_masks(
        predict_fn,
        trajectories,
        tokenizer_manager,
        masks_list,
        prefixs,
        max_n_plots=2,
    )


def create_eval_logs_states(
    predict_fn: Callable,
    trajectories: Dict[str, torch.Tensor],
    tokenizer_manager: TokenizerManager,
) -> Dict[str, Any]:
    assert "states" in trajectories
    eval_logs = {}
    device = trajectories["states"].device
    seq_len = trajectories["states"].shape[1]

    # Given initial state and all actions. Predict future states.
    obs_mask3 = np.ones(seq_len)
    obs_mask3[seq_len // 2 + 2 :] = 0

    obs_use_mask_list = [
        obs_mask3,
    ]

    masks_list = []
    for obs_mask in obs_use_mask_list:
        masks_list.append(
            {
                "states": torch.from_numpy(obs_mask).to(device),
            }
        )

    prefixs = ["prediction"]
    return make_plots_with_masks(
        predict_fn,
        trajectories,
        tokenizer_manager,
        masks_list,
        prefixs,
    )

@dataclass
class TestConfig:
    seed: int = 0
    """RNG seed."""

    batch_size: int = 64
    """Batch size used during testing."""

    n_workers: int = 8
    """Number of workers for loading data."""

    device: str = "cuda"
    """Device to use for testing."""

    mask_ratios: Sequence[float] = (0.15, 0.35, 0.55, 0.75, 0.85, 0.95)

    mask_patterns: Sequence[str] = ("RANDOM",)
    """Indices of masks to use for evaluation."""

    traj_length: int = 1
    """Trajectory length."""

    mode_order: Tuple[str, str, str] = ("states", "returns", "actions")
    """Mode order for autoregressive masking."""

    mode_weights: Tuple[int, int, int] = (0.2, 0.1, 0.7)
    """State action return."""

    model_path: str = ""
    """Path to the trained model checkpoint."""

    save_predictions: bool = True
    """Whether to save predictions to disk."""

    output_dir: str = "test_outputs"
    """Directory to save test outputs."""

    test_name: str = "default_test"
    """Custom name for this test run."""

    load_test_only: bool = True
    """Whether to load only test/validation data (True) or both train and validation data (False).
    Setting to True saves memory and loading time during testing."""

@torch.inference_mode()
def evaluate(
    model: MTM,
    tokenizer_manager: TokenizerManager,
    discrete_map: Dict[str, bool],
    val_batch: Dict[str, torch.Tensor],
    vis_batch: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    
    attention_masks = val_batch.pop("attention_mask", None)
    encoded_batch = tokenizer_manager.encode(val_batch, attention_masks=attention_masks)

    predicted_trajectories = model(encoded_batch, masks, attention_masks=attention_masks)

    model_without_ddp = model.module if hasattr(model, "module") else model
    (
        loss,
        losses_dict,
        masked_losses,
        masked_c_losses,
        masked_c_loss_per_feature_k,
    ) = MTM.forward_loss(
        encoded_batch,
        predicted_trajectories,
        masks,
        discrete_map,
        norm=model_without_ddp.norm,
        reduce_use_sum=model_without_ddp.config.reduce_use_sum,
        loss_keys=model_without_ddp.config.loss_keys,
        attention_masks=attention_masks,
    )

    log_dict = {"val/val_loss": loss.item()}
    for k, v in losses_dict.items():
        log_dict[f"val/full_loss_{k}"] = v.item()
    if len(masked_c_losses.keys()) > 1:
        for k, v in masked_c_losses.items():
            log_dict[f"val/masked_c_loss_{k}"] = v.item()
    else:
        log_dict["val/masked_c_loss"] = masked_c_losses["actions"].item()
    if len(masked_c_loss_per_feature_k.keys()) > 1:
        for k, v in masked_c_loss_per_feature_k.items():
            if v.numel() == 1:
                log_dict[f"val/masked_c_loss_per_feature_{k}"] = v.item()
            else:
                log_dict[f"val/masked_c_loss_per_feature_{k}"] = v.mean().item()
    else:
        v = masked_c_loss_per_feature_k["actions"]
        if v.numel() == 1:
            log_dict["val/masked_c_loss_per_feature"] = v.item()
        else:
            log_dict["val/masked_c_loss_per_feature"] = v.mean().item()
    if isinstance(masked_losses, dict) and len(masked_losses) > 1:
        for k, v in masked_losses.items():
            log_dict[f"val/masked_loss_{k}"] = v.item()
    elif isinstance(masked_losses, dict):
        log_dict["val/masked_loss"] = masked_losses["actions"].item()
    else:
        log_dict["val/masked_loss"] = masked_losses.item()
    if masked_c_losses and isinstance(masked_c_losses, dict) and len(masked_c_losses) > 1:
        log_dict["val/masked_c_loss_sum"] = sum(masked_c_losses.values()).item()
    elif isinstance(masked_c_losses, dict):
        log_dict["val/masked_c_loss_sum"] = masked_c_losses["actions"].item()
    else:
        log_dict["val/masked_c_loss_sum"] = masked_c_losses.item()


    mse_loss = 0
    predictions = tokenizer_manager.decode(predicted_trajectories)

    return log_dict


@torch.inference_mode()
def test_model_on_batch(
    model: MTM,
    tokenizer_manager: TokenizerManager,
    discrete_map: Dict[str, bool],
    batch: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    batch_idx: int = 0,
    mask_pattern: str = "RANDOM",
    save_predictions: bool = True,
    output_dir: str = "test_outputs",
) -> Dict[str, Any]:
    """Test the model on a single batch and optionally save predictions."""
    # Include attention masks in encoding
    attention_masks = batch.pop("attention_mask", None)
    
    # Additional check for empty sequences (secondary validation)
    if attention_masks is not None:
        seq_lengths = attention_masks.sum(dim=-1)
        if (seq_lengths == 0).any():
            print(f"Warning: Empty sequences detected in test_model_on_batch for batch {batch_idx}")
            # Return dummy results for empty batch
            return {
                "test/test_loss": 0.0,
                "test/mse_sum": 0.0,
                "test/ce_sum": 0.0,
                "mask_pattern": mask_pattern,
                "batch_idx": batch_idx,
                "skipped": True
            }

    # Clone the batch
    batch_clone = {k: v.clone() for k, v in batch.items()}
    encoded_batch = tokenizer_manager.encode(batch_clone, attention_masks=attention_masks)
    _ = encoded_batch.pop("attention_masks", None)
    
    # DEBUG: Print mask information
    print(f"\n=== DEBUG MASK INFO for batch {batch_idx} ===")
    for k, mask in masks.items():
        if attention_masks is not None:
            valid_positions = attention_masks.sum().item()
            # CRITICAL: Check mask convention
            visible_positions = ((mask == 1) & (attention_masks == 1)).sum().item()
            masked_positions = ((mask == 0) & (attention_masks == 1)).sum().item()
            
            print(f"{k}: valid={valid_positions}, VISIBLE(mask=1)={visible_positions}, MASKED(mask=0)={masked_positions}")
            print(f"{k} mask convention: 1=VISIBLE/GIVEN, 0=MASKED/TO_PREDICT")
            print(f"{k} mask pattern (first 20): {mask[:20]}")
            print(f"{k} attention mask (first 20): {attention_masks[0][:20]}")
            
            # Verify that we have some masked positions
            if masked_positions == 0:
                print(f"WARNING: No masked positions for {k}! This explains 100% accuracy.")
            
        else:
            visible_positions = (mask == 1).sum().item()
            masked_positions = (mask == 0).sum().item()
            total_positions = len(mask)
            print(f"{k}: total={total_positions}, VISIBLE(mask=1)={visible_positions}, MASKED(mask=0)={masked_positions}")
            print(f"{k} mask pattern (first 20): {mask[:20]}")
            
            if masked_positions == 0:
                print(f"WARNING: No masked positions for {k}! This explains 100% accuracy.")
    print("=== END DEBUG MASK INFO ===\n")
    
    # Forward pass with attention masks
    predicted_trajectories = model(encoded_batch, masks, attention_masks=attention_masks)
    
    print(f"\n=== DEBUG MODEL INPUT/OUTPUT for batch {batch_idx} ===")
    
    # Check the original input vs the encoded input
    for k in batch.keys():
        if k in encoded_batch:
            print(f"\n{k}:")
            print(f"  Original shape: {batch[k].shape}")
            print(f"  Encoded shape: {encoded_batch[k].shape}")
            print(f"  Prediction shape: {predicted_trajectories[k].shape}")
            
            # # Show some sample values for first sequence
            # print(f"  Original values (first 5): {batch[k][0, :5].flatten()}")
            # print(f"  Predicted values (first 5): {predicted_trajectories[k][0, :5].flatten()}")
            
            # Check if the model is just copying visible tokens
            mask = masks[k]
            visible_mask = (mask == 1)
            if attention_masks is not None:
                visible_mask = visible_mask & (attention_masks[0] == 1)
            
            if visible_mask.sum() > 0:
                # For visible positions, compare input vs output to see if model is just copying
                visible_positions = visible_mask.nonzero().flatten()[:3]  # First 3 visible positions
                for pos in visible_positions:
                    orig_val = batch[k][0, pos]
                    pred_val = predicted_trajectories[k][0, pos]
                    print("key:", k)
                    print(f"  Visible pos {pos}: original={orig_val} vs predicted={pred_val}")
    
    print("=== END DEBUG MODEL INPUT/OUTPUT ===\n")
    
    # Compute the loss
    model_without_ddp = model.module if hasattr(model, "module") else model
    loss_keys = model_without_ddp.config.loss_keys

    loss, losses_dict, total_masked_loss, masked_c_losses, masked_c_loss_per_feature = MTM.forward_loss(
        encoded_batch,
        predicted_trajectories,
        masks,
        discrete_map,
        norm=model_without_ddp.norm,
        reduce_use_sum=model_without_ddp.config.reduce_use_sum,
        loss_keys=loss_keys,
        attention_masks=attention_masks,
    )

    # Create a dictionary to log all of the losses
    log_dict = {"test/test_loss": loss.item()}
    for k, v in losses_dict.items():
        log_dict[f"test/full_loss_{k}"] = v.item()
    for k, v in masked_c_losses.items():
        log_dict[f"test/masked_loss_{k}"] = v.item()

    # Calculate MSE losses
    mse_loss = 0
    ce_loss=0
    predictions = tokenizer_manager.decode(predicted_trajectories, attention_masks=attention_masks)
    
    if save_predictions:
        # Save predictions and ground truth
        output_dir= output_dir+"/"+mask_pattern
        os.makedirs(output_dir, exist_ok=True)
        import pickle

        with open(f"{output_dir}/predictions_batch_{batch_idx}.pkl", "wb") as f:
            pickle.dump({k: v.detach().cpu() for k, v in predictions.items()}, f)
        with open(f"{output_dir}/ground_truth_batch_{batch_idx}.pkl", "wb") as f:
            pickle.dump({k: v.detach().cpu() for k, v in batch.items()}, f)
        if attention_masks is not None:
            with open(f"{output_dir}/attention_masks_batch_{batch_idx}.pkl", "wb") as f:
                pickle.dump(attention_masks.detach().cpu(), f)
        with open(f"{output_dir}/masks_batch_{batch_idx}.pkl", "wb") as f:
            pickle.dump({k: v.detach().cpu() for k, v in masks.items()}, f)

    # Calculate accuracy metrics only for masked/predicted positions
    print(f"\n=== DEBUG ACCURACY CALCULATION for batch {batch_idx} ===")
    for k, v in predictions.items():
        if k == "states":
            # For discrete states: calculate cross-entropy and accuracy for MASKED positions only
            mask_positions = (masks[k] == 0) & (attention_masks == 1)  # Masked positions that are valid
            
            print(f"States - Total mask positions: {mask_positions.sum().item()}")
            print(f"States - Prediction shape: {v.shape}, Ground truth shape: {batch[k].shape}")
            
            if mask_positions.sum() > 0:
                # v is logits, batch[k] should be class indices
                predicted_logits = v[mask_positions]  # Shape: [num_masked_tokens, num_classes]
                ground_truth_tensor = batch[k][mask_positions]  # Shape: [num_masked_tokens, num_classes] or [num_masked_tokens]
                
                print(f"States - Predicted logits shape: {predicted_logits.shape}")
                print(f"States - Ground truth tensor shape: {ground_truth_tensor.shape}")
                
                # Check if ground truth is one-hot encoded or class indices
                if len(ground_truth_tensor.shape) > 1 and ground_truth_tensor.shape[1] > 1:
                    # Ground truth is one-hot encoded, convert to class indices
                    ground_truth_indices = torch.argmax(ground_truth_tensor, dim=-1).long()
                    print(f"States - Converting one-hot to indices, new shape: {ground_truth_indices.shape}")
                else:
                    # Ground truth is already class indices
                    ground_truth_indices = ground_truth_tensor.long()
                
                print(f"States - Ground truth sample: {ground_truth_indices[:5]}")
                print(f"States - Predicted classes sample: {torch.argmax(predicted_logits, dim=-1)[:5]}")
                
                # Cross-entropy loss for masked positions
                _ce = F.cross_entropy(predicted_logits, ground_truth_indices, reduction='mean')
                ce_loss += _ce
                
                # Calculate accuracy for masked positions
                predicted_classes = torch.argmax(predicted_logits, dim=-1)
                accuracy = (predicted_classes == ground_truth_indices).float().mean()
                log_dict[f"test/accuracy_{k}"] = accuracy.item()
                log_dict[f"test/num_masked_{k}"] = mask_positions.sum().item()
                
                print(f"States - Accuracy: {accuracy.item():.4f}")
                print(f"States - Cross-entropy loss: {_ce.item():.4f}")
            else:
                log_dict[f"test/accuracy_{k}"] = 0.0
                log_dict[f"test/num_masked_{k}"] = 0
                print(f"States - No masked positions found!")

        elif k == "actions":
            # For continuous actions: calculate MSE for MASKED positions only
            mask_positions = (masks[k] == 0) & (attention_masks == 1)  # Masked positions that are valid
            
            print(f"Actions - Total mask positions: {mask_positions.sum().item()}")
            print(f"Actions - Prediction shape: {v.shape}, Ground truth shape: {batch[k].shape}")
            
            for f in range(v.shape[2]):
                if mask_positions.sum() > 0:
                    predicted_actions = v[mask_positions][:, f].to(torch.float32)
                    ground_truth_actions = batch[k][mask_positions][:, f].to(torch.float32)
                    
                    _mse = F.mse_loss(predicted_actions, ground_truth_actions, reduction='mean')
                    log_dict[f"test/mse_{k}_{f}"] = _mse.item()
                    mse_loss += _mse.item()
                    
                    if f == 0:  # Only print for first action dimension
                        print(f"Actions[{f}] - MSE: {_mse.item():.4f}")
                        print(f"Actions[{f}] - Pred sample: {predicted_actions[:5]}")
                        print(f"Actions[{f}] - GT sample: {ground_truth_actions[:5]}")
                else:
                    log_dict[f"test/mse_{k}_{f}"] = 0.0
            
            if mask_positions.sum() > 0:
                log_dict[f"test/num_masked_{k}"] = mask_positions.sum().item()
            else:
                log_dict[f"test/num_masked_{k}"] = 0
                print(f"Actions - No masked positions found!")
    print("=== END DEBUG ACCURACY CALCULATION ===\n")

    log_dict["test/mse_sum_Actions"] = mse_loss
    log_dict["test/cross_entropy_sum_States"] = ce_loss
    log_dict["skipped"] = False  # Mark as not skipped for normal processing

    return log_dict


def main(hydra_cfg):
    return _main(hydra_cfg)


def _main(hydra_cfg):
    cfg: TestConfig = hydra.utils.instantiate(hydra_cfg.args)
    dp: DistributedParams = get_distributed_params()

    torch.cuda.set_device(dp.local_rank)
    distributed = dp.world_size > 1
    if distributed:
        logger.info(
            f"Initializing rank {dp.rank} (local rank {dp.local_rank}) in total world size {dp.world_size} (local world size {dp.local_world_size}) with master addr:port {dp.master_addr}:{dp.master_port}"
        )
        torch.distributed.init_process_group(
            backend="nccl", rank=dp.rank, world_size=dp.world_size
        )

    set_seed_everywhere(cfg.seed)
    pprint.pp(cfg)

    logger.info(f"Working directory: {os.getcwd()}")

    with open("test_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(hydra_cfg))

    # Load test dataset - optimized to load only validation data if specified
    val_dataset: DatasetProtocol
    print("hydra config", hydra_cfg)
    
    if cfg.load_test_only:
        print("Loading only validation dataset for testing (optimized mode)...")
        
        # Get dataset configuration
        dataset_cfg = hydra_cfg.datasets
        
        # Create a modified version that only loads validation data
        if hasattr(dataset_cfg, '_target_') and 'traj' in dataset_cfg._target_:
            # For trajectory datasets, use optimized test dataset loader
            from research.mtm.datasets.traj import get_test_dataset
            
            val_dataset = get_test_dataset(
                seq_steps=cfg.traj_length,
                env_name=dataset_cfg.env_name,
                seed=getattr(dataset_cfg, 'seed', 42),
                replay_buffer_dir=dataset_cfg.replay_buffer_dir,
                val_max_size=getattr(dataset_cfg, 'val_max_size', 1000000),
                num_workers=getattr(dataset_cfg, 'num_workers', 4),
                train_val_split=getattr(dataset_cfg, 'train_val_split', 0.8),
            )
            
        else:
            # For other dataset types, fall back to original method but only use val_dataset
            print("Using fallback dataset loading method...")
            train_dataset, val_dataset = hydra.utils.call(
                hydra_cfg.datasets, seq_steps=cfg.traj_length)
            del train_dataset  # Free up memory immediately
    else:
        print("Loading both training and validation datasets (full mode)...")
        train_dataset, val_dataset = hydra.utils.call(
            hydra_cfg.datasets, seq_steps=cfg.traj_length)
        # Keep train_dataset in memory for potential use
    
    logger.info(f"Test set size = {len(val_dataset)}")

    # Use validation dataset as our test set
    test_dataset = val_dataset

    # Initialize tokenizers
    if "tokenizers" in hydra_cfg:
        tokenizers: Dict[str, Tokenizer] = {
            k: hydra.utils.call(v, key=k, train_dataset=test_dataset)
            for k, v in hydra_cfg.tokenizers.items()
        }
    else:
        tokenizers: Dict[str, Tokenizer] = {
            k: ContinuousTokenizer.create(k, train_dataset)
            for k in train_dataset[0].keys()
        }
    tokenizer_manager = TokenizerManager(tokenizers).to(cfg.device)

    discrete_map: Dict[str, bool] = {}
    for k, v in tokenizers.items():
        discrete_map[k] = v.discrete
    logger.info(f"Tokenizers: {tokenizers}")

    # Setup test data loader
    if distributed:
        test_sampler = torch.utils.data.DistributedSampler(
            test_dataset, num_replicas=dp.world_size, rank=dp.rank, shuffle=False
        )
    else:
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    with stopwatch("data loader"):
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.n_workers,
            sampler=test_sampler,
        )

    # Get data shapes for model initialization
    test_batch = next(iter(test_loader))
    attention_masks = test_batch.pop("attention_masks", None)
    tokenized = tokenizer_manager.encode(test_batch, attention_masks=attention_masks)

    data_shapes = {}
    for k, v in tokenized.items():
        data_shapes[k] = v.shape[-2:]
    logger.info(f"Data shapes: {data_shapes}")

    # Create the model
    model_config = hydra.utils.instantiate(hydra_cfg.model_config)
    model = model_config.create(data_shapes, cfg.traj_length)
    model.to(cfg.device)

    # Load trained model weights
    if cfg.model_path:
        logger.info(f"Loading model from: {cfg.model_path}")
        checkpoint = torch.load(cfg.model_path, map_location=cfg.device, weights_only=False)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
    else:
        logger.warning("No model path provided! Using randomly initialized model.")

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dp.local_rank], output_device=dp.local_rank
        )

    model.eval()  # Set to evaluation mode

    # Setup WandB logging for test results
    wandb_cfg_log_dict = OmegaConf.to_container(hydra_cfg)
    wandb_cfg_log_dict["*discrete_map"] = discrete_map
    wandb_cfg_log_dict["*test_path"] = str(os.getcwd())
    wandb_cfg_log_dict["*mask_patterns"] = cfg.mask_patterns
    wandb_cfg_log_dict["*model_path"] = cfg.model_path
    wandb_cfg_log_dict["*test_name"] = cfg.test_name
    wandb_cfg_log_dict["*num_parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    wandb_cfg_log = WandBLoggerConfig(
        experiment_id=f"{cfg.test_name}-{dp.job_id}-{dp.rank}",
        project=hydra_cfg.wandb.project + "_test",
        entity=hydra_cfg.wandb.entity or None,
        resume=False,
        group=f"{cfg.test_name}-{dp.job_id}",
    )

    wandb_logger = WandBLogger(wandb_cfg_log, wandb_cfg_log_dict)

    # Create output directory with custom test name
    test_output_dir = os.path.join(cfg.output_dir, cfg.test_name)
    os.makedirs(test_output_dir, exist_ok=True)

    # Setup evaluation masks
    has_rew = "rewards" in test_batch
    has_ret = "returns" in test_batch
    has_img = "images" in test_batch
    
    # Function to find valid masks when sequences are empty
    def find_valid_mask_with_retry(test_mask_functions_map, cfg, attention_mask, pattern, max_attempts=10):
        """
        Try the same mask pattern multiple times with different random seeds until valid.
        Much simpler approach - just try and catch errors.
        """
        
        if attention_mask is None:
            # No attention mask, any mask should work
            mask_func = test_mask_functions_map[MaskType[pattern]]
            return mask_func(), pattern
            
        # Check if all sequences are empty
        
        seq_lengths = attention_mask.sum(dim=-1)
        if not (seq_lengths == 0).any():
            # Some sequences are valid, normal mask should work
            mask_func = test_mask_functions_map[MaskType[pattern]]
            return mask_func(), pattern
            
        print(f"Batch contains empty sequences. Trying {pattern} with different seeds...")
        
        # Try multiple times with different random states
        for attempt in range(max_attempts):
            try:
                # Create mask function - many mask functions use random generation
                mask_func = test_mask_functions_map[MaskType[pattern]]
                masks = mask_func()
                
                # Simple validation: check if mask has any 1s for sequences that have valid tokens
                valid = False
                for key, mask in masks.items():
                    if key not in data_shapes:
                        continue
                    
                    # For each sequence in batch, check if mask overlaps with valid tokens
                    for seq_idx in range(attention_mask.shape[0]):
                        seq_len = int(attention_mask[seq_idx].sum().item())
                        if seq_len > 0:  # Valid sequence
                            # Check if mask has any 1s in the valid part
                            if mask[:seq_len].sum() > 0:
                                valid = True
                                break
                    if valid:
                        break
                
                if valid:
                    if attempt > 0:
                        print(f"Found valid {pattern} mask after {attempt + 1} attempts")
                    return masks, pattern
                    
            except Exception as e:
                # If there's an error, try again
                continue
        
        # If all attempts failed, try a minimal conservative mask
        print(f"All {max_attempts} attempts failed for {pattern}. Using minimal mask.")
        minimal_masks = {}
        for key in data_shapes.keys():
            mask = torch.zeros(cfg.traj_length, device=cfg.device)
            # Only mask first valid timestep for each sequence
            mask[0] = 1  
            minimal_masks[key] = mask
        
        return minimal_masks, f"{pattern}_MINIMAL"

    def create_test_mask_functions(attention_mask=None):
        return {
            MaskType.RANDOM: lambda: create_random_masks(
                data_shapes, cfg.mask_ratios, cfg.traj_length, cfg.device
            ),
            MaskType.FULL_RANDOM: lambda: create_full_random_masks(
                data_shapes, cfg.mask_ratios, cfg.traj_length, cfg.device
            ),
            MaskType.AUTO_MASK: lambda: create_random_autoregressize_mask(
                data_shapes, cfg.mask_ratios, cfg.traj_length, cfg.device, cfg.mode_weights, cfg.mode_order,
            ),
            MaskType.RCBC: lambda: create_rcbc_mask(cfg.traj_length, cfg.device),
            MaskType.GOAL: lambda: maybe_add_rew_to_mask(
                cfg.traj_length,
                cfg.device,
                create_goal_reaching_masks,
                has_rew,
                has_img,
                has_ret,
                attention_mask=attention_mask,
            ),
            MaskType.GOAL_N: lambda: maybe_add_rew_to_mask(
                cfg.traj_length,
                cfg.device,
                create_goal_n_reaching_masks,
                has_rew,
                has_img,
                has_ret,
                attention_mask=attention_mask,
            ),
            MaskType.ID: lambda: maybe_add_rew_to_mask(
                cfg.traj_length,
                cfg.device,
                create_inverse_dynamics_mask,
                has_rew,
                has_img,
                has_ret,
                attention_mask=attention_mask,
            ),
            MaskType.FD: lambda: maybe_add_rew_to_mask(
                cfg.traj_length,
                cfg.device,
                create_forward_dynamics_mask,
                has_rew,
                has_img,
                has_ret,
                attention_mask=attention_mask,
            ),
            MaskType.BC: lambda: maybe_add_rew_to_mask(
                cfg.traj_length,
                cfg.device,
                create_bc_mask,
                has_rew,
                has_img,
                has_ret,
                attention_mask=attention_mask,
            ),
            MaskType.BC_RANDOM: lambda: maybe_add_rew_to_mask(
                cfg.traj_length,
                cfg.device,
                lambda l, d: create_random_bc_masks(l, d, data_shapes, p=0.5),
                has_rew,
                has_img,
                has_ret,
                attention_mask=attention_mask,
            ),
        }

    # Run testing
    logger.info("Starting model testing...")
    
    total_batches = len(test_loader)
    all_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = {k: v.to(cfg.device, non_blocking=True) for k, v in batch.items()}
            
            # Get attention mask for proper variable-length trajectory handling
            attention_mask = batch.get("attention_mask", None)
            
            # Check for empty sequences - use mask cycling instead of skipping
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=-1)
                if (seq_lengths == 0).any():
                    print(f"Warning: Batch {batch_idx} contains empty sequences (lengths: {seq_lengths})")
                    print("Will try to find valid masks...")
            
            # Create mask functions with current attention mask
            test_mask_functions_map = create_test_mask_functions(attention_mask)
            
            # Test with different mask patterns, using retry if needed
            for mask_pattern in cfg.mask_patterns:
                # Try to find a valid mask for this pattern
                try:
                    #import pdb; pdb.set_trace()
                    masks, actual_pattern = find_valid_mask_with_retry(
                        test_mask_functions_map, cfg, attention_mask, mask_pattern
                    )
                    
                    if actual_pattern != mask_pattern:
                        print(f"Original pattern '{mask_pattern}' resulted in empty sequences.")
                        print(f"Using valid fallback pattern: '{actual_pattern}'")
                    
                    if "images" in batch and "images" not in masks:
                        masks["images"] = masks["states"]

                    print(f"Testing with mask pattern: {actual_pattern}")
                    # Test the model on this batch
                    batch_results = test_model_on_batch(
                        model,
                        tokenizer_manager,
                        discrete_map,
                        batch.copy(),
                        masks,
                        batch_idx,
                        mask_pattern=actual_pattern,  # Use the actual pattern that worked
                        save_predictions=cfg.save_predictions,
                        output_dir=test_output_dir,
                    )
                    
                    # Add mask pattern info to results
                    batch_results["mask_pattern"] = actual_pattern
                    batch_results["original_pattern"] = mask_pattern
                    batch_results["batch_idx"] = batch_idx
                    all_results.append(batch_results)
                    
                    # Log progress
                    if batch_idx % 10 == 0:
                        logger.info(f"Processed batch {batch_idx}/{total_batches} with mask {actual_pattern}")
                        
                except Exception as e:
                    print(f"Error processing batch {batch_idx} with mask {mask_pattern}: {e}")
                    # Continue with next mask pattern
                    continue
                    wandb_logger.log(batch_results, step=batch_idx * len(cfg.mask_patterns) + len(all_results) - 1)

            # Run comprehensive evaluation on this batch
            if batch_idx == 0:  # Run detailed evaluation on first batch
                vis_batch = batch.copy()
                eval_masks = create_random_masks(
                    data_shapes, cfg.mask_ratios, cfg.traj_length, cfg.device
                )
                
                eval_results = evaluate(
                    model,
                    tokenizer_manager,
                    discrete_map,
                    batch.copy(),
                    vis_batch,
                    eval_masks,
                )
                
                # Log comprehensive evaluation results
                wandb_logger.log(eval_results, step=0)
                
                # Run specialized evaluations if applicable
                if cfg.traj_length >= 2 and hasattr(test_dataset, "env"):
                    fd_results = eval_fd(model, test_dataset.env, batch.copy(), tokenizer_manager)
                    id_results = eval_id(model, test_dataset.env, batch.copy(), tokenizer_manager)
                    full_id_results = eval_full_id(model, test_dataset.env, batch.copy(), tokenizer_manager)
                    
                    eval_results.update(fd_results)
                    eval_results.update(id_results)
                    eval_results.update(full_id_results)
                    
                    wandb_logger.log(eval_results, step=1)

    # Aggregate and save final results
    logger.info("Computing aggregate statistics...")
    
    # Filter out skipped batches
    valid_results = [result for result in all_results if not result.get("skipped", False)]
    skipped_count = len(all_results) - len(valid_results)
    
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} batches due to empty sequences")
    
    logger.info(f"Processing {len(valid_results)} valid batches for statistics")
    
    # Group results by mask pattern
    results_by_pattern = {}
    for result in valid_results:
        pattern = result["mask_pattern"]
        if pattern not in results_by_pattern:
            results_by_pattern[pattern] = []
        results_by_pattern[pattern].append(result)
    
    # Compute mean statistics for each pattern
    final_stats = {}
    for pattern, pattern_results in results_by_pattern.items():
        pattern_stats = {}
        # Get all metric keys
        metric_keys = set()
        for result in pattern_results:
            metric_keys.update(k for k in result.keys() if isinstance(result[k], (int, float)))
        
        # Compute means
        for key in metric_keys:
            values = [result[key] for result in pattern_results if key in result]
            pattern_stats[f"{pattern}_{key}_mean"] = np.mean(values)
            pattern_stats[f"{pattern}_{key}_std"] = np.std(values)
        
        final_stats.update(pattern_stats)
    
    # Add batch processing statistics
    final_stats["total_batches"] = len(all_results)
    final_stats["valid_batches"] = len(valid_results)
    final_stats["skipped_batches"] = skipped_count
    final_stats["skip_rate"] = skipped_count / len(all_results) if len(all_results) > 0 else 0.0
    
    # Save final statistics
    import json
    with open(f"{test_output_dir}/test_statistics.json", "w") as f:
        json.dump(final_stats, f, indent=2)
    
    # Log final statistics to wandb
    wandb_logger.log(final_stats, step=len(valid_results))
    
    logger.info(f"Testing completed! Results saved to {test_output_dir}")
    logger.info(f"Processed {len(valid_results)}/{len(all_results)} batches (skipped {skipped_count} empty batches)")
    logger.info("Final Statistics:")
    for key, value in final_stats.items():
        logger.info(f"  {key}: {value:.6f}")

    return final_stats


@hydra.main(config_path=".", config_name="test_config", version_base="1.1")
def configure_test(hydra_data: DictConfig) -> None:
    # Convert DictConfig to YAML and print it
    logger.info(
        "\nFull Hydra Configuration:\n%s", OmegaConf.to_yaml(hydra_data, resolve=True)
    )
    main(hydra_data)


if __name__ == "__main__":
    configure_test()
