# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum
from enum import Enum, unique
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

BASIC_MODE = True


@unique
class MaskType(Enum):
    RANDOM = enum.auto()
    ID = enum.auto()
    FD = enum.auto()
    GOAL = enum.auto()
    GOAL_N = enum.auto()
    FULL_RANDOM = enum.auto()
    BC = enum.auto()
    RCBC = enum.auto()
    BC_RANDOM = enum.auto()
    AUTO_MASK = enum.auto()


def create_random_mask(
    traj_length: int,
    mask_ratios: Union[Tuple[float, ...], float],
    device: str,
    rnd_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    # random_mask = np.concatenate(
    #     [
    #         np.ones(6),
    #         np.zeros(traj_length - 6),
    #     ]
    # )
    # return torch.tensor(random_mask, device=device)

    if isinstance(mask_ratios, Sequence):
        if rnd_state is None:
            mask_ratio = np.random.choice(mask_ratios)
        else:
            mask_ratio = rnd_state.choice(mask_ratios)
    else:
        mask_ratio = mask_ratios

    masked = int(traj_length * mask_ratio)
    # random_mask = np.concatenate(
    #     [
    #         np.ones(masked),
    #         np.zeros(traj_length - masked),
    #     ]
    # )

    # FIXED: mask_ratio should be the ratio of MASKED tokens (0s), not visible tokens (1s)
    # So if mask_ratio=0.5, we want 50% masked (0s) and 50% visible (1s)
    random_mask = np.random.choice(
        [1, 0], size=traj_length, p=[1 - mask_ratio, mask_ratio]
    )
    
    # DEBUG: Print mask creation info
    print(f"MASK CREATION DEBUG:")
    print(f"  Mask ratio requested: {mask_ratio} (should be fraction of MASKED tokens)")
    print(f"  Trajectory length: {traj_length}")
    print(f"  Created mask - 1s (visible): {np.sum(random_mask)}")
    print(f"  Created mask - 0s (masked): {np.sum(1 - random_mask)}")
    print(f"  Actual MASKED ratio: {np.sum(1 - random_mask) / traj_length:.3f}")
    print(f"  Actual VISIBLE ratio: {np.sum(random_mask) / traj_length:.3f}")
    print(f"  First 20 mask values: {random_mask[:20]}")
    print(f"  CONVENTION: 1=VISIBLE, 0=MASKED_TO_PREDICT")
    
    #import pdb; pdb.set_trace()
    if rnd_state is None:
        np.random.shuffle(random_mask)
    else:
        rnd_state.shuffle(random_mask)

    # same mask for now
    random_mask = torch.tensor(random_mask, device=device)
    return random_mask


def create_full_random_mask(
    data_shape: Tuple[int, int],
    traj_length: int,
    mask_ratios: Union[Tuple[float, ...], float],
    device: str,
    rnd_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    L = traj_length
    P, _ = data_shape

    if isinstance(mask_ratios, Sequence):
        if rnd_state is None:
            mask_ratio = np.random.choice(mask_ratios)
        else:
            mask_ratio = rnd_state.choice(mask_ratios)
    else:
        mask_ratio = mask_ratios

    masked = int(L * P * float(mask_ratio))
    random_mask = np.concatenate(
        [
            np.ones(masked),
            np.zeros(L * P - masked),
        ]
    )
    if rnd_state is None:
        np.random.shuffle(random_mask)
    else:
        rnd_state.shuffle(random_mask)

    random_mask = torch.tensor(random_mask, device=device)
    return random_mask.reshape(L, P)


def create_goal_reaching_masks(
    traj_length: int,
    device: str,
    rnd_state: Optional[np.random.RandomState] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """
    Create goal-reaching masks that adapt to variable trajectory lengths.
    
    Args:
        traj_length: Maximum trajectory length (278)
        device: Device to place tensors on
        rnd_state: Random state for reproducibility
        attention_mask: (batch_size, seq_len) tensor indicating valid positions (ignored here, handled in model)
    """
    # Create base masks - the attention_mask logic is handled in the model
    state_mask = np.zeros(traj_length)
    action_mask = np.zeros(traj_length)
    
    # Use last 20% of max trajectory length as base pattern

    # HARCODED.

    goal_start_idx = 55  
    state_mask[goal_start_idx:] = 1
    action_mask[goal_start_idx:] = 1
    
    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
    }


def create_goal_n_reaching_masks(
    traj_length: int,
    device: str,
    rnd_state: Optional[np.random.RandomState] = None,
) -> Dict[str, np.ndarray]:
    state_mask = np.zeros(traj_length)
    action_mask = np.zeros(traj_length)

    if traj_length > 1:
        if rnd_state is None:
            end_state = np.random.randint(1, traj_length)
        else:
            end_state = rnd_state.randint(1, traj_length)

        state_mask[:end_state] = 1
        action_mask[: (end_state - 1)] = 1

        if BASIC_MODE:
            state_mask[-1] = 1
        else:
            if rnd_state is None:
                end_state = np.random.randint(1, 4)
            else:
                end_state = np.random.randint(1, 4)
            state_mask[-end_state:] = 1

    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
    }


def create_inverse_dynamics_mask(
    traj_length: int,
    device: str,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """
    Create inverse dynamics masks that adapt to variable trajectory lengths.
    
    Args:
        traj_length: Maximum trajectory length (278)
        device: Device to place tensors on  
        attention_mask: (batch_size, seq_len) tensor indicating valid positions (ignored here, handled in model)
    """
    # Create base masks - the attention_mask logic is handled in the model
    state_mask = np.zeros(traj_length)
    action_mask = np.zeros(traj_length)

    # Use first 20% of max trajectory length as base pattern
    id_end_idx = max(1, int(traj_length * 0.2))  # First 20% of trajectory
    state_mask[0:id_end_idx] = 1
    action_mask[0:id_end_idx] = 1
    
    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
    }


def create_forward_dynamics_mask(
    traj_length: int,
    device: str,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """
    Create forward dynamics masks that adapt to variable trajectory lengths.
    
    Args:
        traj_length: Maximum trajectory length (278)
        device: Device to place tensors on
        attention_mask: (batch_size, seq_len) tensor indicating valid positions (ignored here, handled in model)
    """
    # Create base masks - the attention_mask logic is handled in the model
    state_mask = np.zeros(traj_length)
    action_mask = np.zeros(traj_length)

    # Use middle portion of max trajectory length as base pattern
    start_ratio = np.random.uniform(0.1, 0.6)  # Start between 10%-60%
    end_ratio = np.random.uniform(start_ratio + 0.1, 0.9)  # End between start+10% and 90%
    
    index1 = int(traj_length * start_ratio)
    index2 = int(traj_length * end_ratio)
    
    #harcoded
    index1 = np.random.randint(0, 55)
    index2 = np.random.randint(index1, 56)

    state_mask[index1:index2] = 1
    action_mask[index1:index2] = 1

    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
    }


def create_random_masks(
    data_shapes, mask_ratios, traj_length, device
) -> Dict[str, np.ndarray]:
    masks = {}

    # KEEP THE SAME MASK FOR THE RIGHT TESTING.
    random_mask = create_random_mask(traj_length, mask_ratios, device)
    for k in data_shapes.keys():
        # create a random mask, different mask for each modality
        masks[k] = random_mask
    return masks


def create_full_random_masks(
    data_shapes, mask_ratios, traj_length, device
) -> Dict[str, np.ndarray]:
    masks = {}
    # hardcode mask ratio. make it follow cosin funciton
    mask_ratios = np.linspace(0.15, 0.9, 30)
    mask_ratios = np.cos(mask_ratios * np.pi) / 2 + 0.5  # following mask git
    mask_ratios = mask_ratios.tolist()

    for k, v in data_shapes.items():
        # create a random mask, different mask for each modality
        random_mask = create_full_random_mask(v, traj_length, mask_ratios, device)
        masks[k] = random_mask
    return masks


def maybe_add_rew_to_mask(traj_length, device, mask_fn, add_rew, add_img, add_ret, attention_mask=None):
    """
    Create masks and optionally add reward/return/image masks.
    
    Args:
        traj_length: Maximum trajectory length
        device: Device to place tensors on
        mask_fn: Function to create the base masks
        add_rew: Whether to add reward masks
        add_img: Whether to add image masks
        add_ret: Whether to add return masks
        attention_mask: Optional attention mask for variable-length trajectories
    """
    # Pass attention_mask to the mask function if it supports it
    try:
        # Try calling with attention_mask parameter
        masks = mask_fn(traj_length, device, attention_mask=attention_mask)
    except TypeError:
        # Fall back to original call without attention_mask
        masks = mask_fn(traj_length, device)
    
    if add_rew and "rewards" not in masks:
        masks["rewards"] = masks["actions"].clone()
        if len(masks["rewards"].shape) == 2:
            masks["rewards"] = masks["rewards"][..., 0:1]
    if add_ret and "returns" not in masks:
        masks["returns"] = masks["actions"].clone()
        if len(masks["returns"].shape) == 2:
            masks["returns"] = masks["returns"][..., 0:1]
    if add_img:
        masks["images"] = masks["states"].clone()
    return masks


def create_adaptive_variable_length_goal_mask(
    traj_length: int,
    device: str,
    attention_masks: torch.Tensor,
    goal_ratio: float = 0.8,
) -> Dict[str, torch.Tensor]:
    """
    Create goal masks that adapt to each trajectory's actual length.
    
    Args:
        traj_length: Maximum trajectory length (221)
        device: Device to place tensors on
        attention_masks: (batch_size, seq_len) tensor with 1s for valid positions
        goal_ratio: What percentage from the end to use for goal (0.8 = last 20%)
    
    Returns:
        Dict with masks adapted to each trajectory's actual length
    """
    batch_size, seq_len = attention_masks.shape
    state_masks = torch.zeros_like(attention_masks, dtype=torch.float32)
    action_masks = torch.zeros_like(attention_masks, dtype=torch.float32)
    
    for b in range(batch_size):
        # Find actual trajectory length for this sample
        valid_positions = (attention_masks[b] == 1).nonzero(as_tuple=True)[0]
        if len(valid_positions) > 0:
            actual_length = valid_positions[-1].item() + 1  # +1 because indices are 0-based
            
            # Calculate goal start position based on actual length
            goal_start_idx = max(0, int(actual_length * goal_ratio))
            
            # Set masks only for valid positions
            state_masks[b, goal_start_idx:actual_length] = 1
            action_masks[b, goal_start_idx:actual_length] = 1
    
    return {
        "states": state_masks.to(device),
        "actions": action_masks.to(device),
    }


def create_adaptive_variable_length_id_mask(
    traj_length: int,
    device: str,
    attention_masks: torch.Tensor,
    id_ratio: float = 0.3,
) -> Dict[str, torch.Tensor]:
    """
    Create inverse dynamics masks that adapt to each trajectory's actual length.
    
    Args:
        traj_length: Maximum trajectory length (221)
        device: Device to place tensors on
        attention_masks: (batch_size, seq_len) tensor with 1s for valid positions
        id_ratio: What percentage from the start to use for ID (0.3 = first 30%)
    
    Returns:
        Dict with masks adapted to each trajectory's actual length
    """
    batch_size, seq_len = attention_masks.shape
    state_masks = torch.zeros_like(attention_masks, dtype=torch.float32)
    action_masks = torch.zeros_like(attention_masks, dtype=torch.float32)
    
    for b in range(batch_size):
        # Find actual trajectory length for this sample
        valid_positions = (attention_masks[b] == 1).nonzero(as_tuple=True)[0]
        if len(valid_positions) > 0:
            actual_length = valid_positions[-1].item() + 1  # +1 because indices are 0-based
            
            # Calculate ID end position based on actual length
            id_end_idx = max(1, int(actual_length * id_ratio))
            
            # Set masks only for valid positions
            state_masks[b, 0:id_end_idx] = 1
            action_masks[b, 0:id_end_idx] = 1
    
    return {
        "states": state_masks.to(device),
        "actions": action_masks.to(device),
    }


def create_adaptive_variable_length_fd_mask(
    traj_length: int,
    device: str,
    attention_masks: torch.Tensor,
    min_start_ratio: float = 0.1,
    max_start_ratio: float = 0.6,
    min_length_ratio: float = 0.1,
    max_end_ratio: float = 0.9,
) -> Dict[str, torch.Tensor]:
    """
    Create forward dynamics masks that adapt to each trajectory's actual length.
    
    Args:
        traj_length: Maximum trajectory length (221)
        device: Device to place tensors on
        attention_masks: (batch_size, seq_len) tensor with 1s for valid positions
        min_start_ratio: Minimum start position as ratio of trajectory (0.1 = 10%)
        max_start_ratio: Maximum start position as ratio of trajectory (0.6 = 60%)
        min_length_ratio: Minimum mask length as ratio of trajectory (0.1 = 10%)
        max_end_ratio: Maximum end position as ratio of trajectory (0.9 = 90%)
    
    Returns:
        Dict with masks adapted to each trajectory's actual length
    """
    batch_size, seq_len = attention_masks.shape
    state_masks = torch.zeros_like(attention_masks, dtype=torch.float32)
    action_masks = torch.zeros_like(attention_masks, dtype=torch.float32)
    
    for b in range(batch_size):
        # Find actual trajectory length for this sample
        valid_positions = (attention_masks[b] == 1).nonzero(as_tuple=True)[0]
        if len(valid_positions) > 0:
            actual_length = valid_positions[-1].item() + 1  # +1 because indices are 0-based
            
            # Calculate random start and end positions based on actual length
            start_ratio = np.random.uniform(min_start_ratio, max_start_ratio)
            min_end_ratio = min(start_ratio + min_length_ratio, max_end_ratio)
            end_ratio = np.random.uniform(min_end_ratio, max_end_ratio)
            
            start_idx = int(actual_length * start_ratio)
            end_idx = min(int(actual_length * end_ratio), actual_length)
            
            # Set masks only for valid positions
            if end_idx > start_idx:
                state_masks[b, start_idx:end_idx] = 1
                action_masks[b, start_idx:end_idx] = 1
    
    return {
        "states": state_masks.to(device),
        "actions": action_masks.to(device),
    }


def create_bc_mask(
    traj_length: int,
    device: str,
) -> Dict[str, np.ndarray]:
    state_mask = np.ones(traj_length)
    action_mask = np.ones(traj_length)
    index = np.random.randint(0, traj_length)
    action_mask[index:] = 0
    state_mask[index + 1 :] = 0
    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
    }


def create_rcbc_mask(
    traj_length: int,
    device: str,
) -> Dict[str, np.ndarray]:
    state_mask = np.ones(traj_length)
    action_mask = np.ones(traj_length)
    index = np.random.randint(0, traj_length)
    action_mask[index:] = 0
    state_mask[index + 1 :] = 0
    return_mask = np.ones(traj_length)
    return {
        "states": torch.from_numpy(state_mask).to(device),
        "returns": torch.from_numpy(return_mask).to(
            device
        ),  # returns copies state mask
        "actions": torch.from_numpy(action_mask).to(device),
    }


def create_random_autoregressize_mask(
    data_shapes, mask_ratios, traj_length, device, p_weights=(0.2, 0.1, 0.7), 
    mode_order=["states", "returns", "actions"]
    ) -> Dict[str, np.ndarray]:

    random_mode = np.random.choice(mode_order, p=p_weights)
    random_position = np.random.randint(0, traj_length)
    print("random mode, random position", random_mode, random_position)
    masks = {}

    for k, v in data_shapes.items():
        # create a random mask, different mask for each modality
        masks[k] = create_full_random_mask(v, traj_length, mask_ratios, device)

    end_plus_one = False
    for k in mode_order:
        if k == random_mode:
            end_plus_one = True

        # mask out future
        if k in masks:
            if end_plus_one:
                masks[k][random_position:, :] = 0
            else:
                masks[k][random_position + 1 :, :] = 0

    # print(random_mode, random_position)
    return masks


def create_random_bc_masks(
    traj_length, device, data_shapes, p=0.5
) -> Dict[str, np.ndarray]:
    state_mask = np.ones((traj_length, data_shapes["states"][0]))
    action_mask = np.ones((traj_length, data_shapes["actions"][0]))
    index = np.random.randint(0, traj_length)
    action_mask[index:] = 0
    state_mask[index + 1 :] = 0

    action_mask[:index] = action_mask[:index] * np.random.choice(
        a=[1.0, 0.0], size=action_mask[:index].shape, p=[1 - p, p]
    )
    state_mask[: index + 1] = state_mask[: index + 1] * np.random.choice(
        a=[1.0, 0.0], size=state_mask[: index + 1].shape, p=[1 - p, p]
    )

    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
    }


def main():
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [1], 2, "cpu"
    )
    print(m)
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [1], 2, "cpu"
    )
    print(m)
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [1], 2, "cpu"
    )
    print(m)
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [1], 2, "cpu"
    )
    print(m)
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [1], 2, "cpu"
    )
    print(m)
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [1], 2, "cpu"
    )
    print(m)
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [1], 4, "cpu"
    )
    print(m)
    print()
    print()
    print()
    print()
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [0.35], 4, "cpu"
    )
    print(m)
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [0.35], 4, "cpu"
    )
    print(m)
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [0.35], 4, "cpu"
    )
    print(m)
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [0.35], 4, "cpu"
    )
    print(m)


if __name__ == "__main__":
    main()