import pickle
import random
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from research.exorl.replay_buffer import episode_len, load_episode
from research.mtm.datasets.base import DatasetProtocol, DataStatistics

class OfflineReplayBuffer(Dataset, DatasetProtocol):
    def __init__(
        self,
        # env,
        replay_dir,
        max_size,
        num_workers,
        discount,
        domain,
        traj_length,
        mode=None,
        cfg=None,
    ):
        print("Initializing replay buffer")
        # self._env = env
        self._replay_dir = replay_dir
        self._domain = domain
        self._mode = mode
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._discount = discount
        self._loaded = False
        self._traj_length = traj_length
        self._cfg = cfg

    def trajectory_statistics(self) -> Dict[str, DataStatistics]:
        save_path = self._replay_dir / "statistics.pkl"
        print(f"Statistics save path: {save_path}")

        if save_path.exists():
            print("Loading precomputed statistics.")
            with open(save_path, "rb") as f:
                return pickle.load(f)

        eps_fns = sorted(self._replay_dir.rglob("*.npz"))
        if not eps_fns:
            raise FileNotFoundError(
                f"No episode files (*.npz) found in: {self._replay_dir}"
            )

        keys = ["states", "actions", "att_mask"]
        stats = {key: [] for key in keys}

        for eps_fn in eps_fns:
            episode = load_episode(eps_fn)
            if episode is None:
                print(f"Skipping invalid episode file: {eps_fn}")
                continue

            stats["states"].append(episode["obs"])
            stats["actions"].append(episode["act"])
            stats["att_mask"].append(episode["att_mask"])

        for key in keys:
            stats[key] = np.concatenate(stats[key], axis=0)

        # Calculate statistics where attention_mask is 1

        att_mask = stats["att_mask"]
        # only calculate stats for states, actions.
        keys=["states", "actions"]
        # Calculate separate mean and variance for each feature in the dataset

        data_stats = {
            key: DataStatistics(
                np.mean(stats[key][att_mask == 1], axis=0),
                np.std(stats[key][att_mask == 1], axis=0),
                np.min(stats[key][att_mask == 1], axis=0),
                np.max(stats[key][att_mask == 1], axis=0),
            )
            for key in keys
        }

        # data_stats = {
        #     key: DataStatistics(
        #         np.mean(stats[key], axis=0),
        #         np.std(stats[key], axis=0),
        #         np.min(stats[key], axis=0),
        #         np.max(stats[key], axis=0),
        #     )
        #     for key in keys
        # }

        with open(save_path, "wb") as f:
            pickle.dump(data_stats, f)

        return data_stats

    def _load(self):
        if self._loaded:
            return
        self._loaded = True

        eps_fns = sorted(self._replay_dir.rglob("*.npz"))
        for eps_fn in eps_fns:
            if self._size > self._max_size:
                break
            episode = load_episode(eps_fn)
            self._episode_fns.append(eps_fn)
            self._episodes[eps_fn] = episode
            self._size += episode_len(episode)

    def _load_from_episodes(self, episode_list):
        """Load specific episodes into this dataset buffer."""
        if self._loaded:
            return
        self._loaded = True
        
        current_size = 0
        for eps_fn, episode in episode_list:
            if current_size >= self._max_size:
                break
            self._episode_fns.append(eps_fn)
            self._episodes[eps_fn] = episode
            episode_size = episode_len(episode)
            current_size += episode_size
            self._size += episode_size
        
        print(f"Loaded {len(self._episode_fns)} episodes, total size: {self._size}")

    def _load_from_filenames(self, eps_fns_list):
        """Load episodes from specific filenames into this dataset buffer."""
        if self._loaded:
            return
        self._loaded = True
        
        current_size = 0
        loaded_count = 0
        
        for eps_fn in eps_fns_list:
            if current_size >= self._max_size:
                print(f"Reached max size limit {self._max_size}, stopping at {loaded_count} episodes")
                break
            
            try:
                episode = load_episode(eps_fn)
                self._episode_fns.append(eps_fn)
                self._episodes[eps_fn] = episode
                episode_size = episode_len(episode)
                current_size += episode_size
                self._size += episode_size
                loaded_count += 1
                
                if loaded_count % 1000 == 0:
                    print(f"Loaded {loaded_count} episodes, current size: {current_size}")
                    
            except Exception as e:
                print(f"Error loading episode {eps_fn}: {e}")
                continue
        
        print(f"Final: Loaded {loaded_count} episodes, total size: {self._size}")

    def __len__(self) -> int:
        return self._size

    def _sample_episode(self, episode_idx: Optional[int] = None):
        if episode_idx is None:
            eps_fn = random.choice(self._episode_fns)
        else:
            eps_fn = self._episode_fns[episode_idx % len(self._episode_fns)]
        return self._episodes[eps_fn]

    def sample(self, episode_idx: Optional[int] = None, p_idx: Optional[int] = None):
        episode = self._sample_episode(episode_idx)
        # if p_idx is None:
        #    idx = np.random.randint(0, episode_len(episode) - self._traj_length + 1) + 1
        # else:
        #   idx = p_idx

        # obs = episode["observation"][idx - 1 : idx - 1 + self._traj_length]
        # action = episode["action"][idx : idx + self._traj_length]
        
        obs = episode["obs"]
        action = episode["act"]
        attention_mask = episode["att_mask"]
        return {
            "states": obs.astype(np.float32),
            "actions": action.astype(np.float32),
            "attention_mask": attention_mask.astype(np.float32),
        }

    def _s(self) -> Dict[str, np.ndarray]:
        content = self.sample()
        return {
            "states": content["states"],
            "actions": content["actions"],
            "attention_mask": content["attention_mask"].astype(np.float32),
        }

    def eval_logs(self, model: Callable, tokenizer_manager) -> Dict[str, Any]:
        num_samples = 10
        eval_logs = {}

        for _ in range(num_samples):
            sample = self.sample()
            batch = {
                "states": torch.tensor(sample["states"]).unsqueeze(0),
                "actions": torch.tensor(sample["actions"]).unsqueeze(0),
                "attention_mask": torch.tensor(sample["attention_mask"]).unsqueeze(0),
            }
            
            # Get trajectory length from the sample
            traj_length = batch["attention_mask"].shape[1]
            
            # Create no-masking masks (all zeros = no masking for evaluation)
            masks = {
                "states": torch.zeros(traj_length, dtype=torch.bool, device='cuda'),
                "actions": torch.zeros(traj_length, dtype=torch.bool, device='cuda')
            }
            
            # Encode the batch using tokenizer manager
            encoded_batch = {}
            for key in batch:
                if key != "attention_mask":
                    tokenizer = tokenizer_manager.tokenizers[key]
                    encoded_batch[key] = tokenizer.encode(batch[key])
            
            # Move tensors to GPU
            for key in encoded_batch:
                encoded_batch[key] = encoded_batch[key].to('cuda')
            batch["attention_mask"] = batch["attention_mask"].to('cuda')
            
            # Call model with proper arguments
            predicted_trajectories = model(encoded_batch, masks, attention_masks=batch["attention_mask"])
            
            # Simple MSE calculation on encoded space
            for key in encoded_batch:
                mse = torch.mean((predicted_trajectories[key] - encoded_batch[key]) ** 2)
                eval_logs[f"mse_{key}"] = mse.item()

        return eval_logs

    def __iter__(self):
        while True:
            yield self._s()

    def __getitem__(self, idx: int):
        return self._s()

def get_datasets(
    seq_steps: int,
    env_name: str,
    seed: int,
    replay_buffer_dir: str,
    train_max_size: int,
    val_max_size: int,
    num_workers: int,
    train_val_split: float = 0.8,
):
    # env = dmc.make(env_name, seed=seed)
    domain = env_name.split("_", 1)[0]
    print("domain: ", domain)

    replay_train_dir = Path(replay_buffer_dir)
    print("replay_train_dir: ", replay_train_dir)
    
    # Get all episode filenames and split them (no loading yet)
    eps_fns = sorted(replay_train_dir.rglob("*.npz"))
    
    print(f"Found {len(eps_fns)} episodes for train/val split...")
    
    # Handle case when no episodes are found
    if len(eps_fns) == 0:
        raise ValueError(f"No episodes found in directory: {replay_train_dir}")
    
    # Split episode filenames based on train_val_split ratio
    split_idx = int(len(eps_fns) * train_val_split)
    train_eps_fns = eps_fns[:split_idx]
    val_eps_fns = eps_fns[split_idx:]
    
    print(f"Train episodes: {len(train_eps_fns)}")
    print(f"Val episodes: {len(val_eps_fns)}")
    print(f"Episode-level split: {len(train_eps_fns)/(len(train_eps_fns)+len(val_eps_fns)):.3f}/{len(val_eps_fns)/(len(train_eps_fns)+len(val_eps_fns)):.3f}")
    
    train_dataset = OfflineReplayBuffer(
        # env,
        replay_train_dir,
        train_max_size,
        num_workers,
        discount=0.99,
        domain=domain,
        traj_length=seq_steps,
    )
    val_dataset = OfflineReplayBuffer(
        # env,
        replay_train_dir,
        val_max_size,
        num_workers,
        discount=0.99,
        domain=domain,
        traj_length=seq_steps,
    )

    # Load only the split episode filenames into respective datasets
    train_dataset._load_from_filenames(train_eps_fns)
    val_dataset._load_from_filenames(val_eps_fns)

    return train_dataset, val_dataset


def get_test_dataset(
    seq_steps: int,
    env_name: str,
    seed: int,
    replay_buffer_dir: str,
    val_max_size: int,
    num_workers: int,
    train_val_split: float = 0.8,
):
    """
    Optimized function to load ONLY the validation/test dataset.
    This avoids loading training data during testing, saving memory and time.
    """
    domain = env_name.split("_", 1)[0]
    print("domain: ", domain)

    replay_train_dir = Path(replay_buffer_dir)
    print("replay_train_dir: ", replay_train_dir)
    
    # Get all episode filenames and split them (no loading yet)
    eps_fns = sorted(replay_train_dir.rglob("*.npz"))
    
    print(f"Found {len(eps_fns)} total episodes")
    
    # Handle case when no episodes are found
    if len(eps_fns) == 0:
        raise ValueError(f"No episodes found in directory: {replay_train_dir}")
    
    # Split episode filenames based on train_val_split ratio
    split_idx = int(len(eps_fns) * train_val_split)
    val_eps_fns = eps_fns[split_idx:]  # Only get validation episodes
    
    print(f"Loading only {len(val_eps_fns)} validation episodes for testing")
    print(f"Skipping {len(eps_fns[:split_idx])} training episodes to save memory")
    
    val_dataset = OfflineReplayBuffer(
        replay_train_dir,
        val_max_size,
        num_workers,
        discount=0.99,
        domain=domain,
        traj_length=seq_steps,
    )

    # Load only validation episode filenames
    val_dataset._load_from_filenames(val_eps_fns)

    return val_dataset
