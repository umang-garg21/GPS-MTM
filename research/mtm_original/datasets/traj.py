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

    def eval_logs(self, model: Callable) -> Dict[str, Any]:
        num_samples = 10
        eval_logs = {}

        for _ in range(num_samples):
            batch = {
                "states": torch.tensor(self.sample()["states"]).unsqueeze(0),
                "actions": torch.tensor(self.sample()["actions"]).unsqueeze(0),
            }
            predicted_trajectories = model(batch)
            mse = torch.mean((predicted_trajectories["states"] - batch["states"]) ** 2)
            eval_logs["mse"] = mse.item()

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
):
    # env = dmc.make(env_name, seed=seed)
    domain = env_name.split("_", 1)[0]
    print("domain: ", domain)

    replay_train_dir = Path(replay_buffer_dir)
    print("replay_train_dir: ", replay_train_dir)
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

    train_dataset._load()
    val_dataset._load()

    return train_dataset, val_dataset
