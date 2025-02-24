import pickle
import random
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from research.exorl.replay_buffer import episode_len, load_episode
from research.mtm.datasets.base import DatasetProtocol, DataStatistics
from research.mtm.tokenizers.base import TokenizerManager

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

    def eval_logs(self, model: Callable, tokenizer_manager: TokenizerManager) -> Dict[str, Any]:
        num_samples = 10
        eval_logs = {}
        device = next(model.parameters()).device

        open_loop_mse = {
            # "pre_d_rand_mse_sum": 0,
            # "post_d_rand_mse_sum": 0,
            # "pre_d_full_mse_sum": 0,
            # "post_d_full_mse_sum": 0,
            # "pre_d_feat_full_mse_sum": 0,
            # "post_d_feat_full_mse_sum": 0,
            # "pre_d_feat_mse_sum": 0,
            # "post_d_feat_mse_sum": 0,
        }

        def mse_routine(batch_torch: torch.tensor, masks_torch: dict, k: str, feat_c: int = None):
            encoded_batch = tokenizer_manager.encode(batch_torch, attention_masks=attention_masks)
            predicted_trajectories = model(encoded_batch, masks_torch, attention_masks=attention_masks)
            decoded_trajectories = tokenizer_manager.decode(predicted_trajectories)

            if feat_c is None:
                pre_d_mse = torch.mean((encoded_batch[k] - predicted_trajectories[k]) ** 2)
                post_d_mse = torch.mean((batch_torch[k] - decoded_trajectories[k]) ** 2)
            else:
                pre_d_mse = torch.mean((encoded_batch[k][feat_c] - predicted_trajectories[k][feat_c]) ** 2)
                post_d_mse = torch.mean((batch_torch[k][feat_c] - decoded_trajectories[k][feat_c]) ** 2)

            return pre_d_mse, post_d_mse
        
        for n in range(num_samples):
            batch = self.sample()
            batch_torch = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in batch.items()} # what does attention_mask here?
            attention_masks = batch_torch.pop("attention_mask", None)

            # Update the loss dict at the start of the loop
            if n == 0:
                for k, v in batch_torch.items():
                    open_loop_mse[f"pre_d_rand_{k}"] = 0
                    open_loop_mse[f"post_d_rand_{k}"] = 0
                    open_loop_mse[f"pre_d_full_{k}"] = 0
                    open_loop_mse[f"post_d_full_{k}"] = 0
                    # for feat_c in range(v.shape[-1]):
                    #     open_loop_mse[f"pre_d_{k}_{feat_c}"] = 0
                    #     open_loop_mse[f"post_d_{k}_{feat_c}"] = 0
                    #     open_loop_mse[f"pre_d_full_{k}_{feat_c}"] = 0
                    #     open_loop_mse[f"post_d_full_{k}_{feat_c}"] = 0

            # Using autoregressive masking
            random_mask_start_idx = random.randint(1, batch["actions"].shape[0]-2)
            random_mask = torch.zeros(batch["actions"].shape[0]).to(device)
            random_mask[random_mask_start_idx:] = 1
            empty_mask = torch.ones(batch["actions"].shape[0]).to(device)
            full_mask = torch.zeros(batch["actions"].shape[0]).to(device)
            full_mask[0] = 0

            # # Masking per feature
            # for k, v in batch_torch.items():
            #     other_k = (set(list(batch_torch.keys())) - set([k])).pop()
            #     feat_mask = torch.stack([empty_mask for _ in range(v.shape[-1])], dim=-1)

            #     # Iterating over features
            #     for feat_c in range(v.shape[-1]):
            #         # Partial mask
            #         feat_mask[:, feat_c] = random_mask
            #         masks_torch = {k: feat_mask, other_k: empty_mask}
            #         pre_d_mse, post_d_mse = mse_routine(batch_torch, masks_torch, k, feat_c)
            #         open_loop_mse[f"pre_d_{k}_{feat_c}"] += pre_d_mse.item()
            #         open_loop_mse["pre_d_feat_mse_sum"] += pre_d_mse.item()
            #         open_loop_mse[f"post_d_{k}_{feat_c}"] += post_d_mse.item()
            #         open_loop_mse["post_d_feat_mse_sum"] += post_d_mse.item()

            #         # Full mask
            #         feat_mask[:, feat_c] = full_mask
            #         masks_torch = {k: feat_mask, other_k: empty_mask}
            #         pre_d_mse, post_d_mse = mse_routine(batch_torch, masks_torch, k, feat_c)
            #         open_loop_mse[f"pre_d_full_{k}_{feat_c}"] += pre_d_mse.item()
            #         open_loop_mse["pre_d_feat_full_mse_sum"] += pre_d_mse.item()
            #         open_loop_mse[f"post_d_full_{k}_{feat_c}"] += post_d_mse.item()
            #         open_loop_mse["post_d_feat_full_mse_sum"] += post_d_mse.item()

            # Masking over entire state or action
            for k, v in batch_torch.items():
                # Partial mask
                other_k = (set(list(batch_torch.keys())) - set([k])).pop()
                mask = random_mask
                # mask = torch.stack([random_mask for _ in range(v.shape[-1])], dim=-1)

                # double check that it masks ALL features
                masks_torch = {k: mask.unsqueeze(1), other_k: empty_mask.unsqueeze(1)}
                pre_d_mse, post_d_mse = mse_routine(batch_torch, masks_torch, k)

                open_loop_mse[f"pre_d_rand_{k}"] += pre_d_mse.item()
                open_loop_mse[f"post_d_rand_{k}"] += post_d_mse.item()
                # open_loop_mse["pre_d_rand_mse_sum"] += pre_d_mse.item()
                # open_loop_mse["post_d_rand_mse_sum"] += post_d_mse.item()

                # Full mask
                # mask = torch.stack([full_mask for _ in range(v.shape[-1])], dim=-1)
                mask = full_mask
                masks_torch = {k: mask.unsqueeze(1), other_k: empty_mask.unsqueeze(1)}
                pre_d_mse, post_d_mse = mse_routine(batch_torch, masks_torch, k)

                open_loop_mse[f"pre_d_full_{k}"] += pre_d_mse.item()
                open_loop_mse[f"post_d_full_{k}"] += post_d_mse.item()
                # open_loop_mse["pre_d_full_mse_sum"] += pre_d_mse.item()
                # open_loop_mse["post_d_full_mse_sum"] += post_d_mse.item()

        # Update logs
        for k, v in open_loop_mse.items():
            eval_logs["real_batch/" + k] = v / num_samples

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
