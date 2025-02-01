from pathlib import Path

import numpy as np


def preprocess_gps_data(trajectory_data, output_dir):
    """
    Preprocess GPS data and save as .npz files for MTM training.

    trajectory_data: List[Dict] - Each dict contains a trajectory with keys:
        - 'timestamps': List of timestamps
        - 'locations': List of (latitude, longitude)
        - 'pois': List of POI categories at stop points
    output_dir: str - Directory to save preprocessed .npz files
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i, traj in enumerate(trajectory_data):
        timestamps = np.array(traj["timestamps"])
        locations = np.array(traj["locations"])
        pois = np.array(traj["pois"])  # Encode categories as integers or one-hot
        stop_flags = (timestamps[1:] - timestamps[:-1]) > 10  # Example threshold

        # Calculate deltas for actions
        deltas = locations[1:] - locations[:-1]
        durations = timestamps[1:] - timestamps[:-1]

        observation = np.hstack(
            [
                locations[:-1],  # State: current locations
                timestamps[:-1, None],  # State: timestamps
                stop_flags[:-1, None],  # State: stop point flags
            ]
        )

        action = np.hstack(
            [
                deltas,  # Action: movement deltas
                durations[:, None],  # Action: duration to next state
            ]
        )

        # Rewards and discount (placeholders or computed values)
        reward = np.zeros(len(observation))
        discount = np.ones(len(observation))

        # Save to .npz
        np.savez(
            Path(output_dir) / f"episode_{i:06d}.npz",
            observation=observation,
            action=action,
            reward=reward,
            discount=discount,
        )
