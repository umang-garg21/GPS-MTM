import os
from pathlib import Path

import numpy as np


def inspect_buffer_directory_and_file(buffer_dir):
    buffer_path = Path(buffer_dir)
    if not buffer_path.exists():
        print(f"Buffer directory does not exist: {buffer_dir}")
        return

    print(f"Inspecting buffer directory: {buffer_dir}")
    for root, dirs, files in os.walk(buffer_dir):
        level = root.replace(str(buffer_path), "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 4 * (level + 1)
        for file in files:
            if file.endswith(".npz"):
                file_path = Path(root) / file
                print(f"{sub_indent}{file} (Loading this file)")

                # Load and inspect content of the first .npz file
                data = np.load(file_path)
                print(f"Keys in {file}: {data.files}")
                for key in data.files:
                    print(f"  {key}: shape {data[key].shape}, dtype {data[key].dtype}")
                return  # Stop after the first file


# Replace with your actual buffer directory path
buffer_dir = "/data/home/chandrakanth/a100_code/feb24_code/GPS-MTM/research/exorl/datasets/point_mass_maze/proto/buffer"
buffer_dir = "/data/home/umang/Trajectory_project/anomaly_traj_data/haystac_anomaly_data1/saved_agent_episodes/obs4_act1"

inspect_buffer_directory_and_file(buffer_dir)
