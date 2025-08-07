# ***************************
# Running MTM
# ***************************
$workspace_dir="/data/home/chandrakanth/a100_code/feb24_code/GPS-MTM"

# MTM on D4RL Hoppper Medium-Replay
python research/mtm/train.py +exp_mtm=d4rl_cont wandb.project="11_mtm_d4rl_4_3" args.seed=0 dataset.env_name=hopper-medium-replay-v2 "args.mask_patterns=[AUTO_MASK]"


# ***************************
# Running Heteromodal MTM
# ***************************
# Heteromodal MTM 0.01 with actions 0.95 with states only
python research/mtm/train.py +exp_mtm=d4rl_halfcheetah_o3 "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="11_mtm_d4rl_4_5" args.seed=0 dataset.train_val_split=0.01

# Vanilla MTM 0.01 of the dataset
python research/mtm/train.py +exp_mtm=d4rl_cont dataset.train_val_split=0.01 dataset.env_name=halfcheetah-expert-v2 "args.mask_patterns=[AUTO_MASK]" wandb.project="11_mtm_d4rl_4_5" args.seed=0


# ***************************
# Running MTM on exorl - point mass maze reach top left
python research/mtm/train.py +exp_exorl=exorl_cont dataset.env_name="point_mass_maze_reach_top_left" dataset.replay_buffer_dir="/data/home/chandrakanth/a100_code/feb24_code/GPS-MTM/research/exorl/datasets/"

# Running MTM on GPS trajectory data.
#python research/mtm/train.py +exp_traj=traj_cont ++datasets.env_name="gps_traj_masked_modelling" ++datasets.replay_buffer_dir="/data/home/umang/Trajectory_project/anomaly_traj_data/haystac_anomaly_data1/saved_agents_1000"
python research/mtm/train.py +exp_traj=traj_cont ++datasets.env_name="gps_traj_masked_modelling" ++datasets.replay_buffer_dir="$workspace_dir/anomaly_traj_data/haystac_anomaly_data1/saved_agents_1000"


CUDA_VISIBLE_DEVICES=0 python research/mtm/train.py +exp_traj=traj_cont ++datasets.env_name="gps_traj_masked_modelling" ++datasets.replay_buffer_dir="/data/home/chandrakanth/a100_code/umang_mtm/data_chand/saved_agents_1000/"

CUDA_VISIBLE_DEVICES=0 python research/mtm/train.py +exp_traj=traj_cont ++datasets.env_name="gps_traj_masked_modelling" ++datasets.replay_buffer_dir="/data/home/chandrakanth/a100_code/umang_mtm/data_chand/saved_agents_1000/"

CUDA_VISIBLE_DEVICES=0 python research/mtm/train.py +exp_traj=traj_cont_statediscrete_actcont ++datasets.env_name="gps_traj_masked_modelling" ++datasets.replay_buffer_dir="/data/home/umang/Trajectory_project/anomaly_traj_data/saved_agents_1000_47class"

