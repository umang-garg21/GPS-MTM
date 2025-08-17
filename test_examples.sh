# ***************************
# Running MTM
# ***************************

CUDA_VISIBLE_DEVICES=0 python research/mtm/test.py +exp_traj=traj_cont_statediscrete_actcont ++datasets.env_name="gps_traj_masked_modelling" ++datasets.replay_buffer_dir="/data/home/umang/Trajectory_project/anomaly_traj_data/haystac_anomaly_data1/saved_agent_episodes_new/obs28_act11"

