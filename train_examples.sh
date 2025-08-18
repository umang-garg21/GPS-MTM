######### Running MTM on GPS trajectory data.


# TRAIN ON HAYSTAC DATA
CUDA_VISIBLE_DEVICES=3 python research/mtm/train.py \
    +exp_traj=traj_cont_statediscrete_actcont \
    ++datasets.env_name="gps_traj_masked_modelling" \
    ++datasets.replay_buffer_dir="/data/home/umang/Trajectory_project/anomaly_traj_data/haystac_anomaly_data1/saved_agent_episodes_new/obs28_act11"


# TRAIN ON URBAN ANOMALIES DATA
CUDA_VISIBLE_DEVICES=3 python research/mtm/train.py \
    +exp_traj=traj_cont_statediscrete_actcont \
    ++datasets.env_name="gps_traj_masked_modelling" \
    ++datasets.replay_buffer_dir="/data/home/umang/Trajectory_project/anomaly_traj_data/urban_anomalies/centralized/atlanta_combined_outliers/saved_agent_episodes_new/obs4_act11" \
