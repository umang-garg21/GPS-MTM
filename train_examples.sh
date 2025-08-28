######### Running MTM on GPS trajectory data.


# TRAIN ON HAYSTAC DATA
CUDA_VISIBLE_DEVICES=0 python research/mtm/train.py \
    +exp_traj=traj_cont_statediscrete_actcont \
    ++datasets.env_name="gps_traj_masked_modelling" \
    ++datasets.replay_buffer_dir="/data/home/umang/Trajectory_project/anomaly_traj_data/LA/saved_agent_episodes_new/obs28_act11"\
    ++args.traj_length="221" \
    ++hydra.job.name="train_haystac" \
    ++datasets.train_val_split=0.7 \
    ++tokenizers.states.num_classes="28"\
    ++args.batch_size="32"


# TRAIN ON URBAN ANOMALIES ATLANTA DATA 
CUDA_VISIBLE_DEVICES=1 python research/mtm/train.py \
    +exp_traj=traj_cont_statediscrete_actcont \
    ++datasets.env_name="gps_traj_masked_modelling" \
    ++datasets.replay_buffer_dir="/data/home/umang/Trajectory_project/anomaly_traj_data/urban_anomalies/centralized/atlanta_hunger_outliers/saved_agent_episodes_new/obs4_act11" \
    ++args.traj_length="278"\
    ++hydra.job.name="train_UA_atlanta_hunger_outliers" \
    ++datasets.train_val_split=0.8 \
    ++tokenizers.states.num_classes="4"\
    ++args.batch_size="32"


# TRAIN ON URBAN ANOMALIES BERLIN DATA 
CUDA_VISIBLE_DEVICES=6 python research/mtm/train.py \
    +exp_traj=traj_cont_statediscrete_actcont \
    ++datasets.env_name="gps_traj_masked_modelling" \
    ++datasets.replay_buffer_dir="/data/home/umang/Trajectory_project/anomaly_traj_data/urban_anomalies/centralized/berlin_work_outliers/saved_agent_episodes_new/obs4_act11" \
    ++args.traj_length="266" \
    ++hydra.job.name="train_UA_berlin_work_outliers" \
    ++datasets.train_val_split=0.8 \
    ++tokenizers.states.num_classes="4"\
    ++args.batch_size="32"


# TRAIN ON GEOLIFE
CUDA_VISIBLE_DEVICES=2 python research/mtm/train.py \
    +exp_traj=traj_cont_statediscrete_actcont \
    ++datasets.env_name="gps_traj_masked_modelling" \
    ++datasets.replay_buffer_dir="/data/home/umang/Trajectory_project/anomaly_traj_data/Geolife/saved_agent_episodes_new/obs198_act11" \
    ++args.traj_length="2549" \
    ++hydra.job.name="train_geolife" \
    ++datasets.train_val_split=0.7 \
    ++tokenizers.states.num_classes="198"\
    ++args.batch_size="5"
