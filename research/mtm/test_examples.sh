# ***************************
# Running MTM
# ***************************

## HAYSTAC DATA
CUDA_VISIBLE_DEVICES=3 python research/mtm/test.py \
    +exp_traj=traj_cont_statediscrete_actcont \
    ++datasets.env_name="gps_traj_masked_modelling" \
    ++hydra.job.name="test_haystac" \
    ++datasets.replay_buffer_dir="/data/home/umang/Trajectory_project/anomaly_traj_data/LA/saved_agent_episodes_new/obs28_act11" \
    ++args.mask_ratios="[0.5]" \
    ++args.mask_patterns="[\"RANDOM\", \"GOAL\", \"ID\", \"FD\"]" \
    ++args.model_path="/data/home/umang/Trajectory_project/GPS-MTM/outputs/train_haystac/2025-08-27_17-04-53/model_100000.pt" \
    ++args.test_name="random_masking_0.5_testing" \
    ++args.traj_length="221" \
    ++args.batch_size="1" \
    ++tokenizers.states.num_classes="28"

## URBAN ANOMALIES
CUDA_VISIBLE_DEVICES=7 python research/mtm/test.py \
    +exp_traj=traj_cont_statediscrete_actcont \
    ++datasets.env_name="gps_traj_masked_modelling" \
    ++datasets.replay_buffer_dir="/data/home/umang/Trajectory_project/anomaly_traj_data/urban_anomalies/centralized/atlanta_hunger_outliers/saved_agent_episodes_new/obs4_act11" \
    ++args.model_path="/data/home/umang/Trajectory_project/GPS-MTM/outputs/train_UA_atlanta_hunger_outliers/2025-08-19_10-21-09/model_140010.pt"\
    ++hydra.job.name="test_UA_atlanta_hunger_outliers" \
    ++args.mask_ratios="[0.5]" \
    ++args.mask_patterns="[\"RANDOM\", \"GOAL\", \"ID\", \"FD\"]" \
    ++args.test_name="random_masking_0.5_testing" \
    ++args.traj_length="278" \
    ++args.batch_size="1" \
    ++tokenizers.states.num_classes="4"

## GEOLIFE
CUDA_VISIBLE_DEVICES=0 python research/mtm/test.py \
    +exp_traj=traj_cont_statediscrete_actcont \
    ++datasets.env_name="gps_traj_masked_modelling" \
    ++datasets.replay_buffer_dir="/data/home/umang/Trajectory_project/anomaly_traj_data/Geolife/saved_agent_episodes_new/obs198_act11" \
    ++args.model_path="/data/home/umang/Trajectory_project/GPS-MTM/outputs/train_geolife/2025-08-27_17-01-42/model_55000.pt" \
    ++hydra.job.name="test_geolife" \
    ++args.mask_ratios="[0.5]" \
    ++args.mask_patterns="[\"RANDOM\", \"GOAL\", \"ID\", \"FD\"]" \
    ++args.test_name="random_masking_0.5_testing" \
    ++args.traj_length="2549" \
    ++args.batch_size="1" \
    ++tokenizers.states.num_classes="198"