# ***************************
# Running MTM
# ***************************
CUDA_VISIBLE_DEVICES=3 python research/mtm/test.py \
    +exp_traj=traj_cont_statediscrete_actcont \
    ++datasets.env_name="gps_traj_masked_modelling" \
    ++datasets.replay_buffer_dir="/data/home/umang/Trajectory_project/anomaly_traj_data/haystac_anomaly_data1/saved_agent_episodes_new/obs28_act11" \
    ++args.mask_ratios="[0.8]" \
    ++args.mask_patterns="[\"RANDOM\", \"GOAL\", \"ID\", \"FD\"]" \
    ++args.model_path="/data/home/umang/Trajectory_project/GPS-MTM/outputs/mtm_mae/2025-08-17_00-36-20/model_101000.pt" \
    ++args.test_name="random_masking_0.8_testing"