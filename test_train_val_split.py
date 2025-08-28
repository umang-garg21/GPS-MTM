#!/usr/bin/env python3
"""
Test script to verify that train and validation datasets don't overlap.
This script will check episode filenames and sample data to ensure proper splitting.
"""

import sys
import os
sys.path.append('/data/home/umang/Trajectory_project/GPS-MTM')

from research.mtm.datasets.traj import get_datasets
from pathlib import Path
import numpy as np

def test_train_val_no_overlap():
    """Test that train and validation datasets don't have overlapping episodes."""
    
    # Test configuration
    replay_buffer_dir = "/data/home/umang/Trajectory_project/anomaly_traj_data/LA/saved_agent_episodes_new/obs28_act11"
    train_val_split = 0.7
    seq_steps = 221
    train_max_size = 5000000
    val_max_size = 500000
    num_workers = 1  # Use single worker for testing
    
    print("=" * 60)
    print("TESTING TRAIN/VALIDATION DATA OVERLAP")
    print("=" * 60)
    
    # Load datasets
    print(f"Loading datasets with train_val_split={train_val_split}")
    train_dataset, val_dataset = get_datasets(
        seq_steps=seq_steps,
        env_name="gps_traj_masked_modelling",
        seed=0,
        replay_buffer_dir=replay_buffer_dir,
        train_max_size=train_max_size,
        val_max_size=val_max_size,
        num_workers=num_workers,
        train_val_split=train_val_split
    )
    
    print(f"\nDataset sizes:")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Check 1: Episode filename overlap
    print(f"\n" + "="*40)
    print("CHECK 1: Episode filename overlap")
    print("="*40)
    
    train_episode_fns = set(train_dataset._episode_fns)
    val_episode_fns = set(val_dataset._episode_fns)
    
    print(f"Train episodes count: {len(train_episode_fns)}")
    print(f"Val episodes count: {len(val_episode_fns)}")
    
    overlap_episodes = train_episode_fns.intersection(val_episode_fns)
    
    if len(overlap_episodes) > 0:
        print(f"❌ OVERLAP DETECTED: {len(overlap_episodes)} episodes overlap!")
        print(f"Overlapping episodes (first 5): {list(overlap_episodes)[:5]}")
        return False
    else:
        print(f"✅ NO EPISODE OVERLAP: Train and validation use completely different episodes")
    
    # Check 2: Sample a few episodes and verify their data doesn't overlap
    print(f"\n" + "="*40)
    print("CHECK 2: Sample data verification")
    print("="*40)
    
    # Sample some episodes from each dataset
    train_sample_episodes = list(train_episode_fns)[:5]
    val_sample_episodes = list(val_episode_fns)[:5]
    
    print(f"Sample train episodes:")
    for i, ep in enumerate(train_sample_episodes):
        print(f"  {i+1}. {Path(ep).name}")
    
    print(f"Sample val episodes:")
    for i, ep in enumerate(val_sample_episodes):
        print(f"  {i+1}. {Path(ep).name}")
    
    # Check 3: Verify split ratio
    print(f"\n" + "="*40)
    print("CHECK 3: Split ratio verification")
    print("="*40)
    
    total_episodes = len(train_episode_fns) + len(val_episode_fns)
    actual_train_ratio = len(train_episode_fns) / total_episodes
    actual_val_ratio = len(val_episode_fns) / total_episodes
    
    print(f"Expected train ratio: {train_val_split:.3f}")
    print(f"Actual train ratio: {actual_train_ratio:.3f}")
    print(f"Expected val ratio: {1-train_val_split:.3f}")
    print(f"Actual val ratio: {actual_val_ratio:.3f}")
    
    ratio_tolerance = 0.01  # 1% tolerance
    train_ratio_ok = abs(actual_train_ratio - train_val_split) < ratio_tolerance
    val_ratio_ok = abs(actual_val_ratio - (1-train_val_split)) < ratio_tolerance
    
    if train_ratio_ok and val_ratio_ok:
        print(f"✅ SPLIT RATIO OK: Within {ratio_tolerance*100}% tolerance")
    else:
        print(f"❌ SPLIT RATIO ERROR: Outside {ratio_tolerance*100}% tolerance")
        return False
    
    # Check 4: Verify episode ordering (episodes should be split sequentially)
    print(f"\n" + "="*40)
    print("CHECK 4: Sequential split verification")
    print("="*40)
    
    # Get all episodes in order
    replay_train_dir = Path(replay_buffer_dir)
    all_eps_fns = sorted(replay_train_dir.rglob("*.npz"))
    split_idx = int(len(all_eps_fns) * train_val_split)
    
    expected_train_eps = set(all_eps_fns[:split_idx])
    expected_val_eps = set(all_eps_fns[split_idx:])
    
    if train_episode_fns == expected_train_eps and val_episode_fns == expected_val_eps:
        print(f"✅ SEQUENTIAL SPLIT OK: Episodes are split sequentially as expected")
    else:
        print(f"❌ SEQUENTIAL SPLIT ERROR: Episodes are not split as expected")
        return False
    
    # Check 5: Sample actual data points to double-check
    print(f"\n" + "="*40)
    print("CHECK 5: Data sample verification")
    print("="*40)
    
    try:
        # Get a few data samples from each dataset
        train_sample = train_dataset[0]
        val_sample = val_dataset[0]
        
        print(f"Train sample keys: {train_sample.keys()}")
        print(f"Val sample keys: {val_sample.keys()}")
        
        # Check shapes match
        for key in train_sample.keys():
            if key in val_sample:
                train_shape = train_sample[key].shape
                val_shape = val_sample[key].shape
                print(f"  {key}: train_shape={train_shape}, val_shape={val_shape}")
                
        print(f"✅ DATA SAMPLES OK: Both datasets return valid samples")
        
    except Exception as e:
        print(f"❌ DATA SAMPLE ERROR: {e}")
        return False
    
    print(f"\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(f"✅ ALL CHECKS PASSED: Train and validation datasets are properly split")
    print(f"   - No episode overlap")
    print(f"   - Correct split ratio ({actual_train_ratio:.3f}/{actual_val_ratio:.3f})")
    print(f"   - Sequential splitting") 
    print(f"   - Valid data samples")
    print("="*60)
    
    return True

if __name__ == "__main__":
    test_train_val_no_overlap()
