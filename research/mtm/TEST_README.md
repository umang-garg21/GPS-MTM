# MTM Model Testing Script

This `test.py` script is designed to evaluate a trained MTM (Masked Trajectory Model) on test datasets, in contrast to the `train.py` script which is used for training.

## Key Differences from Training Script

### Removed Training Components:
- **Optimizer and Scheduler**: No gradient updates or learning rate scheduling
- **Training Loop**: Replaced with single-pass evaluation over test data
- **Checkpoint Saving**: Only loads existing checkpoints, doesn't save new ones
- **Training Loss Calculation**: Focused on evaluation metrics only

### Added Testing Features:
- **Model Loading**: Loads pre-trained model weights from checkpoint
- **Comprehensive Evaluation**: Tests multiple mask patterns on each batch
- **Results Aggregation**: Computes mean and std statistics across batches
- **Output Saving**: Saves predictions, ground truth, and attention masks to disk
- **Test-specific Logging**: Uses separate WandB project for test results

## Usage

1. **Configure the test settings** in `test_config.yaml`:
   ```yaml
   args:
     model_path: "/path/to/your/trained_model.pt"  # Required!
     batch_size: 32
     save_predictions: true
     output_dir: "test_outputs"
     test_name: "my_custom_test"  # Custom name for this test run
     mask_patterns: ["RANDOM", "GOAL", "ID", "FD"]
     traj_length: 221  # Should match your training configuration
   ```

2. **Run the test script**:
   ```bash
   # Use the test configuration (default)
   python test.py
   
   # Or explicitly specify the test config
   python test.py --config-name=test_config
   
   # Override specific parameters
   python test.py args.model_path="/path/to/model.pt" args.batch_size=16
   
   # Use a custom test name for better organization
   python test.py args.test_name="gps_trajectory_evaluation"
   ```

3. **Check results**:
   - Test statistics saved to `test_outputs/{test_name}/test_statistics.json`
   - Individual batch predictions saved as pickle files in the test name subdirectory
   - WandB logs available in the test project with custom experiment names

## Configuration Details

### Required Parameters:
- `model_path`: Path to trained model checkpoint (e.g., "model_50000.pt")
- Dataset configuration (must match training setup)
- Model configuration (must match training setup)

### Optional Parameters:
- `save_predictions`: Whether to save individual batch predictions (default: true)
- `output_dir`: Directory for saving results (default: "test_outputs")
- `test_name`: Custom name for the test run - affects output directory and WandB logging (default: "default_test")
- `mask_patterns`: List of masking strategies to test (default: ["RANDOM"])
- `traj_length`: Trajectory length (should match training config)

## Masking Patterns

The test script supports various masking patterns that evaluate different aspects of the model's capabilities:

### Available Mask Patterns:

1. **RANDOM**: Random masking across the trajectory
   - Randomly masks tokens based on the specified mask ratio
   - Tests general trajectory completion ability

2. **GOAL**: Goal-reaching masking ⚠️ **HARDCODED**
   - Shows initial and final states, masks intermediate trajectory
   - **Currently hardcoded to mask everything before index 54**
   - Evaluates goal-conditioned trajectory generation

3. **GOAL_N**: N-step goal reaching
   - Provides partial trajectory from start and goal state
   - Tests ability to connect partial trajectories to goals

4. **ID**: Inverse Dynamics masking ⚠️ **HARDCODED** 
   - Given states, predict actions
   - **Currently hardcoded to show only first 24 timesteps**
   - Tests action prediction from state sequences

5. **FD**: Forward Dynamics masking
   - Given states and actions, predict future states
   - Tests state prediction from action sequences
   - Uses random intervals between indices 0-53

6. **BC**: Behavioral Cloning masking
   - Provides state-action pairs up to a random point
   - Tests imitation learning capabilities

7. **RCBC**: Return-Conditioned Behavioral Cloning
   - Similar to BC but includes return information
   - Tests reward-conditioned policy learning

8. **BC_RANDOM**: Random Behavioral Cloning
   - BC with additional random masking of provided tokens
   - Tests robustness to partial observations

9. **FULL_RANDOM**: Full random masking per feature
   - Different random masks for each feature dimension
   - Uses cosine-scheduled mask ratios

10. **AUTO_MASK**: Autoregressive masking
    - Masks future tokens in autoregressive manner
    - Respects mode ordering (states, returns, actions)

### ⚠️ Important Hardcoding Issues:

Several masking patterns contain hardcoded values that may not generalize:

- **GOAL masking**: Hardcoded to index 54-55 (assumes ~55 timestep trajectories)
- **ID masking**: Hardcoded to first 24 timesteps
- **FD masking**: Hardcoded to work with trajectories up to index 53-55

These hardcoded values were designed for specific trajectory lengths and may need adjustment for different datasets or trajectory lengths.

## Configuration Structure

The `test_config.yaml` inherits most settings from the original training config but uses `TestConfig` instead of `RunConfig`:

```yaml
# Key sections that must match training:
datasets: d4rl  # Same as training
model_config:   # Identical to training config
  _target_: research.mtm.models.mtm_model.MTMConfig
  # ... same parameters

# Test-specific args:
args:
  _target_: research.mtm.test.TestConfig  # Changed from train.RunConfig
  model_path: "/path/to/checkpoint.pt"    # Added for testing
  mask_patterns: ["RANDOM", "GOAL", "ID", "FD"]  # Choose appropriate patterns
  # ... other test parameters
```

## Masking Pattern Usage Recommendations

### For Different Trajectory Lengths:
- **Standard (221 timesteps)**: All patterns should work, but verify hardcoded indices
- **Shorter trajectories (<50)**: Avoid GOAL, ID patterns due to hardcoded indices
- **Longer trajectories (>300)**: GOAL and ID patterns may not utilize full trajectory

### Pattern Selection Guidelines:
```yaml
# Basic evaluation
mask_patterns: ["RANDOM"]

# Comprehensive evaluation 
mask_patterns: ["RANDOM", "GOAL", "ID", "FD", "BC"]

# Research/analysis focused
mask_patterns: ["FULL_RANDOM", "AUTO_MASK", "GOAL_N"]
```

### Troubleshooting Mask-Related Issues:
1. **"Index out of bounds" errors**: Check trajectory length vs hardcoded indices
2. **Poor performance on GOAL/ID**: May indicate hardcoded values don't match your data
3. **Inconsistent results**: Some patterns have random components - run multiple times

## Custom Test Naming

The `test_name` parameter allows you to organize your test runs:
- **WandB Integration**: Creates experiment IDs like `{test_name}-{job_id}-{rank}`
- **Output Organization**: Results saved to `{output_dir}/{test_name}/`
- **Easy Tracking**: Group related test runs with meaningful names

Example usage:
```bash
# Compare different mask patterns
python test.py args.test_name="random_mask_eval" args.mask_patterns=["RANDOM"]
python test.py args.test_name="goal_mask_eval" args.mask_patterns=["GOAL"]

# Test different trajectory lengths
python test.py args.test_name="short_traj" args.traj_length=50
python test.py args.test_name="long_traj" args.traj_length=200
```

## Output Files

The script generates:
- `test_statistics.json`: Aggregate statistics across all test batches
- `predictions_batch_X.pkl`: Model predictions for each batch
- `ground_truth_batch_X.pkl`: Ground truth data for each batch
- `attention_masks_batch_X.pkl`: Attention masks (if applicable)
- `masks_batch_X.pkl`: Applied masks for each test

## Evaluation Metrics

The script computes:
- **Loss metrics**: Test loss, masked losses per modality
- **MSE metrics**: Mean squared error for each output dimension
- **Specialized evaluations**: Forward dynamics, inverse dynamics, goal reaching
- **Aggregate statistics**: Mean and standard deviation across all test batches

## Troubleshooting

### Common Issues:

1. **"RunConfig object has no attribute 'model_path'"**
   - Make sure you're using `--config-name=test_config` or the default is set correctly
   - Verify that `test_config.yaml` has `_target_: research.mtm.test.TestConfig`

2. **Model loading errors**
   - Check that the model_path exists and is accessible
   - Ensure the model architecture in test_config matches the training config

3. **Dataset configuration errors**
   - Make sure the dataset config in test_config.yaml matches your training setup
   - Verify that the same dataset files are accessible

## Notes

- Ensure your test configuration matches the training configuration for datasets, tokenizers, and model architecture
- The script automatically sets the model to evaluation mode (no dropout, etc.)
- Results are logged to a separate WandB project to avoid mixing with training logs
- The script supports distributed testing if multiple GPUs are available
