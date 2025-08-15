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
   ```

3. **Check results**:
   - Test statistics saved to `test_outputs/test_statistics.json`
   - Individual batch predictions saved as pickle files
   - WandB logs available in the test project

## Configuration Details

### Required Parameters:
- `model_path`: Path to trained model checkpoint (e.g., "model_50000.pt")
- Dataset configuration (must match training setup)
- Model configuration (must match training setup)

### Optional Parameters:
- `save_predictions`: Whether to save individual batch predictions (default: true)
- `output_dir`: Directory for saving results (default: "test_outputs")
- `mask_patterns`: List of masking strategies to test (default: ["RANDOM"])
- `traj_length`: Trajectory length (should match training config)

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
  # ... other test parameters
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
