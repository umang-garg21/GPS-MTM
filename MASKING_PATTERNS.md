# MTM Masking Patterns Reference

## Quick Reference Table

| Pattern | Description | Hardcoded? | Best For | Notes |
|---------|-------------|------------|----------|-------|
| RANDOM | Random token masking | ❌ | General evaluation | Safe for all trajectory lengths |
| GOAL | Goal-reaching masking | ⚠️ Yes (idx 54+) | Goal-conditioned tasks | Assumes ~55-step trajectories |
| GOAL_N | N-step goal reaching | ❌ | Partial trajectory completion | Adaptive to trajectory length |
| ID | Inverse dynamics | ⚠️ Yes (first 24) | Action prediction | Limited to first 24 timesteps |
| FD | Forward dynamics | ⚠️ Partial (idx 0-53) | State prediction | Works with trajectories up to ~55 steps |
| BC | Behavioral cloning | ❌ | Imitation learning | Adaptive masking |
| RCBC | Return-conditioned BC | ❌ | Reward-aware imitation | Includes return information |
| BC_RANDOM | Random BC masking | ❌ | Robust imitation | BC with additional noise |
| FULL_RANDOM | Per-feature random | ❌ | Feature-level analysis | Different masks per feature |
| AUTO_MASK | Autoregressive | ❌ | Sequential prediction | Respects mode ordering |

## Hardcoded Values to Watch Out For

### GOAL Pattern (`create_goal_reaching_masks`)
```python
# Hardcoded in masks.py lines ~130-135
state_mask[54:] = 1
action_mask[54:] = 1
```
**Impact**: Only works properly with trajectories of ~55+ timesteps

### ID Pattern (`create_inverse_dynamics_mask`)
```python
# Hardcoded in masks.py lines ~170-175
state_mask[0:24] = 1
action_mask[0:24] = 1
```
**Impact**: Only uses first 24 timesteps, ignores rest of trajectory

### FD Pattern (`create_forward_dynamics_mask`)
```python
# Hardcoded in masks.py lines ~185-190
index1 = np.random.randint(0, 53 - 1)
index2 = np.random.randint(index1 + 1, 55 - 1)
```
**Impact**: Assumes trajectory length ~55, may fail with longer/shorter trajectories

## Recommended Usage

### For GPS Trajectory Data (221 timesteps):
- **Safe patterns**: RANDOM, GOAL_N, BC, RCBC, BC_RANDOM, FULL_RANDOM, AUTO_MASK
- **Problematic patterns**: GOAL (wastes most trajectory), ID (only uses ~11% of data)

### For Shorter Trajectories (<50 timesteps):
- **Safe patterns**: RANDOM, GOAL_N, BC, RCBC, BC_RANDOM, FULL_RANDOM, AUTO_MASK
- **Avoid**: GOAL, ID, FD (all have hardcoded assumptions)

### For Research/Development:
Consider modifying hardcoded patterns or using adaptive alternatives:
```python
# Instead of hardcoded GOAL masking:
goal_start_idx = int(traj_length * 0.8)  # Last 20% of trajectory
state_mask[goal_start_idx:] = 1

# Instead of hardcoded ID masking:
id_end_idx = min(24, int(traj_length * 0.4))  # First 40% or 24, whichever is smaller
state_mask[0:id_end_idx] = 1
```

## Configuration Examples

### Conservative (works with any trajectory length):
```yaml
mask_patterns: ["RANDOM", "GOAL_N", "BC", "AUTO_MASK"]
```

### Comprehensive (verify trajectory length compatibility):
```yaml
mask_patterns: ["RANDOM", "GOAL", "ID", "FD", "BC", "FULL_RANDOM"]
```

### Research-focused:
```yaml
mask_patterns: ["FULL_RANDOM", "AUTO_MASK", "BC_RANDOM"]
```
