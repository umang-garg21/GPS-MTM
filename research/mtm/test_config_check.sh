#!/bin/bash

# Simple test script to check if the test configuration works
echo "Testing MTM test configuration..."

cd /data/home/umang/Trajectory_project/GPS-MTM/research/mtm

# Check if the test_config.yaml is correctly formatted
echo "Checking test_config.yaml..."
python -c "
import yaml
from omegaconf import OmegaConf

try:
    cfg = OmegaConf.load('test_config.yaml')
    print('✓ test_config.yaml loads successfully')
    print(f'✓ Args target: {cfg.args._target_}')
    print(f'✓ Model path: {cfg.args.model_path}')
    print('✓ Configuration looks valid')
except Exception as e:
    print(f'✗ Error loading config: {e}')
    exit(1)
"

echo "Configuration test completed!"
