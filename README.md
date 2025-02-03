# Masked Trajectory Models

This is the official code base for the paper `GPS Masked Trajectory Models for Human Life Travel Pattern Learning"`

## Instructions

# Running The MTM code
Example commands can be found in `train_examples.sh` Run the last command in this file.

The main code is located in the `mtm` folder. 

### Configuring MTM
 * The config file for mtm is located at `research/mtm/config.yaml`
 * Some key parameters
   * `traj_length`: The length of trajectory sub-segments
   * `mask_ratios`: A list of mask ratios that is randomly sampled
   * `mask_pattterns`: A list of masking patterns that are randomly sampled. See `MaskType` under `research/mtm/masks.py` for supported options.
   * `mode_weights`: (Only applies for `AUTO_MASK`) A list of weights that samples which mode is to be the "autoregressive" one. For example, if the mode order is, `states`, `returns`, `actions`, and mode_weights = [0.2, 0.1, 0.7], then with 0.7 probability, the action token and all future tokens will be masked out.

# Code Organization

### pre-commit hooks

pre-commits hooks are great. This will automatically do some checking/formatting. To use the pre-commit hooks, run the following:
```
pip install pre-commit
pre-commit install
```

If you want to make a commit without using the pre-commit hook, you can commit with the -n flag (ie. `git commit -n ...`).

### Datasets
 * all dataset code is located in the `/Trajectory_dataset/anomaly_traj_data` folder. All datasets have to do is return a pytorch dataset that outputs a dict (named set of trajectories).
 * a dataset should follow the `DatasetProtocol` specified in `research/mtm/datasets/base.py`.
 * each dataset should also have a corresponding `get_datasets` function where all the dataset specific construction logic happens. This function can take anything as input (as specified in the corresponding `yaml` config) and output the train and val torch `Dataset`.

### Tokenizers
 * All tokenizer code is found in the `research/mtm/tokenizers` folder. Each tokenizer should inherit from the `Tokenizer` abstract class, found in `research/mtm/tokenizers/base.py`
 * `Tokenizers` must define a `create` method, which can handle dataset specific construction logic.

# License & Acknowledgements
This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree. This is not an official Meta product.

This project builds on top of or utilizes the following third party dependencies.
 * [FangchenLiu/MaskDP_public](https://github.com/FangchenLiu/MaskDP_public): Masked Decision Prediction, which this work builds upon
 * [ikostrikov/jaxrl](https://github.com/ikostrikov/jaxrl): A fast Jax library for RL. We used this environment wrapping and data loading code for all d4rl experiments.
 * [denisyarats/exorl](https://github.com/denisyarats/exorl): ExORL provides datasets collected with unsupervised RL methods which we use in representation learning experiments
 * [vikashplus/robohive](https://github.com/brentyi/tyro): Provides the Adroit environment
 * [aravindr93/mjrl](https://github.com/aravindr93/mjrl): Code for training the policy for generating data on Adroit
 * [brentyi/tyro](https://github.com/brentyi/tyro): Argument parsing and configuration
# GPS-MTM
# GPS-MTM
