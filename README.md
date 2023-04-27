# Masked Trajectory Model

This is the official code base for the paper `Masked Trajectory Models for Prediction, Representation, and Control`

If you find our work useful, consider citing:
```
@misc{wu2023mtm,
  author = {Wu, Philipp and Majumdar, Arjun and Stone, Kevin and Lin, Yixin and Mordatch, Igor and Abbeel, Pieter and Rajeswaran, Aravind},
  title = {Masked Trajectory Models for Prediction, Representation, and Control},
  booktitle = {International Conference on Machine Learning},
  year = {2023},
}
```

## Instructions

### Install python packages from scratch
If you want to make an env from scratch

Make a new conda env
```
conda create -n mtm python=3.10
conda activate mtm
```

Install torch with gpu
https://pytorch.org/get-started/locally/


Run these commands to install all dependencies
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -e .
```

Optionally install dev packages.
```
pip install -r requirements_dev.txt
```

### Adroit Experiments [Optional]
If you wish to run Adroit experiments, please install also install `robohive`.
 * [mjrl](https://github.com/aravindr93/mjrl/tree/pvr_beta_1) - use the `pvr_beta_1` branch
   * `pip install git+https://github.com/aravindr93/mjrl.git@83d35df95eb64274c5e93bb32a0a4e2f6576638a`
 * [robohive](https://github.com/vikashplus/robohive/tree/stable) - use the `stable` branch (note, it uses gitsubmodules, follow install instructions exactly)
   * specifically, we have tested using this commit - `c1557f5572977085f053df63f4e81f4b4e1fb17c`
* Additionally you must download the adroit datasets and put them in the `~/mtm_data` directory

# Running The MTM code
Example commands can be found in `train_examples.sh`
An example notebook is located at `example_train_sinusoid.ipynb` which shows a simple example of how MTM can be used for trajectory prediction on a sinusoid dataset.

The main code is located in the `mtm` folder. Here is how you can run some of the experiments.
 * Simple sinusoidal test data `python research/mtm/train.py +exp_mtm=sinusoid_cont`
 * D4RL `python research/mtm/train.py +exp_mtm=d4rl_cont`
 * Adroit `python research/mtm/train.py +exp_mtm=adroit_cont`

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
 * all dataset code is located in the `research/mtm/datasets` folder. All datasets have to do is return a pytorch dataset that outputs a dict (named set of trajectories).
 * a dataset should follow the `DatasetProtocol` specified in `research/mtm/datasets/base.py`.
 * each dataset should also have corresponding `get_datasets` function where all the dataset specific construction logic happens. This function can take anything as input (as specified in the corresponding `yaml` config) and output the train and val torch `Dataset`.

### Tokenizers
 * All tokenizer code is found in the `research/mtm/tokenizers` folder. Each tokenizer should inherit from the `Tokenizer` abstract class, found in `research/mtm/tokenizers/base.py`
 * `Tokenizers` must define a `create` method, which can handle dataset specific construction logic.

# Acknowledgements
This research would not be possible without building on top of existing open source code. We would like to acknowledge and thank the following:
 * [FangchenLiu/MaskDP_public](https://github.com/FangchenLiu/MaskDP_public): Masked Decision Prediction, which this work builds upon
 * [ikostrikov/jaxrl](https://github.com/ikostrikov/jaxrl): A fast Jax library for RL. We used this environment wrapping and data loading code for all d4rl experiments.
 * [denisyarats/exorl](https://github.com/denisyarats/exorl): ExORL provides datasets collected with unsupervised RL methods which we use in representation learning experiments
 * [vikashplus/robohive](https://github.com/brentyi/tyro): Provides the Adroit environment
 * [aravindr93/mjrl](https://github.com/aravindr93/mjrl): Code for training the policy for generating data on Adroit
 * [brentyi/tyro](https://github.com/brentyi/tyro): Argument parsing and configuration
