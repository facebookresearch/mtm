# Instructions

Clone this repo.


# Use philipp's conda env on fair cluster
A working env is located at:
```
/private/home/philippwu/.conda/envs/dev
```

# Install python packages from scratch
If you want to make an env from scratch

Make a new conda env
```
conda create -n mtm python=3.10
conda activate mtm
```

We will install torch and jax with gpu support
Torch with GPU support
https://pytorch.org/get-started/locally/
Jax with GPU support
https://github.com/google/jax


Run these commands to install all dependencies
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
pip install -e .
```

Optionally install dev packages.
```
pip install -r requirements_dev.txt
```


### Code

#### pre-commit hooks

pre-commits hooks are great. This will automatically do some checking/formatting. To use the pre-commit hooks, run the following:

```
pip install pre-commit
pre-commit install
```

Basically this will run a code formater (black), sort imports (isort), remove unused imports, and clean up ipython notebooks. This is a nice effortless way to keep the code base have a consistent style.

If you want to make a commit without using the pre-commit hook, you can commit with the -n flag (ie. `git commit -n ...`).


# Running The MTM code
 * All code is located in the `mtm` folder.
 * To run the default code directly run `CUDA_VISIBLE_DEVICES=0 python research/mtm/train.py`

# Running specific experiments
 * Experiment configuration yaml files are located in `research/mtm/experiments`
 * to run one you can append a `+experiments="file_name"` to the command. For example, to use the `exorl_discrete.yaml` config run:
   * `CUDA_VISIBLE_DEVICES=0 python research/mtm/train.py +experiments=exorl_discrete`

## Datasets
 * all dataset code is located in the `research/mtm/datasets` folder. All datasets have to do is return a pytorch dataset that outputs a dict (named set of trajectories).
 * a dataset should follow the `DatasetProtocol` specified in `research/mtm/datasets/base.py`.
 * each dataset should also have corresponding `get_datasets` function where all the dataset specific construction logic happens. This function can take anything as input (as specified in the corresponding `yaml` config) and output the train and val torch `Dataset`.

## Tokenizers
 * All tokenizer code is found in the `research/mtm/tokenizers` folder. Each tokenizer should inherit from the `Tokenizer` abstract class, found in `research/mtm/tokenizers/base.py`
 * `Tokenizers` must define a `create` method, which can handle dataset specific construction logic.
