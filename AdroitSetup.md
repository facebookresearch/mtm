# Setting up Adroit [Optional]

## Environment Setup

If you wish to run Adroit experiments, please also install `mjrl` and `robohive`.
 * [mjrl](https://github.com/aravindr93/mjrl/tree/pvr_beta_1) - use the `pvr_beta_1` branch
   * `pip install git+https://github.com/aravindr93/mjrl.git@83d35df95eb64274c5e93bb32a0a4e2f6576638a`
 * [robohive](https://github.com/vikashplus/robohive/tree/stable) - use the `stable` branch (note, it uses gitsubmodules, follow install instructions exactly)
   * specifically, we have tested using this commit - `c1557f5572977085f053df63f4e81f4b4e1fb17c`

## Dataset Download
Additionally you must download the adroit datasets and put them in the `~` directory (or change the `data_dir` in the `research/mtm/datasets/adroit.py` file).
The datasets can be found here https://github.com/facebookresearch/mtm/releases/tag/Datasets-v1

You can download and unzip as follows
```
cd ~
wget https://github.com/facebookresearch/mtm/releases/download/Datasets-v1/adroit_datasets.zip
unzip adroit_datasets.zip
rm -rf adroit_datasets.zip
```

## Run experiments
You are now ready for running experiments. Try:
`python research/mtm/train.py +exp_mtm=adroit_cont`
