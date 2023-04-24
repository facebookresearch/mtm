# ***************************
# Running MTM
# ***************************
# MTM on D4RL Hoppper Medium-Replay
python research/mtm/train.py +exp_mtm=d4rl_cont wandb.project="11_mtm_d4rl_4_3" args.seed=0 dataset.env_name=hopper-medium-replay-v2 "args.mask_patterns=[AUTO_MASK]"


# ***************************
# Running Heteromodal MTM
# ***************************
# Heteromodal MTM 0.01 with actions 0.95 with states only
python research/mtm/train.py +exp_mtm=d4rl_halfcheetah_o3 "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="11_mtm_d4rl_4_5" args.seed=0 dataset.train_val_split=0.01
# Vanilla MTM 0.01 of the dataset
python research/mtm/train.py +exp_mtm=d4rl_cont dataset.train_val_split=0.01 dataset.env_name=halfcheetah-expert-v2 "args.mask_patterns=[AUTO_MASK]" wandb.project="11_mtm_d4rl_4_5" args.seed=0
