# Running
python research/mtm/train.py  -m +exp_exorl=exorl_cont "args.mask_patterns=[AUTO_MASK]" args.traj_length=4,8 wandb.project="11_mtm_exorl_pretrain_long"  args.seed=0,1,2,3 +hydra.launcher.partition=devlab args.num_train_steps=1000010
