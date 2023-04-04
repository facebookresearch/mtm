python research/mtm/train.py  -m +exp_exorl=exorl_cont "args.mask_patterns=[AUTO_MASK]" args.traj_length=8 args.batch_size=1024 wandb.project="exorl_reb_3_16"  args.n_workers=0 args.seed=0,1,2
