# python research/mtm/train.py -m +exp=d4rl_cont_sm wandb.project="d4rl_mtm_cont_1_23" args.seed=10,11,12,13,14,15,16,17,18,19 dataset.env_name=hopper-medium-v2,hopper-medium-replay-v2,hopper-medium-expert-v2,hopper-expert-v2,walker2d-medium-v2,walker2d-medium-replay-v2,walker2d-medium-expert-v2,walker2d-expert-v2,halfcheetah-medium-v2,halfcheetah-medium-replay-v2,halfcheetah-medium-expert-v2,halfcheetah-expert-v2

# python research/mtm/train.py -m +exp=d4rl_q wandb.project="d4rl_mtm_sm" args.seed=0,1,2,3,4 dataset.env_name=hopper-medium-v2,hopper-medium-replay-v2,hopper-medium-expert-v2,hopper-expert-v2,walker2d-medium-v2,walker2d-medium-replay-v2,walker2d-medium-expert-v2,walker2d-expert-v2,halfcheetah-medium-v2,halfcheetah-medium-replay-v2,halfcheetah-medium-expert-v2,halfcheetah-expert-v2

# python research/mtm/train.py -m +exp=adroit_cont wandb.project="adroit_mtm" args.seed=0,1,2,3,4 dataset.env_name=relocate,pen,hammer,door "args.mask_patterns=[AUTO_MASK],[RCBC]" args.log_every=110
# python research/mtm/train.py -m +exp=adroit_cont wandb.project="adroit_paper" args.seed=0,1,2,3,4 dataset.env_name=relocate,pen,hammer,door dataset.d_name="expert","medium_replay" "args.mask_patterns=[AUTO_MASK],[RCBC]" args.log_every=103

# python research/mtm/train.py -m +exp=d4rl_cont wandb.project="d4rl_mtm_paper" args.seed=0,1,2,3,4 dataset.env_name=hopper-medium-v2,hopper-medium-replay-v2,hopper-medium-expert-v2,hopper-expert-v2,walker2d-medium-v2,walker2d-medium-replay-v2,walker2d-medium-expert-v2,walker2d-expert-v2,halfcheetah-medium-v2,halfcheetah-medium-replay-v2,halfcheetah-medium-expert-v2,halfcheetah-expert-v2 "args.mask_patterns=[ID],[FD],[RCBC],[BC]"  args.log_every=110

# fixed FD mask
# python research/mtm/train.py -m +exp=d4rl_cont wandb.project="d4rl_mtm_paper" args.seed=5,6,7,8,9 dataset.env_name=hopper-medium-v2,hopper-medium-replay-v2,hopper-medium-expert-v2,hopper-expert-v2,walker2d-medium-v2,walker2d-medium-replay-v2,walker2d-medium-expert-v2,walker2d-expert-v2,halfcheetah-medium-v2,halfcheetah-medium-replay-v2,halfcheetah-medium-expert-v2,halfcheetah-expert-v2  "args.mask_patterns=[ID],[FD],[RCBC],[BC]" args.log_every=110

# python research/mtm/train.py -m +exp=adroit_cont wandb.project="adroit_final" args.seed=0,1,2,3,4,5,6,7,8,9 dataset.env_name=hammer,relocate,pen,door dataset.d_name="expert","medium_replay" "args.mask_patterns=[AUTO_MASK],[RCBC],[FD],[ID],[BC]"

# python research/mtm/train.py -m +exp=adroit_cont wandb.project="adroit_final" args.seed=0,1,2,3,4 dataset.env_name=hammer,relocate,pen,door dataset.d_name="expert","medium_replay" "args.mask_patterns=[AUTO_MASK],[RCBC],[FD],[ID],[BC]"

# python research/mtm/train.py -m +exp=d4rl_cont_sm wandb.project="d4rl_mtm_cont_1_24" args.seed=0,1,2,3,4,5,6,7,8,9 dataset.env_name=hopper-medium-v2,hopper-medium-replay-v2,hopper-medium-expert-v2,hopper-expert-v2,walker2d-medium-v2,walker2d-medium-replay-v2,walker2d-medium-expert-v2,walker2d-expert-v2,halfcheetah-medium-v2,halfcheetah-medium-replay-v2,halfcheetah-medium-expert-v2,halfcheetah-expert-v2 "args.mask_patterns=[AUTO_MASK],[AUTO_MASK, AUTO_MASK, RCBC]"

# python research/mtm/train.py -m +exp=d4rl_cont_sm wandb.project="d4rl_mtm_mask_ablation" args.seed=0,1,2,3,4,5,6,7,8,9 dataset.env_name=hopper-medium-replay-v2,hopper-expert-v2,walker2d-medium-replay-v2,walker2d-expert-v2,halfcheetah-medium-replay-v2,halfcheetah-expert-v2 "args.mask_patterns=[AUTO_MASK],[RCBC],[BC],[RANDOM],[AUTO_MASK,RCBC], [RANDOM, RCBC]"

# python research/mtm/train.py -m +exp=d4rl_cont_sm wandb.project="d4rl_mtm_cont_1_25" args.seed=0,1,2,3,4 dataset.env_name=halfcheetah-medium-v2,halfcheetah-medium-replay-v2,halfcheetah-medium-expert-v2,halfcheetah-expert-v2 "args.mask_patterns=[AUTO_MASK],[AUTO_MASK,BC],[AUTO_MASK,RCBC]" model_config.n_enc_layers=1,2,3,4
# python research/mtm/train.py -m +exp=adroit_cont wandb.project="adroit_relocate_1_25" args.seed=0,1,2,3,4 dataset.env_name=relocate dataset.d_name="expert","medium_replay" "args.mask_patterns=[AUTO_MASK],[AUTO_MASK,RCBC],[AUTO_MASK, BC]"  model_config.n_enc_layers=1,2,3,4

# python research/mtm/train.py -m +exp=d4rl_cont_sm wandb.project="d4rl_mtm_mask_ablation" args.seed=0,1,2,3,4,5,6,7,8,9 dataset.env_name=halfcheetah-medium-replay-v2,halfcheetah-expert-v2 "args.mask_patterns=[AUTO_MASK],[RCBC],[BC],[RANDOM],[AUTO_MASK,RCBC],[RANDOM, RCBC]"

# python research/mtm/train.py -m +exp=d4rl_cont_sm wandb.project="d4rl_mtm_mask_ablation" args.seed=0,1,2,3,4,5,6,7,8,9 dataset.env_name=halfcheetah-medium-replay-v2,halfcheetah-expert-v2 "args.mask_patterns=[AUTO_MASK],[RCBC],[BC],[RANDOM],[AUTO_MASK,RCBC],[RANDOM, RCBC]"

# python research/mtm/train.py -m +exp=d4rl_cont_sm wandb.project="d4rl_mtm_mask_ablation" args.seed=0,1,2,3,4,5,6,7,8,9 dataset.env_name=halfcheetah-medium-replay-v2,halfcheetah-expert-v2 "args.mask_patterns=[AUTO_MASK],[RCBC],[BC],[RANDOM],[AUTO_MASK,RCBC],[RANDOM, RCBC]"

python research/mtm/train.py -m +exp=d4rl_cont_sm wandb.project="d4rl_1_27_mtm" args.seed=0,1,2,3,4,5 dataset.env_name=hopper-medium-v2,hopper-medium-replay-v2,hopper-medium-expert-v2,hopper-expert-v2,walker2d-medium-v2,walker2d-medium-replay-v2,walker2d-medium-expert-v2,walker2d-expert-v2,halfcheetah-medium-v2,halfcheetah-medium-replay-v2,halfcheetah-medium-expert-v2,halfcheetah-expert-v2 "args.mask_patterns=[AUTO_MASK]"
