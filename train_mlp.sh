# python research/mtm/train_mlp.py -m +exp_mlp=d4rl_discrete dataset.env_name=halfcheetah-medium-replay-v2,halfcheetah-expert-v2,walker2d-medium-replay-v2,walker2d-expert-v2
# python research/mtm/train_mlp.py -m +exp_mlp=d4rl_discrete dataset.env_name=halfcheetah-medium-replay-v2,halfcheetah-expert-v2,walker2d-medium-replay-v2,walker2d-expert-v2 model_config.task="fd" args.traj_length=2
# python research/mtm/train_mlp.py -m +exp_mlp=d4rl_discrete dataset.env_name=halfcheetah-medium-replay-v2,halfcheetah-expert-v2,walker2d-medium-replay-v2,walker2d-expert-v2 model_config.task="id" args.traj_length=2
# python research/mtm/train_mlp.py -m +exp_mlp=d4rl_discrete dataset.env_name=halfcheetah-medium-replay-v2,halfcheetah-expert-v2,walker2d-medium-replay-v2,walker2d-expert-v2 model_config.task="rcbc" args.traj_length=2
# python research/mtm/train_mlp.py -m +exp_mlp=d4rl_discrete dataset.env_name=halfcheetah-medium-replay-v2,halfcheetah-expert-v2,walker2d-medium-replay-v2,walker2d-expert-v2 model_config.task="fd" args.traj_length=2
# python research/mtm/train_mlp.py -m +exp_mlp=d4rl_discrete dataset.env_name=halfcheetah-medium-replay-v2,halfcheetah-expert-v2,walker2d-medium-replay-v2,walker2d-expert-v2 model_config.task="fd" args.traj_length=2

# python research/mtm/train_mlp.py -m +exp_mlp=d4rl_discrete,d4rl_cont dataset.env_name=hopper-medium-v2,hopper-medium-replay-v2,hopper-medium-expert-v2,hopper-expert-v2,walker2d-medium-v2,walker2d-medium-replay-v2,walker2d-medium-expert-v2,walker2d-expert-v2,halfcheetah-medium-v2,halfcheetah-medium-replay-v2,halfcheetah-medium-expert-v2,halfcheetah-expert-v2 wandb.project=mlp_baselines model_config.task="id","fd" args.seed=0,1,2,3,4 args.log_every=101 args.traj_length=2 model_config.dropout=0,0.1
#
# python research/mtm/train_mlp.py -m +exp_mlp=d4rl_cont dataset.env_name=hopper-medium-v2,hopper-medium-replay-v2,hopper-medium-expert-v2,hopper-expert-v2,walker2d-medium-v2,walker2d-medium-replay-v2,walker2d-medium-expert-v2,walker2d-expert-v2,halfcheetah-medium-v2,halfcheetah-medium-replay-v2,halfcheetah-medium-expert-v2,halfcheetah-expert-v2 wandb.project=mlp_baselines model_config.task="id" args.seed=0,1,2,3,4 args.log_every=101 args.traj_length=2

# python research/mtm/train_mlp.py -m +exp_mlp=d4rl_cont dataset.env_name=hopper-expert-v2 wandb.project=d4rl_hetero_o3 model_config.task="bc","rcbc" args.seed=0,1,2,3,4 args.log_every=101 args.traj_length=2 dataset.train_val_split=0.01,0.02,0.05,0.1,0.5





# ############
# # Adroit
# ############
# python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=relocate,pen,hammer,door wandb.project=fixed_baselines_adroit model_config.task="bc","rcbc" args.seed=0,1,2,3,4 args.log_every=100 args.traj_length=1
# python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=relocate,pen,hammer,door wandb.project=fixed_baselines_adroit model_config.task="id","fd" args.seed=0,1,2,3,4 args.log_every=100 args.traj_length=2
# python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=relocate,pen,hammer,door wandb.project=fixed_baselines_adroit model_config.task="bc" args.seed=0,1,2,3,4 args.log_every=100 args.traj_length=1 dataset.train_val_split=0.01,0.05,0.1

python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=relocate,pen,hammer,door wandb.project=adroit_paper model_config.task="bc","rcbc" args.seed=0,1,2,3,4,5,6,7,8,9 args.log_every=101 args.traj_length=1 dataset.d_name="expert","medium_replay"
python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=relocate,pen,hammer,door wandb.project=adroit_paper model_config.task="id","fd" args.seed=0,1,2,3,4,5,6,7,8,9 args.log_every=101 args.traj_length=2 dataset.d_name="expert","medium_replay"
