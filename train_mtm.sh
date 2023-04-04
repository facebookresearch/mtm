# python research/mtm/train.py  -m +exp=d4rl_discrete "args.mask_patterns=[FULL_RANDOM],[AUTO_MASK],[BC],[ID],[FD],[FULL_RANDOM,BC],[FULL_RANDOM,ID],[FULL_RANDOM,FD],[AUTO_MASK,BC],[AUTO_MASK,ID],[AUTO_MASK,FD]" args.traj_length=2,4,6 args.batch_size=1024  dataset.env_name="walker2d-medium-replay-v2,walker2d-expert-v2" wandb.project="d4rl_01_03_discrete_exp"
# python research/mtm/train.py  -m +exp=d4rl_discrete "args.mask_patterns=[FULL_RANDOM],[AUTO_MASK],[BC],[ID],[FD],[FULL_RANDOM,BC],[FULL_RANDOM,ID],[FULL_RANDOM,FD],[AUTO_MASK,BC],[AUTO_MASK,ID],[AUTO_MASK,FD]" args.traj_length=2,4,6 args.batch_size=1024  dataset.env_name="walker2d-medium-expert-v2" wandb.project="d4rl_01_03_med_exp"


# python research/mtm/train.py  -m +exp=d4rl_discrete,d4rl_cont "args.mask_patterns=[FULL_RANDOM],[AUTO_MASK],[BC],[FULL_RANDOM,BC],[AUTO_MASK,BC],[FULL_RANDOM,AUTO_MASK,BC]" args.traj_length=1,2,4,6 dataset.env_name="walker2d-medium-replay-v2,hopper-medium-replay-v2,halfcheetah-medium-replay-v2" wandb.project="d4rl_debug_mtm"

# python research/mtm/train.py -m +exp=d4rl_discrete,d4rl_cont "args.mask_patterns=[AUTO_MASK],[BC],[AUTO_MASK,BC],[FULL_RANDOM,AUTO_MASK,BC]" args.traj_length=1,2,4 dataset.env_name="walker2d-medium-replay-v2,hopper-medium-replay-v2" wandb.project="d4rl_debug_mtm"  args.learning_rate=0.001,0.0003 model_config.n_embd=1024,256 model_config.n_enc_layer=2,4

# python research/mtm/train.py -m +exp=d4rl_cont "args.mask_patterns=[AUTO_MASK],[AUTO_MASK,RCBC]" args.traj_length=4 dataset.env_name="hopper-medium-replay-v2","walker2d-medium-replay-v2" wandb.project="d4rl_debug_mtm" model_config.n_embd=256,512,1024 args.weight_decay=0.001,0.1 args.log_every=102 model_config.latent_dim=null,8,16,64 args.seed=0,1,3

# python research/mtm/train.py -m +exp=d4rl_discrete,d4rl_cont "args.mask_patterns=[AUTO_MASK],[BC,RCBC],[AUTO_MASK,RCBC],[RCBC]" args.traj_length=4,6 dataset.env_name="hopper-medium-replay-v2" wandb.project="d4rl_debug_mtm" model_config.n_embd=64,128,256 args.weight_decay=0.001,0.01 args.log_every=101
#
#
# ####
# python research/mtm/train.py -m +exp=d4rl_q "args.mask_patterns=[AUTO_MASK],[AUTO_MASK,RCBC]" args.traj_length=2 dataset.env_name="hopper-medium-replay-v2" wandb.project="discete_debug_mtm" model_config.n_embd=32,64,128 model_config.latent_dim=4,8 args.seed=0,1,2 model_config.n_head=4,8,12 model_config.n_enc_layer=1,2,4


# python research/mtm/train.py -m +exp=d4rl_cont "args.mask_patterns=[AUTO_MASK],[AUTO_MASK,RCBC],[AUTO_MASK,RCBC,FULL_RANDOM,RCBC]" args.traj_length=4,8 dataset.env_name="hopper-medium-replay-v2" wandb.project="d4rl_debug_mtm" model_config.n_embd=128,256 model_config.latent_dim=null,16 args.seed=0,1,2,3,4 args.learning_rate=0.0001,0.0002
#
# python research/mtm/train.py -m +exp=d4rl_cont "args.mask_patterns=[AUTO_MASK,RCBC]" args.traj_length=6 dataset.env_name="hopper-medium-replay-v2" wandb.project="d4rl_debug_mtm" model_config.n_embd=128 model_config.latent_dim=16 args.seed=0,1,2,3,4 args.learning_rate=0.0003 model_config.dropout=0.0,0.1,0.2

# python research/mtm/train.py -m +exp=d4rl_cont_sm dataset.env_name="hopper-medium-replay-v2" wandb.project="d4rl_mtm_sm" args.seed=0,1,2,3,4

# python research/mtm/train.py -m +exp=d4rl_heterogenous "args.mask_patterns=[AUTO_MASK,RCBC,ID],[AUTO_MASK,RCBC],[AUTO_MASK],[ID,AUTO_MASK],[RCBC]" wandb.project="d4rl_hetero_med" args.seed=0,1,2,3,4 args.traj_length=4 args.log_every=101
# python research/mtm/train.py -m +exp=d4rl_option3 "args.mask_patterns=[AUTO_MASK,RCBC,ID],[AUTO_MASK,RCBC],[AUTO_MASK],[ID,AUTO_MASK],[RCBC]" wandb.project="d4rl_hetero_o3" args.seed=0,1,2,3,4 args.traj_length=4 args.log_every=102 dataset.train_val_split=0.01
# python research/mtm/train.py -m +exp=d4rl_cont "args.mask_patterns=[AUTO_MASK,RCBC,ID],[AUTO_MASK,RCBC],[AUTO_MASK],[ID,AUTO_MASK],[RCBC]" wandb.project="d4rl_hetero_o3" args.seed=0,1,2,3,4 args.traj_length=4 args.log_every=101 dataset.train_val_split=0.01,0.02,0.05,0.1,0.5

# python research/mtm/train.py -m +exp=d4rl_heterogenous1 "args.mask_patterns=[FULL_RANDOM,AUTO_MASK,RCBC,ID],[FULL_RANDOM,AUTO_MASK,RCBC],[FULL_RANDOM,AUTO_MASK],[ID,AUTO_MASK],[FULL_RANDOM,RCBC]" wandb.project="d4rl_hetero_1" args.seed=0,1,2,3,4 args.traj_length=4 args.log_every=101 model_config.n_embd=128 model_config.latent_dim=8

# python research/mtm/train.py -m +exp=d4rl_heterogenous "args.mask_patterns=[AUTO_MASK,RCBC],[AUTO_MASK],[RCBC]" wandb.project="d4rl_mtm_hp_hetero"  model_config.latent_dim=64 args.seed=0,1,2,3 args.traj_length=1,2,4 args.weight_decay=0.001,0.01,0.1 model_config.dropout=0.1,0.0

# python research/mtm/train.py -m +exp=d4rl_split_sm "args.mask_patterns=[AUTO_MASK]" args.traj_length=2 wandb.project="split_models" args.seed=0,1,2,3,4 model_config.n_head=8 tokenizers.states.splits=2 tokenizers.actions.splits=4 dataset.env_name=hopper-medium-v2,hopper-medium-replay-v2,hopper-medium-expert-v2,hopper-expert-v2,walker2d-medium-v2,walker2d-medium-replay-v2,walker2d-medium-expert-v2,walker2d-expert-v2,halfcheetah-medium-v2,halfcheetah-medium-replay-v2,halfcheetah-medium-expert-v2,halfcheetah-expert-v2 "args.mask_patterns=[FULL_RANDOM],[AUTO_MASK],[AUTO_MASK,FULL_RANDOM]"

# python research/mtm/train.py -m +exp=d4rl_split_sm "args.mask_patterns=[FULL_RANDOM],[AUTO_MASK],[AUTO_MASK,FULL_RANDOM]" args.traj_length=2,4 dataset.env_name="hopper-medium-replay-v2" wandb.project="split_mtm" args.seed=0,1 model_config.n_head=4,8 tokenizers.states.splits=1,2,4 tokenizers.actions.splits=1,2,4

# python research/mtm/train.py -m +exp=d4rl_cont_sm wandb.project="_tes_discount" args.seed=0,1,2,3,4 dataset.env_name=hopper-medium-replay-v2,walker2d-medium-replay-v2,halfcheetah-medium-replay-v2 dataset.discount=0.99,1.0,1.5 args.traj_length=8,16 "args.mode_weights=[0.2,0.1,0.7],[0.3,0.3,0.4],[0.5,0.1,0.4]"


# python -m pdb -c continue research/mtm/train.py +exp_urlb=urlb_cont wandb.project="_tes_discount" args.seed=0 dataset.env_name=walker_walk args.traj_length=8 "args.mode_weights=[0.2, 0.0, 0.8]" wandb.resume=null
python research/mtm/train.py -m +exp_urlb=urlb_cont wandb.project="urlb_1M" args.seed=0,1 dataset.env_name=walker_walk args.traj_length=8 "args.mode_weights=[0.2, 0.0, 0.8]" model_config.latent_dim=16,64

