# ***********************
# 4.3 Experiments
# ***********************
#
########
# D4RL
########
# MTM
python research/mtm/train.py -m +exp_mtm=d4rl_cont wandb.project="mtm_d4rl_04_4_3" args.seed=0,1,2,3,4,5 dataset.env_name=hopper-medium-v2,hopper-medium-replay-v2,hopper-medium-expert-v2,hopper-expert-v2,walker2d-medium-v2,walker2d-medium-replay-v2,walker2d-medium-expert-v2,walker2d-expert-v2,halfcheetah-medium-v2,halfcheetah-medium-replay-v2,halfcheetah-medium-expert-v2,halfcheetah-expert-v2 "args.mask_patterns=[AUTO_MASK],[ID],[FD],[BC],[RCBC]" args.log_every=100

# MLP
python research/mtm/train_mlp.py -m +exp_mlp=d4rl_cont wandb.project="mlp_d4rl_04_4_3" args.seed=0,1,2,3,4,5 dataset.env_name=hopper-medium-v2,hopper-medium-replay-v2,hopper-medium-expert-v2,hopper-expert-v2,walker2d-medium-v2,walker2d-medium-replay-v2,walker2d-medium-expert-v2,walker2d-expert-v2,halfcheetah-medium-v2,halfcheetah-medium-replay-v2,halfcheetah-medium-expert-v2,halfcheetah-expert-v2 model_config.task="id","fd" args.log_every=100 args.traj_length=2
python research/mtm/train_mlp.py -m +exp_mlp=d4rl_cont wandb.project="mlp_d4rl_04_4_3" args.seed=0,1,2,3,4,5 dataset.env_name=hopper-medium-v2,hopper-medium-replay-v2,hopper-medium-expert-v2,hopper-expert-v2,walker2d-medium-v2,walker2d-medium-replay-v2,walker2d-medium-expert-v2,walker2d-expert-v2,halfcheetah-medium-v2,halfcheetah-medium-replay-v2,halfcheetah-medium-expert-v2,halfcheetah-expert-v2 model_config.task="bc","rcbc" args.log_every=100 args.traj_length=1

########
# Adroit
########
# MTM
python research/mtm/train.py -m +exp_mtm=adroit_cont wandb.project="mtm_adroit_04_4_3" args.seed=0,1,2,3,4,5 dataset.env_name=relocate,pen,hammer,door dataset.d_name="expert","medium_replay" "args.mask_patterns=[AUTO_MASK],[ID],[FD],[BC],[RCBC]" args.log_every=100

# MLP
python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont wandb.project="mlp_adroit_04_4_3" seed=0,1,2,3,4,5 dataset.env_name=relocate,pen,hammer,door dataset.d_name="expert","medium_replay" model_config.task="bc","rcbc" args.args.log_every=100 args.traj_length=1
python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont wandb.project="mlp_adroit_04_4_3" seed=0,1,2,3,4,5 dataset.env_name=relocate,pen,hammer,door dataset.d_name="expert","medium_replay" model_config.task="id","fd" args.args.log_every=100 args.traj_length=2

# ***********************
# 4.4 Imact of masking pattern
# ***********************
#
########
# D4RL
########
python research/mtm/train.py -m +exp_mtm=d4rl_cont wandb.project="mtm_d4rl_04_4_4" args.seed=0,1,2,3,4,5     dataset.env_name=halfcheetah-medium-replay-v2,halfcheetah-expert-v2 "args.mask_patterns=[AUTO_MASK],[RCBC],[RANDOM]"
python research/mtm/train.py -m +exp_mtm=adroit_cont wandb.project="mtm_adroit_04_4_4" args.seed=0,1,2,3,4,5 dataset.env_name=pen,door dataset.d_name="medium_replay" "args.mask_patterns=[AUTO_MASK],[RCBC],[RANDOM]" args.log_every=100

# ***********************
# 4.5 Hetermodal
# ***********************
#
########
# D4RL
########
# MTM heteromodal 0.01 with actions 0.95 with states only
python research/mtm/train.py -m +exp_mtm=d4rl_halfcheetah_o3,d4rl_walker2d_o3,d4rl_hopper_o3 "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="mtm_d4rl_04_4_5" args.seed=0,1,2,3,4 dataset.train_val_split=0.01
# MTM 0.01 of the dataset
python research/mtm/train.py -m +exp_mtm=d4rl_cont dataset.train_val_split=0.01 dataset.env_name=hopper-expert-v2,walker2d-expert-v2,halfcheetah-expert-v2 "args.mask_patterns=[ID,AUTO_MASK],[AUTO_MASK]" wandb.project="mtm_d4rl_04_4_5" args.seed=0,1,2,3,4
# MLP 0.01 of the dataset
python research/mtm/train_mlp.py -m +exp_mlp=d4rl_cont dataset.env_name=hopper-expert-v2,walker2d-expert-v2,halfcheetah-expert-v2 wandb.project="mtm_d4rl_04_4_5" model_config.task="bc","rcbc" args.seed=0,1,2,3,4 args.traj_length=1 dataset.train_val_split=0.01,0.95

########
# Adroit
########
# MTM heteromodal 0.01 with actions 0.95 with states only
python research/mtm/train.py -m +exp_mtm=adroit_door_option3,adroit_pen_option3,adroit_hammer_option3,adroit_relocate_option3  "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="mtm_adroit_04_4_5" args.seed=0,1,2,3,4
# MTM 0.01 of the dataset
python research/mtm/train.py -m +exp_mtm=adroit_cont dataset.train_val_split=0.01 dataset.env_name="pen","hammer","relocate","door" "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="mtm_adroit_04_4_5" args.seed=0,1,2,3,4
# MLP 0.01 of the dataset
python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=relocate,pen,hammer,door wandb.project="mtm_adroit_04_4_5" model_config.task="bc","rcbc" args.seed=0,1,2,3,4 args.traj_length=1 dataset.d_name="expert" dataset.train_val_split=0.01,0.95

# ***********************
# 4.6 Data Efficiency
# ***********************
#
########
# D4RL
########
python research/mtm/train_mlp.py -m +exp_mlp=d4rl_cont dataset.env_name=hopper-expert-v2 wandb.project="data_eff_hopper_04_4_6" model_config.task="bc","rcbc" args.seed=0,1,2,3,4 args.traj_length=1 dataset.train_val_split=0.005,0.01,0.02,0.05,0.5,0.95
python research/mtm/train.py -m +exp_mtm=d4rl_cont dataset.env_name=hopper-expert-v2 "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="data_eff_hopper_04_4_6" args.seed=0,1,2,3,4 dataset.train_val_split=0.005,0.01,0.02,0.05,0.5,0.95
python research/mtm/train.py -m +exp_mtm=d4rl_hopper_o3 "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="data_eff_hopper_04_4_6" args.seed=0,1,2,3,4 dataset.train_val_split=0.005,0.01,0.02,0.05,0.5,0.95
########
# Adroit
########
python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=pen wandb.project="data_eff_pen_04_4_6" model_config.task="bc","rcbc" args.seed=0,1,2,3,4 args.traj_length=1 dataset.train_val_split=0.005,0.01,0.02,0.05,0.5,0.95
python research/mtm/train.py -m +exp_mtm=adroit_cont  dataset.env_name=pen "args.mask_patterns=[ID,AUTO_MASK]" wand.project="data_eff_pen_04_4_6" args.seed=0,1,2,3,4 dataset.train_val_split=0.005,0.01,0.02,0.05,0.5,0.95
python research/mtm/train.py -m +exp_mtm=adroit_pen_option3 "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="data_eff_pen_04_4_6" args.seed=0,1,2,3,4 dataset.train_val_split=0.005,0.01,0.02,0.05,0.5,0.95

# ***********************
# Trajectory Length (appendix)
# ***********************
#
########
# D4RL
########
python research/mtm/train.py -m +exp_mtm=d4rl_cont wandb.project="mtm_d4rl_04_appendix_0" args.seed=0,1,2,3,4,5 dataset.env_name=hopper-medium-replay-v2,walker2d-medium-replay-v2,halfcheetah-medium-replay-v2 "args.mask_patterns=[AUTO_MASK]" args.traj_length=1,2,4,8,16,32,64
#
########
# Adroit
########
python research/mtm/train.py -m +exp_mtm=adroit_cont wandb.project="mtm_adroit_04_appendix_0" args.seed=0,1,2,3,4,5 dataset.env_name=pen,door dataset.d_name="medium_replay" "args.mask_patterns=[AUTO_MASK]"  args.traj_length=1,2,4,8,16,32,64
