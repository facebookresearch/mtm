# ***********************
# 4.3 Experiments
# ***********************
python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont wandb.project="12_mlp_adroit_4_3" args.seed=0,1,2,3 dataset.env_name=pen,door dataset.d_name="expert","medium_replay" model_config.task="id" args.traj_length=2
python research/mtm/train_mlp.py -m +exp_mlp=d4rl_cont wandb.project="12_mlp_d4rl_4_3" args.seed=0,1,2,3 dataset.env_name=hopper-medium-v2,hopper-medium-replay-v2,hopper-medium-expert-v2,hopper-expert-v2,walker2d-medium-v2,walker2d-medium-replay-v2,walker2d-medium-expert-v2,walker2d-expert-v2,halfcheetah-medium-v2,halfcheetah-medium-replay-v2,halfcheetah-medium-expert-v2,halfcheetah-expert-v2 model_config.task="id" args.traj_length=2
#
########
# D4RL
########
# MTM
# Running
python research/mtm/train.py -m +exp_mtm=d4rl_cont wandb.project="11_mtm_d4rl_4_3" args.seed=0,1,2,3 dataset.env_name=hopper-medium-v2,hopper-medium-replay-v2,hopper-medium-expert-v2,hopper-expert-v2,walker2d-medium-v2,walker2d-medium-replay-v2,walker2d-medium-expert-v2,walker2d-expert-v2,halfcheetah-medium-v2,halfcheetah-medium-replay-v2,halfcheetah-medium-expert-v2,halfcheetah-expert-v2 "args.mask_patterns=[AUTO_MASK],[ID],[FD],[BC],[RCBC]"

# MLP
# Running
python research/mtm/train_mlp.py -m +exp_mlp=d4rl_cont wandb.project="11_mlp_d4rl_4_3" args.seed=0,1,2,3 dataset.env_name=hopper-medium-v2,hopper-medium-replay-v2,hopper-medium-expert-v2,hopper-expert-v2,walker2d-medium-v2,walker2d-medium-replay-v2,walker2d-medium-expert-v2,walker2d-expert-v2,halfcheetah-medium-v2,halfcheetah-medium-replay-v2,halfcheetah-medium-expert-v2,halfcheetah-expert-v2 model_config.task="id","fd" args.traj_length=2
# Running
python research/mtm/train_mlp.py -m +exp_mlp=d4rl_cont wandb.project="11_mlp_d4rl_4_3" args.seed=0,1,2,3 dataset.env_name=hopper-medium-v2,hopper-medium-replay-v2,hopper-medium-expert-v2,hopper-expert-v2,walker2d-medium-v2,walker2d-medium-replay-v2,walker2d-medium-expert-v2,walker2d-expert-v2,halfcheetah-medium-v2,halfcheetah-medium-replay-v2,halfcheetah-medium-expert-v2,halfcheetah-expert-v2 model_config.task="bc","rcbc" args.traj_length=1

########
# Adroit
########
# MTM
# Running
python research/mtm/train.py -m +exp_mtm=adroit_cont wandb.project="11_mtm_adroit_4_3" args.seed=0,1,2,3 dataset.env_name=pen,door dataset.d_name="expert","medium_replay" "args.mask_patterns=[AUTO_MASK],[ID],[FD],[BC],[RCBC]"
# Running
python research/mtm/train.py -m +exp_mtm=adroit_cont wandb.project="12_mtm_adroit_4_3_auto" args.seed=0,1,2,3 dataset.env_name=pen,door dataset.d_name="expert","medium_replay" "args.mask_patterns=[AUTO_MASK]"

# MLP
# Running
python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont wandb.project="11_mlp_adroit_4_3" args.seed=0,1,2,3 dataset.env_name=pen,door dataset.d_name="expert","medium_replay" model_config.task="bc","rcbc" args.traj_length=1
# Running
python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont wandb.project="11_mlp_adroit_4_3" args.seed=0,1,2,3 dataset.env_name=pen,door dataset.d_name="expert","medium_replay" model_config.task="id","fd" args.traj_length=2

# ***********************
# 4.4 Imact of masking pattern
# ***********************
#
########
# D4RL
########
# Running
python research/mtm/train.py -m +exp_mtm=d4rl_cont wandb.project="11_mtm_d4rl_4_4" args.seed=0,1,2,3     dataset.env_name=halfcheetah-medium-replay-v2,halfcheetah-expert-v2 "args.mask_patterns=[AUTO_MASK],[RCBC],[RANDOM]"
# Running
python research/mtm/train.py -m +exp_mtm=adroit_cont wandb.project="11_mtm_adroit_4_4" args.seed=0,1,2,3 dataset.env_name=pen,door dataset.d_name="medium_replay" "args.mask_patterns=[AUTO_MASK],[RCBC],[RANDOM]"

# ***********************
# 4.5 Hetermodal
# ***********************
#
########
# D4RL
########
# MTM heteromodal 0.01 with actions 0.95 with states only
# Running
python research/mtm/train.py -m +exp_mtm=d4rl_halfcheetah_o3,d4rl_walker2d_o3,d4rl_hopper_o3 "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="11_mtm_d4rl_4_5" args.seed=0,1,2,3 dataset.train_val_split=0.01
# MTM 0.01 of the dataset
# Running
python research/mtm/train.py -m +exp_mtm=d4rl_cont dataset.train_val_split=0.01 dataset.env_name=hopper-expert-v2,walker2d-expert-v2,halfcheetah-expert-v2 "args.mask_patterns=[AUTO_MASK]" wandb.project="11_mtm_d4rl_4_5" args.seed=0,1,2,3
# MLP 0.01 of the dataset
# Running
python research/mtm/train_mlp.py -m +exp_mlp=d4rl_cont dataset.env_name=hopper-expert-v2,walker2d-expert-v2,halfcheetah-expert-v2 wandb.project="11_mtm_d4rl_4_5" model_config.task="bc","rcbc" args.seed=0,1,2,3 args.traj_length=1 dataset.train_val_split=0.01

########
# Adroit
########
# MTM heteromodal 0.01 with actions 0.95 with states only
# Running
python research/mtm/train.py -m +exp_mtm=adroit_door_option3,adroit_pen_option3 "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="11_mtm_adroit_4_5" args.seed=0,1,2,3
# MTM 0.01 of the dataset
# Running
python research/mtm/train.py -m +exp_mtm=adroit_cont dataset.train_val_split=0.01 dataset.env_name=pen,door "args.mask_patterns=[AUTO_MASK]" wandb.project="11_mtm_adroit_4_5" args.seed=0,1,2,3
# MLP 0.01 of the dataset
# Running
python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=pen,door wandb.project="11_mtm_adroit_4_5" model_config.task="bc","rcbc" args.seed=0,1,2,3 args.traj_length=1 dataset.d_name="expert" dataset.train_val_split=0.01

# ***********************
# 4.6 Data Efficiency
# ***********************
#
########
# D4RL
########
# Running
python research/mtm/train_mlp.py -m +exp_mlp=d4rl_cont dataset.env_name=hopper-expert-v2 wandb.project="11_data_eff_hopper_4_6" model_config.task="bc","rcbc" args.seed=0,1,2,3 args.traj_length=1 dataset.train_val_split=0.005,0.01,0.02,0.05,0.5,0.95
# Running
python research/mtm/train.py -m +exp_mtm=d4rl_cont dataset.env_name=hopper-expert-v2 "args.mask_patterns=[AUTO_MASK]" wandb.project="11_data_eff_hopper_4_6" args.seed=0,1,2,3 dataset.train_val_split=0.005,0.01,0.02,0.05,0.5,0.95
# Running
python research/mtm/train.py -m +exp_mtm=d4rl_hopper_o3 "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="11_data_eff_hopper_4_6" args.seed=0,1,2,3 dataset.train_val_split=0.005,0.01,0.02,0.05,0.5,0.95
########
# Adroit
########
# Running
python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=pen wandb.project="11_data_eff_pen_4_6" model_config.task="bc","rcbc" args.seed=0,1,2,3 args.traj_length=1 dataset.train_val_split=0.005,0.01,0.02,0.05,0.5,0.95
# Running
python research/mtm/train.py -m +exp_mtm=adroit_cont  dataset.env_name=pen "args.mask_patterns=[AUTO_MASK]" wandb.project="11_data_eff_pen_4_6" args.seed=0,1,2,3 dataset.train_val_split=0.005,0.01,0.02,0.05,0.5,0.95
# Running
python research/mtm/train.py -m +exp_mtm=adroit_pen_option3 "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="11_data_eff_pen_4_6" args.seed=0,1,2,3 dataset.train_val_split=0.005,0.01,0.02,0.05,0.5,0.95

# ***********************
# Trajectory Length (appendix)
# ***********************
#
########
# D4RL
########
# Running
python research/mtm/train.py -m +exp_mtm=d4rl_cont wandb.project="11_mtm_d4rl_appendix_0" args.seed=0,1,2,3 dataset.env_name=hopper-medium-replay-v2,walker2d-medium-replay-v2,halfcheetah-medium-replay-v2 "args.mask_patterns=[AUTO_MASK]" args.traj_length=1,2,4,8,16,32

########
# Adroit
########
# Running
python research/mtm/train.py -m +exp_mtm=adroit_cont wandb.project="11_mtm_adroit_appendix_0" args.seed=0,1,2,3 dataset.env_name=pen,door dataset.d_name="medium_replay" "args.mask_patterns=[AUTO_MASK]"  args.traj_length=1,2,4,8,16,32
