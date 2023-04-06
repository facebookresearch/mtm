# # D4RL
# #######################
# # MTM adroit expert -> 0.01% with actions 0.95 with states only
# python research/mtm/train.py -m +exp_mtm=d4rl_halfcheetah_o3,d4rl_walker2d_o3,d4rl_hopper_o3 "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="d4rl_paper_3_2" args.seed=0,1,2,3,4 args.traj_length=4 dataset.train_val_split=0.01
# # MTM adroit expert -> 0.01% of the dataset
# python research/mtm/train.py -m +exp_mtm=d4rl_cont dataset.train_val_split=0.01 dataset.env_name=hopper-expert-v2,walker2d-expert-v2,halfcheetah-expert-v2 "args.mask_patterns=[ID,AUTO_MASK],[AUTO_MASK]" wandb.project="d4rl_paper_3_2" args.seed=0,1,2,3,4 args.traj_length=4
# # MLP adroit expert BC and RCBC -> 0.01% of the dataset
# python research/mtm/train_mlp.py -m +exp_mlp=d4rl_cont dataset.env_name=hopper-expert-v2,walker2d-expert-v2,halfcheetah-expert-v2 wandb.project="d4rl_paper_3_2" model_config.task="bc","rcbc" args.seed=0,1,2,3,4 args.traj_length=1 dataset.train_val_split=0.01,0.95

# MTM adroit expert -> 0.01% with actions 0.95 with states only
# MTM adroit expert -> 0.01% of the dataset
# MLP adroit expert BC and RCBC -> 0.01% of the dataset


# rerun for walker
# python research/mtm/train.py -m +exp_mtm=d4rl_walker2d_o3 "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="d4rl_paper_3_2" args.seed=0,1,2,3,4 args.traj_length=4 dataset.train_val_split=0.005 args.log_every=200
# python research/mtm/train.py -m +exp_mtm=d4rl_cont dataset.train_val_split=0.005 dataset.env_name=walker2d-expert-v2 "args.mask_patterns=[ID,AUTO_MASK],[AUTO_MASK]" wandb.project="d4rl_paper_3_2" args.seed=0,1,2,3,4 args.traj_length=4 args.log_every=200
# python research/mtm/train_mlp.py -m +exp_mlp=d4rl_cont dataset.env_name=walker2d-expert-v2 wandb.project="d4rl_paper_3_2" model_config.task="bc","rcbc" args.seed=0,1,2,3,4 args.traj_length=1 dataset.train_val_split=0.005 args.log_every=200
#



# # # Adroit
# # #######################
# # MTM adroit expert -> 0.01% with actions 0.95 with states only
# python research/mtm/train.py -m +exp_mtm=adroit_door_option3,adroit_pen_option3,adroit_hammer_option3,adroit_relocate_option3  "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="noisy_adroit_4_2" args.seed=0,1,2,3,4 args.traj_length=4
# # MTM adroit expert -> 0.01% of the dataset
# python research/mtm/train.py -m +exp_mtm=adroit_cont dataset.train_val_split=0.01 dataset.env_name="pen","hammer","relocate","door" "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="noisy_adroit_4_2" args.seed=0,1,2,3,4 args.traj_length=4
# # MLP adroit expert BC and RCBC -> 0.01% of the dataset
# python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=relocate,pen,hammer,door wandb.project="noisy_adroit_4_2" model_config.task="bc","rcbc" args.seed=0,1,2,3,4 args.traj_length=1 dataset.d_name="expert" dataset.train_val_split=0.01,0.95

# # data efficiency
# # - [x] Launch runs for heterogeneous dataset to make data efficiency plot
# #     - [x] 4 versions
# #     - [x] 1 - MLP BC
# #     - [x] 2- MLP RCBC
# #     - [x] 3 - MTM only 0.01
# #     - [x] 4 - MTM with 0.95 state only
# python research/mtm/train_mlp.py -m +exp_mlp=d4rl_cont dataset.env_name=hopper-expert-v2 wandb.project="data_eff" model_config.task="bc","rcbc" args.seed=0,1,2,3,4 args.traj_length=1 dataset.train_val_split=0.005,0.01,0.02,0.05,0.5,0.95
# python research/mtm/train.py -m +exp_mtm=d4rl_cont dataset.env_name=hopper-expert-v2 "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="data_eff" args.seed=0,1,2,3,4 args.traj_length=4 dataset.train_val_split=0.005,0.01,0.02,0.05,0.5,0.95
# python research/mtm/train.py -m +exp_mtm=d4rl_hopper_o3 "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="data_eff" args.seed=0,1,2,3,4 args.traj_length=4 dataset.train_val_split=0.005,0.01,0.02,0.05,0.5,0.95

#
# # trajectory length at low data regimes, note the traj len =4 is already handled in the previous section
# #- [x] MTM data efficiency vs trajectory length
# #     - [x] 0.01 percent data only hopper
# #     - [x] Change trajectory length 1, 2, 4, 8, 16, 32
# #     - [x] Look at RCBC performance
# python research/mtm/train.py -m +exp_mtm=d4rl_cont dataset.env_name=hopper-expert-v2 "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="data_eff" args.seed=0,1,2,3,4 args.traj_length=1,2,8,16,32 dataset.train_val_split=0.01
# python research/mtm/train.py -m +exp_mtm=d4rl_hopper_o3 "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="data_eff" args.seed=0,1,2,3,4 args.traj_length=1,2,8,16,32 dataset.train_val_split=0.01

# debug stuff for adroit
# python research/mtm/train.py -m +exp_mtm=adroit_door_option3  "args.mask_patterns=[AUTO_MASK],[ID,AUTO_MASK]" wandb.project="adroit_paper_4_2" args.seed=0,1,2,3,4 args.traj_length=4 args.tsp_ratio=1,8,100000000 args.log_every=101 +hydra.launcher.partition=devlab

# # data efficiency for MTM on adroit pen
# python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=pen wandb.project="data_eff_adroit" model_config.task="bc","rcbc" args.seed=0,1,2,3,4 args.traj_length=1 dataset.train_val_split=0.005,0.01,0.02,0.05,0.5,0.95
# python research/mtm/train.py -m +exp_mtm=adroit_cont  dataset.env_name=pen "args.mask_patterns=[ID,AUTO_MASK]" wand.project="data_eff_adroit" args.seed=0,1,2,3,4 args.traj_length=4 dataset.train_val_split=0.005,0.01,0.02,0.05,0.5,0.95
# python research/mtm/train.py -m +exp_mtm=adroit_pen_option3 "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="data_eff_adroit" args.seed=0,1,2,3,4 args.traj_length=4 dataset.train_val_split=0.005,0.01,0.02,0.05,0.5,0.95

# # # Adroit, hetero with medium replay
# # #######################
# # medium replay with actions, expert no actions
# python research/mtm/train.py -m +exp_mtm=adroit_door_option3,adroit_pen_option3,adroit_hammer_option3,adroit_relocate_option3  "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="adroit_paper_4_2_" args.seed=0,1,2,3,4 args.traj_length=4 dataset.train_val_split=0.95
# # medium replay
# python research/mtm/train.py -m +exp_mtm=adroit_cont dataset.env_name="pen","hammer","relocate","door" "args.mask_patterns=[ID,AUTO_MASK],[AUTO_MASK]" wandb.project="adroit_paper_4_2_" args.seed=0,1,2,3,4 args.traj_length=4  dataset.d_name="medium_replay"
# # MLP adroit expert BC and RCBC -> 0.01% of the dataset
# python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=relocate,pen,hammer,door wandb.project="adroit_paper_4_2_" model_config.task="bc","rcbc" args.seed=0,1,2,3,4 args.traj_length=1 dataset.d_name="medium_replay"

# # # Adroit relocate
# # #######################
# # MTM adroit expert -> 0.01% with actions 0.95 with states only
# python research/mtm/train.py -m +exp_mtm=adroit_relocate_option3  "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="noisy_adroit_4_2" args.seed=0,1,2,3,4 args.traj_length=4 args.log_every=102
# # MTM adroit expert -> 0.01% of the dataset
# python research/mtm/train.py -m +exp_mtm=adroit_cont dataset.train_val_split=0.05 dataset.env_name="relocate" "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="noisy_adroit_4_2" args.seed=0,1,2,3,4 args.traj_length=4 args.log_every=102
# # MLP adroit expert BC and RCBC -> 0.01% of the dataset
# python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=relocate wandb.project="noisy_adroit_4_2" model_config.task="bc","rcbc" args.seed=0,1,2,3,4 args.traj_length=1 dataset.d_name="expert" dataset.train_val_split=0.05,0.95 args.log_every=102
#
# # # # Adroit hammer
# # # #######################
# # MTM adroit expert -> 0.01% with actions 0.95 with states only
# # python research/mtm/train.py -m +exp_mtm=adroit_hammer_option3  "args.mask_patterns=[AUTO_MASK],[ID,AUTO_MASK],[ID,AUTO_MASK,AUTO_MASK,AUTO_MASK]" wandb.project="noisy_adroit_4_2" args.seed=0,1,2,3,4 args.traj_length=4  model_config.n_enc_layer=2,3  args.tsp_ratio=1,4 args.log_every=200
# # MTM adroit expert -> 0.01% of the dataset
# # python research/mtm/train.py -m +exp_mtm=adroit_cont dataset.train_val_split=0.01 dataset.env_name="hammer" "args.mask_patterns=[AUTO_MASK,ID,AUTO_MASK]" wandb.project="noisy_adroit_4_2" args.seed=0,1,2,3,4 args.traj_length=4 model_config.n_enc_layer=2,3
# # MLP adroit expert BC and RCBC -> 0.01% of the dataset
#
# # # # Adroit
# # # #######################
# # # MTM adroit expert -> 0.01% with actions 0.95 with states only
# python research/mtm/train.py -m +exp_mtm=adroit_hammer_option3 "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="noisy_adroit_4_2" args.seed=0,1,2,3,4 args.traj_length=4 args.log_every=300
# # # MTM adroit expert -> 0.01% of the dataset
# python research/mtm/train.py -m +exp_mtm=adroit_cont dataset.train_val_split=0.005 dataset.env_name="hammer" "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="noisy_adroit_4_2" args.seed=0,1,2,3,4 args.traj_length=4  args.log_every=300
# # # MLP adroit expert BC and RCBC -> 0.01% of the dataset
# python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=hammer wandb.project="noisy_adroit_4_2" model_config.task="bc","rcbc" args.seed=0,1,2,3,4 args.traj_length=1 dataset.d_name="expert" dataset.train_val_split=0.005,0.95  args.log_every=300


# # # Adroit
# # #######################
# # MTM adroit expert -> 0.01% with actions 0.95 with states only
# python research/mtm/train.py -m +exp_mtm=adroit_door_option3,adroit_pen_option3,adroit_hammer_option3,adroit_relocate_option3  "args.mask_patterns=[AUTO_MASK,ID,AUTO_MASK,AUTO_MASK]" wandb.project="noisy_adroit_hetero_final" args.seed=0,1,2,3,4,5,6,7,8,9 args.traj_length=4
# # MTM adroit expert -> 0.01% of the dataset
# python research/mtm/train.py -m +exp_mtm=adroit_cont dataset.train_val_split=0.01 dataset.env_name="pen","hammer","relocate","door" "args.mask_patterns=[AUTO_MASK,ID,AUTO_MASK,AUTO_MASK]" wandb.project="noisy_adroit_hetero_final" args.seed=0,1,2,3,4,5,6,7,8,9 args.traj_length=4
# # MLP adroit expert BC and RCBC -> 0.01% of the dataset
# python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=relocate,pen,hammer,door wandb.project="noisy_adroit_hetero_final" model_config.task="bc" args.seed=0,1,2,3,4,5,6,7,8,9 args.traj_length=1 dataset.d_name="expert" dataset.train_val_split=0.95
# python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=relocate,pen,hammer,door wandb.project="noisy_adroit_hetero_final" model_config.task="bc","rcbc" args.seed=0,1,2,3,4,5,6,7,8,9 args.traj_length=1 dataset.d_name="expert" dataset.train_val_split=0.01


# MTM adroit expert -> 0.01% with actions 0.95 with states only
python research/mtm/train.py -m +exp_mtm=adroit_pen_option3,adroit_hammer_option3,adroit_relocate_option3,adroit_door_option3  "args.mask_patterns=[AUTO_MASK,ID,AUTO_MASK]" wandb.project="adroit_hetero_1_24" args.seed=0,1,2,3,4 args.traj_length=4 args.log_every=200
# MTM adroit expert -> 0.01% of the dataset
python research/mtm/train.py -m +exp_mtm=adroit_cont dataset.train_val_split=0.01 dataset.env_name=relocate,pen,hammer,door "args.mask_patterns=[AUTO_MASK,ID,AUTO_MASK]" wandb.project="adroit_hetero_1_24" args.seed=0,1,2,3,4 args.traj_length=4  args.log_every=200
# MLP adroit expert BC and RCBC -> 0.01% of the dataset
python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=relocate,pen,hammer,door wandb.project="adroit_hetero_1_24" model_config.task="bc" args.seed=0,1,2,3,4 args.traj_length=1 dataset.d_name="expert" dataset.train_val_split=0.95  args.log_every=200
python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=relocate,pen,hammer,door wandb.project="adroit_hetero_1_24" model_config.task="bc","rcbc" args.seed=0,1,2,3,4 args.traj_length=1 dataset.d_name="expert" dataset.train_val_split=0.01  args.log_every=200


# ############################ Pen only
# # MTM adroit expert -> 0.01% with actions 0.95 with states only
# python research/mtm/train.py -m +exp_mtm=adroit_pen_option3 "args.mask_patterns=[AUTO_MASK,ID,AUTO_MASK,AUTO_MASK]" wandb.project="adroit_pen" args.seed=0,1,2,3,4,5,6,7,8,9 args.traj_length=4 args.log_every=500
# # MTM adroit expert -> 0.01% of the dataset
# python research/mtm/train.py -m +exp_mtm=adroit_cont dataset.train_val_split=0.01 dataset.env_name="pen" "args.mask_patterns=[AUTO_MASK,ID,AUTO_MASK,AUTO_MASK]" wandb.project="adroit_pen" args.seed=0,1,2,3,4,5,6,7,8,9 args.traj_length=4  args.log_every=500
# # MLP adroit expert BC and RCBC -> 0.01% of the dataset
# python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=pen wandb.project="adroit_pen" model_config.task="bc" args.seed=0,1,2,3,4,5,6,7,8,9 args.traj_length=1 dataset.d_name="expert" dataset.train_val_split=0.95  args.log_every=500
# python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=pen wandb.project="adroit_pen" model_config.task="bc","rcbc" args.seed=0,1,2,3,4,5,6,7,8,9 args.traj_length=1 dataset.d_name="expert" dataset.train_val_split=0.01  args.log_every=500

python research/mtm/train_mlp.py -m +exp_mlp=adroit_cont dataset.env_name=door wandb.project="data_eff_adroit" model_config.task="bc","rcbc" args.seed=0,1,2,3,4 args.traj_length=1 dataset.train_val_split=0.005,0.01,0.02,0.05,0.95
python research/mtm/train.py -m +exp_mtm=adroit_cont dataset.env_name=door "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="data_eff_adroit" args.seed=0,1,2,3,4 args.traj_length=4 dataset.train_val_split=0.005,0.01,0.02,0.05,0.95
python research/mtm/train.py -m +exp_mtm=adroit_door_option3 "args.mask_patterns=[ID,AUTO_MASK]" wandb.project="data_eff_adroit" args.seed=0,1,2,3,4 args.traj_length=4 dataset.train_val_split=0.005,0.01,0.02,0.05,0.95
