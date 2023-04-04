# python train_offline.py -m agent=td3_mtm agent.end_to_end=true  seed=4,3,2,1,0 task=walker_stand,walker_run,walker_walk
# python train_offline.py -m agent=td3_mtm agent.end_to_end=false seed=4,3,2,1,0 task=walker_stand,walker_run,walker_walk
# python train_offline.py -m agent=td3                            seed=4,3,2,1,0 task=walker_stand,walker_run,walker_walk

python train_offline.py -m agent=td3_mtm agent.end_to_end=true,false  seed=5,6,7,8,9 task=walker_stand,walker_run,walker_walk
python train_offline.py -m agent=td3_mtm agent.end_to_end=false seed=5,6,7,8,9 task=walker_stand,walker_run,walker_walk
python train_offline.py -m agent=td3                            seed=5,6,7,8,9 task=walker_stand,walker_run,walker_walk
