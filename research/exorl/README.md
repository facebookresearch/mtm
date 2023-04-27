# ExORL

The code in this folder is modified from https://github.com/denisyarats/exorl.

See agent/td3_mtm.yaml and agent/td3_mtm.py for for details of how MTM state representations are used.

Follow the original instructions below for setting up additional ExORL dependencies and for downloading the dataset.

```
./download.sh walker proto
```

Since the original creation of ExORL, mujoco has seen many updates. To rerun all the actions and collect a physics consistent data, you may optionally use the update_data.py utility.
```
python update_data.py --num-workers 10 --env-name walker --expl-agent proto
```


# ExORL: Exploratory Data for Offline Reinforcement Learning

This is an original PyTorch implementation of the ExORL framework from

[Don't Change the Algorithm, Change the Data: Exploratory Data for Offline Reinforcement Learning](https://arxiv.org/abs/2201.13425) by

[Denis Yarats*](https://cs.nyu.edu/~dy1042/), [David Brandfonbrener*](https://davidbrandfonbrener.github.io/), [Hao Liu](https://www.haoliu.site/), [Misha Laskin](https://www.mishalaskin.com/), [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/), [Alessandro Lazaric](http://chercheurs.lille.inria.fr/~lazaric/Webpage/Home/Home.html), and [Lerrel Pinto](https://www.lerrelpinto.com).

*Equal contribution.

## Prerequisites

Install [MuJoCo](http://www.mujoco.org/) if it is not already the case:

* Download MuJoCo binaries [here](https://mujoco.org/download).
* Unzip the downloaded archive into `~/.mujoco/`.
* Append the MuJoCo subdirectory bin path into the env variable `LD_LIBRARY_PATH`.

Install the following libraries:
```sh
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 unzip
```

Install dependencies:
```sh
conda env create -f conda_env.yml
conda activate exorl
```

## Datasets
We provide exploratory datasets for 6 DeepMind Control Stuite domains
| Domain | Dataset name | Available task names |
|---|---|---|
| Cartpole | `cartpole` | `cartpole_balance`, `cartpole_balance_sparse`, `cartpole_swingup`, `cartpole_swingup_sparse` |
| Cheetah | `cheetah` | `cheetah_run`, `cheetah_run_backward` |
| Jaco Arm | `jaco` | `jaco_reach_top_left`, `jaco_reach_top_right`, `jaco_reach_bottom_left`, `jaco_reach_bottom_right` |
| Point Mass Maze | `point_mass_maze` | `point_mass_maze_reach_top_left`, `point_mass_maze_reach_top_right`, `point_mass_maze_reach_bottom_left`, `point_mass_maze_reach_bottom_right`  | 
| Quadruped | `quadruped` | `quadruped_walk`, `quadruped_run` |
| Walker | `walker` | `walker_stand`, `walker_walk`, `walker_run` |


For each domain we collected datasets by running 9 unsupervised RL algorithms from [URLB](https://github.com/rll-research/url_benchmark) for total of `10M` steps. Here is the list of algorithms
| Unsupervised RL method | Name | Paper |
|---|---|---|
| APS | `aps` |  [paper](http://proceedings.mlr.press/v139/liu21b.html)|
| APT(ICM) | `icm_apt` |  [paper](https://arxiv.org/abs/2103.04551)|
| DIAYN | `diayn` |[paper](https://arxiv.org/abs/1802.06070)|
| Disagreement | `disagreement` | [paper](https://arxiv.org/abs/1906.04161) |
| ICM | `icm` | [paper](https://arxiv.org/abs/1705.05363)|
| ProtoRL | `proto` | [paper](https://arxiv.org/abs/2102.11271)|
| Random | `random` |  N/A |
| RND | `rnd` |  [paper](https://arxiv.org/abs/1810.12894) |
| SMM | `smm` |  [paper](https://arxiv.org/abs/1906.05274) |

You can download a dataset by running `./download.sh <DOMAIN> <ALGO>`, for example to download ProtoRL dataset for Walker, run
```sh
./download.sh walker proto
```
The script will download the dataset from S3 and store it under `datasets/walker/proto/`, where you can find episodes (under `buffer`) and episode videos (under `video`).

## Offline RL training
We also provide implementation of 5 offline RL algorithms for evaluating the datasets
| Offline RL method | Name | Paper |
|---|---|---|
| Behavior Cloning | `bc` |  [paper](https://proceedings.neurips.cc/paper/1988/file/812b4ba287f5ee0bc9d43bbf5bbe87fb-Paper.pdf)|
| CQL | `cql` |  [paper](https://arxiv.org/pdf/2006.04779.pdf)|
| CRR | `crr` |[paper](https://arxiv.org/pdf/2006.15134.pdf)|
| TD3+BC | `td3_bc` | [paper](https://arxiv.org/pdf/2106.06860.pdf) |
| TD3 | `td3` | [paper](https://arxiv.org/pdf/1802.09477.pdf)|

After downloading required datasets, you can evaluate it using offline RL methon for a specific task. For example, to evaluate a dataset collected by ProtoRL on Walker for the waling task using TD3+BC you can run
```sh
python train_offline.py agent=td3_bc expl_agent=proto task=walker_walk
```
Logs are stored in the `output` folder. To launch tensorboard run:
```sh
tensorboard --logdir output
```

## Citation

If you use this repo in your research, please consider citing the paper as follows:
```
@article{yarats2022exorl,
  title={Don't Change the Algorithm, Change the Data: Exploratory Data for Offline Reinforcement Learning},
  author={Denis Yarats, David Brandfonbrener, Hao Liu, Michael Laskin, Pieter Abbeel, Alessandro Lazaric, Lerrel Pinto},
  journal={arXiv preprint arXiv:2201.13425},
  year={2022}
}
```


## License
The majority of ExORL is licensed under the MIT license, however portions of the project are available under separate license terms: DeepMind is licensed under the Apache 2.0 license.
