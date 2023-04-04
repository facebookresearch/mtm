import dataclasses
import functools
import os
import random
from typing import Dict, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import OmegaConf
from torch.utils.data.dataloader import DataLoader

from research.logger import stopwatch
from research.mtm.models.mtm_model import MaskedDP
from research.mtm.tokenizers.base import Tokenizer, TokenizerManager

os.environ["MUJOCO_GL"] = "egl"
from typing import Dict

import numpy as np
import torch
from omegaconf import OmegaConf


def eval_inverse_dynamics(
    model: MaskedDP,
    tokenizer_manager: TokenizerManager,
    dataset_loader: DataLoader,
    save_path: str,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate the inverse dynamics model.

    Args:
        prediction_fn: Function that takes in a state and outputs a predicted action.
        dataset_loader: loader
        results_folder: Folder to save results to.
    """
    masks = None
    diff_lb = []
    diff_abs = []
    diff_oracle = []

    for batch in dataset_loader:
        if masks is None:
            action_mask = torch.zeros(batch["actions"].shape[1], device=device)
            state_mask = torch.ones(batch["states"].shape[1], device=device)
            reward_mask = torch.zeros(batch["states"].shape[1], device=device)
            masks = {
                "states": state_mask,
                "actions": action_mask,
                "rewards": reward_mask,
            }

        trajectories = next(iter(batch))  # keep this batch for visualization
        trajectories = {k: v.to(device) for k, v in batch.items()}
        encoded_trajectories = tokenizer_manager.encode(trajectories)
        # import ipdb
        # ipdb.set_trace()

        decoded_gt_trajectories = tokenizer_manager.decode(encoded_trajectories)
        predictions = model(encoded_trajectories, masks)
        decoded_trajs = tokenizer_manager.decode(predictions)

        # compute action difference
        action_diff_lower_bound = (
            decoded_gt_trajectories["actions"] - decoded_trajs["actions"]
        )

        action_diff_absolute = trajectories["actions"] - decoded_trajs["actions"]

        oracle_diff = trajectories["actions"] - decoded_gt_trajectories["actions"]

        diff_lb.append(torch.mean(torch.abs(action_diff_lower_bound)).item())
        diff_abs.append(torch.mean(torch.abs(action_diff_absolute)).item())
        diff_oracle.append(torch.mean(torch.abs(oracle_diff)).item())
    return (
        torch.mean(torch.tensor(diff_lb)),
        torch.mean(torch.tensor(diff_abs)),
        torch.mean(torch.tensor(diff_oracle)),
    )


def eval_forward_dynamics(
    model: MaskedDP,
    tokenizer_manager: TokenizerManager,
    dataset_loader: DataLoader,
    save_path: str,
    device: torch.device,
    num_states: int,
) -> Tuple[float, float, float]:
    """Evaluate the inverse dynamics model.

    Args:
        prediction_fn: Function that takes in a state and outputs a predicted action.
        dataset_loader: loader
        results_folder: Folder to save results to.
    """
    masks = None
    diff_lb = []
    diff_abs = []
    diff_oracle = []

    for batch in dataset_loader:
        if masks is None:
            action_mask = torch.ones(batch["actions"].shape[1], device=device)
            state_mask = torch.zeros(batch["states"].shape[1], device=device)
            reward_mask = torch.zeros(batch["states"].shape[1], device=device)
            assert num_states >= 1
            state_mask[:num_states] = 1
            masks = {
                "states": state_mask,
                "actions": action_mask,
                "rewards": reward_mask,
            }

        trajectories = next(iter(batch))  # keep this batch for visualization
        trajectories = {k: v.to(device) for k, v in batch.items()}
        encoded_trajectories = tokenizer_manager.encode(trajectories)
        # import ipdb
        # ipdb.set_trace()

        decoded_gt_trajectories = tokenizer_manager.decode(encoded_trajectories)
        predictions = model(encoded_trajectories, masks)
        decoded_trajs = tokenizer_manager.decode(predictions)

        # compute action difference
        action_diff_lower_bound = (
            decoded_gt_trajectories["states"] - decoded_trajs["states"]
        )
        action_diff_absolute = trajectories["states"] - decoded_trajs["states"]
        oracle_diff = trajectories["states"] - decoded_gt_trajectories["states"]

        diff_lb.append(torch.mean(torch.abs(action_diff_lower_bound)).item())
        diff_abs.append(torch.mean(torch.abs(action_diff_absolute)).item())
        diff_oracle.append(torch.mean(torch.abs(oracle_diff)).item())
    return (
        torch.mean(torch.tensor(diff_lb)),
        torch.mean(torch.tensor(diff_abs)),
        torch.mean(torch.tensor(diff_oracle)),
    )


@dataclasses.dataclass
class Args:
    path: str = "/private/home/philippwu/mtm/outputs/mtm_mae/2022-12-09_11-13-11/4_+experiments=yoga_discrete,args.learning_rate=0.0003,args.mask_patterns=[FULL_RANDOM,RANDOM,GOAL],args.weight_decay=0.001"


# make lru cache for getting the dataset
@functools.lru_cache(maxsize=10)
def get_dataset(dataset, traj_length):
    return hydra.utils.call(dataset, seq_steps=traj_length)


def main(args: Args):
    path = args.path

    # find checkpoints in the directory
    steps = []
    names = []
    paths_ = os.listdir(path)
    for name in [os.path.join(path, n) for n in paths_ if "pt" in n]:
        step = os.path.basename(name).split("_")[-1].split(".")[0]
        steps.append(int(step))
        names.append(name)
    ckpt_path = names[np.argmax(steps)]

    hydra_cfg = OmegaConf.load(os.path.join(path, ".hydra/config.yaml"))
    cfg = hydra.utils.instantiate(hydra_cfg.args)

    train_dataset, val_dataset = get_dataset(
        hydra_cfg.dataset, cfg.model_config.traj_length
    )
    tokenizers: Dict[str, Tokenizer] = {
        k: hydra.utils.call(v, key=k, train_dataset=train_dataset)
        for k, v in hydra_cfg.tokenizers.items()
    }
    tokenizer_manager = TokenizerManager(tokenizers)
    discrete_map: Dict[str, bool] = {}
    for k, v in tokenizers.items():
        discrete_map[k] = v.discrete
    train_loader = DataLoader(
        train_dataset,
        # shuffle=True,
        pin_memory=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
    )

    def seed_init_fn(x):
        seed = 0 + x
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        return

    val_loader = DataLoader(
        val_dataset,
        # shuffle=False,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        worker_init_fn=seed_init_fn,
    )
    train_batch = next(iter(train_loader))
    tokenized = tokenizer_manager.encode(train_batch)
    data_shapes = {}
    for k, v in tokenized.items():
        data_shapes[k] = v.shape[-2:]

    model = MaskedDP(data_shapes, cfg.model_config)
    model.to(cfg.device)
    model.load_state_dict(torch.load(ckpt_path)["model"])
    model.eval()
    print("Model Loaded")

    return_data = {}

    action_lb, action_abs, action_oracle = eval_inverse_dynamics(
        model, tokenizer_manager, val_loader, args.path, cfg.device
    )
    print("Inverse Dynamics")
    print(f"Action LB: {action_lb}")
    print(f"Action Abs: {action_abs}")
    print(f"Action Oracle: {action_oracle}")
    return_data["Inverse Dynamics"] = (action_lb, action_abs, action_oracle)

    error_lb, error_abs, error_oracle = eval_forward_dynamics(
        model, tokenizer_manager, val_loader, args.path, cfg.device, 1
    )
    print("\nForward Dynamics 1 given state")
    print(f"State LB: {error_lb}")
    print(f"State Abs: {error_abs}")
    print(f"State Oracle: {error_oracle}")
    return_data["Forward Dynamics 1"] = (error_lb, error_abs, error_oracle)

    error_lb, error_abs, error_oracle = eval_forward_dynamics(
        model, tokenizer_manager, val_loader, args.path, cfg.device, 3
    )
    print("\nForward Dynamics 3 given state")
    print(f"State LB: {error_lb}")
    print(f"State Abs: {error_abs}")
    print(f"State Oracle: {error_oracle}")
    return_data["Forward Dynamics 3"] = (error_lb, error_abs, error_oracle)

    error_lb, error_abs, error_oracle = eval_forward_dynamics(
        model, tokenizer_manager, val_loader, args.path, cfg.device, 5
    )
    print("\nForward Dynamics 5 given state")
    print(f"State LB: {error_lb}")
    print(f"State Abs: {error_abs}")
    print(f"State Oracle: {error_oracle}")
    return_data["Forward Dynamics 5"] = (error_lb, error_abs, error_oracle)
    return return_data


if __name__ == "__main__":
    # main(dcargs.parse(Args))

    # path_folder = "/private/home/philippwu/mtm/outputs/mtm_mae/2022-12-09_11-13-11"
    # path_list = [os.path.join(path_folder, p) for p in os.listdir(path_folder)]
    # path_list = [p for p in path_list if os.path.isdir(p)]
    #
    # all_data = {}
    # categories = None
    # for path in path_list:
    #     data = None
    #     with stopwatch(path):
    #         try:
    #             data = main(Args(path=path))
    #         except Exception as e:
    #             print(e)
    #
    #     if data is not None:
    #         if categories is None:
    #             categories = list(data.keys())
    #         name = path.split("/")[-1]
    #         # get the string inbetween the brackets []
    #         name = name.split("[")[1].split("]")[0]
    #         all_data[name] = data
    #
    # # make a bar with all the data, one for each name for each category
    # save_location = os.path.join(path_folder, "eval.pdf")
    # with PdfPages(save_location) as pdf:
    #     for category in categories:
    #         fig, ax = plt.subplots()
    #         for idx, (name, data) in enumerate(all_data.items()):
    #             lb, abs, oracle = data[category]
    #             if idx == 0:
    #                 # ax.bar(name, lb, label="LB", color="blue")
    #                 ax.bar(name, abs, label="Abs", color="blue")
    #                 ax.bar(name, oracle, label="Oracle", color="green")
    #             else:
    #                 # ax.bar(name, lb, color="blue")
    #                 ax.bar(name, abs, color="blue")
    #                 ax.bar(name, oracle, color="green")
    #
    #         plt.xticks(rotation="vertical")
    #         ax.set_title(category)
    #         ax.legend()
    #         pdf.savefig(fig, bbox_inches="tight")
    #         plt.close()

    path_folder = "/private/home/philippwu/mtm/outputs/mtm_mae/2022-12-08_17-20-41"
    path_list = [os.path.join(path_folder, p) for p in os.listdir(path_folder)]
    path_list = [p for p in path_list if os.path.isdir(p)]

    path_list = [p for p in path_list if "0.0003" in p]
    path_list = [p for p in path_list if "0.0001" in p]
    # path_list = [p for p in path_list if "walker2d-expert-v2" in p]
    path_list = [p for p in path_list if "walker2d-medium-replay-v2" in p]

    all_data = {}
    categories = None

    names = []
    for path in path_list:
        data = None
        with stopwatch(path):
            try:
                data = main(Args(path=path))
            except Exception as e:
                print(e)

        if data is not None:
            if categories is None:
                categories = list(data.keys())
            name = path.split("/")[-1]
            # extract the characters after traj_length but before ,
            traj_length = name.split("traj_length=")[1].split(",")[0]
            # get the string inbetween the brackets []
            name = name.split("[")[1].split("]")[0]
            name = f"({traj_length}){name}"
            names.append(name)
            all_data[name] = data

    sorted_names = sorted(names)

    # make a bar with all the data, one for each name for each category
    save_location = os.path.join(path_folder, "eval.pdf")
    with PdfPages(save_location) as pdf:
        for category in categories:
            fig, ax = plt.subplots()
            for idx, name in enumerate(sorted_names):
                data = all_data[name]
                lb, abs, oracle = data[category]
                if idx == 0:
                    # ax.bar(name, lb, label="LB", color="blue")
                    ax.bar(name, abs, label="Abs", color="blue")
                    ax.bar(name, oracle, label="Oracle", color="green")
                else:
                    # ax.bar(name, lb, color="blue")
                    ax.bar(name, abs, color="blue")
                    ax.bar(name, oracle, color="green")

            plt.xticks(rotation="vertical")
            ax.set_title(category)
            ax.legend()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()
