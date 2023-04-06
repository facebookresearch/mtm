"""
Main script for training a policy given a dataset.
"""
import os
import wandb
import pprint
import random
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Sequence, Tuple

import hydra
import numpy as np
import torch
import torch.distributed
import torch.multiprocessing
import torch.nn.functional as F
import torch.nn.parallel
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from research.logger import WandBLogger, WandBLoggerConfig, logger, stopwatch
from research.mtm.datasets.base import DatasetProtocol
from research.mtm.distributed_utils import DistributedParams, get_distributed_params
from research.mtm.masks import (
    MaskType,
    create_bc_mask,
    create_forward_dynamics_mask,
    create_full_random_masks,
    create_goal_n_reaching_masks,
    create_goal_reaching_masks,
    create_inverse_dynamics_mask,
    create_random_autoregressize_mask,
    create_random_bc_masks,
    create_random_mask,
    create_random_masks,
    create_rcbc_mask,
    maybe_add_rew_to_mask,
)
from research.mtm.models.mtm_model import MaskedDP, make_plots_with_masks
from research.mtm.tokenizers.base import Tokenizer, TokenizerManager
from research.mtm.tokenizers.continuous import ContinuousTokenizer
from research.mtm.utils import (
    get_cfg_hash,
    get_ckpt_path_from_folder,
    get_git_dirty,
    get_git_hash,
    set_seed_everywhere,
)

dir_path = os.path.dirname(os.path.realpath(__file__))


def eval_fd(
    model: MaskedDP,
    env,
    eval_batch,
    tokenizer_manager,
    ratio: int = 1,
) -> Dict[str, Any]:
    """Evaluate the model on the forward dynamics task.
    Args:
        env (gym.Env): env
        eval_batch (Dict[str, torch.Tensor]): eval_batch
        tokenizer_manager (TokenizerManager): tokenizer_manager
    """
    seq_len = eval_batch["actions"].shape[1]
    device = eval_batch["states"].device
    assert seq_len >= 2, "Forward dynamics eval only works for seq_len=2"

    # Given initial state and all actions. Predict future states.
    obs_mask1 = torch.ones(seq_len, device=device)
    obs_mask1[-1] = 0
    actions_mask1 = torch.zeros(seq_len, device=device)
    actions_mask1[-2] = 1
    returns_mask = torch.zeros(seq_len, device=device)
    masks = {
        "states": obs_mask1,
        "actions": actions_mask1,
        "returns": returns_mask,
    }

    predictions = model.mask_git_forward(
        tokenizer_manager.encode(eval_batch),
        masks,
        ratio=ratio,
    )
    predicted_next_state = tokenizer_manager.decode(predictions)["states"]

    states = eval_batch["states"]
    next_state = states[:, -1]
    state_error = (next_state - predicted_next_state[:, -1, :]) ** 2
    eval_dict = {}
    eval_dict[f"eval/fd_state_error_r={ratio}"] = torch.mean(state_error).item()
    return eval_dict


def eval_id(
    model: MaskedDP, env, eval_batch, tokenizer_manager, ratio: int = 1
) -> Dict[str, Any]:
    """Evaluate the model on the inverse dynamics task.
    Args:
        env (gym.Env): env
        eval_batch (Dict[str, torch.Tensor]): eval_batch
        tokenizer_manager (TokenizerManager): tokenizer_manager
    """
    seq_len = eval_batch["actions"].shape[1]
    B, T, S = eval_batch["states"].shape
    device = eval_batch["states"].device
    assert seq_len >= 2, "Forward dynamics eval only works for seq_len=2"

    # Given all states. Predict second to last action.
    obs_mask1 = torch.ones(seq_len, device=device)
    actions_mask1 = torch.zeros(seq_len, device=device)
    returns_mask = torch.zeros(seq_len, device=device)
    masks = {
        "states": obs_mask1,
        "actions": actions_mask1,
        "returns": returns_mask,
    }

    predictions = model.mask_git_forward(
        tokenizer_manager.encode(eval_batch), masks, ratio=ratio
    )
    predicted_actions = tokenizer_manager.decode(predictions)["actions"]
    predicted_action = predicted_actions[:, -2, :]

    state_error = []
    gt_state_error = []
    action_error = []

    states = eval_batch["states"]
    actions = eval_batch["actions"]
    actions = eval_batch["actions"][:, -2, :]

    action_error = ((predicted_action - actions) ** 2).mean()
    eval_dict = {}
    eval_dict[f"eval/id_action_error_r={ratio}"] = torch.mean(
        torch.tensor(action_error)
    ).item()
    return eval_dict

    for i in range(B):
        # set state to be the second to last state
        env.reset()
        phys_state = np.zeros(S + 2)
        phys_state[2:] = states[i, T - 2].detach().cpu().numpy()
        env.sim.set_state_from_flattened(phys_state.copy())
        env.sim.forward()
        # get the action from the model
        action = predicted_action[i].detach().cpu().numpy()
        action = np.clip(action, -1, 1)

        # get the ground truth action
        gt_action = actions[i, T - 2].detach().cpu().numpy()
        # get the next state
        next_state = states[i, T - 1].detach().cpu().numpy()
        # get the next state from the model
        next_state_model = env.step(action)[0]

        # reset and test groud truth action
        env.reset()
        env.sim.set_state_from_flattened(phys_state.copy())
        env.sim.forward()
        next_state_gt = env.step(gt_action)[0]
        qpos_size = env.sim.data.qpos.shape[0]

        # compute action error
        action_error.append((action - gt_action) ** 2)
        # compute state error
        state_error.append((next_state[:qpos_size] - next_state_model[:qpos_size]) ** 2)
        gt_error = (next_state[:qpos_size] - next_state_gt[:qpos_size]) ** 2

        # if np.sum(gt_error) > 1e-7:
        #     print(gt_error)
        #     import ipdb; ipdb.set_trace();
        #     print("minor")

        gt_state_error.append(gt_error)

    eval_dict = {}
    eval_dict[f"eval/id_state_error_r={ratio}"] = torch.mean(
        torch.tensor(state_error)
    ).item()
    eval_dict[f"eval/id_action_error_r={ratio}"] = torch.mean(
        torch.tensor(action_error)
    ).item()
    eval_dict[f"eval/id_gt_state_error_r={ratio}"] = torch.mean(
        torch.tensor(gt_state_error)
    ).item()
    return eval_dict


def eval_full_id(
    model: MaskedDP, env, eval_batch, tokenizer_manager, ratio: int = 1
) -> Dict[str, Any]:
    """Evaluate the model on the inverse dynamics task.
    Args:
        env (gym.Env): env
        eval_batch (Dict[str, torch.Tensor]): eval_batch
        tokenizer_manager (TokenizerManager): tokenizer_manager
    """
    seq_len = eval_batch["actions"].shape[1]
    B, T, S = eval_batch["states"].shape
    device = eval_batch["states"].device
    assert seq_len >= 2, "Forward dynamics eval only works for seq_len=2"

    # Given all states. Predict ALL actions.
    obs_mask1 = torch.ones(seq_len, device=device)
    actions_mask1 = torch.zeros(seq_len, device=device)
    returns_mask = torch.zeros(seq_len, device=device)
    masks = {
        "states": obs_mask1,
        "actions": actions_mask1,
        "returns": returns_mask,
    }

    predictions = model.mask_git_forward(
        tokenizer_manager.encode(eval_batch), masks, ratio=ratio
    )
    predicted_actions = tokenizer_manager.decode(predictions)["actions"]

    actions = eval_batch["actions"]

    action_error = ((predicted_actions - actions) ** 2).mean()

    eval_dict = {}
    eval_dict[f"eval/full_id_action_error_r={ratio}"] = torch.mean(
        torch.tensor(action_error)
    ).item()
    return eval_dict


def create_eval_logs_states_actions_images(
    predict_fn: Callable,
    trajectories: Dict[str, torch.Tensor],
    tokenizer_manager: TokenizerManager,
) -> Dict[str, Any]:
    eval_logs = {}
    assert "states" in trajectories
    assert "actions" in trajectories
    device = trajectories["states"].device
    seq_len = trajectories["states"].shape[1]

    # Given initial state and all actions. Predict future states.
    obs_mask1 = np.ones(seq_len)
    obs_mask1[1:] = 0
    actions_mask1 = np.ones(seq_len)

    obs_mask2 = np.ones(seq_len)
    obs_mask2[1:-1] = 0
    actions_mask2 = np.zeros(seq_len)

    obs_mask3 = np.ones(seq_len)
    obs_mask3[1:-1] = 0
    obs_mask3[::16] = 1
    actions_mask3 = np.zeros(seq_len)

    obs_mask4 = np.ones(seq_len)
    actions_mask4 = np.zeros(seq_len)

    rnd = np.random.RandomState(0)
    obs_mask5 = create_random_mask(seq_len, 0.15, device, rnd).detach().cpu().numpy()
    actions_mask5 = (
        create_random_mask(seq_len, 0.15, device, rnd).detach().cpu().numpy()
    )

    obs_use_mask_list = [
        obs_mask1,
        obs_mask2,
        obs_mask3,
        obs_mask4,
        obs_mask5,
    ]
    actions_use_mask_list = [
        actions_mask1,
        actions_mask2,
        actions_mask3,
        actions_mask4,
        actions_mask5,
    ]
    masks_list = []
    for obs_mask, actions_mask in zip(obs_use_mask_list, actions_use_mask_list):
        masks_list.append(
            {
                "states": torch.from_numpy(np.zeros_like(obs_mask)).to(device),
                "images": torch.from_numpy(obs_mask).to(device),
                "actions": torch.from_numpy(actions_mask).to(device),
            }
        )

    r1 = create_random_mask(seq_len, 0.15, device, rnd).detach().cpu().numpy()
    r2 = create_random_mask(seq_len, 0.15, device, rnd).detach().cpu().numpy()
    r3 = create_random_mask(seq_len, 0.15, device, rnd).detach().cpu().numpy()
    masks_list.append(
        {
            "states": torch.from_numpy(r1).to(device),
            "images": torch.from_numpy(r2).to(device),
            "actions": torch.from_numpy(r3).to(device),
        }
    )

    prefixs = ["f_dynamics", "goal", "goal_32", "inv_dynamics", "random", "random_all"]
    return make_plots_with_masks(
        predict_fn,
        trajectories,
        tokenizer_manager,
        masks_list,
        prefixs,
        max_n_plots=1,
    )


def create_eval_logs_actions_images(
    predict_fn: Callable,
    trajectories: Dict[str, torch.Tensor],
    tokenizer_manager: TokenizerManager,
    rewards: bool = False,
) -> Dict[str, Any]:
    eval_logs = {}
    assert "images" in trajectories
    assert "actions" in trajectories
    device = trajectories["images"].device
    seq_len = trajectories["images"].shape[1]

    # Given initial state and all actions. Predict future states.
    obs_mask1 = np.ones(seq_len)
    obs_mask1[1:] = 0
    actions_mask1 = np.ones(seq_len)

    obs_mask2 = np.ones(seq_len)
    obs_mask2[1:-1] = 0
    actions_mask2 = np.zeros(seq_len)

    obs_mask3 = np.ones(seq_len)
    obs_mask3[1:-1] = 0
    obs_mask3[::16] = 1
    actions_mask3 = np.zeros(seq_len)

    obs_mask4 = np.ones(seq_len)
    actions_mask4 = np.zeros(seq_len)

    rnd = np.random.RandomState(0)
    obs_mask5 = create_random_mask(seq_len, 0.15, device, rnd).detach().cpu().numpy()
    actions_mask5 = (
        create_random_mask(seq_len, 0.15, device, rnd).detach().cpu().numpy()
    )

    obs_use_mask_list = [
        obs_mask1,
        obs_mask2,
        obs_mask3,
        obs_mask4,
        obs_mask5,
    ]
    actions_use_mask_list = [
        actions_mask1,
        actions_mask2,
        actions_mask3,
        actions_mask4,
        actions_mask5,
    ]
    masks_list = []
    for obs_mask, actions_mask in zip(obs_use_mask_list, actions_use_mask_list):
        masks_list.append(
            {
                "images": torch.from_numpy(obs_mask).to(device),
                "actions": torch.from_numpy(actions_mask).to(device),
            }
        )
        if rewards:
            masks_list[-1]["rewards"] = masks_list[-1]["images"].clone()

    prefixs = ["f_dynamics", "goal", "goal_32", "inv_dynamics", "random"]
    return make_plots_with_masks(
        predict_fn,
        trajectories,
        tokenizer_manager,
        masks_list,
        prefixs,
        max_n_plots=2,
    )


def create_eval_logs_states_actions(
    predict_fn: Callable,
    trajectories: Dict[str, torch.Tensor],
    tokenizer_manager: TokenizerManager,
    rewards: bool = False,
) -> Dict[str, Any]:
    eval_logs = {}
    assert "states" in trajectories
    assert "actions" in trajectories
    device = trajectories["states"].device
    seq_len = trajectories["states"].shape[1]

    # Given initial state and all actions. Predict future states.
    obs_mask1 = np.ones(seq_len)
    obs_mask1[1:] = 0
    actions_mask1 = np.ones(seq_len)

    obs_mask2 = np.ones(seq_len)
    obs_mask2[1:-1] = 0
    actions_mask2 = np.zeros(seq_len)

    obs_mask3 = np.ones(seq_len)
    obs_mask3[1:-1] = 0
    obs_mask3[::16] = 1
    actions_mask3 = np.zeros(seq_len)

    obs_mask4 = np.ones(seq_len)
    actions_mask4 = np.zeros(seq_len)

    rnd = np.random.RandomState(0)
    obs_mask5 = create_random_mask(seq_len, 0.15, device, rnd).detach().cpu().numpy()
    actions_mask5 = (
        create_random_mask(seq_len, 0.15, device, rnd).detach().cpu().numpy()
    )

    obs_use_mask_list = [
        obs_mask1,
        obs_mask2,
        obs_mask3,
        obs_mask4,
        obs_mask5,
    ]
    actions_use_mask_list = [
        actions_mask1,
        actions_mask2,
        actions_mask3,
        actions_mask4,
        actions_mask5,
    ]
    masks_list = []
    for obs_mask, actions_mask in zip(obs_use_mask_list, actions_use_mask_list):
        masks_list.append(
            {
                "states": torch.from_numpy(obs_mask).to(device),
                "actions": torch.from_numpy(actions_mask).to(device),
            }
        )
        if rewards:
            masks_list[-1]["rewards"] = masks_list[-1]["states"].clone()
            masks_list[-1]["returns"] = masks_list[-1]["states"].clone()

    prefixs = ["f_dynamics", "goal", "goal_32", "inv_dynamics", "random"]
    return make_plots_with_masks(
        predict_fn,
        trajectories,
        tokenizer_manager,
        masks_list,
        prefixs,
        max_n_plots=2,
    )


def create_eval_logs_states(
    predict_fn: Callable,
    trajectories: Dict[str, torch.Tensor],
    tokenizer_manager: TokenizerManager,
) -> Dict[str, Any]:
    assert "states" in trajectories
    eval_logs = {}
    device = trajectories["states"].device
    seq_len = trajectories["states"].shape[1]

    # Given initial state and all actions. Predict future states.
    obs_mask1 = np.ones(seq_len)
    obs_mask1[1:] = 0

    obs_mask2 = np.ones(seq_len)
    obs_mask2[1:-1] = 0

    obs_mask3 = np.ones(seq_len)
    obs_mask3[seq_len // 2 + 3 :] = 0

    obs_use_mask_list = [
        obs_mask1,
        obs_mask2,
        obs_mask3,
    ]

    masks_list = []
    for obs_mask in obs_use_mask_list:
        masks_list.append(
            {
                "states": torch.from_numpy(obs_mask).to(device),
            }
        )

    prefixs = ["[6:]", "[1:-1]", "[half+1:]"]
    return make_plots_with_masks(
        predict_fn,
        trajectories,
        tokenizer_manager,
        masks_list,
        prefixs,
    )


@dataclass
class RunConfig:
    seed: int = 0
    """RNG seed."""

    batch_size: int = 64
    """Batch size used during training."""

    n_workers: int = 8
    """Number of workers for loading data."""

    log_every: int = 100
    """Print training loss every N steps."""

    print_every: int = 1000
    """Print training loss every N steps."""

    eval_every: int = 5000
    """Evaluate model every N steps."""

    save_every: int = 5000
    """Evaluate model every N steps."""

    device: str = "cuda"
    """Device to use for training."""

    mask_ratios: Sequence[float] = (0.15, 0.35, 0.55, 0.75, 0.85, 0.95)

    mask_patterns: Sequence[str] = ("RANDOM",)
    """Indices of masks to use for evaluation."""

    warmup_steps: int = 1_000
    """Number of warmup steps for learning rate scheduler."""

    num_train_steps: int = 5_000_000
    """Number of training steps."""

    learning_rate: float = 1e-3
    """Learning rate."""

    weight_decay: float = 1e-5
    """Weight decay."""

    traj_length: int = 1
    """Trajectory length."""

    mode_weights: Tuple[int, int, int] = (0.2, 0.1, 0.7)
    """State action return."""

    tsp_ratio: int = 1
    """Train steps per state only train steps ratio.

    1 means train state only every step.
    2 means train state only every other step, etc.
    """


@torch.inference_mode()
def evaluate(
    model: MaskedDP,
    tokenizer_manager: TokenizerManager,
    discrete_map: Dict[str, bool],
    val_batch: Dict[str, torch.Tensor],
    vis_batch: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    encoded_batch = tokenizer_manager.encode(val_batch)
    predicted_trajectories = model(encoded_batch, masks)
    model_without_ddp = model.module if hasattr(model, "module") else model
    (
        loss,
        losses_dict,
        masked_losses,
        masked_c_losses,
    ) = MaskedDP.forward_loss(
        encoded_batch,
        predicted_trajectories,
        masks,
        discrete_map,
        norm=model_without_ddp.norm,
        reduce_use_sum=model_without_ddp.config.reduce_use_sum,
        loss_keys=model_without_ddp.config.loss_keys,
    )

    log_dict = {"val/val_loss": loss.item()}
    for k, v in losses_dict.items():
        log_dict[f"val/full_loss_{k}"] = v.item()
    for k, v in masked_losses.items():
        log_dict[f"val/masked_loss_{k}"] = v.item()
    for k, v in masked_c_losses.items():
        log_dict[f"val/masked_c_loss_{k}"] = v.item()

    mse_loss = 0
    predictions = tokenizer_manager.decode(predicted_trajectories)
    for k, v in predictions.items():
        _mse = F.mse_loss(v.to(torch.float32), val_batch[k].to(torch.float32)).item()
        log_dict[f"val/mse_{k}"] = _mse
        mse_loss += _mse
    log_dict["val/mse_sum"] = mse_loss

    if "states" in val_batch and "actions" in val_batch and "images" in val_batch:
        log_images = create_eval_logs_states_actions_images(
            model, vis_batch, tokenizer_manager
        )
    elif "states" in val_batch and "actions" in val_batch and "rewards" in val_batch:
        log_images = create_eval_logs_states_actions(
            model, vis_batch, tokenizer_manager, rewards=True
        )
    elif "states" in val_batch and "actions" in val_batch:
        log_images = create_eval_logs_states_actions(
            model, vis_batch, tokenizer_manager
        )
    elif "states" in val_batch:
        log_images = create_eval_logs_states(model, vis_batch, tokenizer_manager)
    elif "images" in val_batch:
        log_images = create_eval_logs_actions_images(
            model, vis_batch, tokenizer_manager
        )
    else:
        raise NotImplementedError
    log_dict.update(log_images)
    return log_dict


def train_one_batch(
    model: MaskedDP,
    optimizer: torch.optim.Optimizer,
    scheduler: Callable,
    tokenizer_manager: TokenizerManager,
    discrete_map: Dict[str, bool],
    batch: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    loss_keys: Sequence[str] = None,
) -> Dict[str, Any]:
    encoded_batch = tokenizer_manager.encode(batch)

    # train the model
    predicted_trajectories = model(encoded_batch, masks)

    # compute the loss
    model_without_ddp = model.module if hasattr(model, "module") else model
    if loss_keys is None:
        loss_keys = model_without_ddp.config.loss_keys

    loss, losses_dict, masked_losses, masked_c_losses = MaskedDP.forward_loss(
        encoded_batch,
        predicted_trajectories,
        masks,
        discrete_map,
        norm=model_without_ddp.norm,
        reduce_use_sum=model_without_ddp.config.reduce_use_sum,
        loss_keys=loss_keys,
    )
    # create a dictionary to log all of the losses
    log_dict = {"train/train_loss": loss.item()}
    log_dict["train/lr"] = scheduler.get_last_lr()[0]
    for k, v in losses_dict.items():
        log_dict[f"train/full_loss_{k}"] = v.item()
    for k, v in masked_losses.items():
        log_dict[f"train/masked_loss_{k}"] = v.item()
    for k, v in masked_c_losses.items():
        log_dict[f"train/masked_c_loss_{k}"] = v.item()

    # backprop
    model.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        mse_loss = 0
        predictions = tokenizer_manager.decode(predicted_trajectories)
        for k, v in predictions.items():
            _mse = F.mse_loss(v.to(torch.float32), batch[k].to(torch.float32)).item()
            log_dict[f"train/mse_{k}"] = _mse
            mse_loss += _mse
        log_dict["train/mse_sum"] = mse_loss
    return log_dict


def main(hydra_cfg):
    _main(hydra_cfg)
    # p = Profiler()
    # with p:
    #     _main(hydra_cfg)
    # p.print()


def _main(hydra_cfg):
    cfg: RunConfig = hydra.utils.instantiate(hydra_cfg.args)
    dp: DistributedParams = get_distributed_params()

    torch.cuda.set_device(dp.local_rank)
    distributed = dp.world_size > 1
    if distributed:
        logger.info(
            f"Initializing rank {dp.rank} (local rank {dp.local_rank}) in total world size {dp.world_size} (local world size {dp.local_world_size}) with master addr:port {dp.master_addr}:{dp.master_port}"
        )
        torch.distributed.init_process_group(
            backend="nccl", rank=dp.rank, world_size=dp.world_size
        )

    set_seed_everywhere(cfg.seed)
    pprint.pp(cfg)

    logger.info(f"Working directory: {os.getcwd()}")

    with open("config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(hydra_cfg))

    train_dataset: DatasetProtocol
    val_dataset: DatasetProtocol

    train_dataset, val_dataset = hydra.utils.call(
        hydra_cfg.dataset, seq_steps=cfg.traj_length
    )
    logger.info(f"Train set size = {len(train_dataset)}")
    logger.info(f"Validation set size = {len(val_dataset)}")

    if hydra_cfg.state_only_dataset is not None:
        state_only_train_dataset, state_only_val_dataset = hydra.utils.call(
            hydra_cfg.state_only_dataset,
            seq_steps=cfg.traj_length,
        )
        logger.info(f"State Only Train set size = {len(state_only_train_dataset)}")
        logger.info(f"State Only Validation set size = {len(state_only_val_dataset)}")

    if "tokenizers" in hydra_cfg:
        tokenizers: Dict[str, Tokenizer] = {
            k: hydra.utils.call(v, key=k, train_dataset=train_dataset)
            for k, v in hydra_cfg.tokenizers.items()
        }
    else:
        tokenizers: Dict[str, Tokenizer] = {
            k: ContinuousTokenizer.create(k, train_dataset)
            for k in train_dataset[0].keys()
        }
    tokenizer_manager = TokenizerManager(tokenizers).to(cfg.device)

    discrete_map: Dict[str, bool] = {}
    for k, v in tokenizers.items():
        discrete_map[k] = v.discrete
    logger.info(f"Tokenizers: {tokenizers}")

    if distributed:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=dp.world_size, rank=dp.rank, shuffle=True
        )
        val_sampler = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=dp.world_size, rank=dp.rank, shuffle=False
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    with stopwatch("data loader"):
        train_loader = DataLoader(
            train_dataset,
            # shuffle=True,
            pin_memory=True,
            batch_size=cfg.batch_size,
            num_workers=cfg.n_workers,
            sampler=train_sampler,
        )
        val_loader = DataLoader(
            val_dataset,
            # shuffle=False,
            batch_size=cfg.batch_size,
            num_workers=1,
            sampler=val_sampler,
        )

    if hydra_cfg.state_only_dataset is not None:
        if distributed:
            state_only_train_sampler = torch.utils.data.DistributedSampler(
                state_only_train_dataset,
                num_replicas=dp.world_size,
                rank=dp.rank,
                shuffle=True,
            )
            state_only_val_sampler = torch.utils.data.DistributedSampler(
                state_only_val_dataset,
                num_replicas=dp.world_size,
                rank=dp.rank,
                shuffle=False,
            )
        else:
            state_only_train_sampler = torch.utils.data.RandomSampler(
                state_only_train_dataset
            )
            state_only_val_sampler = torch.utils.data.SequentialSampler(
                state_only_val_dataset
            )

        with stopwatch("state only loader"):
            state_only_train_loader = DataLoader(
                state_only_train_dataset,
                # shuffle=True,
                pin_memory=True,
                batch_size=cfg.batch_size,
                num_workers=cfg.n_workers,
                sampler=state_only_train_sampler,
            )

            state_only_val_loader = DataLoader(
                state_only_val_dataset,
                # shuffle=False,
                batch_size=cfg.batch_size,
                num_workers=1,
                sampler=state_only_val_sampler,
            )
        state_only_tokenizer_manager = TokenizerManager(
            {
                "states": tokenizers["states"],
                "returns": tokenizers["returns"],
                "actions": tokenizers["actions"],
            }
        ).to(cfg.device)
        state_only_iter = iter(state_only_train_loader)

    train_batch = next(iter(train_loader))
    tokenized = tokenizer_manager.encode(train_batch)
    data_shapes = {}
    for k, v in tokenized.items():
        data_shapes[k] = v.shape[-2:]
    logger.info(f"Data shapes: {data_shapes}")

    # create the model
    model_config = hydra.utils.instantiate(hydra_cfg.model_config)
    model = model_config.create(data_shapes, cfg.traj_length)
    model.to(cfg.device)
    model.train()
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dp.local_rank], output_device=dp.local_rank
        )

    optimizer = MaskedDP.configure_optimizers(
        model,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),  # following BERT
    )

    def _schedule(step):
        # warmp for 1000 steps
        if step < cfg.warmup_steps:
            return step / cfg.warmup_steps

        # then cosine decay
        assert cfg.num_train_steps > cfg.warmup_steps
        step = step - cfg.warmup_steps
        return 0.5 * (
            1 + np.cos(step / (cfg.num_train_steps - cfg.warmup_steps) * np.pi)
        )

    scheduler = LambdaLR(optimizer, lr_lambda=_schedule)

    # create a wandb logger and log params of interest
    wandb_cfg_log_dict = OmegaConf.to_container(hydra_cfg)
    wandb_cfg_log_dict["*discrete_map"] = discrete_map
    wandb_cfg_log_dict["*path"] = str(os.getcwd())
    wandb_cfg_log_dict["*mp"] = cfg.mask_patterns
    wandb_cfg_log_dict["*git_hash"] = get_git_hash()
    wandb_cfg_log_dict["*git_dirty"] = get_git_dirty()
    wandb_cfg_log_dict["*hydra_cfg_hash"] = get_cfg_hash(hydra_cfg)
    wandb_cfg_log_dict["*num_parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    wandb_cfg_log = WandBLoggerConfig(
        experiment_id=f"{dp.job_id}-{dp.rank}",
        project=hydra_cfg.wandb.project,
        entity=hydra_cfg.wandb.entity or None,
        resume=hydra_cfg.wandb.resume,
        group=dp.job_id,
    )

    if wandb_cfg_log.resume:
        exp_id = wandb_cfg_log_dict["*hydra_cfg_hash"]
        wandb_cfg_log = replace(
            wandb_cfg_log,
            experiment_id=exp_id,
        )
    # wandb_logger = WandBLogger(wandb_cfg_log, wandb_cfg_log_dict, enable=False)
    wandb_logger = WandBLogger(wandb_cfg_log, wandb_cfg_log_dict)

    step = 0
    eval_max = defaultdict(lambda: -np.inf)
    if wandb_logger.enable and wandb.run.resumed:
        # find checkpoints in the directory
        logger.info("Trying to resume ...")
        ckpt_path = get_ckpt_path_from_folder(
            os.getcwd()
        )  # hydra/submit-it will resume jobs from the same folder
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=cfg.device)
            logger.info(f"Resuming from checkpoint: {ckpt_path}")
            step = ckpt["step"]
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            eval_max = ckpt["eval_max"]  # keep track of the max even after preempt
        else:
            logger.info(f"No checkpoints found, starting from scratch.")
    logger.info(f"starting from step={step}")

    # train the model
    vis_batch = next(iter(val_loader))  # keep this batch for visualization
    vis_batch = {k: v.to(cfg.device) for k, v in vis_batch.items()}

    has_rew = "rewards" in vis_batch
    has_ret = "returns" in vis_batch
    has_img = "images" in vis_batch
    mask_functions_map = {
        MaskType.RANDOM: lambda: create_random_masks(
            data_shapes, cfg.mask_ratios, cfg.traj_length, cfg.device
        ),
        MaskType.FULL_RANDOM: lambda: create_full_random_masks(
            data_shapes, cfg.mask_ratios, cfg.traj_length, cfg.device
        ),
        MaskType.AUTO_MASK: lambda: create_random_autoregressize_mask(
            data_shapes, cfg.mask_ratios, cfg.traj_length, cfg.device, cfg.mode_weights
        ),
        MaskType.RCBC: lambda: create_rcbc_mask(cfg.traj_length, cfg.device),
        MaskType.GOAL: lambda: maybe_add_rew_to_mask(
            cfg.traj_length,
            cfg.device,
            create_goal_reaching_masks,
            has_rew,
            has_img,
            has_ret,
        ),
        MaskType.GOAL_N: lambda: maybe_add_rew_to_mask(
            cfg.traj_length,
            cfg.device,
            create_goal_n_reaching_masks,
            has_rew,
            has_img,
            has_ret,
        ),
        MaskType.ID: lambda: maybe_add_rew_to_mask(
            cfg.traj_length,
            cfg.device,
            create_inverse_dynamics_mask,
            has_rew,
            has_img,
            has_ret,
        ),
        MaskType.FD: lambda: maybe_add_rew_to_mask(
            cfg.traj_length,
            cfg.device,
            create_forward_dynamics_mask,
            has_rew,
            has_img,
            has_ret,
        ),
        MaskType.BC: lambda: maybe_add_rew_to_mask(
            cfg.traj_length,
            cfg.device,
            create_bc_mask,
            has_rew,
            has_img,
            has_ret,
        ),
        MaskType.BC_RANDOM: lambda: maybe_add_rew_to_mask(
            cfg.traj_length,
            cfg.device,
            lambda l, d: create_random_bc_masks(l, d, data_shapes, p=0.5),
            has_rew,
            has_img,
            has_ret,
        ),
    }

    mask_functions = [mask_functions_map[MaskType[i]] for i in cfg.mask_patterns]
    eval_masks = create_random_masks(
        data_shapes, cfg.mask_ratios, cfg.traj_length, cfg.device
    )

    epoch = 0

    batch_iter = iter(train_loader)
    while True:
        B = time.time()
        log_dict = {}
        log_dict["train/epochs"] = epoch

        if hydra_cfg.state_only_dataset is not None and step % (cfg.tsp_ratio + 1) == 0:
            s_t = time.time()
            try:
                state_only_batch = next(state_only_iter)
            except StopIteration:
                state_only_iter = iter(state_only_train_loader)
                state_only_batch = next(state_only_iter)

            state_only_batch = {
                k: v.to(cfg.device, non_blocking=True)
                for k, v in state_only_batch.items()
            }
            # ranodmly select mask
            while True:
                masks = random.choice(mask_functions)()
                # check that the mask for states is not all ones
                if masks["states"].sum() != np.prod(masks["states"].shape):
                    break
            masks["actions"] = masks["actions"] * 0  # ignore all actions
            state_only_batch["actions"] = (
                state_only_batch["actions"] * 0
            )  # ignore all actions

            # check if state mask is all ones
            state_only_log_dict = train_one_batch(
                model,
                optimizer,
                scheduler,
                state_only_tokenizer_manager,
                discrete_map,
                state_only_batch,
                masks,
                loss_keys=["states", "returns"],
            )
            for k, v in state_only_log_dict.items():
                log_dict[f"state_only_{k}"] = v
            log_dict["time/state_only_train"] = time.time() - s_t
        else:
            start_time = time.time()
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_loader)
                batch = next(batch_iter)
                epoch += 1

            # cycle between different types

            # ranodmly select mask
            masks = random.choice(mask_functions)()
            while torch.stack([v.sum() for v in masks.values()]).sum() == 0:
                masks = random.choice(mask_functions)()

            if "images" in batch and "images" not in masks:
                masks["images"] = masks["states"]

            batch = {k: v.to(cfg.device, non_blocking=True) for k, v in batch.items()}
            _log_dict = train_one_batch(
                model,
                optimizer,
                scheduler,
                tokenizer_manager,
                discrete_map,
                batch,
                masks,
            )
            log_dict.update(_log_dict)
            # log train step time = time to process a batch
            log_dict["time/train_step"] = time.time() - start_time

        if step % cfg.print_every == 0:
            try:
                train_loss = log_dict["train/train_loss"]
            except:
                train_loss = -1
            logger.info(f"Step: {step}, Train Loss: {train_loss}")

        if dp.rank == 0 and step % cfg.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "eval_max": dict(eval_max),
                },
                f"model_{step}.pt",
            )
            try:
                if step > 3 * cfg.save_every:
                    remove_step = step - 3 * cfg.save_every
                    if (remove_step // cfg.save_every) % 10 != 0:
                        os.remove(f"model_{remove_step}.pt")
            except Exception as e:
                logger.error(f"Failed to remove model file! {e}")

        if step % cfg.eval_every == 0 and step != 0:
            # if step % cfg.eval_every == 0:
            # evaluate the model
            start_time = time.time()
            model.eval()
            val_batch = next(iter(val_loader))
            val_batch = {
                k: v.to(cfg.device, non_blocking=True) for k, v in val_batch.items()
            }
            if hydra_cfg.state_only_dataset is not None:
                log_dict = state_only_val_dataset.eval_logs(model, tokenizer_manager)
            else:
                log_dict = val_dataset.eval_logs(model, tokenizer_manager)

            _val_dict = evaluate(
                model,
                tokenizer_manager,
                discrete_map,
                val_batch,
                vis_batch,
                eval_masks,
            )
            log_dict.update(_val_dict)

            if hydra_cfg.state_only_dataset is not None:
                # take state only batch and inspect mse error on action prediction
                state_val_batch = next(iter(state_only_val_loader))
                state_val_batch = {
                    k: v.to(cfg.device, non_blocking=True)
                    for k, v in state_val_batch.items()
                }
                log_dict.update(
                    eval_full_id(
                        model,
                        state_only_val_dataset.env,
                        state_val_batch,
                        state_only_tokenizer_manager,
                    )
                )

            # for everything with eval prefix keep the max
            max_log = {}
            for k, v in log_dict.items():
                if k.startswith("eval"):
                    eval_max[k] = max(eval_max[k], v)
                    max_log[f"max_{k}"] = eval_max[k]
            log_dict.update(max_log)

            wandb_logger.log(
                {f"p_{k}": v for k, v in max_log.items()},
                step=0,  # use step 0 to log to the same bar plot
            )
            log_dict["time/eval_time"] = time.time() - start_time

            if cfg.traj_length >= 2 and hasattr(val_dataset, "env"):
                log_dict.update(
                    eval_fd(model, val_dataset.env, val_batch, tokenizer_manager)
                )
                log_dict.update(
                    eval_id(model, val_dataset.env, val_batch, tokenizer_manager)
                )

            wandb_logger.log(log_dict, step=step)
            val_loss = log_dict["val/val_loss"]
            logger.info(f"Step: {step}, Val Loss: {val_loss}")
            model.train()

        log_dict["time/iteration_step_time"] = time.time() - B

        # if step % cfg.log_every == 0:
        if random.randint(0, cfg.log_every) == 0:
            logger.info(f"Step {step}")
            wandb_logger.log(log_dict, step=step)

        step += 1
        if step >= cfg.num_train_steps:
            break


@hydra.main(config_path=".", config_name="config", version_base="1.1")
def configure_jobs(hydra_data: DictConfig) -> None:
    logger.info(hydra_data)
    main(hydra_data)


if __name__ == "__main__":
    configure_jobs()
