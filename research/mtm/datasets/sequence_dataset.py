# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Tuple

import gym
import numpy as np
import torch
import tqdm
import wandb
from torch.utils.data import IterableDataset

from research.jaxrl.datasets.d4rl_dataset import D4RLDataset
from research.jaxrl.utils import make_env
from research.mtm.datasets.base import DataStatistics
from research.mtm.tokenizers.base import TokenizerManager


@dataclass(frozen=True)
class Trajectory:
    """Immutable container for a Trajectory.

    Each has shape (T, X).
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray

    def __post_init__(self):
        assert self.observations.shape[0] == self.actions.shape[0]
        assert self.observations.shape[0] == self.rewards.shape[0]

    def __len__(self) -> int:
        return self.observations.shape[0]

    @staticmethod
    def create_empty(
        observation_shape: Sequence[int], action_shape: Sequence[int]
    ) -> "Trajectory":
        """Create an empty trajectory."""
        return Trajectory(
            observations=np.zeros((0,) + observation_shape),
            actions=np.zeros((0,) + action_shape),
            rewards=np.zeros((0, 1)),
        )

    def append(
        self, observation: np.ndarray, action: np.ndarray, reward: float
    ) -> "Trajectory":
        """Append a new observation, action, and reward to the trajectory."""
        assert observation.shape == self.observations.shape[1:]
        assert action.shape == self.actions.shape[1:]
        return Trajectory(
            observations=np.concatenate((self.observations, observation[None])),
            actions=np.concatenate((self.actions, action[None])),
            rewards=np.concatenate((self.rewards, np.array([reward])[None])),
        )


SampleActions = Callable[[np.ndarray, Trajectory], np.ndarray]
"""Sample an action given the current observation and past history.

Parameters
----------
observation : np.ndarray, shape=(O,)
    Observation at time t.
trajectory_history : Trajectory
    History of observations, actions, and rewards from 0 to t-1.

Returns
-------
jnp.ndarray, shape=(A,)
    The sampled action.
"""


def segment(observations, terminals, max_path_length):
    """
    segment `observations` into trajectories according to `terminals`
    """
    assert len(observations) == len(terminals)
    observation_dim = observations.shape[1]

    trajectories = [[]]
    for obs, term in zip(observations, terminals):
        trajectories[-1].append(obs)
        if term.squeeze():
            trajectories.append([])

    if len(trajectories[-1]) == 0:
        trajectories = trajectories[:-1]

    ## list of arrays because trajectories lengths will be different
    trajectories = [np.stack(traj, axis=0) for traj in trajectories]

    n_trajectories = len(trajectories)
    path_lengths = [len(traj) for traj in trajectories]

    ## pad trajectories to be of equal length
    trajectories_pad = np.zeros(
        (n_trajectories, max_path_length, observation_dim), dtype=trajectories[0].dtype
    )
    early_termination = np.zeros((n_trajectories, max_path_length), dtype=bool)
    for i, traj in enumerate(trajectories):
        path_length = path_lengths[i]
        trajectories_pad[i, :path_length] = traj
        early_termination[i, path_length:] = 1

    return trajectories_pad, early_termination, path_lengths


class SequenceDataset:
    # For trajectory transformer
    # this is SequenceDataset and DiscreteDataset combined into one class. (trying to reduce class inheritance, in
    # janners code its literally only SequenceDataset is only used as a parent class to DiscreteDataset)
    # (https://github.com/jannerm/trajectory-transformer/blob/e0b5f12677a131ee87c65bc01179381679b3cfef/trajectory/datasets/sequence.py#L44)
    def __init__(
        self,
        dataset: D4RLDataset,
        discount: float = 0.99,
        sequence_length: int = 32,
        max_path_length: int = 1000,
        use_reward: bool = True,
        name: str = "",
    ):
        self.env = dataset.env
        self.dataset = dataset
        self.max_path_length = max_path_length
        self.sequence_length = sequence_length
        self._use_reward = use_reward
        self._name = name

        # extract data from Dataset
        self.observations_raw = dataset.observations
        self.actions_raw = dataset.actions
        self.rewards_raw = dataset.rewards.reshape(-1, 1)
        self.terminals_raw = dataset.dones_float

        ## segment
        self.actions_segmented, self.termination_flags, self.path_lengths = segment(
            self.actions_raw, self.terminals_raw, max_path_length
        )
        self.observations_segmented, *_ = segment(
            self.observations_raw, self.terminals_raw, max_path_length
        )
        self.rewards_segmented, *_ = segment(
            self.rewards_raw, self.terminals_raw, max_path_length
        )

        if discount > 1.0:
            self.discount = 1.0
            self.use_avg = True
        else:
            self.discount = discount
            self.use_avg = False

        self.discounts = (self.discount ** np.arange(self.max_path_length))[:, None]

        ## [ n_paths x max_path_length x 1 ]
        self.values_segmented = np.zeros(self.rewards_segmented.shape)

        for t in range(max_path_length):
            ## [ n_paths x 1 ]
            V = (self.rewards_segmented[:, t + 1 :] * self.discounts[: -t - 1]).sum(
                axis=1
            )
            self.values_segmented[:, t] = V

        N_p, Max_Path_Len, _ = self.values_segmented.shape
        if self.use_avg:
            divisor = np.arange(1, Max_Path_Len + 1)[::-1][None, :, None]
            self.values_segmented = self.values_segmented / divisor

        values_raw = self.values_segmented.squeeze(axis=-1).reshape(-1)
        values_mask = ~self.termination_flags.reshape(-1)
        self.values_raw = values_raw[values_mask, None]

        self.observation_dim = self.observations_raw.shape[1]
        self.action_dim = self.actions_raw.shape[1]

        assert (
            self.observations_segmented.shape[0]
            == self.actions_segmented.shape[0]
            == self.rewards_segmented.shape[0]
            == self.values_segmented.shape[0]
        )
        #  assert len(set(self.path_lengths)) == 1
        keep_idx = []
        index_map = {}
        count = 0
        traj_count = 0
        for idx, pl in enumerate(self.path_lengths):
            if pl < sequence_length:
                pass
            else:
                keep_idx.append(idx)
                for i in range(pl - sequence_length + 1):
                    index_map[count] = (traj_count, i)
                    count += 1
                traj_count += 1

        self.index_map = index_map
        self.path_lengths = np.array(self.path_lengths)[keep_idx]
        self.observations_segmented = self.observations_segmented[keep_idx]
        self.actions_segmented = self.actions_segmented[keep_idx]
        self.rewards_segmented = self.rewards_segmented[keep_idx]
        self.values_segmented = self.values_segmented[keep_idx]
        self.num_trajectories = self.observations_segmented.shape[0]

        self.raw_data = {
            "states": self.observations_raw,
            "actions": self.actions_raw,
            "rewards": self.rewards_raw,
            "returns": self.values_raw,
        }

    def __len__(self) -> int:
        # return self.num_trajectories
        return len(self.index_map)

    @property
    def num_traj(self) -> int:
        return len(self.path_lengths)

    def get_trajectory(self, traj_index: int) -> Dict[str, np.ndarray]:
        if self._use_reward:
            return {
                "states": self.observations_segmented[traj_index],
                "actions": self.actions_segmented[traj_index],
                "rewards": self.rewards_segmented[traj_index],
                "returns": self.values_segmented[traj_index],
            }
        else:
            return {
                "states": self.observations_segmented[traj_index],
                "actions": self.actions_segmented[traj_index],
            }

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Return a trajectories of the form (observations, actions, rewards, values).

        A random trajectory with self.sequence_length is returned.
        """
        idx, start_idx = self.index_map[index]
        traj = self.get_trajectory(idx)
        return {
            k: v[start_idx : start_idx + self.sequence_length] for k, v in traj.items()
        }

    def eval_logs(
        self, model: Callable, tokenizer_manager: TokenizerManager
    ) -> Dict[str, Any]:
        log_data = {}
        observation_shape = self.observations_raw.shape
        action_shape = self.actions_raw.shape
        device = next(model.parameters()).device

        bc_sampler = lambda o, t: sample_action_bc(
            o, t, model, tokenizer_manager, observation_shape, action_shape, device
        )
        results, videos = evaluate(
            bc_sampler,
            self.dataset.env,
            20,
            (self.observation_dim,),
            (self.action_dim,),
            num_videos=0,
        )
        for k, v in results.items():
            log_data[f"eval_bc/{k}"] = v
        for idx, v in enumerate(videos):
            log_data[f"eval_bc_video_{idx}/video"] = wandb.Video(
                v.transpose(0, 3, 1, 2), fps=10, format="gif"
            )

        if "returns" in tokenizer_manager.tokenizers:
            for p in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]:
                bc_sampler = lambda o, t: sample_action_bc2(
                    o,
                    t,
                    model,
                    tokenizer_manager,
                    observation_shape,
                    action_shape,
                    device,
                    percentage=p,
                )
                results, videos = evaluate(
                    bc_sampler,
                    self.dataset.env,
                    20,
                    (self.observation_dim,),
                    (self.action_dim,),
                    num_videos=0,
                )
                for k, v in results.items():
                    log_data[f"eval2/p={p}_{k}"] = v
                for idx, v in enumerate(videos):
                    log_data[f"eval2_video_{idx}/p={p}_video"] = wandb.Video(
                        v.transpose(0, 3, 1, 2), fps=10, format="gif"
                    )

        if "returns" in tokenizer_manager.tokenizers:
            for p in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]:
                bc_sampler = lambda o, t: sample_action_bc_two_stage(
                    o,
                    t,
                    model,
                    tokenizer_manager,
                    observation_shape,
                    action_shape,
                    device,
                    percentage=p,
                )
                results, videos = evaluate(
                    bc_sampler,
                    self.dataset.env,
                    20,
                    (self.observation_dim,),
                    (self.action_dim,),
                    num_videos=0,
                )
                for k, v in results.items():
                    log_data[f"eval_ts/p={p}_{k}"] = v
                for idx, v in enumerate(videos):
                    log_data[f"eval_ts_video_{idx}/p={p}_video"] = wandb.Video(
                        v.transpose(0, 3, 1, 2), fps=10, format="gif"
                    )

        return log_data

    def trajectory_statistics(self) -> Dict[str, DataStatistics]:
        """Shapes of the trajectories in the dataset."""

        if self.use_avg:
            # special hard-coded case for using avg reward to go
            save_path = Path(f"/tmp/d4rl/d4rl_statistics_{self._name}_avg.pkl")
        else:
            if self.discount == 1.0:
                save_path = Path(f"/tmp/d4rl/d4rl_statistics_{self._name}_d=1.0.pkl")
            elif self.discount == 0.99:
                save_path = Path(f"/tmp/d4rl/d4rl_statistics_{self._name}.pkl")
            else:
                raise NotImplementedError

        if save_path.exists():
            with open(save_path, "rb") as f:
                ret_dict = pickle.load(f)
                if "values" in ret_dict:
                    ret_dict["returns"] = ret_dict["values"]
                    ret_dict.pop("values")
                return ret_dict

        trajectories = {
            "states": self.observations_segmented,
            "actions": self.actions_segmented,
            "rewards": self.rewards_segmented,
            "returns": self.values_segmented,
        }

        # average over samples and time
        ret_dict = {
            k: DataStatistics(
                mean=v.mean(axis=(0, 1)),
                std=v.std(axis=(0, 1)),
                min=v.min(axis=(0, 1)),
                max=v.max(axis=(0, 1)),
            )
            for k, v in trajectories.items()
        }
        try:
            # make sure the directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(ret_dict, f)
        except Exception as e:
            print(f"Failed to dump trajectory statistics: {e}")
        return ret_dict


@torch.inference_mode()
def sample_action_git(
    observation: np.ndarray,
    traj: Trajectory,
    model,
    tokenizer_manager,
    observation_shape,
    action_shape,
    device,
):
    traj_len = model.max_len

    # make observations and actions
    observations = np.zeros((traj_len, observation_shape[1]))
    actions = np.zeros((traj_len, action_shape[1]))
    rewards = np.zeros((traj_len, 1))
    returns = np.zeros((traj_len, 1))

    observations[0] = observation
    mask = np.zeros(traj_len)
    obs_mask = np.zeros(traj_len)
    obs_mask[0] = 1
    masks = {
        "states": obs_mask,
        "actions": mask,
        "rewards": mask,
        "returns": mask,
    }
    trajectories = {
        "states": observations,
        "actions": actions,
        "rewards": rewards,
        "returns": returns,
    }
    torch_trajectories = {
        k: torch.tensor(v, device=device)[None] for k, v in trajectories.items()
    }
    torch_masks = {k: torch.tensor(v, device=device) for k, v in masks.items()}
    encoded_trajectories = tokenizer_manager.encode(torch_trajectories)

    predicted = model.mask_git_forward(encoded_trajectories, torch_masks, ratio=0.34)
    decode = tokenizer_manager.decode(predicted)

    # extract_action
    a = decode["actions"][0][0].cpu().numpy()
    return a


@torch.inference_mode()
def sample_action_bc(
    observation: np.ndarray,
    traj: Trajectory,
    model,
    tokenizer_manager,
    observation_shape,
    action_shape,
    device,
):
    traj_len = model.max_len

    # make observations and actions
    observations = np.zeros((traj_len, observation_shape[1]))
    actions = np.zeros((traj_len, action_shape[1]))
    rewards = np.zeros((traj_len, 1))
    returns = np.zeros((traj_len, 1))

    observations[0] = observation
    mask = np.zeros(traj_len)
    obs_mask = np.zeros(traj_len)
    obs_mask[0] = 1
    masks = {
        "states": obs_mask,
        "actions": mask,
        "rewards": mask,
        "returns": mask,
    }
    trajectories = {
        "states": observations,
        "actions": actions,
        "rewards": rewards,
        "returns": returns,
    }
    torch_trajectories = {
        k: torch.tensor(v, device=device)[None] for k, v in trajectories.items()
    }
    torch_masks = {k: torch.tensor(v, device=device) for k, v in masks.items()}
    encoded_trajectories = tokenizer_manager.encode(torch_trajectories)

    predicted = model(encoded_trajectories, torch_masks)
    decode = tokenizer_manager.decode(predicted)

    # extract_action
    a = decode["actions"][0][0].cpu().numpy()
    return a


@torch.inference_mode()
def sample_action_bc2(
    observation: np.ndarray,
    traj: Trajectory,
    model,
    tokenizer_manager,
    observation_shape,
    action_shape,
    device,
    percentage=1.0,
):
    traj_len = model.max_len
    # make observations and actions

    observations = np.zeros((traj_len, observation_shape[1]))
    actions = np.zeros((traj_len, action_shape[1]))

    return_max = tokenizer_manager.tokenizers["returns"].stats.max
    return_min = tokenizer_manager.tokenizers["returns"].stats.min

    return_value = return_min + (return_max - return_min) * percentage
    return_to_go = float(return_value)
    returns = return_to_go * np.ones((traj_len, 1))

    masks = np.zeros(traj_len)

    i = -1
    max_len = min(traj_len - 1, len(traj))
    assert max_len < traj_len
    for i in range(max_len):
        observations[i] = traj.observations[-max_len + i]
        actions[i] = traj.actions[-max_len + i]
        # rewards[i] = traj.rewards[-max_len + i]
        masks[i] = 1

    assert i == max_len - 1
    # fill in the rest with the current observation
    observations[i + 1] = observation
    obs_mask = np.copy(masks)
    obs_mask[i + 1] = 1

    # pass through tokenizer
    trajectories = {
        "states": observations,
        "actions": actions,
        "returns": returns,
    }

    reward_mask = np.ones(traj_len)
    masks = {
        "states": obs_mask,
        "actions": masks,
        "returns": reward_mask,
    }

    # convert to tensors and add
    torch_trajectories = {
        k: torch.tensor(v, device=device)[None] for k, v in trajectories.items()
    }
    torch_masks = {k: torch.tensor(v, device=device) for k, v in masks.items()}

    encoded_trajectories = tokenizer_manager.encode(torch_trajectories)

    predicted = model(encoded_trajectories, torch_masks)
    decode = tokenizer_manager.decode(predicted)

    # extract_action
    a = decode["actions"][0][i + 1].cpu().numpy()
    return a


@torch.inference_mode()
def sample_action_bc5(
    observation: np.ndarray,
    traj: Trajectory,
    model,
    tokenizer_manager,
    observation_shape,
    action_shape,
    device,
    percentage=1.0,
):
    traj_len = model.max_len

    # make observations and actions
    observations = np.zeros((traj_len, observation_shape[1]))
    actions = np.zeros((traj_len, action_shape[1]))
    rewards = np.zeros((traj_len, 1))

    return_max = tokenizer_manager.tokenizers["returns"].stats.max
    return_min = tokenizer_manager.tokenizers["returns"].stats.min

    return_value = return_min + (return_max - return_min) * percentage
    return_to_go = float(return_value)
    returns = return_to_go * np.ones((traj_len, 1))

    observations[0] = observation
    mask = np.zeros(traj_len)
    obs_mask = np.zeros(traj_len)
    obs_mask[0] = 1
    masks = {
        "states": obs_mask,
        "actions": mask,
        "rewards": mask,
        "returns": obs_mask,
    }
    trajectories = {
        "states": observations,
        "actions": actions,
        "rewards": rewards,
        "returns": returns,
    }
    torch_trajectories = {
        k: torch.tensor(v, device=device)[None] for k, v in trajectories.items()
    }
    torch_masks = {k: torch.tensor(v, device=device) for k, v in masks.items()}
    encoded_trajectories = tokenizer_manager.encode(torch_trajectories)
    predicted = model(encoded_trajectories, torch_masks)
    decode = tokenizer_manager.decode(predicted)

    # extract_action
    a = decode["actions"][0][0].cpu().numpy()
    return a


@torch.inference_mode()
def sample_action_bc4(
    observation: np.ndarray,
    traj: Trajectory,
    model,
    tokenizer_manager,
    observation_shape,
    action_shape,
    device,
    percentage=1.0,
):
    traj_len = model.max_len

    # make observations and actions
    observations = np.zeros((traj_len, observation_shape[1]))
    actions = np.zeros((traj_len, action_shape[1]))
    rewards = np.zeros((traj_len, 1))

    return_max = tokenizer_manager.tokenizers["returns"].stats.max
    return_min = tokenizer_manager.tokenizers["returns"].stats.min

    return_value = return_min + (return_max - return_min) * percentage
    return_to_go = float(return_value)
    returns = return_to_go * np.ones((traj_len, 1))

    observations[0] = observation
    mask = np.zeros(traj_len)
    obs_mask = np.zeros(traj_len)
    obs_mask[0] = 1
    ret_mask = np.ones(traj_len)
    masks = {
        "states": obs_mask,
        "actions": mask,
        "rewards": mask,
        "returns": ret_mask,
    }
    trajectories = {
        "states": observations,
        "actions": actions,
        "rewards": rewards,
        "returns": returns,
    }
    torch_trajectories = {
        k: torch.tensor(v, device=device)[None] for k, v in trajectories.items()
    }
    torch_masks = {k: torch.tensor(v, device=device) for k, v in masks.items()}
    encoded_trajectories = tokenizer_manager.encode(torch_trajectories)
    predicted = model(encoded_trajectories, torch_masks)
    decode = tokenizer_manager.decode(predicted)

    # extract_action
    a = decode["actions"][0][0].cpu().numpy()
    return a


@torch.inference_mode()
def sample_action_bc3(
    observation: np.ndarray,
    traj: Trajectory,
    model,
    tokenizer_manager,
    observation_shape,
    action_shape,
    device,
    N=256,
    top_k=50,
    cem_iterations=2,
):
    traj_len = model.max_len
    # make observations and actions
    observations = np.zeros((traj_len, observation_shape[1]))
    actions = np.zeros((traj_len, action_shape[1]))
    rewards = np.zeros((traj_len, 1))

    observations[0] = observation
    mask = np.zeros(traj_len)
    obs_mask = np.zeros(traj_len)
    obs_mask[0] = 1
    masks = {
        "states": obs_mask,
        "actions": mask,
        "rewards": mask,
    }
    trajectories = {
        "states": observations,
        "actions": actions,
        "rewards": rewards,
    }
    torch_trajectories = {
        k: torch.tensor(v, device=device)[None] for k, v in trajectories.items()
    }
    torch_masks = {k: torch.tensor(v, device=device) for k, v in masks.items()}
    encoded_trajectories = tokenizer_manager.encode(torch_trajectories)

    predicted = model(encoded_trajectories, torch_masks)
    decode = tokenizer_manager.decode(predicted)

    # extract_action sequence make new torch sequence with 1024 copies
    torch_trajectories = {
        k: torch.tensor(v, device=device)[None].repeat(N, 1, 1)
        for k, v in trajectories.items()
    }
    torch_trajectories["actions"] = decode["actions"].repeat(N, 1, 1)
    # add noise to the actions and clip
    torch_trajectories["actions"] += (
        torch.randn_like(torch_trajectories["actions"]) * 0.1
    )
    torch_trajectories["actions"] = torch.clamp(torch_trajectories["actions"], -1, 1)

    # compute rewards
    for i in range(cem_iterations):
        # pass through tokenizer

        encoded_trajectories = tokenizer_manager.encode(torch_trajectories)
        predicted = model(encoded_trajectories, torch_masks)
        decode = tokenizer_manager.decode(predicted)
        value_return = decode["rewards"].sum(dim=1)
        sorted_values, sorted_indices = torch.sort(value_return, descending=True)
        # get top k actions
        top_k_actions = torch_trajectories["actions"][sorted_indices[:top_k][:, 0]]
        # compute mean
        torch_trajectories["actions"] = top_k_actions.mean(dim=0, keepdim=True).repeat(
            N, 1, 1
        )
        action_std = top_k_actions.std(dim=0, keepdim=True)

        # add noise to the actions and clip
        torch_trajectories["actions"] += (
            torch.randn_like(torch_trajectories["actions"], device=device) * action_std
        )
        torch_trajectories["actions"] = torch.clamp(
            torch_trajectories["actions"], -1, 1
        )

    # extract_action
    a = torch_trajectories["actions"][0][0].cpu().numpy()
    return a


@torch.inference_mode()
def sample_action_bc_two_stage(
    observation: np.ndarray,
    traj: Trajectory,
    model,
    tokenizer_manager,
    observation_shape,
    action_shape,
    device,
    percentage=1.0,
):
    traj_len = model.max_len

    # make observations and actions
    observations = np.zeros((traj_len, observation_shape[1]))
    actions = np.zeros((traj_len, action_shape[1]))
    rewards = np.zeros((traj_len, 1))

    return_max = tokenizer_manager.tokenizers["returns"].stats.max
    return_min = tokenizer_manager.tokenizers["returns"].stats.min

    return_value = return_min + (return_max - return_min) * percentage
    return_to_go = float(return_value)
    returns = return_to_go * np.ones((traj_len, 1))

    observations[0] = observation
    mask = np.zeros(traj_len)
    obs_mask = np.zeros(traj_len)
    obs_mask[0] = 1
    ret_mask = np.zeros(traj_len)
    ret_mask[0] = 1
    masks = {
        "states": obs_mask,
        "actions": mask,
        "rewards": mask,
        "returns": ret_mask,
    }
    trajectories = {
        "states": observations,
        "actions": actions,
        "rewards": rewards,
        "returns": returns,
    }
    torch_trajectories = {
        k: torch.tensor(v, device=device)[None] for k, v in trajectories.items()
    }
    torch_masks = {k: torch.tensor(v, device=device) for k, v in masks.items()}
    encoded_trajectories = tokenizer_manager.encode(torch_trajectories)
    predicted = model(encoded_trajectories, torch_masks)
    decode = tokenizer_manager.decode(predicted)

    # fill in predicted states
    torch_trajectories["states"] = torch_trajectories["states"] * torch_masks["states"][
        None, :, None
    ] + decode["states"] * (1 - torch_masks["states"][None, :, None])
    # fill in predicted returns
    torch_trajectories["returns"] = torch_trajectories["returns"] * torch_masks[
        "returns"
    ][None, :, None] + decode["returns"] * (1 - torch_masks["returns"][None, :, None])

    # update masks
    masks = {
        "states": np.ones(traj_len),
        "actions": mask,
        "rewards": mask,
        "returns": np.ones(traj_len),
    }
    torch_masks = {k: torch.tensor(v, device=device) for k, v in masks.items()}
    encoded_trajectories = tokenizer_manager.encode(torch_trajectories)
    predicted = model(encoded_trajectories, torch_masks)
    decode = tokenizer_manager.decode(predicted)

    # extract_action
    a = decode["actions"][0][0].cpu().numpy()
    return a


def evaluate(
    sample_actions: SampleActions,
    env: gym.Env,
    num_episodes: int,
    observation_space: Tuple[int, ...],
    action_space: Tuple[int, ...],
    disable_tqdm: bool = True,
    verbose: bool = False,
    all_results: bool = False,
    num_videos: int = 3,
) -> Dict[str, Any]:
    # stats: Dict[str, Any] = {"return": [], "length": []}
    stats: Dict[str, Any] = defaultdict(list)
    successes = None

    pbar = tqdm.tqdm(range(num_episodes), disable=disable_tqdm, ncols=85)

    videos = []

    for i in pbar:
        observation, done = env.reset(), False
        trajectory_history = Trajectory.create_empty(observation_space, action_space)
        if len(videos) < num_videos:
            try:
                imgs = [env.sim.render(64, 48, camera_name="track")[::-1]]
            except:
                imgs = [env.render()[::-1]]

        while not done:
            action = sample_actions(observation, trajectory_history)
            action = np.clip(action, -1, 1)
            new_observation, reward, done, info = env.step(action)
            trajectory_history = trajectory_history.append(observation, action, reward)
            observation = new_observation
            if len(videos) < num_videos:
                try:
                    imgs.append(env.sim.render(64, 48, camera_name="track")[::-1])
                except:
                    imgs.append(env.render()[::-1])

        if len(videos) < num_videos:
            videos.append(np.array(imgs[:-1]))

        if "episode" in info:
            for k in stats.keys():
                stats[k].append(float(info["episode"][k]))
                if verbose:
                    print(f"{k}: {info['episode'][k]}")

            ret = info["episode"]["return"]
            mean = np.mean(stats["return"])
            pbar.set_description(f"iter={i}\t last={ret:.2f} mean={mean}")
            if "is_success" in info:
                if successes is None:
                    successes = 0.0
                successes += info["is_success"]
        else:
            stats["return"].append(trajectory_history.rewards.sum())
            stats["length"].append(len(trajectory_history.rewards))
            stats["achieved"].append(int(info["goal_achieved"]))

    new_stats = {}
    for k, v in stats.items():
        new_stats[k + "_mean"] = float(np.mean(v))
        new_stats[k + "_std"] = float(np.std(v))
    if all_results:
        new_stats.update(stats)
    stats = new_stats

    if successes is not None:
        stats["success"] = successes / num_episodes

    return stats, videos


@torch.inference_mode()
def sample_action_cem(
    observation: np.ndarray,
    traj: Trajectory,
    model,
    tokenizer_manager,
    observation_shape,
    action_shape,
    device,
    N=1024,
    top_k=128,
    cem_iterations=2,
):
    traj_len = model.max_len
    # make observations and actions
    observations = np.zeros((traj_len, observation_shape[1]))
    actions = np.zeros((traj_len, action_shape[1]))
    returns = np.zeros((traj_len, 1))

    observations[0] = observation
    mask = np.zeros(traj_len)
    obs_mask = np.zeros(traj_len)
    obs_mask[0] = 1
    masks = {
        "states": obs_mask,
        "actions": mask,
        "returns": mask,
    }
    trajectories = {
        "states": observations,
        "actions": actions,
        "returns": returns,
    }
    torch_trajectories = {
        k: torch.tensor(v, device=device)[None] for k, v in trajectories.items()
    }
    torch_masks = {k: torch.tensor(v, device=device) for k, v in masks.items()}
    encoded_trajectories = tokenizer_manager.encode(torch_trajectories)

    predicted = model(encoded_trajectories, torch_masks)
    decode = tokenizer_manager.decode(predicted)

    # extract_action sequence make new torch sequence with 1024 copies
    torch_trajectories = {
        k: torch.tensor(v, device=device)[None].repeat(N, 1, 1)
        for k, v in trajectories.items()
    }
    torch_trajectories["actions"] = decode["actions"].repeat(N, 1, 1)
    # add noise to the actions and clip
    torch_trajectories["actions"] += (
        torch.randn_like(torch_trajectories["actions"]) * 0.1
    )
    torch_trajectories["actions"] = torch.clamp(torch_trajectories["actions"], -1, 1)

    # compute rewards
    for i in range(cem_iterations):
        # pass through tokenizer

        encoded_trajectories = tokenizer_manager.encode(torch_trajectories)
        predicted = model(encoded_trajectories, torch_masks)
        decode = tokenizer_manager.decode(predicted)
        value_return = decode["returns"].sum(dim=1)
        sorted_values, sorted_indices = torch.sort(value_return, descending=True)
        # get top k actions
        top_k_actions = torch_trajectories["actions"][sorted_indices[:top_k][:, 0]]
        # compute mean
        torch_trajectories["actions"] = top_k_actions.mean(dim=0, keepdim=True).repeat(
            N, 1, 1
        )
        action_std = top_k_actions.std(dim=0, keepdim=True)

        # add noise to the actions and clip
        torch_trajectories["actions"] += (
            torch.randn_like(torch_trajectories["actions"], device=device) * action_std
        )
        torch_trajectories["actions"] = torch.clamp(
            torch_trajectories["actions"], -1, 1
        )

    # extract_action
    a = torch_trajectories["actions"][0][0].cpu().numpy()
    return a


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn, domain, obs):
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


def relable_episode(env, episode):
    rewards = []
    reward_spec = env.reward_spec()
    states = episode["physics"]
    for i in range(states.shape[0]):
        with env.physics.reset_context():
            env.physics.set_state(states[i])
        reward = env.task.get_reward(env.physics)
        reward = np.full(reward_spec.shape, reward, reward_spec.dtype)
        rewards.append(reward)
    episode["reward"] = np.array(rewards, dtype=reward_spec.dtype)
    return episode


class OfflineReplayBuffer(IterableDataset):
    def __init__(
        self,
        env,
        replay_dir,
        max_size,
        num_workers,
        discount,
        domain,
        traj_length,
        mode,
        cfg,
        relabel,
        obs,
    ):
        print("in SINGLE replay buffer")
        # self._env = env
        self._replay_dir = replay_dir
        self._domain = domain
        self._mode = mode
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._discount = discount
        self._loaded = False
        self._traj_length = traj_length
        self._cfg = cfg
        self._relabel = relabel
        self._obs = obs
        # self.env = env
        # print('seed', np.random.get_state()[1][0])
        # random.seed(np.random.get_state()[1][0])

    def _load(self, relable=True):
        if relable:
            print("Labeling data...")
        else:
            print("loading reward free data...")
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(
            self._replay_dir.rglob("*.npz")
        )  # get all episodes recursively
        for eps_fn in eps_fns:
            if self._size > self._max_size:
                print("over size", self._max_size)
                break
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            episode = load_episode(eps_fn, self._domain, self._obs)
            if relable:
                episode = self._relable_reward(episode)
            self._episode_fns.append(eps_fn)
            self._episodes[eps_fn] = episode
            self._size += episode_len(episode)

    def __len__(self) -> int:
        return self._size

    def _sample_episode(self):
        if not self._loaded:
            self._load(self._relabel)
            self._loaded = True
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _relable_reward(self, episode):
        return relable_episode(self._env, episode)

    def _sample(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._traj_length + 1) + 1
        obs = episode["observation"][idx - 1 : idx - 1 + self._traj_length]
        action = episode["action"][idx : idx + self._traj_length]
        next_obs = episode["observation"][idx : idx + self._traj_length]
        reward = episode["reward"][idx : idx + self._traj_length]
        discount = episode["discount"][idx : idx + self._traj_length] * self._discount
        timestep = np.arange(idx - 1, idx + self._traj_length - 1)[:, np.newaxis]
        return {
            "observations": obs,
            "actions": action,
            "rewards": reward,
            "discount": discount,
            "next_observations": next_obs,
            "timestep": 0,
        }

    def _sample_goal(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        start_idx = np.random.randint(0, 200)
        length = np.random.randint(15, 20)
        start_obs = episode["observation"][start_idx]
        start_physics = episode["physics"][start_idx]
        goal_obs = episode["observation"][start_idx + length - 1]
        goal_physics = episode["physics"][start_idx + length - 1]
        timestep = length - 1
        # print(action.shape)
        return (start_obs, start_physics, goal_obs, goal_physics, timestep)

    def _sample_multiple_goal(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        start_idx = np.random.randint(0, 180)
        time_budget = np.array([12, 24, 36, 48, 60])

        start_obs = episode["observation"][start_idx]
        start_physics = episode["physics"][start_idx]

        goal = episode["observation"][start_idx + time_budget]
        goal_physics = episode["physics"][start_idx + time_budget]

        # print(action.shape)
        return (start_obs, start_physics, goal, goal_physics, time_budget)

    def _sample_context(self):
        episode = self._sample_episode()
        context_length = self._cfg.context_length
        forecast_length = self._cfg.forecast_length
        # add +1 for the first dummy transition
        # idx = np.random.randint(0, 50 - context_length+ 1) + 1
        start_idx = np.random.randint(100, 850)
        obs = episode["observation"][
            start_idx - 1 : start_idx + context_length
        ]  # last state is the initial obs
        action = episode["action"][start_idx : start_idx + context_length]
        reward = episode["reward"][
            start_idx + context_length : start_idx + context_length + forecast_length
        ]
        physics = episode["physics"][start_idx - 1 : start_idx + context_length]
        remaining = episode["action"][
            start_idx + context_length : start_idx + context_length + forecast_length
        ]
        return (obs, action, physics, reward, remaining)

    def _s(self) -> Dict[str, np.ndarray]:
        if self._mode is None:
            return self._sample()
        elif self._mode == "goal":
            return self._sample_goal()
        elif self._mode == "multi_goal":
            return self._sample_multiple_goal()
        elif self._mode == "prompt":
            return self._sample_context()
        else:
            raise NotImplementedError

    def __iter__(self):
        while True:
            yield self._s()

    def __getitem__(self, idx: int):
        return self._s()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(
    env,
    replay_dir,
    max_size,
    num_workers,
    discount,
    domain,
    traj_length=1,
    mode=None,
    cfg=None,
    relabel=True,
    obs="states",
):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = OfflineReplayBuffer(
        env,
        replay_dir,
        max_size_per_worker,
        num_workers,
        discount,
        domain,
        traj_length,
        mode,
        cfg,
        relabel,
        obs,
    )
    return iterable
