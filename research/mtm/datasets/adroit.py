# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import wandb
from mjrl.utils.gym_env import GymEnv
from torch.utils.data import Dataset

from research.mtm.datasets.base import DatasetProtocol, DataStatistics
from research.mtm.datasets.sequence_dataset import (
    evaluate,
    sample_action_bc,
    sample_action_bc2,
    sample_action_bc_two_stage,
)
from research.mtm.tokenizers.base import TokenizerManager


def episode_len(episode):
    return next(iter(episode.values())).shape[0]


def load_episodes(fn):
    with fn.open("rb") as f:
        episodes = pickle.load(f)
        return episodes


class UseAchievedGym(GymEnv):
    def step(self, *args, **kwargs):
        obs, _old_reward, done, info = super().step(*args, **kwargs)
        new_reward = info["goal_achieved"]
        return obs, new_reward, done, info


def get_datasets(
    seq_steps: bool,
    env_name: int,
    d_name: str = "expert",
    train_val_split: float = 0.95,
    discount: float = 1.5,
    use_achieved: bool = False,
    data_dir: str = "~/arxiv_data",
):
    assert env_name in ["pen", "door"]
    assert d_name in ["expert", "medium_replay"]

    env_n = f"{env_name}-v0"
    if use_achieved:
        env = UseAchievedGym(env_n)
    else:
        env = GymEnv(env_n)

    replay_dir = Path(data_dir).expanduser() / env_name
    data_path = replay_dir / f"{d_name}_dataset.pickle"

    episodes = load_episodes(data_path)
    for ep in episodes:
        for k in ep.keys():
            ep[k] = np.array(ep[k])

    # split into train and val
    cutoff = int(len(episodes) * train_val_split)
    train_episodes = episodes[:cutoff]
    val_episodes = episodes[cutoff:]

    if len(train_episodes) < 20:
        # copy 100 times speed up training
        train_episodes = train_episodes * 100

    if len(val_episodes) < 20:
        # copy 100 times speed up training
        val_episodes = val_episodes * 100

    train_dataset = OfflineReplayBuffer(
        env,
        train_episodes,
        discount,
        seq_steps,
        f"{env_name}_{d_name}",
        use_achieved=use_achieved,
    )
    val_dataset = OfflineReplayBuffer(
        env,
        val_episodes,
        discount,
        seq_steps,
        f"{env_name}_{d_name}",
        use_achieved=use_achieved,
    )
    return train_dataset, val_dataset


class OfflineReplayBuffer(Dataset, DatasetProtocol):
    def __init__(
        self,
        env,
        episodes,
        discount,
        traj_length,
        name,
        use_achieved: bool = False,
    ):
        print("in SINGLE replay buffer")
        self._env = env
        self.env = env
        self._episodes = episodes
        self._traj_length = traj_length
        self.name = name
        self.use_achieved = use_achieved

        # computations

        self._path_lengths = [episode_len(episode) for episode in self._episodes]

        self.max_path_length = max(self._path_lengths)
        # check that all path lengths are the same
        assert all(
            [path_len == self.max_path_length for path_len in self._path_lengths]
        )

        self.actions = np.array([ep["actions"] for ep in self._episodes])
        self.states = np.array([ep["observations"] for ep in self._episodes])
        if self.use_achieved:
            self.rewards = np.array(
                [ep["env_infos"].tolist()["goal_achieved"] for ep in self._episodes],
                dtype=float,
            )
        else:
            self.rewards = np.array([ep["rewards"] for ep in self._episodes])
        self.returns = np.zeros(self.rewards.shape)

        if discount > 1.0:
            self.discount = 1.0
            self.use_avg = True
        else:
            self.discount = discount
            self.use_avg = False
        discounts = (self.discount ** np.arange(self.max_path_length))[None, :]

        for t in range(self.max_path_length):
            ## [ n_paths x 1 ]

            if False:
                ret = (self.rewards[:, t + 1 :] * discounts[:, : -t - 1]).sum(axis=1)
            else:
                dis = discounts if t == 0 else discounts[:, :-t]
                ret = (self.rewards[:, t:] * dis).sum(axis=1)
            self.returns[:, t] = ret
        _, Max_Path_Len = self.returns.shape
        if self.use_avg:
            divisor = np.arange(1, Max_Path_Len + 1)[::-1][None, :]
            self.returns = self.returns / divisor

        index_map = {}
        count = 0
        traj_count = 0
        for idx, pl in enumerate(self._path_lengths):
            for i in range(pl - self._traj_length + 1):
                index_map[count] = (traj_count, i)
                count += 1
            traj_count += 1

        self.index_map = index_map
        self.num_trajectories = self.returns.shape[0]
        self.returns = self.returns[..., None]
        self.rewards = self.rewards[..., None]

        self.ts = self.trajectory_statistics()

        self.raw_data = {
            "states": self.states.reshape(
                self.num_trajectories * self.max_path_length, -1
            ),
            "actions": self.actions.reshape(
                self.num_trajectories * self.max_path_length, -1
            ),
            "rewards": self.rewards.reshape(
                self.num_trajectories * self.max_path_length, -1
            ),
            "returns": self.returns.reshape(
                self.num_trajectories * self.max_path_length, -1
            ),
        }

        print()

    def trajectory_statistics(self) -> Dict[str, DataStatistics]:
        if self.use_avg:
            save_path = f"/tmp/adroit/adroit_{self.name}_statistics_avg.pkl"
        else:
            save_path = f"/tmp/adroit/adroit_{self.name}_statistics_{self.discount}.pkl"

        if self.use_achieved:
            save_path = save_path.replace(".pkl", "_achieved.pkl")
        save_path = Path(save_path)

        if save_path.exists():
            with open(save_path, "rb") as f:
                ret_dict = pickle.load(f)
                return ret_dict

        keys = ["states", "actions", "rewards", "returns"]
        stats = {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "returns": self.returns,
        }

        data_stats = {}
        for key in keys:
            d = stats[key]
            B, T, X = d.shape
            d = d.reshape(B * T, X)
            data_stats[key] = DataStatistics(
                np.mean(d, axis=0),
                np.std(d, axis=0),
                np.min(d, axis=0),
                np.max(d, axis=0),
            )

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(data_stats, f)
        return data_stats

    def __len__(self) -> int:
        # return self.num_trajectories
        return len(self.index_map)

    def get_trajectory(self, traj_index: int) -> Dict[str, np.ndarray]:
        return {
            "states": self.states[traj_index],
            "actions": self.actions[traj_index],
            "rewards": self.rewards[traj_index],
            "returns": self.returns[traj_index],
        }

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Return a trajectories of the form (observations, actions, rewards, values).

        A random trajectory with self.sequence_length is returned.
        """
        idx, start_idx = self.index_map[index]
        traj = self.get_trajectory(idx)
        return {
            k: v[start_idx : start_idx + self._traj_length] for k, v in traj.items()
        }

    def eval_logs(
        self, model: Callable, tokenizer_manager: TokenizerManager
    ) -> Dict[str, Any]:
        state_shape = self._env.observation_space.shape
        action_shape = self._env.action_space.shape

        log_data = {}
        device = next(model.parameters()).device

        bc_sampler = lambda o, t: sample_action_bc(
            o,
            t,
            model,
            tokenizer_manager,
            (0,) + state_shape,
            (0,) + action_shape,
            device,
        )

        results, videos = evaluate(
            bc_sampler,
            self._env,
            20,
            state_shape,
            action_shape,
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
                    (0,) + state_shape,
                    (0,) + action_shape,
                    device,
                    percentage=p,
                )
                results, videos = evaluate(
                    bc_sampler,
                    self._env,
                    20,
                    state_shape,
                    action_shape,
                    num_videos=0,
                )
                for k, v in results.items():
                    log_data[f"eval2/p={p}_{k}"] = v
                for idx, v in enumerate(videos):
                    log_data[f"eval2_video_{idx}/p={p}_video"] = wandb.Video(
                        v.transpose(0, 3, 1, 2), fps=10, format="gif"
                    )

        if True:
            if "returns" in tokenizer_manager.tokenizers:
                for p in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]:
                    bc_sampler = lambda o, t: sample_action_bc_two_stage(
                        o,
                        t,
                        model,
                        tokenizer_manager,
                        (0,) + state_shape,
                        (0,) + action_shape,
                        device,
                        percentage=p,
                    )
                    results, videos = evaluate(
                        bc_sampler,
                        self._env,
                        20,
                        state_shape,
                        action_shape,
                        num_videos=0,
                    )
                    for k, v in results.items():
                        log_data[f"eval_ts/p={p}_{k}"] = v
                    for idx, v in enumerate(videos):
                        log_data[f"eval_ts_video_{idx}/p={p}_video"] = wandb.video(
                            v.transpose(0, 3, 1, 2), fps=10, format="gif"
                        )

        return log_data


def main():
    for t in [True, False]:
        for d in [0.99, 1, 1.5]:
            for e in ["relocate", "pen", "hammer", "door"]:
                for de in ["expert", "medium_replay", "full_replay"]:
                    train_dataset, val_dataset = get_datasets(
                        seq_steps=10,
                        env_name=e,
                        d_name=de,
                        discount=d,
                        use_achieved=t,
                    )

    print(train_dataset)


if __name__ == "__main__":
    main()
