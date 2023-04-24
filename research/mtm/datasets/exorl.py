"""
Dataset used for training a policy. Formed from a collection of
HDF5 files and wrapped into a PyTorch Dataset.
"""
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import tqdm
import wandb
from gym import spaces
from torch.utils.data import Dataset

from research.exorl import dmc
from research.exorl.replay_buffer import episode_len, load_episode, relable_episode
from research.mtm.datasets.base import DatasetProtocol, DataStatistics
from research.mtm.tokenizers.base import TokenizerManager
from research.utils.plot_utils import PlotHandler as ph


def get_datasets(
    seq_steps: bool,
    env_name: int,
    seed: str,
    replay_buffer_dir: str,
    train_max_size: int,
    val_max_size: int,
    num_workers: int,
    use_rewards: bool = False,
    use_qpos: bool = False,
    use_remove_vel: bool = False,
):
    env = dmc.make(env_name, seed=seed)
    if env_name.startswith("point_mass_maze"):
        domain = "point_mass_maze"
    else:
        domain = env_name.split("_", 1)[0]

    replay_train_dir = Path(replay_buffer_dir) / domain / "proto" / "buffer_updated"
    train_dataset = OfflineReplayBuffer(
        env,
        replay_train_dir,
        train_max_size,
        num_workers,
        0.99,
        domain,
        seq_steps,
        None,
        None,
        False,
        use_rewards=use_rewards,
        use_qpos=use_qpos,
        use_remove_vel=use_remove_vel,
    )
    val_dataset = OfflineReplayBuffer(
        env,
        replay_train_dir,
        val_max_size,
        num_workers,
        0.99,
        domain,
        seq_steps,
        None,
        None,
        False,
        use_rewards=use_rewards,
        use_qpos=use_qpos,
        use_remove_vel=use_remove_vel,
    )
    train_dataset._load()
    val_dataset._load()
    return train_dataset, val_dataset


class OfflineReplayBuffer(Dataset, DatasetProtocol):
    def __init__(
        self,
        env,
        replay_dir,
        max_size,
        num_workers,
        discount,
        domain,
        traj_length,
        mode=None,
        cfg=None,
        relabel=False,
        use_rewards: bool = False,
        use_qpos: bool = False,
        use_remove_vel: bool = False,
    ):
        print("in SINGLE replay buffer")
        self._env = env
        # self.env = env
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
        self._sample_trajs = []
        self._use_rewards = use_rewards
        self._use_qpos = use_qpos
        self._use_remove_vel = use_remove_vel
        if self._use_qpos and self._use_remove_vel:
            raise NotImplementedError

        action_spec = env.action_spec()
        self._action_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=action_spec.shape,
            dtype=np.float32,
        )

    def trajectory_statistics(self) -> Dict[str, DataStatistics]:
        # save_path = self._replay_dir / "statistics_11_11.pkl"
        save_path = self._replay_dir / "statistics.pkl"
        print(save_path)

        if save_path.exists():
            with open(save_path, "rb") as f:
                ret_dict = pickle.load(f)
                if self._use_qpos:
                    ret_dict["states"] = ret_dict["qpos"]
                if self._use_remove_vel:
                    new = DataStatistics(
                        ret_dict["states"].mean[:15],
                        ret_dict["states"].std[:15],
                        ret_dict["states"].min[:15],
                        ret_dict["states"].max[:15],
                    )
                    ret_dict["states"] = new
                return ret_dict

        eps_fns = sorted(
            self._replay_dir.rglob("*.npz")
        )  # get all episodes recursively
        keys = ["states", "actions", "rewards", "qpos"]
        stats = {key: [] for key in keys}
        print("saving to: ", save_path)
        for eps_fn in tqdm.tqdm(eps_fns):
            episode = load_episode(eps_fn)
            episode = self._relable_reward(episode)
            stats["states"].append(episode["observation"])
            stats["actions"].append(episode["action"])
            stats["rewards"].append(episode["reward"])
            stats["qpos"].append(episode["qpos"])

        for key in keys:
            stats[key] = np.concatenate(stats[key], axis=0)

        data_stats = {}
        for key in keys:
            data_stats[key] = DataStatistics(
                np.mean(stats[key], axis=0),
                np.std(stats[key], axis=0),
                np.min(stats[key], axis=0),
                np.max(stats[key], axis=0),
            )

        with open(save_path, "wb") as f:
            pickle.dump(data_stats, f)
        if self._use_qpos:
            data_stats["states"] = data_stats["qpos"]
        return data_stats

    def _load(self, relable=True):
        if self._loaded:
            return
        self._loaded = True
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
        print(self._replay_dir)
        print(len(list(self._replay_dir.rglob("*.npz"))))
        for eps_fn in eps_fns:
            if self._size > self._max_size:
                print("over size", self._max_size)
                break
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            episode = load_episode(eps_fn)
            if relable:
                episode = self._relable_reward(episode)
            self._episode_fns.append(eps_fn)
            self._episodes[eps_fn] = episode
            self._size += episode_len(episode)

    def __len__(self) -> int:
        return self._size

    def _sample_episode(self, episode_idx: Optional[int] = None):
        if episode_idx is None:
            eps_fn = random.choice(self._episode_fns)
        else:
            eps_fn = self._episode_fns[episode_idx % len(self._episode_fns)]
        return self._episodes[eps_fn]

    def _relable_reward(self, episode):
        return relable_episode(self._env, episode)

    def sample(self, episode_idx: Optional[int] = None, p_idx: Optional[int] = None):
        episode = self._sample_episode(episode_idx)
        # add +1 for the first dummy transition
        if p_idx is None:
            idx = np.random.randint(0, episode_len(episode) - self._traj_length + 1) + 1
        else:
            idx = p_idx  # choose first transition
        obs = episode["observation"][idx - 1 : idx - 1 + self._traj_length]
        qpos = episode["qpos"][idx - 1 : idx - 1 + self._traj_length]
        action = episode["action"][idx : idx + self._traj_length]
        next_obs = episode["observation"][idx : idx + self._traj_length]
        reward = episode["reward"][idx : idx + self._traj_length]
        discount = episode["discount"][idx : idx + self._traj_length] * self._discount
        timestep = np.arange(idx - 1, idx + self._traj_length - 1)[:, np.newaxis]
        physics = episode["physics"][idx - 1 : idx - 1 + self._traj_length]
        return {
            "observations": obs.astype(np.float32),
            "qpos": qpos.astype(np.float32),
            "actions": action.astype(np.float32),
            "rewards": reward.astype(np.float32).reshape(-1, 1),
            "discount": discount.astype(np.float32),
            "next_observations": next_obs.astype(np.float32),
            "timestep": 0,
            "physics": physics,
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
            content = self.sample()
        elif self._mode == "goal":
            content = self._sample_goal()
        elif self._mode == "multi_goal":
            content = self._sample_multiple_goal()
        elif self._mode == "prompt":
            content = self._sample_context()
        else:
            raise NotImplementedError

        if self._use_qpos:
            return_dict = {
                "states": content["qpos"],
                "actions": content["actions"],
            }
        else:
            if self._use_remove_vel:
                return_dict = {
                    "states": content["observations"][:, :15],
                    "actions": content["actions"],
                }
            else:
                return_dict = {
                    "states": content["observations"],
                    "actions": content["actions"],
                }

        if self._use_rewards:
            return_dict["rewards"] = content["rewards"]
        return return_dict

    def __iter__(self):
        while True:
            yield self._s()

    def __getitem__(self, idx: int):
        return self._s()

    def _step_env(self, a):
        """Thin wrapper to try things out with different observation spaces"""
        obs = self._env.step(a)
        if self._use_qpos:
            o = self._env.physics.data.qpos.copy()
        elif self._use_remove_vel:
            o = obs["observation"][:15]
        else:
            o = obs["observation"]
        return o, obs

    def eval_logs(
        self, model: Callable, tokenizer_manager: TokenizerManager
    ) -> Dict[str, Any]:
        num_samples = 10
        num_samples_to_plot = 3
        while len(self._sample_trajs) < num_samples:
            s = self.sample(len(self._sample_trajs))
            if self._use_qpos:
                s["observations"] = s["qpos"]
            if self._use_remove_vel:
                s["observations"] = s["observations"][:, :15]
            self._sample_trajs.append(s)

        open_loop_mse = 0

        _eval_logs = {}
        for b_idx in range(num_samples):
            _sample_traj = self._sample_trajs[b_idx]

            state0 = _sample_traj["physics"][0]
            self._env.reset()
            with self._env.physics.reset_context():
                self._env.physics.set_state(state0)

            device = next(model.parameters()).device
            batch_torch = {
                "states": torch.from_numpy(_sample_traj["observations"])
                .to(device)
                .unsqueeze(0),
                "actions": torch.from_numpy(_sample_traj["actions"])
                .to(device)
                .unsqueeze(0),
            }
            if self._use_rewards:
                batch_torch["rewards"] = torch.from_numpy(_sample_traj["rewards"])
                batch_torch["rewards"] = batch_torch["rewards"].to(device).unsqueeze(0)

            # goalreaching mask
            state_mask = torch.zeros(batch_torch["states"].shape[1])
            state_mask[:3] = 1
            state_mask[-3:] = 1
            action_mask = torch.zeros(batch_torch["actions"].shape[1])
            masks = {"states": state_mask, "actions": action_mask}
            if self._use_rewards:
                reward_mask = torch.zeros(batch_torch["rewards"].shape[1])
                masks["rewards"] = reward_mask
            masks_torch = {k: v.to(device) for k, v in masks.items()}

            encoded_batch = tokenizer_manager.encode(batch_torch)
            decoded_gt_trajectories = tokenizer_manager.decode(encoded_batch)
            predicted_trajectories = model(encoded_batch, masks_torch)
            decoded_trajectories = tokenizer_manager.decode(predicted_trajectories)

            images_open_loop = [self._env.physics.render(480, 640, 0)]
            execute_actions = (
                decoded_trajectories["actions"].squeeze(0).detach().cpu().numpy()
            )
            traj_real_ol = defaultdict(list)
            traj_real_ol["states"].append(_sample_traj["observations"][0])
            for idx, action in enumerate(execute_actions):
                traj_real_ol["actions"].append(action)
                o, obs = self._step_env(action)
                traj_real_ol["states"].append(o)
                traj_real_ol["rewards"].append([obs["reward"]])
                image = self._env.physics.render(480, 640, 0)
                images_open_loop.append(image)
            traj_real_ol["states"] = traj_real_ol["states"][:-1]
            images_open_loop = np.array(images_open_loop[:-1])

            goal_state = _sample_traj["observations"][-1]
            diff_ol = np.sum((goal_state - traj_real_ol["states"][-1]) ** 2)
            open_loop_mse += diff_ol
            if b_idx >= num_samples_to_plot:
                continue

            max_n_plots = 3

            # set env physics
            self._env.reset()
            with self._env.physics.reset_context():
                self._env.physics.set_state(state0)

            # rollout actions
            actions = _sample_traj["actions"]
            _obs = _sample_traj["observations"]
            images = [self._env.physics.render(480, 640, 0)]
            for idx, action in enumerate(actions):
                o, _ = self._step_env(action)
                image = self._env.physics.render(480, 640, 0)
                if idx < len(actions) - 1:
                    np.testing.assert_allclose(o, _obs[idx + 1], atol=1e-3, rtol=1e-3)
                images.append(image)
            images = np.array(images[:-1])

            _eval_logs.update(
                {
                    f"real_batch={b_idx}/dataset_execution": wandb.Video(
                        images.transpose(0, 3, 1, 2), fps=10, format="gif"
                    ),
                    f"real_batch={b_idx}/open_loop_execution": wandb.Video(
                        images_open_loop.transpose(0, 3, 1, 2), fps=10, format="gif"
                    ),
                    f"real_batch={b_idx}/open_loop_goal_error": diff_ol,
                    f"real_batch={b_idx}/start_img": wandb.Image(images[0]),
                    f"real_batch={b_idx}/goal_img": wandb.Image(images[-1]),
                    f"real_batch={b_idx}/open_loop_goal_img": wandb.Image(
                        images_open_loop[-1]
                    ),
                    f"real_batch={b_idx}/open_loop_image_diff": wandb.Image(
                        images[-1] - images_open_loop[-1]
                    ),
                }
            )

            for k, _ in decoded_trajectories.items():
                traj = batch_torch[k][0].detach().cpu().numpy()
                pred_traj = decoded_trajectories[k][0].detach().cpu().numpy()
                dec_gt_traj = decoded_gt_trajectories[k][0].detach().cpu().numpy()
                mask = masks[k]
                for i in range(min(max_n_plots, traj.shape[-1])):
                    gt_i = traj[:, i]
                    re_i = pred_traj[:, i]
                    dec_gt_i = dec_gt_traj[:, i]
                    real_i = np.array(traj_real_ol[k])[:, i]
                    if len(mask.shape) == 1:
                        # only along time dimension: repeat across the given dimension
                        mask = mask[:, None].repeat(1, traj.shape[1])
                    select_mask = mask[:, i].cpu().numpy()
                    unmasked_gt_i = gt_i[select_mask == 1]
                    unmasked_gt_i_index = np.arange(len(gt_i))[select_mask == 1]
                    vmax = max(np.max(gt_i), np.max(re_i))
                    vmin = min(np.min(gt_i), np.min(re_i))
                    y_range = vmax - vmin
                    with ph.plot_context() as (fig, ax):
                        ax.plot(gt_i, "-o", label="ground truth")
                        ax.plot(
                            re_i, "-o", label="reconstructed", markerfacecolor="none"
                        )
                        # blue color
                        ax.plot(
                            dec_gt_i,
                            "--o",
                            label="gt_reconstructed",
                            markerfacecolor="none",
                            color="blue",
                        )
                        ax.plot(
                            unmasked_gt_i_index,
                            unmasked_gt_i,
                            "o",
                            label="unmasked ground truth",
                        )
                        ax.plot(real_i, ".", label="real")
                        ax.set_ylim(
                            vmin - y_range / 5,
                            vmax + y_range / 5,
                        )
                        ax.legend()
                        ax.set_title(f"{k}_{i}")
                        _eval_logs[f"real_batch={b_idx}/{k}_{i}"] = wandb.Image(
                            ph.plot_as_image(fig)
                        )

        _eval_logs["val/open_loop_execution_mse"] = open_loop_mse / num_samples
        return _eval_logs


def main():
    env_name = "walker_walk"

    env = dmc.make(env_name, seed=0)
    if env_name.startswith("point_mass_maze"):
        domain = "point_mass_maze"
    else:
        domain = env_name.split("_", 1)[0]
    replay_buffer_dir = "~/exorl_data"
    replay_train_dir = Path(replay_buffer_dir) / domain / "proto" / "buffer_updated"

    sequence_length = 16
    for d in [0.99, 1, 1.5]:
        iterable = OfflineReplayBuffer(
            env,
            replay_train_dir,
            100,
            1,
            d,
            domain,
            sequence_length,
            None,
            None,
            False,
        )

    # instance = iterable[0]
    statistics = iterable.trajectory_statistics()
    # import ipdb
    #
    # ipdb.set_trace()
    # print()


if __name__ == "__main__":
    main()
