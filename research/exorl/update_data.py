# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import warnings
from multiprocessing import cpu_count, current_process, get_context
from typing import Any, List, Literal

import tqdm
import tyro

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import dmc
import numpy as np
import utils
from replay_buffer import load_episode
from train_offline import get_domain


def relable_episode(env, episode):
    image_list = []
    joint_state_list = []
    reward_spec = env.reward_spec()
    states = episode["physics"]
    state0 = states[0]
    og_actions = episode["action"]

    env.reset()
    with env.physics.reset_context():
        env.physics.set_state(state0)
    obs = [np.hstack([o for o in env.task.get_observation(env.physics).values()])]
    physics = [env.physics.get_state()]
    new_actions = [og_actions[0]]
    rewards = [episode["reward"][0]]
    discount = [episode["discount"][0]]
    qpos = [env.physics.data.qpos.copy()]

    size = 84
    img = env.physics.render(camera_id=0, height=size, width=size)
    image_list = [img]

    for action in og_actions[1:]:
        time_step = env.step(action)
        qpos.append(env.physics.data.qpos.copy())
        obs.append(time_step["observation"])
        physics.append(env.physics.get_state())
        new_actions.append(action)
        rewards.append(time_step.reward)
        discount.append(time_step.discount)

        img = env.physics.render(camera_id=0, height=size, width=size)
        image_list.append(img)
    episode = {
        "observation": np.array(obs),
        "qpos": np.array(qpos),
        "physics": np.array(physics),
        "action": np.array(new_actions),
        "reward": np.array(rewards),
        "discount": np.array(discount),
        "image": np.stack(image_list),
    }
    return episode


def main(
    seed: int = 0,
    num_workers: int = 0,
    env_name: Literal[
        "walker",
        "cartpole",
        "cheetah",
        "jaco",
        "point_mass_maze",
        "quadruped",
    ] = "walker",
    expl_agent: Literal[
        "aps",
        "icm_apt",
        "diayn",
        "disagreement",
        "icm",
        "proto",
        "random",
        "rnd",
        "smm",
    ] = "proto",
    replay_buffer_dir: str = "datasets",
):
    if env_name == "walker":
        task = "walker_stand"
    elif env_name == "cartpole":
        task = "cartpole_balance"
    elif env_name == "cheetah":
        task = "cheetah_run"
    elif env_name == "jaco":
        task = "jaco_reach_top_left"
    elif env_name == "point_mass_maze":
        task = "point_mass_maze_reach_top_left"
    elif env_name == "quadruped":
        task = "quadruped_run"

    work_dir = Path.cwd()
    utils.set_seed_everywhere(seed)

    # create envs

    # create data storage
    domain = get_domain(task)
    datasets_dir = work_dir / replay_buffer_dir
    replay_dir = datasets_dir.resolve() / domain / expl_agent / "buffer"
    replay_img_dir = datasets_dir.resolve() / domain / expl_agent / "buffer_updated"
    os.makedirs(replay_img_dir, exist_ok=True)
    print(f"replay dir: {replay_dir}")

    eps_fns = sorted(replay_dir.glob("*.npz"))

    print(f"using {num_workers} workers out of {cpu_count()}")
    if num_workers == 0:
        save_new_files(eps_fns, seed, task, replay_img_dir, env_name)
    else:
        ctx = get_context("spawn")
        list_eps = np.array_split(np.array(eps_fns), num_workers)
        assert len(list_eps) == num_workers
        with ctx.Pool(num_workers) as pool:
            f = functools.partial(
                save_new_files,
                seed=seed,
                task=task,
                replay_img_dir=replay_img_dir,
                env_name=env_name,
            )
            pool.map(f, list_eps)


def save_new_files(
    eps_fns: List[Any],
    seed: int,
    task: str,
    replay_img_dir: Path,
    env_name: str,
):
    env = dmc.make(task, seed=seed)

    try:
        disable = current_process()._identity[0] != 1
    except:
        disable = False

    for eps_fn in tqdm.tqdm(eps_fns, disable=disable):
        eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
        episode = load_episode(eps_fn)
        episode = relable_episode(env, episode)
        file_name = eps_fn.name
        np.savez(replay_img_dir / file_name, **episode)


if __name__ == "__main__":
    tyro.cli(main)
