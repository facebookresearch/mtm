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

# TODO add other environments
env_joints_map = {
    "walker": [
        "right_hip",
        "right_knee",
        "right_ankle",
        "left_hip",
        "left_knee",
        "left_ankle",
    ]
}


def relable_episode(env, episode, env_name, size=84):
    image_list = []
    joint_state_list = []
    reward_spec = env.reward_spec()
    states = episode["physics"]

    for i in range(states.shape[0]):
        with env.physics.reset_context():
            env.physics.set_state(states[i])
        img = env.physics.render(camera_id=0, height=size, width=size)
        image_list.append(img)

        joint_state = env.physics.named.data.qpos[env_joints_map[env_name]]
        joint_state_list.append(joint_state)

    episode["image"] = np.stack(image_list)
    episode["joint_state"] = np.stack(joint_state_list)
    return episode


def main(
    seed: int = 0,
    num_workers: int = 0,
    env_name: Literal[
        "walker",
        # TODO -  only walker is supported now: see other TODO at the top of the page
        # "cartpole",
        # "cheetah",
        # "jaco",
        # "point_mass_maze",
        # "quadruped",
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
    size: int = 84,
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
    replay_img_dir = datasets_dir.resolve() / domain / expl_agent / "buffer_img"
    os.makedirs(replay_img_dir, exist_ok=True)
    print(f"replay dir: {replay_dir}")

    eps_fns = sorted(replay_dir.glob("*.npz"))

    print(f"using {num_workers} workers out of {cpu_count()}")
    if num_workers == 0:
        save_new_files(eps_fns, seed, task, replay_img_dir, env_name, size)
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
                size=size,
            )
            pool.map(f, list_eps)


def save_new_files(
    eps_fns: List[Any],
    seed: int,
    task: str,
    replay_img_dir: Path,
    env_name: str,
    size: int = 84,
):
    env = dmc.make(task, seed=seed)

    try:
        disable = current_process()._identity[0] != 1
    except:
        disable = False

    for eps_fn in tqdm.tqdm(eps_fns, disable=disable):
        eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
        episode = load_episode(eps_fn)
        episode = relable_episode(env, episode, env_name=env_name, size=size)
        file_name = eps_fn.name
        np.savez(replay_img_dir / file_name, **episode)


if __name__ == "__main__":
    tyro.cli(main)
