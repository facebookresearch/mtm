# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from research.jaxrl.datasets.d4rl_dataset import D4RLDataset
from research.jaxrl.utils import make_env
from research.mtm.datasets.sequence_dataset import SequenceDataset


def get_datasets(
    seq_steps: bool,
    env_name: int,
    seed: int = 0,
    use_reward: bool = True,
    discount: int = 1.5,
    train_val_split: float = 0.95,
):
    env = make_env(env_name, seed)
    d4rl_dataset = D4RLDataset(env)
    train_d, val_d = d4rl_dataset.train_validation_split(train_val_split)

    # hack to add env to class for eval
    train_d.env = env
    val_d.env = env

    train_dataset = SequenceDataset(
        train_d,
        discount=discount,
        sequence_length=seq_steps,
        use_reward=use_reward,
        name=env_name,
    )
    val_dataset = SequenceDataset(
        val_d,
        discount=discount,
        sequence_length=seq_steps,
        use_reward=use_reward,
        name=env_name,
    )
    return train_dataset, val_dataset


def main():
    env_names = [
        "hopper-medium-v2",
        "hopper-medium-replay-v2",
        "hopper-medium-expert-v2",
        "hopper-expert-v2",
        "walker2d-medium-v2",
        "walker2d-medium-replay-v2",
        "walker2d-medium-expert-v2",
        "walker2d-expert-v2",
        "halfcheetah-medium-v2",
        "halfcheetah-medium-replay-v2",
        "halfcheetah-medium-expert-v2",
        "halfcheetah-expert-v2",
    ]
    for d in [0.99, 1, 1.5]:
        for e in env_names:
            train_dataset, val_dataset = get_datasets(32, e, discount=d)
            train_dataset.trajectory_statistics()


if __name__ == "__main__":
    main()
