# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset used for training a policy. Formed from a collection of
HDF5 files and wrapped into a PyTorch Dataset.
"""

from typing import Any, Callable, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from research.mtm.datasets.base import DatasetProtocol, DataStatistics
from research.mtm.tokenizers.base import TokenizerManager


def get_datasets(
    seq_steps: bool,
    noise: float,
    train_dataset_size: int,
    val_dataset_size: int,
):
    noise = 0.03
    train_dataset = SinusoidDataset(seq_steps, noise, train_dataset_size)
    val_dataset = SinusoidDataset(seq_steps, noise, val_dataset_size)
    return train_dataset, val_dataset


def _get_data(dataset_size, traj_length, noise, multimodal_data=True):
    """synthetic sinusoidal data"""
    amp = 1.0
    phase = torch.where(torch.rand(dataset_size, 1, 1) > 0.5, 0, np.pi)
    freq = torch.ones(dataset_size, traj_length, 1)
    if multimodal_data:
        # Randomly invert the second half of half of the trajectories
        half0, half1 = freq[:, : traj_length // 2, :], freq[:, traj_length // 2 :, :]
        do_flip = torch.rand(dataset_size, 1, 1) > 0.5
        half1 = torch.where(do_flip, half1, -half1)
        freq = torch.concat([half0, half1], dim=1)
    noise_values = noise * torch.randn(dataset_size, traj_length, 1)
    t = torch.linspace(0, 2 * np.pi, traj_length).reshape(1, traj_length, 1)
    states = torch.clamp(amp * torch.sin(freq * t + phase) + noise_values, -1.0, 1.0)
    return states


class SinusoidDataset(Dataset, DatasetProtocol):
    def __init__(
        self,
        traj_length: int,
        noise: float,
        dataset_size: int,
    ):
        self._size = dataset_size
        self._data = _get_data(dataset_size, traj_length, noise)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int):
        return dict(states=self._data[idx].numpy())

    def eval_logs(
        self, model: Callable, tokenizer_manager: TokenizerManager
    ) -> Dict[str, Any]:
        return {}

    def trajectory_statistics(self) -> Dict[str, DataStatistics]:
        return {
            "states": DataStatistics(
                mean=self._data.mean().numpy(),
                std=self._data.std().numpy(),
                min=self._data.min().numpy(),
                max=self._data.max().numpy(),
            )
        }
