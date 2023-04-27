# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from research.mtm.datasets.base import DataStatistics
from research.mtm.tokenizers.base import Tokenizer


class UniformBinningTokenizer(Tokenizer):
    def __init__(
        self,
        num_bins: int,
        data_min: np.array,
        data_max: np.array,
        stats: DataStatistics,
    ):
        super().__init__()
        self._num_bins = num_bins  # dummy example argument
        self._encode_min = nn.Parameter(
            torch.tensor(data_min, dtype=torch.float32), requires_grad=False
        )
        self._encode_max = nn.Parameter(
            torch.tensor(data_max, dtype=torch.float32), requires_grad=False
        )
        self.stats = stats

    @classmethod
    def create(
        cls,
        key: str,
        train_dataset: Dataset,
        num_bins: int,
    ) -> "UniformBinningTokenizer":
        # add some slack
        stats = train_dataset.trajectory_statistics()[key]
        data_min = stats.min
        data_max = stats.max
        data_range = data_max - data_min

        # data_min -= data_range
        # data_max += data_range

        print(f"{key}: min = {data_min}")
        print(f"{key}: max = {data_max}")
        return cls(num_bins, data_min, data_max, stats)

    @property
    def discrete(self) -> bool:
        return True

    def encode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        assert trajectory.dim() == 3
        dmin = self._encode_min.to(trajectory.device)
        dmax = self._encode_max.to(trajectory.device)
        diff = dmax - dmin
        diff[diff < 0.1] = 1  # do not normalize if std is too small

        tokenized_trajectory = torch.floor((trajectory - dmin) / diff * self._num_bins)
        tokenized_trajectory = torch.clamp(tokenized_trajectory, 0, self._num_bins - 1)
        one_hot_trajectory = torch.nn.functional.one_hot(
            tokenized_trajectory.long(), self._num_bins
        )
        return one_hot_trajectory.float()

    def decode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        assert trajectory.dim() == 4
        decoded_trajectory = torch.argmax(trajectory, dim=3).float()
        dmin = self._encode_min.to(trajectory.device)
        dmax = self._encode_max.to(trajectory.device)
        decoded_trajectory = decoded_trajectory / self._num_bins * (dmax - dmin) + dmin

        decoded_trajectory = (
            decoded_trajectory + (dmax - dmin) / self._num_bins / 2
        )  # add half bin width to get center of bin
        return decoded_trajectory
