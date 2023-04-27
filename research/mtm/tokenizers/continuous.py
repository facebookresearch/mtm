# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from numpy.typing import ArrayLike

from research.mtm.datasets.base import DatasetProtocol, DataStatistics
from research.mtm.tokenizers.base import Tokenizer


class ContinuousTokenizer(Tokenizer):
    def __init__(
        self,
        data_mean: ArrayLike,
        data_std: ArrayLike,
        stats: DataStatistics,
        normalize: bool = True,
    ):
        super().__init__()
        self._data_mean = torch.nn.Parameter(
            torch.tensor(data_mean, dtype=torch.float32), requires_grad=False
        )
        self._data_std = torch.nn.Parameter(
            torch.tensor(data_std, dtype=torch.float32), requires_grad=False
        )
        self.stats = stats
        self.normalize = normalize

    @classmethod
    def create(
        cls, key: str, train_dataset: DatasetProtocol, normalize: bool = True
    ) -> "ContinuousTokenizer":
        data = []
        stats = train_dataset.trajectory_statistics()[key]
        data_mean = stats.mean
        data_std = stats.std
        data_std[data_std < 0.1] = 1  # do not normalize if std is too small
        return cls(data_mean, data_std, stats, normalize=normalize)

    @property
    def discrete(self) -> bool:
        return False

    def encode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        assert trajectory.dim() == 3

        if self.normalize:
            mean = self._data_mean.to(trajectory.device)
            std = self._data_std.to(trajectory.device)
            # normalize trajectory
            trajectory = (trajectory - mean) / std
        return trajectory.unsqueeze(2).to(torch.float32)

    def decode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        assert trajectory.dim() == 4
        assert trajectory.size(2) == 1
        if self.normalize:
            mean = self._data_mean.to(trajectory.device)
            std = self._data_std.to(trajectory.device)

            # denormalize trajectory
            return trajectory.squeeze(2) * std + mean
        else:
            return trajectory.squeeze(2)
