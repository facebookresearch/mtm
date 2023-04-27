# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from numpy.typing import ArrayLike

from research.mtm.datasets.base import DatasetProtocol, DataStatistics
from research.mtm.tokenizers.base import Tokenizer


class SplitContinuousTokenizer(Tokenizer):
    def __init__(
        self,
        data_mean: ArrayLike,
        data_std: ArrayLike,
        stats: DataStatistics,
        normalize: bool = True,
        splits: int = 4,
    ):
        super().__init__()
        self._data_mean = torch.nn.Parameter(
            torch.tensor(data_mean, dtype=torch.float32), requires_grad=False
        )
        self._data_std = torch.nn.Parameter(
            torch.tensor(data_std, dtype=torch.float32), requires_grad=False
        )
        self.split = splits
        self.stats = stats
        self.normalize = normalize

    @classmethod
    def create(
        cls,
        key: str,
        train_dataset: DatasetProtocol,
        normalize: bool = True,
        splits: int = 4,
    ) -> "ContinuousTokenizer":
        data = []
        stats = train_dataset.trajectory_statistics()[key]
        data_mean = stats.mean
        data_std = stats.std
        data_std[data_std < 0.1] = 1  # do not normalize if std is too small
        return cls(data_mean, data_std, stats, normalize=normalize, splits=splits)

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

        # split trajectory into multiple channels
        B, L, D = trajectory.shape

        # zero_padded
        padded_length = (D // self.split + 1) * self.split
        zero_padded = torch.zeros(B, L, padded_length, device=trajectory.device)
        zero_padded[:, :, :D] = trajectory

        # split
        split = zero_padded.reshape(B, L, self.split, -1)
        return split

    def decode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        assert trajectory.dim() == 4
        assert trajectory.size(2) == self.split

        # merge
        B, L, _, _ = trajectory.shape
        merged = trajectory.reshape(B, L, -1)

        D = self._data_mean.shape[-1]
        if self.normalize:
            mean = self._data_mean.to(trajectory.device)
            std = self._data_std.to(trajectory.device)

            # denormalize trajectory
            return merged.squeeze(2)[..., :D] * std + mean
        else:
            return merged.squeeze(2)[..., :D]
