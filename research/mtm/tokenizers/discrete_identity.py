# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils.data import Dataset

from research.mtm.tokenizers.base import Tokenizer


class DiscreteIdentity(Tokenizer):
    """Dummy tokenizer for trajectories that are already discrete."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    @classmethod
    def create(
        cls, key: str, train_dataset: Dataset, num_classes: int
    ) -> "DiscreteIdentity":
        # add some slack
        return cls(num_classes)

    @property
    def discrete(self) -> bool:
        return True

    def encode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        trajectory = torch.nn.functional.one_hot(
            trajectory, num_classes=self.num_classes
        )
        assert trajectory.dim() == 3
        return trajectory.unsqueeze(2).to(torch.float32)

    def decode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        assert trajectory.dim() == 4
        assert trajectory.size(2) == 1
        # denormalize trajectory
        trajectory = trajectory.squeeze(2)
        trajectory = torch.argmax(trajectory, dim=-1)
        return trajectory
