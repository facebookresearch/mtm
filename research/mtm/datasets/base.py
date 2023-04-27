# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Callable, Dict, Protocol

import numpy as np

from research.mtm.tokenizers.base import TokenizerManager


@dataclass
class DataStatistics:
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray

    def __post_init__(self):
        self.mean = np.array(self.mean, dtype=np.float32)
        self.std = np.array(self.std, dtype=np.float32)
        self.min = np.array(self.min, dtype=np.float32)
        self.max = np.array(self.max, dtype=np.float32)

        # check shapes
        assert self.mean.shape == self.std.shape == self.min.shape == self.max.shape

        # check ordering
        assert np.all(self.min <= self.max)


class DatasetProtocol(Protocol):
    @property
    def trajectory_statistics(self) -> Dict[str, DataStatistics]:
        ...
        """Shapes of the trajectories in the dataset."""

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Returns the observation, action, and return for the given index.

        Args:
            idx: The index of the data point to return.

        Returns:
            trajectories: A dictionary of trajectories.
        """

    def eval_logs(
        self, model: Callable, tokenizer_manager: TokenizerManager
    ) -> Dict[str, Any]:
        """Returns the evaluation logs for the given model.

        Args:
            model: The model to evaluate.

        Returns:
            logs: A dictionary of evaluation logs.
        """
