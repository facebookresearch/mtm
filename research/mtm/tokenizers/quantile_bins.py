import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from research.mtm.datasets.base import DataStatistics
from research.mtm.tokenizers.base import Tokenizer


def to_np(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


def largest_nonzero_index(x, dim):
    N = x.shape[dim]
    arange = np.arange(N) + 1

    for i in range(dim):
        arange = np.expand_dims(arange, axis=0)
    for i in range(dim + 1, x.ndim):
        arange = np.expand_dims(arange, axis=-1)

    inds = np.argmax(x * arange, axis=0)
    ## masks for all `False` or all `True`
    lt_mask = (~x).all(axis=0)
    gt_mask = (x).all(axis=0)

    inds[lt_mask] = 0
    inds[gt_mask] = N

    return inds


class QuantileBinningTokenizer(Tokenizer):
    def __init__(
        self,
        num_bins: int,
        data_min: np.array,
        data_max: np.array,
        raw_data: np.array,  # shape=(N, X)
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
        self.tokenizer = QuantileDiscretizer(
            raw_data,
            num_bins,
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
        return cls(num_bins, data_min, data_max, train_dataset.raw_data[key], stats)

    @property
    def discrete(self) -> bool:
        return True

    def encode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        assert trajectory.dim() == 3
        B, T, X = trajectory.shape
        reshape = trajectory.reshape(B * T, X)
        reshape = to_np(reshape)
        encoded_trajectory = self.tokenizer.discretize(reshape)
        encoded_trajectory = torch.tensor(
            encoded_trajectory, device=trajectory.device
        ).reshape(B, T, X)

        one_hot_trajectory = torch.nn.functional.one_hot(
            encoded_trajectory.long(), self._num_bins
        )
        return one_hot_trajectory.float()

    def decode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        assert trajectory.dim() == 4
        B, T, X, _num_bins = trajectory.shape
        assert _num_bins == self._num_bins
        arg_max_index = np.argmax(to_np(trajectory), axis=-1)
        reshape = arg_max_index.reshape(B * T, X)
        decoded_trajectory = self.tokenizer.reconstruct(reshape)
        decoded_trajectory = torch.tensor(
            decoded_trajectory, dtype=torch.float32, device=trajectory.device
        )
        decoded_trajectory = decoded_trajectory.reshape(B, T, X)
        return decoded_trajectory


class QuantileDiscretizer:
    def __init__(self, data, N):
        self.data = data
        self.N = N

        n_points_per_bin = int(np.ceil(len(data) / N))
        obs_sorted = np.sort(data, axis=0)
        thresholds = obs_sorted[::n_points_per_bin, :]
        maxs = data.max(axis=0, keepdims=True)

        ## [ (N + 1) x dim ]
        self.thresholds = np.concatenate([thresholds, maxs], axis=0)

        # threshold_inds = np.linspace(0, len(data) - 1, N + 1, dtype=int)
        # obs_sorted = np.sort(data, axis=0)

        # ## [ (N + 1) x dim ]
        # self.thresholds = obs_sorted[threshold_inds, :]

        ## [ N x dim ]
        self.diffs = self.thresholds[1:] - self.thresholds[:-1]

        ## for sparse reward tasks
        # if (self.diffs[:,-1] == 0).any():
        # 	raise RuntimeError('rebin for sparse reward tasks')

        self._test()

    def __call__(self, x):
        indices = self.discretize(x)
        recon = self.reconstruct(indices)
        error = np.abs(recon - x).max(0)
        return indices, recon, error

    def _test(self):
        print("[ utils/discretization ] Testing...", end=" ", flush=True)
        inds = np.random.randint(0, len(self.data), size=1000)
        X = self.data[inds]
        indices = self.discretize(X)
        recon = self.reconstruct(indices)
        ## make sure reconstruction error is less than the max allowed per dimension
        error = np.abs(X - recon).max(0)
        assert (error <= self.diffs.max(axis=0)).all()
        ## re-discretize reconstruction and make sure it is the same as original indices
        indices_2 = self.discretize(recon)
        assert (indices == indices_2).all()
        ## reconstruct random indices
        ## @TODO: remove duplicate thresholds
        # randint = np.random.randint(0, self.N, indices.shape)
        # randint_2 = self.discretize(self.reconstruct(randint))
        # assert (randint == randint_2).all()
        print("✓")

    def discretize(self, x, subslice=(None, None)):
        """
        x : [ B x observation_dim ]
        """

        ## enforce batch mode
        if x.ndim == 1:
            x = x[None]

        ## [ N x B x observation_dim ]
        start, end = subslice
        thresholds = self.thresholds[:, start:end]

        gt = x[None] >= thresholds[:, None]
        indices = largest_nonzero_index(gt, dim=0)

        if indices.min() < 0 or indices.max() >= self.N:
            indices = np.clip(indices, 0, self.N - 1)

        return indices

    def reconstruct(self, indices, subslice=(None, None)):
        if torch.is_tensor(indices):
            indices = to_np(indices)

        ## enforce batch mode
        if indices.ndim == 1:
            indices = indices[None]

        if indices.min() < 0 or indices.max() >= self.N:
            print(
                f"[ utils/discretization ] indices out of range: ({indices.min()}, {indices.max()}) | N: {self.N}"
            )
            indices = np.clip(indices, 0, self.N - 1)

        start, end = subslice
        thresholds = self.thresholds[:, start:end]

        left = np.take_along_axis(thresholds, indices, axis=0)
        right = np.take_along_axis(thresholds, indices + 1, axis=0)
        recon = (left + right) / 2.0
        return recon
