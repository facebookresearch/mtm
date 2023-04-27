# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from research.mtm.tokenizers.uniform_bins import UniformBinningTokenizer


def test_binning_simple():
    # generate random data
    rnd = np.random.RandomState(0)

    X = rnd.rand(1, 2, 3)
    data_min = np.zeros(X.shape[2:])
    data_max = np.ones(X.shape[2:])
    num_bins = 2

    max_diffs = torch.tensor((data_max - data_min) / num_bins / 2)
    tokenizer = UniformBinningTokenizer(
        num_bins, torch.zeros(data_min.shape), torch.ones(data_max.shape), None
    )
    test_data = torch.tensor(X)

    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens)

    ## make sure reconstruction error is less than the max allowed per dimension
    error_np = torch.abs(test_data - recon).numpy()
    error_np = np.max(error_np, (0, 1))
    max_diffs = max_diffs.numpy()
    np.testing.assert_array_less(error_np, max_diffs)

    ## re-discretize reconstruction and make sure it is the same as original indices
    tokens_2 = tokenizer.encode(recon)
    np.testing.assert_allclose(tokens.cpu().numpy(), tokens_2.cpu().numpy())


def test_binning_simple_big():
    # generate random data
    rnd = np.random.RandomState(0)

    X = rnd.rand(10000, 2, 3)
    data_min = np.zeros(X.shape[2:])
    data_max = np.ones(X.shape[2:])
    num_bins = 10

    max_diffs = torch.tensor((data_max - data_min) / num_bins / 2)
    tokenizer = UniformBinningTokenizer(
        num_bins, torch.zeros(data_min.shape), torch.ones(data_max.shape), None
    )
    test_data = torch.tensor(X)

    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens)

    ## make sure reconstruction error is less than the max allowed per dimension
    error_np = torch.abs(test_data - recon).numpy()
    error_np = np.max(error_np, (0, 1))
    max_diffs = max_diffs.numpy()
    np.testing.assert_array_less(error_np, max_diffs)

    ## re-discretize reconstruction and make sure it is the same as original indices
    tokens_2 = tokenizer.encode(recon)
    np.testing.assert_allclose(tokens.cpu().numpy(), tokens_2.cpu().numpy())


def test_binning_rnd():
    # generate random data
    rnd = np.random.RandomState(0)

    X = rnd.randn(1, 2, 3)
    data_min = np.min(X, axis=(0, 1))
    data_max = np.max(X, axis=(0, 1))
    num_bins = 2

    max_diffs = torch.tensor((data_max - data_min) / num_bins / 2)
    tokenizer = UniformBinningTokenizer(
        num_bins, torch.tensor(data_min), torch.tensor(data_max), None
    )
    test_data = torch.tensor(X)

    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens)

    ## make sure reconstruction error is less than the max allowed per dimension
    error_np = torch.abs(test_data - recon).numpy()
    error_np = np.max(error_np, (0, 1))
    max_diffs = max_diffs.numpy()
    np.testing.assert_array_less(error_np, max_diffs + 1e-6)

    ## re-discretize reconstruction and make sure it is the same as original indices
    tokens_2 = tokenizer.encode(recon)
    np.testing.assert_allclose(tokens.cpu().numpy(), tokens_2.cpu().numpy())


def test_binning_large():
    # generate random data
    rnd = np.random.RandomState(0)

    X = rnd.randn(1024, 8, 5)
    data_min = np.min(X, axis=(0, 1))
    data_max = np.max(X, axis=(0, 1))
    num_bins = 11

    max_diffs = torch.tensor((data_max - data_min) / num_bins / 2)
    tokenizer = UniformBinningTokenizer(
        num_bins, torch.tensor(data_min), torch.tensor(data_max), None
    )
    test_data = torch.tensor(X)

    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens)

    ## make sure reconstruction error is less than the max allowed per dimension
    error_np = torch.abs(test_data - recon).numpy()
    error_np = np.max(error_np, (0, 1))
    max_diffs = max_diffs.numpy()
    np.testing.assert_array_less(error_np, max_diffs + 1e-6)

    ## re-discretize reconstruction and make sure it is the same as original indices
    tokens_2 = tokenizer.encode(recon)
    np.testing.assert_allclose(tokens.cpu().numpy(), tokens_2.cpu().numpy())
