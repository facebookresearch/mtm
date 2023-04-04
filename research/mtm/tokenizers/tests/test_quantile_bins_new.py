import numpy as np
import torch

from research.mtm.tokenizers.quantile_bins import QuantileBinningTokenizer


def test_binning_simple():
    # generate random data
    rnd = np.random.RandomState(0)

    X = rnd.rand(4, 2, 3)
    data_min = np.zeros(X.shape[2:])
    data_max = np.ones(X.shape[2:])
    num_bins = 2

    max_diffs = torch.tensor((data_max - data_min) / num_bins / 2)
    tokenizer = QuantileBinningTokenizer(
        num_bins,
        torch.zeros(data_min.shape),
        torch.ones(data_max.shape),
        X.reshape(-1, 3),
        None,
    )
    test_data = torch.tensor(X)

    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens)

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
    tokenizer = QuantileBinningTokenizer(
        num_bins,
        torch.zeros(data_min.shape),
        torch.ones(data_max.shape),
        X.reshape(-1, 3),
        None,
    )
    test_data = torch.tensor(X)

    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens)

    ## re-discretize reconstruction and make sure it is the same as original indices
    tokens_2 = tokenizer.encode(recon)
    np.testing.assert_allclose(tokens.cpu().numpy(), tokens_2.cpu().numpy())


def test_binning_rnd():
    # generate random data
    rnd = np.random.RandomState(0)

    X = rnd.randn(4, 2, 3)
    data_min = np.min(X, axis=(0, 1))
    data_max = np.max(X, axis=(0, 1))
    num_bins = 2

    max_diffs = torch.tensor((data_max - data_min) / num_bins / 2)
    tokenizer = QuantileBinningTokenizer(
        num_bins, torch.tensor(data_min), torch.tensor(data_max), X.reshape(-1, 3), None
    )
    test_data = torch.tensor(X)

    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens)

    ## re-discretize reconstruction and make sure it is the same as original indices
    tokens_2 = tokenizer.encode(recon)
    np.testing.assert_allclose(tokens.cpu().numpy(), tokens_2.cpu().numpy())
