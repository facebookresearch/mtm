# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from research.mtm.tokenizers.continuous_binned import ContinuousBinnedTokenizer


def test_binning_simple():
    # generate random data
    rnd = np.random.RandomState(0)

    X = [
        [0, 0, 0.1],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0],
    ]
    X = np.array(X)[None]
    tokenizer = ContinuousBinnedTokenizer([0, 0.1, 0.2])

    test_data = torch.tensor(X)
    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens)

    np.testing.assert_allclose(recon, test_data)


def test_binning_simple_with_others():
    # generate random data
    rnd = np.random.RandomState(0)

    X = [
        [0, 0, 0.1],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0],
    ]
    X = np.array(X)[None]
    tokenizer = ContinuousBinnedTokenizer([-0.1, 0, 0.1, 0.2, 0.4, 0.5])

    test_data = torch.tensor(X)
    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens)

    np.testing.assert_allclose(recon, test_data)


def test_binning_simple_logits():
    # generate random data
    rnd = np.random.RandomState(0)

    X = [
        [0, 0, 0.1],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0],
    ]
    X = np.array(X)[None]
    tokenizer = ContinuousBinnedTokenizer([-0.1, 0, 0.1, 0.2, 0.4, 0.5])

    test_data = torch.tensor(X)
    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens * 100)

    np.testing.assert_allclose(recon, test_data)
