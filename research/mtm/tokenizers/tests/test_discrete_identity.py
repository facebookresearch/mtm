# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from research.mtm.tokenizers.discrete_identity import DiscreteIdentity


def test_di_simple():
    # generate random data
    tokenizer = DiscreteIdentity(4)
    test_data = torch.tensor([[0, 2, 3]])

    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens)

    ## make sure reconstruction error is less than the max allowed per dimension
    np.testing.assert_allclose(recon, test_data)

    # tokens, shape=(1, 3, 1, 4)
    test_noise = torch.tensor([[[[1, 0, 0, 0]], [[0, 0, 1, 0]], [[0, 0, 0, 1]]]])
    recon = tokenizer.decode(test_noise)
    np.testing.assert_allclose(recon, test_data)

    # tokens, shape=(1, 3, 1, 4)
    test_noise = torch.tensor(
        [[[[0.9, 0.1, 0.1, 0.1]], [[-0.1, 0.1, 1.1, 0.1]], [[-0.1, -10, -2, 0.1]]]]
    )
    recon = tokenizer.decode(test_noise)
    np.testing.assert_allclose(recon, test_data)
