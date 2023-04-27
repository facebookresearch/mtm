# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from research.mtm.tokenizers.patchify import PatchifyTokenizer


def test_patchify_simple():
    # generate random data
    rnd = np.random.RandomState(0)
    H = 64
    W = 64
    X = np.floor(rnd.rand(1, 2, H, W, 3) * 256)
    tokenizer = PatchifyTokenizer(patch_size=16)
    test_data = torch.tensor(X)
    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens)

    ## make sure reconstruction error is less than the max allowed per dimension
    np.testing.assert_allclose(test_data.cpu().numpy(), recon.cpu().numpy())

    ## re-discretize reconstruction and make sure it is the same as original indices
    tokens_2 = tokenizer.encode(recon)
    np.testing.assert_allclose(tokens.cpu().numpy(), tokens_2.cpu().numpy())
