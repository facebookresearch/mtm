# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from research.mtm.models.mtm_model import MTM, MTMConfig


def test_maskdp_model_simple():
    # smoke test to check that the model can run
    features_dim = 13
    action_dim = 7
    n_layer = 1
    dropout = 0.0
    traj_length = 20

    data_shapes = {
        "actions": (3, action_dim),
        "states": (1, features_dim),
    }

    model = MTM(
        data_shapes,
        traj_length,
        MTMConfig(
            n_embd=128,
            n_head=2,
            n_enc_layer=n_layer,
            n_dec_layer=n_layer,
            dropout=dropout,
        ),
    )

    trajectories_torch = {
        "actions": torch.randn(5, traj_length, *data_shapes["actions"]),
        "states": torch.randn(5, traj_length, *data_shapes["states"]),
    }
    masks = {
        "actions": torch.ones(traj_length, 3),
        "states": torch.ones(traj_length, 1),
    }

    out_trajs = model(trajectories_torch, masks)
    for k, v in trajectories_torch.items():
        assert v.shape == out_trajs[k].shape
