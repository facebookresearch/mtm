import torch

from research.mtm.models.mtm_model import MaskedDP, MTMConfig


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

    model = MaskedDP(
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

    # masks = {
    #     "actions": torch.ones(100, 3),
    #     "states": torch.ones(100, 1),
    # }
    out_trajs = model(trajectories_torch, masks)
    for k, v in trajectories_torch.items():
        assert v.shape == out_trajs[k].shape


# def test_maskdp_model_complex():
#     # smoke test to check that the model can run
#     features_dim = 13
#     action_dim = 7
#     n_layer = 1
#     dropout = 0.0
#     traj_length = 9
#
#     model = MaskedDP(
#         features_dim,
#         action_dim,
#         MTMConfig(
#             n_embd=128,
#             n_head=2,
#             n_enc_layer=n_layer,
#             n_dec_layer=n_layer,
#             dropout=dropout,
#             traj_length=traj_length,
#         ),
#     )
#     obs = torch.randn(5, traj_length, features_dim)
#     action = torch.randn(5, traj_length, action_dim)
#     mask1 = torch.ones(traj_length)
#     mask1[1:-1] = 0
#     mask2 = torch.zeros(traj_length)
#     model(obs, action, mask1, mask2)
#
#
# def test_maskdp_model_forward():
#     # smoke test to check that the model can run
#     features_dim = 13
#     action_dim = 7
#     n_layer = 1
#     dropout = 0.0
#     traj_length = 9
#
#     model = MaskedDP(
#         features_dim,
#         action_dim,
#         MTMConfig(
#             n_embd=128,
#             n_head=2,
#             n_enc_layer=n_layer,
#             n_dec_layer=n_layer,
#             dropout=dropout,
#             traj_length=traj_length,
#         ),
#     )
#
#     mask1 = torch.ones(traj_length)
#     mask1[1:-1] = 0
#     x, ids_restore, keep_len = model._index(mask1[None, :, None], mask1)
#
#     assert ids_restore.shape == (traj_length,)
#     assert keep_len == 2
#
#     assert x.sum() == 2
#     assert (x == torch.tensor([1, 1])).all()
#
#     new_tokens = torch.concat([x[0, :, 0], torch.zeros(traj_length - keep_len)])
#     assert (new_tokens[ids_restore] == mask1).all()
#
#
# def test_maskpd_policy():
#     # smoke test to check that the poilcy can run
#     # TODO
#     pass
