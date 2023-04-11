import copy
import os
from typing import Dict, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from omegaconf import OmegaConf
from torch.utils.data.dataloader import DataLoader

from research.mtm.models.mtm_model import MTM
from research.mtm.tokenizers.base import Tokenizer, TokenizerManager


def get_mtm_model(
    path: str,
) -> Tuple[MTM, TokenizerManager, Dict[str, Tuple[int, int]]]:
    def _get_dataset(dataset, traj_length):
        return hydra.utils.call(dataset, seq_steps=traj_length)

    # find checkpoints in the directory
    steps = []
    names = []
    paths_ = os.listdir(path)
    for name in [os.path.join(path, n) for n in paths_ if "pt" in n]:
        step = os.path.basename(name).split("_")[-1].split(".")[0]
        steps.append(int(step))
        names.append(name)
    ckpt_path = names[np.argmax(steps)]

    hydra_cfg = OmegaConf.load(os.path.join(path, ".hydra/config.yaml"))
    cfg = hydra.utils.instantiate(hydra_cfg.args)
    hydra_cfg.dataset.train_max_size = 10000
    train_dataset, val_dataset = _get_dataset(hydra_cfg.dataset, cfg.traj_length)
    tokenizers: Dict[str, Tokenizer] = {
        k: hydra.utils.call(v, key=k, train_dataset=train_dataset)
        for k, v in hydra_cfg.tokenizers.items()
    }
    tokenizer_manager = TokenizerManager(tokenizers)
    discrete_map: Dict[str, bool] = {}
    for k, v in tokenizers.items():
        discrete_map[k] = v.discrete
    train_loader = DataLoader(
        train_dataset,
        # shuffle=True,
        pin_memory=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
    )
    train_batch = next(iter(train_loader))
    tokenized = tokenizer_manager.encode(train_batch)
    data_shapes = {}
    for k, v in tokenized.items():
        data_shapes[k] = v.shape[-2:]

    model_config = hydra.utils.instantiate(hydra_cfg.model_config)
    model = MTM(data_shapes, cfg.traj_length, model_config)
    model.load_state_dict(torch.load(ckpt_path)["model"])
    model.eval()

    # freeze the model
    for param in model.parameters():
        param.requires_grad = False

    return model, tokenizer_manager, data_shapes


class ObsEncoder(nn.Module):
    def __init__(
        self, obs_dim, device, tokenizer_manager, mtm_model, keep_obs=False
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.device = device
        self.tokenizer_manager = tokenizer_manager
        self.mtm_model = mtm_model
        self.keep_obs = (
            keep_obs  # append MTM representation to obs, instead of replacing it
        )

    def forward(self, obs):
        # obs shape = (batch, obs_dim)
        batch_size = obs.shape[0]

        # make empty dict
        traj = {
            "states": obs.unsqueeze(1).clone().to(self.device),
        }
        masks = {
            "states": torch.ones(1, device=self.device),
        }
        token_representation = self.tokenizer_manager.encode(traj)
        state_representation = self.mtm_model.encode(token_representation, masks)[
            "states"
        ]
        if self.keep_obs:
            state_representation = torch.cat(
                [state_representation, obs.unsqueeze(-1)], dim=-1
            )
        state_representation = state_representation.reshape(batch_size, -1)
        return state_representation


class ObsActionEncoder(nn.Module):
    def __init__(self, obs_dim, act_dim, device, tokenizer_manager, mtm_model) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.tokenizer_manager = tokenizer_manager
        self.mtm_model = mtm_model

    def forward(self, obs, act):
        # obs shape = (batch, obs_dim)
        # act shape = (batch, act_dim)
        batch_size = obs.shape[0]

        # make empty dict
        traj = {
            "states": obs.unsqueeze(1).clone().to(self.device),
            "actions": act.unsqueeze(1).clone().to(self.device),
        }
        masks = {
            "states": torch.ones(1, device=self.device),
            "actions": torch.ones(1, device=self.device),
        }
        token_representation = self.tokenizer_manager.encode(traj)

        rep = self.mtm_model.encode(token_representation, masks)
        state_representation = rep["states"]
        action_representation = rep["actions"]
        return state_representation[:, 0], action_representation[:, 0]


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, obs_encoder=None):
        super().__init__()

        self.obs_encoder = obs_encoder
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        if self.obs_encoder is not None:
            obs = self.obs_encoder(obs)
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim,
        obs_encoder=None,
    ):
        super().__init__()

        self.obs_encoder = obs_encoder
        self.q1_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.q2_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        if self.obs_encoder is not None:
            obs = self.obs_encoder(obs)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.q1_net(obs_action)
        q2 = self.q2_net(obs_action)
        return q1, q2


class TD3Agent:
    def __init__(
        self,
        name,
        obs_shape,
        action_shape,
        device,
        lr,
        hidden_dim,
        critic_target_tau,
        stddev_schedule,
        nstep,
        batch_size,
        stddev_clip,
        use_tb,
        has_next_action=False,
        path=None,
        end_to_end=False,  # finetune MTM model
        keep_obs=False,  # append MTM representation to obs, instead of replacing it
        use_state_action_rep=False,  # use state-action representation instead of state representation
    ):
        assert path is not None
        self.mtm_model, self.tokenizer_manager, self.data_shapes = get_mtm_model(path)
        self.mtm_model.to(device)
        self.tokenizer_manager.to(device)

        # OG td3
        self.action_dim = action_shape[0]
        self.obs_dim = obs_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        latent_dim = (
            self.mtm_model.config.n_embd
            if self.mtm_model.config.latent_dim is None
            else self.mtm_model.config.latent_dim
        )
        self._mtm_obs_dim = (latent_dim + int(keep_obs)) * self.data_shapes["states"][0]
        self.keep_obs = keep_obs

        self.obs_encoder = ObsEncoder(
            obs_dim=self.obs_dim,
            device=self.device,
            tokenizer_manager=self.tokenizer_manager,
            mtm_model=self.mtm_model,
            keep_obs=self.keep_obs,
        )
        self.obs_encoder.requires_grad_(False)
        if use_state_action_rep:
            self.state_obs_rep = ObsActionEncoder(
                obs_dim=self.obs_dim,
                act_dim=self.action_dim,
                device=self.device,
                tokenizer_manager=self.tokenizer_manager,
                mtm_model=self.mtm_model,
            )
            self.state_obs_rep.requires_grad_(False)
        else:
            self.state_obs_rep = None

        self.end_to_end = end_to_end

        # models
        self.actor = Actor(
            self._mtm_obs_dim,
            action_shape[0],
            hidden_dim,
            obs_encoder=copy.deepcopy(self.obs_encoder) if end_to_end else None,
        ).to(device)

        if use_state_action_rep:
            assert end_to_end is False
            self.critic = Critic(
                self._mtm_obs_dim,
                self._mtm_obs_dim,
                hidden_dim,
                obs_encoder=copy.deepcopy(self.obs_encoder) if end_to_end else None,
            ).to(device)

            self.critic_target = Critic(
                self._mtm_obs_dim,
                self._mtm_obs_dim,
                hidden_dim,
                obs_encoder=copy.deepcopy(self.obs_encoder) if end_to_end else None,
            ).to(device)
        else:
            self.critic = Critic(
                self._mtm_obs_dim,
                action_shape[0],
                hidden_dim,
                obs_encoder=copy.deepcopy(self.obs_encoder) if end_to_end else None,
            ).to(device)

            self.critic_target = Critic(
                self._mtm_obs_dim,
                action_shape[0],
                hidden_dim,
                obs_encoder=copy.deepcopy(self.obs_encoder) if end_to_end else None,
            ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        if self.end_to_end:
            self.actor.obs_encoder.load_state_dict(self.obs_encoder.state_dict().copy())
            self.critic.obs_encoder.load_state_dict(
                self.obs_encoder.state_dict().copy()
            )
            self.critic_target.obs_encoder.load_state_dict(
                self.obs_encoder.state_dict().copy()
            )

            self.actor.obs_encoder.requires_grad_(True)
            self.critic.obs_encoder.requires_grad_(True)
            self.critic_target.obs_encoder.requires_grad_(True)

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, raw_obs, step, eval_mode):
        obs = torch.as_tensor(raw_obs, device=self.device).unsqueeze(0)
        if not self.end_to_end:
            with torch.no_grad():
                obs = self.obs_encoder(obs)  # encode obs

        stddev = utils.schedule(self.stddev_schedule, step)
        policy = self.actor(obs, stddev)
        if eval_mode:
            action = policy.mean
        else:
            action = policy.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(
        self, obs, action, reward, discount, next_obs, step, raw_obs, raw_next_obs
    ):
        # raw next obs is always the og raw envs obs
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            if self.state_obs_rep is not None:
                next_obs, next_action = self.state_obs_rep(raw_next_obs, next_action)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        if self.state_obs_rep is not None:
            obs, action = self.state_obs_rep(raw_obs, action)
        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        return metrics

    def update_actor(self, obs, action, step, raw_obs):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        policy = self.actor(obs, stddev)

        policy_action = policy.sample(clip=self.stddev_clip)
        if self.state_obs_rep:
            obs, policy_action = self.state_obs_rep(raw_obs, policy_action)

        Q1, Q2 = self.critic(obs, policy_action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_ent"] = policy.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        _raw_obs, action, reward, discount, _next_obs = utils.to_torch(
            batch, self.device
        )
        if self.end_to_end:
            obs = _raw_obs.clone()
            next_obs = _next_obs.clone()
        else:
            with torch.no_grad():
                obs = self.obs_encoder(_raw_obs)  # encode obs
                next_obs = self.obs_encoder(_next_obs)  # encode obs

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()
        # update critic
        metrics.update(
            self.update_critic(
                obs, action, reward, discount, next_obs, step, _raw_obs, _next_obs
            )
        )

        # update actor
        metrics.update(self.update_actor(obs, action, step, _raw_obs))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics
