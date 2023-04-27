# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from functools import partial
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb

from research.mtm.datasets.sequence_dataset import Trajectory, evaluate


class MLP(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        hidden_dim,
        n_layers,
        activation=nn.ReLU(),
        bnorm=False,
        input_norm=None,
    ) -> None:
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation

        self.net = []

        dims = [in_channel] + [hidden_dim] * n_layers + [out_channel]
        net = []
        if input_norm:
            net.append(input_norm)
        for h1, h2 in zip(dims[:-1], dims[1:]):
            net.append(nn.Linear(h1, h2))
            if bnorm:
                net.append(nn.BatchNorm1d(h2))
            net.append(activation)
        net.pop()

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


@dataclasses.dataclass
class MLPConfig:
    n_embd: int = 1024
    n_layer: int = 2
    reduce_use_sum: bool = False
    task: str = "bc"
    """Should be one of "bc", "id", "fd", "rcbc" """

    def create(self, data_shape, traj_length):
        return MLP_BC(data_shape, traj_length, self)


class MLP_BC(nn.Module):
    def __init__(
        self,
        data_shapes: Dict[str, Tuple[int, ...]],
        traj_length: int,
        config: MLPConfig,
    ):
        """Initialize a masked model.

        Args:
            data_shapes (Dict[str, Tuple[int, int]]): data_shapes
            config (MTMConfig): config
        """
        super().__init__()
        self.config = config
        if config.task == "bc":
            self.out_shape = data_shapes["actions"][0] * data_shapes["actions"][1]
            self.in_shape = data_shapes["states"][0] * data_shapes["states"][1]
        elif config.task == "id":
            assert traj_length >= 2
            # given T states, predict the T-1 action
            self.out_shape = data_shapes["actions"][0] * data_shapes["actions"][1]
            self.in_shape = (
                data_shapes["states"][0] * data_shapes["states"][1] * traj_length
            )
        elif config.task == "fd":
            assert traj_length >= 2
            # given the T-1 states, and the T-1th action, predict the Tth state
            self.out_shape = data_shapes["states"][0] * data_shapes["states"][1]
            self.in_shape = (
                data_shapes["states"][0] * data_shapes["states"][1] * (traj_length - 1)
                + data_shapes["actions"][0] * data_shapes["actions"][1]
            )
        elif config.task == "rcbc":
            self.out_shape = data_shapes["actions"][0] * data_shapes["actions"][1]
            self.in_shape = (
                data_shapes["states"][0] * data_shapes["states"][1]
                + data_shapes["returns"][0] * data_shapes["returns"][1]
            )
        else:
            raise NotImplementedError

        self.data_shapes = data_shapes
        self.mlp = MLP(
            in_channel=self.in_shape,
            out_channel=self.out_shape,
            hidden_dim=config.n_embd,
            n_layers=config.n_layer,
            activation=nn.GELU(),
        )

    def forward(self, trajectories, discrete_map: Dict[str, bool], compute_loss=True):
        """
        Args:
            trajectories (Dict[str, torch.Tensor]): trajectories
            discrete_map (Dict[str, bool]): discrete_map
            compute_loss (bool, optional): compute_loss. Defaults to True.
        """
        if self.config.task == "bc":
            states = trajectories["states"]
            B, T, tokens_per_time, feature_dim = states.shape
            states = states.reshape(B * T, tokens_per_time * feature_dim)
            pred_logits = self.mlp(states)
            pred_logits = pred_logits.reshape(
                B, T, self.data_shapes["actions"][0], self.data_shapes["actions"][1]
            )
            if compute_loss:
                target = trajectories["actions"]
                is_discrete = discrete_map["actions"]
        elif self.config.task == "id":
            states = trajectories["states"]
            B, T, tokens_per_time, feature_dim = states.shape
            states = states.reshape(B, T * tokens_per_time * feature_dim)
            pred_logits = self.mlp(states)
            pred_logits = pred_logits.reshape(
                B, 1, self.data_shapes["actions"][0], self.data_shapes["actions"][1]
            )
            if compute_loss:
                target = trajectories["actions"][:, T - 2 : T - 1]
                is_discrete = discrete_map["actions"]
        elif self.config.task == "fd":
            states = trajectories["states"]
            actions = trajectories["actions"]
            B, T, tokens_per_time_states, feature_dim_states = states.shape

            input_states = states[:, : T - 1, :, :]
            input_actions = actions[:, T - 1 : T, :, :]
            input_states = input_states.reshape(
                B, (T - 1) * tokens_per_time_states * feature_dim_states
            )
            input_actions = input_actions.reshape(B, -1)
            input_features = torch.cat([input_states, input_actions], dim=1)
            pred_logits = self.mlp(input_features)
            pred_logits = pred_logits.reshape(
                B, 1, self.data_shapes["states"][0], self.data_shapes["states"][1]
            )
            if compute_loss:
                target = states[:, T - 1 : T]
                is_discrete = discrete_map["states"]
        elif self.config.task == "rcbc":
            states = trajectories["states"]
            B, T, tokens_per_time, feature_dim = states.shape
            states = states.reshape(B * T, tokens_per_time * feature_dim)

            returns = trajectories["returns"]
            B, T, tokens_per_time, feature_dim = returns.shape
            returns = returns.reshape(B * T, tokens_per_time * feature_dim)

            features = torch.cat([states, returns], dim=1)
            pred_logits = self.mlp(features)
            pred_logits = pred_logits.reshape(
                B, T, self.data_shapes["actions"][0], self.data_shapes["actions"][1]
            )

            if compute_loss:
                target = trajectories["actions"]
                is_discrete = discrete_map["actions"]
        else:
            raise NotImplementedError

        if compute_loss:
            loss = self._compute_loss(pred_logits, target, is_discrete)
        else:
            loss = None
        return pred_logits, loss

    def _compute_loss(self, pred_logits, actions, is_discrete):
        """Compute loss for the model.

        Args:
            pred_logits (torch.Tensor): (batch_size, T, tokens_per_time, feature_dim)
            actions (torch.Tensor): (batch_size, T, tokens_per_time, feature_dim)
            discrete_map (Dict[str, bool]): discrete_map
        """
        if is_discrete:
            loss = nn.CrossEntropyLoss(reduction="none")(
                pred_logits.permute(0, 3, 1, 2), actions.permute(0, 3, 1, 2)
            ).unsqueeze(3)
        else:
            loss = nn.MSELoss(reduction="none")(pred_logits, actions)
        if self.config.reduce_use_sum:
            loss = loss.sum(dim=(2, 3)).mean()
        else:
            loss = loss.mean(dim=(2, 3)).mean()
        return loss

    def evaluate(self, env, batch, tokenizer_manager, discrete_map):
        """Evaluate the model.

        Args:
            env (gym.Env): env
            batch (Dict[str, torch.Tensor]): batch
            tokenizer_manager (TokenizerManager): tokenizer_manager
            discrete_map (Dict[str, bool]): discrete_map
        """
        device = next(self.parameters()).device
        eval_dict = {}
        batch = {k: v.to(device) for k, v in batch.items()}
        encoded_batch = tokenizer_manager.encode(batch)
        pred_logits, loss = self(encoded_batch, discrete_map)
        eval_dict["val/val_loss"] = loss.item()

        if self.config.task == "bc" or self.config.task == "rcbc":
            task_log = self.eval_bc(env, tokenizer_manager, discrete_map)
        elif self.config.task == "id":
            task_log = self.eval_id(env, batch, tokenizer_manager, discrete_map)
        elif self.config.task == "fd":
            task_log = self.eval_fd(env, batch, tokenizer_manager, discrete_map)
        else:
            raise NotImplementedError

        eval_dict.update(task_log)
        return eval_dict

    def eval_bc(self, env, tokenizer_manager, discrete_map) -> Dict[str, Any]:
        device = next(self.parameters()).device

        @torch.inference_mode()
        def sample_action_bc(
            observation: np.ndarray,
            traj: Trajectory,
        ):
            """Sample action from the model.

            Args:
                observation (np.ndarray): observation
                traj (Trajectory): traj
            """

            input_ = {"states": torch.from_numpy(observation)[None, None].to(device)}
            logits, _ = self(
                tokenizer_manager.encode(input_), discrete_map, compute_loss=False
            )
            decoded_logits = tokenizer_manager.decode({"actions": logits})
            a = decoded_logits["actions"][0].detach().cpu().numpy()
            return a[0]

        observation_shape = env.observation_space.shape
        action_shape = env.action_space.shape

        percentages = [1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6]

        @torch.inference_mode()
        def sample_action_rcbc(
            observation: np.ndarray,
            traj: Trajectory,
            percentage: float,
        ):
            """Sample action from the model.

            Args:
                observation (np.ndarray): observation
                traj (Trajectory): traj
            """
            return_max = tokenizer_manager.tokenizers["returns"].stats.max
            return_min = tokenizer_manager.tokenizers["returns"].stats.min

            return_value = return_min + (return_max - return_min) * percentage
            return_to_go = float(return_value)
            # return_to_go = float(return_value - traj.rewards.sum())

            input_ = {
                "states": torch.from_numpy(observation)[None, None].to(device),
                "returns": return_to_go * torch.ones((1, 1, 1), device=device),
            }
            encoded_inputs = tokenizer_manager.encode(input_)

            logits, _ = self(encoded_inputs, discrete_map, compute_loss=False)
            decoded_logits = tokenizer_manager.decode({"actions": logits})
            a = decoded_logits["actions"][0].detach().cpu().numpy()
            return a[0]

        eval_dict = {}
        if self.config.task == "bc":
            fn = sample_action_bc
            results, videos = evaluate(
                fn,
                env,
                20,
                observation_shape,
                action_shape,
                num_videos=0,
            )

            for k, v in results.items():
                eval_dict[f"eval_bc/{k}"] = v
            for idx, v in enumerate(videos):
                eval_dict[f"eval_bc_video_{idx}/video"] = wandb.Video(
                    v.transpose(0, 3, 1, 2), fps=10, format="gif"
                )
        elif self.config.task == "rcbc":
            for p in percentages:
                fn = partial(sample_action_rcbc, percentage=p)
                results, videos = evaluate(
                    fn,
                    env,
                    20,
                    observation_shape,
                    action_shape,
                    num_videos=0,
                )

                for k, v in results.items():
                    eval_dict[f"eval_rcbc/{k}_{p}"] = v
                for idx, v in enumerate(videos):
                    eval_dict[f"eval_rcbc_video_{idx}_{p}/video"] = wandb.Video(
                        v.transpose(0, 3, 1, 2), fps=10, format="gif"
                    )
        else:
            raise NotImplementedError(f"{self.task}")

        return eval_dict

    def eval_id(
        self, env, eval_batch, tokenizer_manager, discrete_map
    ) -> Dict[str, Any]:
        """Evaluate the model on the inverse dynamics task.
        Args:
            env (gym.Env): env
            eval_batch (Dict[str, torch.Tensor]): eval_batch
            tokenizer_manager (TokenizerManager): tokenizer_manager
            discrete_map (Dict[str, bool]): discrete_map
        """
        states = eval_batch["states"]
        actions = eval_batch["actions"]
        B, T, S = states.shape

        predict_logits = self(
            tokenizer_manager.encode(eval_batch), discrete_map, compute_loss=False
        )[0]
        predicted_action = tokenizer_manager.decode({"actions": predict_logits})[
            "actions"
        ]

        action_error = []

        for i in range(B):
            # set state to be the second to last state
            phys_state = np.zeros(S + 2)
            phys_state[2:] = states[i, T - 2].detach().cpu().numpy()
            # get the action from the model
            action = predicted_action[i, 0].detach().cpu().numpy()
            # get the ground truth action
            gt_action = actions[i, T - 2].detach().cpu().numpy()
            # compute action error
            action_error.append((action - gt_action) ** 2)
            # compute state error

        eval_dict = {}
        eval_dict["eval/id_action_error"] = torch.mean(
            torch.tensor(action_error)
        ).item()
        return eval_dict

    def eval_fd(
        self, env, eval_batch, tokenizer_manager, discrete_map
    ) -> Dict[str, Any]:
        """Evaluate the model on the forward dynamics task.
        Args:
            env (gym.Env): env
            eval_batch (Dict[str, torch.Tensor]): eval_batch
            tokenizer_manager (TokenizerManager): tokenizer_manager
            discrete_map (Dict[str, bool]): discrete_map
        """
        states = eval_batch["states"]
        actions = eval_batch["actions"]
        encoded_batch = tokenizer_manager.encode(eval_batch)

        B, T, _ = states.shape

        predict_logits = self(
            tokenizer_manager.encode(eval_batch), discrete_map, compute_loss=False
        )[0]
        predicted_next_state = tokenizer_manager.decode({"states": predict_logits})[
            "states"
        ]

        next_state = states[:, -1]
        state_error = (next_state - predicted_next_state[:, 0, :]) ** 2
        eval_dict = {}
        eval_dict["eval/fd_state_error"] = torch.mean(state_error).item()
        return eval_dict
