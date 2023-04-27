# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Main script for training a policy given a dataset.
"""
import os
import pprint
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict

import hydra
import numpy as np
import torch
import torch.distributed
import torch.multiprocessing
import torch.nn.parallel
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from research.logger import WandBLogger, WandBLoggerConfig, logger
from research.mtm.datasets.base import DatasetProtocol
from research.mtm.distributed_utils import DistributedParams, get_distributed_params
from research.mtm.models.mtm_model import MTMConfig
from research.mtm.tokenizers.base import Tokenizer, TokenizerManager
from research.mtm.tokenizers.continuous import ContinuousTokenizer
from research.mtm.utils import (
    get_cfg_hash,
    get_ckpt_path_from_folder,
    get_git_dirty,
    get_git_hash,
    set_seed_everywhere,
)

dir_path = os.path.dirname(os.path.realpath(__file__))


@dataclass
class RunConfig:
    seed: int = 0
    """RNG seed."""

    batch_size: int = 64
    """Batch size used during training."""

    n_workers: int = 8
    """Number of workers for loading data."""

    model_config: MTMConfig = field(default_factory=MTMConfig)
    """Model configuration."""

    log_every: int = 100
    """Print training loss every N steps."""

    print_every: int = 1000
    """Print training loss every N steps."""

    eval_every: int = 5000
    """Evaluate model every N steps."""

    save_every: int = 5000
    """Evaluate model every N steps."""

    device: str = "cuda"
    """Device to use for training."""

    warmup_steps: int = 1_000
    """Number of warmup steps for learning rate scheduler."""

    num_train_steps: int = 5_000_000
    """Number of training steps."""

    learning_rate: float = 1e-3
    """Learning rate."""

    weight_decay: float = 1e-5
    """Weight decay."""

    traj_length: int = 1
    """Trajectory length."""


def train_one_batch(
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Callable,
    tokenizer_manager: TokenizerManager,
    discrete_map: Dict[str, bool],
    batch: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    encoded_batch = tokenizer_manager.encode(batch)

    # train the model
    prediction, loss = model(encoded_batch, discrete_map)

    # create a dictionary to log all of the losses
    log_dict = {"train/train_loss": loss.item()}
    log_dict["train/lr"] = scheduler.get_last_lr()[0]

    # backprop
    model.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return log_dict


def main(hydra_cfg):
    _main(hydra_cfg)
    # p = Profiler()
    # with p:
    #     _main(hydra_cfg)
    # p.print()


def _main(hydra_cfg):
    cfg: RunConfig = hydra.utils.instantiate(hydra_cfg.args)
    dp: DistributedParams = get_distributed_params()

    if cfg.device != "cpu":
        torch.cuda.set_device(dp.local_rank)

    distributed = dp.world_size > 1
    if distributed:
        logger.info(
            f"Initializing rank {dp.rank} (local rank {dp.local_rank}) in total world size {dp.world_size} (local world size {dp.local_world_size}) with master addr:port {dp.master_addr}:{dp.master_port}"
        )
        torch.distributed.init_process_group(
            backend="nccl", rank=dp.rank, world_size=dp.world_size
        )

    set_seed_everywhere(cfg.seed)
    pprint.pp(cfg)

    logger.info(f"Working directory: {os.getcwd()}")

    with open("config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(hydra_cfg))

    train_dataset: DatasetProtocol
    val_dataset: DatasetProtocol

    train_dataset, val_dataset = hydra.utils.call(
        hydra_cfg.dataset, seq_steps=cfg.traj_length
    )
    logger.info(f"Train set size = {len(train_dataset)}")
    logger.info(f"Validation set size = {len(val_dataset)}")

    if "tokenizers" in hydra_cfg:
        tokenizers: Dict[str, Tokenizer] = {
            k: hydra.utils.call(v, key=k, train_dataset=train_dataset)
            for k, v in hydra_cfg.tokenizers.items()
        }
    else:
        tokenizers: Dict[str, Tokenizer] = {
            k: ContinuousTokenizer.create(k, train_dataset)
            for k in train_dataset[0].keys()
        }
    tokenizer_manager = TokenizerManager(tokenizers).to(cfg.device)

    discrete_map: Dict[str, bool] = {}
    for k, v in tokenizers.items():
        discrete_map[k] = v.discrete
    logger.info(f"Tokenizers: {tokenizers}")

    if distributed:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=dp.world_size, rank=dp.rank, shuffle=True
        )
        val_sampler = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=dp.world_size, rank=dp.rank, shuffle=False
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        # shuffle=True,
        pin_memory=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_dataset,
        # shuffle=False,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        sampler=val_sampler,
    )
    train_batch = next(iter(train_loader))
    tokenized = tokenizer_manager.encode(train_batch)
    data_shapes = {}
    for k, v in tokenized.items():
        data_shapes[k] = v.shape[-2:]
    logger.info(f"Data shapes: {data_shapes}")

    # create the model
    model_config = hydra.utils.instantiate(hydra_cfg.model_config)
    model = model_config.create(data_shapes, cfg.traj_length)
    model.to(cfg.device)
    model.train()
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dp.local_rank], output_device=dp.local_rank
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    def _schedule(step):
        return 1

        # warmp for 1000 steps
        if step < cfg.warmup_steps:
            return step / cfg.warmup_steps

        # then cosine decay
        assert cfg.num_train_steps > cfg.warmup_steps
        step = step - cfg.warmup_steps
        return 0.5 * (
            1 + np.cos(step / (cfg.num_train_steps - cfg.warmup_steps) * np.pi)
        )

    scheduler = LambdaLR(optimizer, lr_lambda=_schedule)

    # create a wandb logger and log params of interest
    wandb_cfg_log_dict = OmegaConf.to_container(hydra_cfg)
    wandb_cfg_log_dict["*discrete_map"] = discrete_map
    wandb_cfg_log_dict["*path"] = str(os.getcwd())
    wandb_cfg_log_dict["*git_hash"] = get_git_hash()
    wandb_cfg_log_dict["*git_dirty"] = get_git_dirty()
    wandb_cfg_log_dict["*hydra_cfg_hash"] = get_cfg_hash(hydra_cfg)
    wandb_cfg_log_dict["*num_parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    wandb_cfg_log = WandBLoggerConfig(
        experiment_id=f"{dp.job_id}-{dp.rank}",
        project=hydra_cfg.wandb.project,
        entity=hydra_cfg.wandb.entity or None,
        resume=hydra_cfg.wandb.resume,
        group=dp.job_id,
    )

    if wandb_cfg_log.resume:
        exp_id = wandb_cfg_log_dict["*hydra_cfg_hash"]
        wandb_cfg_log = replace(
            wandb_cfg_log,
            experiment_id=exp_id,
        )
    # wandb_logger = WandBLogger(wandb_cfg_log, wandb_cfg_log_dict, enable=False)
    wandb_logger = WandBLogger(wandb_cfg_log, wandb_cfg_log_dict)

    step = 0
    if wandb_logger.enable and wandb.run.resumed:
        # find checkpoints in the directory
        logger.info("Trying to resume ...")
        ckpt_path = get_ckpt_path_from_folder(
            os.getcwd()
        )  # hydra/submit-it will resume jobs from the same folder
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=cfg.device)
            logger.info(f"Resuming from checkpoint: {ckpt_path}")
            step = ckpt["step"]
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            logger.info(f"No checkpoints found, starting from scratch.")
    logger.info(f"starting from step={step}")

    # train the model
    vis_batch = next(iter(val_loader))  # keep this batch for visualization
    vis_batch = {k: v.to(cfg.device) for k, v in vis_batch.items()}

    epoch = 0

    iteration_start_time = time.time()
    eval_max = defaultdict(lambda: -np.inf)
    while True:
        for batch in train_loader:
            # cycle between different types
            start_time = time.time()

            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            log_dict = train_one_batch(
                model,
                optimizer,
                scheduler,
                tokenizer_manager,
                discrete_map,
                batch,
            )
            log_dict["train/epochs"] = epoch

            # log train step time = time to process a batch
            log_dict["time/train_step"] = time.time() - start_time

            if step % cfg.print_every == 0:
                train_loss = log_dict["train/train_loss"]
                logger.info(f"Step: {step}, Train Loss: {train_loss}")
            if dp.rank == 0 and step % cfg.save_every == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                    },
                    f"model_{step}.pt",
                )
                try:
                    if step > 3 * cfg.save_every:
                        remove_step = step - 3 * cfg.save_every
                        if (remove_step // cfg.save_every) % 10 != 0:
                            os.remove(f"model_{remove_step}.pt")
                except Exception as e:
                    logger.error(f"Failed to remove model file! {e}")

            if step % cfg.eval_every == 0:
                # evaluate the model
                start_time = time.time()
                model.eval()
                val_batch = next(iter(val_loader))
                eval_log = model.evaluate(
                    val_dataset.env, val_batch, tokenizer_manager, discrete_map
                )
                eval_log["time/eval_time"] = time.time() - start_time

                # for everything with eval prefix keep the max
                max_log = {}
                for k, v in eval_log.items():
                    if k.startswith("eval/"):
                        eval_max[k] = max(eval_max[k], v)
                        max_log[f"max_{k}"] = eval_max[k]
                eval_log.update(max_log)

                # log the eval results
                wandb_logger.log(eval_log, step=step)
                val_loss = eval_log["val/val_loss"]
                logger.info(f"Step: {step}, Val Loss: {val_loss}")
                model.train()

            log_dict["time/iteration_time"] = time.time() - iteration_start_time
            iteration_start_time = time.time()

            if step % cfg.log_every == 0:
                logger.info(f"Step {step}")
                wandb_logger.log(log_dict, step=step)

            step += 1
            if step >= cfg.num_train_steps:
                break

        if step >= cfg.num_train_steps:
            break
        epoch += 1


@hydra.main(config_path=".", config_name="config_mlp", version_base="1.1")
def configure_jobs(hydra_data: DictConfig) -> None:
    logger.info(hydra_data)
    main(hydra_data)


if __name__ == "__main__":
    configure_jobs()
