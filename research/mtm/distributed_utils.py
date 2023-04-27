# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import logging
import os
import time

import torch

log = logging.getLogger(__name__)


@dataclasses.dataclass
class DistributedParams:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    local_world_size: int = torch.cuda.device_count()
    master_addr: str = "localhost"
    master_port: int = 25900
    job_id: str = f"{int(time.time_ns())}"
    env_loaded: str = "default_local"


def get_distributed_params() -> DistributedParams:
    try:
        log.info(f"Loading distributed job info from submitit...")
        import submitit

        submitit_job_env = submitit.slurm.slurm.SlurmJobEnvironment()

        result = DistributedParams(
            rank=submitit_job_env.global_rank,
            local_rank=submitit_job_env.local_rank,
            world_size=submitit_job_env.num_tasks,
            local_world_size=torch.cuda.device_count(),
            master_addr=submitit_job_env.hostnames[0],
            master_port=29500,
            job_id=submitit_job_env.job_id,
            env_loaded="submitit",
        )
        os.environ["RANK"] = f"{result.rank}"
        os.environ["LOCAL_RANK"] = f"{result.local_rank}"
        os.environ["WORLD_SIZE"] = f"{result.world_size}"
        os.environ["LOCAL_WORLD_SIZE"] = f"{result.local_world_size}"
        os.environ["MASTER_ADDR"] = f"{result.master_addr}"
        os.environ["MASTER_PORT"] = f"{result.master_port}"

        return result
    except Exception as e:
        log.warning(f"Unable to load from submitit JobEnvironment: {e}")

    try:
        log.info(f"Loading distributed job info from environment variables...")
        result = DistributedParams(
            rank=int(os.environ["RANK"]),
            local_rank=int(os.environ["LOCAL_RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
            local_world_size=int(os.environ["LOCAL_WORLD_SIZE"]),
            master_addr=os.environ["MASTER_ADDR"],
            master_port=os.environ["MASTER_PORT"],
            job_id=os.environ["TORCHELASTIC_RUN_ID"],
            env_loaded="env_variables",
        )
        return result
    except Exception as e:
        log.warning(f"Unable to load from environment variables: {e}")

    return DistributedParams()
