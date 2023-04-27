# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import dataclasses
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, replace
from socket import gethostname
from typing import Any, Dict, Generator, Optional

import cloudpickle as pickle
import numpy as np
import termcolor
import wandb

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_stopwatch_nest_counter: int = 0


@contextlib.contextmanager
def stopwatch(label: str = "block") -> Generator[None, None, None]:
    """Helper for printing the runtime of a block of code.
    Example:
    ```
    with fannypack.utils.stopwatch("name"):
        time.sleep(1.0)
    ```
    Args:
        label (str): Label for block that's running.
    Returns:
        Generator: Context manager to place in `with` statement.
    """
    start_time = time.time()

    def log_red(*args, **kwargs):
        if len(args) > 0:
            global _stopwatch_nest_counter
            prefix = ("    " * _stopwatch_nest_counter) + f"[stopwatch: {label}] "
            args = (termcolor.colored(prefix + args[0], color="yellow"),) + args[1:]
        logger.info(*args, **kwargs)

    log_red("Starting!")

    global _stopwatch_nest_counter
    _stopwatch_nest_counter += 1
    yield
    _stopwatch_nest_counter -= 1

    log_red(
        f"Completed in {termcolor.colored(str(time.time() - start_time) + ' seconds', attrs=['bold'])}"
    )


@dataclass(frozen=False)
class WandBLoggerConfig:
    project: str
    output_dir: str = "logs"
    prefix: str = ""
    online: bool = True
    experiment_id: Optional[str] = None
    notes: str = ""
    entity: Optional[str] = None
    resume: bool = False
    group: Optional[str] = None

    # to prevent overloading the wandb backend by spawning too many wandb instances
    random_delay: float = 20.0


class WandBLogger:
    def __init__(
        self,
        config: WandBLoggerConfig,
        log_exp_config_dict: Dict[str, Any],
        enable: bool = True,
    ):
        assert dataclasses.is_dataclass(config)
        self.enable = enable
        self.config = config

        if self.config.experiment_id is None:
            self.config = replace(self.config, experiment_id=uuid.uuid4().hex)

        if self.config.prefix != "":
            project = "{}--{}".format(self.config.prefix, self.config.project)
            self.config = replace(self.config, project=project)
        if self.enable:
            if self.config.output_dir == "":
                self.save_dir = tempfile.mkdtemp()
            else:
                self.save_dir = os.path.join(
                    self.config.output_dir, self.config.experiment_id
                )
                os.makedirs(self.save_dir, exist_ok=True)

        if self.enable:
            if self.config.random_delay > 0:
                time.sleep(np.random.uniform(0, self.config.random_delay))

            log_exp_config_dict["hostname"] = gethostname()
            self.run = wandb.init(
                reinit=True,
                config=log_exp_config_dict,
                project=self.config.project,
                dir=self.save_dir,
                id=self.config.experiment_id,
                notes=self.config.notes,
                entity=self.config.entity,
                settings=wandb.Settings(
                    start_method="thread",
                    # _disable_stats=True,
                ),
                mode="online" if self.config.online else "offline",
                resume="allow" if self.config.resume else None,
                group=self.config.group,
            )
        else:
            self.run = None

    def log(self, *args: Any, **kwargs: Any) -> None:
        if self.enable:
            self.run.log(*args, **kwargs)

    def add_prefix_and_log(
        self, metrics: Dict[str, Any], prefix: str, **kwargs: Any
    ) -> None:
        new_metrics = {}
        for k, v in metrics.items():
            new_metrics[f"{prefix}/{k}"] = v
        self.log(new_metrics, **kwargs)

    def save_pickle(self, obj: Any, filename: str) -> None:
        with open(os.path.join(self.save_dir, filename), "wb") as fout:
            pickle.dump(obj, fout)

    @property
    def experiment_id(self) -> str:
        return self.config.experiment_id

    @property
    def output_dir(self) -> str:
        return self.save_dir

    @contextlib.contextmanager
    def stopwatch(self, label: str, step: int) -> Generator[None, None, None]:
        """Time and log something in a convenient context manager."""
        start_time = time.time()
        yield
        run_time = time.time() - start_time
        self.log({f"timing/{label}": run_time}, step=step)
