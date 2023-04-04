from functools import lru_cache
from typing import Callable, List

import wandb


@lru_cache(maxsize=5)
def extract_runs(api: wandb.Api, project: str, entity: str) -> List[wandb.run]:
    return api.runs(f"{entity}/{project}")


def filter_runs(
    select_fn: Callable[[wandb.run], bool], runs: List[wandb.run]
) -> List[wandb.run]:
    return [r for r in runs if select_fn(r)]
