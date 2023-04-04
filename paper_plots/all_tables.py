import pathlib
from collections import defaultdict
from typing import Callable, List

import numpy as np
import wandb

from research.utils.latex import generate_latex_table
from research.utils.wandb import extract_runs, filter_runs

d4rl_env_map = {
    "walker": [
        "walker2d-expert-v2",
        "walker2d-medium-expert-v2",
        "walker2d-medium-v2",
        "walker2d-medium-replay-v2",
    ],
    "hopper": [
        "hopper-expert-v2",
        "hopper-medium-expert-v2",
        "hopper-medium-v2",
        "hopper-medium-replay-v2",
    ],
    "halfcheetah": [
        "halfcheetah-expert-v2",
        "halfcheetah-medium-expert-v2",
        "halfcheetah-medium-v2",
        "halfcheetah-medium-replay-v2",
    ],
}
adroit_env_map = {
    "hammer": [
        "hammer-expert",
        "hammer-medium_replay",
    ],
    "pen": [
        "pen-expert",
        "pen-medium_replay",
    ],
    "relocate": [
        "relocate-expert",
        "relocate-medium_replay",
    ],
    "door": [
        "door-expert",
        "door-medium_replay",
    ],
}
tasks = ["bc", "rcbc", "id", "fd"]
FRAC_DONE = 0.8


def _make_mlp_d4rl_select_fn(env_name: str, task: str) -> Callable[[wandb.run], bool]:
    def _s(r: wandb.run) -> bool:
        if r.config["dataset"]["env_name"] == env_name:
            if (
                "task" in r.config["model_config"]
                and r.config["model_config"]["task"] == task
            ):
                if "Continuous" in r.config["tokenizers"]["states"]["_target_"]:
                    if "_step" in r.summary:
                        if (
                            r.config["args"]["num_train_steps"] * FRAC_DONE
                            < r.summary["_step"]
                        ):
                            return True
        return False

    return _s


def _make_mlp_adroit_select_fn(env_name: str, task: str) -> Callable[[wandb.run], bool]:
    t_name, d_name = env_name.split("-")

    def _s(r: wandb.run) -> bool:
        if (
            r.config["dataset"]["env_name"] == t_name
            and r.config["dataset"]["d_name"] == d_name
        ):
            if (
                "task" in r.config["model_config"]
                and r.config["model_config"]["task"] == task
            ):
                if "Continuous" in r.config["tokenizers"]["states"]["_target_"]:
                    if "_step" in r.summary:
                        if (
                            r.config["args"]["num_train_steps"] * FRAC_DONE
                            < r.summary["_step"]
                        ):
                            return True
        return False

    return _s


def mlp_table_entry(
    runs: List[wandb.run], env_name: str, task: str, domain: str = "d4rl"
) -> List[wandb.run]:
    assert task in ["bc", "id", "fd", "rcbc"]

    if domain == "d4rl":
        select_fn = _make_mlp_d4rl_select_fn(env_name, task)
    elif domain == "adroit":
        select_fn = _make_mlp_adroit_select_fn(env_name, task)
    else:
        raise ValueError(f"Unknown domain {domain}")
    runs = filter_runs(select_fn, runs)
    if len(runs) < 5:
        print(f"Warning mlp_table_entry: {env_name} {task} has {len(runs)} runs")
    else:
        runs = runs[:5]

    values = []
    if task == "rcbc":
        candidates = defaultdict(list)
        for r in runs:
            keys = [f"eval/return_mean_{i}" for i in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]]
            metrics = r.history(keys=keys)

            for k in keys:
                candidates[k].append(np.max(metrics[k]))

        candidates_means = {}
        for k in keys:
            candidates_means[k] = np.mean(candidates[k])

        best_key = max(candidates_means, key=candidates.get)
        values = candidates[best_key]

    else:
        for r in runs:
            try:
                if task == "bc":
                    metrics = r.history(keys=["eval/return_mean"])
                    max_metric = np.max(metrics["eval/return_mean"])
                    values.append(max_metric)
                elif task == "id":
                    try:
                        metrics = r.history(keys=["eval/id_action_error"])
                        max_metric = np.min(metrics["eval/id_action_error"])
                    except:
                        metrics = r.history(keys=["eval/id_action_error_r"])
                        max_metric = np.min(metrics["eval/id_action_error_r"])
                    values.append(max_metric * 100)
                elif task == "fd":
                    metrics = r.history(keys=["eval/fd_state_error"])
                    max_metric = np.min(metrics["eval/fd_state_error"])
                    values.append(max_metric * 100)
                else:
                    raise ValueError(f"Unknown task {task}")
            except Exception as e:
                print(e)
                import ipdb

                ipdb.set_trace()

                print(r.name)

    return np.mean(values), np.std(values)


FRAC_DONE = 0.6


def _make_mtm_d4rl_select_fn(
    env_name: str, mp: List[str]
) -> Callable[[wandb.run], bool]:
    def _s(r: wandb.run) -> bool:
        if r.config["dataset"]["env_name"] == env_name:
            if set(r.config["args"]["mask_patterns"]) == set(mp):
                # insure trained to at least 0.8 of desired steps
                if "_step" in r.summary:
                    if (
                        r.config["args"]["num_train_steps"] * FRAC_DONE
                        < r.summary["_step"]
                    ):
                        if "Continuous" in r.config["tokenizers"]["states"]["_target_"]:
                            return True
        return False

    return _s


def _make_mtm_adroit_select_fn(
    env_name: str, mp: List[str]
) -> Callable[[wandb.run], bool]:
    t_name, d_name = env_name.split("-")

    def _s(r: wandb.run) -> bool:
        if (
            r.config["dataset"]["env_name"] == t_name
            and r.config["dataset"]["d_name"] == d_name
        ):
            if set(r.config["args"]["mask_patterns"]) == set(mp):
                # insure trained to at least 0.8 of desired steps
                if "_step" in r.summary:
                    if (
                        r.config["args"]["num_train_steps"] * FRAC_DONE
                        < r.summary["_step"]
                    ):
                        if "Continuous" in r.config["tokenizers"]["states"]["_target_"]:
                            return True
        return False

    return _s


def _mtm_get_value(run: wandb.run, task: str) -> float:
    if task == "bc":
        k = "eval1/return_mean"
        metrics = run.history(keys=[k])
        return np.max(metrics[k])
    elif task == "id":
        k = "eval/id_action_error_r=1"
        metrics = run.history(keys=[k])
        return np.min(metrics[k]) * 100
    elif task == "fd":
        k = "eval/fd_state_error_r=1"
        metrics = run.history(keys=[k])
        return np.min(metrics[k]) * 100
    else:
        raise ValueError(f"Unknown task {task}")


def get_max_key(dict_values):
    best_value = -np.inf
    best_key = ""
    for k, v in dict_values.items():
        if v > best_value:
            best_value = v
            best_key = k
    return best_key


def mtm_table_entry(
    runs: List[wandb.run],
    env_name: str,
    task: str,
    mp: List[str],
    domain: str = "d4rl",
) -> List[wandb.run]:
    if domain == "d4rl":
        select_fn = _make_mtm_d4rl_select_fn(env_name, mp)
    elif domain == "adroit":
        select_fn = _make_mtm_adroit_select_fn(env_name, mp)
    else:
        raise ValueError(f"Unknown domain {domain}")

    runs = filter_runs(select_fn, runs)
    if len(runs) < 5:
        print(f"Warning mtm_table_entry: {env_name} {task} has {len(runs)} runs")
    else:
        runs = runs[:5]

    values = []
    if task == "rcbc":
        avalible = set(runs[0].summary.keys())

        candidates = defaultdict(list)
        for r in runs:
            keys = [f"eval2/p={i}_return_mean" for i in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]]
            keys += [f"eval5/p={i}_return_mean" for i in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]]
            keys += [
                f"eval_ts/p={i}_return_mean" for i in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
            ]
            keys += [
                f"eval_cem/p={i}_return_mean" for i in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
            ]
            keys = [k for k in keys if k in avalible]
            metrics = r.history(keys=keys)
            for k in keys:
                candidates[k].append(np.max(metrics[k]))

        candidates_means = {}
        for k in keys:
            candidates_means[k] = np.mean(candidates[k])

        best_key = get_max_key(candidates_means)
        values = candidates[best_key]
    else:
        for r in runs:
            values.append(_mtm_get_value(r, task))

    return np.mean(values), np.std(values)


def mean_std_to_str(mean: float, std: float) -> str:
    return f"{mean:.2f} $\pm$ {std:.2f}"


def d4rl(api, entity, test=True) -> None:
    def _name_to_des(name: str) -> str:
        names = name.split("-")[1:-1]
        return " ".join([n.capitalize() for n in names])

    if test:
        table = [["Dataset", "Task", "MLP", "MTM-Specalized"]]
        env_names = d4rl_env_map["walker"]

        project = "d4rl_mtm_paper"
        runs = extract_runs(api, project, entity)
        mtm_specialized_entries = {}
        for n in env_names:
            for t in tasks:
                mp = [t.upper()]
                mean, std = mtm_table_entry(runs, n, t, mp)
                mtm_specialized_entries[(n, t)] = (mean, std)

        project = "mlp_baselines"
        mlp_runs = extract_runs(api, project, entity)
        mlp_entries = {}
        for n in env_names:
            for t in tasks:
                mean, std = mlp_table_entry(mlp_runs, n, t)
                mlp_entries[(n, t)] = (mean, std)

        # build rows
        for n in env_names:
            for t in tasks:
                row = [_name_to_des(n), t.upper()]
                row.append(mean_std_to_str(*mlp_entries[(n, t)]))
                row.append(mean_std_to_str(*mtm_specialized_entries[(n, t)]))
                table.append(row)
            table.append(None)

        print("\n")
        print(f"Table: walker")
        print("\n")
        print(generate_latex_table(table))
        print("\n")
    else:
        # for env in d4rl_env_map.keys():
        for env in ["halfcheetah"]:
            # table = [["Dataset", "Task", "MLP", "MTM-Specalized", "MTM", "MTM-Shared"]]
            table = [["Dataset", "Task", "MLP", "MTM-Specalized", "MTM"]]
            env_names = d4rl_env_map[env]

            project = "mlp_baselines"
            mlp_runs = extract_runs(api, project, entity)
            mlp_entries = {}
            for n in env_names:
                for t in tasks:
                    mean, std = mlp_table_entry(mlp_runs, n, t)
                    mlp_entries[(n, t)] = (mean, std)

            project = "d4rl_mtm_paper"
            runs = extract_runs(api, project, entity)
            mtm_specialized_entries = {}
            for n in env_names:
                for t in tasks:
                    mp = [t.upper()]
                    mean, std = mtm_table_entry(runs, n, t, mp)
                    mtm_specialized_entries[(n, t)] = (mean, std)

            project = "d4rl_mtm_cont_1_23"
            project = "d4rl_mtm_cont_1_24"
            runs = extract_runs(api, project, entity)
            mtm_entries = {}
            for n in env_names:
                for t in tasks:
                    mp = ["AUTO_MASK"]
                    mean, std = mtm_table_entry(runs, n, t, mp)
                    mtm_entries[(n, t)] = (mean, std)

            # project = "mtm_specialized"
            # runs = extract_runs(api, project, entity)
            # mtm_shared_entries = {}
            # for n in env_names:
            #     for t in tasks:
            #         mean, std = d4rl_mtm_shared_table_entry(runs, n, t)
            #         mtm_shared_entries[(n, t)] = (mean, std)

            # build rows
            for n in env_names:
                for t in tasks:
                    row = [_name_to_des(n), t.upper()]
                    row.append(mean_std_to_str(*mlp_entries[(n, t)]))
                    row.append(mean_std_to_str(*mtm_specialized_entries[(n, t)]))
                    row.append(mean_std_to_str(*mtm_entries[(n, t)]))
                    # row.append(mean_std_to_str(*mtm_shared_entries[(n, t)]))
                    table.append(row)
                table.append(None)

            caption = f"D4RL {env.capitalize()} Results"
            label = f"tab:d4rl_{env}"
            print(f"\nTable: {env}\n")
            latex_table = generate_latex_table(table, caption, label)
            print(latex_table)
            # save to file
            path = pathlib.Path(f"tables/d4rl_{env}.tex")
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(latex_table)


def adroit(api, entity, test=True) -> None:
    def _name_to_des(name: str) -> str:
        names = name.split("-")[1:]
        return " ".join([n.capitalize() for n in names])

    if test:
        table = [["Dataset", "Task", "MLP", "MTM-Specalized"]]
        env_names = adroit_env_map["door"]

        project = "adroit_final"
        runs = extract_runs(api, project, entity)
        mtm_specialized_entries = {}
        for n in env_names:
            for t in tasks:
                mp = [t.upper()]
                mean, std = mtm_table_entry(runs, n, t, mp, domain="adroit")
                mtm_specialized_entries[(n, t)] = (mean, std)

        project = "adroit_paper"
        mlp_runs = extract_runs(api, project, entity)
        mlp_entries = {}
        for n in env_names:
            for t in tasks:
                mean, std = mlp_table_entry(mlp_runs, n, t, domain="adroit")
                mlp_entries[(n, t)] = (mean, std)

        # build rows
        for n in env_names:
            for t in tasks:
                row = [_name_to_des(n), t.upper()]
                row.append(mean_std_to_str(*mlp_entries[(n, t)]))
                row.append(mean_std_to_str(*mtm_specialized_entries[(n, t)]))
                table.append(row)
            table.append(None)

        print("\n")
        print(f"Table: walker")
        print("\n")
        print(generate_latex_table(table))
        print("\n")
    else:
        for env in adroit_env_map.keys():
            # table = [["Dataset", "Task", "MLP", "MTM-Specalized", "MTM", "MTM-Shared"]]
            table = [["Dataset", "Task", "MLP", "MTM-Specalized", "MTM"]]
            env_names = adroit_env_map[env]

            project = "adroit_paper"
            mlp_runs = extract_runs(api, project, entity)
            mlp_entries = {}
            for n in env_names:
                for t in tasks:
                    mean, std = mlp_table_entry(mlp_runs, n, t, domain="adroit")
                    mlp_entries[(n, t)] = (mean, std)

            project = "adroit_final"
            runs = extract_runs(api, project, entity)
            mtm_specialized_entries = {}
            for n in env_names:
                for t in tasks:
                    mp = [t.upper()]
                    mean, std = mtm_table_entry(runs, n, t, mp, domain="adroit")
                    mtm_specialized_entries[(n, t)] = (mean, std)

            project = "adroit_final"
            runs = extract_runs(api, project, entity)
            mtm_entries = {}
            for n in env_names:
                for t in tasks:
                    mp = ["AUTO_MASK"]
                    mean, std = mtm_table_entry(runs, n, t, mp, domain="adroit")
                    mtm_entries[(n, t)] = (mean, std)

            # project = "mtm_specialized"
            # runs = extract_runs(api, project, entity)
            # mtm_shared_entries = {}
            # for n in env_names:
            #     for t in tasks:
            #         mean, std = d4rl_mtm_shared_table_entry(runs, n, t, domain="adroit")
            #         mtm_shared_entries[(n, t)] = (mean, std)

            # build rows
            for n in env_names:
                for t in tasks:
                    row = [_name_to_des(n), t.upper()]
                    row.append(mean_std_to_str(*mlp_entries[(n, t)]))
                    row.append(mean_std_to_str(*mtm_specialized_entries[(n, t)]))
                    row.append(mean_std_to_str(*mtm_entries[(n, t)]))
                    # row.append(mean_std_to_str(*mtm_shared_entries[(n, t)]))
                    table.append(row)
                table.append(None)

            caption = f"Adroit {env.capitalize()} Results"
            label = f"tab:adroit_{env}"
            print(f"\nTable: {env}\n")
            latex_table = generate_latex_table(table, caption, label)
            print(latex_table)
            # save to file
            path = pathlib.Path(f"tables/adroit_{env}.tex")
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(latex_table)


if __name__ == "__main__":
    api = wandb.Api(timeout=60)
    entity = "mtm_team"

    if True:
        project = "mlp_baselines"
        project = "d4rl_mtm_cont"
        project = "adroit_final"
        project = "d4rl_mtm_cont_1_24"
        runs = extract_runs(api, project, entity)
        # mean, std = mlp_table_entry(runs, "walker2d-medium-replay-v2", "rcbc")
        # mean, std = mtm_table_entry(runs, "walker2d-medium-replay-v2", "rcbc")
        # mean, std = mtm_table_entry(
        #     runs, "door-medium_replay", "rcbc", ["RCBC"], domain="adroit"
        # )
        # print(mean, std)
        mean, std = mtm_table_entry(
            runs, "hopper-medium-replay-v2", "rcbc", ["AUTO_MASK"]
        )
        print(mean, std)
        exit()

    # d4rl(api, entity, test=False)
    # adroit(api, entity, test=True)
