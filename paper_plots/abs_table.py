import hashlib
import pathlib
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb

from research.utils.latex import color_text, create_color_map, generate_latex_table
from research.utils.plots import save_fig
from research.utils.wandb import extract_runs, filter_runs

mpl.rc("hatch", color="k", linewidth=1 / 4)


def _domain_to_name(s: str):
    if s == "d4rl":
        return "D4RL"
    elif s == "adroit":
        return "Adroit"
    else:
        raise NotImplementedError


def _dataset_to_env(s: str):
    if "hopper" in s:
        return "Hopper"
    elif "walker2d" in s:
        return "Walker2D"
    elif "halfcheetah" in s:
        return "HalfCheetah"
    elif "door" in s:
        return "Door"
    elif "hammer" in s:
        return "Hammer"
    elif "relocate" in s:
        return "Relocate"
    elif "pen" in s:
        return "Pen"
    else:
        raise NotImplementedError(f"In _dataset_to_env {s}")


def _dataset_to_name(s: str):
    if "medium" in s and "expert" in s:
        return "Medium Expert"
    elif "expert" in s:
        return "Expert"
    elif "medium" in s and "replay" in s:
        return "Medium Replay"
    elif "medium" in s:
        return "Medium"
    else:
        raise NotImplementedError


def get_max_key(dict_values):
    best_value = -np.inf
    best_key = ""
    for k, v in dict_values.items():
        if v > best_value:
            best_value = v
            best_key = k
    return best_key


SEEDS = 4
FRAC_DONE = 0.9

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
    if len(runs) < SEEDS:
        print(f"Warning mlp_table_entry: {env_name} {task} has {len(runs)} runs")
    else:
        runs = runs[:SEEDS]

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

        best_key = get_max_key(candidates_means)
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
                    values.append(max_metric)
                elif task == "fd":
                    metrics = r.history(keys=["eval/fd_state_error"])
                    max_metric = np.min(metrics["eval/fd_state_error"])
                    values.append(max_metric)
                else:
                    raise ValueError(f"Unknown task {task}")
            except Exception as e:
                print(e)
                import ipdb

                ipdb.set_trace()

                print(r.name)

    return np.mean(values), np.std(values)


FRAC_DONE = 0.2


def _make_mtm_d4rl_select_fn(
    env_name: str, mp: List[str]
) -> Callable[[wandb.run], bool]:
    def _s(r: wandb.run) -> bool:
        if r.config["dataset"]["env_name"] == env_name:
            if set(r.config["args"]["mask_patterns"]) == set(mp):
                # insure trained to at least 0.8 of desired steps
                if "_step" in r.summary:
                    if (
                        # r.config["args"]["num_train_steps"] * FRAC_DONE
                        91000
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
                        # r.config["args"]["num_train_steps"] * FRAC_DONE
                        91000
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
        return np.min(metrics[k])
    elif task == "fd":
        k = "eval/fd_state_error_r=1"
        metrics = run.history(keys=[k])
        return np.min(metrics[k])
    else:
        raise ValueError(f"Unknown task {task}")


def mtm_table_entry(
    runs: List[wandb.run],
    env_name: str,
    task: str,
    mp: List[str],
    domain: str = "d4rl",
):
    if domain == "d4rl":
        select_fn = _make_mtm_d4rl_select_fn(env_name, mp)
    elif domain == "adroit":
        select_fn = _make_mtm_adroit_select_fn(env_name, mp)
    else:
        raise ValueError(f"Unknown domain {domain}")

    runs = filter_runs(select_fn, runs)
    if len(runs) < SEEDS:
        print(f"Warning mtm_table_entry: {env_name} {task} has {len(runs)} runs")
    else:
        runs = runs[:SEEDS]

    values = []
    select_key = None
    if task == "rcbc":
        avalible = set(runs[0].summary.keys())
        candidates = defaultdict(list)
        for r in runs:
            keys = [f"eval2/p={i}_return_mean" for i in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]]
            keys += [f"eval5/p={i}_return_mean" for i in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]]
            keys += [
                f"eval_ts/p={i}_return_mean" for i in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
            ]
            # keys += [
            #     f"eval_cem/p={i}_return_mean" for i in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
            # ]
            keys = [k for k in keys if k in avalible]
            metrics = r.history(keys=keys)
            for k in keys:
                try:
                    candidates[k].append(np.max(metrics[k]))
                except:
                    candidates[k].append(np.max(r.history(keys=[k])[k]))

        candidates_means = {}
        for k in keys:
            candidates_means[k] = np.mean(candidates[k])

        best_key = get_max_key(candidates_means)
        select_key = best_key
        values = candidates[best_key]
    else:
        for r in runs:
            values.append(_mtm_get_value(r, task))

    return np.mean(values), np.std(values), select_key


def mtm_table_entries(
    runs: List[wandb.run],
    env_name: str,
    tasks: List[str],
    mp: List[str],
    domain: str = "d4rl",
):
    task_to_key_map = {
        "bc": "eval1/return_mean",
        "id": "eval/id_action_error_r=1",
        "fd": "eval/fd_state_error_r=1",
    }
    if domain == "d4rl":
        select_fn = _make_mtm_d4rl_select_fn(env_name, mp)
    elif domain == "adroit":
        select_fn = _make_mtm_adroit_select_fn(env_name, mp)
    else:
        raise ValueError(f"Unknown domain {domain}")

    runs = filter_runs(select_fn, runs)
    if len(runs) < SEEDS:
        print(f"Warning mtm_table_entry: {env_name} {tasks} has {len(runs)} runs")
    else:
        runs = runs[:SEEDS]

    # first task is rcbc
    avalible = set(runs[0].summary.keys())
    candidates = defaultdict(
        list
    )  # mapping from rcbc return condition percentage to a list of max returns per run
    all_values = defaultdict(
        lambda: defaultdict(list)
    )  # mapping from                             task to a list of values per run for the best rcbc value

    keys_of_interest = []
    for t in tasks:
        if t == "rcbc":
            pass
        else:
            keys_of_interest.append(task_to_key_map[t])

    for r in runs:
        keys = [f"eval2/p={i}_return_mean" for i in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]]
        keys += [f"eval5/p={i}_return_mean" for i in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]]
        keys += [f"eval_ts/p={i}_return_mean" for i in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]]
        # keys += [f"eval_cem/p={i}_return_mean" for i in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]]
        keys = [k for k in keys if k in avalible]
        # deep copy of keys
        rcbc_keys = keys[:]

        metrics = r.history(keys=keys + keys_of_interest)
        for k in rcbc_keys:
            index_of_max = np.argmax(metrics[k])
            candidates[k].append(metrics[k][index_of_max])
            for koi in keys_of_interest:
                all_values[koi][k].append(metrics[koi][index_of_max])

    # the above code extracts, for each run, the best rcbc value and the corresponding values for the other tasks

    candidates_means = {}
    for k in keys:
        candidates_means[k] = np.mean(candidates[k])
    best_key = get_max_key(candidates_means)
    # best_key is the rcbc return condition percentage that gives the best mean return

    # finally we can extract the values for the other tasks
    means = []
    stds = []
    for t in tasks:
        if t == "rcbc":
            means.append(np.mean(candidates[best_key]))
            stds.append(np.std(candidates[best_key]))
        else:
            means.append(np.mean(all_values[task_to_key_map[t]][best_key]))
            stds.append(np.std(all_values[task_to_key_map[t]][best_key]))

    return means, stds, best_key


def mean_std_to_str(
    mean: float, std: float, normalize_value: float = 1.0, use_three: bool = False
) -> str:
    mean = mean / normalize_value
    std = std / normalize_value
    if use_three:
        return f"{mean:.3f} $\pm$ {std:.3f}"
    else:
        return f"{mean:.2f} $\pm$ {std:.2f}"


def compute_data_stds():
    from research.mtm.datasets.adroit import get_datasets as get_adroit_datasets
    from research.mtm.datasets.d4rl_ds import get_datasets as get_d4rl_datasets

    # for each environment
    # compute s_t+1 - s_t-1 mean
    # compute a_t+1 - a_t-1 mean

    normalize_factor = {}
    for e in ["relocate", "pen", "hammer", "door"]:
        for de in ["expert", "medium_replay", "full_replay"]:
            name = f"adroit_{e}_{de}"
            train_dataset, val_dataset = get_adroit_datasets(
                seq_steps=10,
                env_name=e,
                d_name=de,
                use_achieved=True,
            )
            states = train_dataset.states
            state_diff = states[:, 1, :] - states[:, 0, :]
            actions = train_dataset.actions
            action_diff = actions[:, 1, :] - actions[:, 0, :]
            normalize_factor[name] = {
                "states": state_diff.mean(),
                "actions": action_diff.mean(),
            }

    for e in ["walker2d", "hopper", "halfcheetah"]:
        for de in ["medium-replay-v2", "medium-v2", "expert-v2", "medium-expert-v2"]:
            name = f"d4rl_{e}_{de}"
            env_name = f"{e}-{de}"
            train_dataset, val_dataset = get_d4rl_datasets(
                seq_steps=10, env_name=env_name
            )
            action_diffs = []
            state_diffs = []
            for state_traj, action_traj in zip(
                train_dataset.observations_segmented, train_dataset.actions_segmented
            ):
                state_diff = state_traj[1:] - state_traj[:-1]
                action_diff = action_traj[1:] - action_traj[:-1]
                state_diffs.extend(state_diff)
                action_diffs.extend(action_diff)

            state_diffs = np.array(state_diffs)
            action_diffs = np.array(action_diffs)
            normalize_factor[name] = {
                "states": state_diffs.mean(axis=0),
                "actions": action_diffs.mean(axis=0),
            }

        # save to file
        np.save("data_stds.npy", normalize_factor)


def _hash_string(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def _get_project(model: str, domain: str, dataset: str, mask: str = None) -> str:
    if model == "MLP" and domain == "d4rl":
        project = "mlp_baselines"
    elif model == "MTM" and domain == "d4rl" and mask == "auto":
        # project = "d4rl_mtm_cont_1_24"
        # if "halfcheetah" in dataset:
        #     project = "half"
        project = "d4rl_1_26_mtm"
    elif model == "MTM" and domain == "d4rl":
        project = "d4rl_mtm_paper"
    elif model == "MLP" and domain == "adroit":
        project = "adroit_paper"
    elif model == "MTM" and domain == "adroit":
        project = "adroit_final"
        if "relocate" in dataset:
            project = "adroit_relocate_1_25"
    else:
        raise NotImplementedError
    return project


def generate_mtm_table(
    api: wandb.Api,
    entity: str,
    domain_dataset: List[Tuple[str, str]],
    name: str,
    # ignore_cols: Tuple[str] = (),
) -> None:
    """Generate a table for a single domain/dataset/task combination."""

    with open("data_stds.npy", "rb") as f:
        data_stds = np.load(f, allow_pickle=True).item()

    table = [
        [
            "Domain",
            "Dataset",
            "Task",
            "MLP",
            "S-MTM (Ours)",
            "MTM (Ours)",
            "(MTM) - (S-MTM)",
            # "MTM-Best (Ours)",
        ]
    ]

    combined_string = "".join(table[0])
    for domain, dataset in domain_dataset:
        combined_string += f"{domain}_{dataset} "

    hash_code = _hash_string(combined_string)
    path = Path(f"hash_{hash_code}.npy")

    if not path.exists():
        mlp_entries = {}
        mtm_specialized_entries = {}
        mtm_entries = {}
        mtm_shared_entries = {}

        for domain, dataset in domain_dataset:
            model = "MLP"
            project = _get_project(model, domain, dataset)
            runs = extract_runs(api, project, entity)
            for task in tasks:
                mean, std = mlp_table_entry(runs, dataset, task, domain=domain)
                mlp_entries[(domain, dataset, task)] = (mean, std)

            model = "MTM"
            project = _get_project(model, domain, dataset)
            runs = extract_runs(api, project, entity)
            for task in tasks:
                mp = [task.upper()]
                mean, std, _ = mtm_table_entry(runs, dataset, task, mp, domain=domain)
                mtm_specialized_entries[(domain, dataset, task)] = (mean, std)

            model = "MTM"
            project = _get_project(model, domain, dataset, mask="auto")
            runs = extract_runs(api, project, entity)
            mp = ["AUTO_MASK"]

            for task in tasks:
                mean, std, select_key = mtm_table_entry(
                    runs, dataset, task, mp, domain=domain
                )
                mtm_entries[(domain, dataset, task)] = (mean, std)

            model = "MTM"
            project = _get_project(model, domain, dataset, mask="auto")
            runs = extract_runs(api, project, entity)
            means, stds, _ = mtm_table_entries(runs, dataset, tasks, mp, domain=domain)
            for task, mean, std in zip(tasks, means, stds):
                mtm_shared_entries[(domain, dataset, task)] = (mean, std)

        # save to_file
        np.save(
            path,
            {
                "mlp": mlp_entries,
                "mtm_specialized": mtm_specialized_entries,
                "mtm": mtm_entries,
                "mtm_shared": mtm_shared_entries,
            },
        )
    else:
        with open(path, "rb") as f:
            data = np.load(f, allow_pickle=True).item()
            mlp_entries = data["mlp"]
            mtm_specialized_entries = data["mtm_specialized"]
            mtm_entries = data["mtm"]
            mtm_shared_entries = data["mtm_shared"]

    # build rows
    count = 0
    for domain, dataset in domain_dataset:
        for idx, task in enumerate(tasks):
            if idx in [0, 3]:
                row = ["", _dataset_to_name(dataset), task.upper()]
            elif idx == 1:
                row = [
                    domain.upper() if "4" in domain else domain.capitalize(),
                    _dataset_to_name(dataset),
                    task.upper(),
                ]
            elif idx == 2:
                row = [
                    _dataset_to_env(dataset),
                    _dataset_to_name(dataset),
                    task.upper(),
                ]
            else:
                raise NotImplementedError

            if domain == "d4rl":
                _temp = dataset.replace("-", "_", 1)
                index = f"{domain}_{_temp}"
            else:
                _temp = dataset.replace("-", "_")
                index = f"{domain}_{_temp}"

            if task == "id":
                normalize_value = data_stds[index]["actions"]
            elif task == "fd":
                normalize_value = data_stds[index]["states"]
            else:
                normalize_value = 1

            row.append(
                mean_std_to_str(
                    *mlp_entries[(domain, dataset, task)],
                    normalize_value=normalize_value,
                    use_three=normalize_value != 1,
                )
            )
            row.append(
                mean_std_to_str(
                    *mtm_specialized_entries[(domain, dataset, task)],
                    normalize_value=normalize_value,
                    use_three=normalize_value != 1,
                )
            )
            if idx == 0:
                add_m = f"\\tikzmark{{top left {count}}}"
                add_e = ""
            elif idx == 3:
                add_m = ""
                add_e = f"\\tikzmark{{bottom right {count}}}"
                count += 1
            else:
                add_m = ""
                add_e = ""
            row.append(
                add_m
                + mean_std_to_str(
                    *mtm_shared_entries[(domain, dataset, task)],
                    normalize_value=normalize_value,
                    use_three=normalize_value != 1,
                )
                + add_e
            )

            m1, s1 = mtm_specialized_entries[(domain, dataset, task)]
            m2, s2 = mtm_shared_entries[(domain, dataset, task)]
            mean, std = m2 - m1, s2 + s1
            entry = mean_std_to_str(
                mean,
                std,
                normalize_value=normalize_value,
                use_three=normalize_value != 1,
            )

            if task in ["id", "fd"]:
                n_c = "0.0,1.0,0.0"
                p_c = "1.0,0.0,0.0"
            else:
                n_c = "1.0,0.0,0.0"
                p_c = "0.0,1.0,0.0"

            if normalize_value == 1:
                neg_color = create_color_map(n_c, "0.0,0.0,0.0", -2.00, -0.10, 100)
                pos_color = create_color_map("0.0,0.0,0.0", p_c, 0.10, 2.00, 2)
            else:
                neg_color = create_color_map(n_c, "0.0,0.0,0.0", -1.0, -0.1, 100)
                pos_color = create_color_map("0.0,0.0,0.0", p_c, 0.1, 1.0, 2)
            pos_color[100] = p_c

            if mean > 0:
                row.append(color_text(entry, mean / normalize_value, pos_color))
            else:
                row.append(color_text(entry, mean / normalize_value, neg_color))

            # row.append(
            #     mean_std_to_str(
            #         *mtm_entries[(domain, dataset, task)],
            #         normalize_value=normalize_value,
            #     )
            # )

            # remove_idxs = []
            # for idx, header in enumerate(table[0]):
            #     if header in ignore_cols:
            #         remove_idxs.append(idx)
            #
            # for idx in sorted(remove_idxs, reverse=True):
            #     del row[idx]

            table.append(row)
        table.append(None)

    add_extra = ""
    for i in range(count):
        add_extra += (
            "\DrawBox[ultra thick, gray]{top left "
            + str(i)
            + "}{bottom right "
            + str(i)
            + "}\n"
        )

    caption = f"{name} Results".replace("-", " ").replace("_", " ")
    label = f"tab:{name}"
    print(f"\nTable: MTM\n")
    latex_table = generate_latex_table(
        table, caption, label, small=True, table_extras=add_extra
    )
    print(latex_table)
    # save to file
    path = pathlib.Path(f"tables/{name}.tex")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(latex_table)


def core_table(
    api: wandb.Api,
    entity: str,
) -> None:
    domain_dataset: List[Tuple[str, str]] = [
        ("d4rl", "hopper-expert-v2"),
        ("d4rl", "hopper-medium-replay-v2"),
        ("adroit", "door-expert"),
        ("adroit", "door-medium_replay"),
    ]
    generate_mtm_table(api, entity, domain_dataset, "core_mtm")


def d4rl_tables(api: wandb.Api, entity: str) -> None:
    # hopper table
    domain_dataset: List[Tuple[str, str]] = [
        ("d4rl", "hopper-expert-v2"),
        ("d4rl", "hopper-medium-expert-v2"),
        ("d4rl", "hopper-medium-v2"),
        ("d4rl", "hopper-medium-replay-v2"),
        ("d4rl", "walker2d-expert-v2"),
        ("d4rl", "walker2d-medium-expert-v2"),
        ("d4rl", "walker2d-medium-v2"),
        ("d4rl", "walker2d-medium-replay-v2"),
        ("d4rl", "halfcheetah-expert-v2"),
        ("d4rl", "halfcheetah-medium-expert-v2"),
        ("d4rl", "halfcheetah-medium-v2"),
        ("d4rl", "halfcheetah-medium-replay-v2"),
    ]
    generate_mtm_table(api, entity, domain_dataset, "d4rl_all")


def adroit_tables(api: wandb.Api, entity: str) -> None:
    # door table
    domain_dataset: List[Tuple[str, str]] = [
        ("adroit", "door-expert"),
        ("adroit", "door-medium_replay"),
        ("adroit", "pen-expert"),
        ("adroit", "pen-medium_replay"),
        # ("adroit", "relocate-expert"),
        # ("adroit", "relocate-medium_replay"),
        # ("adroit", "hammer-expert"),
        # ("adroit", "hammer-medium_replay"),
    ]
    generate_mtm_table(api, entity, domain_dataset, "adroit_all")


def extract_masked_results(
    api: wandb.Api,
    entity: str,
    domain_dataset: List[Tuple[str, str]],
    masks_list: List[List[str]],
    name: str,
    task="rcbc",
) -> None:
    """Generate a table for a single domain/dataset/task combination."""
    blue_colors = sns.color_palette("light:#1F77B4", 3)
    green_colors = sns.color_palette("light:#5A9", 3)
    purple_colors = sns.color_palette("light:#9184DB", 3)
    # colors = [*blue_colors[1:], *green_colors[1:], *purple_colors[1:]]
    colors = [blue_colors[2], purple_colors[2], green_colors[2]]
    # hatches = [None] * 10
    # hatches = [None, "/", ".", "-", "O", "\\", "|", "+", "x", "o", "*"]
    # hatches = [None, "\\", None, "\\", None, None]
    hatches = ["//", "*", None, "//", None, None]
    # hatches = [None, "//", None, "//", None, None]

    with open("data_stds.npy", "rb") as f:
        data_stds = np.load(f, allow_pickle=True).item()

    flat_masks = [item for sublist in masks_list for item in sublist]
    combined_string = task + name + "".join(flat_masks)
    for domain, dataset in domain_dataset:
        combined_string += f"{domain}_{dataset} "

    hash_code = _hash_string(combined_string)
    path = Path(f"hash_{hash_code}.npy")
    print(path)

    if not path.exists():
        mtm_entries = {}
        for domain, dataset in domain_dataset:
            for masks in masks_list:
                model = "MTM"
                u_masks = masks
                # if (
                #     masks == ["AUTO_MASK"]
                #     or masks == ["AUTO_MASK", "RCBC"]
                #     and dataset != "hopper-expert-v2"
                # ):
                #     project = "d4rl_mtm_cont_1_24"
                #     if masks == ["AUTO_MASK", "RCBC"]:
                #         u_masks = ["AUTO_MASK", "AUTO_MASK", "RCBC"]
                # elif dataset == "halfcheetah-expert-v2":
                #     project = "half"
                if masks == ["AUTO_MASK"]:
                    project = "d4rl_1_26_mtm"
                else:
                    project = "d4rl_mtm_mask_ablation"
                runs = extract_runs(api, project, entity)
                mean, std, _ = mtm_table_entry(
                    runs, dataset, task, u_masks, domain=domain
                )
                mtm_entries[(domain, dataset, tuple(masks))] = (mean, std)

        # save to_file
        np.save(path, {"mtm": mtm_entries})
    else:
        with open(path, "rb") as f:
            data = np.load(f, allow_pickle=True).item()
            mtm_entries = data["mtm"]

    # make bar plot
    _mask_names = [str(m) for m in masks_list]
    _mask_names = [m.replace("[", "").replace("]", "") for m in _mask_names]
    masks_names = []
    for m in _mask_names:
        if "RCBC" in m:
            masks_names.append("RCBC (Specialized)")
        elif "AUTO_MASK" in m:
            masks_names.append("Random Autoregressive (Ours)")
        elif "RANDOM" in m:
            masks_names.append("Random (BERT / MAE)")
        else:
            raise NotADirectoryError()

    # Create the data for the plot
    # Each row represents a mask, and the columns represent the mean for each environment

    # MTM
    data_mean = []
    data_std = []

    for mask in masks_list:
        row_means = []
        row_stds = []
        for domain, dataset in domain_dataset:
            mean, std = mtm_entries[(domain, dataset, tuple(mask))]
            row_means.append(mean)
            row_stds.append(std)
        data_mean.append(row_means)
        data_std.append(row_stds)

    # convert to np arrays
    data_mean = np.array(data_mean)
    data_std = np.array(data_std)

    # plot
    # fig size 4, 3
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(data_mean.shape[1])
    bar_width = 0.2
    eps_space = 0.005

    # specialized_perf = data_mean[2, :]
    specialized_perf = 1

    for i in range(data_mean.shape[0]):
        x_coord = x + i * bar_width + i * eps_space + bar_width
        ax.bar(
            x_coord,
            data_mean[i, :] / specialized_perf,
            bar_width,
            yerr=data_std[i, :] / np.sqrt(SEEDS) / specialized_perf,
            capsize=3,
            color=colors[i],
            label=masks_names[i],
            hatch=hatches[i],
            align="edge" if data_mean.shape[0] % 2 == 0 else "center",
        )

    # Add labels and title
    ax.set_xlabel("Environment")
    # ax.set_ylabel("relative performance")
    ax.set_ylabel("Return")
    ax.set_xticks(x + 2 * bar_width)

    names = []
    for _, dataset in domain_dataset:
        n = f"{_dataset_to_env(dataset)}\n{_dataset_to_name(dataset)}"
        names.append(n)
    ax.set_xticklabels(names)

    # legend outside on left side of plot
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        #  title="Train Mask Pattern",
        title_fontsize=12,
        ncol=1,
        fontsize=12,
        handleheight=2,
        handlelength=3,
    )

    # ax.legend(
    #     loc="center left",
    #     bbox_to_anchor=(1, 0.5),
    #     title="Train Mask Pattern",
    #     title_fontsize=12,
    #     fontsize=12,
    # )

    # Show the plot
    save_fig(fig, f"plots/mask_ablation_{task}_{name}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    api = wandb.Api(timeout=30)
    entity = "mtm_team"
    for task in ["rcbc", "id"]:
        extract_masked_results(
            api,
            entity,
            [
                ("d4rl", "hopper-expert-v2"),
                ("d4rl", "hopper-medium-replay-v2"),
                ("d4rl", "walker2d-expert-v2"),
                ("d4rl", "walker2d-medium-replay-v2"),
                # ("d4rl", "halfcheetah-expert-v2"),
                # ("d4rl", "halfcheetah-medium-replay-v2"),
            ],
            [
                ["RANDOM"],
                ["AUTO_MASK"],
                ["RCBC"],
                # ["RANDOM", "RCBC"],
                # ["AUTO_MASK", "RCBC"],
                # ["BC"],
            ],
            "paper",
            task,
        )

    # if False:
    #     project = "mlp_baselines"
    #     project = "d4rl_mtm_cont"
    #     project = "adroit_final"
    #     project = "d4rl_mtm_cont_1_24"
    #     runs = extract_runs(api, project, entity)
    #     # mean, std = mlp_table_entry(runs, "walker2d-medium-replay-v2", "rcbc")
    #     # mean, std = mtm_table_entry(runs, "walker2d-medium-replay-v2", "rcbc")
    #     # mean, std = mtm_table_entry(
    #     #     runs, "door-medium_replay", "rcbc", ["RCBC"], domain="adroit"
    #     # )
    #     # print(mean, std)
    #     mean, std = mtm_table_entry(
    #         runs, "hopper-medium-replay-v2", "rcbc", ["AUTO_MASK"]
    #     )
    # #     print(mean, std)
    # #     exit()
    #
    # core_table(api, entity)
    # d4rl_tables(api, entity)
    # adroit_tables(api, entity)
