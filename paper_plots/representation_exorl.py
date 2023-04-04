import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from matplotlib.lines import Line2D

from research.utils.plots import save_fig


def reduce(values, reducer=np.nanmean, *args, **kwargs):
    with warnings.catch_warnings():  # Buckets can be empty.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return reducer(values, *args, **kwargs)


def binning(xs, ys, borders, reducer=np.nanmean, fill="nan"):
    xs = xs if isinstance(xs, np.ndarray) else np.array(xs)
    ys = ys if isinstance(ys, np.ndarray) else np.array(ys)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    binned = []
    for start, stop in zip(borders[:-1], borders[1:]):
        left = (xs <= start).sum()
        right = (xs <= stop).sum()
        if left < right:
            value = reduce(ys[left:right], reducer)
        elif binned:
            value = {"nan": np.nan, "last": binned[-1]}[fill]
        else:
            value = np.nan
        binned.append(value)
    return borders[1:], np.array(binned)


api = wandb.Api(timeout=20)
entity, project = "mtm_team", "rep_exorl_experiments"

runs = api.runs(entity + "/" + project)
print(f"Found {len(runs)} runs in {entity}/{project}")

# Group runs


def g(r):
    return r.config["task"]


task_map = defaultdict(list)
for run in runs:
    task_map[g(run)].append(run)


# Grab the eval sub-runs for each seed.


def g(r):
    if r.config["agent"]["name"] == "td3":
        return "td3"
    else:
        if r.config["agent"]["end_to_end"]:
            return "td3_mtm_e2e"
        else:
            return "td3_mtm"


plot_map = defaultdict(lambda: defaultdict(list))
for k, runs in task_map.items():
    for run in runs:
        plot_map[k][g(run)].append(run.history(keys=["eval/episode_reward"]))
        print(run.name)

green_colors = sns.color_palette("light:#5A9", 5)
purple_colors = sns.color_palette("light:#9184DB", 5)
blue_colors = sns.color_palette("light:#1F77B4", 5)
colors = [green_colors[4], blue_colors[4], purple_colors[4]]

color_map = {
    "td3": colors[0],
    "td3_mtm": colors[1],
    "td3_mtm_e2e": colors[2],
}
gray_color = (0.5, 0.5, 0.5)


bins = 300
borders = np.linspace(0, 100000, bins)
for env in plot_map.keys():
    # Plot the time series, with the mean and standard deviation over the 1M steps.
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    for model_name, series in plot_map[env].items():
        rew_per_step = defaultdict(list)
        for s in series:
            for r in np.array(s):
                rew_per_step[r[0]].append(r[1])
        xs = []
        ys = []
        stds = []
        sorted_steps = list(rew_per_step.keys())
        sorted_steps.sort()
        for step in sorted_steps:
            xs.append(step)
            ys.append(np.mean(rew_per_step[step]))
            stds.append(np.std(rew_per_step[step]))

        xs = np.array(xs)
        ys = np.array(ys)
        stds = np.array(stds)

        _, ys = binning(xs, ys, borders, reducer=np.nanmean)
        xs, stds = binning(xs, stds, borders, reducer=np.nanmean)

        if model_name == "td3":
            max_ret = np.mean(ys[-10:][~np.isnan(ys[-10:])])
            # plot the max return as a dotted horizontal line
            ax.axhline(
                max_ret,
                color=gray_color,
                linestyle="--",
                linewidth=3,
                alpha=0.5,
                label="Asymptotic TD3",
            )

        truncate = 75

        # Make x-axis steps in units of 1M.
        xs = xs[:truncate]
        ys = ys[:truncate]
        stds = stds[:truncate]

        ax.plot(
            xs,
            ys,
            label=f"model_name={model_name}",
            color=color_map[model_name],
            linewidth=3,
        )
        ax.fill_between(
            xs, ys - stds, ys + stds, alpha=0.2, color=color_map[model_name]
        )

    ax.set_xlabel("Training Steps", fontsize=16)
    ax.set_xlim(right=25001)
    ax.set_ylabel("Return", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=12)
    # ax.set_title(f"Task: {env}"6

    ax.grid(which="major", axis="x", color="lightgray", linestyle="--")
    ax.grid(which="major", axis="y", color="lightgray", linestyle="--")

    save_fig(fig, f"rep_plots/representation_{env}.pdf", bbox_inches="tight")

# make the legend
fig, ax = plt.subplots()

# Create legend handles manually
labels = ["Base TD3", "MTM (Frozen)", "MTM (Finetuned)"]
# thicker_line
handles = [
    Line2D([0], [0], color=color_map[m], label=labels[idx], linewidth=3)
    for idx, m in enumerate(["td3", "td3_mtm", "td3_mtm_e2e"])
]
# add legend for asymptotic return
handles.append(
    Line2D(
        [0],
        [0],
        color=gray_color,
        linestyle="--",
        alpha=0.5,
        label="Asymptotic TD3",
        linewidth=3,
    )
)


# Create legend
leg = plt.legend(handles=handles, ncol=4, handlelength=3)
bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

# Get current axes object and turn off axis
plt.gca().set_axis_off()
save_fig(fig, f"rep_plots/legend.pdf", bbox_inches=bbox)
# save_fig(fig, f"rep_plots/legend.pdf")
