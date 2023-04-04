import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from research.utils.plots import save_fig

green_colors = sns.color_palette("light:#5A9", 5)
purple_colors = sns.color_palette("light:#9184DB", 5)
blue_colors = sns.color_palette("light:#1F77B4", 5)
colors = [green_colors[4], blue_colors[4], purple_colors[4]]

colors = [
    colors[0],
    colors[1],
    colors[2],
]

# mlp_data_means = [110.72, 111.03, 99.48, 81.5, 45.44, 20.62]
# x_ax = [0.95, 0.5, 0.05, 0.02, 0.01, 0.005]
#
# mtm_data_means = [110.79, 110.8, 107.02, 75.61, 73.74, 53.13]
# x_ax = [0.95, 0.5, 0.05, 0.02, 0.01, 0.005]
#
# mtm_s_data_means = [110.97, 110.64, 109.92, 91.12, 85.95, 73.65]
# mtm_s_dp = [0.95, 0.5, 0.05, 0.02, 0.01, 0.005]
mlp_data_means = [110.72, 99.48, 81.5, 45.44, 20.62]
x_ax = np.array([0.95, 0.05, 0.02, 0.01, 0.005]) * 100

mtm_data_means = [110.79, 107.02, 75.61, 73.74, 53.13]

mtm_s_data_means = [110.97, 109.92, 91.12, 85.95, 73.65]


fig, axs = plt.subplots(2, 1, figsize=(4.2 * 1.2, 4 * 1.2), sharex=True)
ax = axs[0]
ax.plot(
    x_ax,
    mlp_data_means,
    label="MLP-BC",
    color=colors[0],
    marker="o",
    linewidth=2,
    markersize=8,
)
ax.plot(
    x_ax,
    mtm_data_means,
    label="MTM",
    color=colors[1],
    marker="o",
    linewidth=2,
    markersize=8,
)
ax.plot(
    x_ax,
    mtm_s_data_means,
    label="MTM-S",
    color=colors[2],
    marker="o",
    linewidth=2,
    markersize=8,
)
ax.set_xscale("log")
ax.set_ylabel("Return\n(D4RL - Hopper)")
# ax.set_title("Data efficiency")
ax.grid(which="major", axis="x", color="lightgray", linestyle="--")
ax.grid(which="major", axis="y", color="lightgray", linestyle="--")

labels = ["MLP", "MTM (Ours)", "Heteromodal MTM (Ours)"]
# thicker_line
handles = [
    Line2D([0], [0], color=colors[idx], label=labels[idx], linewidth=5)
    for idx in range(len(labels))
]

# Create legend
ax.legend(handles=handles, handlelength=3)
# ax.legend()

# set x axis to have numbers
ax.set_xticks(x_ax)
ax.set_xticklabels(x_ax)
ax.set_ylim(bottom=0)

ax = axs[1]
mlp_data_means = [147.7, 72.1, 22.4, 6.678, 4.71]

mtm_data_means = [148.4, 129.2, 115.5, 103.3, 47.7]

mtm_s_data_means = [148.5, 126.8, 116.3, 114.2, 91.03]


# fig, ax = plt.subplots()
ax.plot(
    x_ax,
    mlp_data_means,
    label="MLP-BC",
    color=colors[0],
    marker="o",
    linewidth=2,
    markersize=8,
)
ax.plot(
    x_ax,
    mtm_data_means,
    label="MTM",
    color=colors[1],
    marker="o",
    linewidth=2,
    markersize=8,
)
ax.plot(
    x_ax,
    mtm_s_data_means,
    label="MTM-S",
    color=colors[2],
    marker="o",
    linewidth=2,
    markersize=8,
)
ax.set_xscale("log")
ax.set_xlabel("% of Dataset")
ax.set_ylabel("Return\n(Adroit - Door)")
# ax.legend()

# set x axis to have numbers
ax.set_xticks(x_ax)
ax.set_xticklabels(x_ax)
ax.grid(which="major", axis="x", color="lightgray", linestyle="--")
ax.grid(which="major", axis="y", color="lightgray", linestyle="--")
ax.set_ylim(bottom=0)

# remove extra space
plt.subplots_adjust(hspace=0.05)

save_fig(fig, f"plots/sample_eff.pdf", bbox_inches="tight")

# make the legend
fig, ax = plt.subplots()

# Create legend handles manually
labels = ["MLP", "MTM (Our)", "Heteromodal MTM (Our)"]
# thicker_line
handles = [
    Line2D([0], [0], color=colors[idx], label=labels[idx], linewidth=5)
    for idx in range(len(labels))
]

# Create legend
leg = plt.legend(handles=handles, ncol=4, handlelength=3)
bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# Get current axes object and turn off axis
plt.gca().set_axis_off()
save_fig(fig, f"plots/sample_eff_legend.pdf", bbox_inches=bbox)
