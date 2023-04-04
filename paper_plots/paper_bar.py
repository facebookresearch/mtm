import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from research.utils.plots import save_fig

mpl.rc("hatch", color="k", linewidth=1 / 4)


# Colors
blue_colors = sns.color_palette("light:#1F77B4", 3)
green_colors = sns.color_palette("light:#5A9", 5)
purple_colors = sns.color_palette("light:#9184DB", 10)
colors = [*blue_colors[0:3], purple_colors[3], purple_colors[9]]
# hatches = [None] * 10
hatches = [None, "/", "*", "--", "O", "\\", "|", "+", "x", "o", "*"]

model_names = [
    "MLP-BC (Full)",
    "MLP-BC",
    "MLP-RCBC",
    "MTM (Ours)",
    "Heteromodal MTM (Ours)",
]


# might change to use AUTO_MASK, ID, RCBC masks

# Create the data for the plot
# Each row represents a model, and the columns represent the mean for each environment
data_mean = np.array(
    [
        [110.7, 93.22, 108.94, 147.8, 61.5],
        [53.52, 1.184, 81.9, 5.796, 23.1],
        [64.54, 1.736, 84.9, 12.76, 25.6],
        [73.74, 13.88, 88.7, 92.03, 21.2],
        [87.7, 56.5, 96.4, 113.5, 29.2],
    ]
)

# Create the data for the std of the plot
data_std = np.array(
    [
        [0.248, 0.0336, 0.0873, 0.374, 2.85],
        [16.6, 0.0344, 19.9, 1.997, 4.39],
        [24.58, 0.227, 22.7, 14.2, 3.59],
        [24.33, 8.91, 7.47, 18.1, 6.69],
        [3.37, 15.71, 8.9, 1.34, 3.76],
    ]
)
data_std = data_std / np.sqrt(5)  # to make std error


env_names = [
    "D4RL\nHopper",
    "D4RL\nHalfCheetah",
    "D4RL\nWalker2D",
    "Adroit\nDoor",
    "Adroit\nPen",
]

assert data_mean.shape == data_std.shape
assert data_mean.shape[0] == len(
    model_names
), f"The number of models must match the number of rows in data_mean. {data_mean.shape[0]} != {len(model_names)}"
assert data_mean.shape[1] == len(env_names)


# Create the stacked bar chart
fig, ax = plt.subplots()
x = np.arange(data_mean.shape[1])
bar_width = 0.15
eps_space = 0.005


for i in range(data_mean.shape[0]):
    x_coord = x + i * bar_width + i * eps_space
    ax.bar(
        x_coord,
        data_mean[i, :],
        bar_width,
        yerr=data_std[i, :],
        capsize=3,
        color=colors[i],
        label=model_names[i],
        hatch=hatches[i],
        align="edge",
    )

# Add labels and title
ax.set_xlabel("Environment")
ax.set_ylabel("Return")
ax.set_xticks(x + 2 * bar_width)
ax.set_xticklabels(env_names)
# legend outside on left side of plot

ax.legend(
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    title="Model",
    title_fontsize=12,
    fontsize=12,
)


# Show the plot
save_fig(fig, f"plots/bar.pdf", bbox_inches="tight")

# make next fig
model_names = ["MLP-BC", "MLP-RCBC", "MTM (Ours)", "Heteromodal MTM (Ours)"]
# remove full bc
full_m = data_mean[0, :]
data_mean = data_mean[1:, :]
data_std = data_std[1:, :]

# normalize by MTM column
m = data_mean[3, :]
data_mean = data_mean / m
data_std = data_std / m

# Create the stacked bar chart
fig, ax = plt.subplots(figsize=(6, 3))
x = np.arange(data_mean.shape[1])
bar_width = 0.15
eps_space = 0.005


for i in range(data_mean.shape[0]):
    x_coord = x + i * bar_width + i * eps_space
    ax.bar(
        x_coord,
        data_mean[i, :],
        bar_width,
        yerr=data_std[i, :],
        capsize=3,
        color=colors[i + 1],
        label=model_names[i],
        hatch=hatches[i + 1],
        align="edge",
    )

# Show the plot
# Add labels and title
ax.set_xlabel("Environment")
ax.set_ylabel("Relative Performance")
ax.set_xticks(x + 2 * bar_width)
ax.set_xticklabels(env_names)
# legend outside on left side of plot

# ax.legend(
#     loc="center left",
#     bbox_to_anchor=(1, 0.5),
#     title="Model",
#     title_fontsize=12,
#     fontsize=12,
# )

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.38),
    # title="Model",
    title_fontsize=12,
    fontsize=12,
    ncol=2,
    handleheight=2,
    handlelength=3,
)

save_fig(fig, f"plots/paper_bar.pdf", bbox_inches="tight")


# # normalize by full column
# data_mean = data_mean * m / full_m * 100
# data_std = data_std * m / full_m * 100
#
# # Create the stacked bar chart
# fig, ax = plt.subplots()
# x = np.arange(data_mean.shape[1])
# bar_width = 0.15
# eps_space = 0.005
#
#
# for i in range(data_mean.shape[0]):
#     x_coord = x + i * bar_width + i * eps_space
#     ax.bar(
#         x_coord,
#         data_mean[i, :],
#         bar_width,
#         yerr=data_std[i, :],
#         capsize=5,
#         color=colors[i + 1],
#         label=model_names[i],
#         hatch=hatches[i + 1],
#     )
#
# # Show the plot
# # Add labels and title
# ax.set_xlabel("Environment")
# ax.set_ylabel("Percent of Max Performance")
# ax.set_xticks(x + 2 * bar_width)
# ax.set_xticklabels(env_names)
# # legend outside on left side of plot
#
# ax.legend(
#     loc="upper center",
#     bbox_to_anchor=(0.5, 1.25),
#     title="Model",
#     title_fontsize=12,
#     fontsize=12,
#     ncol=2,
# )
#
# save_fig(fig, f"plots/paper_bar_percent.pdf", bbox_inches="tight")

# make the legend
#  fig, ax = plt.subplots()
#
#  # Create legend handles manually
#  handles = [
#      Patch(
#          facecolor=colors[i + 1],
#          label=model_names[i],
#          hatch=hatches[i + 1],
#          edgecolor="black",
#      )
#      for i in range(len(model_names))
#  ]
#  # Create legend
#  leg = plt.legend(handles=handles, ncol=5, handleheight=4 / 2, handlelength=6 / 2)
#  # leg = plt.legend(handles=handles, ncol=5)
#  bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#
#  # Get current axes object and turn off axis
#  plt.gca().set_axis_off()
#  save_fig(fig, f"plots/paper_hetero_legend.pdf", bbox_inches=bbox)
