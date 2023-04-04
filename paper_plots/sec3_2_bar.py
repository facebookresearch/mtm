import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

from research.utils.plots import save_fig

model_names = [
    "MLP-BC (Full)",
    "MLP-BC",
    "MLP-RCBC",
    "MTM (Ours)",
    "Heteromodal MTM (Ours)",
]

# Colors
blue_colors = sns.color_palette("light:#1F77B4", 3)
green_colors = sns.color_palette("light:#5A9", 5)
purple_colors = sns.color_palette("light:#9184DB", 10)
colors = [*blue_colors[0:3], purple_colors[3], purple_colors[9]]
# hatches = [None] * 10
hatches = [None, "/", ".", "-", "O", "\\", "|", "+", "x", "o", "*"]


# Create the data for the plot
# Each row represents a model, and the columns represent the mean for each environment
data_mean = np.array(
    [
        [110.7, 93.22, 108.94],
        [53.52, 1.184, 81.9],
        [64.54, 1.736, 84.9],
        [73.74, 13.88, 88.7],
        [87.7, 56.5, 96.4],
    ]
)

# Create the data for the std of the plot
data_std = np.array(
    [
        [0.248, 0.0336, 0.0873],
        [16.6, 0.0344, 19.9],
        [24.58, 0.227, 22.7],
        [24.33, 8.91, 7.47],
        [3.37, 15.71, 8.9],
    ]
)
data_std = data_std / np.sqrt(5)  # to make std error


env_names = [
    "D4RL-Hopper",
    "D4RL-HalfCheetah",
    "D4RL-Walker2D",
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
        capsize=5,
        color=colors[i],
        label=model_names[i],
        hatch=hatches[i],
    )

# Add labels and title
ax.set_xlabel("Environment")
ax.set_ylabel("Return")
ax.set_title("Low data regime")
ax.set_xticks(x + 2 * bar_width)
ax.set_xticklabels(env_names)

# Show the plot
save_fig(fig, f"plots/d4rl_heter_bar.pdf", bbox_inches="tight")


########## Adroit

data_mean = np.array(
    [
        [160.3, 54.85, 147.68, 119.97],
        [96.32, 16.94, 6.796, 0.2],
        [86.26, 17.024, 13.524, 0.22],
        [140.32, 18.38, 110.54, 5.54],
        [124.6, 32.06, 134.36, 9.88],
    ]
)

# Create the data for the std of the plot
data_std = np.array(
    [
        [0.414, 1.78, 0.5263, 67.102],
        [8.27, 2.92, 2.057, 0.086],
        [6.99, 2.249, 11.652, 0.13],
        [24.90, 3.311, 32.75, 4.739],
        [27.319, 7.71, 8.199, 6.387],
    ]
)
data_std = data_std / np.sqrt(5)  # to make std error
env_names = [
    "Adroit-Hammer",
    "Adroit-Pen",
    "Adroit-Door",
    "Adroit-Relocate",
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
        capsize=5,
        color=colors[i],
        label=model_names[i],
        hatch=hatches[i],
    )

# Add labels and title
ax.set_xlabel("Environment")
ax.set_ylabel("Return")
ax.set_title("Low data regime")
ax.set_xticks(x + 2 * bar_width)
ax.set_xticklabels(env_names)

# Show the plot
save_fig(fig, f"plots/adroit_heter_bar.pdf", bbox_inches="tight")


# make the legend
fig, ax = plt.subplots()

# Create legend handles manually
handles = [
    Patch(
        facecolor=colors[i], label=model_names[i], hatch=hatches[i], edgecolor="black"
    )
    for i in range(1, len(model_names))
]
# Create legend
leg = plt.legend(handles=handles, ncol=5, handleheight=4, handlelength=4)
bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

# Get current axes object and turn off axis
plt.gca().set_axis_off()
save_fig(fig, f"plots/legend_heter_bar.pdf", bbox_inches=bbox)
