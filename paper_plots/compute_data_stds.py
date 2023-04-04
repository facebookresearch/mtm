import numpy as np

from research.mtm.datasets.adroit import get_datasets as get_adroit_datasets
from research.mtm.datasets.d4rl_ds import get_datasets as get_d4rl_datasets


def compute_data_stds():
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
            state_diff = (states[:, 11:41, :] - states[:, 10:40, :]) ** 2
            actions = train_dataset.actions
            action_diff = (actions[:, 11:41, :] - actions[:, 10:40, :]) ** 2
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
                state_diff = (state_traj[1:] - state_traj[:-1]) ** 2
                action_diff = (action_traj[1:] - action_traj[:-1]) ** 2
                state_diffs.extend(state_diff)
                action_diffs.extend(action_diff)

            state_diffs = np.array(state_diffs)
            action_diffs = np.array(action_diffs)
            normalize_factor[name] = {
                "states": state_diffs.mean(),
                "actions": action_diffs.mean(),
            }

        # save to file
        np.save("data_stds.npy", normalize_factor)

    import pandas as pd

    # convert the dictionary to a pandas dataframe
    df = pd.DataFrame.from_dict(normalize_factor, orient="index")

    # display the dataframe
    print(df)


if __name__ == "__main__":
    compute_data_stds()
