# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Point-mass domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources
from dm_env import specs
import numpy as np
import os

_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()


TASKS = [
    ("reach_top_left", np.array([-0.15, 0.15, 0.01])),
    ("reach_top_right", np.array([0.15, 0.15, 0.01])),
    ("reach_bottom_left", np.array([-0.15, -0.15, 0.01])),
    ("reach_bottom_right", np.array([0.15, -0.15, 0.01])),
]


def make(task, task_kwargs=None, environment_kwargs=None, visualize_reward=False):
    task_kwargs = task_kwargs or {}
    if environment_kwargs is not None:
        task_kwargs = task_kwargs.copy()
        task_kwargs["environment_kwargs"] = environment_kwargs
    env = SUITE[task](**task_kwargs)
    env.task.visualize_reward = visualize_reward
    return env


def get_model_and_assets(task):
    """Returns a tuple containing the model XML string and a dict of assets."""
    root_dir = os.path.dirname(os.path.dirname(__file__))
    xml = resources.GetResource(
        os.path.join(root_dir, "custom_dmc_tasks", f"point_mass_maze_{task}.xml")
    )
    return xml, common.ASSETS


@SUITE.add("benchmarking")
def reach_top_left(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets("reach_top_left"))
    task = MultiTaskPointMassMaze(target_id=0, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


@SUITE.add("benchmarking")
def reach_top_right(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets("reach_top_right"))
    task = MultiTaskPointMassMaze(target_id=1, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


@SUITE.add("benchmarking")
def reach_bottom_left(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets("reach_bottom_left"))
    task = MultiTaskPointMassMaze(target_id=2, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


@SUITE.add("benchmarking")
def reach_bottom_right(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets("reach_bottom_right"))
    task = MultiTaskPointMassMaze(target_id=3, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


class Physics(mujoco.Physics):
    """physics for the point_mass domain."""

    def mass_to_target_dist(self, target):
        """Returns the distance from mass to the target."""
        d = target - self.named.data.geom_xpos["pointmass"]
        return np.linalg.norm(d)


class MultiTaskPointMassMaze(base.Task):
    """A point_mass `Task` to reach target with smooth reward."""

    def __init__(self, target_id, random=None):
        """Initialize an instance of `PointMassMaze`.

        Args:
          randomize_gains: A `bool`, whether to randomize the actuator gains.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._target = TASKS[target_id][1]
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

           If _randomize_gains is True, the relationship between the controls and
           the joints is randomized, so that each control actuates a random linear
           combination of joints.

        Args:
          physics: An instance of `mujoco.Physics`.
        """
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        physics.data.qpos[0] = np.random.uniform(-0.29, -0.15)
        physics.data.qpos[1] = np.random.uniform(0.15, 0.29)
        # import ipdb; ipdb.set_trace()
        physics.named.data.geom_xpos["target"][:] = self._target

        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the state."""
        obs = collections.OrderedDict()
        obs["position"] = physics.position()
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward_spec(self):
        return specs.Array(shape=(1,), dtype=np.float32, name="reward")

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        target_size = 0.015
        control_reward = rewards.tolerance(
            physics.control(), margin=1, value_at_margin=0, sigmoid="quadratic"
        ).mean()
        small_control = (control_reward + 4) / 5
        near_target = rewards.tolerance(
            physics.mass_to_target_dist(self._target),
            bounds=(0, target_size),
            margin=target_size,
        )
        reward = near_target * small_control
        return reward
