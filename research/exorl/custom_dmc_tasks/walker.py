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
"""Planar Walker Domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources
from dm_control import suite
from dm_env import specs

import numpy as np

_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = 0.025

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 1.2

# Horizontal speeds (meters/second) above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 8
_SPIN_SPEED = 5

SUITE = containers.TaggedTasks()


def make(task, task_kwargs=None, environment_kwargs=None, visualize_reward=False):
    task_kwargs = task_kwargs or {}
    if environment_kwargs is not None:
        task_kwargs = task_kwargs.copy()
        task_kwargs["environment_kwargs"] = environment_kwargs
    env = SUITE[task](**task_kwargs)
    env.task.visualize_reward = visualize_reward
    return env


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    root_dir = os.path.dirname(os.path.dirname(__file__))
    xml = resources.GetResource(
        os.path.join(root_dir, "custom_dmc_tasks", "walker.xml")
    )
    return xml, common.ASSETS


@SUITE.add("benchmarking")
def flip(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PlanarWalker(move_speed=_RUN_SPEED, flip=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs
    )


@SUITE.add("benchmarking")
def multitask(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = MultiTaskPlanarWalker(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs
    )


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Walker domain."""

    def torso_upright(self):
        """Returns projection from z-axes of torso to the z-axes of world."""
        return self.named.data.xmat["torso", "zz"]

    def torso_height(self):
        """Returns the height of the torso."""
        return self.named.data.xpos["torso", "z"]

    def horizontal_velocity(self):
        """Returns the horizontal velocity of the center-of-mass."""
        return self.named.data.sensordata["torso_subtreelinvel"][0]

    def orientations(self):
        """Returns planar orientations of all bodies."""
        return self.named.data.xmat[1:, ["xx", "xz"]].ravel()

    def angmomentum(self):
        """Returns the angular momentum of torso of the Cheetah about Y axis."""
        return self.named.data.subtree_angmom["torso"][1]


class PlanarWalker(base.Task):
    """A planar walker task."""

    def __init__(self, move_speed, flip=False, random=None):
        """Initializes an instance of `PlanarWalker`.

        Args:
          move_speed: A float. If this value is zero, reward is given simply for
            standing up. Otherwise this specifies a target horizontal velocity for
            the walking task.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._move_speed = move_speed
        self._flip = flip
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        In 'standing' mode, use initial orientation and small velocities.
        In 'random' mode, randomize joint angles and let fall to the floor.

        Args:
          physics: An instance of `Physics`.

        """
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of body orientations, height and velocites."""
        obs = collections.OrderedDict()
        obs["orientations"] = physics.orientations()
        obs["height"] = physics.torso_height()
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        standing = rewards.tolerance(
            physics.torso_height(),
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 2,
        )
        upright = (1 + physics.torso_upright()) / 2
        stand_reward = (3 * standing + upright) / 4

        if self._flip:
            move_reward = rewards.tolerance(
                physics.angmomentum(),
                bounds=(_SPIN_SPEED, float("inf")),
                margin=_SPIN_SPEED,
                value_at_margin=0,
                sigmoid="linear",
            )
        else:
            move_reward = rewards.tolerance(
                physics.horizontal_velocity(),
                bounds=(self._move_speed, float("inf")),
                margin=self._move_speed / 2,
                value_at_margin=0.5,
                sigmoid="linear",
            )

        return stand_reward * (5 * move_reward + 1) / 6


class MultiTaskPlanarWalker(base.Task):
    """A planar walker task."""

    def __init__(self, random=None):
        """Initializes an instance of `PlanarWalker`.

        Args:
          move_speed: A float. If this value is zero, reward is given simply for
            standing up. Otherwise this specifies a target horizontal velocity for
            the walking task.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        In 'standing' mode, use initial orientation and small velocities.
        In 'random' mode, randomize joint angles and let fall to the floor.

        Args:
          physics: An instance of `Physics`.

        """
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of body orientations, height and velocites."""
        obs = collections.OrderedDict()
        obs["orientations"] = physics.orientations()
        obs["height"] = physics.torso_height()
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward_spec(self):
        return specs.Array(shape=(4,), dtype=np.float32, name="reward")

    def get_reward(self, physics):
        """Returns a reward to the agent."""

        # compute stand reward
        standing = rewards.tolerance(
            physics.torso_height(),
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 2,
        )
        upright = (1 + physics.torso_upright()) / 2

        stand_reward = (3 * standing + upright) / 4

        # compute walk reward
        walking = rewards.tolerance(
            physics.horizontal_velocity(),
            bounds=(_WALK_SPEED, float("inf")),
            margin=_WALK_SPEED / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        walk_reward = stand_reward * (5 * walking + 1) / 6

        # compute run reward
        running = rewards.tolerance(
            physics.horizontal_velocity(),
            bounds=(_RUN_SPEED, float("inf")),
            margin=_RUN_SPEED / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        run_reward = stand_reward * (5 * running + 1) / 6

        # compute flip reward
        flipping = rewards.tolerance(
            physics.angmomentum(),
            bounds=(_SPIN_SPEED, float("inf")),
            margin=_SPIN_SPEED,
            value_at_margin=0,
            sigmoid="linear",
        )

        flip_reward = stand_reward * (5 * flipping + 1) / 6

        reward = np.array([stand_reward, walk_reward, run_reward, flip_reward])
        return reward.astype(np.float32)
