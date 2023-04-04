# Copyright 2019 The dm_control Authors.
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
"""A task where the goal is to move the hand close to a target prop or site."""

import collections

from dm_control import composer
from dm_control.composer import initializers
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions
from dm_control.entities import props
from dm_control.manipulation.shared import arenas
from dm_control.manipulation.shared import cameras
from dm_control.manipulation.shared import constants
from dm_control.manipulation.shared import observations
from dm_control.manipulation.shared import registry
from dm_control.manipulation.shared import robots
from dm_control.manipulation.shared import tags
from dm_control.manipulation.shared import workspaces
from dm_control.utils import rewards
from dm_env import specs
import numpy as np

_ReachWorkspace = collections.namedtuple(
    "_ReachWorkspace", ["target_bbox", "tcp_bbox", "arm_offset"]
)

# Ensures that the props are not touching the table before settling.
_PROP_Z_OFFSET = 0.001

_DUPLO_WORKSPACE = _ReachWorkspace(
    target_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, _PROP_Z_OFFSET), upper=(0.1, 0.1, _PROP_Z_OFFSET)
    ),
    tcp_bbox=workspaces.BoundingBox(lower=(-0.1, -0.1, 0.2), upper=(0.1, 0.1, 0.4)),
    arm_offset=robots.ARM_OFFSET,
)

_SITE_WORKSPACE = _ReachWorkspace(
    target_bbox=workspaces.BoundingBox(lower=(-0.2, -0.2, 0.02), upper=(0.2, 0.2, 0.4)),
    tcp_bbox=workspaces.BoundingBox(lower=(-0.2, -0.2, 0.02), upper=(0.2, 0.2, 0.4)),
    arm_offset=robots.ARM_OFFSET,
)

_TARGET_RADIUS = 0.05
_TIME_LIMIT = 10.0

TASKS = [
    ("reach_top_left", np.array([-0.09, 0.09, _PROP_Z_OFFSET])),
    ("reach_top_right", np.array([0.09, 0.09, _PROP_Z_OFFSET])),
    ("reach_bottom_left", np.array([-0.09, -0.09, _PROP_Z_OFFSET])),
    ("reach_bottom_right", np.array([0.09, -0.09, _PROP_Z_OFFSET])),
]


def make(task_id, obs_type, seed):
    obs_settings = (
        observations.VISION if obs_type == "pixels" else observations.PERFECT_FEATURES
    )
    task = _reach(task_id, obs_settings=obs_settings, use_site=True)
    return composer.Environment(task, time_limit=_TIME_LIMIT, random_state=seed)


class MultiTaskReach(composer.Task):
    """Bring the hand close to a target prop or site."""

    def __init__(
        self, task_id, arena, arm, hand, prop, obs_settings, workspace, control_timestep
    ):
        """Initializes a new `Reach` task.

        Args:
          arena: `composer.Entity` instance.
          arm: `robot_base.RobotArm` instance.
          hand: `robot_base.RobotHand` instance.
          prop: `composer.Entity` instance specifying the prop to reach to, or None
            in which case the target is a fixed site whose position is specified by
            the workspace.
          obs_settings: `observations.ObservationSettings` instance.
          workspace: `_ReachWorkspace` specifying the placement of the prop and TCP.
          control_timestep: Float specifying the control timestep in seconds.
        """
        self._arena = arena
        self._arm = arm
        self._hand = hand
        self._arm.attach(self._hand)
        self._arena.attach_offset(self._arm, offset=workspace.arm_offset)
        self.control_timestep = control_timestep
        self._tcp_initializer = initializers.ToolCenterPointInitializer(
            self._hand,
            self._arm,
            position=distributions.Uniform(*workspace.tcp_bbox),
            quaternion=workspaces.DOWN_QUATERNION,
        )

        # Add custom camera observable.
        self._task_observables = cameras.add_camera_observables(
            arena, obs_settings, cameras.FRONT_CLOSE
        )

        if task_id == "reach_multitask":
            self._targets = [target for (_, target) in TASKS]
        else:
            self._targets = [target for (task, target) in TASKS if task == task_id]

        # target_pos_distribution = distributions.Uniform(*TASKS[task_id])
        self._prop = prop
        if prop:
            # The prop itself is used to visualize the target location.
            self._make_target_site(parent_entity=prop, visible=False)
            self._target = self._arena.add_free_entity(prop)
            self._prop_placer = initializers.PropPlacer(
                props=[prop],
                position=target_pos_distribution,
                quaternion=workspaces.uniform_z_rotation,
                settle_physics=True,
            )
        else:
            if len(self._targets) == 1:
                self._target = self._make_target_site(parent_entity=arena, visible=True)

            # obs = observable.MJCFFeature('pos', self._target)
            # obs.configure(**obs_settings.prop_pose._asdict())
            # self._task_observables['target_position'] = obs

        # Add sites for visualizing the prop and target bounding boxes.
        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=workspace.tcp_bbox.lower,
            upper=workspace.tcp_bbox.upper,
            rgba=constants.GREEN,
            name="tcp_spawn_area",
        )
        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=workspace.target_bbox.lower,
            upper=workspace.target_bbox.upper,
            rgba=constants.BLUE,
            name="target_spawn_area",
        )

    def _make_target_site(self, parent_entity, visible):
        return workspaces.add_target_site(
            body=parent_entity.mjcf_model.worldbody,
            radius=_TARGET_RADIUS,
            visible=visible,
            rgba=constants.RED,
            name="target_site",
        )

    @property
    def root_entity(self):
        return self._arena

    @property
    def arm(self):
        return self._arm

    @property
    def hand(self):
        return self._hand

    def get_reward_spec(self):
        n = len(self._targets)
        return specs.Array(shape=(n,), dtype=np.float32, name="reward")

    @property
    def task_observables(self):
        return self._task_observables

    def get_reward(self, physics):
        hand_pos = physics.bind(self._hand.tool_center_point).xpos
        rews = []
        for target_pos in self._targets:
            distance = np.linalg.norm(hand_pos - target_pos)
            reward = rewards.tolerance(
                distance, bounds=(0, _TARGET_RADIUS), margin=_TARGET_RADIUS
            )
            rews.append(reward)
        rews = np.array(rews).astype(np.float32)
        if len(self._targets) == 1:
            return rews[0]
        return rews

    def initialize_episode(self, physics, random_state):
        self._hand.set_grasp(physics, close_factors=random_state.uniform())
        self._tcp_initializer(physics, random_state)
        if self._prop:
            self._prop_placer(physics, random_state)
        else:
            if len(self._targets) == 1:
                physics.bind(self._target).pos = self._targets[0]


def _reach(task_id, obs_settings, use_site):
    """Configure and instantiate a `Reach` task.

    Args:
      obs_settings: An `observations.ObservationSettings` instance.
      use_site: Boolean, if True then the target will be a fixed site, otherwise
        it will be a moveable Duplo brick.

    Returns:
      An instance of `reach.Reach`.
    """
    arena = arenas.Standard()
    arm = robots.make_arm(obs_settings=obs_settings)
    hand = robots.make_hand(obs_settings=obs_settings)
    if use_site:
        workspace = _SITE_WORKSPACE
        prop = None
    else:
        workspace = _DUPLO_WORKSPACE
        prop = props.Duplo(
            observable_options=observations.make_options(
                obs_settings, observations.FREEPROP_OBSERVABLES
            )
        )
    task = MultiTaskReach(
        task_id,
        arena=arena,
        arm=arm,
        hand=hand,
        prop=prop,
        obs_settings=obs_settings,
        workspace=workspace,
        control_timestep=constants.CONTROL_TIMESTEP,
    )
    return task
