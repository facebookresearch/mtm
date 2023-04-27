# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict, deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs

import research.exorl.custom_dmc_tasks as cdmc


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    physics: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class FlattenJacoObservationWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        self._obs_spec = OrderedDict()
        wrapped_obs_spec = env.observation_spec().copy()
        if "front_close" in wrapped_obs_spec:
            spec = wrapped_obs_spec["front_close"]
            # drop batch dim
            self._obs_spec["pixels"] = specs.BoundedArray(
                shape=spec.shape[1:],
                dtype=spec.dtype,
                minimum=spec.minimum,
                maximum=spec.maximum,
                name="pixels",
            )
            wrapped_obs_spec.pop("front_close")

        for key, spec in wrapped_obs_spec.items():
            assert spec.dtype == np.float64
            assert type(spec) == specs.Array
        dim = np.sum(
            np.fromiter(
                (np.int(np.prod(spec.shape)) for spec in wrapped_obs_spec.values()),
                np.int32,
            )
        )

        self._obs_spec["observations"] = specs.Array(
            shape=(dim,), dtype=np.float32, name="observations"
        )

    def _transform_observation(self, time_step):
        obs = OrderedDict()

        if "front_close" in time_step.observation:
            pixels = time_step.observation["front_close"]
            time_step.observation.pop("front_close")
            pixels = np.squeeze(pixels)
            obs["pixels"] = pixels

        features = []
        for feature in time_step.observation.values():
            features.append(feature.ravel())
        obs["observations"] = np.concatenate(features, axis=0)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += time_step.reward * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key="pixels"):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(
            shape=np.concatenate(
                [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
            ),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ObservationDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._dtype = dtype
        wrapped_obs_spec = env.observation_spec()["observations"]
        self._obs_spec = specs.Array(wrapped_obs_spec.shape, dtype, "observation")

    def _transform_observation(self, time_step):
        obs = time_step.observation["observations"].astype(self._dtype)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        physics = env.physics.state()
        self._physics_spec = specs.Array(
            physics.shape, dtype=physics.dtype, name="physics"
        )

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        def default_on_none(value, default):
            if value is None:
                return default
            return value

        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=default_on_none(time_step.reward, 0.0),
            discount=default_on_none(time_step.discount, 1.0),
            physics=self._env.physics.state(),
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reward_spec(self):
        spec = self._env.reward_spec()
        if hasattr(self._task, "get_reward_spec"):
            task_spec = self._task.get_reward_spec()
            if task_spec is not None:
                spec = task_spec
        if len(spec.shape) == 0:
            spec = spec.replace(shape=tuple((1,)), dtype=np.float32)
        return spec

    def physics_spec(self):
        return self._physics_spec

    def discount_spec(self):
        spec = self._env.discount_spec()
        if hasattr(self._task, "get_discount_spec"):
            task_spec = self._task.get_discount_spec()
            if task_spec is not None:
                spec = task_spec
        if len(spec.shape) == 0:
            spec = spec.replace(shape=tuple((1,)), dtype=np.float32)
        return spec

    def __getattr__(self, name):
        return getattr(self._env, name)


def _make_jaco(obs_type, domain, task, frame_stack, action_repeat, seed):
    env = cdmc.make_jaco(task, obs_type, seed)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FlattenJacoObservationWrapper(env)
    return env


def _make_dmc(obs_type, domain, task, frame_stack, action_repeat, seed):
    visualize_reward = False
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(
            domain,
            task,
            task_kwargs=dict(random=seed),
            environment_kwargs=dict(flat_observation=True),
            visualize_reward=visualize_reward,
        )
    else:
        env = cdmc.make(
            domain,
            task,
            task_kwargs=dict(random=seed),
            environment_kwargs=dict(flat_observation=True),
            visualize_reward=visualize_reward,
        )

    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    if obs_type == "pixels":
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=84, width=84, camera_id=camera_id)
        env = pixels.Wrapper(env, pixels_only=True, render_kwargs=render_kwargs)
    return env


def make(name, obs_type="states", frame_stack=1, action_repeat=1, seed=1):
    assert obs_type in ["states", "pixels"]
    if name.startswith("point_mass_maze"):
        domain = "point_mass_maze"
        _, _, _, task = name.split("_", 3)
    else:
        domain, task = name.split("_", 1)
    domain = dict(cup="ball_in_cup").get(domain, domain)

    make_fn = _make_jaco if domain == "jaco" else _make_dmc
    env = make_fn(obs_type, domain, task, frame_stack, action_repeat, seed)

    if obs_type == "pixels":
        env = FrameStackWrapper(env, frame_stack)
    else:
        env = ObservationDTypeWrapper(env, np.float32)

    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = ExtendedTimeStepWrapper(env)
    return env
