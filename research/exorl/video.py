# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import imageio
import numpy as np


class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20, camera_id=0):
        if root_dir is not None:
            self.save_dir = root_dir / "eval_video"
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.camera_id = camera_id

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            if hasattr(env, "physics"):
                frame = env.physics.render(
                    height=self.render_size,
                    width=self.render_size,
                    camera_id=self.camera_id,
                )
            else:
                frame = env.render()
            self.frames.append(frame)

    def log_to_wandb(self):
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log(
            {
                "eval/video": wandb.Video(
                    frames[::skip, :, ::2, ::2], fps=fps, format="gif"
                )
            }
        )

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
