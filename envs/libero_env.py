"""Wrapper Gym-style quanh benchmark LIBERO dùng cho eval/rollout Giai đoạn 5.

TODO: implement dựa trên lerobot.envs / libero-gym khi bắt đầu code thật.
"""


class LiberoEnv:
    def __init__(self, task_suite: str, task_id: int | None = None, gui: bool = False):
        self.task_suite = task_suite
        self.task_id = task_id
        self.gui = gui
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
