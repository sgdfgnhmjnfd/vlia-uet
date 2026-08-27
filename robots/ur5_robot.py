"""Driver UR5 cho triển khai thật (Giai đoạn 5 — sau khi so sánh Success Rate trong sim).

TODO: implement giao tiếp UR5 (RTDE/ROS driver) khi bắt đầu deploy phần cứng.
Robot mục tiêu của VLIA là UR5 (khác với repo tham khảo dùng UR3e).
"""


class UR5Robot:
    def __init__(self, robot_ip: str | None = None):
        self.robot_ip = robot_ip
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def send_action(self, action):
        raise NotImplementedError
