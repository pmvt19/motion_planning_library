from motion_planning.space import RobotSpace


class HolonomicRobot(RobotSpace):
    def __init__(self):
        super().__init__()

    def sample_task(self, task_type="random"):
        if task_type == "random":
            start = self.sample_valid_point()
            target = self.sample_valid_point()
        elif task_type == "structured":
            raise NotImplementedError
        return start, target
