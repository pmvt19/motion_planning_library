from motion_planning.space import RobotSpace
from motion_planning.tools import NumpyState

class HolonomicRobot(RobotSpace):
    def __init__(self):
        super().__init__()

    def sample_task(self, task_type: str = "random") -> tuple[NumpyState, NumpyState]:
        if task_type == "random":
            start = self.sample_valid_point()
            target = self.sample_valid_point()
        elif task_type == "structured":
            raise NotImplementedError
        return start, target
