import numpy as np


class NumpyState():
    def __init__(self, value: np.ndarray):
        self.value: np.ndarray = value

    def __hash__(self):
        return hash(self.value.tobytes())

    def __eq__(self, other: "NumpyState"):
        return np.all(np.isclose(self.value, other.value))
    
class AngularNumpyState(NumpyState):
    def __init__(self, value: np.ndarray, angular_dims_start: int):
        super().__init__(value)
        self.twopi: float = np.pi * 2
        self.angular_dims_start: int = angular_dims_start
        self.value[self.angular_dims_start: ] %= self.twopi
    
