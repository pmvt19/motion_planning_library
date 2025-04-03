import numpy as np

class NumpyState():
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value.tobytes())

    def __eq__(self, other):
        return np.all(np.isclose(self.value, other.value))
    
class AngularNumpyState(NumpyState):
    def __init__(self, value, angular_dims_start):
        super().__init__(value)
        self.twopi = np.pi * 2
        self.angular_dims_start = angular_dims_start
        self.value[self.angular_dims_start: ] %= self.twopi
    
