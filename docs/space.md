# Spaces

## Creating Your Own Space

There are a only a few required functions to implement a new `RobotSpace`. Implementing these functions will allow
use with most of the search algorithms.

```python
class CustomRobotSpace(RobotSpace):
    def __init__(self):
        RobotSpace.__init__(self)

        # Required Instance Variables
        self.angular_dims_start: int = 2
        self.edge_validity_delta: float = 200.0

        self.state_dims: int = 3
        self.c_space_lower_bound: np.ndarray = np.array([-10, -10, 0.0])
        self.c_space_upper_bound: np.ndarray = np.array([10, 10, 2*np.pi])

    def is_valid(self, state):
        # Required To Be Implemented
    
    def make_state(self, state: np.ndarray):
        # Required To Be Implemented
    
    def sample_point(self):
        # Required To Be Implemented
    
    def dist(self, state1, state2):
        # Required To Be Implemented

    def batch_is_valid(self, states: np.ndarray):
        # Optional (Can Speed Up Collision Detection if implemented)

    def draw_environment(self, ax):
        # Optional (Helps visualize robot environment)
    
    def draw_state(self, ax, state):
        # Option (Helps visualize robot state)
```