# Temp file for containing the requred fields for batched robot representations
# Would require separate objects for 2d and 3d representations

import numpy as np
from dataclasses import dataclass

@dataclass
class BatchRobotState:
    # TODO: Templating required fields
    {
        "rectangles": np.empty((0, 4)),
        "segments": segments,
        "points": np.empty((0, 2)),
        "points_radius": 0.0,
        "segments_radii": radii,
    }