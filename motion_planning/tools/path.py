from motion_planning.tools import NumpyState

class Path:
    def __init__(self, path: list[NumpyState] = []):
        assert isinstance(path, list), "Path must be of type list"
        self.path: list[NumpyState] = path

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        return self.path[idx]


class KinodynamicPath(Path):
    def __init__(self, path: list = [], controls: list = [], dt: float = None):
        super().__init__(path=path)
        self.controls: list = controls
        self.dt: float = dt
