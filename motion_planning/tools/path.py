import numpy as np 

class Path():
    def __init__(self, path=[]):
        assert(isinstance(path, list)), "Path must be of type list"
        self.path : list = path 
    
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx):
        return self.path[idx]
    
class KinodynamicPath(Path):
    def __init__(self, path=[], controls=[], dt=None):
        super().__init__(path=path)
        self.controls = controls
        self.dt = dt