import numpy as np
class SquaredError:
    def __init__(self):
        pass
    def calc(self, pred, gt):
        return np.linalg.norm(pred-gt)**2
        