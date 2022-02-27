import numpy as np    
class EntropyLoss:
    def __init__(self):
        pass
    def forward(self, x, gt):
        self.x = x
        self.gt = gt
        return -np.log(x).dot(gt)   
    def backward(self, loss):
        #print(1/self.x)
        #print(-loss*(1/self.x).dot(self.gt))
        return -loss*np.diag(1/self.x).dot(self.gt)
