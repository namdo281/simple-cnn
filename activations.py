import numpy as np

class Sigmoid:
    def __init__(self, input_size):
        self.W = np.array(input_size).T
        self.b = 0
    def forward(self, x):
        self.x = x
        return 1/1+ np.e^(-self.W.dot(x) + self.b)    

    def backward(self, dy):
        dW = dy*self.forward(self.x)*(1-self.forward(self.x))*self.x.T
        db = dy*self.forward(self.x)*(1-self.forward(self.x))
        dx = dy*self.forward(self.x)*(1-self.forward(self.x))*self.W.T
        self.W -= dW
        self.db -= db
        return dx