import numpy as np
class FullyConnected:
    def __init__(self, input_size, output_size):
        self.W = np.zeros((input_size, output_size))
        self.b = np.zeros((1, output_size))

    def forward(self, x):
        self.x = x 
        assert x.shape == self.input_size
        return x @ self.W + self.b
    def backward(self, df):
        assert df.shape[1] == self.W.shape[0]
        dW = self.x.T @ df
        dx = df @ self.W.T
        db = df
        self.W -= dW
        self.b -= db
        return dx