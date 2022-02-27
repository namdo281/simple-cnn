import numpy as np
class FullyConnected:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.ones((input_size, output_size))/(input_size*output_size)
        self.b = np.ones((1, output_size))/output_size

    def forward(self, x):
        try:
            self.x = x 
            # print(x.shape)
            # print(self.input_size)
            assert x.shape[0] == self.input_size
            # print("fc x: ", x)
            return x.dot(self.W) + self.b
        except:
            print("fc x: ", x)
            
    def backward(self, dy):
        # print(dy.shape)
        # print(self.x.T
        assert dy.shape[0] == self.W.shape[1]
        dW = self.x.reshape(16,1) @ dy.reshape(1, 10)
        dx = dy @ self.W.T
        db = dy
        self.W -= 1e-3*dW
        self.b -= 1e-3*db
        return dx