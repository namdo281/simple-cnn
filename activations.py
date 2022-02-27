import numpy as np
class Sigmoid:
    def __init__(self, input_size):
        #print("input size: ", input_size)
        pass
    def sigmoid(self, x):
        #print(x.shape)
        return 1/(1+np.e**(-x)) 
    def forward(self, x):
        try:
            #print(x)
            return self.sigmoid(x)
        except:
            print(x)
    def backward(self, dy):
        # print(dy)
        return np.multiply(self.sigmoid(dy), -self.sigmoid(dy) +1)