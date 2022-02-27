import numpy as np
from convolution import Convolution
from fully_connected import FullyConnected
from pooling import MaxPooling
from activations import Sigmoid
from loss_functions import EntropyLoss

class Model:
    def __init__(self):
        self.conv = Convolution(input_size=(1, 8,8), n_filter=1)
        self.pool = MaxPooling(input_size=(1,8,8))
        self.fc = FullyConnected(input_size=4*4, output_size=10)
        self.sigmoid = Sigmoid((1, 10))
        self.loss = EntropyLoss()
    def forward(self, x):
        #print("ini x: ", x)
        x = self.conv.forward(x)
        #print("conv x: ", x)
        x = self.pool.forward(x)
        #print("pool x: ", x)
        x = x.reshape(-1)
        #print("reshape x: ", x)
        x = self.fc.forward(x)[0]
        #print("fc x: ", x)
        x = self.sigmoid.forward(x)
        #print("s x: ", x)
        x = x/(sum(x))
        #print(x)
        return x
    def backward(self, y):
        #print("ini y: ", y)
        y = self.sigmoid.backward(y)
        #print("sigmoid y: ", y)
        y = self.fc.backward(y)
        #print("fc y: ", y)
        y = y.reshape(1, 4, 4)
        #print("reshape y :", y)
        y = self.pool.backward(y)
        #print("pool y: ", y)
        y = self.conv.backward(y)
        #print("conv y: ", y)
    def train(self, examples, targets):
        for i in range(20):
            overall_loss = 0
            for j, x in enumerate(examples):
                x_pred = self.forward(x)
                target_onehot = np.zeros(10)
                #print(targets[j])
                target_onehot[targets[j]] = 1
                #print(target_onehot)
                loss = self.loss.forward(x_pred, target_onehot)
                #print("loss: ", loss)
                self.backward(self.loss.backward(loss))
                overall_loss += loss
            print("overall loss: ", overall_loss/examples.shape[0])
        pass

