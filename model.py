from re import I
import numpy
from .convolution import Convolution
from .fully_connected import FullyConnected
from .pooling import MaxPooling
from .activations import Sigmoid
from .loss_functions import SquaredError
class Model:
    def __init__(self):
        self.conv = Convolution(input_size=(1, 64,64), n_filter=10)
        self.pool = MaxPooling(input_size=(10,64,64))
        self.fc = FullyConnected(input_size=(1, 32*32), output_size=(1, 10))
        self.sigmoid = Sigmoid((1, 10))
        self.loss = SquaredError()
    def forward(self, x):
        pass
    def backward():
        pass
    def train():
        pass

