import numpy as np
import pandas as pd
from sklearn import datasets, train_test_split
data = datasets.load_digits().data
print(data.shape)

data = data.reshape(1797, 1, -1)




