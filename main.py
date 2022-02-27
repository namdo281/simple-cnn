import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from model import Model
data = datasets.load_digits()
examples = data['data']
targets = data['target']
#print(examples.shape)
#print(targets.shape)

examples = examples.reshape(-1, 1, 8, 8)

model = Model()
model.train(examples[:100], targets[:100])


