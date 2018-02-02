# coding=utf-8
from sklearn import preprocessing
import numpy as np

x = np.array([[0,0,0], [1,0,1], [1,0,0], [1,1,0]]).T
l = len(x)
s = x.shape
x_mean = np.mean(x, axis=1)
c = np.cov(x, bias=1)
print(x)