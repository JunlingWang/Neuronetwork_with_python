# test
import creatAndPlotData as cp
import numpy as np
import math
from playsound import playsound
import random
from time import sleep
from progress.bar import Bar

values = np.array([1, 2, 3, 4])
demands = np.array([1, 2])

def get_adjust_matrix(values, demands):
    plain_weights = np.full((len(values), len(demands)), 1)
    print(plain_weights)
    plain_weights_T = plain_weights.T
    print(plain_weights_T)
    weights_adjust_matrix = (plain_weights_T*values).T*demands#batch内所有修改矩阵相加
    print(plain_weights_T*values)
    return weights_adjust_matrix

print(get_adjust_matrix(values, demands))

