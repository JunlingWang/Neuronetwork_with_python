# data generator
# 本模块用来生成训练和测试数据
import numpy as np
import matplotlib.pyplot as plt
import math
import random

NUM_DATA = 500
CLASIFICATION_TYPE ='cross'

def tag_entry(x, y):
    if CLASIFICATION_TYPE == 'circle':
        if x**2+y**2 > 1: # 如果距离原点半径大于某值，则标为1
            tag = 1
        else:# 小于某值则标为0
            tag = 0
    
    elif CLASIFICATION_TYPE == 'line':
        if x > 0: # 如果距离原点半径大于某值，则标为1
            tag = 1
        else:# 小于某值则标为0
            tag = 0
            
    elif CLASIFICATION_TYPE == 'cross':
        if x*y > 0: # 如果距离原点半径大于某值，则标为1
            tag = 1
        else:# 小于某值则标为0
            tag = 0            
    
    return tag

    

def creat_data(n_entries, purpose="ref"):
    half_n_entries = math.ceil(n_entries/2)
    entry_list = []
    num_of_0s = 0
    num_of_1s = 0
    
    i = 0
    while i < half_n_entries * 2:
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        tag = tag_entry(x, y)
        entry = [x, y, tag]
        if tag == 0 and num_of_0s < half_n_entries:
            entry_list.append(entry)
            num_of_0s += 1
            i += 1
        elif tag == 1 and num_of_1s < half_n_entries:
            entry_list.append(entry)
            num_of_1s += 1
            i += 1
            
    return np.array(entry_list)
            
def plot_data(data, title):
    color = [] # 根据标签值定义颜色
    for i in data[:, 2]:
        if i == 0:
            color.append("orange")
        else:
            color.append("blue")
            
    plt.scatter(data[:, 0],data[:, 1],c=color) # scatter 的两个参数都是类似于数组的形式，分别代表x、y轴
    plt.title(title)
    plt.show()    

if __name__ == "__main__":
    data = creat_data(NUM_DATA, "train")
    print(data)
    plot_data(data, 'demo data')

