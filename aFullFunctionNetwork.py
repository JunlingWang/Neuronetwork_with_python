# a network built with class
# 成功做环形分类之前做了3个改动，
# 1. 普遍标准化，作用不明显
# 2. 数据中两个分类比例相同，防止造成偏差，效果仍然不明显
# 3. 修改了损失函数，使其值简单地落在0到1之间
# 4. 修改反馈机制，凡是分类正确的条目都是0反馈（不反馈修正值），
#    这样可以只针对分类错误的数据条目做优化，防止卡死在同一个分类
#    （这应该是起关键作用的一次修正）
# 5. 修改学习率，防止跳过最优点
# 6. 损失函数进一步修改，设置了“粗”、“细”两个损失函数，只要其中一个减小，就可以优化
# 7. 解决梯度消失问题，在网络“僵住”的时候引入随机变化

import numpy as np
import creatAndPlotData as cp
import math
from playsound import playsound
import copy


BATCH_SIZE = 25 
LEARNING_RATE = 0.015
NETWORK_SHAPE = (2, 10, 11, 12, 2) # 各层神经元数
n_improved = 0 #在一个batch的训练中，有多少条数据产生了优化（能让损失函数减小，因此修改了参数）
n_not_improved = 0 #在一个batch的训练中，有多少条数据没有产生优化（不能让损失函数减小，因此没有修改参数）
force_train = False #如果n_improved太少，即使不能让损失函数减小也强制修改参数，以让网络走出梯度消失状态
random_train = False #如果n_improved为0（此时参数修改值往往也是0,强制修改无效）,让网络随机修改，以走出梯度消失状态

# 把np.array里的数据标准化
def normalize(array):
    max_number = np.max(np.absolute(array), axis=1, keepdims=True)
    scale_rate = np.where(max_number == 0, 1, 1/max_number)
    norm = array * scale_rate
    return norm
# def normalize(array):
#     max_ab = np.max(np.absolute(array)) # 绝对值中的最大值
#     if max_ab == 0:
#         scale_rate = 1
#     else:
#         scale_rate = 1/max_ab # max_ab * scale_rate == 1
#     norm = array * scale_rate
#     return norm

# 本类定义了一个层的特征，包括权重矩阵和偏置向量
class Layer_Specs:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) # 权重矩阵的行数等于输入矩阵的列数
        self.biases = np.zeros((1, n_neurons)) # 偏置矩阵的列数与权重矩阵（以及输出矩阵）的列数相等
        # 注意此处np.zeros需要双层括号，而np.random.randn只需要单层的
        
        # __init__ 函数用来规定创建一个对象时需要输入哪些参数（这里是n_inputs, n_neurons）
        # 以及这个对象具有哪些属性（attributes）（这里是weights，biases）

    #前向传播并生成本层的输出值，返回输出值向量
    def layer_forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
    
    #反向传播并生成本层参数的修改值（但不修改参数），返回修改值（偏置和权重）
    def layer_backward(self, preWeights_values, aftWeights_demands): # aft_weights_demands 的行数为数据条数、列数为神经元数、值为-1到1之间
        preWeights_demands = np.dot(aftWeights_demands, self.weights.T)
            
        condition = (preWeights_values > 0) # 正数映射到1,负数映射到0
        value_derivatives = np.where(condition, 1, 0) # 根据条件用0、1代替浮点数（ReLU是分段函数，负区间导数为0,正区间导数为1）
        preActs_demands = value_derivatives * preWeights_demands
        norm_preActs_demands = normalize(preActs_demands)
        
        weights_adjust_matrix = self.get_weights_adjust_matrix(preWeights_values, aftWeights_demands)
        norm_weights_adjust_matrix = normalize(weights_adjust_matrix)
        
        return (norm_preActs_demands, norm_weights_adjust_matrix) # 输入和输出都是标准化的demands，此值同时用作修正biases
    
    #对于每一个单一的权重来说，它的修改值 = preWeights_values × aftWeights_demands
    #这可以理解为导数的链式规则preWeights_values是权重对于aftWeights_value的导数，
    #而aftWeights_demands是aftWeights_value相对于目标值（让损失函数减小）的导数
    #本函数通过一个“全1矩阵”前后相乘的方式，把每个权重的修改值一次算完
    def get_weights_adjust_matrix(self, preWeights_values, aftWeights_demands):
        plain_weights = np.full(self.weights.shape, 1)
        weights_adjust_matrix = np.full(self.weights.shape, 0.0)
        plain_weights_T = plain_weights.T

        for i in range(BATCH_SIZE):
            weights_adjust_matrix += (plain_weights_T*preWeights_values[i, :]).T*aftWeights_demands[i, :]#batch内所有修改矩阵相加
        weights_adjust_matrix = weights_adjust_matrix/BATCH_SIZE # 求平均值
        return weights_adjust_matrix #返回batch内修改矩阵的平均值
#-------------------------------------------------------------------------------------------------------------
class Network:
    def __init__(self, network_shape):
        self.layers = []
        for i in range(len(network_shape)-1):
            layer_specs = Layer_Specs(network_shape[i], network_shape[i+1])
            self.layers.append(layer_specs)
        self.shape = network_shape

    def network_forward(self, data):
        inputs = data[:, (0,1)] #取前两列作为输入
        outputs = [inputs] # inputs 就是数据集的outputs
        for i in range(len(self.layers)): #最后一层激活函数不一样，所以要-1
            if i < len(self.layers)-1:
                layer_output = activation_ReLU(self.layers[i].layer_forward(outputs[i]))
                layer_output = normalize(layer_output)
            else:
                layer_output = activation_softmax(self.layers[i].layer_forward(outputs[i]))
            outputs.append(layer_output)
            
        probabilities = outputs[-1]
        # n个概率数组成的一维array，和为1
        return outputs #  outputs[-1] 就是probabilities
    
    def network_backward(self, layer_outputs, target_vector):
        backup_network = copy.deepcopy(self)
        preAct_demands = get_final_layer_preAct_demands(layer_outputs[-1], target_vector)
        for i in range(len(self.layers)):
            layer = backup_network.layers[len(self.layers)-(1+i)] # 倒序
            if i != 0: # 输出层的biases不修正
                layer.biases += LEARNING_RATE * np.mean(preAct_demands, axis=0) # 调整backup_network的biases
                layer.biases = normalize(layer.biases)
            outputs = layer_outputs[len(layer_outputs)-(2+i)]
            demands_list = layer.layer_backward(outputs, preAct_demands)
            preAct_demands =demands_list[0]
            weight_demands =demands_list[1]
            layer.weights += LEARNING_RATE * weight_demands # 调整backup_network的biases
            layer.weights = normalize(layer.weights)

        return backup_network
            
    def train(self, n_entries):
        global n_improved, n_not_improved, force_train, random_train
        n_improved = 0
        n_not_improved = 0

        n_batches = math.ceil(n_entries / BATCH_SIZE)
        for i in range(n_batches):
            batch = cp.creat_data(BATCH_SIZE, 'train') #生成数据, 'train'意为打标的训练数据
            self.one_batch_train(batch)
        Improvement_rate = n_improved/(n_improved+n_not_improved)
        print('Improvement rate:')
        print(format(Improvement_rate, ".0%"))
        print(str(n_improved)+'/'+str(n_improved+n_not_improved))
        if Improvement_rate < 0.05:
            force_train = True
        else:
            force_train = False
            
        if n_improved == 0:
            random_train = True
        else:
            random_train = False
            
        data = cp.creat_data(100, 'ref') #生成数据
        cp.plot_data(data, "Original training set classification")
        inputs = data[:, (0,1)] #取前两列作为输入
        outputs = self.network_forward(data)
        classification = classify(outputs[-1])
        data[:, 2] = classification#     print(data)
        cp.plot_data(data, "After-training classification")

    def one_batch_train(self, batch):
        global n_improved, n_not_improved
        
        inputs = batch[:, (0,1)] #取前两列作为输入
        targets = batch[:, 2].astype(int)  # 取数据的第3列用于训练的目标值
        outputs = self.network_forward(inputs)
        
        #-------for test-------------
        print('pre_training outputs')
        outputs_values = outputs[-1][:, 1]
        condition = (outputs_values > 0.5) # 正数映射到1,负数映射到0
        # Use np.where to replace values based on the condition
        outputs_vector = np.where(condition, 1, 0)
        print(outputs_vector)
        print('targets')
        print(targets)
        #-------test ends------------
        
        loss = loss_function(outputs[-1], targets)
        precise_loss = precise_loss_function(outputs[-1], targets)
        
        if np.mean(loss) <= 0.1:
            print('no need for modification')
        else:
            backup_network = self.network_backward(outputs, targets)
            backup_outputs = backup_network.network_forward(inputs)
            
            #-------for test-------------
            print('post_training_outputs')
            backup_outputs_values = backup_outputs[-1][:, 1]
            condition = (backup_outputs_values > 0.5) # 正数映射到1,负数映射到0
            # Use np.where to replace values based on the condition
            backup_outputs_vector = np.where(condition, 1, 0)
            print(backup_outputs_vector)
            #-------test ends------------
            
            backup_loss = loss_function(backup_outputs[-1], targets)
            precise_backup_loss = precise_loss_function(backup_outputs[-1], targets)
            if np.mean(loss) > np.mean(backup_loss) or np.mean(precise_loss) > np.mean(precise_backup_loss):
                for i in range(len(self.layers)):
                    self.layers[i].weights = backup_network.layers[i].weights.copy()
                    self.layers[i].biases = backup_network.layers[i].biases.copy()
#                 self= copy.deepcopy(backup_network)
                print('Improved')
                print(np.mean(backup_loss))
                n_improved += 1
            else:
                if force_train:
                    for i in range(len(self.layers)):
                        self.layers[i].weights = backup_network.layers[i].weights.copy()
                        self.layers[i].biases = backup_network.layers[i].biases.copy()
                    print('Force modified')
                if random_train:
                    self.random_update()
                    print('Randomly modified')
                    
                print('No improvement')
                print(np.mean(backup_loss))
                n_not_improved += 1
        print('--------------------------')
    
    def random_update(self):
        random_network = Network(NETWORK_SHAPE)
        for i in range(len(self.layers)):
            weights_change = random_network.layers[i].weights
            biases_change = random_network.layers[i].biases
            self.layers[i].weights += weights_change
            self.layers[i].biases += biases_change
##class Network ends
##------------------------------------------------------------------------------------------------------------------------

def activation_ReLU(inputs): # ReLU, rectified linear unit,输入负值输出0,输入正值输出原值
    return np.maximum(0, inputs)

def activation_softmax(inputs): # 把每条输入分别做指数计算之后再标准化,为便于理解将计算过程拆分
    max_values = np.max(inputs, axis=1, keepdims=True) # 选出每一条输入的最大值并组成向量
    slided_inputs = inputs - max_values # 把所有输入值平移到负值或零，避免指数爆炸并保持倍数不变
    exp_values = np.exp(slided_inputs) # 做指数计算
    norm_base = np.sum(exp_values, axis=1, keepdims=True) # 把每一行分别取和
    return exp_values/norm_base # 标准化 normalization

def loss_function(predicted, real):
    condition = (predicted > 0.5) 
    binary_predicted = np.where(condition, 1, 0)
    real_matrix = np.zeros((len(real), 2))
    real_matrix[:, 1] = real
    real_matrix[:, 0] = - (real - 1)
    dot_product = np.sum(binary_predicted*real_matrix, axis=1) 
    return 1-dot_product
    
def precise_loss_function(predicted, real):
    real_matrix = np.zeros((len(real), 2))
    real_matrix[:, 1] = real
    real_matrix[:, 0] = - (real - 1)
    dot_product = np.sum(predicted*real_matrix, axis=1) 
    return 1-dot_product

def classify(probabilities):
    classification = np.rint(probabilities[:, 1])
    # 把概率值转化成分类值（取第1列四舍五入到0、1即可）
    # 之所以取第1列，是因为0意思是第0列为真，1意为第1列为真
    return classification

def get_final_layer_preAct_demands(predicted_values, target_vector): # ?
    target = np.zeros((len(target_vector), 2))
    target[:, 1] = target_vector
    target[:, 0] = - (target_vector - 1)
    
    for i in range(len(target_vector)):
        if np.dot(target[i],predicted_values[i]) > 0.5:
            target[i] = np.array([0, 0])    
        else:
            target[i] = (target[i] - 0.5) * 2
        
    return target

    
#----------------------------执行-----------------------------------
def main():
    data = cp.creat_data(100, 'ref') #生成数据

    # 选择一个合适的起始网络
    use_this_network = 'n'
    while use_this_network != 'Y' and use_this_network != 'y':
        network = Network(NETWORK_SHAPE)
        outputs = network.network_forward(data)
        classification = classify(outputs[-1])
        data[:, 2] = classification
        cp.plot_data(data, "Initial network classification")
        print('Use this network? Y to yes, N to no')
        use_this_network = input()
        
    print('Train? Y to yes, N to no')
    do_train = input()
    while do_train == 'Y' or do_train == 'y' or do_train.isnumeric() == True:
        if do_train.isnumeric() == True:
            n_entries = int(do_train)
        else: 
            print('Enter the number of data entries used to train.')
            n_entries = int(input())
        
        network.train(n_entries)
        print('Train? Y to yes, N to no')
        do_train = input()
        
    print('inference')
    for i in range(1):
        data = cp.creat_data(100, 'ref') #生成数据
        inputs = data[:, (0,1)] #取前两列作为输入
        outputs = network.network_forward(data)
        classification = classify(outputs[-1])
        data[:, 2] = classification
        cp.plot_data(data, 'After-training classification')
#---------------------------------------------------------------------------
main()