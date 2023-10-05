#Python手搓神经网络
#这是B站教程的相关代码请结合下面地址中的教程使用：
#https://www.bilibili.com/video/BV1py4y1F7vp/?spm_id_from=333.999.section.playall&vd_source=036afdc841b709a63dcc3989925f751a
import numpy as np
import createDataAndPlot as cp
import copy
import math

NETWORK_SHAPE = [2, 100, 200, 100, 50, 2]
BATCH_SIZE = 30
LEARNING_RATE = 0.015
LOSS_THRESHOLD = 0.1
FORCE_TRAIN_THRESHOLD = 0.05

force_train = False
random_train = False
n_improved = 0
n_not_improved = 0
current_loss = 1


#标准化函数
def normalize(array):
    max_number = np.max(np.absolute(array), axis=1, keepdims=True)
    scale_rate = np.where(max_number == 0, 1, 1/max_number)
    norm = array * scale_rate
    return norm

#向量标准化函数
def vector_normalize(array):
    max_number = np.max(np.absolute(array))
    scale_rate = np.where(max_number == 0, 1, 1/max_number)
    norm = array * scale_rate
    return norm

# 激活函数
def activation_ReLU(inputs):
    return np.maximum(0, inputs)

#分类函数
def classify(probabilities):
    classification = np.rint(probabilities[:, 1])
    return classification
    
# softmax激活函数
def activation_softmax(inputs):
    max_values = np.max(inputs, axis=1, keepdims=True)
    slided_inputs = inputs - max_values
    exp_values = np.exp(slided_inputs)
    norm_base = np.sum(exp_values, axis=1, keepdims=True)
    norm_values = exp_values/norm_base
    return norm_values

#损失函数1
def precise_loss_function(predicted, real):
    real_matrix = np.zeros((len(real), 2))
    real_matrix[:, 1] = real
    real_matrix[:, 0] = 1 - real
    product = np.sum(predicted*real_matrix, axis=1)
    return 1 - product

#损失函数2
def loss_function(predicted, real):
    condition = (predicted > 0.5)
    binary_predicted = np.where(condition, 1, 0)
    real_matrix = np.zeros((len(real), 2))
    real_matrix[:, 1] = real
    real_matrix[:, 0] = 1 - real
    product = np.sum(predicted*real_matrix, axis=1)
    return 1 - product

#需求函数
def get_final_layer_preAct_damands(predicted_values, target_vector):
    target = np.zeros((len(target_vector), 2))
    target[:, 1] = target_vector
    target[:, 0] = 1 - target_vector
    
    for i in range(len(target_vector)):
        if np.dot(target[i], predicted_values[i]) > 0.5:
            target[i] = np.array([0, 0])
        else:
            target[i] = (target[i] - 0.5) * 2
    return target

#-------------------------------------------------------------
#定义一个层类
class Layer: 
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.random.randn(n_neurons)
    
    def layer_forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
    
    def layer_backward(self, preWeights_values, afterWeights_demands):
        preWeights_damands = np.dot(afterWeights_demands, self.weights.T)
        
        condition = (preWeights_values > 0)
        value_derivatives = np.where(condition, 1, 0)

        preActs_demands = value_derivatives*preWeights_damands
        norm_preActs_demands = normalize(preActs_demands)
        
        weight_adjust_matrix = self.get_weight_adjust_matrix(preWeights_values, afterWeights_demands)
        norm_weight_adjust_matrix = normalize(weight_adjust_matrix)
        
        return (norm_preActs_demands, norm_weight_adjust_matrix)
    
    def get_weight_adjust_matrix(self, preWeights_values, aftWeights_demands):
        plain_weights = np.full(self.weights.shape, 1)
        weights_adjust_matrix = np.full(self.weights.shape, 0.0)
        plain_weights_T = plain_weights.T
        
        for i in range(BATCH_SIZE):
            weights_adjust_matrix += (plain_weights_T*preWeights_values[i, :]).T*aftWeights_demands[i, :]
        weights_adjust_matrix = weights_adjust_matrix/BATCH_SIZE
        return weights_adjust_matrix
        
#-----------------------------------------------------------------   
#定义一个网络类
class Network:
    def __init__(self, network_shape):
        self.shape = network_shape
        self.layers = []
        for i in range(len(network_shape)-1):
            layer = Layer(network_shape[i], network_shape[i+1])
            self.layers.append(layer)
     
    #前馈运算函数 
    def network_forward(self, inputs):
        outputs = [inputs]
        for i in range(len(self.layers)):
            layer_sum = self.layers[i].layer_forward(outputs[i])
            if i < len(self.layers)-1:
                layer_output = activation_ReLU(layer_sum)
                layer_output = normalize(layer_output)
            else:
                layer_output = activation_softmax(layer_sum)
            outputs.append(layer_output)
        return outputs
    
    #反向传播函数
    def network_backward(self, layer_outputs, target_vector):
        backup_network = copy.deepcopy(self) # 备用网络
        preAct_demands = get_final_layer_preAct_damands(layer_outputs[-1], target_vector)
        for i in range(len(self.layers)):
            layer = backup_network.layers[len(self.layers) - (1+i)] # 倒序
            if i != 0:
                layer.biases += LEARNING_RATE * np.mean(preAct_demands, axis=0)
                layer.biases = vector_normalize(layer.biases)
                
            outputs = layer_outputs[len(layer_outputs) - (2+i)]
            results_list = layer.layer_backward(outputs, preAct_demands)
            preAct_demands = results_list[0]
            weights_adjust_matrix = results_list[1]
            layer.weights += LEARNING_RATE * weights_adjust_matrix
            layer.weights = normalize(layer.weights)
        return backup_network
    
    #单批次训练
    def one_batch_train(self, batch):
        global force_train, random_train, n_improved, n_not_improved

        inputs = batch[:,(0, 1)]
        targets = copy.deepcopy(batch[:, 2]).astype(int) # 标准答案
        outputs = self.network_forward(inputs)
        precise_loss = precise_loss_function(outputs[-1], targets)
        loss = loss_function(outputs[-1], targets)
            
        if np.mean(loss) <= LOSS_THRESHOLD:#损失函数小于这个值就不需要训练了
            print('No need for training')
        else:
            backup_network = self.network_backward(outputs, targets)
            backup_outputs = backup_network.network_forward(inputs)
            backup_precise_loss = precise_loss_function(backup_outputs[-1], targets)
            backup_loss = loss_function(backup_outputs[-1], targets)
            
            if np.mean(precise_loss) >= np.mean(backup_precise_loss) or np.mean(loss) >= np.mean(backup_loss):
                for i in range(len(self.layers)):
                    self.layers[i].weights = backup_network.layers[i].weights.copy()
                    self.layers[i].biases = backup_network.layers[i].biases.copy()
                print('Improved')
                n_improved += 1
            
            else:
                if force_train:
                    for i in range(len(self.layers)):
                        self.layers[i].weights = backup_network.layers[i].weights.copy()
                        self.layers[i].biases = backup_network.layers[i].biases.copy()
                    print('Force train')
                if random_train:
                    self.random_update()
                    print("Random update")
                else:  
                    print('No improvement')
                n_not_improved += 1
        print('-----------------------------------------')
            
    #多批次训练
    def train(self, n_entries):
        global force_train, random_train, n_improved, n_not_improved
        n_improved = 0
        n_not_improved = 0

        n_batches = math.ceil(n_entries/BATCH_SIZE)
        for i in range(n_batches):
            batch = cp.creat_data(BATCH_SIZE)
            self.one_batch_train(batch)
        improvement_rate =  n_improved/(n_improved + n_not_improved)
        print("Improvement rate")
        print(format(improvement_rate, ".0%"))
        if improvement_rate <= FORCE_TRAIN_THRESHOLD:
            force_train = True
        else:
            force_train = False
        if n_improved == 0:
            random_train = True
        else:
            random_train = False
        
        data = cp.creat_data(800)
        inputs = data[:, (0, 1)]
        outputs = self.network_forward(inputs)
        classification = classify(outputs[-1])
        data[:, 2] = classification
        cp.plot_data(data, "After training")
        
    #随机更新
    def random_update(self):
        random_network = Network(NETWORK_SHAPE)
        for i in range(len(self.layers)):
            weights_change = random_network.layers[i].weights
            biases_change = random_network.layers[i].biases
            self.layers[i].weights += weights_change
            self.layers[i].biases += biases_change
            
    #核验LOSS值
    def check_loss(self):
        data = cp.creat_data(1000)
        inputs = data[:, (0, 1)]
        targets = copy.deepcopy(data[:, 2]).astype(int) # 标准答案
        outputs = self.network_forward(inputs)
        loss = loss_function(outputs[-1], targets)
        return np.mean(loss)
        
#-------------MAIN-------------------------
def main():
    global current_loss
    data = cp.creat_data(800) #生成数据
    cp.plot_data(data, "Right classification")

    #选择起始网络
    use_this_network = 'n' #No
    while use_this_network != 'Y' and use_this_network != 'y':
        network = Network(NETWORK_SHAPE)
        inputs = data[:, (0, 1)]
        outputs = network.network_forward(inputs)
        classification = classify(outputs[-1])
        data[:, 2] = classification
        cp.plot_data(data, "Choose network")
        use_this_network = input("Use this network? Y to yes, N to No \n")
    
    #进行训练
    do_train = input("Train? Y to yes, N to No \n")
    while do_train == 'Y' or do_train == 'y' or do_train.isnumeric() == True:
        if do_train.isnumeric() == True:
            n_entries = int(do_train)
        else:
            n_entries = int(input("Enter the number of data entries used to train. \n"))
            
        network.train(n_entries)
        do_train = input("Train? Y to yes, N to No \n")
        
    #演示训练效果
    inputs = data[:, (0, 1)]
    outputs = network.network_forward(inputs)
    classification = classify(outputs[-1])
    data[:, 2] = classification
    cp.plot_data(data, "After training")
    print("谢谢，再见！")
#----------------TEST-------------------------
def test():
    pass

#--------------运行---------------------
main()
