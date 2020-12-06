import numpy as np 
from scipy import special
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

class NeuralNetwork: 
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learing_rate): 
        '''
        初始化网络
        '''
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learing_rate = learing_rate

        '''
        初始化链路权重矩阵, 在-0.5 与 0.5 之间
        '''
        self.wih = np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        self.who = np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5
        #print(self.wih)
        #print(self.who)

        pass

    def fit(self, input_list, target_list): 
        '''
        训练网络
        '''
        # 第一步，根据输入的训练数据更新节点的链路权重
        inputs = np.array(input_list, ndmin = 2).T #转换成numpy支持的二维矩阵
        targets = np.array(target_list, ndmin = 2).T #转换成numpy支持的二维矩阵

        #计算信号经过输入层后产生的信号量
        hidden_inputs = np.dot(self.wih, inputs)

        #中间层神经元对输入的信号做激活函数后得到输出信号
        sigmoid = lambda x:special.expit(x)
        hidden_outputs = sigmoid(hidden_inputs)

        #输出层接收到来自中间层的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        
        #输出层对信号量做激活函数后得到最终的输出信号
        final_outputs = sigmoid(final_inputs)

        # 第二步，计算误差
        output_errors = targets - final_outputs
        
        # 第三步，反向传播误差
        hidden_errors = np.dot(self.who.T, output_errors)
        # 第四步，根据误差计算链接权重的更新量，然后把更新量加到原来的链路权重上
        self.who += self.learing_rate * np.dot(output_errors * final_outputs * (1 - final_outputs), np.transpose(hidden_outputs))
        self.wih += self.learing_rate * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), np.transpose(inputs))

        pass

    def evaluate(self, inputs): 
        '''
        对新输入的数据进行结果判断, inputs = 输入信号量
        中间层得到的总信号量 = 输入层与中间层链接参数所构成的矩阵 * 输入信号量
        '''
        hidden_inputs = np.dot(self.wih, inputs)

        # 对总信号量调用激活函数从而生成该神经元的输出信号量
        sigmoid = lambda x:special.expit(x)
        hidden_outputs = sigmoid(hidden_inputs)

        # 再将输出信号量输入到输出层, 计算输出层神经元接收到的信号量
        final_inputs = np.dot(self.who, hidden_outputs)

        # 再次调用sigmoid函数计算最外层神经元的输出信号量
        final_outputs = sigmoid(final_inputs)

        return final_outputs

        pass



# 加载图片数据集
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

# 将图片像素点值归一化，即转换成0到1之间的数
train_images = train_images.reshape(60000, 28*28)
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape(10000, 28*28)
test_images = test_images.astype('float32') / 255

#将标签转换成向量格式
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# 初始化一个神经网络
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learing_rate = 0.3

net = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learing_rate)

# 将训练图片与图片对应的标签配对后输入到网络进行训练
for train_image, train_label in zip(train_images, train_labels):
    net.fit(train_image, train_label)

print('train complete')

# 使用测试数据检验网络训练的结果， 计算网络判断的准确率
scopes = []
for test_image, test_label in zip(test_images, test_labels):
    out_put = net.evaluate(test_image)
    evaluate_label = np.argmax(out_put)
    correct_label = np.argmax(test_label) 
    
    if evaluate_label == correct_label: 
        scopes.append(1)
    else: 
        scopes.append(0) 

scopes_array = np.asarray(scopes)
accuracy = scopes_array.sum() / scopes_array.size
print('准确率为', accuracy)

#测试一张图片
res = net.evaluate(test_images[0])
print('第6张图片是', np.argmax(res))

(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
plt.imshow(test_images[5])
plt.show()

