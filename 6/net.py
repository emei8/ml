from keras.datasets import imdb
import numpy as np 
from keras import models
from keras import layers
from keras import losses
from keras import optimizers
from keras import metrics

#加载影视数据
(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words=10000) 

print(train_data[0])
print(train_labels[0])

#获取词与频率映射关系，key = words  value = rate 
word_index = imdb.get_word_index() 
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#train_data中 1，2，3 对应的不是单词，1=填充，2=文本起始，3=未知，4开始对应单词
text = ''
for index in train_data[0]: 
    if index > 3: 
        text += reverse_word_index.get(index - 3)
        text += ' '
    else: 
        text += "?"

print(text) 

#将文本向量化
def oneHotVectorText(allText, dimension = 10000): 
    oneHotMatrix = np.zeros((len(allText), dimension))
    for i, wordFrequency in enumerate(allText): 
        oneHotMatrix[i, wordFrequency] = 1.0
    return oneHotMatrix

x_train = oneHotVectorText(train_data) 
x_test = oneHotVectorText(test_data) 
print(x_train[0]) 
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32') 

#开始构建三层网络
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#设置损失函数和学习率
model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
train_result = history.history
print(train_result.keys())

acc = train_result['acc']
val_acc = train_result['val_acc']
loss = train_result['loss']
val_loss = train_result['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='trainning loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title(' train and valid loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
