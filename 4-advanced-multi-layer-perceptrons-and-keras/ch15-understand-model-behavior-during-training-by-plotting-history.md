### 第15章 模型训练效果可视化

查看训练效果的历史数据大有裨益。本章关于将模型的训练效果进行可视化。本章教你：

- 如何观察历史训练数据
- 如何在训练时绘制数据准确性图像
- 如何在训练时绘制损失图像

我们开始吧。

#### 15.1 取历史数据

上一章说到Keras支持回调API，其中默认调用```History```函数，每轮训练收集损失和准确率，如果有测试集，也会收集测试集的数据。

历史数据会收集```fit()```函数的返回值，在```history```对象中。看一下到底收集了什么数据：

```python
# list all data in history
print(history.history.keys())
```

如果是第7章的二分类问题：

```python
['acc', 'loss', 'val_acc', 'val_loss']
```

可以用这些数据画折线图，直观看到：

- 模型收敛的速度（斜率）
- 模型是否已经收敛（稳定性）
- 模型是否过拟合（验证数据集）

以及更多。

#### 15.2 可视化Keras模型训练

收集一下第7章皮马人糖尿病模型的历史数据，绘制：

1. 训练和验证集的准确度
2. 训练和验证集的损失

```python
# Visualize training history
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10, verbose=0) # list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left') plt.show()
# summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left') plt.show()
```

图像如下。最后几轮的准确率还在上升，有可能有点过度学习；但是两个数据集的效果差不多，应该没有过拟合。

![15.1 准确度图像](https://i.imgur.com/ngFUnWu.png)

从损失图像看，两个数据集的性能差不多。如果两条线开始分开，有可能应该提前终止训练。

![15.2 损失图像](https://i.imgur.com/r5XHhJy.png)

#### 15.3 总结

本章关于在训练时绘制图像。总结一下：

- 如何看历史对象
- 如何绘制历史性能
- 如何绘制两个数据集的不同性能

##### 15.3.1 下一章

Dropout可以有效防止过拟合：下一章关于这个技术、如何在Keras中实现以及最佳实践。


