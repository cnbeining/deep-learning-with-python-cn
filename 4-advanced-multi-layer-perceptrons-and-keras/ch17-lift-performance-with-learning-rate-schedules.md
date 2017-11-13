### 第17章 学习速度设计

神经网络的训练是很困难的优化问题。传统的随机梯度下降算法配合设计好的学习速度有时效果更好。本章包括：

- 调整学习速度的的原因
- 如何使用按时间变化的学习速度
- 如何使用按训练次数变化的学习速度

我们开始吧。

#### 17.1 学习速度

随机梯度下降算法配合设计好的速度可以增强效果，减少训练时间：也叫学习速度退火或可变学习速度。其实就是慢慢调整学习速度，而传统的方法中学习速度不变。

最简单的调整方法是学习速度随时间下降，一开始做大的调整加速训练，后面慢慢微调性能。两个简单的方法：

- 根据训练轮数慢慢下降
- 到某个点下降到某个值

我们分别探讨一下。

#### 17.2 电离层分类数据集

本章使用电离层二分类数据集，研究电离层中的自由电子。分类g（好）意味着电离层中有某个结构；b（坏）代表没有，信号通过了电离层。数据有34个属性，351个数据。

10折检验下最好的模型可以达到94~98%的准确度。数据在本书的data目录下，也可以自行下载，重命名为```ionosphere.csv```。数据集详情请参见UCI机器学习网站。

#### 17.3 基于时间的学习速度调度

Keras内置了一个基于时间的学习速度调度器：Keras的随机梯度下降```SGD```类有```decay```参数，按下面的公式调整速度：

```
LearnRate = LearnRate x (1 / 1 + decay x epoch)
```

默认值是0：不起作用。

```python
LearningRate = 0.1 * 1/(1 + 0.0 * 1)
LearningRate = 0.1
```

如果衰减率大于1，例如0.001，效果是：

```python
Epoch Learning Rate 
1 0.1
2 0.0999000999 
3 0.0997006985 
4 0.09940249103 
5 0.09900646517
```

到100轮的图像：

![17.2 按时间](https://i.imgur.com/aGgYVXf.png)

可以这样设计：

```python
Decay = LearningRate / Epochs
Decay = 0.1 / 100
Decay = 0.001
```

下面的代码按时间减少学习速度。神经网络有1个隐层，34个神经元，激活函数是整流函数。输出层是1个神经元，激活函数是S型函数，输出一个概率。学习率设到0.1，训练50轮，衰减率0.002，也就是0.1/50。学习速度调整一般配合动量使用：动量设成0.8。代码如下：

```python
import pandas
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("ionosphere.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:34].astype(float)
Y = dataset[:,34]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
# create model
model = Sequential()
model.add(Dense(34, input_dim=34, init='normal', activation='relu')) model.add(Dense(1, init='normal', activation='sigmoid'))
# Compile model
epochs = 50
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False) model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Fit the model
model.fit(X, Y, validation_split=0.33, nb_epoch=epochs, batch_size=28)
```

训练67%，测试33%的数据，准确度达到了99.14%，高于不使用任何优化的95.69%：

```
235/235 [==============================] - 0s - loss: 0.0607 - acc: 0.9830 - val_loss:
    0.0732 - val_acc: 0.9914
Epoch 46/50
235/235 [==============================] - 0s - loss: 0.0570 - acc: 0.9830 - val_loss:
    0.0867 - val_acc: 0.9914
Epoch 47/50
235/235 [==============================] - 0s - loss: 0.0584 - acc: 0.9830 - val_loss:
    0.0808 - val_acc: 0.9914
Epoch 48/50
235/235 [==============================] - 0s - loss: 0.0610 - acc: 0.9872 - val_loss:
    0.0653 - val_acc: 0.9828
Epoch 49/50
235/235 [==============================] - 0s - loss: 0.0591 - acc: 0.9830 - val_loss:
    0.0821 - val_acc: 0.9914
Epoch 50/50
235/235 [==============================] - 0s - loss: 0.0598 - acc: 0.9872 - val_loss:
    0.0739 - val_acc: 0.9914
```

#### 17.3 基于轮数的学习速度调度

也可以固定调度：到某个轮数就用某个速度，每次的速度是上次的一半。例如，初始速度0.1，每10轮降低一半。画图就是：

![17.3 轮数](https://i.imgur.com/xKKILzV.png)

Keras的```LearningRateScheduler```作为回调参数可以控制学习速度，取当前的轮数，返回应有的速度。还是刚才的网络，加入一个```step_decay```函数，生成如下学习率：

```Python
LearnRate = InitialLearningRate x Droprate ^ floor((1+Epoch)/EpochDrop)
```

InitialLearningRate是初始的速度，DropRate是减速频率，EpochDrop是降低多少：

```python
import pandas
import pandas
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import LearningRateScheduler
# learning rate schedule
def step_decay(epoch):
  initial_lrate = 0.1
  drop = 0.5
  epochs_drop = 10.0
  lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
  return lrate
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("../data/ionosphere.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:34].astype(float)
Y = dataset[:,34]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
# create model
model = Sequential()
model.add(Dense(34, input_dim=34, init='normal', activation='relu'))
model.add(Dense(1, init='normal', activation='sigmoid'))
# Compile model
sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False) model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# learning schedule callback
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
# Fit the model
model.fit(X, Y, validation_split=0.33, nb_epoch=50, batch_size=28, callbacks=callbacks_list)
```

效果也是99.14%，比什么都不做好：

```python
Epoch 45/50
235/235 [==============================] - 0s - loss: 0.0546 - acc: 0.9830 - val_loss:
    0.0705 - val_acc: 0.9914
Epoch 46/50
235/235 [==============================] - 0s - loss: 0.0542 - acc: 0.9830 - val_loss:
    0.0676 - val_acc: 0.9914
Epoch 47/50
235/235 [==============================] - 0s - loss: 0.0538 - acc: 0.9830 - val_loss:
    0.0668 - val_acc: 0.9914
Epoch 48/50
235/235 [==============================] - 0s - loss: 0.0539 - acc: 0.9830 - val_loss:
    0.0708 - val_acc: 0.9914
Epoch 49/50
235/235 [==============================] - 0s - loss: 0.0539 - acc: 0.9830 - val_loss:
    0.0674 - val_acc: 0.9914
Epoch 50/50
235/235 [==============================] - 0s - loss: 0.0531 - acc: 0.9830 - val_loss:
    0.0694 - val_acc: 0.9914
```

#### 17.5 调整学习速度的技巧

这些技巧可以帮助调参：

1. 增加初始学习速度。因为速度后面会降低，一开始速度快点可以加速收敛。
2. 动量要大。这样后期学习速度下降时如果方向一样，还可以继续收敛。
3. 多试验。这个问题没有定论，需要多尝试。也试试指数下降和什么都不做。

#### 17.6 总结

本章关于调整学习速度。总结一下：

- 调整学习速度为什么有效
- 如何在Keras使用基于时间的学习速度下降
- 如何自己编写下降速度函数

##### 17.6.1 下一章

第四章到此结束，包括Keras的一些高级函数和调参的高级方法。下一章研究卷积神经网络（CNN），在图片和自然语言处理上尤为有效。


