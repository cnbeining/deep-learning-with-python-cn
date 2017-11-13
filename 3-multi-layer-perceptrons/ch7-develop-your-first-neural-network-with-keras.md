### 第7章 使用Keras开发神经网络

Keras基于Python，开发深度学习模型很容易。Keras将Theano和TensorFlow的数值计算封装好，几句话就可以配置并训练神经网络。本章开始使用Keras开发神经网络。本章将：

- 将CSV数据读入Keras
- 用Keras配置并编译多层感知器模型
- 用验证数据集验证Keras模型

我们开始吧。

#### 7.1 简介

虽然代码量不大，但是我们还是慢慢来。大体分几步：

1. 导入数据
2. 定义模型
3. 编译模型
4. 训练模型
5. 测试模型
6. 写出程序

#### 7.2 皮马人糖尿病数据集

我们使用皮马人糖尿病数据集（Pima Indians onset of diabetes），在UCI的机器学习网站可以免费下载。数据集的内容是皮马人的医疗记录，以及过去5年内是否有糖尿病。所有的数据都是数字，问题是（是否有糖尿病是1或0），是二分类问题。数据的数量级不同，有8个属性：

1. 怀孕次数
2. 2小时口服葡萄糖耐量试验中的血浆葡萄糖浓度
3. 舒张压（毫米汞柱）
4. 2小时血清胰岛素（mu U/ml)
5. 体重指数（BMI）
6. 糖尿病血系功能
7. 年龄（年）
8. 类别：过去5年内是否有糖尿病

所有的数据都是数字，可以直接导入Keras。本书后面也会用到这个数据集。数据有768行，前5行的样本长这样：

```
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
```

数据在本书代码的```data``` 目录，也可以在UCI机器学习的网站下载。把数据和Python文件放在一起，改名：

```
pima-indians-diabetes.csv
```

基准准确率是65.1%，在10次交叉验证中最高的正确率是77.7%。在UCI机器学习的网站可以得到数据集的更多资料。

#### 7.3 导入资料

使用随机梯度下降时最好固定随机数种子，这样你的代码每次运行的结果都一致。这种做法在演示结果、比较算法或debug时特别有效。你可以随便选种子：

```python
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
```

现在导入皮马人数据集。NumPy的```loadtxt()```函数可以直接带入数据，输入变量是8个，输出1个。导入数据后，我们把数据分成输入和输出两组以便交叉检验：

```python
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
```

这样我们的数据每次结果都一致，可以定义模型了。

#### 7.4 定义模型

Keras的模型由层构成：我们建立一个```Sequential```模型，一层层加入神经元。第一步是确定输入层的数目正确：在创建模型时用```input_dim```参数确定。例如，有8个输入变量，就设成8。

隐层怎么设置？这个问题很难回答，需要慢慢试验。一般来说，如果网络够大，即使存在问题也不会有影响。这个例子里我们用3层全连接网络。

全连接层用```Dense```类定义：第一个参数是本层神经元个数，然后是初始化方式和激活函数。这里的初始化方法是0到0.05的连续型均匀分布（```uniform```），Keras的默认方法也是这个。也可以用高斯分布进行初始化（```normal```）。

前两层的激活函数是线性整流函数（```relu```），最后一层的激活函数是S型函数（```sigmoid```）。之前大家喜欢用S型和正切函数，但现在线性整流函数效果更好。为了保证输出是0到1的概率数字，最后一层的激活函数是S型函数，这样映射到0.5的阈值函数也容易。前两个隐层分别有12和8个神经元，最后一层是1个神经元（是否有糖尿病）。

```python
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu')) model.add(Dense(8, init='uniform', activation='relu')) model.add(Dense(1, init='uniform', activation='sigmoid'))
```

网络的结构如图：

![7.1 神经网络结构](https://i.imgur.com/6TQODzB.png)

#### 7.5 编译模型

定义好的模型可以编译：Keras会调用Theano或者TensorFlow编译模型。后端会自动选择表示网络的最佳方法，配合你的硬件。这步需要定义几个新的参数。训练神经网络的意义是：找到最好的一组权重，解决问题。

我们需要定义损失函数和优化算法，以及需要收集的数据。我们使用```binary_crossentropy```，错误的对数作为损失函数；```adam```作为优化算法，因为这东西好用。想深入了解请查阅：Adam: A Method for Stochastic Optimization论文。因为这个问题是分类问题，我们收集每轮的准确率。

#### 7.6 训练模型

终于开始训练了！调用模型的```fit()```方法即可开始训练。

网络按轮训练，通过```nb_epoch```参数控制。每次送入的数据（批尺寸）可以用```batch_size```参数控制。这里我们只跑150轮，每次10个数据。多试试就知道了。

```python
# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10)
```

现在CPU或GPU开始煎鸡蛋了。

#### 7.7 测试模型

我们把测试数据拿出来检验一下模型的效果。注意这样不能测试在新数据的预测能力。应该将数据分成训练和测试集。

调用模型的```evaluation()```方法，传入训练时的数据。输出是平均值，包括平均误差和其他的数据，例如准确度。

```python
# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```

#### 7.8 写出程序

用Keras做机器学习就是这么简单。我们把代码放在一起：

```python
# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
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
model.add(Dense(12, input_dim=8, init='uniform', activation='relu')) model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```

训练时每轮会输出一次损失和正确率，以及最终的效果。在我的CPU上用Theano大约跑10秒：

```python
...
Epoch 143/150
768/768 [==============================] - 0s - loss: 0.4614 - acc: 0.7878
Epoch 144/150
768/768 [==============================] - 0s - loss: 0.4508 - acc: 0.7969
Epoch 145/150
768/768 [==============================] - 0s - loss: 0.4580 - acc: 0.7747
Epoch 146/150
768/768 [==============================] - 0s - loss: 0.4627 - acc: 0.7812
Epoch 147/150
768/768 [==============================] - 0s - loss: 0.4531 - acc: 0.7943
Epoch 148/150
768/768 [==============================] - 0s - loss: 0.4656 - acc: 0.7734
Epoch 149/150
768/768 [==============================] - 0s - loss: 0.4566 - acc: 0.7839
Epoch 150/150
768/768 [==============================] - 0s - loss: 0.4593 - acc: 0.7839
768/768 [==============================] - 0s
acc: 79.56%
```
#### 7.9 总结

本章关于利用Keras创建神经网络。总结一下：

- 如何导入数据
- 如何用Keras定义神经网络
- 如何调用后端编译模型
- 如何训练模型
- 如何测试模型

##### 7.9.1 下一章

现在你已经知道如何如何用Keras开发神经网络：下一章讲讲如何在新的数据上进行测试。

