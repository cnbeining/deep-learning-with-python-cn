### 第12章 项目：波士顿住房价格回归

本章关于如何使用Keras和社交网络解决回归问题。本章将：

- 导入CSV数据
- 创建回归问题的神经网络模型
- 使用scikit-learn对Keras的模型进行交叉验证
- 预处理数据以增加效果
- 微调网络参数

我们开始吧。

#### 12.1 波士顿住房价格数据

本章我们研究波士顿住房价格数据集，即波士顿地区的住房信息。我们关心的是住房价格，单位是千美金：所以，这个问题是回归问题。数据有13个输入变量，代表房屋不同的属性：

1. CRIM：人均犯罪率
2. ZN：25,000平方英尺以上民用土地的比例
3. INDUS：城镇非零售业商用土地比例
4. CHAS：是否邻近查尔斯河，1是邻近，0是不邻近
5. NOX：一氧化氮浓度（千万分之一）
6. RM：住宅的平均房间数
7. AGE：自住且建于1940年前的房屋比例
8. DIS：到5个波士顿就业中心的加权距离
9. RAD：到高速公路的便捷度指数
10. TAX：每万元的房产税率
11. PTRATIO：城镇学生教师比例
12. B： 1000(Bk − 0.63)2 其中Bk是城镇中黑人比例
13. LSTAT：低收入人群比例
14. MEDV：自住房中位数价格，单位是千元

这个问题已经被深入研究过，所有的数据都是数字。数据的前5行是：

```
0.00632 18.00 2.310 0 0.5380 6.5750 65.20 4.0900 1 296.0 15.30 396.90 4.98 24.00
0.02731 0.00 7.070 0 0.4690 6.4210 78.90 4.9671 2 242.0 17.80 396.90 9.14 21.60 
0.02729 0.00 7.070 0 0.4690 7.1850 61.10 4.9671 2 242.0 17.80 392.83 4.03 34.70 
0.03237 0.00 2.180 0 0.4580 6.9980 45.80 6.0622 3 222.0 18.70 394.63 2.94 33.40 
0.06905 0.00 2.180 0 0.4580 7.1470 54.20 6.0622 3 222.0 18.70 396.90 5.33 36.20
```

数据在本书的data目录下，也可以自行下载，重命名为```housing.csv```。普通模型的均方误差（MSE）大约是20，和方差（SSE）是$4,500美金。关于数据集详情，请到UCI机器学习网站查看。


#### 12.2 简单的神经网络

先创建一个简单的回归神经网络。导入所有的库和函数：

```
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
```

源文件是CSV格式，分隔符是空格：可以用pandas导入，然后分成输入（X）和输出（Y）变量。

```
# load dataset
dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]
```

Keras可以把模型封装好，交给scikit-learn使用，方便测试模型。我们写一个函数，创建神经网络。

代码如下。有一个全连接层，神经元数量和输入变量数一致（13），激活函数还是整流函数。输出层没有激活函数，因为在回归问题中我们希望直接取结果。

优化函数是Adam，损失函数是MSE，和我们要优化的函数一致：这样可以对模型的预测有直观的理解，因为MSE乘方就是千美元计的误差。

```
# define base mode
def baseline_model():
# create model
model = Sequential()
model.add(Dense(13, input_dim=13, init='normal', activation='relu')) model.add(Dense(1, init='normal'))
  # Compile model
model.compile(loss='mean_squared_error', optimizer='adam') return model
```

使用```KerasRegressor```封装这个模型，任何其他的变量都会传入```fit()```函数中，例如训练次数和批次大小，这里我们取默认值。老规矩，为了可以复现结果，指定一下随机数种子：

```
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
```

可以测试一下基准模型的结果了：用10折交叉检验看看。

```
kfold = KFold(n=len(X), n_folds=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
```

结果是10次检验的误差均值和标准差。

```
Results: 38.04 (28.15) MSE
```

#### 12.3 预处理数据以增加性能

这个数据集的特点是变量的尺度不一致，所以标准化很有用。

scikit-learn的```Pipeline```可以直接进行均一化处理并交叉检验，这样模型不会预先知道新的数据。代码如下：

```
# evaluate model with standardized dataset
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, nb_epoch=50,
    batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n=len(X), n_folds=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
```

效果直接好了一万刀：

```
Standardized: 28.24 (26.25) MSE
```

也可以将数据标准化，在最后一层用S型函数作为激活函数，将比例拉到一样。

#### 12.4 调整模型的拓扑

神经网络有很多可调的参数：最可玩的是网络的结构。这次我们用一个更深的和一个更宽的模型试试。


##### 12.4.1 更深的模型

增加神经网络的层数可以提高效果，这样模型可以提取并组合更多的特征。我们试着加几层隐层：加几句话就行。代码从上面复制下来，在第一层后加一层隐层，神经元数量是上层的一半：

```
def larger_model():
# create model
model = Sequential()
model.add(Dense(13, input_dim=13, init='normal', activation='relu')) model.add(Dense(6, init='normal', activation='relu')) model.add(Dense(1, init='normal'))
  # Compile model
model.compile(loss='mean_squared_error', optimizer='adam') return model
```

这样的结构是：

```
13 inputs -> [13 -> 6] -> 1 output
```

测试的方法一样，数据正则化一下：

```
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, nb_epoch=50, batch_size=5,
    verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n=len(X), n_folds=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))
```

效果好了一点，MSE从28变成24：

```
Larger: 24.60 (25.65) MSE
```

##### 12.4.1 更宽的模型

加宽模型可以增加网络容量。我们减去一层，把隐层的神经元数量加大，从13加到20：

```
def wider_model():
# create model
model = Sequential()
model.add(Dense(20, input_dim=13, init='normal', activation='relu')) model.add(Dense(1, init='normal'))
  # Compile model
model.compile(loss='mean_squared_error', optimizer='adam') return model
```

网络的结构是：

```
13 inputs -> [20] -> 1 output
```

跑一下试试：

```
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, nb_epoch=100, batch_size=5,
    verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n=len(X), n_folds=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))
```

MSE下降到21，效果不错了。

```
Wider: 21.64 (23.75) MSE
```

很难想到，加宽模型比加深模型效果更好：这就是欧皇的力量。


#### 12.5 总结

本章关于使用Keras开发回归深度学习项目。总结一下：

- 如何导入数据
- 如何预处理数据提高性能
- 如何调整网络结构提高性能

##### 12.5.1 下一章

第三部分到此结束：你可以处理一般的机器学习问题了。下一章我们用一些奇技淫巧，使用一些Keras的高级API。


