### 第11章 项目：声呐返回值分类

本章我们使用Keras开发一个二分类网络。本章包括：

- 将数据导入Keras
- 为表格数据定义并训练模型
- 在未知数据上测试Keras模型的性能
- 处理数据以提高准确率
- 调整Keras模型的拓扑和配置

我们开始吧。

#### 11.1 声呐物体分类数据

本章使用声呐数据，包括声呐在不同物体的返回。数据有60个变量，代表不同角度的返回值。目标是将石头和金属筒（矿石）分开。

所有的数据都是连续的，从0到1；输出变量中M代表矿石，R代表石头，需要转换为1和0。数据集有208条数据，在本书的data目录下，也可以自行下载，重命名为```sonar.csv```。 

此数据集可以作为性能测试标准：我们知道什么程度的准确率代表模型是优秀的。交叉检验后，一般的网络可以达到84%的准确率，最高可以达到88%。关于数据集详情，请到UCI机器学习网站查看。

#### 11.2 简单的神经网络

先创建一个简单的神经网络试试看。导入所有的库和函数：

```
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
```

初始化随机数种子，这样每次的结果都一样，帮助debug：

```
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
```

用pandas读入数据：前60列是输入变量（X），最后一列是输出变量（Y）。pandas处理带字符的数据比NumPy更容易。

```
# load dataset
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
```

输出变量现在是字符串：需要编码成数字0和1。scikit-learn的```LabelEncoder```可以做到：先将数据用```fit()```方法导入，然后用```transform()```函数编码，加入一列：

```
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
```

现在可以用Keras创建神经网络模型了。我们用scikit-learn进行随机K折验证，测试模型效果。Keras的模型用```KerasClassifier```封装后可以在scikit-learn中调用，取的变量建立的模型；其他变量会传入```fit()```方法中，例如训练次数和批尺寸。我们写一个函数创建这个模型：只有一个全连接层，神经元数量和输入变量数一样，作为最基础的模型。

模型的权重是比较小的高斯随机数，激活函数是整流函数，输出层只有一个神经元，激活函数是S型函数，代表某个类的概率。损失函数还是对数损失函数（```binary_crossentropy```），这个函数适用于二分类问题。优化算法是Adam随机梯度下降，每轮收集模型的准确率。

```
# baseline model
def create_baseline():
# create model
    model = Sequential()
    model.add(Dense(60, input_dim=60, init='normal', activation='relu')) model.add(Dense(1, init='normal', activation='sigmoid'))
# Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) return model
```

用scikit-learn测试一下模型。向```KerasClassifier```传入训练次数（默认值），关闭日志：

```
# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(y=encoded_Y, n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

输出内容是测试的平均数和标准差。

```
Baseline: 81.68% (5.67%)
```

不用很累很麻烦效果也可以很好。

#### 11.3 预处理数据以增加性能

预处理数据是个好习惯。神经网络喜欢输入类型的比例和分布一致，为了达到这点可以使用正则化，让数据的平均值是0，标准差是1，这样可以保留数据的分布情况。

scikit-learn的```StandardScaler```可以做到这点。不应该在整个数据集上直接应用正则化：应该只在测试数据上交叉验证时进行正则化处理，使正则化成为交叉验证的一环，让模型没有新数据的先验知识，防止模型发散。

scikit-learn的```Pipeline```可以直接做到这些。我们先定义一个```StandardScaler```，然后进行验证：

```
# evaluate baseline model with standardized dataset
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, nb_epoch=100,
    batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(y=encoded_Y, n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

结果如下，平均效果有一点进步。

```
Standardized: 84.07% (6.23%)
```

#### 11.4 调整模型的拓扑和神经元

神经网络有很多参数，例如初始化权重、激活函数、优化算法等等。我们一直没有说到调整网络的拓扑结构：扩大或缩小网络。我们试验一下：

##### 11.4.1 缩小网络

有可能数据中有冗余：原始数据是不同角度的信号，有可能其中某些角度有相关性。我们把第一层隐层缩小一些，强行提取特征试试。

我们把之前的模型隐层的60个神经元减半到30个，这样神经网络需要挑选最重要的信息。之前的正则化有效果：我们也一并做一下.

```
# smaller model
def create_smaller():
# create model
model = Sequential()
model.add(Dense(30, input_dim=60, init='normal', activation='relu')) model.add(Dense(1, init='normal', activation='sigmoid'))
  # Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) return model
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, nb_epoch=100,
    batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(y=encoded_Y, n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

结果如下。平均值有少许提升，方差减少很多：这么做果然有效，因为这次的训练时间只需要之前的一半！

```
Smaller: 84.61% (4.65%)
```

##### 11.4.2 扩大网络

扩大网络后，神经网络更有可能提取关键特征，以非线性方式组合。我们对之前的网络简单修改一下：在原来的隐层后加入一层30个神经元的隐层。现在的网络是：

```
60 inputs -> [60 -> 30] -> 1 output
```

我们希望在缩减信息前可以对所有的变量建模，和缩小网络时的想法类似。这次我们加一层，帮助网络挑选信息：

```
# larger model
def create_larger():
# create model
model = Sequential()
model.add(Dense(60, input_dim=60, init='normal', activation='relu')) model.add(Dense(30, init='normal', activation='relu')) model.add(Dense(1, init='normal', activation='sigmoid'))
  # Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) return model
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, nb_epoch=100,
    batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(y=encoded_Y, n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

这次的结果好了很多，几乎达到业界最优。

```
Larger: 86.47% (3.82%)
```

继续微调网络的结果会更好。你能做到如何？

#### 11.5 总结

本章关于使用Keras开发二分类深度学习项目。总结一下：

- 如何导入数据
- 如何创建基准模型
- 如何用scikit-learn通过K折随机交叉检验测试Keras的模型
- 如何预处理数据
- 如何微调网络

##### 11.5.1 下一章

多分类和二分类介绍完了：下一章是回归问题。
