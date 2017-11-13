### 第10章 项目：多类花朵分类

本章我们使用Keras为多类分类开发并验证一个神经网络。本章包括：

- 将CSV导入Keras
- 为Keras预处理数据
- 使用scikit-learn验证Keras模型

我们开始吧。

#### 10.1 鸢尾花分类数据集

本章我们使用经典的鸢尾花数据集。这个数据集已经被充分研究过，4个输入变量都是数字，量纲都是厘米。每个数据代表花朵的不同参数，输出是分类结果。数据的属性是（厘米）：

1. 萼片长度
2. 萼片宽度
3. 花瓣长度
4. 花瓣宽度
5. 类别

这个问题是多类分类的：有两种以上的类别需要预测，确切的说，3种。这种问题需要对神经网络做出特殊调整。数据有150条：前5行是：

```python
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
```

鸢尾花数据集已经被充分研究，模型的准确率可以达到95%到97%，作为目标很不错。本书的data目录下附带了示例代码和数据，也可以从UCI机器学习网站下载，重命名为```iris.csv```。数据集的详情请在UCI机器学习网站查询。

#### 10.2 导入库和函数

我们导入所需要的库和函数，包括深度学习包Keras、数据处理包pandas和模型测试包scikit-learn。

```python
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
```

#### 10.3 指定随机数种子

我们指定一个随机数种子，这样重复运行的结果会一致，以便复现随机梯度下降的结果：

```python
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
```

#### 10.4 导入数据

数据可以直接导入。因为数据包含字符，用pandas更容易。然后可以将数据的属性（列）分成输入变量（X）和输出变量（Y）：

```python
# load dataset
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]
```

#### 10.5 输出变量编码

数据的类型是字符串：在使用神经网络时应该将类别编码成矩阵，每行每列代表所属类别。可以使用独热编码，或者加入一列。这个数据中有3个类别：```Iris-setosa```、```Iris-versicolor```和```Iris-virginica```。如果数据是

```python
Iris-setosa
Iris-versicolor
Iris-virginica
```

用独热编码可以编码成这种矩阵：

```python
Iris-setosa, Iris-versicolor, Iris-virginica 1, 0, 0
0, 1, 0
0, 0, 1
```

scikit-learn的```LabelEncoder```可以将类别变成数字，然后用Keras的```to_categorical()```函数编码：

```python
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
```

#### 10.6 设计神经网络

Keras提供了```KerasClassifier```，可以将网络封装，在scikit-learn上用。```KerasClassifier```的初始化变量是模型名称，返回供训练的神经网络模型。

我们写一个函数，为鸢尾花分类问题创建一个神经网络：这个全连接网络只有1个带有4个神经元的隐层，和输入的变量数相同。为了效果，隐层使用整流函数作为激活函数。因为我们用了独热编码，网络的输出必须是3个变量，每个变量代表一种花，最大的变量代表预测种类。网络的结构是：

```python
4个神经元 输入层 -> [4个神经元 隐层] -> 3个神经元 输出层
```

输出层的函数是S型函数，把可能性映射到概率的0到1。优化算法选择ADAM随机梯度下降，损失函数是对数函数，在Keras中叫```categorical_crossentropy```：

```python
# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=4, init='normal', activation='relu')) model.add(Dense(3, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) return model
```

可以用这个模型创建```KerasClassifier```，也可以传入其他参数，这些参数会传递到```fit()```函数中。我们将训练次数```nb_epoch```设成150，批尺寸```batch_size```设成5，```verbose```设成0以关闭调试信息：

```python
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
```

#### 10.7 用K折交叉检验测试模型

现在可以测试模型效果了。scikit-learn有很多种办法可以测试模型，其中最重要的就是K折检验。我们先设定模型的测试方法：K设为10（默认值很好），在分割前随机重排数据：

```python
kfold = KFold(n=len(X), n_folds=10, shuffle=True, random_state=seed)
```

这样我们就可以在数据集（```X```和```dummy_y```）上用10折交叉检验（```kfold```）测试性能了。模型需要10秒钟就可以跑完，每次检验输出结果：

```python
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

输出结果的均值和标准差，这样可以验证模型的预测能力，效果拔群：

```python
Baseline: 95.33% (4.27%)
```

#### 10.8 总结

本章关于使用Keras开发深度学习项目。总结一下：

- 如何导入数据
- 如何使用独热编码处理多类分类数据
- 如何与scikit-learn一同使用Keras
- 如何用Keras定义多类分类神经网络
- 如何用scikit-learn通过K折交叉检验测试Keras的模型


#### 10.8.1 下一章

本章完整描述了Keras项目的开发：下一章我们开发一个二分类网络，并调优。

