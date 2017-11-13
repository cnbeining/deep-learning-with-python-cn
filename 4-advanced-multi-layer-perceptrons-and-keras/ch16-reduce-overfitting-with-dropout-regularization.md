### 第16章 使用Dropout正则化防止过拟合

Dropout虽然简单，但可以有效防止过拟合。本章关于如何在Keras中使用Dropout。本章包括：

- dropout的原理
- dropout的使用
- 在隐层上使用dropout

我们开始吧。

#### 16.1 Dropout正则化

译者鄙校的Srivastava等大牛在2014年的论文《[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)》提出了Dropout正则化。Dropout的意思是：每次训练时随机忽略一部分神经元，这些神经元dropped-out了。换句话讲，这些神经元在正向传播时对下游的启动影响被忽略，反向传播时也不会更新权重。

神经网络的所谓“学习”是指，让各个神经元的权重符合需要的特性。不同的神经元组合后可以分辨数据的某个特征。每个神经元的邻居会依赖邻居的行为组成的特征，如果过度依赖，就会造成过拟合。如果每次随机拿走一部分神经元，那么剩下的神经元就需要补上消失神经元的功能，整个网络变成很多独立网络（对同一问题的不同解决方法）的合集。

Dropout的效果是，网络对某个神经元的权重变化更不敏感，增加泛化能力，减少过拟合。

#### 16.2 在Keras中使用Dropout正则化

Dropout就是每次训练按概率拿走一部分神经元，只在训练时使用。后面我们会研究其他的用法。

以下的例子是声呐数据集（第11章），用scikit-learn进行10折交叉检验，这样可以看出区别。输入变量有60个，输出1个，数据经过正则化。基线模型有2个隐层，第一个有60个神经元，第二个有30个。训练方法是随机梯度下降，学习率和动量较低。下面是基线模型的代码：

```python
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# baseline
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=60, init='normal', activation='relu')) model.add(Dense(30, init='normal', activation='relu')) model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) 
    return model
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, nb_epoch=300,
    batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(y=encoded_Y, n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

不使用Dropout的准确率是82%。

```python
Accuracy: 82.68% (3.90%)
```

#### 16.3 对输入层使用Dropout正则化

可以对表层用Dropout：这里我们对输入层（表层）和第一个隐层用Dropout，比例是20%，意思是每轮训练每5个输入随机去掉1个变量。

原论文推荐对每层的权重加限制，保证模不超过3：在定义全连接层时用```W_constraint```可以做到。学习率加10倍，动量加到0.9，原论文也如此推荐。对上面的模型进行修改：

```python

# dropout in the input layer with weight constraint
def create_model1():
    # create model
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(60,)))
    model.add(Dense(60, init='normal', activation='relu', W_constraint=maxnorm(3))) model.add(Dense(30, init='normal', activation='relu', W_constraint=maxnorm(3))) model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) 
    return model
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model1, nb_epoch=300,
    batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(y=encoded_Y, n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

准确率提升到86%：

```python
Accuracy: 86.04% (6.33%)
```

#### 16.4 对隐层使用Dropout正则化

隐层当然也可以用Dropout。和上次一样，这次对两个隐层都做Dropout，概率还是20%：

```python
# dropout in hidden layers with weight constraint
def create_model2():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=60, init='normal', activation='relu',
W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(30, init='normal', activation='relu', W_constraint=maxnorm(3))) 
    model.add(Dropout(0.2))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)  
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) 
    return model
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model2, nb_epoch=300,
    batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(y=encoded_Y, n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

然而并没有什么卯月，效果更差了。有可能需要多训练一些吧。

```python
Accuracy: 82.16% (6.16%)
```

#### 16.5 使用Dropout正则化的技巧

原论文对很多标准机器学习问题做出了比较，并提出了下列建议：

1. Dropout概率不要太高，从20%开始，试到50%。太低的概率效果不好，太高有可能欠拟合。
2. 网络要大。更大的网络学习到不同方法的几率更大。
3. 每层都做Dropout，包括输入层。效果更好。
4. 学习率（带衰减的）和动量要大。直接对学习率乘10或100，动量设到0.9或0.99。
5. 限制每层的权重。学习率增大会造成权重增大，把每层的模限制到4或5的效果更好。

#### 16.6 总结

本章关于使用Dropout正则化避免过拟合。总结一下：

- Dropout的工作原理是什么
- 如何使用Dropout
- Dropout的最佳实践是什么

##### 16.6.1 下一章

在训练中调节学习率会提升性能。下一章会研究不同学习率的效果，以及如何在Keras中使用。


