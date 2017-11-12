### 第8章 测试神经网络

深度学习有很多参数要调：大部分都是拍脑袋的。所以测试特别重要：本章我们讨论几种测试方法。本章将：

- 使用Keras进行自动验证
- 使用Keras进行手工验证
- 使用Keras进行K折交叉验证

我们开始吧。

#### 8.1 口算神经网络

创建神经网络时有很多参数：很多时候可以从别人的网络上抄，但是最终还是需要一点点做实验。无论是网络的拓扑结构（层数、大小、每层类型）还是小参数（损失函数、激活函数、优化算法、训练次数）等。

一般深度学习的数据集都很大，数据有几十万乃至几亿个。所以测试方法至关重要。

#### 8.2 分割数据

数据量大和网络复杂会造成训练时间很长，所以需要将数据分成训练、测试或验证数据集。Keras提供两种办法：

1. 自动验证
2. 手工验证

##### 8.2.1 自动验证

Keras可以将数据自动分出一部分，每次训练后进行验证。在训练时用```validation_split```参数可以指定验证数据的比例，一般是总数据的20%或者33%。下面的代码在第七章上加入了自动验证：

```
# MLP with automatic validation set
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
model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10)
```

训练时，每轮会显示训练和测试数据的数据：

```
Epoch 145/150
514/514 [==============================] - 0s - loss: 0.4885 - acc: 0.7743 - val_loss:
    0.5016 - val_acc: 0.7638
Epoch 146/150
514/514 [==============================] - 0s - loss: 0.4862 - acc: 0.7704 - val_loss:
    0.5202 - val_acc: 0.7323
Epoch 147/150
514/514 [==============================] - 0s - loss: 0.4959 - acc: 0.7588 - val_loss:
    0.5012 - val_acc: 0.7598
Epoch 148/150
514/514 [==============================] - 0s - loss: 0.4966 - acc: 0.7665 - val_loss:
    0.5244 - val_acc: 0.7520
Epoch 149/150
514/514 [==============================] - 0s - loss: 0.4863 - acc: 0.7724 - val_loss:
    0.5074 - val_acc: 0.7717
Epoch 150/150
514/514 [==============================] - 0s - loss: 0.4884 - acc: 0.7724 - val_loss:
    0.5462 - val_acc: 0.7205
```

##### 8.2.2 手工验证

Keras也可以手工进行验证。我们定义一个```train_test_split```函数，将数据分成2：1的测试和验证数据集。在调用```fit()```方法时需要加入```validation_data```参数作为验证数据，数组的项目分别是输入和输出数据。

```
# MLP with manual validation set
from keras.models import Sequential
from keras.layers import Dense
from sklearn.cross_validation import train_test_split
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed) # create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test,y_test), nb_epoch=150, batch_size=10)
```

和自动化验证一样，每轮训练后，Keras会输出训练和验证结果：

```
...
Epoch 145/150
514/514 [==============================] - 0s - loss: 0.5001 - acc: 0.7685 - val_loss:
    0.5617 - val_acc: 0.7087
Epoch 146/150
514/514 [==============================] - 0s - loss: 0.5041 - acc: 0.7529 - val_loss:
    0.5423 - val_acc: 0.7362
Epoch 147/150
514/514 [==============================] - 0s - loss: 0.4936 - acc: 0.7685 - val_loss:
    0.5426 - val_acc: 0.7283
Epoch 148/150
514/514 [==============================] - 0s - loss: 0.4957 - acc: 0.7685 - val_loss:
    0.5430 - val_acc: 0.7362
Epoch 149/150
514/514 [==============================] - 0s - loss: 0.4953 - acc: 0.7685 - val_loss:
    0.5403 - val_acc: 0.7323
Epoch 150/150
514/514 [==============================] - 0s - loss: 0.4941 - acc: 0.7743 - val_loss:
    0.5452 - val_acc: 0.7323
```

#### 8.3 手工K折交叉验证

机器学习的金科玉律是K折验证，以验证模型对未来数据的预测能力。K折验证的方法是：将数据分成K组，留下1组验证，其他数据用作训练，直到每种分发的性能一致。

深度学习一般不用交叉验证，因为对算力要求太高。例如，K折的次数一般是5或者10折：每组都需要训练并验证，训练时间成倍上升。然而，如果数据量小，交叉验证的效果更好，误差更小。

scikit-learn有```StratifiedKFold```类，我们用它把数据分成10组。抽样方法是分层抽样，尽可能保证每组数据量一致。然后我们在每组上训练模型，使用```verbose=0```参数关闭每轮的输出。训练后，Keras会输出模型的性能，并存储模型。最终，Keras输出性能的平均值和标准差，为性能估算提供更准确的估计：

```
# MLP for Pima Indians Dataset with 10-fold cross validation
from keras.models import Sequential
from keras.layers import Dense
from sklearn.cross_validation import StratifiedKFold
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
  # split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# define 10-fold cross validation test harness
kfold = StratifiedKFold(y=Y, n_folds=10, shuffle=True, random_state=seed)
cvscores = []
for i, (train, test) in enumerate(kfold):
  # create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu')) model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
model.fit(X[train], Y[train], nb_epoch=150, batch_size=10, verbose=0)
# evaluate the model
scores = model.evaluate(X[test], Y[test], verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)) cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
```

输出是：

```
acc: 77.92%
acc: 79.22%
acc: 76.62%
acc: 77.92%
acc: 75.32%
acc: 74.03%
acc: 77.92%
acc: 71.43%
acc: 71.05%
acc: 75.00%
75.64% (+/- 2.67%)
```

每次循环都需要重新生成模型，使用对应的数据训练。下一章我们用scikit-learn直接使用Keras的模型。

#### 8.4 总结

本章关于测试神经网络的性能。总结一下：

- 如何自动将数据分成训练和测试组
- 如何人工对数据分组
- 如何使用K折法测试性能

##### 8.4.1 下一章

现在你已经知道如何如何测试神经网络的性能：下一章讲讲如何在scikit-learn中直接使用Keras的模型。

