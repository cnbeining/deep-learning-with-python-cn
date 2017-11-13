### 第14章 使用保存点保存最好的模型

深度学习有可能需要跑很长时间，如果中间断了（特别是在竞价式实例上跑的时候）就要亲命了。本章关于在训练时中途保存模型。本章将：

- 保存点很重要！
- 每轮打保存点！
- 挑最好的模型！

我们开始吧。

#### 14.1 使用保存点

长时间运行的程序需要能中途保存，加强健壮性。保存的程序应该可以继续运行，或者直接运行。深度学习的保存点用来存储模型的权重：这样可以继续训练，或者直接开始预测。

Keras有回调API，配合```ModelCheckpoint```可以每轮保存网络信息，可以定义文件位置、文件名和保存时机等。例如，损失函数或准确率达到某个标准就保存，文件名的格式可以加入时间和准确率等。```ModelCheckpoint```需要传入```fit()```函数，也需要安装```h5py```库。

#### 14.2 效果变好就保存

好习惯：每轮如果效果变好就保存一下。还是用第7章的模型，用33%的数据测试。

每轮后在测试数据集上验证，如果比之前效果好就保存权重（monitor='val_acc', mode='max'）。文件名格式是```weights-improvement-val_acc=.2f.hdf5```。

```python
# Checkpoint the weights when validation accuracy improves
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
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
# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
mode='max')
callbacks_list = [checkpoint]
# Fit the model
model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10,
    callbacks=callbacks_list, verbose=0)
```

输出的结果如下：如果效果更好就保存。

```python
...
Epoch 00134: val_acc did not improve
Epoch 00135: val_acc did not improve
Epoch 00136: val_acc did not improve
Epoch 00137: val_acc did not improve
Epoch 00138: val_acc did not improve
Epoch 00139: val_acc did not improve
Epoch 00140: val_acc improved from 0.83465 to 0.83858, saving model to
    weights-improvement-140-0.84.hdf5
Epoch 00141: val_acc did not improve
Epoch 00142: val_acc did not improve
Epoch 00143: val_acc did not improve
Epoch 00144: val_acc did not improve
Epoch 00145: val_acc did not improve
Epoch 00146: val_acc improved from 0.83858 to 0.84252, saving model to
    weights-improvement-146-0.84.hdf5
Epoch 00147: val_acc did not improve
Epoch 00148: val_acc improved from 0.84252 to 0.84252, saving model to
    weights-improvement-148-0.84.hdf5
Epoch 00149: val_acc did not improve
```

目录下会保存每次的模型：

```python
...
weights-improvement-74-0.81.hdf5
weights-improvement-81-0.82.hdf5
weights-improvement-91-0.82.hdf5
weights-improvement-93-0.83.hdf5
```

这种方法有效，但是文件较多。当然最好的模型肯定保存下来了。

#### 14.3 保存最好的模型

也可以只保存最好的模型：每次如果效果变好就覆盖之前的权重文件，把之前的文件名改成固定的就可以：

```python
# Checkpoint the weights for best model on validation accuracy
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
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
model.add(Dense(12, input_dim=8, init='uniform', activation='relu')) model.add(Dense(8, init='uniform', activation='relu')) model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
mode='max')
callbacks_list = [checkpoint]
# Fit the model
model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10,
    callbacks=callbacks_list, verbose=0)
```

结果如下：

```python
...
Epoch 00136: val_acc did not improve
Epoch 00137: val_acc did not improve
Epoch 00138: val_acc did not improve
Epoch 00139: val_acc did not improve
Epoch 00140: val_acc improved from 0.83465 to 0.83858, saving model to weights.best.hdf5
Epoch 00141: val_acc did not improve
Epoch 00142: val_acc did not improve
Epoch 00143: val_acc did not improve
Epoch 00144: val_acc did not improve
Epoch 00145: val_acc did not improve
Epoch 00146: val_acc improved from 0.83858 to 0.84252, saving model to weights.best.hdf5
Epoch 00147: val_acc did not improve
Epoch 00148: val_acc improved from 0.84252 to 0.84252, saving model to weights.best.hdf5
Epoch 00149: val_acc did not improve
```

网络保存在：

```
weights.best.hdf5
```

#### 14.4 导入保存的模型

保存点只保存权重，网络结构需要预先保存。参见第13章，代码如下：

```python
# How to load and use weights from a checkpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
 import matplotlib.pyplot as plt
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu')) model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# load weights
model.load_weights("weights.best.hdf5")
# Compile model (required to make predictions) model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) print("Created model and loaded weights from file")
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# estimate accuracy on whole dataset using loaded weights
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```

结果如下：

```
Created model and loaded weights from file
acc: 77.73%
```

#### 14.5 总结

本章关于在训练时保存检查点。总结一下：

- 如何在优化时保存网络
- 如何保存最好的网络
- 如何导入网络

##### 14.5.1 下一章

本章关于建立保存点：下一章关于在训练时画性能图表。


