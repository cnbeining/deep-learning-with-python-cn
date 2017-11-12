### 第9章 使用Scikit-Learn调用Keras的模型

scikit-learn是最受欢迎的Python机器学习库。本章我们将使用scikit-learn调用Keras生成的模型。本章将：

- 使用scikit-learn封装Keras的模型
- 使用scikit-learn对Keras的模型进行交叉验证
- 使用scikit-learn，利用网格搜索调整Keras模型的超参

我们开始吧。

#### 9.1 简介

Keras在深度学习很受欢迎，但是只能做深度学习：Keras是最小化的深度学习库，目标在于快速搭建深度学习模型。基于SciPy的scikit-learn，数值运算效率很高，适用于普遍的机器学习任务，提供很多机器学习工具，包括但不限于：

- 使用K折验证模型
- 快速搜索并测试超参

Keras为scikit-learn封装了```KerasClassifier```和```KerasRegressor```。本章我们继续使用第7章的模型。

#### 9.2 使用交叉验证检验深度学习模型

Keras的```KerasClassifier```和```KerasRegressor```两个类接受```build_fn```参数，传入编译好的模型。我们加入```nb_epoch=150```和```batch_size=10```这两个参数：这两个参数会传入模型的```fit()```方法。我们用scikit-learn的```StratifiedKFold```类进行10折交叉验证，测试模型在未知数据的性能，并使用```cross_val_score()```函数检测模型，打印结果。

```
# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
import numpy
import pandas
# Function to create model, required for KerasClassifier
def create_model():
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu')) model.add(Dense(8, init='uniform', activation='relu')) model.add(Dense(1, init='uniform', activation='sigmoid'))
  # Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, nb_epoch=150, batch_size=10)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(y=Y, n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

每轮训练会输出一次结果，加上最终的平均性能：

```
...
Epoch 145/150
692/692 [==============================] - 0s - loss: 0.4671 - acc: 0.7803
Epoch 146/150
692/692 [==============================] - 0s - loss: 0.4661 - acc: 0.7847
Epoch 147/150
692/692 [==============================] - 0s - loss: 0.4581 - acc: 0.7803
Epoch 148/150
692/692 [==============================] - 0s - loss: 0.4657 - acc: 0.7688
Epoch 149/150
692/692 [==============================] - 0s - loss: 0.4660 - acc: 0.7659
Epoch 150/150
692/692 [==============================] - 0s - loss: 0.4574 - acc: 0.7702
76/76 [==============================] - 0s
0.756442244065
```

比起手工测试，使用scikit-learn容易的多。

#### 9.3 使用网格搜索调整深度学习模型的参数

使用scikit-learn封装Keras的模型十分简单。进一步想：我们可以给```fit()```方法传入参数，```KerasClassifier```的```build_fn```方法也可以传入参数。可以利用这点进一步调整模型。

我们用网格搜索测试不同参数的性能：```create_model()```函数可以传入```optimizer```和```init```参数，虽然都有默认值。那么我们可以用不同的优化算法和初始权重调整网络。具体说，我们希望搜索：

- 优化算法：搜索权重的方法
- 初始权重：初始化不同的网络
- 训练次数：对模型训练的次数
- 批次大小：每次训练的数据量

所有的参数组成一个字典，传入scikit-learn的```GridSearchCV```类：```GridSearchCV```会对每组参数（2×3×3×3）进行训练，进行3折交叉检验。

计算量巨大：耗时巨长。如果模型小还可以取一部分数据试试。第7章的模型可以用，因为网络和数据集都不大（1000个数据内，9个参数）。最后scikit-learn会输出最好的参数和模型，以及平均值。

```
# MLP for Pima Indians Dataset with grid search via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
import numpy
import pandas
# Function to create model, required for KerasClassifier
def create_model(optimizer='rmsprop', init='glorot_uniform'):
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init=init, activation='relu')) model.add(Dense(8, init=init, activation='relu')) model.add(Dense(1, init=init, activation='sigmoid'))
  # Compile model
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model)
# grid search epochs, batch size and optimizer
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = numpy.array([50, 100, 150])
batches = numpy.array([5, 10, 20])
param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init) grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
```

用CPU差不多要5分钟，结果如下。我们发现使用均匀分布初始化，```rmsprop```优化算法，150轮，批尺寸为5时效果最好，正确率约75%：

```
Best: 0.751302 using {'init': 'uniform', 'optimizer': 'rmsprop', 'nb_epoch': 150, 'batch_size': 5}
0.653646 (0.031948) with: {'init': 'glorot_uniform', 'optimizer': 'rmsprop', 'nb_epoch': 50, 'batch_size': 5}
0.665365 (0.004872) with: {'init': 'glorot_uniform', 'optimizer': 'adam', 'nb_epoch': 50, 'batch_size': 5}
0.683594 (0.037603) with: {'init': 'glorot_uniform', 'optimizer': 'rmsprop', 'nb_epoch': 100, 'batch_size': 5}
0.709635 (0.034987) with: {'init': 'glorot_uniform', 'optimizer': 'adam', 'nb_epoch': 100, 'batch_size': 5}
0.699219 (0.009568) with: {'init': 'glorot_uniform', 'optimizer': 'rmsprop', 'nb_epoch': 150, 'batch_size': 5}
0.725260 (0.008027) with: {'init': 'glorot_uniform', 'optimizer': 'adam', 'nb_epoch': 150, 'batch_size': 5}
0.686198 (0.024774) with: {'init': 'normal', 'optimizer': 'rmsprop', 'nb_epoch': 50, 'batch_size': 5}
0.718750 (0.014616) with: {'init': 'normal', 'optimizer': 'adam', 'nb_epoch': 50, 'batch_size': 5}
0.725260 (0.028940) with: {'init': 'normal', 'optimizer': 'rmsprop', 'nb_epoch': 100, 'batch_size': 5}
0.727865 (0.028764) with: {'init': 'normal', 'optimizer': 'adam', 'nb_epoch': 100, 'batch_size': 5}
0.748698 (0.035849) with: {'init': 'normal', 'optimizer': 'rmsprop', 'nb_epoch': 150, 'batch_size': 5}
0.712240 (0.039623) with: {'init': 'normal', 'optimizer': 'adam', 'nb_epoch': 150, 'batch_size': 5}
0.699219 (0.024910) with: {'init': 'uniform', 'optimizer': 'rmsprop', 'nb_epoch': 50, 'batch_size': 5}
0.703125 (0.011500) with: {'init': 'uniform', 'optimizer': 'adam', 'nb_epoch': 50, 'batch_size': 5}
0.720052 (0.015073) with: {'init': 'uniform', 'optimizer': 'rmsprop', 'nb_epoch': 100, 'batch_size': 5}
0.712240 (0.034987) with: {'init': 'uniform', 'optimizer': 'adam', 'nb_epoch': 100, 'batch_size': 5}
0.751302 (0.031466) with: {'init': 'uniform', 'optimizer': 'rmsprop', 'nb_epoch': 150, 'batch_size': 5}
0.734375 (0.038273) with: {'init': 'uniform', 'optimizer': 'adam', 'nb_epoch': 150, 'batch_size': 5}
...
```
#### 9.4 总结

本章关于使用scikit-learn封装并测试神经网络的性能。总结一下：

- 如何使用scikit-learn封装Keras模型
- 如何使用scikit-learn测试Keras模型的性能
- 如何使用scikit-learn调整Keras模型的超参

使用scikit-learn调整参数比手工调用Keras简便的多。

##### 9.4.1 下一章

现在你已经知道如何如何在scikit-learn调用Keras模型：可以开工了。接下来几章我们会用Keras创造不同的端到端模型，从多类分类问题开始。

