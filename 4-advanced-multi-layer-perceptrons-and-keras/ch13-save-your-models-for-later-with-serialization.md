### 第13章 用序列化保存模型

深度学习的模型有可能需要好几天才能训练好，如果没有SL大法就完蛋了。本章关于如何保存和加载模型。本章将：

- 使用HDF5格式保存模型
- 使用JSON格式保存模型
- 使用YAML格式保存模型

我们开始吧。

#### 13.1 简介

Keras中，模型的结构和权重数据是分开的：权重的文件格式是HDF5，这种格式保存数字矩阵效率很高。模型的结构用JSON或YAML导入导出。

本章包括如何手工修改HDF5文件，使用的模型是第7章的皮马人糖尿病模型。

##### 13.1.1 HDF5文件

分层数据格式，版本5（HDF5）可以高效保存大实数矩阵，例如神经网络的权重。HDF5的包需要安装：

```python
sudo pip install h5py
```

#### 13.2 使用JSON保存网络结构

JSON的格式很简单，Keras可以用```to_json()```把模型导出为JSON格式，再用```model_from_json()```加载回来。

```save_weights()```和```load_weights()```可以保存和加载模型权重。下面的代码把之前的模型保存到JSON文件```model.json```，权重保存到HDF5文件```model.h5```，然后加载回来。

模型和权重加载后需要编译一次，让Keras正确调用后端。模型的验证方法和之前一致：

导出：

```python
# MLP for Pima Indians Dataset serialize to JSON and HDF5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
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
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10, verbose=0)
# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
```

导入：

```python
# later...
# load json and create model
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
```

结果如下。导入的模型和之前导出时一致：


```python
acc: 79.56%
Saved model to disk
Loaded model from disk
acc: 79.56%
```

JSON文件类似：

```python
{
    "class_name": "Sequential",
    "config": [{
            "class_name": "Dense",
            "config": {
                "W_constraint": null,
                "b_constraint": null,
                "name": "dense_1",
                "output_dim": 12,
                "activity_regularizer": null,
                "trainable": true,
                "init": "uniform",
                "input_dtype": "float32",
                "input_dim": 8,
                "b_regularizer": null,
                "W_regularizer": null,
                "activation": "relu",
                "batch_input_shape": [
                    null,
                    8
                ]
            }
        },
        {
            "class_name": "Dense",
            "config": {
                "W_constraint": null,
                "b_constraint": null,
                "name": "dense_2",
                "activity_regularizer": null,
                "trainable": true,
                "init": "uniform",
                "input_dim": null,
                "b_regularizer": null,
                "W_regularizer": null,
                "activation": "relu",
                "output_dim": 8
            }
        },
        {
            "class_name": "Dense",
            "config": {
                "W_constraint": null,
                "b_constraint": null,
                "name": "dense_3",
                "activity_regularizer": null,
                "trainable": true,
                "init": "uniform",
                "input_dim": null,
                "b_regularizer": null,
                "W_regularizer": null,
                "activation": "sigmoid",
                "output_dim": 1
            }
        }
    ]
}
```

#### 13.3 使用YAML保存网络结构

和之前JSON类似，只不过文件格式变成YAML，使用的函数变成了```to_yaml()```和```model_from_yaml()```：

```python
# MLP for Pima Indians Dataset serialize to YAML and HDF5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml
import numpy
import os
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
model.fit(X, Y, nb_epoch=150, batch_size=10, verbose=0)
# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# later...
# load YAML and create model
yaml_file = open('model.yaml', 'r') loaded_model_yaml = yaml_file.read() yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml) 
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy']) score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
```

结果和之前的一样：

```python
acc: 79.56%
Saved model to disk
Loaded model from disk
acc: 79.56%
```

YAML文件长这样：

```yaml
class_name: Sequential
config:
- class_name: Dense
  config:
    W_constraint: null
    W_regularizer: null
   activation: relu
    activity_regularizer: null
    b_constraint: null
    b_regularizer: null
    batch_input_shape: !!python/tuple [null, 8]
    init: uniform
    input_dim: 8
    input_dtype: float32
    name: dense_1
    output_dim: 12
    trainable: true
- class_name: Dense
  config: {W_constraint: null, W_regularizer: null, activation: relu, activity_regularizer:
      null,
    b_constraint: null, b_regularizer: null, init: uniform, input_dim: null, name: dense_2,
    output_dim: 8, trainable: true}
- class_name: Dense
  config: {W_constraint: null, W_regularizer: null, activation: sigmoid,
      activity_regularizer: null,
    b_constraint: null, b_regularizer: null, init: uniform, input_dim: null, name: dense_3,
    output_dim: 1, trainable: true}
```

#### 13.4 总结

本章关于导入导出Keras模型。总结一下：

- 如何用HDF5保存加载权重
- 如何用JSON保存加载模型
- 如何用YAML保存加载模型

##### 13.4.1 下一章

模型可以保存了：下一章关于使用保存点。


