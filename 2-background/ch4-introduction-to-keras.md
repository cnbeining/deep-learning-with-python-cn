### 第4章 Keras入门

Python的科学计算包主要是Theano和TensorFlow：很强大，但有点难用。Keras可以基于这两种包之一方便地建立神经网络。本章包括：

- 使用Keras进行深度学习
- 如何配置Keras的后端
- Keras的常见操作

我们开始吧。

#### 4.1 Keras是什么？

Keras可以基于Theano或TensorFlow建立深度学习模型，方便研究和开发。Keras可以在Python 2.7或3.5运行，无痛调用后端的CPU或GPU网络。Keras由Google的Francois Chollet开发，遵循以下原则：

- 模块化：每个模块都是单独的流程或图，深度学习的所有问题都可以通过组装模块解决
- 简单化：提供解决问题的最简单办法，不加装饰，最大化可读性
- 扩展性：新模块的添加特别容易，方便试验新想法
- Python：不使用任何自创格式，只使用原生Python

#### 4.2 安装Keras

Keras很好安装，但是你需要至少安装Theano或TensorFlow之一。

使用PyPI安装Keras：

```
sudo pip install keras
```

本书完成时，Keras的最新版本是1.0.1。下面这句话可以看Keras的版本：

```
python -c "import keras; print keras.__version__"
```

Python会显示Keras的版本号，例如：

```
1.0.1
```

Keras的升级也是一句话：

```
sudo pip install --upgrade keras
```

#### 4.3 配置Keras的后端

Keras是Theano和TensorFlow的轻量级API，所以必须配合后端使用。后端配置只需要一个文件：

```
~/.keras/keras.json
```

里面是：

```
{"epsilon": 1e-07, "floatx": "float32", "backend": "theano"}
```

默认的后端是```theano ```，可以改成```tensorflow```。下面这行命令会显示Keras的后端：

```
python -c "from keras import backend; print backend._BACKEND"
```

默认会显示：

```
Using Theano backend.
theano
```

变量```KERAS_BACKEND```可以控制Keras的后端，例如：

```
KERAS_BACKEND=tensorflow python -c "from keras import backend; print backend._BACKEND"
```

会输出：

```
Using TensorFlow backend.
tensorflow
```

#### 4.4 使用Keras搭建深度学习模型

Keras的目标就是搭建模型。最主要的模型是```Sequential```：不同层的叠加。模型创建后可以编译，调用后端进行优化，可以指定损失函数和优化方式。

编译后的模型需要导入数据：可以一批批加入数据，也可以一次性全加入。所有的计算在这步进行。训练后的模型就可以做预测或分类了。大体上的步骤是：

1. 定义模型：创建```Sequential```模型，加入每一层
2. 编译模型：指定损失函数和优化方式，使用模型的```compile()```方法
3. 拟合数据：使用模型的```fit()```方法拟合数据
4. 进行预测：使用模型的```evaluate()``` 或 ```predict()```方法进行预测

#### 4.5 总结

本章关于Keras。总结一下：

- Keras是Theano和TensorFlow的封装，降低了复杂性
- Keras是最小化、模块化的封装，可以迅速上手
- Keras可以通过定义-编译-拟合搭建模型，进行预测

##### 4.5.1 下一章

这是Python机器学习的最前沿：下个项目我们一步步在云上搭建机器学习的环境。
