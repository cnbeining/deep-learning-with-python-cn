### 第3章 TensorFlow入门

TensorFlow是Google创造的数值运算库，作为深度学习的底层使用。本章包括：

- TensorFlow介绍
- 如何用TensorFlow定义、编译并运算表达式
- 如何寻求帮助

注意：TensorFlow暂时不支持Windows，你可以用Docker或虚拟机。Windows用户可以不看这章。

#### 3.1 TensorFlow是什么？

TensorFlow是开源数学计算引擎，由Google创造，用Apache 2.0协议发布。TF的API是Python的，但底层是C++。和Theano不同，TF兼顾了工业和研究，在RankBrain、DeepDream等项目中使用。TF可以在单个CPU或GPU，移动设备以及大规模分布式系统中使用。

#### 3.2 安装TensorFlow

TF支持Python 2.7和3.3以上。安装很简单：

```
sudo pip install TensorFlow
```

就好了。

#### 3.3 TensorFlow例子

TF的计算是用图表示的：

- 节点：节点进行计算，有一个或者多个输入输出。节点间的数据叫张量：多维实数数组。
- 边缘：定义数据、分支、循环和覆盖的图，也可以进行高级操作，例如等待某个计算完成。
- 操作：取一个输入值，得出一个输出值，例如，加减乘除。

#### 3.4 简单的TensorFlow

简单说一下TensorFlow：我们定义a和b两个浮点变量，定义一个表达式（c=a+b），将表达式变成函数，编译，进行计算：

```
import tensorflow as tf
# declare two symbolic floating-point scalars
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
# create a simple symbolic expression using the add function add = tf.add(a, b)
# bind 1.5 to 'a', 2.5 to 'b', and evaluate 'c'
sess = tf.Session()
binding = {a: 1.5, b: 2.5}
c = sess.run(add, feed_dict=binding)
print(c)
```

结果是4： 1.5+2.5=4.0。大的矩阵操作类似。

#### 3.5 其他深度学习模型

TensorFlow自带很多模型，可以直接调用：首先，看看TensorFlow的安装位置：

```
python -c 'import os; import inspect; import tensorflow; print(os.path.dirname(inspect.getfile(tensorflow)))'
```

结果类似于：

```
/usr/lib/python2.7/site-packages/tensorflow
```

进入该目录，可以看见很多例子：

- 多线程word2vec mini-batch Skip-Gram模型
- 多线程word2vec Skip-Gram模型
- CIFAR-10的CNN模型
- 类似LeNet-5的端到端的MNIST模型
- 带注意力机制的端到端模型

example目录带有MNIST数据集的例子，TensorFlow的网站也很有帮助，包括不同的网络、数据集。TensorFlow也有个网页版，可以直接试验。

#### 3.6 总结

本章关于TensorFlow。总结一下：

- TensorFlow和Theano一样，都是数值计算库
- TensorFlow和Theano一样可以直接开发模型
- TensorFlow比Theano包装的好一些

##### 3.6.1 下一章

下一章我们研究Keras：我们用这个库开发深度学习模型。