---
layout: cover
background: /pietro-jeng-n6B49lTx7NM-unsplash.jpg
title: tfjs 介绍及应用
---

# tfjs 介绍及应用

Introduction and application of tfjs

---
---
## 机器学习
Machine Learning

### 🤔 定义

<v-clicks>

- 机器学习理论主要是设计和分析一些让计算机可以 **自动「学习」** 的算法。

- 机器学习算法是一类从数据中自动分析获得规律，并利用规律对未知数据进行 **预测** 的算法。

- 因为学习算法中涉及了大量的统计学理论，机器学习与 **推断统计学** 联系尤为密切，也被称为统计学习理论。

- 分类
  - **人工神经网络**
  - 贝叶斯分类器
  - 马尔可夫链
  - ...

</v-clicks>
---
---
## 机器学习
Machine Learning

- 普通函数
```ts {6-7|1-4|9}
// 由程序员来编写函数的功能
function add(a, b) {
  return a + b
}

const a = 1
const b = 2

const c = add(a, b)
```

---
---
## 机器学习
Machine Learning

- 机器学习

```ts {9-11|1-7|12-13|15-18}
function learn(a, b, c) {
  return (a, b) => {
    // 通过学习得到的函数来预测 c
    // 在这里则是学习 c = a + b
    return a + b
  }
}

const a = 1
const b = 2
const c = 3
// 生成了一个函数
const add = learn(a, b, c)

// 使用 add 函数
const x = 2
const y = 3
const z = add(x, y)

```

---
---
## 深度学习
Deep Learning

### 🤔 定义

<v-clicks>

- 深度学习是机器学习的分支，是一种以人工神经网络为架构，对资料进行表征学习的算法。

- 深度学习中的形容词“深度”是指在网络中使用多层。

</v-clicks>

---
---
## 深度学习
Deep Learning

### 相关概念

<v-clicks depth=2>

- 神经元
  - 神经元是机器学习中的一个基本概念，它是一种数学模型，用于模拟人类神经系统中的神经元。
  - 在机器学习中，神经元通常被用作构建神经网络的基本单元。
  - 每个神经元接收一组输入，对这些输入进行加权处理，并通过一个 **激活函数** 将结果输出。
  - 神经元的输出可以被传递给其他神经元，从而构建出一个复杂的神经网络，用于解决各种机器学习问题。

</v-clicks>

---
---
## 深度学习
Deep Learning

### 相关概念

<v-clicks depth=2>

- 层
  - 在深度学习中，层是指神经网络中的一个组成部分，它由多个神经元组成，通常被用于对输入数据进行特征提取和转换。
  - 每一层接收上一层的输出作为输入，并对其进行一定的变换，然后将结果传递给下一层。
  - 深度学习中的神经网络通常由多个层组成，每一层都有自己的权重和偏置，用于对输入数据进行不同的变换和处理。
  - 通过不断堆叠多个层，深度学习模型可以学习到更加复杂的特征和模式，从而实现更加准确的预测和分类。

</v-clicks>

---
---
## 深度学习
Deep Learning

### 一个简单的神经网络

![神经网络](/20180413145539360.jpg)

---
---
## 一个简单的神经元

神经元可以理解为一个类，它包含一个权重和一个偏置，输入几个值，输出一个值。

<v-click>

```ts {6-7|12-18|1-3|all}
function activation(x: number): number { // 激活函数
  return x > 0 ? 1 : 0
}

class Node {
  private weights: number[] // 权重
  private bias: number // 偏置
  constructor(weights: number[], bias: number) {
    this.weights = weights
    this.bias = bias
  }
  public forward(inputs: number[]): number {
    let sum = 0
    for (let i = 0; i < inputs.length; i++) {
      sum += inputs[i] * this.weights[i] + this.bias // 通过权重和偏置计算出结果
    }
    return activation(sum) // 通过一个激活函数输出
  }
}
```

</v-click>

<v-click>

\* 缺陷： 没有学习能力

</v-click>
---
---
## 反向传播

```ts {11-22|16-18|19-21|1-7|all}
const getOptimizer = (learnRate: number) {
  return (weight: number, gradient: number) => {
    return weight - learnRate * gradient
  }
}
const lr = 1e-3
const optimizer = getOptimizer(lr)

class Node {
  ...
  /**
   * @param target 目标值
   * @param result 实际值
   */
  function backward(target: number, result: number, input: number, idx: number) {
    // 计算梯度
    const loss = result - target
    const gradient = loss * input
    // 更新权重和偏置
    this.weights[idx] = optimizer(this.weights[idx], gradient)
    this.bias = optimizer(this.bias, gradient)
  }
  ...
}
```

---
---
## 一个简单的神经网络

将多个神经元连接起来。

<v-click>

```ts
const node1 = new Node([Math.random(), Math.random()], Math.random())
const node2 = new Node([Math.random(), Math.random()], Math.random())

const node3 = new Node([Math.random(), Math.random()], Math.random())
const node4 = new Node([Math.random(), Math.random()], Math.random())
const node5 = new Node([Math.random(), Math.random()], Math.random())

const node6 = new Node([Math.random(), Math.random(), Math.random()], Math.random())

const layers = [
  [node1, node2],        // 2个输入神经元
  [node3, node4, node5], // 3个隐藏神经元
  [node6]                // 1个输出神经元
]

```

</v-click>

---
---
## tfjs 框架介绍

<v-click>

> TensorFlow.js 是一个用于使用 JavaScript 进行机器学习开发的库。
> 
> 使用 JavaScript 开发机器学习模型，并直接在浏览器或 Node.js 中使用机器学习模型。

</v-click>

<v-click>

```ts
import * as tf from '@tensorflow/tfjs';

function getModel() {
  const model = tf.sequential();

  model.add(tf.layers.dense({
    inputShape: [1], // 一维向量
    units: 2
  }));
  model.add(tf.layers.dense({units: 3}));
  model.add(tf.layers.dense({units: 1}));

  return model
}

```

</v-click>

---
---
## 运行一下

<v-click>

<tfjs-run-1></tfjs-run-1>

</v-click>

<v-click>

为什么每次值都不一样？

</v-click>

<v-click>

```vue {5-6|2-3}
<script setup lang=ts>
// 使用同一个模型
// const model = getModel()
function run() {
  //每次都生成新的模型
  const model = getModel()
}
</script>
<template>
  <button @click="run"></button>
</template>
```

</v-click>

<v-click>

<tfjs-run-2></tfjs-run-2>

</v-click>

---
---
## 模型训练

<v-click>

### 1. 准备数据

</v-click>

<v-click>

```ts
type TrainData = {
  addends: number[]
  res: number
}

type DataSet = {
  trainSet: TrainData[]
  testSet: TrainData[]
}

const getDateSet = () => {
  return {
    trainSet: [
      ...
    ],
    testSet: [
      ...
    ],
  } as DataSet
}

```

</v-click>

---
---

## 模型训练

<v-click>

### 2. 训练模型

</v-click>

<v-click>

```ts
const model = getModel()
const dataSet = getDateSet()

const trainDataAddends = dataSet.trainSet.map(item => item.addends)
const trainDataRes = dataSet.trainSet.map(item => item.res)
const testDataAddends = dataSet.testSet.map(item => item.addends)
const testDataRes = dataSet.testSet.map(item => item.res)

model.compile({
  optimizer: tf.train.adam(),
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
})

await model.fit(trainDataAddends, trainDataRes, {
  validationData: [testDataAddends, testDataRes],
  epochs: 10,
})
```

</v-click>

<v-click>

🎇 训练完成！

</v-click>

---
---
## tfjs 的优势和劣势

<v-clicks>

- 优势
  - 用户不需要设置开发环境，只要有浏览器就可以
  - 使用 js/ts 进行开发，不需要学习新的语言
  - 训练好的模型可以直接在浏览器中运行
- 劣势
  - 无法充分使用硬件资源，训练速度慢
  - 由于浏览器限制，只能训练小模型

</v-clicks>

---
---
## tfjs 相关资源

- [tfjs 官网](https://www.tensorflow.org/js)
- [教程 - 使用 CNN 识别手写数字](https://codelabs.developers.google.com/codelabs/tfjs-training-classfication/index.html?hl=zh-cn#0)
- [TensoeFlow Hub](https://tfhub.dev/s?deployment-format=tfjs)

---
---
## tensorflow 框架的简单介绍

tensorflow 是基于 Python 语言的深度学习框架。

<v-click>

```python
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
# 数据归一化
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```

</v-click>

---
---
## 模型转换

<v-click>

我们可以将使用 tensorflow 训练得到的模型，转换为 tfjs 所能接受的格式。

这样我们就可以使用 python 进行模型的训练，然后使用 js/ts 运行训练好的模型。

</v-click>

<v-click>

```shell
pip install tensorflowjs

tensorflowjs_converter \
    --input_format=tf_saved_model \     # 输入模型的格式
    --output_node_names='web_model' \   # 输出模型的名称
    /mobilenet/saved_model \            # 输入模型的路径
    /mobilenet/web_model                # 转换后输出的路径
```

转换后生成的文件路径

- /web_model
  - **model.json**
  - /group1-shard

</v-click>

---
---
## 模型转换

在浏览器中加载转换后的模型

<v-click>

```ts {2|4-5|7}
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';

// 可以是本地地址，或者网络地址
const MODEL_URL = 'model_directory/model.json';

const model = await loadGraphModel(MODEL_URL);
```

</v-click>

<v-click>

**注意:**

</v-click>

<v-clicks>

- 转换后的模型的输入和输出，一般要与 python 版本一致。

- 如果 python 版本输入的是一个灰度图像，那么 js 版本的输入也应该是一个灰度图像。灰度图像就是说，图片不再由红绿蓝三个通道组成，而是只有一个灰度通道。

</v-clicks>

---
---
## 试一下

<tfjs-face-detection></tfjs-face-detection>
