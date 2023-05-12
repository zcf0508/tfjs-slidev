---
layout: cover
background: ./images/pietro-jeng-n6B49lTx7NM-unsplash.jpg
title: tfjs 介绍及应用
---

# tfjs 介绍及应用

Introduction and application of tfjs

---
---
## 机器学习
Machine Learning

### 🤔 定义

- 机器学习理论主要是设计和分析一些让计算机可以 **自动「学习」** 的算法。

- 机器学习算法是一类从数据中自动分析获得规律，并利用规律对未知数据进行 **预测** 的算法。

- 因为学习算法中涉及了大量的统计学理论，机器学习与 **推断统计学** 联系尤为密切，也被称为统计学习理论。

- 分类
  - **人工神经网络**
  - 贝叶斯分类器
  - 马尔可夫链
  - ...

---
---
## 机器学习
Machine Learning

- 普通函数
```ts
// 有程序员来编写函数的功能
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

```ts
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

- 深度学习是机器学习的分支，是一种以人工神经网络为架构，对资料进行表征学习的算法。

- 深度学习中的形容词“深度”是指在网络中使用多层。


---
---
## 深度学习
Deep Learning

### 相关概念

- 神经元
  - 神经元是机器学习中的一个基本概念，它是一种数学模型，用于模拟人类神经系统中的神经元。
  - 在机器学习中，神经元通常被用作构建神经网络的基本单元。
  - 每个神经元接收一组输入，对这些输入进行加权处理，并通过一个 **激活函数** 将结果输出。
  - 神经元的输出可以被传递给其他神经元，从而构建出一个复杂的神经网络，用于解决各种机器学习问题。

---
---
## 深度学习
Deep Learning

### 相关概念

- 层
  - 在深度学习中，层是指神经网络中的一个组成部分，它由多个神经元组成，通常被用于对输入数据进行特征提取和转换。
  - 每一层接收上一层的输出作为输入，并对其进行一定的变换，然后将结果传递给下一层。
  - 深度学习中的神经网络通常由多个层组成，每一层都有自己的权重和偏置，用于对输入数据进行不同的变换和处理。
  - 通过不断堆叠多个层，深度学习模型可以学习到更加复杂的特征和模式，从而实现更加准确的预测和分类。

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

```ts
// 激活函数
function activation(x: number): number {
  return x > 0 ? 1 : 0
}

class Node {
  private weight: number // 权重
  private bias: number // 偏置

  // 设置默认值
  constructor(weight: number, bias: number) {
    this.weight = weight
    this.bias = bias
  }

  public run(inputs: number[]): number {
    // 通过权重和偏置计算出结果
    const sum = inputs.reduce((sum, input) => sum + input * this.weight + this.bias, 0) 
    // 通过一个激活函数输出
    return activation(sum)
  }
}
```

\* 缺陷： 没有学习能力

---
---
## 一个简单的神经网络

将多个神经元连接起来。

```ts
const node1 = new Node(0.1, 0.1)
const node2 = new Node(0.1, -0.1)

const node3 = new Node(-0.1, 0.1)
const node4 = new Node(0.2, 0.05)
const node5 = new Node(-0.1, 0.2)

const node6 = new Node(0.1, 0.1)

const layers = [
  [node1, node2],        // 2个输入神经元
  [node3, node4, node5], // 3个隐藏神经元
  [node6]                // 1个输出神经元
]

```

---
---
## tfjs 框架介绍

> TensorFlow.js 是一个用于使用 JavaScript 进行机器学习开发的库。
> 
> 使用 JavaScript 开发机器学习模型，并直接在浏览器或 Node.js 中使用机器学习模型。

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

---
---
## 运行一下

<tfjs-run-1></tfjs-run-1>

为什么每次值都不一样？
```vue
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

<tfjs-run-2></tfjs-run-2>
