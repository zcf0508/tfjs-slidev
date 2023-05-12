<script setup lang="ts">
import * as tf from '@tensorflow/tfjs';
import { ref, watch } from 'vue';

const inputs = ref('');
const result = ref('');


function getModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [1],
    units: 2,
  }));
  model.add(tf.layers.dense({units: 3}));
  model.add(tf.layers.dense({units: 1}));

  // 设置优化函数
  // const optimizer = tf.train.adam();
  // model.compile({
  //   optimizer: optimizer,
  //   loss: 'meanSquaredError',
  // });
  
  return model;
}

const model = getModel();
async function run() {  
  const res = await (model.predict(
    tf.tensor1d(inputs.value.split(',').map(Number)),
  ) as tf.Tensor<tf.Rank>).array();

  result.value = res[0][0]
}


</script>

<template>
  <div class="my-2">
    <div class="my-2">
      <input class="border border-gray-300 px-2 py-1 mr-4 rounded w-[300px]" v-model="inputs" placeholder="输入两个数字，用 `,` 分隔">
      <button class="border border-gray-300 p-2 rounded" @click="run">运行</button>
    </div>  
    <template v-if="result">
      {{ result }}
    </template>
  </div>
</template>

<style scoped>

</style>
