<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick } from 'vue';
import '@tensorflow/tfjs-backend-webgl';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);


import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import { drawResults } from './camera';

const videoRef = ref<HTMLVideoElement>();
const canvasRef = ref<HTMLCanvasElement>();


const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;

let detector = null as unknown as faceLandmarksDetection.FaceLandmarksDetector

async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ 
    'audio': false,
      video: {
      facingMode: 'user',
      width: 640,
      height: 480,
      frameRate: {
        ideal: 60,
      },
    } 
});
  videoRef.value!.srcObject = stream;

  await new Promise((resolve) => {
    videoRef.value!.onloadedmetadata = () => {
      resolve(videoRef.value!);
    };
  });

  videoRef.value!.play()

  const videoWidth = videoRef.value!.videoWidth;
  const videoHeight = videoRef.value!.videoHeight;

  videoRef.value!.width = videoWidth
  videoRef.value!.height = videoHeight

  canvasRef.value!.width = videoWidth
  canvasRef.value!.height = videoHeight

  const ctx = canvasRef.value!.getContext('2d')!;

  ctx.translate(videoRef.value!.videoWidth, 0);
  ctx.scale(-1, 1);

  detector = await faceLandmarksDetection.createDetector(model, {
    runtime: 'tfjs',
    refineLandmarks: true,
    maxFaces: 3,
  })
}

async function detectFace() {
  if (videoRef.value!.readyState < 2) {
    await new Promise((resolve) => {
      videoRef.value!.onloadeddata = () => {
        resolve(videoRef.value!);
      };
    });
  }
  const ctx = canvasRef.value!.getContext('2d')!;
  const faces = await detector.estimateFaces(videoRef.value! , {flipHorizontal: false});
  ctx.drawImage(videoRef.value!, 0, 0, videoRef.value!.videoWidth, videoRef.value!.videoHeight);
  faces && faces.length > 0 && drawResults(ctx, faces, true, true);
  requestAnimationFrame(detectFace);
}

onMounted(() => {
  nextTick(async () => { 
    await initCamera()
    detectFace()
  })
})

onUnmounted(() => {
  detector && detector.dispose()
})

</script>

<template>
  <div class="relative">
    <canvas class="absolute z-10" ref="canvasRef"></canvas>
    <video ref="videoRef" playsinline style="
        -webkit-transform: scaleX(-1);
        transform: scaleX(-1);
        visibility: hidden;
        width: auto;
        height: auto;
      "/>
    
  </div>
</template>


<style scoped>

</style>
