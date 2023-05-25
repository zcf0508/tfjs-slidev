import { defineConfig } from 'vite';

export default defineConfig({
  optimizeDeps: {
    include: [
      '@tensorflow/tfjs', 
      '@tensorflow/tfjs-backend-webgl', 
      '@tensorflow/tfjs-backend-wasm',
      '@tensorflow-models/face-landmarks-detection',
    ],
  }
})
