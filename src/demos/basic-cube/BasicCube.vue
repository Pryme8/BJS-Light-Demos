<script setup lang="ts">
import { ref } from "vue";
import DemoViewport from "@/components/DemoViewport.vue";
import { useLiteScene } from "@/composables/useLiteScene";
import { buildScene } from "./scene";
import source from "./scene.ts?raw";

const canvasRef = ref<HTMLCanvasElement | null>(null);

useLiteScene(canvasRef, buildScene);
</script>

<template>
  <DemoViewport
    v-model:canvas="canvasRef"
    title="Basic Cube"
    description="Box mesh · hemispheric light · arc-rotate camera"
    :source="source"
  />
</template>
