<script setup lang="ts">
import { ref } from "vue";
import DemoViewport from "@/components/DemoViewport.vue";
import ControlPanel from "@/components/ControlPanel.vue";
import { useSimulation } from "@/composables/useSimulation";
import { buildSimScene } from "./scene";
import source from "./scene.ts?raw";

const canvasRef = ref<HTMLCanvasElement | null>(null);
const sim = useSimulation(canvasRef, buildSimScene);
</script>

<template>
  <DemoViewport v-model:canvas="canvasRef" title="Conway's Game of Life" description="Cellular automata · B3/S23 rules · CPU or GPU scalable grid" :source="source">
    <template #overlay>
      <ControlPanel v-if="sim.handle.value" :schema="sim.handle.value.schema" :params="sim.handle.value.params" :readouts="sim.handle.value.readouts" :fps="sim.fps.value" :running="sim.running.value" @play="sim.play()" @pause="sim.pause()" @step="sim.step()" @reset="sim.reset()" />
    </template>
  </DemoViewport>
</template>
