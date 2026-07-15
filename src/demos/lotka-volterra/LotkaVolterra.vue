<script setup lang="ts">
import { ref } from "vue";
import DemoViewport from "@/components/DemoViewport.vue";
import ControlPanel from "@/components/ControlPanel.vue";
import GraphOverlay from "@/components/GraphOverlay.vue";
import { useSimulation } from "@/composables/useSimulation";
import { buildSimScene } from "./scene";
import source from "./scene.ts?raw";

const canvasRef = ref<HTMLCanvasElement | null>(null);
const sim = useSimulation(canvasRef, buildSimScene);
</script>

<template>
  <DemoViewport v-model:canvas="canvasRef" title="Lotka–Volterra" description="Predator-prey population dynamics · RK4 integration" :source="source">
    <template #overlay>
      <ControlPanel v-if="sim.handle.value" :schema="sim.handle.value.schema" :params="sim.handle.value.params" :readouts="sim.handle.value.readouts" :fps="sim.fps.value" :running="sim.running.value" @play="sim.play()" @pause="sim.pause()" @step="sim.step()" @reset="sim.reset()" />
      <GraphOverlay v-if="sim.handle.value?.getSeries" :get-series="sim.handle.value.getSeries" :labels="sim.handle.value.seriesLabels ?? []" :colors="sim.handle.value.seriesColors" />
    </template>
  </DemoViewport>
</template>
