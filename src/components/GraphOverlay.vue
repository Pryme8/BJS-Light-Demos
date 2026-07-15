<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount } from "vue";
import { TimeSeriesGraph } from "@/lib/graph";

const props = defineProps<{
  getSeries: () => number[];
  labels: string[];
  colors?: string[];
  sampleEvery?: number;
}>();

const canvasRef = ref<HTMLCanvasElement | null>(null);
let graph: TimeSeriesGraph | null = null;
let rafId = 0;
let frameCount = 0;
const SampleEvery = props.sampleEvery ?? 3;

function loop() {
  const canvas = canvasRef.value;
  if (!canvas || !graph) { rafId = requestAnimationFrame(loop); return; }
  const ctx = canvas.getContext("2d");
  if (!ctx) { rafId = requestAnimationFrame(loop); return; }

  frameCount++;
  if (frameCount % SampleEvery === 0) {
    graph.push(props.getSeries());
  }
  graph.draw(ctx);
  rafId = requestAnimationFrame(loop);
}

onMounted(() => {
  const series = props.labels.map((label, i) => ({
    label,
    color: props.colors?.[i],
  }));
  graph = new TimeSeriesGraph(series, 300);
  rafId = requestAnimationFrame(loop);
});

onBeforeUnmount(() => {
  cancelAnimationFrame(rafId);
});
</script>

<template>
  <canvas
    ref="canvasRef"
    class="graph-overlay"
    width="220"
    height="120"
  />
</template>

<style scoped>
.graph-overlay {
  position: absolute;
  right: 12px;
  bottom: 12px;
  border-radius: var(--radius-md);
  pointer-events: none;
}
</style>
