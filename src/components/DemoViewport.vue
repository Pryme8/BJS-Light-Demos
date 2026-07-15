<script setup lang="ts">
import { ref } from "vue";
import type { ComponentPublicInstance } from "vue";
import CodeViewer from "@/components/CodeViewer.vue";

defineProps<{
  source: string;
  title: string;
  description?: string;
}>();

const canvasRef = defineModel<HTMLCanvasElement | null>("canvas");
const showSource = ref(false);

function onCanvasMount(el: Element | ComponentPublicInstance | null) {
  if (el instanceof HTMLCanvasElement) {
    canvasRef.value = el;
  }
}
</script>

<template>
  <div class="viewport">
    <header class="viewport__header">
      <div class="viewport__title-group">
        <span class="viewport__badge font-mono">demo</span>
        <h1 class="viewport__title">{{ title }}</h1>
        <p v-if="description" class="viewport__description">{{ description }}</p>
      </div>
      <button
        class="viewport__source-btn font-mono"
        :class="{ 'viewport__source-btn--active': showSource }"
        @click="showSource = !showSource"
      >
        {{ showSource ? "hide source" : "view source" }}
      </button>
    </header>

    <div class="viewport__body">
      <div class="viewport__canvas-wrap">
        <canvas :ref="onCanvasMount" class="viewport__canvas" />
        <div class="viewport__webgpu-warn" v-if="false">
          WebGPU is not supported in this browser.
          Use Chrome or Edge 113+.
        </div>
        <!-- Overlay slot: ControlPanel, GraphOverlay, etc. -->
        <slot name="overlay" />
      </div>

      <transition name="slide">
        <CodeViewer v-if="showSource" :source="source" class="viewport__code" />
      </transition>
    </div>
  </div>
</template>

<style scoped>
.viewport {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--bg-void);
}

.viewport__header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 20px;
  height: var(--header-h);
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border-subtle);
  flex-shrink: 0;
}

.viewport__title-group {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: 1;
  overflow: hidden;
}

.viewport__badge {
  font-size: 9px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--cyan);
  border: 1px solid var(--cyan-dim);
  border-radius: var(--radius-sm);
  padding: 2px 6px;
  white-space: nowrap;
  flex-shrink: 0;
}

.viewport__title {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
  letter-spacing: 0.01em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.viewport__description {
  font-size: 12px;
  color: var(--text-muted);
  white-space: normal;
  line-height: 1.4;
}

.viewport__source-btn {
  font-size: 11px;
  letter-spacing: 0.08em;
  padding: 5px 12px;
  background: transparent;
  color: var(--text-secondary);
  border: 1px solid var(--border-dim);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s, box-shadow 0.15s;
  white-space: nowrap;
  flex-shrink: 0;
}

.viewport__source-btn:hover,
.viewport__source-btn--active {
  color: var(--cyan);
  border-color: var(--cyan-dim);
  box-shadow: 0 0 6px rgba(0, 229, 255, 0.2);
}

.viewport__body {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.viewport__canvas-wrap {
  flex: 1;
  position: relative;
  overflow: hidden;
}

.viewport__canvas {
  display: block;
  width: 100%;
  height: 100%;
  box-shadow: inset 0 0 0 1px var(--border-subtle);
}

.viewport__webgpu-warn {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-mono);
  font-size: 13px;
  color: var(--text-muted);
  background: var(--bg-base);
}

.viewport__code {
  width: 42%;
  min-width: 320px;
  max-width: 640px;
  flex-shrink: 0;
}

/* slide transition */
.slide-enter-active,
.slide-leave-active {
  transition: width 0.2s ease, opacity 0.2s ease;
  overflow: hidden;
}
.slide-enter-from,
.slide-leave-to {
  width: 0 !important;
  opacity: 0;
}
</style>
