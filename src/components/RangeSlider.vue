<script setup lang="ts">
import { computed } from "vue";

const props = defineProps<{
  min: number;
  max: number;
  step: number;
  lo: number;
  hi: number;
  label: string;
}>();

const emit = defineEmits<{
  "update:lo": [v: number];
  "update:hi": [v: number];
}>();

const span = computed(() => props.max - props.min);
const loFrac = computed(() => (props.lo - props.min) / span.value);
const hiFrac = computed(() => (props.hi - props.min) / span.value);

const fillStyle = computed(() => ({
  left:  `${loFrac.value * 100}%`,
  width: `${(hiFrac.value - loFrac.value) * 100}%`,
}));

function onLo(e: Event) {
  const raw = parseFloat((e.target as HTMLInputElement).value);
  emit("update:lo", Math.min(raw, props.hi - props.step));
}
function onHi(e: Event) {
  const raw = parseFloat((e.target as HTMLInputElement).value);
  emit("update:hi", Math.max(raw, props.lo + props.step));
}
</script>

<template>
  <div class="rs">
    <div class="rs__header">
      <span class="rs__label font-mono">{{ label }}</span>
      <span class="rs__vals font-mono">
        {{ step < 1 ? lo.toFixed(2) : lo.toFixed(0) }}
        <span class="rs__dash">–</span>
        {{ step < 1 ? hi.toFixed(2) : hi.toFixed(0) }}
      </span>
    </div>
    <div class="rs__wrap">
      <div class="rs__track">
        <div class="rs__fill" :style="fillStyle" />
      </div>
      <!-- Lo thumb (z-index below hi, so hi wins overlap near the top) -->
      <input type="range" class="rs__input rs__input--lo"
        :min="min" :max="max" :step="step" :value="lo"
        @input="onLo"
      />
      <!-- Hi thumb -->
      <input type="range" class="rs__input rs__input--hi"
        :min="min" :max="max" :step="step" :value="hi"
        @input="onHi"
      />
    </div>
  </div>
</template>

<style scoped>
.rs { display: flex; flex-direction: column; gap: 3px; }

.rs__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.rs__label { font-size: 10px; color: var(--text-secondary); letter-spacing: 0.05em; }

.rs__vals { font-size: 10px; color: var(--cyan); }
.rs__dash  { color: var(--text-muted); margin: 0 2px; }

.rs__wrap {
  position: relative;
  height: 14px;
}

.rs__track {
  position: absolute;
  left: 0; right: 0;
  top: 50%; transform: translateY(-50%);
  height: 4px;
  background: var(--bg-hover);
  border-radius: 2px;
  pointer-events: none;
}

.rs__fill {
  position: absolute;
  top: 0; height: 100%;
  background: var(--cyan);
  border-radius: 2px;
  box-shadow: 0 0 6px rgba(0,229,255,0.35);
}

.rs__input {
  position: absolute;
  left: 0; right: 0; top: 0;
  width: 100%;
  height: 14px;
  appearance: none;
  background: transparent;
  cursor: pointer;
  outline: none;
  margin: 0;
  padding: 0;
  pointer-events: none;
}

/* Only the thumb captures pointer events */
.rs__input::-webkit-slider-thumb {
  pointer-events: all;
  appearance: none;
  width: 12px; height: 12px;
  border-radius: 50%;
  background: var(--cyan);
  box-shadow: 0 0 4px rgba(0,229,255,0.6);
  cursor: grab;
  transition: transform 0.1s;
}
.rs__input::-webkit-slider-thumb:active { transform: scale(1.25); cursor: grabbing; }

.rs__input--lo { z-index: 1; }
.rs__input--hi { z-index: 2; }
</style>
