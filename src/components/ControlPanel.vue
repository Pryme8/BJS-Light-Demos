<script setup lang="ts">
import { ref, reactive } from "vue";
import type { ParamSchema } from "@/types/sim";
import type { Ref } from "vue";
import RangeSlider from "@/components/RangeSlider.vue";

const props = defineProps<{
  running: boolean;
  fps: number;
  schema: ParamSchema[];
  params: Record<string, unknown>;
  readouts?: Record<string, Ref<number | string>>;
}>();

const emit = defineEmits<{
  play: [];
  pause: [];
  step: [];
  reset: [];
}>();

const collapsed = ref(false);

// Track collapsed state per section key.
const sectionCollapsed = reactive<Record<string, boolean>>({});
function isSectionCollapsed(key: string) { return sectionCollapsed[key] ?? false; }
function toggleSection(key: string) { sectionCollapsed[key] = !isSectionCollapsed(key); }

// Compute which section (if any) each item belongs to, so items inside a
// collapsed section can be hidden.
function isVisible(idx: number): boolean {
  for (let i = idx - 1; i >= 0; i--) {
    if (props.schema[i].type === "section") {
      return !isSectionCollapsed(props.schema[i].key);
    }
  }
  return true; // no parent section → always visible
}

// Sliders commit to params only on release. While dragging, the value lives in
// dragValues (local) so the thumb + number stay live without re-running the sim
// every frame; on `change` (pointer release / keyboard commit) it is written to
// params.
const dragValues = reactive<Record<string, number>>({});

function getSliderValue(key: string): number {
  return (props.params[key] ?? 0) as number;
}
/** Display value: the in-progress drag value if dragging, else the committed param. */
function getSliderDisplay(key: string): number {
  return dragValues[key] ?? getSliderValue(key);
}
/** During drag: update the local value only (does not touch params). */
function onSliderInput(key: string, v: string): void {
  dragValues[key] = parseFloat(v);
}
/** On release: commit the dragged value to params, then drop the local override. */
function onSliderChange(key: string, v: string): void {
  (props.params as Record<string, unknown>)[key] = parseFloat(v);
  delete dragValues[key];
}
function getToggleValue(key: string): boolean {
  return props.params[key] as boolean;
}
function toggleValue(key: string): void {
  (props.params as Record<string, unknown>)[key] = !props.params[key];
}
function getSelectValue(key: string): string {
  return props.params[key] as string;
}
function setSelectValue(key: string, v: string): void {
  (props.params as Record<string, unknown>)[key] = v;
}
function getRangeVal(key: string): number {
  return (props.params[key] ?? 0) as number;
}
function setRangeVal(key: string, v: number): void {
  (props.params as Record<string, unknown>)[key] = v;
}
</script>

<template>
  <div class="cp" :class="{ 'cp--collapsed': collapsed }">
    <!-- Header row: transport + FPS + collapse toggle -->
    <div class="cp__header">
      <div class="cp__transport">
        <button v-if="!running" class="cp__btn cp__btn--play" @click="emit('play')" title="Play">▶</button>
        <button v-else class="cp__btn cp__btn--pause" @click="emit('pause')" title="Pause">⏸</button>
        <button class="cp__btn" @click="emit('step')" title="Step one frame">⏭</button>
        <button class="cp__btn" @click="emit('reset')" title="Reset">↺</button>
      </div>
      <span class="cp__fps font-mono">{{ fps }} fps</span>
      <button class="cp__collapse font-mono" @click="collapsed = !collapsed">
        {{ collapsed ? "▸" : "▾" }}
      </button>
    </div>

    <!-- Body: params + readouts -->
    <div v-show="!collapsed" class="cp__body">
      <!-- Readouts -->
      <div v-if="readouts && Object.keys(readouts).length" class="cp__readouts">
        <div v-for="(val, key) in readouts" :key="key" class="cp__readout font-mono">
          <span class="cp__readout-key">{{ key }}</span>
          <span class="cp__readout-val">{{ val.value }}</span>
        </div>
      </div>

      <!-- Params by schema — items inside a collapsed section are hidden -->
      <template v-for="(def, idx) in schema" :key="def.type === 'range' ? def.keyLo : def.key">

        <!-- Section header (collapsible sub-panel) -->
        <template v-if="def.type === 'section'">
          <button class="cp__section" @click="toggleSection(def.key)">
            <span class="cp__section-arrow">{{ isSectionCollapsed(def.key) ? '▸' : '▾' }}</span>
            <span class="cp__section-label font-mono">{{ def.label }}</span>
          </button>
        </template>

        <!-- Everything else — hidden when inside a collapsed section -->
        <div v-else-if="isVisible(idx)" class="cp__param">

          <!-- Slider (commits on release; live-updates the display while dragging) -->
          <template v-if="def.type === 'slider'">
            <div class="cp__param-row">
              <label class="cp__label font-mono">{{ def.label }}</label>
              <span class="cp__val font-mono">{{ (getSliderDisplay(def.key) ?? 0).toFixed(def.step < 1 ? 2 : 0) }}</span>
            </div>
            <input
              type="range"
              class="cp__slider"
              :min="def.min"
              :max="def.max"
              :step="def.step"
              :value="getSliderDisplay(def.key)"
              @input="onSliderInput(def.key, ($event.target as HTMLInputElement).value)"
              @change="onSliderChange(def.key, ($event.target as HTMLInputElement).value)"
            />
          </template>

          <!-- Dual-thumb range -->
          <template v-else-if="def.type === 'range'">
            <RangeSlider
              :label="def.label"
              :min="def.min"
              :max="def.max"
              :step="def.step"
              :lo="getRangeVal(def.keyLo)"
              :hi="getRangeVal(def.keyHi)"
              @update:lo="setRangeVal(def.keyLo, $event)"
              @update:hi="setRangeVal(def.keyHi, $event)"
            />
          </template>

          <!-- Toggle -->
          <template v-else-if="def.type === 'toggle'">
            <button
              class="cp__toggle"
              :class="{ 'cp__toggle--on': getToggleValue(def.key) }"
              @click="toggleValue(def.key)"
            >
              <span class="cp__toggle-dot" />
              <span class="cp__label font-mono">{{ def.label }}</span>
            </button>
          </template>

          <!-- Select -->
          <template v-else-if="def.type === 'select'">
            <label class="cp__label font-mono">{{ def.label }}</label>
            <select
              class="cp__select font-mono"
              :value="getSelectValue(def.key)"
              @change="setSelectValue(def.key, ($event.target as HTMLSelectElement).value)"
            >
              <option v-for="opt in def.options" :key="opt" :value="opt">{{ opt }}</option>
            </select>
          </template>

          <!-- Button -->
          <template v-else-if="def.type === 'button'">
            <button class="cp__action font-mono" @click="def.action()">{{ def.label }}</button>
          </template>

        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.cp {
  position: absolute;
  left: 12px;
  bottom: 12px;
  width: 220px;
  background: rgba(8, 11, 15, 0.88);
  border: 1px solid var(--border-bright);
  border-radius: var(--radius-md);
  backdrop-filter: blur(8px);
  box-shadow: 0 4px 24px rgba(0,0,0,0.5), var(--glow-canvas);
  z-index: 10;
  user-select: none;
}

.cp__header {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 10px;
  border-bottom: 1px solid var(--border-subtle);
}

.cp__transport {
  display: flex;
  gap: 4px;
  flex: 1;
}

.cp__btn {
  width: 26px;
  height: 26px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-elevated);
  border: 1px solid var(--border-dim);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 12px;
  line-height: 1;
  transition: color 0.12s, border-color 0.12s, box-shadow 0.12s;
}

.cp__btn:hover,
.cp__btn--play,
.cp__btn--pause {
  color: var(--cyan);
  border-color: var(--cyan-dim);
  box-shadow: 0 0 6px rgba(0, 229, 255, 0.2);
}

.cp__fps {
  font-size: 10px;
  color: var(--text-muted);
  letter-spacing: 0.06em;
}

.cp__collapse {
  background: transparent;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 12px;
  padding: 0 2px;
}

.cp__collapse:hover { color: var(--text-primary); }

.cp__body {
  padding: 8px 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 380px;
  overflow-y: auto;
}

/* Readouts */
.cp__readouts {
  display: flex;
  flex-direction: column;
  gap: 2px;
  border-bottom: 1px solid var(--border-subtle);
  padding-bottom: 6px;
}

.cp__readout {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
}

.cp__readout-key { color: var(--text-muted); letter-spacing: 0.06em; }
.cp__readout-val { color: var(--cyan); }

/* Params */
.cp__param {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.cp__param-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.cp__label {
  font-size: 10px;
  color: var(--text-secondary);
  letter-spacing: 0.05em;
}

.cp__val {
  font-size: 10px;
  color: var(--cyan);
}

.cp__slider {
  width: 100%;
  height: 4px;
  appearance: none;
  background: var(--bg-hover);
  border-radius: 2px;
  cursor: pointer;
  outline: none;
}

.cp__slider::-webkit-slider-thumb {
  appearance: none;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--cyan);
  box-shadow: 0 0 4px rgba(0,229,255,0.5);
  cursor: pointer;
}

.cp__toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  background: transparent;
  border: 1px solid var(--border-dim);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  cursor: pointer;
  padding: 4px 8px;
  transition: border-color 0.12s;
}

.cp__toggle:hover { border-color: var(--border-bright); }
.cp__toggle--on { border-color: var(--cyan-dim); color: var(--cyan); }

.cp__toggle-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--text-muted);
  flex-shrink: 0;
  transition: background 0.12s;
}

.cp__toggle--on .cp__toggle-dot {
  background: var(--cyan);
  box-shadow: 0 0 4px rgba(0,229,255,0.6);
}

.cp__select {
  width: 100%;
  background: var(--bg-elevated);
  border: 1px solid var(--border-dim);
  border-radius: var(--radius-sm);
  color: var(--text-primary);
  font-size: 11px;
  padding: 3px 6px;
  cursor: pointer;
  outline: none;
}

.cp__select:focus { border-color: var(--cyan-dim); }

.cp__action {
  width: 100%;
  padding: 5px 10px;
  background: var(--bg-elevated);
  border: 1px solid var(--border-dim);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  font-size: 11px;
  cursor: pointer;
  transition: color 0.12s, border-color 0.12s;
}

.cp__action:hover {
  color: var(--magenta);
  border-color: var(--magenta-dim);
}

/* Section header */
.cp__section {
  display: flex;
  align-items: center;
  gap: 5px;
  width: 100%;
  padding: 5px 0 3px;
  background: transparent;
  border: none;
  border-top: 1px solid var(--border-subtle);
  color: var(--text-muted);
  cursor: pointer;
  text-align: left;
  margin-top: 2px;
}
.cp__section:hover { color: var(--text-secondary); }
.cp__section-arrow { font-size: 10px; flex-shrink: 0; }
.cp__section-label { font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase; }
</style>
