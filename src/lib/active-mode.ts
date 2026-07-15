import { reactive } from "vue";

/**
 * Live CPU/GPU mode for whichever demo is currently mounted.
 *
 * Only one demo scene is ever mounted at a time (the one matching the current
 * route), so a single flat singleton is enough — no per-demo-id map needed.
 * `useSimulation` updates `gpuMode` from that demo's `params.gpuMode` (when
 * present) right after building the handle and on every frame; the sidebar
 * reads it to light the GPU/CPU badge that matches the demo's *actual*
 * current mode, rather than always showing GPU as available.
 */
export const activeMode = reactive({
  gpuMode: false,
});
