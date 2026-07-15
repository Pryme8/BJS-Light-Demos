import { ref, shallowRef, type Ref } from "vue";
import { onBeforeRender } from "@babylonjs/lite";
import { useLiteScene } from "@/composables/useLiteScene";
import { activeMode } from "@/lib/active-mode";
import type { BuildSimSceneFn, SimControls, SimHandle } from "@/types/sim";

export function useSimulation(
  canvasRef: Ref<HTMLCanvasElement | null>,
  buildSimScene: BuildSimSceneFn
): SimControls {
  const running = ref(true);
  const fps = ref(0);
  const handle = shallowRef<SimHandle | null>(null);

  let fpsFrames = 0;
  let fpsAccum = 0;

  useLiteScene(canvasRef, async (engine, scene, canvas) => {
    const h = await buildSimScene(engine, scene, canvas);
    handle.value = h;

    // Demos that support both a CPU and GPU path expose a boolean
    // `params.gpuMode`; publish it so the sidebar badge reflects the mode
    // this demo is actually running in, not just that GPU is available.
    const gpuModeOf = () => (h.params as { gpuMode?: boolean }).gpuMode;
    if (typeof gpuModeOf() === "boolean") activeMode.gpuMode = gpuModeOf()!;

    onBeforeRender(scene, (dt) => {
      fpsAccum += dt;
      fpsFrames++;
      if (fpsAccum >= 500) {
        fps.value = Math.round((fpsFrames * 1000) * 0.001 / (fpsAccum * 0.001));
        fpsFrames = 0;
        fpsAccum = 0;
      }
      const g = gpuModeOf();
      if (typeof g === "boolean") activeMode.gpuMode = g;
      if (running.value) h.update(dt);
    });

    return h.detach;
  });

  function play() { running.value = true; }
  function pause() { running.value = false; }

  function step() {
    if (!running.value && handle.value) {
      handle.value.update(16.667);
    }
  }

  function reset() {
    handle.value?.reset();
  }

  return { running, fps, handle, play, pause, step, reset };
}
