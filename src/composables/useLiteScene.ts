import { onBeforeUnmount, onMounted, type Ref } from "vue";
import {
  createEngine,
  createSceneContext,
  disposeEngine,
  disposeScene,
  registerScene,
  resizeEngine,
  startEngine,
  stopEngine,
} from "@babylonjs/lite";
import type { EngineContext, SceneContext } from "@babylonjs/lite";

/**
 * A scene builder receives engine + scene + canvas.
 * It may optionally return a detach function (e.g. the return value of
 * attachControl / attachFreeControl) which will be called on unmount.
 */
export type BuildSceneFn = (
  engine: EngineContext,
  scene: SceneContext,
  canvas: HTMLCanvasElement
) => (() => void) | void | Promise<(() => void) | void>;

export function useLiteScene(
  canvasRef: Ref<HTMLCanvasElement | null>,
  buildScene: BuildSceneFn
) {
  let engine: EngineContext | null = null;
  let scene: SceneContext | null = null;
  let detachControl: (() => void) | null = null;
  let resizeObserver: ResizeObserver | null = null;

  function cleanup() {
    resizeObserver?.disconnect();
    resizeObserver = null;
    detachControl?.();
    detachControl = null;
    if (scene) { disposeScene(scene); scene = null; }
    if (engine) { stopEngine(engine); disposeEngine(engine); engine = null; }
  }

  onMounted(async () => {
    const canvas = canvasRef.value;
    if (!canvas) return;

    if (!navigator.gpu) {
      canvas.setAttribute("data-webgpu-unsupported", "true");
      return;
    }

    try {
      engine = await createEngine(canvas);
      scene = createSceneContext(engine);

      const result = await buildScene(engine, scene, canvas);
      if (typeof result === "function") detachControl = result;

      await registerScene(scene);
      await startEngine(engine);

      resizeObserver = new ResizeObserver(() => {
        if (engine) resizeEngine(engine);
      });
      resizeObserver.observe(canvas);
    } catch (err) {
      console.error("[useLiteScene] initialization failed:", err);
      cleanup();
    }
  });

  onBeforeUnmount(cleanup);
}
