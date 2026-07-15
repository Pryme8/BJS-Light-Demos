import {
  addToScene,
  attachControl,
  createArcRotateCamera,
  createBox,
  createHemisphericLight,
  createPbrMaterial,
} from "@babylonjs/lite";
import type { EngineContext, SceneContext } from "@babylonjs/lite";

export function buildScene(
  engine: EngineContext,
  scene: SceneContext,
  canvas: HTMLCanvasElement
) {
  // Camera — arc-rotate orbiting the origin
  const camera = createArcRotateCamera(-Math.PI * 0.5, Math.PI * 0.4, 5, {
    x: 0,
    y: 0,
    z: 0,
  });
  scene.camera = camera;
  const detachControl = attachControl(camera, canvas, scene);

  // Light — soft hemispheric from above
  addToScene(scene, createHemisphericLight([0, 1, 0], 1.0));

  // Cube with a PBR material
  const cube = createBox(engine);
  cube.material = createPbrMaterial({
    baseColorFactor: [0.05, 0.7, 0.9, 1],
    metallicFactor: 0.3,
    roughnessFactor: 0.4,
  });
  addToScene(scene, cube);

  return detachControl;
}
