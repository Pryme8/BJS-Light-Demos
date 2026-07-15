import { reactive, ref } from "vue";
import {
  addToScene,
  attachControl,
  createArcRotateCamera,
  createHemisphericLight,
  createDirectionalLight,
  createPlane,
  createStandardMaterial,
  createTexture2DFromPixels,
  updateTexture2DFromPixels,
} from "@babylonjs/lite";
import type { EngineContext, SceneContext } from "@babylonjs/lite";
import type { SimHandle } from "@/types/sim";

/** Reaction-Diffusion + Erosion:
 *  Gray-Scott grows a chemical V-pattern, whose concentration is mapped to terrain height.
 *  Thermal erosion is then applied to the height field, wearing down peaks and filling valleys.
 *  The result is visualized as a colorized texture on a plane:
 *    - V-concentration → base color (cyan/dark gradient)
 *    - Height (post-erosion) → brightness/tint
 *  You can toggle between pure-grow, pure-erode, or combined modes.
 */

const N = 192;
const PixelCount = N * N;

export function buildSimScene(
  engine: EngineContext,
  scene: SceneContext,
  canvas: HTMLCanvasElement
): SimHandle {
  const camera = createArcRotateCamera(-Math.PI * 0.4, Math.PI * 0.28, 45, { x: 0, y: 0, z: 0 });
  scene.camera = camera;
  const detach = attachControl(camera, canvas, scene);

  addToScene(scene, createHemisphericLight([0, 1, 0], 0.6));
  addToScene(scene, createDirectionalLight([-0.5, -1, 0.4]));

  const params = reactive({
    rdFeed: 0.055,
    rdKill: 0.062,
    rdSteps: 6,
    erosionRate: 0.025,
    erosionSteps: 3,
    phase: "both",
  });

  const stepsRef = ref(0);

  // Gray-Scott grids
  let u = new Float32Array(PixelCount);
  let v = new Float32Array(PixelCount);
  let nu = new Float32Array(PixelCount);
  let nv = new Float32Array(PixelCount);

  // Height derived from v, further modified by erosion
  const height = new Float32Array(PixelCount);
  const pixels = new Uint8Array(PixelCount * 4);

  const tex = createTexture2DFromPixels(engine, pixels, N, N, { format: "rgba8unorm", generateMipmaps: false });
  const plane = createPlane(engine, { width: 38, height: 38 });
  const mat = createStandardMaterial();
  mat.diffuseTexture = tex;
  plane.material = mat;
  addToScene(scene, plane);

  function seedRD() {
    u.fill(1); v.fill(0);
    const cx = N >> 1;
    for (let dy = -10; dy <= 10; dy++) {
      for (let dx = -10; dx <= 10; dx++) {
        const i = (cx + dx) + (cx + dy) * N;
        if (i >= 0 && i < PixelCount) { u[i] = 0.5; v[i] = 0.25; }
      }
    }
    height.fill(0);
    stepsRef.value = 0;
  }
  seedRD();

  function rdStep() {
    const f = params.rdFeed, k = params.rdKill;
    const Du = 0.18, Dv = 0.09;
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const i = x + y * N;
        const l = (x > 0 ? x - 1 : N - 1) + y * N;
        const r = (x < N - 1 ? x + 1 : 0) + y * N;
        const up = x + (y > 0 ? y - 1 : N - 1) * N;
        const dn = x + (y < N - 1 ? y + 1 : 0) * N;
        const ui = u[i], vi = v[i];
        const lapU = u[l] + u[r] + u[up] + u[dn] - 4 * ui;
        const lapV = v[l] + v[r] + v[up] + v[dn] - 4 * vi;
        const uvv = ui * vi * vi;
        nu[i] = Math.max(0, Math.min(1, ui + Du * lapU - uvv + f * (1 - ui)));
        nv[i] = Math.max(0, Math.min(1, vi + Dv * lapV + uvv - (f + k) * vi));
      }
    }
    const tmp = u; u = nu; nu = tmp;
    const tv = v; v = nv; nv = tv;
  }

  function thermalErosion() {
    const rate = params.erosionRate;
    const talus = 0.35 / N;
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const i = x + y * N;
        const hi = height[i];
        const ns = [
          x > 0 ? x - 1 + y * N : -1,
          x < N - 1 ? x + 1 + y * N : -1,
          y > 0 ? x + (y - 1) * N : -1,
          y < N - 1 ? x + (y + 1) * N : -1,
        ] as const;
        for (const ni of ns) {
          if (ni < 0) continue;
          const diff = hi - height[ni];
          if (diff > talus) {
            const move = rate * (diff - talus);
            height[i] -= move;
            height[ni] += move;
          }
        }
      }
    }
  }

  function writeTexture() {
    for (let i = 0; i < PixelCount; i++) {
      const vi = v[i];
      const h = height[i];
      // RD pattern → base: dark navy → cyan
      const t = Math.min(vi * 3.5, 1);
      const rBase = t * t * 15;
      const gBase = vi * 160 + t * 70;
      const bBase = vi * 220 + 30;
      // Height adds warm highlights (sand/rock color)
      const hBright = h * 180;
      const r = Math.round(Math.min(255, rBase + hBright * 0.9)) & 0xff;
      const g = Math.round(Math.min(255, gBase + hBright * 0.6)) & 0xff;
      const b = Math.round(Math.min(255, bBase + hBright * 0.2)) & 0xff;
      const o = i * 4;
      pixels[o] = r; pixels[o + 1] = g; pixels[o + 2] = b; pixels[o + 3] = 255;
    }
    updateTexture2DFromPixels(engine, tex, pixels);
  }

  function update(_dt: number) {
    const phase = params.phase as string;
    if (phase === "grow" || phase === "both") {
      for (let i = 0; i < params.rdSteps; i++) rdStep();
      // Transfer V to height (smoothly blend in)
      for (let i = 0; i < PixelCount; i++) height[i] = height[i] * 0.95 + v[i] * 0.05;
    }
    if (phase === "erode" || phase === "both") {
      for (let i = 0; i < params.erosionSteps; i++) thermalErosion();
    }
    writeTexture();
    stepsRef.value++;
  }

  function reset() { seedRD(); }

  return {
    params,
    schema: [
      { type: "select", key: "phase", label: "Phase", options: ["grow", "erode", "both"] },
      { type: "slider", key: "rdFeed", label: "Feed Rate (f)", min: 0.01, max: 0.1, step: 0.001 },
      { type: "slider", key: "rdKill", label: "Kill Rate (k)", min: 0.04, max: 0.075, step: 0.0005 },
      { type: "slider", key: "rdSteps", label: "RD Steps/Frame", min: 1, max: 12, step: 1 },
      { type: "slider", key: "erosionRate", label: "Erosion Rate", min: 0.001, max: 0.06, step: 0.001 },
      { type: "slider", key: "erosionSteps", label: "Erosion Steps", min: 1, max: 10, step: 1 },
      { type: "button", key: "_reseed", label: "Reseed", action: reset },
    ],
    readouts: { steps: stepsRef },
    update,
    reset,
    detach,
  };
}
