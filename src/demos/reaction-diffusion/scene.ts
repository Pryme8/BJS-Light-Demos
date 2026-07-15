import { reactive, ref } from "vue";
import {
  addToScene,
  attachControl,
  createArcRotateCamera,
  createHemisphericLight,
  createPlane,
  createStandardMaterial,
  createTexture2DFromPixels,
  updateTexture2DFromPixels,
} from "@babylonjs/lite";
import type { EngineContext, SceneContext } from "@babylonjs/lite";
import type { SimHandle } from "@/types/sim";

const GridN = 256;
const PixelCount = GridN * GridN;

export function buildSimScene(
  engine: EngineContext,
  scene: SceneContext,
  canvas: HTMLCanvasElement
): SimHandle {
  // createPlane builds a vertical plane in the XY plane (normal -Z), so view it
  // head-on from the -Z axis (alpha -PI/2, beta PI/2).
  const camera = createArcRotateCamera(-Math.PI / 2, Math.PI / 2, 38, { x: 0, y: 0, z: 0 });
  scene.camera = camera;
  const detach = attachControl(camera, canvas, scene);

  addToScene(scene, createHemisphericLight([0, 1, 0], 1));

  const params = reactive({
    feed: 0.055,
    kill: 0.062,
    Du: 1.0,
    Dv: 0.5,
    steps: 8,
    preset: "coral",
  });

  const stepsTaken = ref(0);

  const Presets: Record<string, { feed: number; kill: number }> = {
    coral:  { feed: 0.0545, kill: 0.062 },
    spots:  { feed: 0.035,  kill: 0.065 },
    stripes:{ feed: 0.060,  kill: 0.062 },
    waves:  { feed: 0.025,  kill: 0.051 },
    maze:   { feed: 0.029,  kill: 0.057 },
  };

  // Gray-Scott state: two interleaved Float32 grids
  let u = new Float32Array(PixelCount);
  let v = new Float32Array(PixelCount);
  let nu = new Float32Array(PixelCount);
  let nv = new Float32Array(PixelCount);

  // RGBA pixels for texture
  const pixels = new Uint8Array(PixelCount * 4);

  function seedCenter() {
    u.fill(1);
    v.fill(0);
    // Seed a small square of v in center
    const cx = GridN >> 1;
    const r = 12;
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const i = (cx + dx) + (cx + dy) * GridN;
        u[i] = 0.5; v[i] = 0.25;
      }
    }
    stepsTaken.value = 0;
  }

  seedCenter();

  // Texture — will be updated each frame
  const tex = createTexture2DFromPixels(engine, pixels, GridN, GridN, { format: "rgba8unorm", generateMipmaps: false });

  const plane = createPlane(engine, { width: 36, height: 36 });
  const mat = createStandardMaterial();
  // Self-lit so the pattern shows at full brightness regardless of lighting;
  // double-sided so it's visible from either face.
  mat.emissiveTexture = tex;
  mat.backFaceCulling = false;
  plane.material = mat;
  addToScene(scene, plane);

  function applyPreset() {
    const p = Presets[params.preset as string] ?? Presets.coral;
    params.feed = p.feed;
    params.kill = p.kill;
  }

  function grayScottStep() {
    const f = params.feed, k = params.kill;
    const Du = params.Du * 0.2;
    const Dv = params.Dv * 0.2;

    for (let y = 0; y < GridN; y++) {
      for (let x = 0; x < GridN; x++) {
        const i = x + y * GridN;
        const left  = (x === 0 ? GridN - 1 : x - 1) + y * GridN;
        const right = (x === GridN - 1 ? 0 : x + 1) + y * GridN;
        const up    = x + (y === 0 ? GridN - 1 : y - 1) * GridN;
        const down  = x + (y === GridN - 1 ? 0 : y + 1) * GridN;

        const ui = u[i], vi = v[i];
        const lapU = u[left] + u[right] + u[up] + u[down] - 4 * ui;
        const lapV = v[left] + v[right] + v[up] + v[down] - 4 * vi;
        const uvv = ui * vi * vi;

        nu[i] = Math.max(0, Math.min(1, ui + Du * lapU - uvv + f * (1 - ui)));
        nv[i] = Math.max(0, Math.min(1, vi + Dv * lapV + uvv - (f + k) * vi));
      }
    }
    // Swap buffers
    const tmp = u; u = nu; nu = tmp;
    const tmpV = v; v = nv; nv = tmpV;
  }

  function writeTexture() {
    for (let i = 0; i < PixelCount; i++) {
      const vi = v[i];
      // Map v concentration to color: dark→blue→cyan→white
      const t = Math.min(vi * 4, 1);
      const r = Math.round(t * t * 20 + vi * 30) & 0xff;
      const g = Math.round(vi * 180 + t * 60) & 0xff;
      const b = Math.round(vi * 255 + (1 - vi) * 40) & 0xff;
      const o = i * 4;
      pixels[o] = r; pixels[o + 1] = g; pixels[o + 2] = b; pixels[o + 3] = 255;
    }
    updateTexture2DFromPixels(engine, tex, pixels);
  }

  function update(dt: number) {
    const steps = params.steps;
    for (let s = 0; s < steps; s++) grayScottStep();
    stepsTaken.value += steps;
    writeTexture();
  }

  function reset() {
    seedCenter();
    u.fill(1); v.fill(0);
    seedCenter();
  }

  return {
    params,
    schema: [
      { type: "select", key: "preset", label: "Preset", options: Object.keys(Presets) },
      { type: "button", key: "_preset", label: "Apply Preset", action: applyPreset },
      { type: "slider", key: "feed", label: "Feed Rate (f)", min: 0.01, max: 0.1, step: 0.001 },
      { type: "slider", key: "kill", label: "Kill Rate (k)", min: 0.04, max: 0.07, step: 0.0005 },
      { type: "slider", key: "Du", label: "Diffusion U", min: 0.1, max: 2, step: 0.05 },
      { type: "slider", key: "Dv", label: "Diffusion V", min: 0.05, max: 1, step: 0.025 },
      { type: "slider", key: "steps", label: "Steps/Frame", min: 1, max: 20, step: 1 },
      { type: "button", key: "_reset", label: "Reseed Center", action: () => { seedCenter(); } },
    ],
    readouts: { steps: stepsTaken },
    update,
    reset,
    detach,
  };
}
