import { reactive, ref } from "vue";
import {
  addToScene,
  attachControl,
  createArcRotateCamera,
  createHemisphericLight,
  createPlane,
  createStandardMaterial,
  createTexture2DFromPixels,
  rebuildMaterial,
  updateTexture2DFromPixels,
} from "@babylonjs/lite";
import type { EngineContext, SceneContext } from "@babylonjs/lite";
import type { SimHandle, SliderParam } from "@/types/sim";
import { ReactionDiffusionGpu, GpuMaxGrid } from "./gpu";

const GridN = 256;
const PixelCount = GridN * GridN;

export function buildSimScene(
  engine: EngineContext,
  scene: SceneContext,
  canvas: HTMLCanvasElement
): SimHandle {
  const camera = createArcRotateCamera(-Math.PI / 2, Math.PI / 2, 38, { x: 0, y: 0, z: 0 });
  scene.camera = camera;
  const detach = attachControl(camera, canvas, scene);

  addToScene(scene, createHemisphericLight([0, 1, 0], 1));

  const params = reactive({
    feed:     0.055,
    kill:     0.062,
    Du:       1.0,
    Dv:       0.5,
    steps:    8,
    preset:   "coral",
    gpuMode:  false,
    gridSize: 512,
  });

  const stepsTaken = ref(0);
  const gridRef    = ref<string>(`${GridN}x${GridN}`);

  // Reactive grid-size slider — only meaningful in GPU mode.
  const gridSlider = reactive<SliderParam>({
    type:  "slider",
    key:   "gridSize",
    label: "Grid Size (GPU)",
    min:   256,
    max:   GpuMaxGrid,
    step:  64,
  });

  const Presets: Record<string, { feed: number; kill: number }> = {
    coral:   { feed: 0.0545, kill: 0.062 },
    spots:   { feed: 0.035,  kill: 0.065 },
    stripes: { feed: 0.060,  kill: 0.062 },
    waves:   { feed: 0.025,  kill: 0.051 },
    maze:    { feed: 0.029,  kill: 0.057 },
  };

  // ── CPU path state ────────────────────────────────────────────────────────
  let u = new Float32Array(PixelCount);
  let v = new Float32Array(PixelCount);
  let nu = new Float32Array(PixelCount);
  let nv = new Float32Array(PixelCount);
  const pixels = new Uint8Array(PixelCount * 4);

  function cpuSeedCenter() {
    u.fill(1);
    v.fill(0);
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
  cpuSeedCenter();

  const cpuTex = createTexture2DFromPixels(engine, pixels, GridN, GridN, {
    format: "rgba8unorm",
    generateMipmaps: false,
  });

  const plane = createPlane(engine, { width: 36, height: 36 });
  const mat   = createStandardMaterial();
  mat.emissiveTexture = cpuTex;
  mat.backFaceCulling = false;
  plane.material = mat;
  addToScene(scene, plane);

  function cpuGrayScottStep() {
    const f = params.feed, k = params.kill;
    const Du = params.Du * 0.2;
    const Dv = params.Dv * 0.2;
    for (let y = 0; y < GridN; y++) {
      for (let x = 0; x < GridN; x++) {
        const i     = x + y * GridN;
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
    const tmp = u; u = nu; nu = tmp;
    const tmpV = v; v = nv; nv = tmpV;
  }

  function cpuWriteTexture() {
    for (let i = 0; i < PixelCount; i++) {
      const vi = v[i];
      const t  = Math.min(vi * 4, 1);
      const r  = Math.round(t * t * 20 + vi * 30) & 0xff;
      const g  = Math.round(vi * 180 + t * 60) & 0xff;
      const b  = Math.round(vi * 255 + (1 - vi) * 40) & 0xff;
      const o  = i * 4;
      pixels[o] = r; pixels[o+1] = g; pixels[o+2] = b; pixels[o+3] = 255;
    }
    updateTexture2DFromPixels(engine, cpuTex, pixels);
  }

  // ── GPU path ──────────────────────────────────────────────────────────────
  const gpu          = new ReactionDiffusionGpu(engine);
  const gpuAvailable = gpu.ok;

  function switchToGpu(): void {
    if (!gpuAvailable) { params.gpuMode = false; return; }
    const t = gpu.setGrid(params.gridSize);
    mat.emissiveTexture = t;
    rebuildMaterial(scene, mat, { rebuildFrameGraph: true });
    gridRef.value = `${params.gridSize}x${params.gridSize}`;
  }

  function switchToCpu(): void {
    mat.emissiveTexture = cpuTex;
    rebuildMaterial(scene, mat, { rebuildFrameGraph: true });
    gridRef.value = `${GridN}x${GridN}`;
  }

  let prevGpu     = false;
  let prevGrid    = params.gridSize;
  let prevSteps   = params.steps;
  void prevSteps;

  function applyPreset() {
    const p = Presets[params.preset as string] ?? Presets.coral;
    params.feed = p.feed;
    params.kill = p.kill;
  }

  function update(dt: number) {
    void dt;
    const isGpu = params.gpuMode;

    if (isGpu !== prevGpu) {
      prevGpu = isGpu;
      if (isGpu) switchToGpu(); else switchToCpu();
    }

    if (!isGpu) {
      // ── CPU path ────────────────────────────────────────────────────────
      const steps = params.steps;
      for (let s = 0; s < steps; s++) cpuGrayScottStep();
      stepsTaken.value += steps;
      cpuWriteTexture();
    } else {
      // ── GPU path ─────────────────────────────────────────────────────────
      if (params.gridSize !== prevGrid) {
        prevGrid = params.gridSize;
        const t = gpu.setGrid(params.gridSize);
        mat.emissiveTexture = t;
        rebuildMaterial(scene, mat, { rebuildFrameGraph: true });
        gridRef.value = `${params.gridSize}x${params.gridSize}`;
      }
      gpu.dispatch(params.steps, {
        feed: params.feed,
        kill: params.kill,
        Du:   params.Du,
        Dv:   params.Dv,
      });
      stepsTaken.value += params.steps;
    }
  }

  function reset() {
    stepsTaken.value = 0;
    if (params.gpuMode && gpuAvailable) {
      gpu.seedCenter();
    } else {
      cpuSeedCenter();
    }
  }

  return {
    params,
    schema: [
      {
        type:  "toggle",
        key:   "gpuMode",
        label: gpuAvailable ? "GPU Mode" : "GPU Mode (unavailable)",
      },
      gridSlider,
      { type: "select", key: "preset",  label: "Preset",        options: Object.keys(Presets) },
      { type: "button", key: "_preset", label: "Apply Preset",  action: applyPreset },
      { type: "slider", key: "feed",    label: "Feed Rate (f)", min: 0.01,  max: 0.1,   step: 0.001 },
      { type: "slider", key: "kill",    label: "Kill Rate (k)", min: 0.04,  max: 0.07,  step: 0.0005 },
      { type: "slider", key: "Du",      label: "Diffusion U",   min: 0.1,   max: 2,     step: 0.05 },
      { type: "slider", key: "Dv",      label: "Diffusion V",   min: 0.05,  max: 1,     step: 0.025 },
      { type: "slider", key: "steps",   label: "Steps/Frame",   min: 1,     max: 20,    step: 1 },
      { type: "button", key: "_reset",  label: "Reseed Center", action: () => reset() },
    ],
    readouts: { steps: stepsTaken, grid: gridRef },
    update,
    reset,
    detach,
  };
}
