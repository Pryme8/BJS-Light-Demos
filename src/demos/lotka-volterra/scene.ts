import { reactive, ref } from "vue";
import {
  addToScene,
  attachControl,
  createArcRotateCamera,
  createSphere,
  createHemisphericLight,
  createPbrMaterial,
  invalidateRenderBundles,
  setMeshVisible,
  setThinInstanceCount,
} from "@babylonjs/lite";
import type { EngineContext, SceneContext } from "@babylonjs/lite";
import { AgentBuffer } from "@/lib/agents";
import type { SimHandle } from "@/types/sim";
import { LotkaVolterraGpu, GpuMaxPrey, GpuMaxPred, GpuBound } from "./gpu";

// ── CPU-path constants ────────────────────────────────────────────────────

const MaxPrey = 800;
const MaxPred = 200;
const Bound   = 18;

// ── GPU-path scaling ──────────────────────────────────────────────────────
// Agents-per-ODE-population-unit on the GPU path.
// At peak (prey ~80): 80 × 400 = 32 000 prey; pred peaks ~9 × 400 = 3 600.
const GpuDensity = 400;

export function buildSimScene(
  engine: EngineContext,
  scene: SceneContext,
  canvas: HTMLCanvasElement
): SimHandle {
  const camera = createArcRotateCamera(-Math.PI * 0.5, Math.PI * 0.3, 45, { x: 0, y: 0, z: 0 });
  scene.camera = camera;
  const detach = attachControl(camera, canvas, scene);

  addToScene(scene, createHemisphericLight([0, 1, 0], 0.8));

  // ── CPU path: prey and predator meshes ───────────────────────────────────
  const preyMesh = createSphere(engine, { segments: 6, diameter: 0.6 });
  preyMesh.material = createPbrMaterial({ baseColorFactor: [0.05, 0.85, 0.9, 1], metallicFactor: 0.1, roughnessFactor: 0.6 });
  addToScene(scene, preyMesh);

  const predMesh = createSphere(engine, { segments: 6, diameter: 1.0 });
  predMesh.material = createPbrMaterial({ baseColorFactor: [1, 0.05, 0.7, 1], metallicFactor: 0.3, roughnessFactor: 0.4 });
  addToScene(scene, predMesh);

  const params = reactive({
    alpha:     1.0,
    beta:      0.15,
    delta:     0.1,
    gamma:     0.8,
    speed:     0.5,
    preySpeed: 4.0,
    predSpeed: 3.0,
    gpuMode:   false,
  });

  const preyCountRef = ref(0);
  const predCountRef = ref(0);

  // ODE state (continuous populations).
  let preyPop = 40;
  let predPop = 9;

  // CPU visual agent state.
  let preyN = 0;
  let predN = 0;
  const preyPx = new Float32Array(MaxPrey); const preyPy = new Float32Array(MaxPrey); const preyPz = new Float32Array(MaxPrey);
  const preyVx = new Float32Array(MaxPrey); const preyVy = new Float32Array(MaxPrey); const preyVz = new Float32Array(MaxPrey);
  const predPx = new Float32Array(MaxPred); const predPy = new Float32Array(MaxPred); const predPz = new Float32Array(MaxPred);
  const predVx = new Float32Array(MaxPred); const predVy = new Float32Array(MaxPred); const predVz = new Float32Array(MaxPred);

  const preyBuf = new AgentBuffer(MaxPrey);
  const predBuf = new AgentBuffer(MaxPred);
  preyBuf.attach(engine, preyMesh, false);
  predBuf.attach(engine, predMesh, false);

  function spawnAgent(
    px: Float32Array, py: Float32Array, pz: Float32Array,
    vx: Float32Array, vy: Float32Array, vz: Float32Array,
    i: number
  ) {
    px[i] = (Math.random() - 0.5) * Bound * 2;
    py[i] = (Math.random() - 0.5) * 2;
    pz[i] = (Math.random() - 0.5) * Bound * 2;
    const a = Math.random() * Math.PI * 2;
    vx[i] = Math.cos(a); vy[i] = 0; vz[i] = Math.sin(a);
  }

  function resetPops() {
    preyPop = 40; predPop = 9;
    preyN = Math.min(40, MaxPrey);
    predN = Math.min(9, MaxPred);
    for (let i = 0; i < MaxPrey; i++) spawnAgent(preyPx, preyPy, preyPz, preyVx, preyVy, preyVz, i);
    for (let i = 0; i < MaxPred; i++) spawnAgent(predPx, predPy, predPz, predVx, predVy, predVz, i);
  }
  resetPops();

  function moveAgents(
    n: number, spd: number, s: number,
    px: Float32Array, py: Float32Array, pz: Float32Array,
    vx: Float32Array, vy: Float32Array, vz: Float32Array
  ) {
    void py; void vy;   // y is unchanged; suppress unused-var warnings
    for (let i = 0; i < n; i++) {
      vx[i] += (Math.random() - 0.5) * 0.4;
      vz[i] += (Math.random() - 0.5) * 0.4;
      const len = Math.sqrt(vx[i] * vx[i] + vz[i] * vz[i]);
      if (len > 0.001) { vx[i] /= len; vz[i] /= len; }
      if (Math.abs(px[i]) > Bound) vx[i] -= Math.sign(px[i]) * 0.3;
      if (Math.abs(pz[i]) > Bound) vz[i] -= Math.sign(pz[i]) * 0.3;
      px[i] += vx[i] * spd * s;
      pz[i] += vz[i] * spd * s;
    }
  }

  // ── GPU path ──────────────────────────────────────────────────────────────
  const gpu          = new LotkaVolterraGpu(engine, scene);
  const gpuAvailable = gpu.ok;

  // ── Visibility management ─────────────────────────────────────────────────
  // Thin-instanced meshes with count 0 still draw one base instance from the
  // stale GPU buffer — use setMeshVisible (which bumps the visibility epoch so
  // the opaque bundle re-records) to hide the idle set outright.
  function applyVisibility(): void {
    const g = params.gpuMode && gpuAvailable;
    setMeshVisible(preyMesh,      !g);
    setMeshVisible(predMesh,      !g);
    setMeshVisible(gpu.preyMesh,   g);
    setMeshVisible(gpu.predMesh,   g);
  }

  // ── Mode switch helpers ───────────────────────────────────────────────────

  function switchToGpu(): void {
    if (!gpuAvailable) { params.gpuMode = false; return; }
    // Zero-out CPU meshes.
    setThinInstanceCount(preyMesh, 0);
    setThinInstanceCount(predMesh, 0);
    invalidateRenderBundles(engine);
    gpu.seedRandom(GpuBound);
    camera.radius = GpuBound * 2.5;
    applyVisibility();
  }

  function switchToCpu(): void {
    if (gpuAvailable) {
      gpu.setPreyCount(0);
      gpu.setPredCount(0);
    }
    camera.radius = 45;
    applyVisibility();
  }

  // Establish initial visibility (GPU meshes hidden at startup).
  let prevGpu = false;
  applyVisibility();

  // ── Per-frame update ──────────────────────────────────────────────────────

  function update(dt: number) {
    const s = dt * 0.001 * params.speed;
    const isGpu = params.gpuMode;

    // RK4 ODE runs always — drives the graph in both modes.
    const { alpha, beta, delta, gamma } = params;
    function dPrey(x: number, y: number) { return alpha * x - beta * x * y; }
    function dPred(x: number, y: number) { return delta * x * y - gamma * y; }

    const k1x = dPrey(preyPop, predPop);
    const k1y = dPred(preyPop, predPop);
    const k2x = dPrey(preyPop + k1x * s * 0.5, predPop + k1y * s * 0.5);
    const k2y = dPred(preyPop + k1x * s * 0.5, predPop + k1y * s * 0.5);
    const k3x = dPrey(preyPop + k2x * s * 0.5, predPop + k2y * s * 0.5);
    const k3y = dPred(preyPop + k2x * s * 0.5, predPop + k2y * s * 0.5);
    const k4x = dPrey(preyPop + k3x * s, predPop + k3y * s);
    const k4y = dPred(preyPop + k3x * s, predPop + k3y * s);

    preyPop = Math.max(0.1, preyPop + (k1x + 2*k2x + 2*k3x + k4x) * s * 0.1667);
    predPop = Math.max(0.1, predPop + (k1y + 2*k2y + 2*k3y + k4y) * s * 0.1667);

    // React to mode toggle.
    if (isGpu !== prevGpu) {
      prevGpu = isGpu;
      if (isGpu) switchToGpu(); else switchToCpu();
    }

    if (!isGpu) {
      // ── CPU path ────────────────────────────────────────────────────────
      const targetPrey = Math.min(Math.round(preyPop * 5), MaxPrey);
      const targetPred = Math.min(Math.round(predPop * 5), MaxPred);
      while (preyN < targetPrey) { spawnAgent(preyPx, preyPy, preyPz, preyVx, preyVy, preyVz, preyN); preyN++; }
      preyN = Math.max(0, Math.min(preyN, targetPrey));
      while (predN < targetPred) { spawnAgent(predPx, predPy, predPz, predVx, predVy, predVz, predN); predN++; }
      predN = Math.max(0, Math.min(predN, targetPred));

      const ds = dt * 0.001;
      moveAgents(preyN, params.preySpeed, ds, preyPx, preyPy, preyPz, preyVx, preyVy, preyVz);
      moveAgents(predN, params.predSpeed, ds, predPx, predPy, predPz, predVx, predVy, predVz);

      for (let i = 0; i < preyN; i++) preyBuf.writeScale(i, preyPx[i], preyPy[i], preyPz[i]);
      preyBuf.commit(preyMesh, preyN, false);
      for (let i = 0; i < predN; i++) predBuf.writeScale(i, predPx[i], predPy[i], predPz[i]);
      predBuf.commit(predMesh, predN, false);

      // Readouts: show actual rendered agent count so the number matches the visual.
      preyCountRef.value = preyN;
      predCountRef.value = predN;
    } else {
      // ── GPU path ─────────────────────────────────────────────────────────
      const targetPrey = Math.min(Math.round(preyPop * GpuDensity), GpuMaxPrey);
      const targetPred = Math.min(Math.round(predPop * GpuDensity), GpuMaxPred);
      gpu.setPreyCount(targetPrey);
      gpu.setPredCount(targetPred);
      gpu.dispatch(dt, {
        preyCount: targetPrey,
        predCount: targetPred,
        preySpeed: params.preySpeed,
        predSpeed: params.predSpeed,
        bound:     GpuBound,
      });

      // Readouts: show actual rendered agent count so the number matches the visual.
      preyCountRef.value = targetPrey;
      predCountRef.value = targetPred;
    }
  }

  function reset() {
    resetPops();
    if (params.gpuMode && gpuAvailable) gpu.seedRandom(GpuBound);
  }

  return {
    params,
    schema: [
      {
        type:  "toggle",
        key:   "gpuMode",
        label: gpuAvailable ? "GPU Mode" : "GPU Mode (unavailable)",
      },
      { type: "slider", key: "alpha",     label: "Prey Birth (α)",  min: 0.1, max: 3,   step: 0.05 },
      { type: "slider", key: "beta",      label: "Predation (β)",   min: 0.01, max: 0.5, step: 0.01 },
      { type: "slider", key: "delta",     label: "Pred. Birth (δ)", min: 0.01, max: 0.5, step: 0.01 },
      { type: "slider", key: "gamma",     label: "Pred. Death (γ)", min: 0.1, max: 3,   step: 0.05 },
      { type: "slider", key: "speed",     label: "Time Scale",      min: 0.1, max: 3,   step: 0.1  },
    ],
    readouts: { prey: preyCountRef, predators: predCountRef },
    seriesLabels: ["Prey", "Predators"],
    seriesColors: ["#00e5ff", "#ff00cc"],
    // Graph tracks the same rendered count shown in the readouts so the lines
    // match the numbers (and the on-screen population shape is identical in
    // both modes — just scaled by 5× CPU or GpuDensity× GPU).
    getSeries: () => [preyCountRef.value as number, predCountRef.value as number],
    update,
    reset,
    detach,
  };
}
