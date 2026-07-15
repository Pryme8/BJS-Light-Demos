import { reactive, ref } from "vue";
import {
  addToScene,
  attachControl,
  createArcRotateCamera,
  createCapsule,
  createHemisphericLight,
  createPbrMaterial,
  invalidateRenderBundles,
  setThinInstanceCount,
} from "@babylonjs/lite";
import type { EngineContext, SceneContext } from "@babylonjs/lite";
import { AgentBuffer } from "@/lib/agents";
import type { SimHandle, SliderParam } from "@/types/sim";
import { CpuBoids } from "./cpu";
import { GpuBoids, GpuCapacity, gpuBoundForCount } from "./gpu";
import type { CpuBoidsParams } from "./cpu";

// Per-mode count ceilings.
const CpuCapacity = 3_000;         // CPU physics is O(n) per frame — keep it modest
const GpuMax      = GpuCapacity;   // 200k — spatial-grid compute + GPU-driven render

export function buildSimScene(
  engine: EngineContext,
  scene: SceneContext,
  canvas: HTMLCanvasElement
): SimHandle {
  // ── Camera + light ────────────────────────────────────────────────────────
  const camera = createArcRotateCamera(-Math.PI * 0.5, Math.PI * 0.35, 55, { x: 0, y: 0, z: 0 });
  scene.camera = camera;
  const detach = attachControl(camera, canvas, scene);
  addToScene(scene, createHemisphericLight([0, 1, 0], 0.9));

  // ── CPU path: its own PBR + AgentBuffer capsule mesh ──────────────────────
  const cpuMesh = createCapsule(engine, { height: 1.2, radius: 0.35, tessellation: 6 });
  cpuMesh.material = createPbrMaterial({ baseColorFactor: [0.05, 0.8, 1, 1], metallicFactor: 0.2, roughnessFactor: 0.5 });
  addToScene(scene, cpuMesh);

  const agentBuf = new AgentBuffer(CpuCapacity);
  agentBuf.attach(engine, cpuMesh);

  const cpu = new CpuBoids(CpuCapacity);
  cpu.spawnAll();

  // ── GPU path: GpuBoids owns its own compute-driven PBR mesh ───────────────
  const gpu = new GpuBoids(engine, scene);
  const gpuAvailable = gpu.ok;

  // ── Shared params ─────────────────────────────────────────────────────────
  const params = reactive({
    count: 300,
    speed: 6,
    separation: 1.8,
    alignment: 1.0,
    cohesion: 0.6,
    radius: 5.0,
    separationRadius: 2.0,
    gpuMode: false,
    collision: false,
    collisionRadius: 0.4,
    collisionStrength: 0.5,
  });

  const agentsRef = ref<number | string>(params.count);
  const modeRef   = ref<string>("CPU");

  // Reactive count slider — `max` swaps between CpuCapacity and GpuMax with mode.
  const countSlider = reactive<SliderParam>({
    type: "slider",
    key: "count",
    label: "Count",
    min: 10,
    max: gpuAvailable ? (params.gpuMode ? GpuMax : CpuCapacity) : CpuCapacity,
    step: 10,
  });

  let prevGpu         = false;
  let prevGpuCount    = -1;  // last count pushed to gpu.setCount
  let readbackPending = false;

  // Frame the camera to the density-scaled GPU world so the whole flock is visible.
  function frameGpu(n: number): void {
    camera.radius = Math.max(45, gpuBoundForCount(n) * 2.4);
  }

  // ── Mode switch helpers ───────────────────────────────────────────────────

  function switchToGpu(): void {
    if (!gpuAvailable) {
      params.gpuMode = false;
      return;
    }
    modeRef.value = "GPU";
    countSlider.max = GpuMax;
    // Hide the CPU mesh and drop its stale bundle.
    setThinInstanceCount(cpuMesh, 0);
    invalidateRenderBundles(engine);
    // Seamless: seed GPU state from the current CPU flock, keep the count.
    gpu.setCount(params.count);
    gpu.seedFrom(params.count, cpu.px, cpu.py, cpu.pz, cpu.vx, cpu.vy, cpu.vz);
    prevGpuCount = params.count;
    frameGpu(params.count);
  }

  function switchToCpu(): void {
    if (readbackPending) return;
    modeRef.value = "CPU";
    // Clamp the slider range and snap an over-CPU-cap count down.
    countSlider.max = CpuCapacity;
    if (params.count > CpuCapacity) params.count = CpuCapacity;
    // Hide the GPU mesh.
    gpu.setCount(0);
    prevGpuCount = 0;

    if (!gpuAvailable) return;

    // Async GPU→CPU handoff so the flock continues seamlessly.
    readbackPending = true;
    const n = Math.min(params.count, CpuCapacity);
    gpu.readbackInto(n, cpu.px, cpu.py, cpu.pz, cpu.vx, cpu.vy, cpu.vz)
      .then(() => { readbackPending = false; })
      .catch(() => { readbackPending = false; });
  }

  // ── Per-frame update ──────────────────────────────────────────────────────

  function update(dt: number): void {
    const n     = params.count;
    const isGpu = params.gpuMode;

    // React to mode toggle
    if (isGpu !== prevGpu) {
      prevGpu = isGpu;
      if (isGpu) switchToGpu();
      else       switchToCpu();
    }

    if (!isGpu || readbackPending) {
      // ── CPU physics path (writes AgentBuffer, PBR renders it) ────────────
      // All slots are pre-spawned by spawnAll(), so raising the count simply
      // reveals already-initialized boids — no per-frame respawn needed.
      const cpuCount = Math.min(n, CpuCapacity);
      const cpuParams: CpuBoidsParams = {
        count: cpuCount,
        speed: params.speed,
        separation: params.separation,
        alignment: params.alignment,
        cohesion: params.cohesion,
        radius: params.radius,
        separationRadius: params.separationRadius,
        collision: params.collision,
        collisionRadius: params.collisionRadius,
        collisionStrength: params.collisionStrength,
      };
      cpu.update(dt, cpuParams, agentBuf, cpuMesh);
      agentsRef.value = cpuCount;
      modeRef.value   = "CPU";
    } else {
      // ── GPU physics + GPU-driven render (no readback) ────────────────────
      // Count changes (committed on slider release) re-spread the flock across
      // the density-scaled world and reframe the camera.
      if (n !== prevGpuCount) {
        gpu.setCount(n);
        gpu.spawnSpread(n);
        frameGpu(n);
        prevGpuCount = n;
      }
      gpu.dispatch(dt, {
        count: n,
        speed: params.speed,
        separation: params.separation,
        alignment: params.alignment,
        cohesion: params.cohesion,
        radius: params.radius,
        separationRadius: params.separationRadius,
        collision: params.collision,
        collisionRadius: params.collisionRadius,
        collisionStrength: params.collisionStrength,
      });
      agentsRef.value = n;
      modeRef.value   = "GPU";
    }
  }

  function reset(): void {
    cpu.spawnAll();
    agentsRef.value = params.count;
    if (params.gpuMode && gpuAvailable) {
      gpu.seedFrom(params.count, cpu.px, cpu.py, cpu.pz, cpu.vx, cpu.vy, cpu.vz);
    }
  }

  return {
    params,
    schema: [
      {
        type: "toggle",
        key: "gpuMode",
        label: gpuAvailable ? "GPU Mode" : "GPU Mode (unavailable)",
      },
      countSlider,
      { type: "slider", key: "speed",            label: "Speed",           min: 1,   max: 20, step: 0.5 },
      { type: "slider", key: "radius",           label: "Neighbor Radius", min: 1,   max: 15, step: 0.5 },
      { type: "slider", key: "separationRadius", label: "Sep. Radius",     min: 0.5, max: 6,  step: 0.25 },
      { type: "slider", key: "separation",       label: "Separation",      min: 0,   max: 5,  step: 0.1 },
      { type: "slider", key: "alignment",        label: "Alignment",       min: 0,   max: 5,  step: 0.1 },
      { type: "slider", key: "cohesion",         label: "Cohesion",        min: 0,   max: 5,  step: 0.1 },
      { type: "toggle", key: "collision",        label: "Collision" },
      { type: "slider", key: "collisionRadius",  label: "Collide Radius",  min: 0.2, max: 1.5, step: 0.05 },
      { type: "slider", key: "collisionStrength",label: "Collide Strength",min: 0,   max: 1,   step: 0.05 },
    ],
    readouts: {
      mode: modeRef,
      agents: agentsRef,
    },
    update,
    reset,
    detach,
  };
}
