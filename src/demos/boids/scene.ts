import { reactive, ref } from "vue";
import {
  addToScene,
  attachControl,
  createArcRotateCamera,
  createCapsule,
  createHemisphericLight,
  createPbrMaterial,
  invalidateRenderBundles,
} from "@babylonjs/lite";
import type { EngineContext, SceneContext } from "@babylonjs/lite";
import { AgentBuffer, upToDirQuat } from "@/lib/agents";
import type { SimHandle, SliderParam } from "@/types/sim";
import { CpuBoids } from "./cpu";
import { GpuBoids, GpuCapacity } from "./gpu";
import type { CpuBoidsParams } from "./cpu";

// Per-mode count ceilings.
const CpuCapacity = 3_000;   // CPU physics is O(n) per frame — keep it modest
const GpuMax      = GpuCapacity;   // 200k — spatial-grid compute on the GPU

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

  // ── GPU compute system ────────────────────────────────────────────────────
  const gpu = new GpuBoids(engine);
  const gpuAvailable = gpu.ok;

  // ── Shared rendering mesh + AgentBuffer ───────────────────────────────────
  // One mesh, one AgentBuffer — used for BOTH CPU and GPU paths.
  // Capacity is GpuCapacity when GPU is available (for tens of thousands).
  const renderCapacity = gpuAvailable ? GpuCapacity : CpuCapacity;

  const mesh = createCapsule(engine, { height: 1.2, radius: 0.35, tessellation: 6 });
  mesh.material = createPbrMaterial({ baseColorFactor: [0.05, 0.8, 1, 1], metallicFactor: 0.2, roughnessFactor: 0.5 });
  addToScene(scene, mesh);

  const agentBuf = new AgentBuffer(renderCapacity);
  agentBuf.attach(engine, mesh);

  // ── CPU physics ────────────────────────────────────────────────────────────
  const cpu = new CpuBoids(CpuCapacity);
  cpu.spawnAll();

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
  });

  const agentsRef = ref<number | string>(params.count);
  const modeRef   = ref<string>("CPU");

  // Reactive count slider — its `max` swaps between CpuCapacity and GpuMax with
  // the active mode. ControlPanel binds :max="def.max", so mutating it here
  // updates the slider range live.
  const countSlider = reactive<SliderParam>({
    type: "slider",
    key: "count",
    label: "Count",
    min: 10,
    max: gpuAvailable ? (params.gpuMode ? GpuMax : CpuCapacity) : CpuCapacity,
    step: 10,
  });

  let prevCount = params.count;
  let prevGpu   = false;
  let readbackPending = false;

  // ── Mode switch helpers ───────────────────────────────────────────────────

  function switchToGpu(): void {
    if (!gpuAvailable) {
      params.gpuMode = false;
      return;
    }
    modeRef.value = "GPU";
    countSlider.max = GpuMax;   // unlock the full GPU range
    // Seed the GPU state from current CPU positions so the flock continues.
    gpu.seedFrom(params.count, cpu.px, cpu.py, cpu.pz, cpu.vx, cpu.vy, cpu.vz);
    invalidateRenderBundles(engine);
  }

  function switchToCpu(): void {
    if (readbackPending) return;
    modeRef.value = "CPU";
    // Clamp the slider range and snap an over-CPU-cap count back down.
    countSlider.max = CpuCapacity;
    if (params.count > CpuCapacity) params.count = CpuCapacity;
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
      if (isGpu) {
        switchToGpu();
      } else {
        switchToCpu();
      }
    }

    if (!isGpu || readbackPending) {
      // ── CPU physics path ────────────────────────────────────────────────
      const cpuCount = Math.min(n, CpuCapacity);
      if (cpuCount !== prevCount) {
        for (let i = prevCount; i < cpuCount; i++) cpu.spawnOne(i);
        prevCount = cpuCount;
      }
      const cpuParams: CpuBoidsParams = {
        count: cpuCount,
        speed: params.speed,
        separation: params.separation,
        alignment: params.alignment,
        cohesion: params.cohesion,
        radius: params.radius,
        separationRadius: params.separationRadius,
      };
      cpu.update(dt, cpuParams, agentBuf, mesh);
      agentsRef.value = cpuCount;
      modeRef.value   = "CPU";
    } else {
      // ── GPU physics path ────────────────────────────────────────────────
      // Dispatch compute (spatial-grid physics, O(n) neighbor search).
      gpu.dispatch(dt, {
        count: n,
        speed: params.speed,
        separation: params.separation,
        alignment: params.alignment,
        cohesion: params.cohesion,
        radius: params.radius,
        separationRadius: params.separationRadius,
      });

      // Consume the latest async readback data (populated ~1 frame after dispatch).
      // seedFrom() pre-populates the arrays so the first frame shows something.
      const rbCount = gpu.readbackCount;
      if (rbCount > 0) {
        const spd = params.speed;
        for (let i = 0; i < rbCount; i++) {
          const vLen = Math.sqrt(gpu.vx[i] ** 2 + gpu.vy[i] ** 2 + gpu.vz[i] ** 2);
          const [qx, qy, qz, qw] = upToDirQuat(gpu.vx[i], gpu.vy[i], gpu.vz[i]);
          agentBuf.writeTransform(i, gpu.px[i], gpu.py[i], gpu.pz[i], qx, qy, qz, qw, 1, 1, 1);
          const t = Math.min(vLen / (spd * 1.5), 1);
          agentBuf.writeColor(i, t, 1 - t * 0.8, 1 - t * 0.4);
        }
        agentBuf.commit(mesh, rbCount);
      }

      agentsRef.value = n;
      modeRef.value   = "GPU";
    }
  }

  function reset(): void {
    cpu.spawnAll();
    prevCount = params.count;
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
      { type: "slider", key: "speed",            label: "Speed",           min: 1,   max: 20,       step: 0.5 },
      { type: "slider", key: "radius",           label: "Neighbor Radius", min: 1,   max: 15,       step: 0.5 },
      { type: "slider", key: "separationRadius", label: "Sep. Radius",     min: 0.5, max: 6,        step: 0.25 },
      { type: "slider", key: "separation",       label: "Separation",      min: 0,   max: 5,        step: 0.1 },
      { type: "slider", key: "alignment",        label: "Alignment",       min: 0,   max: 5,        step: 0.1 },
      { type: "slider", key: "cohesion",         label: "Cohesion",        min: 0,   max: 5,        step: 0.1 },
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
