import { reactive, ref } from "vue";
import {
  addToScene,
  attachControl,
  createArcRotateCamera,
  createSphere,
  createCylinder,
  createHemisphericLight,
  createPbrMaterial,
  invalidateRenderBundles,
  setMeshVisible,
  setThinInstanceCount,
} from "@babylonjs/lite";
import type { EngineContext, SceneContext } from "@babylonjs/lite";
import { AgentBuffer } from "@/lib/agents";
import type { SimHandle, SliderParam } from "@/types/sim";
import { DoublePendulumGpu, GpuMaxPendulums } from "./gpu";

const MaxPendulums = 24;
const TrailLen     = 200;

// GPU pivot matches the CPU pivot y-coordinate (oy = 5 in CPU update).
const GpuPivotY = 5;

export function buildSimScene(
  engine: EngineContext,
  scene: SceneContext,
  canvas: HTMLCanvasElement
): SimHandle {
  const camera = createArcRotateCamera(-Math.PI / 2, Math.PI / 2, 32, { x: 0, y: -2, z: 0 });
  scene.camera = camera;
  const detach = attachControl(camera, canvas, scene);

  addToScene(scene, createHemisphericLight([0, 1, 0], 0.7));

  // ── CPU path meshes ───────────────────────────────────────────────────────
  const arm1Mesh = createCylinder(engine, { height: 1, diameter: 0.15, tessellation: 8 });
  arm1Mesh.material = createPbrMaterial({ baseColorFactor: [0.9, 0.9, 0.9, 1], metallicFactor: 0.8, roughnessFactor: 0.3 });
  addToScene(scene, arm1Mesh);

  const arm2Mesh = createCylinder(engine, { height: 1, diameter: 0.15, tessellation: 8 });
  arm2Mesh.material = createPbrMaterial({ baseColorFactor: [0.6, 0.6, 0.6, 1], metallicFactor: 0.8, roughnessFactor: 0.3 });
  addToScene(scene, arm2Mesh);

  const bobMesh = createSphere(engine, { segments: 8, diameter: 0.5 });
  bobMesh.material = createPbrMaterial({ baseColorFactor: [0.05, 0.85, 1, 1], metallicFactor: 0.4, roughnessFactor: 0.3 });
  addToScene(scene, bobMesh);

  const trailMesh = createSphere(engine, { segments: 4, diameter: 0.12 });
  trailMesh.material = createPbrMaterial({ baseColorFactor: [1, 0.05, 0.8, 1], metallicFactor: 0, roughnessFactor: 1 });
  addToScene(scene, trailMesh);

  const params = reactive({
    count:   8,
    L1:      5.0,
    L2:      5.0,
    m1:      1.0,
    m2:      1.0,
    g:       9.81,
    speed:   1.0,
    spread:  0.001,
    gpuMode: false,
  });

  const chaosRef      = ref(0);
  const pendulumsRef  = ref<number | string>(params.count);

  // Reactive count slider — max swaps between CPU and GPU ceiling.
  const countSlider = reactive<SliderParam>({
    type:  "slider",
    key:   "count",
    label: "Pendulums",
    min:   1,
    max:   MaxPendulums,
    step:  1,
  });

  // ── CPU state ─────────────────────────────────────────────────────────────
  const theta1 = new Float64Array(MaxPendulums);
  const theta2 = new Float64Array(MaxPendulums);
  const omega1 = new Float64Array(MaxPendulums);
  const omega2 = new Float64Array(MaxPendulums);

  const trailX      = new Float32Array(MaxPendulums * TrailLen);
  const trailY      = new Float32Array(MaxPendulums * TrailLen);
  const trailZ      = new Float32Array(MaxPendulums * TrailLen);
  const trailHeads  = new Int32Array(MaxPendulums);
  const trailCounts = new Int32Array(MaxPendulums);

  const arm1Buf  = new AgentBuffer(MaxPendulums);
  const arm2Buf  = new AgentBuffer(MaxPendulums);
  const bobBuf   = new AgentBuffer(MaxPendulums);
  const trailBuf = new AgentBuffer(MaxPendulums * TrailLen);
  arm1Buf.attach(engine, arm1Mesh);
  arm2Buf.attach(engine, arm2Mesh);
  bobBuf.attach(engine, bobMesh);
  trailBuf.attach(engine, trailMesh);

  function spawnPendulums() {
    const n = params.count;
    const baseAngle1 = Math.PI * 0.7;
    const baseAngle2 = Math.PI * 0.3;
    for (let i = 0; i < n; i++) {
      theta1[i] = baseAngle1 + i * params.spread;
      theta2[i] = baseAngle2 + i * params.spread * 0.8;
      omega1[i] = 0;
      omega2[i] = 0;
      trailHeads[i]  = 0;
      trailCounts[i] = 0;
    }
    chaosRef.value = 0;
  }
  spawnPendulums();

  function pendHue(i: number, n: number): [number, number, number] {
    const t = n > 1 ? i / (n - 1) : 0;
    return [t, 1 - t * 0.8, 1 - t * 0.4];
  }

  function doublePendulumAccel(
    t1: number, t2: number, o1: number, o2: number,
    L1: number, L2: number, m1: number, m2: number, g: number
  ): [number, number] {
    const dt = t2 - t1;
    const denom1 = (m1 + m2) * L1 - m2 * L1 * Math.cos(dt) * Math.cos(dt);
    const denom2 = (L2 / L1) * denom1;
    const a1 = (m2 * L1 * o1 * o1 * Math.sin(dt) * Math.cos(dt)
      + m2 * g * Math.sin(t2) * Math.cos(dt)
      + m2 * L2 * o2 * o2 * Math.sin(dt)
      - (m1 + m2) * g * Math.sin(t1)) / denom1;
    const a2 = (-(m1 + m2) * L1 * o1 * o1 * Math.sin(dt) * Math.cos(dt)
      + (m1 + m2) * g * Math.sin(t1) * Math.cos(dt)
      - (m1 + m2) * L2 * o2 * o2 * Math.sin(dt)
      - (m1 + m2) * g * Math.sin(t2)) / denom2;
    return [a1, a2];
  }

  function rk4Step(i: number, s: number) {
    const { L1, L2, m1, m2, g } = params;
    const t1 = theta1[i], t2 = theta2[i], o1 = omega1[i], o2 = omega2[i];
    const [k1a1, k1a2] = doublePendulumAccel(t1, t2, o1, o2, L1, L2, m1, m2, g);
    const [k2a1, k2a2] = doublePendulumAccel(t1 + o1*s*0.5, t2 + o2*s*0.5, o1 + k1a1*s*0.5, o2 + k1a2*s*0.5, L1, L2, m1, m2, g);
    const [k3a1, k3a2] = doublePendulumAccel(t1 + o1*s*0.5, t2 + o2*s*0.5, o1 + k2a1*s*0.5, o2 + k2a2*s*0.5, L1, L2, m1, m2, g);
    const [k4a1, k4a2] = doublePendulumAccel(t1 + o1*s, t2 + o2*s, o1 + k3a1*s, o2 + k3a2*s, L1, L2, m1, m2, g);
    omega1[i] += (k1a1 + 2*k2a1 + 2*k3a1 + k4a1) * s * 0.1667;
    omega2[i] += (k1a2 + 2*k2a2 + 2*k3a2 + k4a2) * s * 0.1667;
    theta1[i] += omega1[i] * s;
    theta2[i] += omega2[i] * s;
  }

  // ── GPU path ──────────────────────────────────────────────────────────────
  const gpu          = new DoublePendulumGpu(engine, scene);
  const gpuAvailable = gpu.ok;

  // ── Visibility management ─────────────────────────────────────────────────
  function applyVisibility(): void {
    const g = params.gpuMode && gpuAvailable;
    setMeshVisible(arm1Mesh,     !g);
    setMeshVisible(arm2Mesh,     !g);
    setMeshVisible(bobMesh,      !g);
    setMeshVisible(trailMesh,    !g);
    setMeshVisible(gpu.arm1Mesh,  g);
    setMeshVisible(gpu.arm2Mesh,  g);
    setMeshVisible(gpu.bobMesh,   g);
  }

  function reframeCamera(): void {
    if (params.gpuMode && gpuAvailable) {
      camera.radius = (params.L1 + params.L2) * 2.2;
    } else {
      camera.radius = 32;
    }
  }

  function switchToGpu(): void {
    if (!gpuAvailable) { params.gpuMode = false; return; }
    countSlider.max = GpuMaxPendulums;
    // Zero out CPU meshes.
    setThinInstanceCount(arm1Mesh,  0);
    setThinInstanceCount(arm2Mesh,  0);
    setThinInstanceCount(bobMesh,   0);
    setThinInstanceCount(trailMesh, 0);
    invalidateRenderBundles(engine);
    gpu.setCount(params.count);
    gpu.seedSpread(params.count, params.spread);
    reframeCamera();
    applyVisibility();
  }

  function switchToCpu(): void {
    countSlider.max = MaxPendulums;
    if (params.count > MaxPendulums) params.count = MaxPendulums;
    if (gpuAvailable) gpu.setCount(0);
    reframeCamera();
    applyVisibility();
  }

  let prevGpu   = false;
  let prevCount = params.count;
  applyVisibility();

  // ── Per-frame update ──────────────────────────────────────────────────────

  function update(dt: number) {
    const n     = params.count;
    const isGpu = params.gpuMode;

    if (isGpu !== prevGpu) {
      prevGpu = isGpu;
      if (isGpu) switchToGpu(); else switchToCpu();
    }

    pendulumsRef.value = n;

    if (!isGpu) {
      // ── CPU path ────────────────────────────────────────────────────────
      const s        = dt * 0.001 * params.speed;
      const substeps = 4;
      const ss       = s / substeps;
      const { L1, L2 } = params;

      for (let i = 0; i < n; i++) {
        for (let k = 0; k < substeps; k++) rk4Step(i, ss);

        const ox = i * 2.5 - (n - 1) * 1.25;
        const oy = GpuPivotY;

        const b1x = ox + L1 * Math.sin(theta1[i]);
        const b1y = oy - L1 * Math.cos(theta1[i]);
        const b2x = b1x + L2 * Math.sin(theta2[i]);
        const b2y = b1y - L2 * Math.cos(theta2[i]);

        const a1cx = (ox + b1x) * 0.5, a1cy = (oy + b1y) * 0.5;
        const a1ang = Math.atan2(b1x - ox, oy - b1y);
        const a1len = Math.sqrt((b1x - ox) ** 2 + (b1y - oy) ** 2);
        arm1Buf.writeTransform(i, a1cx, a1cy, 0, 0, 0, Math.sin(a1ang * 0.5), Math.cos(a1ang * 0.5), 1, a1len * 2, 1);

        const a2cx = (b1x + b2x) * 0.5, a2cy = (b1y + b2y) * 0.5;
        const a2ang = Math.atan2(b2x - b1x, b1y - b2y);
        const a2len = Math.sqrt((b2x - b1x) ** 2 + (b2y - b1y) ** 2);
        arm2Buf.writeTransform(i, a2cx, a2cy, 0, 0, 0, Math.sin(a2ang * 0.5), Math.cos(a2ang * 0.5), 1, a2len * 2, 1);

        bobBuf.writeScale(i, b2x, b2y, 0);
        const [r, g, b] = pendHue(i, n);
        bobBuf.writeColor(i, r, g, b);
        arm1Buf.writeColor(i, 0.7, 0.7, 0.7);
        arm2Buf.writeColor(i, 0.5, 0.5, 0.5);

        const th   = trailHeads[i];
        const base = i * TrailLen + th;
        trailX[base] = b2x; trailY[base] = b2y; trailZ[base] = 0;
        trailHeads[i] = (th + 1) % TrailLen;
        if (trailCounts[i] < TrailLen) trailCounts[i]++;
      }

      let tc = 0;
      for (let i = 0; i < n; i++) {
        const cnt  = trailCounts[i];
        const head = trailHeads[i];
        for (let t = 0; t < cnt; t++) {
          const alpha = t / cnt;
          const slot  = i * TrailLen + ((head - cnt + t + TrailLen) % TrailLen);
          trailBuf.writeScale(tc, trailX[slot], trailY[slot], trailZ[slot], 0.06 + 0.08 * alpha);
          const [r, g, b] = pendHue(i, n);
          trailBuf.writeColor(tc, r * alpha, g * alpha, b * alpha, alpha);
          tc++;
        }
      }

      arm1Buf.commit(arm1Mesh, n);
      arm2Buf.commit(arm2Mesh, n);
      bobBuf.commit(bobMesh, n);
      trailBuf.commit(trailMesh, tc);

      if (n > 1) {
        let maxDiff = 0;
        for (let i = 1; i < n; i++) {
          const diff = Math.abs(theta2[i] - theta2[0]);
          if (diff > maxDiff) maxDiff = diff;
        }
        chaosRef.value = Math.round(maxDiff * 10) * 0.1;
      }
    } else {
      // ── GPU path ─────────────────────────────────────────────────────────
      if (n !== prevCount) {
        prevCount = n;
        gpu.setCount(n);
        gpu.seedSpread(n, params.spread);
        reframeCamera();
      }
      gpu.dispatch(dt, {
        count:    n,
        L1:       params.L1,
        L2:       params.L2,
        m1:       params.m1,
        m2:       params.m2,
        g:        params.g,
        speed:    params.speed,
        substeps: 4,
        pivotY:   GpuPivotY,
      });
    }
  }

  function reset() {
    if (params.gpuMode && gpuAvailable) {
      gpu.seedSpread(params.count, params.spread);
    } else {
      spawnPendulums();
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
      countSlider,
      { type: "slider", key: "L1",     label: "Arm 1 Length",    min: 1,      max: 8,    step: 0.25 },
      { type: "slider", key: "L2",     label: "Arm 2 Length",    min: 1,      max: 8,    step: 0.25 },
      { type: "slider", key: "m1",     label: "Mass 1",          min: 0.1,    max: 5,    step: 0.1 },
      { type: "slider", key: "m2",     label: "Mass 2",          min: 0.1,    max: 5,    step: 0.1 },
      { type: "slider", key: "g",      label: "Gravity",         min: 1,      max: 20,   step: 0.5 },
      { type: "slider", key: "speed",  label: "Time Scale",      min: 0.1,    max: 3,    step: 0.1 },
      { type: "slider", key: "spread", label: "Initial Spread",  min: 0.0001, max: 0.01, step: 0.0001 },
    ],
    readouts: { pendulums: pendulumsRef, "chaos Δ": chaosRef },
    update,
    reset,
    detach,
  };
}
