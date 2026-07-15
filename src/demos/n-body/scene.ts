import { reactive, ref } from "vue";
import {
  addToScene,
  attachControl,
  createArcRotateCamera,
  createHemisphericLight,
  createSphere,
  createCylinder,
  createPbrMaterial,
  createDirectionalLight,
  invalidateRenderBundles,
  setMeshVisible,
  setThinInstanceCount,
} from "@babylonjs/lite";
import type { EngineContext, SceneContext } from "@babylonjs/lite";
import { AgentBuffer, upToDirQuat } from "@/lib/agents";
import type { SimHandle, SliderParam } from "@/types/sim";
import { NBodyGpu, GpuMaxBodies } from "./gpu";

const MaxBodies = 64;
const TrailLen  = 120;

const Presets: Record<string, () => void> = {};

export function buildSimScene(
  engine: EngineContext,
  scene: SceneContext,
  canvas: HTMLCanvasElement
): SimHandle {
  const camera = createArcRotateCamera(-Math.PI * 0.5, Math.PI * 0.35, 40, { x: 0, y: 0, z: 0 });
  scene.camera = camera;
  const detach = attachControl(camera, canvas, scene);

  addToScene(scene, createHemisphericLight([0, 1, 0], 0.4));
  const sun = createDirectionalLight([0.3, -1, 0.5]);
  addToScene(scene, sun);

  // ── CPU path meshes (order matches the working boids demo) ────────────────
  const bodyMesh = createSphere(engine, { segments: 8, diameter: 1 });
  bodyMesh.material = createPbrMaterial({ baseColorFactor: [1, 0.85, 0.3, 1], metallicFactor: 0.6, roughnessFactor: 0.3 });
  addToScene(scene, bodyMesh);

  const trailMesh = createCylinder(engine, { height: 1, diameter: 1, tessellation: 6 });
  trailMesh.material = createPbrMaterial({ unlit: true, baseColorFactor: [1, 1, 1, 1], needAlphaBlending: true });
  addToScene(scene, trailMesh);

  const buf      = new AgentBuffer(MaxBodies);
  const trailBuf = new AgentBuffer(MaxBodies * TrailLen);
  buf.attach(engine, bodyMesh);
  trailBuf.attach(engine, trailMesh);

  const px   = new Float64Array(MaxBodies);
  const py   = new Float64Array(MaxBodies);
  const pz   = new Float64Array(MaxBodies);
  const vx   = new Float64Array(MaxBodies);
  const vy   = new Float64Array(MaxBodies);
  const vz   = new Float64Array(MaxBodies);
  const mass = new Float64Array(MaxBodies);

  const trailPx = new Float32Array(MaxBodies * TrailLen);
  const trailPy = new Float32Array(MaxBodies * TrailLen);
  const trailPz = new Float32Array(MaxBodies * TrailLen);
  let trailHead = 0;

  const bodyHues = [0, 30, 60, 120, 180, 210, 270, 300, 15, 45, 135, 240, 330, 90, 150, 195,
    10, 40, 70, 100, 160, 220, 250, 310, 20, 50, 80, 140, 200, 260, 280, 320];

  function hslToRgb(h: number, s: number, l: number): [number, number, number] {
    const a = s * Math.min(l, 1 - l);
    function f(n: number) {
      const k = (n + h * 0.0833333) % 12;
      return l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    }
    return [f(0), f(8), f(4)];
  }

  function spawnRandom(n: number) {
    let cx = 0, cy = 0, cz = 0, cvx = 0, cvy = 0, cvz = 0, totalM = 0;
    for (let i = 0; i < n; i++) {
      mass[i] = 0.5 + Math.random() * 4;
      px[i]   = (Math.random() - 0.5) * 30;
      py[i]   = (Math.random() - 0.5) * 30;
      pz[i]   = (Math.random() - 0.5) * 30;
      vx[i]   = (Math.random() - 0.5) * 4;
      vy[i]   = (Math.random() - 0.5) * 4;
      vz[i]   = (Math.random() - 0.5) * 4;
      cx += px[i]*mass[i]; cy += py[i]*mass[i]; cz += pz[i]*mass[i];
      cvx += vx[i]*mass[i]; cvy += vy[i]*mass[i]; cvz += vz[i]*mass[i];
      totalM += mass[i];
    }
    const invM = 1 / totalM;
    for (let i = 0; i < n; i++) {
      px[i]-=cx*invM; py[i]-=cy*invM; pz[i]-=cz*invM;
      vx[i]-=cvx*invM; vy[i]-=cvy*invM; vz[i]-=cvz*invM;
    }
    trailHead = 0;
  }

  function spawnBinary() {
    params.bodies = 2;
    mass[0] = 10; mass[1] = 10;
    px[0] = -8; py[0] = 0; pz[0] = 0;
    px[1] =  8; py[1] = 0; pz[1] = 0;
    vx[0] = 0; vy[0] =  4; vz[0] = 0;
    vx[1] = 0; vy[1] = -4; vz[1] = 0;
    trailHead = 0;
  }

  function spawnBody(i: number) {
    mass[i] = 0.5 + Math.random() * 4;
    px[i] = (Math.random() - 0.5) * 30;
    py[i] = (Math.random() - 0.5) * 30;
    pz[i] = (Math.random() - 0.5) * 30;
    vx[i] = (Math.random() - 0.5) * 4;
    vy[i] = (Math.random() - 0.5) * 4;
    vz[i] = (Math.random() - 0.5) * 4;
  }

  // Preset wrappers — registered after params is declared so they can read it.
  // (spawnRandom / spawnBinary themselves do NOT reference params.)
  spawnRandom(12); // initial seed — params not yet declared, so pass literal

  // ── GPU path ──────────────────────────────────────────────────────────────
  const gpu          = new NBodyGpu(engine, scene);
  const gpuAvailable = gpu.ok;

  // ── Shared params ─────────────────────────────────────────────────────────
  const params = reactive({
    bodies:     12,
    G:          80,
    softening:  1.5,
    speed:      1.0,
    showTrails: true,
    preset:     "random",
    gpuMode:    false,
  });

  // Register Presets wrappers now that params is available.
  Presets.random = () => spawnRandom(params.bodies);
  Presets.binary = spawnBinary;

  // Reactive bodies slider — max swaps between CPU and GPU ceiling.
  const bodiesSlider = reactive<SliderParam>({
    type:  "slider",
    key:   "bodies",
    label: "Bodies",
    min:   2,
    max:   gpuAvailable ? (params.gpuMode ? GpuMaxBodies : MaxBodies) : MaxBodies,
    step:  1,
  });

  const bodiesRef = ref(params.bodies);
  let prevBodies    = params.bodies;
  let prevGpuBodies = -1;
  let frameCount    = 0;
  let prevGpu       = false;
  let prevShowTrails = params.showTrails;

  // Establish initial visibility (CPU meshes shown, GPU meshes hidden) so an
  // idle GPU mesh never draws its base geometry before the first mode toggle.
  applyVisibility();

  // ── Mode switch helpers ───────────────────────────────────────────────────

  function seedGpu(): void {
    const n = params.bodies;
    if (params.preset === "binary") {
      params.bodies = 2;
      gpu.setBodyCount(2);
      gpu.setTrailCount(2);
      gpu.seedBinary();
      prevGpuBodies = 2;
    } else {
      gpu.setBodyCount(n);
      gpu.setTrailCount(n);
      gpu.seedRandom(n);
      prevGpuBodies = n;
    }
  }

  // A thin-instanced mesh with an active count of 0 still draws its BASE
  // geometry once, reading instance-0's transform from its GPU buffer. After a
  // GPU session that buffer holds a real body transform, so the idle mesh would
  // leave one stray sphere on screen. Toggling `visible` (which bumps the render
  // visibility epoch so the opaque bundle re-records) hides the idle set outright.
  function applyVisibility(): void {
    const g = params.gpuMode && gpuAvailable;
    const t = params.showTrails;
    setMeshVisible(bodyMesh, !g);
    setMeshVisible(trailMesh, !g && t);
    setMeshVisible(gpu.bodyMesh, g);
    setMeshVisible(gpu.trailMesh, g && t);
  }

  function switchToGpu(): void {
    if (!gpuAvailable) { params.gpuMode = false; return; }
    bodiesSlider.max = GpuMaxBodies;
    // Hide CPU meshes and drop their stale render bundles.
    setThinInstanceCount(bodyMesh, 0);
    setThinInstanceCount(trailMesh, 0);
    invalidateRenderBundles(engine);
    seedGpu();
    applyVisibility();
  }

  function switchToCpu(): void {
    bodiesSlider.max = MaxBodies;
    if (params.bodies > MaxBodies) params.bodies = MaxBodies;
    // Hide GPU meshes.
    if (gpuAvailable) {
      gpu.setBodyCount(0);
      gpu.setTrailCount(0);
    }
    // Re-run the active preset on the CPU side.
    (Presets[params.preset] ?? Presets.random)();
    prevBodies = params.bodies;
    bodiesRef.value = params.bodies;
    applyVisibility();
  }

  // ── Per-frame update ──────────────────────────────────────────────────────

  function update(dt: number) {
    const n      = params.bodies;
    const isGpu  = params.gpuMode;

    if (isGpu !== prevGpu) {
      prevGpu = isGpu;
      if (isGpu) switchToGpu(); else switchToCpu();
    }

    // Trails toggled within the active mode: refresh mesh visibility so an idle
    // (count 0) trail mesh never leaves a stray base cylinder behind.
    if (params.showTrails !== prevShowTrails) {
      prevShowTrails = params.showTrails;
      applyVisibility();
    }

    if (!isGpu) {
      // ── CPU physics (Float64, O(n²), existing logic) ──────────────────
      if (n !== prevBodies) {
        for (let i = prevBodies; i < n; i++) spawnBody(i);
        prevBodies = n;
        bodiesRef.value = n;
      }

      const s    = (dt * 0.001) * params.speed;
      const G    = params.G;
      const eps2 = params.softening * params.softening;

      for (let i = 0; i < n; i++) {
        let fx = 0, fy = 0, fz = 0;
        for (let j = 0; j < n; j++) {
          if (i === j) continue;
          const dx = px[j]-px[i], dy = py[j]-py[i], dz = pz[j]-pz[i];
          const d2 = dx*dx + dy*dy + dz*dz + eps2;
          const d3 = d2 * Math.sqrt(d2);
          const f  = G * mass[j] / d3;
          fx += f*dx; fy += f*dy; fz += f*dz;
        }
        vx[i] += fx*s; vy[i] += fy*s; vz[i] += fz*s;
      }
      for (let i = 0; i < n; i++) {
        px[i] += vx[i]*s; py[i] += vy[i]*s; pz[i] += vz[i]*s;
      }

      for (let i = 0; i < n; i++) {
        const s2 = Math.cbrt(mass[i]) * 0.7;
        buf.writeScale(i, px[i], py[i], pz[i], s2);
        const [r, g, b] = hslToRgb(bodyHues[i % bodyHues.length], 0.9, 0.65);
        buf.writeColor(i, r, g, b);
      }
      buf.commit(bodyMesh, n);

      frameCount++;
      if (params.showTrails && frameCount % 2 === 0) {
        const base = trailHead * MaxBodies;
        for (let i = 0; i < n; i++) {
          trailPx[base+i] = px[i]; trailPy[base+i] = py[i]; trailPz[base+i] = pz[i];
        }
        trailHead = (trailHead + 1) % TrailLen;

        const HeadThick = 0.22;
        let tc = 0;
        for (let t = 0; t < TrailLen - 1; t++) {
          const f       = t / (TrailLen - 1);
          const slotNew = ((trailHead - t - 1 + TrailLen) % TrailLen) * MaxBodies;
          const slotOld = ((trailHead - t - 2 + TrailLen) % TrailLen) * MaxBodies;
          for (let i = 0; i < n; i++) {
            const ax = trailPx[slotNew+i], ay = trailPy[slotNew+i], az = trailPz[slotNew+i];
            const bx = trailPx[slotOld+i], by = trailPy[slotOld+i], bz = trailPz[slotOld+i];
            const dx = bx-ax, dy = by-ay, dz = bz-az;
            const len = Math.sqrt(dx*dx + dy*dy + dz*dz);
            if (len < 0.01) continue;
            const mx = (ax+bx)*0.5, my = (ay+by)*0.5, mz = (az+bz)*0.5;
            const thickness = HeadThick * (1 - f);
            const [qx, qy, qz, qw] = upToDirQuat(dx, dy, dz);
            trailBuf.writeTransform(tc, mx, my, mz, qx, qy, qz, qw, thickness, len, thickness);
            const [r, g, b] = hslToRgb(bodyHues[i % bodyHues.length], 0.9, 0.65 * (1 - f*0.85));
            trailBuf.writeColor(tc, r, g, b, 1 - f*0.85);
            tc++;
          }
        }
        trailBuf.commit(trailMesh, tc);
      } else if (!params.showTrails) {
        trailBuf.commit(trailMesh, 0);
      }

      bodiesRef.value = n;
    } else {
      // ── GPU physics + GPU-driven render ───────────────────────────────
      if (n !== prevGpuBodies) {
        gpu.setBodyCount(n);
        gpu.setTrailCount(n);
        gpu.seedRandom(n);
        prevGpuBodies = n;
      }
      gpu.dispatch(dt, {
        bodies:     n,
        G:          params.G,
        softening:  params.softening,
        speed:      params.speed,
        showTrails: params.showTrails,
      });
      bodiesRef.value = n;
    }
  }

  function reset() {
    if (params.gpuMode && gpuAvailable) {
      // Seed the GPU with the active preset.
      seedGpu();
    } else {
      (Presets[params.preset] ?? Presets.random)();
      prevBodies = params.bodies;
      bodiesRef.value = params.bodies;
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
      bodiesSlider,
      { type: "slider", key: "G",          label: "Gravity G",  min: 10,  max: 300, step: 5 },
      { type: "slider", key: "softening",  label: "Softening",  min: 0.2, max: 5,   step: 0.1 },
      { type: "slider", key: "speed",      label: "Time Scale", min: 0.1, max: 3,   step: 0.1 },
      { type: "toggle", key: "showTrails", label: "Show Trails" },
      { type: "select", key: "preset",     label: "Preset",     options: ["random", "binary"] },
      { type: "button", key: "_reset",     label: "Apply Preset", action: () => reset() },
    ],
    readouts: { bodies: bodiesRef },
    update,
    reset,
    detach,
  };
}
