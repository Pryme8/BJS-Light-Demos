/**
 * Voronoi 3D Landscape — a G×G grid of thin-instanced columns whose height is
 * driven by a selectable Voronoi metric, giving orbitable shattered-crystal
 * terrain that morphs live as seeds animate or Lloyd-relax.
 *
 * Height modes:
 *   ridge — (d2-d1)/norm: mounds inside cells, sharp valleys at Voronoi edges.
 *   cone  — 1 - d1/norm: peaks at each seed, slopes away.
 *   mesa  — regionArea/maxArea: flat-topped plateaus; bigger cells taller.
 *   noise — animated smooth noise blended with a per-region offset.
 *
 * CPU: O(G² × seeds) per frame; default gridRes=96 keeps it comfortable.
 * GPU: compute shader does the heavy per-cell pass; scales to 512×512 / 256 seeds.
 */
import { reactive, ref } from "vue";
import {
  addToScene,
  attachControl,
  createArcRotateCamera,
  createBox,
  createDirectionalLight,
  createHemisphericLight,
  createPbrMaterial,
  createSphere,
  invalidateRenderBundles,
  setMeshVisible,
  setThinInstanceCount,
} from "@babylonjs/lite";
import type { EngineContext, SceneContext } from "@babylonjs/lite";
import { AgentBuffer } from "@/lib/agents";
import type { SimHandle, SliderParam } from "@/types/sim";
import { VoronoiGpu, GpuMaxGrid, GpuMaxSeeds } from "./gpu";

const WorldSize  = 40;
const MaxGrid    = 160;
const MaxColumns = MaxGrid * MaxGrid;
const MaxSeeds   = 64;   // CPU ceiling (GPU uses GpuMaxSeeds = 256)

// Coarse Lloyd / mesa grid used in GPU mode (keeps CPU work cheap).
const LloydGrid = 96;

// Height-mode index for GPU UBO.
const HEIGHT_MODE: Record<string, number> = { ridge: 0, cone: 1, mesa: 2, noise: 3 };

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  function f(t: number) {
    if (t < 0) t += 1; if (t > 1) t -= 1;
    if (t < 0.1667) return p + (q - p) * 6 * t;
    if (t < 0.5)    return q;
    if (t < 0.6667) return p + (q - p) * (0.6667 - t) * 6;
    return p;
  }
  return [f(h + 0.3333), f(h), f(h - 0.3333)];
}

function smoothNoise(x: number, y: number): number {
  function hash(ix: number, iy: number): number {
    let h = (ix * 1619 + iy * 31337 + 6791) | 0;
    h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
    h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
    return ((h >>> 0) / 0x100000000);
  }
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy);
  return (hash(ix,iy)   * (1-ux) + hash(ix+1,iy)   * ux) * (1-uy)
       + (hash(ix,iy+1) * (1-ux) + hash(ix+1,iy+1) * ux) * uy;
}

export function buildSimScene(
  engine: EngineContext,
  scene: SceneContext,
  canvas: HTMLCanvasElement
): SimHandle {
  const camera = createArcRotateCamera(-Math.PI * 0.25, Math.PI * 0.3, WorldSize * 1.6, { x: 0, y: 0, z: 0 });
  scene.camera = camera;
  const detach = attachControl(camera, canvas, scene);

  addToScene(scene, createHemisphericLight([0, 1, 0], 0.4));
  addToScene(scene, createDirectionalLight([-0.5, -1, 0.4]));

  // ── CPU column mesh ───────────────────────────────────────────────────────
  const columnMesh = createBox(engine, 1);
  columnMesh.material = createPbrMaterial({ baseColorFactor: [0.8, 0.8, 0.8, 1], metallicFactor: 0.2, roughnessFactor: 0.7 });
  addToScene(scene, columnMesh);

  const colBuf = new AgentBuffer(MaxColumns);
  colBuf.attach(engine, columnMesh);

  // ── Seed marker spheres (shared CPU/GPU — always CPU-driven) ──────────────
  // Sized at GpuMaxSeeds so markers work at full GPU seed count.
  const seedMesh = createSphere(engine, { segments: 5, diameter: 0.9 });
  seedMesh.material = createPbrMaterial({ baseColorFactor: [1, 1, 1, 1], metallicFactor: 0.9, roughnessFactor: 0.1 });
  addToScene(scene, seedMesh);

  const markerBuf = new AgentBuffer(GpuMaxSeeds);
  markerBuf.attach(engine, seedMesh);

  // ── GPU path ──────────────────────────────────────────────────────────────
  const gpu          = new VoronoiGpu(engine, scene);
  const gpuAvailable = gpu.ok;

  // ── Reactive params ───────────────────────────────────────────────────────
  const params = reactive({
    seeds:       20,
    speed:       0.8,
    lloyd:       false,
    gridRes:     96,
    heightMode:  "ridge",
    heightScale: 8,
    gpuMode:     false,
  });

  const seedsRef   = ref(0);
  const columnsRef = ref(0);

  // Reactive sliders whose max depends on mode.
  const gridSlider = reactive<SliderParam>({
    type:  "slider",
    key:   "gridRes",
    label: "Grid Res",
    min:   32,
    max:   MaxGrid,
    step:  8,
  });
  const seedSlider = reactive<SliderParam>({
    type:  "slider",
    key:   "seeds",
    label: "Seeds",
    min:   2,
    max:   MaxSeeds,
    step:  1,
  });

  // ── Seed state (sized at GpuMaxSeeds for both paths) ─────────────────────
  const sx   = new Float32Array(GpuMaxSeeds);
  const sy   = new Float32Array(GpuMaxSeeds);
  const svx  = new Float32Array(GpuMaxSeeds);
  const svy  = new Float32Array(GpuMaxSeeds);
  const hues = new Float32Array(GpuMaxSeeds);

  function spawnSeeds() {
    const n = params.seeds;
    for (let i = 0; i < n; i++) {
      sx[i]   = Math.random();
      sy[i]   = Math.random();
      const a = Math.random() * Math.PI * 2;
      svx[i]  = Math.cos(a) * 0.15;
      svy[i]  = Math.sin(a) * 0.15;
      hues[i] = i / n;
    }
  }
  spawnSeeds();

  // ── Lloyd relaxation ──────────────────────────────────────────────────────
  const lloydAccX = new Float32Array(GpuMaxSeeds);
  const lloydAccY = new Float32Array(GpuMaxSeeds);
  const lloydCnt  = new Int32Array(GpuMaxSeeds);

  function lloydRelaxation(G: number) {
    const n = params.seeds;
    lloydAccX.fill(0, 0, n);
    lloydAccY.fill(0, 0, n);
    lloydCnt.fill(0, 0, n);
    for (let gy = 0; gy < G; gy++) {
      const fy = (gy + 0.5) / G;
      for (let gx = 0; gx < G; gx++) {
        const fx = (gx + 0.5) / G;
        let minD = Infinity, best = 0;
        for (let i = 0; i < n; i++) {
          const dx = fx - sx[i], dy = fy - sy[i];
          const d2 = dx*dx + dy*dy;
          if (d2 < minD) { minD = d2; best = i; }
        }
        lloydAccX[best] += fx;
        lloydAccY[best] += fy;
        lloydCnt[best]++;
      }
    }
    for (let i = 0; i < n; i++) {
      if (lloydCnt[i] > 0) {
        sx[i] = lloydAccX[i] / lloydCnt[i];
        sy[i] = lloydAccY[i] / lloydCnt[i];
      }
    }
  }

  // ── CPU working arrays ────────────────────────────────────────────────────
  const cellId     = new Int32Array(MaxColumns);
  const cellD1     = new Float32Array(MaxColumns);
  const cellD2     = new Float32Array(MaxColumns);
  const regionArea = new Int32Array(GpuMaxSeeds);
  // Per-seed mesa heights (normalized), reused by both CPU and GPU paths.
  const mesaH      = new Float32Array(GpuMaxSeeds);

  let simTime  = 0;
  let prevGpu  = false;
  let prevGrid = params.gridRes;

  // ── Visibility ────────────────────────────────────────────────────────────
  // CPU columnMesh vs GPU gpu.columnMesh; seedMesh (markers) always visible.
  function applyVisibility(): void {
    const g = params.gpuMode && gpuAvailable;
    setMeshVisible(columnMesh,      !g);
    setMeshVisible(gpu.columnMesh,   g);
  }

  function switchToGpu(): void {
    if (!gpuAvailable) { params.gpuMode = false; return; }
    gridSlider.max = GpuMaxGrid;
    seedSlider.max = GpuMaxSeeds;
    // Hide CPU columns.
    setThinInstanceCount(columnMesh, 0);
    invalidateRenderBundles(engine);
    gpu.setGrid(params.gridRes);
    prevGrid = params.gridRes;
    applyVisibility();
  }

  function switchToCpu(): void {
    gridSlider.max = MaxGrid;
    seedSlider.max = MaxSeeds;
    if (params.gridRes > MaxGrid)  params.gridRes  = MaxGrid;
    if (params.seeds   > MaxSeeds) params.seeds    = MaxSeeds;
    if (gpuAvailable) gpu.setGrid(0);
    applyVisibility();
  }

  // Initial visibility: GPU mesh hidden at startup.
  applyVisibility();

  // ── CPU buildLandscape (unchanged) ───────────────────────────────────────
  function buildLandscape() {
    const G  = params.gridRes;
    const n  = params.seeds;
    const G2 = G * G;
    const cellW     = WorldSize / G;
    const cellNorm  = Math.max(0.5 / Math.sqrt(n), 0.05);

    for (let gy = 0; gy < G; gy++) {
      const fy = (gy + 0.5) / G;
      for (let gx = 0; gx < G; gx++) {
        const fx = (gx + 0.5) / G;
        let d1 = Infinity, d2 = Infinity, id = 0;
        for (let i = 0; i < n; i++) {
          const dx = fx - sx[i], dy = fy - sy[i];
          const d2_ = dx*dx + dy*dy;
          if (d2_ < d1) { d2 = d1; d1 = d2_; id = i; }
          else if (d2_ < d2) { d2 = d2_; }
        }
        const idx = gy * G + gx;
        cellId[idx]  = id;
        cellD1[idx]  = Math.sqrt(d1);
        cellD2[idx]  = Math.sqrt(d2);
      }
    }

    let maxArea = 1;
    if (params.heightMode === "mesa") {
      regionArea.fill(0, 0, n);
      for (let k = 0; k < G2; k++) regionArea[cellId[k]]++;
      for (let i = 0; i < n; i++) if (regionArea[i] > maxArea) maxArea = regionArea[i];
    }

    const hs   = params.heightScale;
    const mode = params.heightMode;

    for (let gy = 0; gy < G; gy++) {
      const fy = (gy + 0.5) / G;
      const wz = (fy - 0.5) * WorldSize;
      for (let gx = 0; gx < G; gx++) {
        const fx = (gx + 0.5) / G;
        const wx  = (fx - 0.5) * WorldSize;
        const idx = gy * G + gx;
        const id  = cellId[idx];
        const d1  = cellD1[idx];
        const d2  = cellD2[idx];

        let hNorm: number;
        if (mode === "ridge") {
          hNorm = Math.min((d2 - d1) / cellNorm, 1);
        } else if (mode === "cone") {
          hNorm = Math.max(1 - d1 / cellNorm, 0);
        } else if (mode === "mesa") {
          hNorm = regionArea[id] / maxArea;
        } else {
          const rOff = hues[id] * 3.7;
          hNorm = Math.max(0, Math.min(1,
            0.5 + 0.5 * (smoothNoise(fx * 4 + rOff, fy * 4 + simTime * 0.2) - 0.5) * 2,
          ));
        }

        const h = Math.max(hNorm * hs, 0.05);
        const l = 0.32 + 0.32 * hNorm;
        const [r, g, b] = hslToRgb(hues[id], 0.72, l);
        colBuf.writeTRS(idx, wx, h * 0.5, wz, cellW * 0.96, h, cellW * 0.96);
        colBuf.writeColor(idx, r, g, b);
      }
    }

    colBuf.commit(columnMesh, G2);
    columnsRef.value = G2;

    // Seed markers at the column under each seed's position.
    const hs2 = params.heightScale;
    for (let i = 0; i < n; i++) {
      const wx = (sx[i] - 0.5) * WorldSize;
      const wz = (sy[i] - 0.5) * WorldSize;
      const gx  = Math.min(Math.max(Math.floor(sx[i] * G), 0), G - 1);
      const gy  = Math.min(Math.max(Math.floor(sy[i] * G), 0), G - 1);
      const idx = gy * G + gx;
      const d1  = cellD1[idx];
      const d2  = cellD2[idx];
      let hNormI: number;
      if (mode === "ridge") {
        hNormI = Math.min((d2 - d1) / cellNorm, 1);
      } else if (mode === "cone") {
        hNormI = Math.max(1 - d1 / cellNorm, 0);
      } else if (mode === "mesa") {
        hNormI = regionArea[i] / maxArea;
      } else {
        const rOff = hues[i] * 3.7;
        hNormI = Math.max(0, Math.min(1,
          0.5 + 0.5 * (smoothNoise(sx[i] * 4 + rOff, sy[i] * 4 + simTime * 0.2) - 0.5) * 2,
        ));
      }
      const wy = Math.max(hNormI * hs2, 0.05) + 0.6;
      markerBuf.writeScale(i, wx, wy, wz, 0.7);
      const [r, g, b] = hslToRgb(hues[i], 0.5, 0.88);
      markerBuf.writeColor(i, r, g, b);
    }
    markerBuf.commit(seedMesh, n);
    seedsRef.value = n;
  }

  // ── GPU marker heights (O(seeds²) only, called each GPU frame) ───────────
  // For each seed, find its 2nd-nearest neighbour to compute the height at its
  // own position — the same formula the GPU shader uses for that cell.
  function buildGpuMarkers(n: number, cellNorm: number, maxArea: number) {
    const mode = params.heightMode;
    const hs   = params.heightScale;
    for (let i = 0; i < n; i++) {
      let d1 = Infinity, d2 = Infinity;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const dx = sx[i] - sx[j], dy = sy[i] - sy[j];
        const d2_ = dx*dx + dy*dy;
        if (d2_ < d1) { d2 = d1; d1 = d2_; }
        else if (d2_ < d2) { d2 = d2_; }
      }
      const dist1 = Math.sqrt(d1);
      const dist2 = d2 === Infinity ? dist1 : Math.sqrt(d2);
      let hNorm: number;
      if (mode === "ridge") {
        hNorm = Math.min((dist2 - dist1) / cellNorm, 1);
      } else if (mode === "cone") {
        hNorm = Math.max(1 - dist1 / cellNorm, 0);
      } else if (mode === "mesa") {
        hNorm = mesaH[i];
      } else {
        const rOff = hues[i] * 3.7;
        hNorm = Math.max(0, Math.min(1,
          0.5 + 0.5 * (smoothNoise(sx[i] * 4 + rOff, sy[i] * 4 + simTime * 0.2) - 0.5) * 2,
        ));
      }
      const wx = (sx[i] - 0.5) * WorldSize;
      const wz = (sy[i] - 0.5) * WorldSize;
      const wy = Math.max(hNorm * hs, 0.05) + 0.6;
      markerBuf.writeScale(i, wx, wy, wz, 0.7);
      const [r, g, bl] = hslToRgb(hues[i], 0.5, 0.88);
      markerBuf.writeColor(i, r, g, bl);
    }
    markerBuf.commit(seedMesh, n);
    seedsRef.value = n;
    void maxArea;
  }

  // ── Per-frame update ──────────────────────────────────────────────────────
  function update(dt: number) {
    const n     = params.seeds;
    const s     = dt * 0.001 * params.speed;
    const isGpu = params.gpuMode;
    simTime += dt * 0.001;

    // React to mode flip.
    if (isGpu !== prevGpu) {
      prevGpu = isGpu;
      if (isGpu) switchToGpu(); else switchToCpu();
    }

    // Animate seeds or Lloyd-relax.
    if (params.lloyd) {
      // In GPU mode use a coarse grid to keep CPU work cheap.
      lloydRelaxation(isGpu ? LloydGrid : params.gridRes);
    } else {
      for (let i = 0; i < n; i++) {
        svx[i] += (Math.random() - 0.5) * 0.02;
        svy[i] += (Math.random() - 0.5) * 0.02;
        const spd = Math.sqrt(svx[i]*svx[i] + svy[i]*svy[i]);
        if (spd > 0.3) { svx[i] *= 0.3 / spd; svy[i] *= 0.3 / spd; }
        sx[i] += svx[i] * s;
        sy[i] += svy[i] * s;
        if (sx[i] < 0) { sx[i] = -sx[i]; svx[i] = -svx[i]; }
        if (sx[i] > 1) { sx[i] = 2 - sx[i]; svx[i] = -svx[i]; }
        if (sy[i] < 0) { sy[i] = -sy[i]; svy[i] = -svy[i]; }
        if (sy[i] > 1) { sy[i] = 2 - sy[i]; svy[i] = -svy[i]; }
      }
    }

    if (!isGpu) {
      buildLandscape();
    } else {
      // GPU path: compute mesa heights on CPU (coarse grid, only in mesa mode).
      const G = params.gridRes;
      const cellNorm = Math.max(0.5 / Math.sqrt(n), 0.05);
      let maxArea = 1;

      if (params.heightMode === "mesa") {
        regionArea.fill(0, 0, n);
        for (let gy = 0; gy < LloydGrid; gy++) {
          for (let gx = 0; gx < LloydGrid; gx++) {
            const fx = (gx + 0.5) / LloydGrid;
            const fy = (gy + 0.5) / LloydGrid;
            let minD = Infinity, best = 0;
            for (let i = 0; i < n; i++) {
              const dx = fx - sx[i], dy = fy - sy[i];
              const d2 = dx*dx + dy*dy;
              if (d2 < minD) { minD = d2; best = i; }
            }
            regionArea[best]++;
          }
        }
        for (let i = 0; i < n; i++) if (regionArea[i] > maxArea) maxArea = regionArea[i];
        for (let i = 0; i < n; i++) mesaH[i] = regionArea[i] / maxArea;
      } else {
        mesaH.fill(0, 0, n);
      }

      // Upload seeds and dispatch GPU compute.
      gpu.uploadSeeds(sx, sy, hues, mesaH, n);

      if (G !== prevGrid) {
        prevGrid = G;
        gpu.setGrid(G);
      }

      gpu.dispatch({
        gridRes:     G,
        seedCount:   n,
        heightMode:  HEIGHT_MODE[params.heightMode] ?? 0,
        heightScale: params.heightScale,
        cellNorm,
        worldSize:   WorldSize,
        time:        simTime,
      });

      columnsRef.value = G * G;
      buildGpuMarkers(n, cellNorm, maxArea);
    }
  }

  function reset() { spawnSeeds(); }

  return {
    params,
    schema: [
      {
        type:  "toggle",
        key:   "gpuMode",
        label: gpuAvailable ? "GPU Mode" : "GPU Mode (unavailable)",
      },
      seedSlider,
      { type: "slider", key: "speed",       label: "Speed",        min: 0,  max: 3,   step: 0.1 },
      { type: "select", key: "heightMode",  label: "Height Mode",  options: ["ridge", "cone", "mesa", "noise"] },
      { type: "slider", key: "heightScale", label: "Height Scale", min: 1,  max: 16,  step: 0.5 },
      gridSlider,
      { type: "toggle", key: "lloyd",       label: "Lloyd Relax" },
      { type: "button", key: "_respawn",    label: "Respawn",      action: spawnSeeds },
    ],
    readouts: { seeds: seedsRef, columns: columnsRef },
    update,
    reset,
    detach,
  };
}
