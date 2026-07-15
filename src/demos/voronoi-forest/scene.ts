import { reactive, ref } from "vue";
import {
  addToScene,
  attachControl,
  createArcRotateCamera,
  createHemisphericLight,
  createDirectionalLight,
  createCylinder,
  createPlane,
  createSphere,
  createPbrMaterial,
  createStandardMaterial,
  createTexture2DFromPixels,
  updateTexture2DFromPixels,
} from "@babylonjs/lite";
import type { EngineContext, SceneContext } from "@babylonjs/lite";
import { AgentBuffer } from "@/lib/agents";
import type { SimHandle } from "@/types/sim";

/** Voronoi + L-Systems:
 *  - Voronoi partitions the world into N biomes (each with a unique species)
 *  - Each biome grows a small L-system tree (precomputed segments)
 *  - Trees are placed as thin-instanced cylinder segments within their biome cell
 *  - Seeds animate slowly; Lloyd relaxation keeps cells even
 */

const TexN = 128;
const TexPixels = TexN * TexN;
const MaxSeeds = 12;
const MaxTreeSegments = 3000;
const WorldSize = 40;

interface Biome {
  hue: number;
  branchAngle: number;   // degrees
  shrink: number;
  rule: string;
  axiom: string;
  iters: number;
}

const Biomes: Biome[] = [
  { hue: 0.3,  branchAngle: 25, shrink: 0.65, axiom: "X", rule: "X→F+[[X]-X]-F[-FX]+X;F→FF", iters: 3 },
  { hue: 0.08, branchAngle: 35, shrink: 0.7,  axiom: "F", rule: "F→FF+[+F-F-F]-[-F+F+F]", iters: 3 },
  { hue: 0.45, branchAngle: 20, shrink: 0.75, axiom: "X", rule: "X→F[+X]F[-X]+X;F→FF", iters: 4 },
  { hue: 0.15, branchAngle: 30, shrink: 0.6,  axiom: "X", rule: "X→F[+X][-X]FX;F→FF", iters: 4 },
  { hue: 0.55, branchAngle: 18, shrink: 0.8,  axiom: "F", rule: "F→F[+F]F[-F][F]", iters: 4 },
  { hue: 0.25, branchAngle: 40, shrink: 0.65, axiom: "X", rule: "X→F+[[X]-X]-F[-FX]+X;F→FF", iters: 3 },
  { hue: 0.60, branchAngle: 22, shrink: 0.72, axiom: "F", rule: "F→FF-[-F+F+F]+[+F-F-F]", iters: 3 },
  { hue: 0.10, branchAngle: 28, shrink: 0.68, axiom: "X", rule: "X→F[-X][+X]FX;F→FF", iters: 4 },
  { hue: 0.38, branchAngle: 32, shrink: 0.66, axiom: "F", rule: "F→F[+F[-F]F]F[-F[+F]F]F", iters: 3 },
  { hue: 0.20, branchAngle: 15, shrink: 0.78, axiom: "X", rule: "X→F[+X]F[-X]FX;F→FF", iters: 4 },
  { hue: 0.50, branchAngle: 45, shrink: 0.60, axiom: "X", rule: "X→F+[[X]-X]-F[-FX]+X;F→FF", iters: 3 },
  { hue: 0.70, branchAngle: 25, shrink: 0.70, axiom: "F", rule: "F→FF+[+F-F-F]-[-F+F+F]", iters: 3 },
];

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  function f(t: number) {
    if (t < 0) t += 1; if (t > 1) t -= 1;
    if (t < 0.1667) return p + (q - p) * 6 * t;
    if (t < 0.5) return q;
    if (t < 0.6667) return p + (q - p) * (0.6667 - t) * 6;
    return p;
  }
  return [f(h + 0.3333), f(h), f(h - 0.3333)];
}

function expandLSystem(axiom: string, ruleStr: string, iters: number): string {
  const rules: Record<string, string> = {};
  for (const part of ruleStr.split(";")) {
    const [lhs, rhs] = part.split("→");
    if (lhs && rhs) rules[lhs.trim()] = rhs.trim();
  }
  let s = axiom;
  for (let i = 0; i < iters; i++) {
    let ns = "";
    for (const c of s) ns += rules[c] ?? c;
    s = ns;
    if (s.length > 20000) break;
  }
  return s;
}

interface Seg { mx: number; my: number; depth: number; qy: number; qw: number; len: number; wid: number; }

function buildTreeSegments(biome: Biome): Seg[] {
  const str = expandLSystem(biome.axiom, biome.rule, biome.iters);
  const segs: Seg[] = [];
  const ang = biome.branchAngle * (Math.PI / 180);

  interface TState { x: number; y: number; dir: number; len: number; wid: number; depth: number; }
  const stack: TState[] = [];
  const state: TState = { x: 0, y: 0, dir: Math.PI * 0.5, len: 1.2, wid: 0.12, depth: 0 };

  for (const ch of str) {
    if (segs.length >= 400) break;
    if (ch === "F") {
      const nx = state.x + Math.cos(state.dir) * state.len;
      const ny = state.y + Math.sin(state.dir) * state.len;
      const cx = (state.x + nx) * 0.5, cy = (state.y + ny) * 0.5;
      const halfAng = (Math.PI * 0.5 - state.dir) * 0.5;
      segs.push({ mx: cx, my: cy, depth: state.depth, qy: Math.sin(halfAng), qw: Math.cos(halfAng), len: state.len, wid: state.wid });
      state.x = nx; state.y = ny;
    } else if (ch === "+") {
      state.dir += ang;
    } else if (ch === "-") {
      state.dir -= ang;
    } else if (ch === "[") {
      stack.push({ ...state });
      state.len *= biome.shrink; state.wid *= biome.shrink; state.depth++;
    } else if (ch === "]") {
      const s2 = stack.pop(); if (s2) Object.assign(state, s2);
    }
  }
  return segs;
}

export function buildSimScene(
  engine: EngineContext,
  scene: SceneContext,
  canvas: HTMLCanvasElement
): SimHandle {
  const camera = createArcRotateCamera(-Math.PI * 0.4, Math.PI * 0.3, 60, { x: 0, y: 0, z: 0 });
  scene.camera = camera;
  const detach = attachControl(camera, canvas, scene);

  addToScene(scene, createHemisphericLight([0, 1, 0], 0.65));
  addToScene(scene, createDirectionalLight([-0.4, -1, 0.3]));

  // Ground texture
  const pixels = new Uint8Array(TexPixels * 4);
  const tex = createTexture2DFromPixels(engine, pixels, TexN, TexN, { format: "rgba8unorm", generateMipmaps: false });
  const ground = createPlane(engine, { width: WorldSize, height: WorldSize });
  const groundMat = createStandardMaterial();
  groundMat.diffuseTexture = tex;
  ground.material = groundMat;
  addToScene(scene, ground);

  // Tree segments — one shared instanced cylinder
  const segMesh = createCylinder(engine, { height: 1, diameter: 1, tessellation: 5 });
  segMesh.material = createPbrMaterial({ baseColorFactor: [0.3, 0.6, 0.1, 1], metallicFactor: 0, roughnessFactor: 0.9 });
  addToScene(scene, segMesh);

  // Seed markers
  const seedMesh = createSphere(engine, { segments: 4, diameter: 0.8 });
  seedMesh.material = createPbrMaterial({ baseColorFactor: [1, 1, 1, 1], metallicFactor: 0.7, roughnessFactor: 0.2 });
  addToScene(scene, seedMesh);

  const params = reactive({
    seeds: 8,
    speed: 0.3,
    lloyd: true,
    treeScale: 1.0,
  });

  const segCountRef = ref(0);

  // Voronoi seed data
  const sx = new Float32Array(MaxSeeds);
  const sy = new Float32Array(MaxSeeds);
  const svx = new Float32Array(MaxSeeds);
  const svy = new Float32Array(MaxSeeds);
  const nearest = new Int32Array(TexPixels);

  const segBuf = new AgentBuffer(MaxTreeSegments);
  const seedBuf = new AgentBuffer(MaxSeeds);
  segBuf.attach(engine, segMesh);
  seedBuf.attach(engine, seedMesh);

  // Precomputed tree segments per biome
  const treeSegments: Seg[][] = Biomes.map(b => buildTreeSegments(b));

  function spawnSeeds() {
    const n = params.seeds;
    for (let i = 0; i < n; i++) {
      sx[i] = 0.1 + Math.random() * 0.8;
      sy[i] = 0.1 + Math.random() * 0.8;
      const a = Math.random() * Math.PI * 2;
      svx[i] = Math.cos(a) * 0.08;
      svy[i] = Math.sin(a) * 0.08;
    }
  }
  spawnSeeds();

  function computeVoronoi(n: number) {
    for (let py2 = 0; py2 < TexN; py2++) {
      const fy = py2 / TexN;
      for (let px2 = 0; px2 < TexN; px2++) {
        const fx = px2 / TexN;
        let minD = Infinity, best = 0;
        for (let i = 0; i < n; i++) {
          const dx = fx - sx[i], dy = fy - sy[i];
          const d = dx * dx + dy * dy;
          if (d < minD) { minD = d; best = i; }
        }
        nearest[px2 + py2 * TexN] = best;
      }
    }
  }

  function lloydRelax(n: number) {
    computeVoronoi(n);
    const accX = new Float32Array(n), accY = new Float32Array(n);
    const cnt = new Int32Array(n);
    for (let i = 0; i < TexPixels; i++) {
      const id = nearest[i];
      accX[id] += (i % TexN) / TexN;
      accY[id] += Math.floor(i / TexN) / TexN;
      cnt[id]++;
    }
    for (let i = 0; i < n; i++) {
      if (cnt[i] > 0) { sx[i] = accX[i] / cnt[i]; sy[i] = accY[i] / cnt[i]; }
    }
  }

  function writeScene(n: number) {
    // Texture
    computeVoronoi(n);
    for (let i = 0; i < TexPixels; i++) {
      const id = nearest[i];
      const biome = Biomes[id % Biomes.length];
      const [r, g, b] = hslToRgb(biome.hue, 0.5, 0.32);
      const o = i * 4;
      pixels[o] = Math.round(r * 255); pixels[o+1] = Math.round(g * 255); pixels[o+2] = Math.round(b * 255); pixels[o+3] = 255;
    }
    updateTexture2DFromPixels(engine, tex, pixels);

    // Tree segments per biome, placed at seed position
    let tc = 0;
    const sc = params.treeScale;

    for (let i = 0; i < n && tc < MaxTreeSegments - 100; i++) {
      const biome = Biomes[i % Biomes.length];
      const segs = treeSegments[i % treeSegments.length];
      const worldX = (sx[i] - 0.5) * WorldSize;
      const worldZ = (sy[i] - 0.5) * WorldSize;

      for (const seg of segs) {
        if (tc >= MaxTreeSegments) break;
        const wx = worldX + seg.mx * sc;
        const wy = seg.my * sc;
        const [r, g, b] = hslToRgb(biome.hue, 0.65, 0.25 + seg.depth * 0.04);
        segBuf.writeTransform(tc, wx, wy, worldZ, 0, seg.qy, 0, seg.qw, seg.wid * sc, seg.len * sc, seg.wid * sc);
        segBuf.writeColor(tc, r, g, b);
        tc++;
      }
    }
    segCountRef.value = tc;
    segBuf.commit(segMesh, tc);

    // Seed markers
    for (let i = 0; i < n; i++) {
      const wx = (sx[i] - 0.5) * WorldSize;
      const wz = (sy[i] - 0.5) * WorldSize;
      seedBuf.writeScale(i, wx, 0.5, wz, 0.5);
      const [r, g, b] = hslToRgb(Biomes[i % Biomes.length].hue, 0.9, 0.7);
      seedBuf.writeColor(i, r, g, b);
    }
    seedBuf.commit(seedMesh, n);
  }

  function update(dt: number) {
    const n = params.seeds;
    const s = dt * 0.001 * params.speed;

    if (params.lloyd) {
      lloydRelax(n);
    } else {
      for (let i = 0; i < n; i++) {
        svx[i] += (Math.random() - 0.5) * 0.01;
        svy[i] += (Math.random() - 0.5) * 0.01;
        const spd = Math.sqrt(svx[i] ** 2 + svy[i] ** 2);
        if (spd > 0.15) { svx[i] *= 0.15 / spd; svy[i] *= 0.15 / spd; }
        sx[i] = Math.max(0.05, Math.min(0.95, sx[i] + svx[i] * s));
        sy[i] = Math.max(0.05, Math.min(0.95, sy[i] + svy[i] * s));
      }
    }

    writeScene(n);
  }

  function reset() { spawnSeeds(); }

  return {
    params,
    schema: [
      { type: "slider", key: "seeds", label: "Biomes", min: 2, max: MaxSeeds, step: 1 },
      { type: "slider", key: "treeScale", label: "Tree Scale", min: 0.3, max: 2.5, step: 0.1 },
      { type: "slider", key: "speed", label: "Drift Speed", min: 0, max: 1, step: 0.05 },
      { type: "toggle", key: "lloyd", label: "Lloyd Relaxation" },
      { type: "button", key: "_respawn", label: "Respawn Biomes", action: spawnSeeds },
    ],
    readouts: { segments: segCountRef },
    update,
    reset,
    detach,
  };
}
