import { reactive, ref } from "vue";
import {
  addToScene,
  attachControl,
  createArcRotateCamera,
  createHemisphericLight,
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

const TexN = 256;
const MaxSeeds = 64;
const PixelCount = TexN * TexN;

function hue2rgb(p: number, q: number, t: number): number {
  if (t < 0) t += 1;
  if (t > 1) t -= 1;
  if (t < 0.1667) return p + (q - p) * 6 * t;
  if (t < 0.5) return q;
  if (t < 0.6667) return p + (q - p) * (0.6667 - t) * 6;
  return p;
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [hue2rgb(p, q, h + 0.3333), hue2rgb(p, q, h), hue2rgb(p, q, h - 0.3333)];
}

export function buildSimScene(
  engine: EngineContext,
  scene: SceneContext,
  canvas: HTMLCanvasElement
): SimHandle {
  // createPlane is a vertical XY plane (normal -Z); view it front-on from -Z.
  const camera = createArcRotateCamera(-Math.PI / 2, Math.PI * 0.5, 42, { x: 0, y: 0, z: 0 });
  scene.camera = camera;
  const detach = attachControl(camera, canvas, scene);

  addToScene(scene, createHemisphericLight([0, 1, 0], 1));

  const params = reactive({
    seeds: 20,
    speed: 0.8,
    lloyd: false,
  });

  const seedsRef = ref(0);

  const pixels = new Uint8Array(PixelCount * 4);
  const tex = createTexture2DFromPixels(engine, pixels, TexN, TexN, { format: "rgba8unorm", generateMipmaps: false });

  const plane = createPlane(engine, { width: 36, height: 36 });
  const mat = createStandardMaterial();
  mat.emissiveTexture = tex;      // self-lit so the regions read at full color
  mat.backFaceCulling = false;
  plane.material = mat;
  addToScene(scene, plane);

  // Seed marker spheres
  const seedMesh = createSphere(engine, { segments: 5, diameter: 0.7 });
  seedMesh.material = createPbrMaterial({ baseColorFactor: [1, 1, 1, 1], metallicFactor: 0.8, roughnessFactor: 0.2 });
  addToScene(scene, seedMesh);

  const buf = new AgentBuffer(MaxSeeds);
  buf.attach(engine, seedMesh);

  // Seed positions [0..1]
  const sx = new Float32Array(MaxSeeds);
  const sy = new Float32Array(MaxSeeds);
  const svx = new Float32Array(MaxSeeds);
  const svy = new Float32Array(MaxSeeds);
  const hues = new Float32Array(MaxSeeds);

  // Nearest-seed lookup (precomputed per pixel)
  const nearest = new Int32Array(PixelCount);

  function spawnSeeds() {
    const n = params.seeds;
    for (let i = 0; i < n; i++) {
      sx[i] = Math.random();
      sy[i] = Math.random();
      const angle = Math.random() * Math.PI * 2;
      svx[i] = Math.cos(angle) * 0.15;
      svy[i] = Math.sin(angle) * 0.15;
      hues[i] = i / n;
    }
  }
  spawnSeeds();

  function computeVoronoi() {
    const n = params.seeds;
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

  function writeTexture() {
    const n = params.seeds;
    computeVoronoi();
    for (let i = 0; i < PixelCount; i++) {
      const id = nearest[i];
      const [r, g, b] = hslToRgb(hues[id], 0.75, 0.45);
      const o = i * 4;
      pixels[o]   = Math.round(r * 255);
      pixels[o+1] = Math.round(g * 255);
      pixels[o+2] = Math.round(b * 255);
      pixels[o+3] = 255;
    }
    updateTexture2DFromPixels(engine, tex, pixels);

    // Write seed markers ON the plane (XY), aligned with their region, popped
    // slightly toward the camera (-Z) so each dot sits at the center of its cell.
    for (let i = 0; i < n; i++) {
      const wx = (sx[i] - 0.5) * 36;
      const wy = (sy[i] - 0.5) * 36;
      buf.writeScale(i, wx, wy, -0.6, 0.6);
      // Bright near-white core tinted with the region hue, so the site reads
      // clearly against its matching-colored cell.
      const [r, g, b] = hslToRgb(hues[i], 0.5, 0.9);
      buf.writeColor(i, r, g, b);
    }
    buf.commit(seedMesh, n);
    seedsRef.value = n;
  }

  function lloydRelaxation() {
    const n = params.seeds;
    computeVoronoi();
    const accX = new Float32Array(n);
    const accY = new Float32Array(n);
    const cnt = new Int32Array(n);
    for (let i = 0; i < PixelCount; i++) {
      const id = nearest[i];
      accX[id] += (i % TexN) / TexN;
      accY[id] += Math.floor(i / TexN) / TexN;
      cnt[id]++;
    }
    for (let i = 0; i < n; i++) {
      if (cnt[i] > 0) {
        sx[i] = accX[i] / cnt[i];
        sy[i] = accY[i] / cnt[i];
      }
    }
  }

  function update(dt: number) {
    const n = params.seeds;
    const s = dt * 0.001 * params.speed;

    if (params.lloyd) {
      lloydRelaxation();
    } else {
      // Animate seeds
      for (let i = 0; i < n; i++) {
        svx[i] += (Math.random() - 0.5) * 0.02;
        svy[i] += (Math.random() - 0.5) * 0.02;
        const spd = Math.sqrt(svx[i] * svx[i] + svy[i] * svy[i]);
        if (spd > 0.3) { svx[i] *= 0.3 / spd; svy[i] *= 0.3 / spd; }
        sx[i] += svx[i] * s;
        sy[i] += svy[i] * s;
        if (sx[i] < 0) { sx[i] = -sx[i]; svx[i] = -svx[i]; }
        if (sx[i] > 1) { sx[i] = 2 - sx[i]; svx[i] = -svx[i]; }
        if (sy[i] < 0) { sy[i] = -sy[i]; svy[i] = -svy[i]; }
        if (sy[i] > 1) { sy[i] = 2 - sy[i]; svy[i] = -svy[i]; }
      }
    }

    writeTexture();
  }

  function reset() { spawnSeeds(); }

  return {
    params,
    schema: [
      { type: "slider", key: "seeds", label: "Seed Count", min: 2, max: MaxSeeds, step: 1 },
      { type: "slider", key: "speed", label: "Speed", min: 0, max: 3, step: 0.1 },
      { type: "toggle", key: "lloyd", label: "Lloyd Relaxation" },
      { type: "button", key: "_respawn", label: "Respawn", action: spawnSeeds },
    ],
    readouts: { seeds: seedsRef },
    update,
    reset,
    detach,
  };
}
