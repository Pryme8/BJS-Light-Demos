import { reactive, ref } from "vue";
import {
  addToScene,
  attachControl,
  createArcRotateCamera,
  createCapsule,
  createHemisphericLight,
  createPbrMaterial,
} from "@babylonjs/lite";
import type { EngineContext, SceneContext } from "@babylonjs/lite";
import { AgentBuffer, upToDirQuat } from "@/lib/agents";
import type { SimHandle } from "@/types/sim";

const Capacity = 5000;
const Bound = 20;

export function buildSimScene(
  engine: EngineContext,
  scene: SceneContext,
  canvas: HTMLCanvasElement
): SimHandle {
  const camera = createArcRotateCamera(-Math.PI * 0.5, Math.PI * 0.35, 55, { x: 0, y: 0, z: 0 });
  scene.camera = camera;
  const detach = attachControl(camera, canvas, scene);

  addToScene(scene, createHemisphericLight([0, 1, 0], 0.9));

  const mesh = createCapsule(engine, { height: 1.2, radius: 0.35, tessellation: 6 });
  mesh.material = createPbrMaterial({ baseColorFactor: [0.05, 0.8, 1, 1], metallicFactor: 0.2, roughnessFactor: 0.5 });
  addToScene(scene, mesh);

  const buf = new AgentBuffer(Capacity);
  buf.attach(engine, mesh);

  const params = reactive({
    count: 300,
    speed: 6,
    separation: 1.8,
    alignment: 1.0,
    cohesion: 0.6,
    radius: 5.0,
    separationRadius: 2.0,
  });

  // Agent state
  const px = new Float32Array(Capacity);
  const py = new Float32Array(Capacity);
  const pz = new Float32Array(Capacity);
  const vx = new Float32Array(Capacity);
  const vy = new Float32Array(Capacity);
  const vz = new Float32Array(Capacity);

  const agentsRef = ref(params.count);
  let prevCount = params.count;

  function spawnOne(i: number) {
    px[i] = (Math.random() - 0.5) * Bound * 1.5;
    py[i] = (Math.random() - 0.5) * Bound;
    pz[i] = (Math.random() - 0.5) * Bound * 1.5;
    const angle = Math.random() * Math.PI * 2;
    const pitch = (Math.random() - 0.5) * Math.PI;
    vx[i] = Math.cos(angle) * Math.cos(pitch);
    vy[i] = Math.sin(pitch);
    vz[i] = Math.sin(angle) * Math.cos(pitch);
  }

  function spawnAll() {
    for (let i = 0; i < Capacity; i++) spawnOne(i);
  }
  spawnAll();

  // Preallocate neighbor accumulators
  const sx2 = new Float32Array(Capacity), sy2 = new Float32Array(Capacity), sz2 = new Float32Array(Capacity);
  const ax = new Float32Array(Capacity), ay = new Float32Array(Capacity), az = new Float32Array(Capacity);
  const cx2 = new Float32Array(Capacity), cy2 = new Float32Array(Capacity), cz2 = new Float32Array(Capacity);
  const nc = new Int32Array(Capacity), ns = new Int32Array(Capacity);

  function update(dt: number) {
    const n = params.count;

    if (n !== prevCount) {
      for (let i = prevCount; i < n; i++) spawnOne(i);
      prevCount = n;
      agentsRef.value = n;
    }

    const s = dt * 0.001;
    const spd = params.speed;
    const sepW = params.separation;
    const aliW = params.alignment;
    const cohW = params.cohesion;
    const r = params.radius;
    const r2 = r * r;
    const sr = params.separationRadius;
    const sr2 = sr * sr;

    // Reset accumulators
    for (let i = 0; i < n; i++) {
      sx2[i] = sy2[i] = sz2[i] = 0;
      ax[i] = ay[i] = az[i] = 0;
      cx2[i] = cy2[i] = cz2[i] = 0;
      nc[i] = ns[i] = 0;
    }

    // Neighbor scan O(n²)
    for (let i = 0; i < n - 1; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = px[j] - px[i];
        const dy = py[j] - py[i];
        const dz = pz[j] - pz[i];
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < r2) {
          // Cohesion + alignment
          ax[i] += vx[j]; ay[i] += vy[j]; az[i] += vz[j];
          ax[j] += vx[i]; ay[j] += vy[i]; az[j] += vz[i];
          cx2[i] += px[j]; cy2[i] += py[j]; cz2[i] += pz[j];
          cx2[j] += px[i]; cy2[j] += py[i]; cz2[j] += pz[i];
          nc[i]++; nc[j]++;
          if (d2 < sr2) {
            // Separation (push away)
            sx2[i] -= dx; sy2[i] -= dy; sz2[i] -= dz;
            sx2[j] += dx; sy2[j] += dy; sz2[j] += dz;
            ns[i]++; ns[j]++;
          }
        }
      }
    }

    // Integrate
    for (let i = 0; i < n; i++) {
      let fx = 0, fy = 0, fz = 0;

      // Separation
      if (ns[i] > 0) { fx += sx2[i] * sepW; fy += sy2[i] * sepW; fz += sz2[i] * sepW; }

      // Alignment
      if (nc[i] > 0) {
        const inv = 1 / nc[i];
        fx += (ax[i] * inv - vx[i]) * aliW;
        fy += (ay[i] * inv - vy[i]) * aliW;
        fz += (az[i] * inv - vz[i]) * aliW;
      }

      // Cohesion
      if (nc[i] > 0) {
        const inv = 1 / nc[i];
        fx += (cx2[i] * inv - px[i]) * cohW * 0.05;
        fy += (cy2[i] * inv - py[i]) * cohW * 0.05;
        fz += (cz2[i] * inv - pz[i]) * cohW * 0.05;
      }

      // Boundary avoidance — steer back toward origin
      const margin = Bound * 0.15;
      const bx = Math.abs(px[i]) - (Bound - margin);
      const by = Math.abs(py[i]) - (Bound * 0.5 - margin);
      const bz = Math.abs(pz[i]) - (Bound - margin);
      if (bx > 0) fx -= Math.sign(px[i]) * bx * 0.5;
      if (by > 0) fy -= Math.sign(py[i]) * by * 0.5;
      if (bz > 0) fz -= Math.sign(pz[i]) * bz * 0.5;

      vx[i] += fx * s; vy[i] += fy * s; vz[i] += fz * s;

      // Normalize to target speed
      const vLen = Math.sqrt(vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
      if (vLen > 0.001) {
        const scale = spd / vLen;
        vx[i] *= scale; vy[i] *= scale; vz[i] *= scale;
      }

      px[i] += vx[i] * s;
      py[i] += vy[i] * s;
      pz[i] += vz[i] * s;

      // Orientation from velocity — capsule +Y points along travel
      const [qx, qy, qz, qw] = upToDirQuat(vx[i], vy[i], vz[i]);
      buf.writeTransform(i, px[i], py[i], pz[i], qx, qy, qz, qw, 1, 1, 1);
      // Color by speed — cyan to magenta
      const t = Math.min(vLen / (spd * 1.5), 1);
      buf.writeColor(i, t, 1 - t * 0.8, 1 - t * 0.4);
    }

    buf.commit(mesh, n);
  }

  function reset() {
    spawnAll();
    prevCount = params.count;
    agentsRef.value = params.count;
  }

  return {
    params,
    schema: [
      { type: "slider", key: "count", label: "Count", min: 10, max: Capacity, step: 10 },
      { type: "slider", key: "speed", label: "Speed", min: 1, max: 20, step: 0.5 },
      { type: "slider", key: "radius", label: "Neighbor Radius", min: 1, max: 15, step: 0.5 },
      { type: "slider", key: "separationRadius", label: "Sep. Radius", min: 0.5, max: 6, step: 0.25 },
      { type: "slider", key: "separation", label: "Separation", min: 0, max: 5, step: 0.1 },
      { type: "slider", key: "alignment", label: "Alignment", min: 0, max: 5, step: 0.1 },
      { type: "slider", key: "cohesion", label: "Cohesion", min: 0, max: 5, step: 0.1 },
    ],
    readouts: { agents: agentsRef },
    update,
    reset,
    detach,
  };
}
