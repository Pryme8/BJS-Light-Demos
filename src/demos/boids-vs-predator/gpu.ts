/**
 * GPU Ecosystem — full faithful Boids-vs-Predator on the GPU.
 *
 * Architecture: ping-pong stream compaction.
 *   Each frame reads from "A" buffers and emits survivors + newborns
 *   densely into "B" buffers via atomic append counters, then A↔B swap.
 *   No free-lists; no fragmentation; natural cap enforcement.
 *
 * Six compute passes per frame (all in one CommandEncoder):
 *   1. clear    — zero cell counts + output counters
 *   2. scatter  — scatter prey into 2D toroidal grid + reset prey alive flags
 *   3. pred     — hunt prey (direct scan), atomic claim, integrate, emit to B_pred
 *   4. prey     — flock (grid), flee preds (direct), seek food (direct, atomic claim),
 *                 integrate, emit to B_prey with hunger-shaded color
 *   5. preyRend — write prey instance mat4 + color from B_prey
 *   6. food     — emit alive food to B_food + spawn new food + write instance data
 *
 * Same-type collision: prey-prey via the 2D grid (soft pushout like boids GPU).
 * Cross-type collision: NONE (predators must be able to reach prey to eat them).
 *
 * Scale (GPU default): ~32k prey / 1k pred / 8k food — 40× the CPU caps.
 *
 * Storage binding counts per pass (stay ≤ 8, safe on all WebGPU devices):
 *   clear:4  scatter:4  pred:7  prey:8  preyRend:3  food:6
 *
 * NOTE: population dynamics use f32, hash RNG (pcg), and limited-neighbor grids,
 * so individual agent trajectories differ from the CPU f64 simulation — the
 * emergent population oscillations are qualitatively identical, not bit-exact.
 */

import {
  addToScene,
  createCapsule,
  createPbrMaterial,
  createSphere,
  invalidateRenderBundles,
  setMeshVisible,
  setThinInstanceColors,
  setThinInstanceCount,
  setThinInstances,
} from "@babylonjs/lite";
import type { EngineContext, Mesh, SceneContext } from "@babylonjs/lite";

// ── Exported constants ─────────────────────────────────────────────────────

export const GpuMaxPrey  = 32_768;
export const GpuMaxPred  = 1_024;
export const GpuMaxFood  = 8_192;

// Grid constants (2D toroidal, prey only).
const GridDimMax = 64;
const MaxPerCell = 64;
const MaxCells   = GridDimMax * GridDimMax;

// Agent state: 2 × vec4<f32> = 32 bytes
//   pos = (x, z, vx, vz)  — flat XZ plane, y is always 0
//   sta = (energy, age, eaten, _)
const BytesPerAgent = 32;

// Food: 4 × u32 = 16 bytes per slot (atomic array)
//   u32[0] = alive flag (atomic)
//   u32[1] = bitcast<f32>(x)
//   u32[2] = bitcast<f32>(z)
//   u32[3] = unused
const BytesPerFood  = 16;

// Output counters buffer: 3 atomic<u32> = 12 bytes → padded to 16.
const CounterBytes  = 16;

// UBO: 36 u32/f32 = 144 bytes.
const UBOBytes = 144;

// Reference energies for colour shading (match CPU scene.ts constants).
const PreyFull_GPU = 50;
const PredFull_GPU = 90;

// Thin-instance internal fields we mutate for buffer ownership.
type ThinInstData = {
  _gpuBuffer: GPUBuffer | null;       _gpuBufferStorage: boolean;
  _gpuVersion: number;                _version: number;
  _colorGpuBuffer: GPUBuffer | null;  _colorGpuBufferStorage: boolean;
  _colorGpuVersion: number;           _colorVersion: number;
};

// ── WGSL shared preamble ──────────────────────────────────────────────────

const WGSL_COMMON = /* wgsl */ `
struct Params {
  nPrey          : u32,   //   0 — current A-side prey count
  nPred          : u32,   //   4
  nFood          : u32,   //   8
  foodSpawnCount : u32,   //  12 — new food items to append this frame
  preyGridDim    : u32,   //  16
  maxPerCell     : u32,   //  20
  preyGridInvCS  : f32,   //  24 — 1 / cellSize
  worldX         : f32,   //  28
  worldZ         : f32,   //  32
  dt             : f32,   //  36 — dt * 0.001 * timeScale
  preySpeed      : f32,   //  40
  predSpeed      : f32,   //  44
  separation     : f32,   //  48
  alignment      : f32,   //  52
  cohesion       : f32,   //  56
  foodEnergy     : f32,   //  60
  preyMetabolism : f32,   //  64
  preyFoodRepro  : u32,   //  68
  predGain       : f32,   //  72
  predMetabolism : f32,   //  76
  predPreyRepro  : u32,   //  80
  lifespan       : f32,   //  84
  maxPrey        : u32,   //  88
  maxPred        : u32,   //  92
  maxFood        : u32,   //  96
  flockR2        : f32,   // 100
  sepR2          : f32,   // 104
  fleeR2         : f32,   // 108
  collRadius     : f32,   // 112
  collStrength   : f32,   // 116
  foodEatR2      : f32,   // 120
  preyEatR2      : f32,   // 124
  seed           : u32,   // 128
  preyFull       : f32,   // 132
  predFull       : f32,   // 136
  _pad           : u32,   // 140
}

struct Agent {
  pos : vec4<f32>,  // (x, z, vx, vz)
  sta : vec4<f32>,  // (energy, age, eaten, _)
}

// Exact current-frame agent counts — copied on the GPU timeline from the
// previous frame's output atomics (never CPU-written), so they always match
// the contents of the "A" buffers being read this frame.
struct Counts {
  nPrey : u32,
  nPred : u32,
  nFood : u32,
  _cpad : u32,
}

// Buffer capacities. The emit counter can overshoot capacity (atomicAdd keeps
// counting past the last written slot), so every count MUST be clamped to its
// capacity before use — that clamp equals the exact number of slots written.
const CAP_PREY : u32 = ${GpuMaxPrey}u;
const CAP_PRED : u32 = ${GpuMaxPred}u;
const CAP_FOOD : u32 = ${GpuMaxFood}u;

// Toroidal nearest-image delta.
fn ni(d : f32, w : f32) -> f32 {
  let h = w * 0.5;
  return select(select(d, d - w, d > h), d + w, d < -h);
}

// PCG hash → random u32.
fn pcg(v : u32) -> u32 {
  var s = v * 747796405u + 2891336453u;
  s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (s >> 22u) ^ s;
}
// → random f32 in [0,1).
fn pcgF(v : u32) -> f32 { return f32(pcg(v)) / 4294967296.0; }

// Quaternion: rotate local +Y onto direction (dx, 0, dz).
fn upToDirQuat(dx : f32, dz : f32) -> vec4<f32> {
  let len = sqrt(dx*dx + dz*dz);
  if (len < 1e-6) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
  let fx = dx / len; let fz = dz / len;
  // +Y → (fx, 0, fz): y component of the world dir is 0.
  // Using standard upToDirQuat but with 3D dir = (fx, 0, fz).
  let fy = 0.0;
  if (fy < -0.999999) { return vec4<f32>(1.0, 0.0, 0.0, 0.0); }
  let w_   = 1.0 + fy;
  let inv  = 1.0 / sqrt(2.0 * w_);
  return vec4<f32>(fz * inv, 0.0, -fx * inv, w_ * inv);
}

// Write column-major TRS mat4 for a capsule facing (dx, 0, dz) at (px, 0, pz).
fn writeCapsuleMat(mb : u32, px : f32, pz : f32, vx : f32, vz : f32,
                   matBuf : ptr<storage, array<f32>, read_write>) {
  let q = upToDirQuat(vx, vz);
  let qx = q.x; let qy = q.y; let qz = q.z; let qw = q.w;
  let x2 = qx+qx; let y2 = qy+qy; let z2 = qz+qz;
  let xx = qx*x2; let xy = qx*y2; let xz_ = qx*z2;
  let yy = qy*y2; let yz_ = qy*z2; let zz = qz*z2;
  let wx_ = qw*x2; let wy_ = qw*y2; let wz_ = qw*z2;
  (*matBuf)[mb+ 0u]=1.0-(yy+zz); (*matBuf)[mb+ 1u]=xy+wz_;      (*matBuf)[mb+ 2u]=xz_-wy_;     (*matBuf)[mb+ 3u]=0.0;
  (*matBuf)[mb+ 4u]=xy-wz_;      (*matBuf)[mb+ 5u]=1.0-(xx+zz); (*matBuf)[mb+ 6u]=yz_+wx_;     (*matBuf)[mb+ 7u]=0.0;
  (*matBuf)[mb+ 8u]=xz_+wy_;     (*matBuf)[mb+ 9u]=yz_-wx_;     (*matBuf)[mb+10u]=1.0-(xx+yy); (*matBuf)[mb+11u]=0.0;
  (*matBuf)[mb+12u]=px;           (*matBuf)[mb+13u]=0.0;          (*matBuf)[mb+14u]=pz;           (*matBuf)[mb+15u]=1.0;
}

// Write scale-1 translation mat4 (for pred spheres + food).
fn writeTransMat(mb : u32, px : f32, pz : f32,
                 matBuf : ptr<storage, array<f32>, read_write>) {
  (*matBuf)[mb+ 0u]=1.0; (*matBuf)[mb+ 1u]=0.0; (*matBuf)[mb+ 2u]=0.0; (*matBuf)[mb+ 3u]=0.0;
  (*matBuf)[mb+ 4u]=0.0; (*matBuf)[mb+ 5u]=1.0; (*matBuf)[mb+ 6u]=0.0; (*matBuf)[mb+ 7u]=0.0;
  (*matBuf)[mb+ 8u]=0.0; (*matBuf)[mb+ 9u]=0.0; (*matBuf)[mb+10u]=1.0; (*matBuf)[mb+11u]=0.0;
  (*matBuf)[mb+12u]=px;  (*matBuf)[mb+13u]=0.0; (*matBuf)[mb+14u]=pz;  (*matBuf)[mb+15u]=1.0;
}

// 2D toroidal cell index.
fn cellOf(px : f32, pz : f32, dim : u32, invCS : f32, halfX : f32, halfZ : f32) -> u32 {
  let cx = u32(clamp((px + halfX) * invCS, 0.0, f32(dim - 1u)));
  let cz = u32(clamp((pz + halfZ) * invCS, 0.0, f32(dim - 1u)));
  return cx + cz * dim;
}
`;

// ── Pass 1: clear ──────────────────────────────────────────────────────────

const WGSL_CLEAR = /* wgsl */ `
${WGSL_COMMON}
@group(0) @binding(0) var<uniform>             params      : Params;
@group(0) @binding(1) var<storage, read_write> preyGCount  : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> outCounters : array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  // Clear prey grid counts.
  let cells = params.preyGridDim * params.preyGridDim;
  if (i < cells) { atomicStore(&preyGCount[i], 0u); }
  // Clear output counters (only 3 needed; thread 0 handles all).
  if (i == 0u) {
    atomicStore(&outCounters[0], 0u);  // nPrey_out
    atomicStore(&outCounters[1], 0u);  // nPred_out
    atomicStore(&outCounters[2], 0u);  // nFood_out
  }
}
`;

// ── Pass 2: scatter prey into 2D grid + reset alive flags ─────────────────

const WGSL_SCATTER = /* wgsl */ `
${WGSL_COMMON}
@group(0) @binding(0) var<uniform>             params     : Params;
@group(0) @binding(1) var<storage, read_write> preyStateA : array<Agent>;
@group(0) @binding(2) var<storage, read_write> preyAliveA : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> preyGCount : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> preyGSlots : array<u32>;
@group(0) @binding(5) var<uniform>             counts     : Counts;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= min(counts.nPrey, CAP_PREY)) { return; }
  // Reset alive flag so predators can claim this prey this frame.
  atomicStore(&preyAliveA[i], 1u);
  // Scatter into 2D grid.
  // Agent pos = (x, z, vx, vz) — z position is pos.y, NOT pos.z.
  let px = preyStateA[i].pos.x; let pz = preyStateA[i].pos.y;
  let c    = cellOf(px, pz, params.preyGridDim, params.preyGridInvCS,
                    params.worldX * 0.5, params.worldZ * 0.5);
  let slot = atomicAdd(&preyGCount[c], 1u);
  if (slot < params.maxPerCell) {
    preyGSlots[c * params.maxPerCell + slot] = i;
  }
}
`;

// ── Pass 3: predator step ──────────────────────────────────────────────────
// Hunts prey (direct scan), eats, pred-pred collision (direct), integrates,
// emits survivors + offspring to B_pred with instance data.

const WGSL_PRED = /* wgsl */ `
${WGSL_COMMON}
@group(0) @binding(0) var<uniform>             params      : Params;
@group(0) @binding(1) var<storage, read_write> predStateA  : array<Agent>;
@group(0) @binding(2) var<storage, read_write> predStateB  : array<Agent>;
@group(0) @binding(3) var<storage, read_write> preyStateA  : array<Agent>;
@group(0) @binding(4) var<storage, read_write> preyAliveA  : array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> predMatBuf  : array<f32>;
@group(0) @binding(6) var<storage, read_write> predColBuf  : array<f32>;
@group(0) @binding(7) var<storage, read_write> outCounters : array<atomic<u32>>;
@group(0) @binding(8) var<uniform>             counts      : Counts;

fn emitPred(sta : vec4<f32>, pos : vec4<f32>, outCounters : ptr<storage, array<atomic<u32>>, read_write>,
            predStateB : ptr<storage, array<Agent>, read_write>,
            predMatBuf : ptr<storage, array<f32>, read_write>,
            predColBuf : ptr<storage, array<f32>, read_write>,
            maxPred : u32, predFull : f32) {
  let slot = atomicAdd(&(*outCounters)[1], 1u);
  if (slot >= maxPred) { return; }
  (*predStateB)[slot] = Agent(pos, sta);
  // pos layout: .x = world-x, .y = world-z — use .y for the z translation.
  writeTransMat(slot * 16u, pos.x, pos.y, predMatBuf);
  let frac = clamp(sta.x / predFull, 0.0, 1.0);
  let cb = slot * 4u;
  (*predColBuf)[cb+0u] = 0.4 + 0.6*frac;
  (*predColBuf)[cb+1u] = 0.05;
  (*predColBuf)[cb+2u] = 0.3 + 0.35*frac;
  (*predColBuf)[cb+3u] = 1.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let p = gid.x;
  let nPred = min(counts.nPred, CAP_PRED);
  let nPrey = min(counts.nPrey, CAP_PREY);
  if (p >= nPred) { return; }

  let dt    = params.dt;
  let wX    = params.worldX; let wZ = params.worldZ;
  var pos   = predStateA[p].pos;  // packed as (x, z, vx, vz)
  var sta   = predStateA[p].sta;  // (energy, age, eaten, _)
  var eaten = u32(sta.z);
  // Agent vec4 layout: .x = world-x, .y = world-z, .z = vel-x, .w = vel-z
  var px = pos.x; var pz = pos.y; var vx = pos.z; var vz = pos.w;

  // Hunt: find nearest alive prey (direct scan).
  var bestD2 = 1e30;
  var bestJ  = nPrey; // sentinel = not found
  for (var k = 0u; k < nPrey; k++) {
    if (atomicLoad(&preyAliveA[k]) == 0u) { continue; }
    let kx = preyStateA[k].pos.x;
    let kz = preyStateA[k].pos.y;
    let dx = ni(kx - px, wX); let dz = ni(kz - pz, wZ);
    let d2 = dx*dx + dz*dz;
    if (d2 < bestD2) { bestD2 = d2; bestJ = k; }
  }

  // Steer toward nearest prey.
  if (bestJ < nPrey) {
    let kx = preyStateA[bestJ].pos.x;
    let kz = preyStateA[bestJ].pos.y;
    let dx = ni(kx - px, wX); let dz = ni(kz - pz, wZ);
    let len = sqrt(dx*dx + dz*dz) + 0.001;
    vx += (dx/len - vx) * 0.08;
    vz += (dz/len - vz) * 0.08;
    // Eat if close enough.
    if (bestD2 < params.preyEatR2) {
      // Atomic exchange: we claim the prey (set alive=0).
      let prev = atomicExchange(&preyAliveA[bestJ], 0u);
      if (prev == 1u) {
        // We won the race — we eat this prey.
        sta.x += params.predGain;
        eaten++;
      }
    }
  }

  // Pred-pred soft collision (direct scan, nPred ≤ 1024).
  var collPush = vec2<f32>(0.0);
  for (var k = 0u; k < nPred; k++) {
    if (k == p) { continue; }
    let kx = predStateA[k].pos.x; let kz = predStateA[k].pos.y;
    let dx = ni(kx - px, wX); let dz = ni(kz - pz, wZ);
    let d2 = dx*dx + dz*dz;
    let collD = params.collRadius * 2.0;
    if (d2 < collD*collD && d2 > 1e-8) {
      let d    = sqrt(d2);
      let push = (collD - d) * 0.5 * params.collStrength / d;
      collPush -= vec2<f32>(dx, dz) * push;
    }
  }

  // Normalize velocity to predSpeed.
  let vLen = sqrt(vx*vx + vz*vz);
  if (vLen > 0.001) { vx = vx * params.predSpeed / vLen; vz = vz * params.predSpeed / vLen; }

  // Integrate with toroidal wrap.
  px = px + vx * dt + collPush.x;
  pz = pz + vz * dt + collPush.y;
  px = px - wX * floor((px + wX*0.5) / wX);
  pz = pz - wZ * floor((pz + wZ*0.5) / wZ);

  // Metabolism + aging.
  sta.x -= params.predMetabolism * dt;
  sta.y += dt;

  let survive = sta.x > 0.0 && sta.y < params.lifespan;
  if (survive) {
    // Reproduction decision (matches CPU: split energy + reset eaten counter).
    let doRepro = eaten >= params.predPreyRepro;
    if (doRepro) {
      sta.x = sta.x * 0.5;   // parent splits its energy with the child
      eaten = 0u;            // reset prey-to-reproduce counter
    }
    sta.z = f32(eaten);

    // Emit survivor (parent, with post-repro energy/counter).
    emitPred(sta, vec4<f32>(px, pz, vx, vz), &outCounters, &predStateB, &predMatBuf, &predColBuf,
             params.maxPred, params.predFull);

    // Emit offspring.
    if (doRepro && atomicLoad(&outCounters[1]) < params.maxPred) {
      var childSta = sta;   // already-halved energy, eaten (sta.z) = 0
      let rng = pcgF(p * 73856093u + params.seed);
      let ang = rng * 6.28318;
      let cx = px + cos(ang) * 1.5; let cz = pz + sin(ang) * 1.5;
      let cwx = cx - wX * floor((cx + wX*0.5) / wX);
      let cwz = cz - wZ * floor((cz + wZ*0.5) / wZ);
      emitPred(childSta, vec4<f32>(cwx, cwz, -vx, -vz), &outCounters, &predStateB, &predMatBuf, &predColBuf,
               params.maxPred, params.predFull);
    }
  }
}
`;

// ── Pass 4: prey step ──────────────────────────────────────────────────────
// Flock (grid), flee preds (direct), seek/eat food (direct atomic), integrate,
// emit survivors + offspring to B_prey.

const WGSL_PREY = /* wgsl */ `
${WGSL_COMMON}
@group(0) @binding(0) var<uniform>             params      : Params;
@group(0) @binding(1) var<storage, read_write> preyStateA  : array<Agent>;
@group(0) @binding(2) var<storage, read_write> preyAliveA  : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> preyGCount  : array<u32>;
@group(0) @binding(4) var<storage, read_write> preyGSlots  : array<u32>;
@group(0) @binding(5) var<storage, read_write> predStateA  : array<Agent>;
@group(0) @binding(6) var<storage, read_write> foodBufA    : array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> preyStateB  : array<Agent>;
@group(0) @binding(8) var<storage, read_write> outCounters : array<atomic<u32>>;
@group(0) @binding(9) var<uniform>             counts      : Counts;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  let nPrey = min(counts.nPrey, CAP_PREY);
  let nPred = min(counts.nPred, CAP_PRED);
  let nFood = min(counts.nFood, CAP_FOOD);
  if (i >= nPrey) { return; }
  // Skip prey that were eaten by a predator this frame.
  if (atomicLoad(&preyAliveA[i]) == 0u) { return; }

  let dt = params.dt;
  let wX = params.worldX; let wZ = params.worldZ;
  var px = preyStateA[i].pos.x; var pz = preyStateA[i].pos.y;
  var vx = preyStateA[i].pos.z; var vz = preyStateA[i].pos.w;
  var sta = preyStateA[i].sta;  // (energy, age, foodEaten, _)
  var foodEaten = u32(sta.z);

  let dim = params.preyGridDim;
  let halfX = wX * 0.5; let halfZ = wZ * 0.5;
  let invCS = params.preyGridInvCS;
  let mpc = params.maxPerCell;

  // ── Prey-prey flocking + same-type collision (2D grid scan) ──
  var sepX = 0.0; var sepZ = 0.0;
  var aliX = 0.0; var aliZ = 0.0;
  var cohX = 0.0; var cohZ = 0.0;
  var coll = vec2<f32>(0.0);
  var nc = 0u; var ns = 0u;

  let gcx = i32(clamp((px + halfX) * invCS, 0.0, f32(dim - 1u)));
  let gcz = i32(clamp((pz + halfZ) * invCS, 0.0, f32(dim - 1u)));

  for (var dz = -1; dz <= 1; dz++) {
    for (var dx = -1; dx <= 1; dx++) {
      let nx = (gcx + dx + i32(dim)) % i32(dim);
      let nz = (gcz + dz + i32(dim)) % i32(dim);
      let cell = u32(nx) + u32(nz) * dim;
      let cnt  = min(preyGCount[cell], mpc);
      for (var s = 0u; s < cnt; s++) {
        let j = preyGSlots[cell * mpc + s];
        if (j == i) { continue; }
        let dx2 = ni(preyStateA[j].pos.x - px, wX);
        let dz2 = ni(preyStateA[j].pos.y - pz, wZ);
        let d2  = dx2*dx2 + dz2*dz2;
        if (d2 >= params.flockR2) { continue; }
        aliX += preyStateA[j].pos.z;  aliZ += preyStateA[j].pos.w;
        cohX += dx2; cohZ += dz2; nc++;
        if (d2 < params.sepR2) { sepX -= dx2; sepZ -= dz2; ns++; }
        // Same-type collision.
        let cDist = params.collRadius * 2.0;
        if (d2 < cDist*cDist && d2 > 1e-8) {
          let d    = sqrt(d2);
          let push = (cDist - d) * 0.5 * params.collStrength / d;
          coll -= vec2<f32>(dx2, dz2) * push;
        }
      }
    }
  }

  var fx = 0.0; var fz_ = 0.0;
  if (ns > 0u) { fx += sepX * params.separation; fz_ += sepZ * params.separation; }
  if (nc > 0u) {
    let inv = 1.0 / f32(nc);
    fx += (aliX * inv - vx) * params.alignment;
    fz_ += (aliZ * inv - vz) * params.alignment;
    fx += cohX * inv * params.cohesion * 0.05;
    fz_ += cohZ * inv * params.cohesion * 0.05;
  }

  // ── Flee nearest predator (direct scan, nPred ≤ 1024) ──
  var fNd = 1e30; var fDx = 0.0; var fDz = 0.0;
  for (var k = 0u; k < nPred; k++) {
    let kx = predStateA[k].pos.x; let kz = predStateA[k].pos.y;
    let dx2 = ni(kx - px, wX); let dz2 = ni(kz - pz, wZ);
    let d2  = dx2*dx2 + dz2*dz2;
    if (d2 < fNd) { fNd = d2; fDx = dx2; fDz = dz2; }
  }
  if (fNd < params.fleeR2) {
    let dist  = sqrt(fNd) + 0.001;
    let fleeF = (sqrt(params.fleeR2) - dist) / dist * 3.0;
    fx -= fDx * fleeF; fz_ -= fDz * fleeF;
  }

  // ── Seek + eat nearest food (direct scan, nFood ≤ 8192) ──
  var fndFood = 1e30; var fFdx = 0.0; var fFdz = 0.0; var fFoodIdx = nFood;
  for (var k = 0u; k < nFood; k++) {
    if (atomicLoad(&foodBufA[k*4u]) == 0u) { continue; }
    let ffx = bitcast<f32>(atomicLoad(&foodBufA[k*4u + 1u]));
    let ffz = bitcast<f32>(atomicLoad(&foodBufA[k*4u + 2u]));
    let dx2 = ni(ffx - px, wX); let dz2 = ni(ffz - pz, wZ);
    let d2  = dx2*dx2 + dz2*dz2;
    if (d2 < fndFood) { fndFood = d2; fFdx = dx2; fFdz = dz2; fFoodIdx = k; }
  }
  if (fFoodIdx < nFood) {
    let hunger = 1.0 - clamp(sta.x / params.preyFull, 0.0, 1.0);
    let seek = 0.6 + hunger * 1.4;
    let len  = sqrt(fndFood) + 0.001;
    fx += (fFdx/len) * seek; fz_ += (fFdz/len) * seek;
    if (fndFood < params.foodEatR2) {
      // Atomic claim: exchange alive flag to 0; only one prey wins.
      let prev = atomicExchange(&foodBufA[fFoodIdx*4u], 0u);
      if (prev == 1u) {
        sta.x += params.foodEnergy;
        foodEaten++;
      }
    }
  }

  // ── Integrate ──
  vx += fx * dt; vz += fz_ * dt;
  let vLen = sqrt(vx*vx + vz*vz);
  if (vLen > 0.001) { vx = vx * params.preySpeed / vLen; vz = vz * params.preySpeed / vLen; }
  px = px + vx * dt + coll.x;
  pz = pz + vz * dt + coll.y;
  px = px - wX * floor((px + wX*0.5) / wX);
  pz = pz - wZ * floor((pz + wZ*0.5) / wZ);

  // ── Energy + aging ──
  sta.x -= params.preyMetabolism * dt;
  sta.y += dt;

  if (sta.x <= 0.0 || sta.y > params.lifespan) { return; } // dies

  // ── Reproduction decision (matches CPU: split energy + reset food counter) ──
  let doRepro = foodEaten >= params.preyFoodRepro;
  if (doRepro) {
    sta.x = sta.x * 0.5;   // parent splits its energy with the child
    foodEaten = 0u;        // reset food-to-reproduce counter
  }
  sta.z = f32(foodEaten);

  // ── Emit survivor (parent, with post-repro energy/counter) ──
  let slot = atomicAdd(&outCounters[0], 1u);
  if (slot < params.maxPrey) {
    preyStateB[slot] = Agent(vec4<f32>(px, pz, vx, vz), sta);
  }

  // ── Emit offspring ──
  if (doRepro && atomicLoad(&outCounters[0]) < params.maxPrey) {
    var cSta = sta;   // already-halved energy, foodEaten (sta.z) = 0
    let rng  = pcgF(i * 1664525u + params.seed);
    let ang  = rng * 6.28318;
    let cx   = px + cos(ang); let cz = pz + sin(ang);
    let cwx  = cx - wX * floor((cx + wX*0.5) / wX);
    let cwz  = cz - wZ * floor((cz + wZ*0.5) / wZ);
    let cSlot = atomicAdd(&outCounters[0], 1u);
    if (cSlot < params.maxPrey) {
      preyStateB[cSlot] = Agent(vec4<f32>(cwx, cwz, -vx, -vz), cSta);
    }
  }
}
`;

// ── Pass 5: prey render ────────────────────────────────────────────────────
// Write mat4 + color from the freshly written B_prey state.

const WGSL_PREY_RENDER = /* wgsl */ `
${WGSL_COMMON}
@group(0) @binding(0) var<uniform>             params     : Params;
@group(0) @binding(1) var<storage, read_write> preyStateB : array<Agent>;
@group(0) @binding(2) var<storage, read_write> preyMatBuf : array<f32>;
@group(0) @binding(3) var<storage, read_write> preyColBuf : array<f32>;
@group(0) @binding(4) var<storage, read_write> outCounts  : array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  // Use output count from pred/prey passes (it's final by the time this runs).
  let n = atomicLoad(&outCounts[0]);
  if (i >= n) { return; }
  let px = preyStateB[i].pos.x; let pz = preyStateB[i].pos.y;
  let vx = preyStateB[i].pos.z; let vz = preyStateB[i].pos.w;
  let en = preyStateB[i].sta.x;
  writeCapsuleMat(i * 16u, px, pz, vx, vz, &preyMatBuf);
  let frac = clamp(en / params.preyFull, 0.0, 1.0);
  let cb = i * 4u;
  preyColBuf[cb+0u] = 0.05;
  preyColBuf[cb+1u] = 0.3 + 0.6 * frac;
  preyColBuf[cb+2u] = 0.45 + 0.55 * frac;
  preyColBuf[cb+3u] = 1.0;
}
`;

// ── Pass 6: food step ──────────────────────────────────────────────────────
// Emit alive food to B + spawn new food + write food instance data.

const WGSL_FOOD = /* wgsl */ `
${WGSL_COMMON}
@group(0) @binding(0) var<uniform>             params      : Params;
@group(0) @binding(1) var<storage, read_write> foodBufA    : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> foodBufB    : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> outCounters : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> foodMatBuf  : array<f32>;
@group(0) @binding(5) var<storage, read_write> foodColBuf  : array<f32>;
@group(0) @binding(6) var<uniform>             counts      : Counts;

fn emitFood(fx : f32, fz : f32, outCounters : ptr<storage, array<atomic<u32>>, read_write>,
            foodBufB : ptr<storage, array<atomic<u32>>, read_write>,
            foodMatBuf : ptr<storage, array<f32>, read_write>,
            foodColBuf : ptr<storage, array<f32>, read_write>, maxFood : u32) {
  let slot = atomicAdd(&(*outCounters)[2], 1u);
  if (slot >= maxFood) { return; }
  atomicStore(&(*foodBufB)[slot*4u + 0u], 1u);
  atomicStore(&(*foodBufB)[slot*4u + 1u], bitcast<u32>(fx));
  atomicStore(&(*foodBufB)[slot*4u + 2u], bitcast<u32>(fz));
  atomicStore(&(*foodBufB)[slot*4u + 3u], 0u);
  writeTransMat(slot * 16u, fx, fz, foodMatBuf);
  let cb = slot * 4u;
  (*foodColBuf)[cb+0u] = 0.55;
  (*foodColBuf)[cb+1u] = 0.95;
  (*foodColBuf)[cb+2u] = 0.2;
  (*foodColBuf)[cb+3u] = 1.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  let nFood = min(counts.nFood, CAP_FOOD);
  let total = nFood + params.foodSpawnCount;
  if (i >= total) { return; }

  if (i < nFood) {
    // Emit existing food if still alive.
    let alive = atomicLoad(&foodBufA[i*4u]);
    if (alive == 1u) {
      let fx = bitcast<f32>(atomicLoad(&foodBufA[i*4u + 1u]));
      let fz = bitcast<f32>(atomicLoad(&foodBufA[i*4u + 2u]));
      emitFood(fx, fz, &outCounters, &foodBufB, &foodMatBuf, &foodColBuf, params.maxFood);
    }
  } else {
    // Spawn a new food item at a random position.
    let si  = i - nFood;
    let r1  = pcgF(si * 134775813u + params.seed + 7u);
    let r2  = pcgF(si * 1664525u   + params.seed + 13u);
    let fx  = (r1 - 0.5) * params.worldX;
    let fz  = (r2 - 0.5) * params.worldZ;
    emitFood(fx, fz, &outCounters, &foodBufB, &foodMatBuf, &foodColBuf, params.maxFood);
  }
}
`;

// ── EcosystemGpuParams (passed by the scene each frame) ───────────────────

export interface EcosystemGpuParams {
  nPrey: number; nPred: number; nFood: number;
  foodSpawnCount: number;
  worldX: number; worldZ: number;
  dt: number;
  preySpeed: number; predSpeed: number;
  separation: number; alignment: number; cohesion: number;
  foodEnergy: number;
  preyMetabolism: number; preyFoodToRepro: number;
  predGain: number; predMetabolism: number; predPreyToRepro: number;
  lifespan: number;
  maxPrey: number; maxPred: number; maxFood: number;
  collRadius: number; collStrength: number;
}

// ── EcosystemGpu ──────────────────────────────────────────────────────────

export class EcosystemGpu {
  readonly preyMesh: Mesh;
  readonly predMesh: Mesh;
  readonly foodMesh: Mesh;

  private readonly engine: EngineContext;
  private readonly device: GPUDevice;
  private readonly preyTi: ThinInstData;
  private readonly predTi: ThinInstData;
  private readonly foodTi: ThinInstData;

  // Owned instance buffers.
  private readonly preyMatBuf: GPUBuffer;
  private readonly preyColBuf: GPUBuffer;
  private readonly predMatBuf: GPUBuffer;
  private readonly predColBuf: GPUBuffer;
  private readonly foodMatBuf: GPUBuffer;
  private readonly foodColBuf: GPUBuffer;

  // Ping-pong state: A is always the "read" side this frame.
  private preyStateA: GPUBuffer;
  private preyStateB: GPUBuffer;
  private predStateA: GPUBuffer;
  private predStateB: GPUBuffer;
  private foodBufA:   GPUBuffer;
  private foodBufB:   GPUBuffer;

  // Alive flags (only one set needed — reset by scatter each frame).
  private readonly preyAliveA: GPUBuffer;

  // Prey spatial grid.
  private readonly preyGCount: GPUBuffer;
  private readonly preyGSlots: GPUBuffer;

  // Output counters (atomic, reset each frame by clear pass).
  private readonly outCounters: GPUBuffer;
  // Staging buffer for async readback of the 3 counters (display only).
  private readonly staging: GPUBuffer;
  // Exact per-frame input counts — copied on the GPU timeline from the previous
  // frame's outCounters, so shaders always agree with the buffer contents.
  private readonly countsUbo: GPUBuffer;

  // UBO.
  private readonly ubo:     GPUBuffer;
  private readonly uboData: Uint32Array;

  // Pipelines.
  private readonly pipeClear:     GPUComputePipeline;
  private readonly pipeScatter:   GPUComputePipeline;
  private readonly pipePred:      GPUComputePipeline;
  private readonly pipePrey:      GPUComputePipeline;
  private readonly pipePreyRend:  GPUComputePipeline;
  private readonly pipeFood:      GPUComputePipeline;

  // Two sets of bind groups (AtoB and BtoA) to avoid per-frame BG creation.
  // Active set is selected based on which buffer is currently preyStateA.
  private bgClear!:       GPUBindGroup;  // fixed (cleared + counters always same buffers)
  private bgScatterAtoB!: GPUBindGroup;
  private bgScatterBtoA!: GPUBindGroup;
  private bgPredAtoB!:    GPUBindGroup;
  private bgPredBtoA!:    GPUBindGroup;
  private bgPreyAtoB!:    GPUBindGroup;
  private bgPreyBtoA!:    GPUBindGroup;
  private bgPreyRendAtoB!:GPUBindGroup;
  private bgPreyRendBtoA!:GPUBindGroup;
  private bgFoodAtoB!:    GPUBindGroup;
  private bgFoodBtoA!:    GPUBindGroup;

  // State tracking.
  private _nPrey = 0;
  private _nPred = 0;
  private _nFood = 0;
  private _frameCount = 0;
  private _gridDim = 1;
  private _readbackPending = false;
  private _frontIsA = true; // true = preyStateA is the current "front" buffer

  private _ok = false;
  get ok(): boolean { return this._ok; }

  get counts(): { nPrey: number; nPred: number; nFood: number } {
    return { nPrey: this._nPrey, nPred: this._nPred, nFood: this._nFood };
  }

  constructor(engine: EngineContext, scene: SceneContext) {
    this.engine = engine;
    const dev = (engine as unknown as { _device: GPUDevice })._device;
    this.device = dev;

    const inst = GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST;
    this.preyMatBuf = dev.createBuffer({ label: "eco-preyMat", size: GpuMaxPrey * 64, usage: inst });
    this.preyColBuf = dev.createBuffer({ label: "eco-preyCol", size: GpuMaxPrey * 16, usage: inst });
    this.predMatBuf = dev.createBuffer({ label: "eco-predMat", size: GpuMaxPred * 64, usage: inst });
    this.predColBuf = dev.createBuffer({ label: "eco-predCol", size: GpuMaxPred * 16, usage: inst });
    this.foodMatBuf = dev.createBuffer({ label: "eco-foodMat", size: GpuMaxFood * 64, usage: inst });
    this.foodColBuf = dev.createBuffer({ label: "eco-foodCol", size: GpuMaxFood * 16, usage: inst });

    // Build meshes with owned instance buffers.
    this.preyMesh = this.buildMesh(engine, scene,
      createCapsule(engine, { height: 0.9, radius: 0.25, tessellation: 6 }),
      createPbrMaterial({ baseColorFactor: [1, 1, 1, 1], metallicFactor: 0.1, roughnessFactor: 0.5 }),
      GpuMaxPrey, this.preyMatBuf, this.preyColBuf);
    this.preyTi = (this.preyMesh as unknown as { thinInstances: ThinInstData }).thinInstances;

    this.predMesh = this.buildMesh(engine, scene,
      createSphere(engine, { segments: 7, diameter: 1.5 }),
      createPbrMaterial({ baseColorFactor: [1, 1, 1, 1], metallicFactor: 0.3, roughnessFactor: 0.4 }),
      GpuMaxPred, this.predMatBuf, this.predColBuf);
    this.predTi = (this.predMesh as unknown as { thinInstances: ThinInstData }).thinInstances;

    this.foodMesh = this.buildMesh(engine, scene,
      createSphere(engine, { segments: 5, diameter: 0.5 }),
      createPbrMaterial({ unlit: true, baseColorFactor: [1, 1, 1, 1] }),
      GpuMaxFood, this.foodMatBuf, this.foodColBuf);
    this.foodTi = (this.foodMesh as unknown as { thinInstances: ThinInstData }).thinInstances;

    // Ping-pong agent state buffers.
    const stUsg = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.preyStateA = dev.createBuffer({ label: "eco-preyA", size: GpuMaxPrey * BytesPerAgent, usage: stUsg });
    this.preyStateB = dev.createBuffer({ label: "eco-preyB", size: GpuMaxPrey * BytesPerAgent, usage: stUsg });
    this.predStateA = dev.createBuffer({ label: "eco-predA", size: GpuMaxPred * BytesPerAgent, usage: stUsg });
    this.predStateB = dev.createBuffer({ label: "eco-predB", size: GpuMaxPred * BytesPerAgent, usage: stUsg });
    this.foodBufA   = dev.createBuffer({ label: "eco-foodA", size: GpuMaxFood * BytesPerFood,  usage: stUsg });
    this.foodBufB   = dev.createBuffer({ label: "eco-foodB", size: GpuMaxFood * BytesPerFood,  usage: stUsg });

    // Alive flags (atomic u32, one per prey slot).
    this.preyAliveA = dev.createBuffer({
      label: "eco-alive",
      size: GpuMaxPrey * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // 2D prey grid.
    this.preyGCount = dev.createBuffer({
      label: "eco-gCount",
      size: MaxCells * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.preyGSlots = dev.createBuffer({
      label: "eco-gSlots",
      size: MaxCells * MaxPerCell * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Output counters + readback staging.
    this.outCounters = dev.createBuffer({
      label: "eco-counters",
      size: CounterBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.staging = dev.createBuffer({
      label: "eco-staging",
      size: CounterBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.countsUbo = dev.createBuffer({
      label: "eco-countsUbo",
      size: CounterBytes,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // UBO.
    this.uboData = new Uint32Array(UBOBytes / 4);
    this.ubo = dev.createBuffer({
      label: "eco-ubo",
      size: UBOBytes,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Build pipelines.
    try {
      this.pipeClear    = this.buildPipeline("eco-clear",     WGSL_CLEAR,      this.bglClear());
      this.pipeScatter  = this.buildPipeline("eco-scatter",   WGSL_SCATTER,    this.bglScatter());
      this.pipePred     = this.buildPipeline("eco-pred",      WGSL_PRED,       this.bglPred());
      this.pipePrey     = this.buildPipeline("eco-prey",      WGSL_PREY,       this.bglPrey());
      this.pipePreyRend = this.buildPipeline("eco-preyRend",  WGSL_PREY_RENDER,this.bglPreyRend());
      this.pipeFood     = this.buildPipeline("eco-food",      WGSL_FOOD,       this.bglFood());
    } catch (e) {
      console.error("[EcosystemGpu] pipeline build failed:", e);
      this.pipeClear = null!; this.pipeScatter = null!; this.pipePred = null!;
      this.pipePrey = null!; this.pipePreyRend = null!; this.pipeFood = null!;
      return;
    }

    // Build both A↔B bind group sets.
    this.rebuildBindGroups();
    this._ok = true;
  }

  // ── Mesh builder ──────────────────────────────────────────────────────────
  private buildMesh(engine: EngineContext, scene: SceneContext, mesh: Mesh, mat: ReturnType<typeof createPbrMaterial>,
                    max: number, matBuf: GPUBuffer, colBuf: GPUBuffer): Mesh {
    mesh.material = mat;
    addToScene(scene, mesh);
    const ids = new Float32Array(max * 16);
    for (let i = 0; i < max; i++) { const b = i*16; ids[b]=1; ids[b+5]=1; ids[b+10]=1; ids[b+15]=1; }
    setThinInstances(mesh, ids, max);
    setThinInstanceColors(mesh, new Float32Array(max * 4).fill(1));
    setThinInstanceCount(mesh, 0);
    const ti = (mesh as unknown as { thinInstances: ThinInstData }).thinInstances;
    ti._gpuBuffer             = matBuf; ti._gpuBufferStorage      = false; ti._gpuVersion      = ti._version;
    ti._colorGpuBuffer        = colBuf; ti._colorGpuBufferStorage = false; ti._colorGpuVersion = ti._colorVersion;
    return mesh;
  }

  // ── BGL / pipeline helpers ─────────────────────────────────────────────
  private bglEntry(b: number, t: GPUBufferBindingType): GPUBindGroupLayoutEntry {
    return { binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: t } };
  }
  private bglClear(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({ label: "eco-bgl-clr", entries: [
      this.bglEntry(0, "uniform"), this.bglEntry(1, "storage"), this.bglEntry(2, "storage"),
    ]});
  }
  private bglScatter(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({ label: "eco-bgl-sc", entries: [
      this.bglEntry(0, "uniform"), this.bglEntry(1, "storage"), this.bglEntry(2, "storage"),
      this.bglEntry(3, "storage"), this.bglEntry(4, "storage"), this.bglEntry(5, "uniform"),
    ]});
  }
  private bglPred(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({ label: "eco-bgl-pred", entries: [
      this.bglEntry(0, "uniform"), this.bglEntry(1, "storage"), this.bglEntry(2, "storage"),
      this.bglEntry(3, "storage"), this.bglEntry(4, "storage"),
      this.bglEntry(5, "storage"), this.bglEntry(6, "storage"), this.bglEntry(7, "storage"),
      this.bglEntry(8, "uniform"),
    ]});
  }
  private bglPrey(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({ label: "eco-bgl-prey", entries: [
      this.bglEntry(0, "uniform"), this.bglEntry(1, "storage"), this.bglEntry(2, "storage"),
      this.bglEntry(3, "storage"), this.bglEntry(4, "storage"), this.bglEntry(5, "storage"),
      this.bglEntry(6, "storage"), this.bglEntry(7, "storage"), this.bglEntry(8, "storage"),
      this.bglEntry(9, "uniform"),
    ]});
  }
  private bglPreyRend(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({ label: "eco-bgl-pr", entries: [
      this.bglEntry(0, "uniform"), this.bglEntry(1, "storage"), this.bglEntry(2, "storage"),
      this.bglEntry(3, "storage"), this.bglEntry(4, "storage"),
    ]});
  }
  private bglFood(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({ label: "eco-bgl-food", entries: [
      this.bglEntry(0, "uniform"), this.bglEntry(1, "storage"), this.bglEntry(2, "storage"),
      this.bglEntry(3, "storage"), this.bglEntry(4, "storage"), this.bglEntry(5, "storage"),
      this.bglEntry(6, "uniform"),
    ]});
  }
  private buildPipeline(label: string, wgsl: string, bgl: GPUBindGroupLayout): GPUComputePipeline {
    const mod = this.device.createShaderModule({ label, code: wgsl });
    return this.device.createComputePipeline({
      label, layout: this.device.createPipelineLayout({ label: `${label}-layout`, bindGroupLayouts: [bgl] }),
      compute: { module: mod, entryPoint: "main" },
    });
  }
  private bg(label: string, pipeline: GPUComputePipeline, entries: { binding: number; resource: GPUBindingResource }[]): GPUBindGroup {
    const layout = pipeline.getBindGroupLayout(0);
    return this.device.createBindGroup({ label, layout, entries });
  }
  private buf(b: GPUBuffer): GPUBindingResource { return { buffer: b }; }

  // ── Rebuild bind groups for both A→B and B→A directions ──────────────────
  private rebuildBindGroups(): void {
    const [pA, pB] = [this.preyStateA, this.preyStateB];
    const [dA, dB] = [this.predStateA, this.predStateB];
    const [fA, fB] = [this.foodBufA,   this.foodBufB];
    const oc = this.outCounters;

    this.bgClear = this.bg("eco-bg-clr", this.pipeClear, [
      { binding: 0, resource: this.buf(this.ubo) },
      { binding: 1, resource: this.buf(this.preyGCount) },
      { binding: 2, resource: this.buf(oc) },
    ]);
    this.bgScatterAtoB = this.bg("eco-bg-sc-AB", this.pipeScatter, [
      { binding: 0, resource: this.buf(this.ubo) }, { binding: 1, resource: this.buf(pA) },
      { binding: 2, resource: this.buf(this.preyAliveA) }, { binding: 3, resource: this.buf(this.preyGCount) },
      { binding: 4, resource: this.buf(this.preyGSlots) }, { binding: 5, resource: this.buf(this.countsUbo) },
    ]);
    this.bgScatterBtoA = this.bg("eco-bg-sc-BA", this.pipeScatter, [
      { binding: 0, resource: this.buf(this.ubo) }, { binding: 1, resource: this.buf(pB) },
      { binding: 2, resource: this.buf(this.preyAliveA) }, { binding: 3, resource: this.buf(this.preyGCount) },
      { binding: 4, resource: this.buf(this.preyGSlots) }, { binding: 5, resource: this.buf(this.countsUbo) },
    ]);
    this.bgPredAtoB = this.bg("eco-bg-pred-AB", this.pipePred, [
      { binding: 0, resource: this.buf(this.ubo) }, { binding: 1, resource: this.buf(dA) },
      { binding: 2, resource: this.buf(dB) },       { binding: 3, resource: this.buf(pA) },
      { binding: 4, resource: this.buf(this.preyAliveA) }, { binding: 5, resource: this.buf(this.predMatBuf) },
      { binding: 6, resource: this.buf(this.predColBuf) }, { binding: 7, resource: this.buf(oc) },
      { binding: 8, resource: this.buf(this.countsUbo) },
    ]);
    this.bgPredBtoA = this.bg("eco-bg-pred-BA", this.pipePred, [
      { binding: 0, resource: this.buf(this.ubo) }, { binding: 1, resource: this.buf(dB) },
      { binding: 2, resource: this.buf(dA) },       { binding: 3, resource: this.buf(pB) },
      { binding: 4, resource: this.buf(this.preyAliveA) }, { binding: 5, resource: this.buf(this.predMatBuf) },
      { binding: 6, resource: this.buf(this.predColBuf) }, { binding: 7, resource: this.buf(oc) },
      { binding: 8, resource: this.buf(this.countsUbo) },
    ]);
    this.bgPreyAtoB = this.bg("eco-bg-prey-AB", this.pipePrey, [
      { binding: 0, resource: this.buf(this.ubo) }, { binding: 1, resource: this.buf(pA) },
      { binding: 2, resource: this.buf(this.preyAliveA) }, { binding: 3, resource: this.buf(this.preyGCount) },
      { binding: 4, resource: this.buf(this.preyGSlots) }, { binding: 5, resource: this.buf(dA) },
      { binding: 6, resource: this.buf(fA) },        { binding: 7, resource: this.buf(pB) },
      { binding: 8, resource: this.buf(oc) },        { binding: 9, resource: this.buf(this.countsUbo) },
    ]);
    this.bgPreyBtoA = this.bg("eco-bg-prey-BA", this.pipePrey, [
      { binding: 0, resource: this.buf(this.ubo) }, { binding: 1, resource: this.buf(pB) },
      { binding: 2, resource: this.buf(this.preyAliveA) }, { binding: 3, resource: this.buf(this.preyGCount) },
      { binding: 4, resource: this.buf(this.preyGSlots) }, { binding: 5, resource: this.buf(dB) },
      { binding: 6, resource: this.buf(fB) },        { binding: 7, resource: this.buf(pA) },
      { binding: 8, resource: this.buf(oc) },        { binding: 9, resource: this.buf(this.countsUbo) },
    ]);
    this.bgPreyRendAtoB = this.bg("eco-bg-pr-AB", this.pipePreyRend, [
      { binding: 0, resource: this.buf(this.ubo) }, { binding: 1, resource: this.buf(pB) },
      { binding: 2, resource: this.buf(this.preyMatBuf) }, { binding: 3, resource: this.buf(this.preyColBuf) },
      { binding: 4, resource: this.buf(oc) },
    ]);
    this.bgPreyRendBtoA = this.bg("eco-bg-pr-BA", this.pipePreyRend, [
      { binding: 0, resource: this.buf(this.ubo) }, { binding: 1, resource: this.buf(pA) },
      { binding: 2, resource: this.buf(this.preyMatBuf) }, { binding: 3, resource: this.buf(this.preyColBuf) },
      { binding: 4, resource: this.buf(oc) },
    ]);
    this.bgFoodAtoB = this.bg("eco-bg-food-AB", this.pipeFood, [
      { binding: 0, resource: this.buf(this.ubo) }, { binding: 1, resource: this.buf(fA) },
      { binding: 2, resource: this.buf(fB) },        { binding: 3, resource: this.buf(oc) },
      { binding: 4, resource: this.buf(this.foodMatBuf) }, { binding: 5, resource: this.buf(this.foodColBuf) },
      { binding: 6, resource: this.buf(this.countsUbo) },
    ]);
    this.bgFoodBtoA = this.bg("eco-bg-food-BA", this.pipeFood, [
      { binding: 0, resource: this.buf(this.ubo) }, { binding: 1, resource: this.buf(fB) },
      { binding: 2, resource: this.buf(fA) },        { binding: 3, resource: this.buf(oc) },
      { binding: 4, resource: this.buf(this.foodMatBuf) }, { binding: 5, resource: this.buf(this.foodColBuf) },
      { binding: 6, resource: this.buf(this.countsUbo) },
    ]);
  }

  /** Seed the ecosystem from CPU arrays. Call before the first dispatch(). */
  seed(prey: { x: Float32Array; z: Float32Array; vx: Float32Array; vz: Float32Array; energy: Float32Array; n: number },
       pred: { x: Float32Array; z: Float32Array; energy: Float32Array; n: number },
       food: { x: Float32Array; z: Float32Array; n: number }) {
    const dev = this.device;

    // Pack prey into Agent structs.
    const preyData = new Float32Array(prey.n * 8);
    for (let i = 0; i < prey.n; i++) {
      preyData[i*8  ] = prey.x[i]; preyData[i*8+1] = prey.z[i];
      preyData[i*8+2] = prey.vx[i]; preyData[i*8+3] = prey.vz[i];
      preyData[i*8+4] = prey.energy[i];
    }
    dev.queue.writeBuffer(this.preyStateA, 0, preyData.buffer, 0, prey.n * 32);

    const predData = new Float32Array(pred.n * 8);
    for (let i = 0; i < pred.n; i++) {
      const a = Math.random() * Math.PI * 2;
      predData[i*8  ] = pred.x[i]; predData[i*8+1] = pred.z[i];
      predData[i*8+2] = Math.cos(a); predData[i*8+3] = Math.sin(a);
      predData[i*8+4] = pred.energy[i];
    }
    dev.queue.writeBuffer(this.predStateA, 0, predData.buffer, 0, pred.n * 32);

    // Pack food: 4 u32 per item = [alive=1, x_bits, z_bits, 0].
    const foodData = new Uint32Array(food.n * 4);
    for (let i = 0; i < food.n; i++) {
      foodData[i*4  ] = 1;
      foodData[i*4+1] = new DataView(new Float32Array([food.x[i]]).buffer).getUint32(0, true);
      foodData[i*4+2] = new DataView(new Float32Array([food.z[i]]).buffer).getUint32(0, true);
    }
    dev.queue.writeBuffer(this.foodBufA, 0, foodData.buffer, 0, food.n * 16);

    // Initialize the exact GPU-side counts (first frame reads these).
    dev.queue.writeBuffer(this.countsUbo, 0, new Uint32Array([prey.n, pred.n, food.n, 0]).buffer);

    this._nPrey = prey.n; this._nPred = pred.n; this._nFood = food.n;
    this._frontIsA = true;
  }

  /** Dispatch one frame. Updates draw counts from last readback. */
  dispatch(p: EcosystemGpuParams): void {
    if (!this._ok) return;
    this._frameCount++;

    // Compute prey grid dim from world size.
    const flockR = Math.sqrt(36); // matches CPU r2=36
    const dim = Math.min(GridDimMax, Math.max(4, Math.floor(Math.min(p.worldX, p.worldZ) / flockR)));
    this._gridDim = dim;
    const cellSize = Math.max(p.worldX, p.worldZ) / dim;

    // Write UBO.
    this.writeUBO(p, dim, cellSize);

    // Dispatch full capacity — the exact agent count lives in countsUbo and each
    // shader guards `i >= counts.n*`, so over-dispatching is correct and cheap.
    const wgPrey = Math.ceil(GpuMaxPrey / 64);
    const wgPred = Math.ceil(GpuMaxPred / 64);
    const wgFood = Math.ceil(GpuMaxFood / 64);
    const wgGrid = Math.ceil(dim * dim / 64);

    const ab = this._frontIsA;
    const bgSc  = ab ? this.bgScatterAtoB   : this.bgScatterBtoA;
    const bgPr  = ab ? this.bgPredAtoB      : this.bgPredBtoA;
    const bgPy  = ab ? this.bgPreyAtoB      : this.bgPreyBtoA;
    const bgPyR = ab ? this.bgPreyRendAtoB  : this.bgPreyRendBtoA;
    const bgFd  = ab ? this.bgFoodAtoB      : this.bgFoodBtoA;

    const enc  = this.device.createCommandEncoder({ label: "eco-compute" });
    const pass = enc.beginComputePass({ label: "eco" });

    pass.setPipeline(this.pipeClear);    pass.setBindGroup(0, this.bgClear);  pass.dispatchWorkgroups(wgGrid);
    pass.setPipeline(this.pipeScatter);  pass.setBindGroup(0, bgSc);          pass.dispatchWorkgroups(wgPrey);
    pass.setPipeline(this.pipePred);     pass.setBindGroup(0, bgPr);          pass.dispatchWorkgroups(wgPred);
    pass.setPipeline(this.pipePrey);     pass.setBindGroup(0, bgPy);          pass.dispatchWorkgroups(wgPrey);
    pass.setPipeline(this.pipePreyRend); pass.setBindGroup(0, bgPyR);         pass.dispatchWorkgroups(wgPrey);
    pass.setPipeline(this.pipeFood);     pass.setBindGroup(0, bgFd);          pass.dispatchWorkgroups(wgFood);
    pass.end();

    // Copy this frame's output counts into: (a) the GPU-side countsUbo so next
    // frame's shaders read the exact count, and (b) the CPU staging buffer for
    // the population graph/readout (async, display-only).
    enc.copyBufferToBuffer(this.outCounters, 0, this.countsUbo, 0, CounterBytes);
    enc.copyBufferToBuffer(this.outCounters, 0, this.staging,   0, CounterBytes);
    this.device.queue.submit([enc.finish()]);

    // Ping-pong via bind-group selection (buffers themselves are NOT swapped —
    // the static AtoB/BtoA bind groups encode the direction).
    this._frontIsA = !this._frontIsA;

    // Update draw counts from the latest readback (1-2 frames stale, imperceptible).
    this.setDrawCounts(this._nPrey, this._nPred, this._nFood);

    // Async readback of this frame's output counters (display only).
    this.readbackAsync();
  }

  private readbackAsync(): void {
    if (this._readbackPending) return;
    this._readbackPending = true;
    this.staging.mapAsync(GPUMapMode.READ, 0, CounterBytes)
      .then(() => {
        const data = new Uint32Array(this.staging.getMappedRange(0, CounterBytes));
        this._nPrey = Math.min(data[0], GpuMaxPrey);
        this._nPred = Math.min(data[1], GpuMaxPred);
        this._nFood = Math.min(data[2], GpuMaxFood);
        this.staging.unmap();
        this._readbackPending = false;
      })
      .catch(() => { this._readbackPending = false; });
  }

  private setDrawCounts(nPrey: number, nPred: number, nFood: number): void {
    setThinInstanceCount(this.preyMesh, nPrey);
    this.preyTi._gpuVersion      = this.preyTi._version;
    this.preyTi._colorGpuVersion = this.preyTi._colorVersion;
    setThinInstanceCount(this.predMesh, nPred);
    this.predTi._gpuVersion      = this.predTi._version;
    this.predTi._colorGpuVersion = this.predTi._colorVersion;
    setThinInstanceCount(this.foodMesh, nFood);
    this.foodTi._gpuVersion      = this.foodTi._version;
    this.foodTi._colorGpuVersion = this.foodTi._colorVersion;
    invalidateRenderBundles(this.engine);
  }

  /** Hide all GPU meshes (call on switch to CPU). */
  hideMeshes(): void {
    this.setDrawCounts(0, 0, 0);
    setMeshVisible(this.preyMesh, false);
    setMeshVisible(this.predMesh, false);
    setMeshVisible(this.foodMesh, false);
  }

  private writeUBO(p: EcosystemGpuParams, dim: number, cellSize: number): void {
    const u  = this.uboData;
    const fv = new DataView(u.buffer);
    u[0]  = p.nPrey; u[1] = p.nPred; u[2] = p.nFood; u[3] = p.foodSpawnCount;
    u[4]  = dim;     u[5] = MaxPerCell;
    fv.setFloat32( 6*4, 1 / cellSize,        true);
    fv.setFloat32( 7*4, p.worldX,            true);
    fv.setFloat32( 8*4, p.worldZ,            true);
    fv.setFloat32( 9*4, p.dt,                true);
    fv.setFloat32(10*4, p.preySpeed,         true);
    fv.setFloat32(11*4, p.predSpeed,         true);
    fv.setFloat32(12*4, p.separation,        true);
    fv.setFloat32(13*4, p.alignment,         true);
    fv.setFloat32(14*4, p.cohesion,          true);
    fv.setFloat32(15*4, p.foodEnergy,        true);
    fv.setFloat32(16*4, p.preyMetabolism,    true);
    u[17] = p.preyFoodToRepro;
    fv.setFloat32(18*4, p.predGain,          true);
    fv.setFloat32(19*4, p.predMetabolism,    true);
    u[20] = p.predPreyToRepro;
    fv.setFloat32(21*4, p.lifespan,          true);
    u[22] = p.maxPrey; u[23] = p.maxPred; u[24] = p.maxFood;
    fv.setFloat32(25*4, 36,                  true); // flockR2 = 6² (matches CPU r2)
    fv.setFloat32(26*4, 6,                   true); // sepR2   = sqrt(6)² (matches CPU sr2)
    fv.setFloat32(27*4, 144,                 true); // fleeR2  = 12² (matches CPU FleeR2)
    fv.setFloat32(28*4, p.collRadius,        true);
    fv.setFloat32(29*4, p.collStrength,      true);
    fv.setFloat32(30*4, 1.8 * 1.8,          true); // foodEatR2 (matches CPU FoodEatR2)
    fv.setFloat32(31*4, 1.8 * 1.8,          true); // preyEatR2 (matches CPU PreyEatR2)
    u[32] = (this._frameCount * 1664525 + 1013904223) >>> 0; // seed
    fv.setFloat32(33*4, PreyFull_GPU,        true);
    fv.setFloat32(34*4, PredFull_GPU,        true);
    u[35] = 0;
    this.device.queue.writeBuffer(this.ubo, 0, u.buffer);
  }

  destroy(): void {
    this.preyMatBuf.destroy(); this.preyColBuf.destroy();
    this.predMatBuf.destroy(); this.predColBuf.destroy();
    this.foodMatBuf.destroy(); this.foodColBuf.destroy();
    this.preyStateA.destroy(); this.preyStateB.destroy();
    this.predStateA.destroy(); this.predStateB.destroy();
    this.foodBufA.destroy();   this.foodBufB.destroy();
    this.preyAliveA.destroy(); this.preyGCount.destroy(); this.preyGSlots.destroy();
    this.outCounters.destroy(); this.staging.destroy(); this.countsUbo.destroy(); this.ubo.destroy();
  }
}
