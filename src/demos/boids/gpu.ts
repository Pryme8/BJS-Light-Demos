/**
 * GPU Boids system using raw WebGPU compute + thin-instanced ShaderMaterial.
 *
 * Compute passes per frame (single encoder / single submit):
 *   1. count   — each boid atomically increments its cell counter
 *   2. scan    — single-thread sequential prefix sum → cellStart[], sentinel
 *                also zeroes cellCount[] so scatter can reuse as offset counter
 *   3. scatter — place each boid index into sortedIndices at its cell slot
 *   4. simulate— each boid scans its 3×3×3 neighbor cells, integrates forces,
 *                writes to the ping-pong back buffer
 *
 * Grid: world ≈ ±22 units. GridDim=8 → cellSize≈5.5u → TotalCells=512.
 * 512 fits in a single sequential scan pass (no workgroup-size limit risk).
 *
 * Rendering: a thin-instanced capsule whose ShaderMaterial vertex shader reads
 * each boid's position + velocity directly from the compute output buffer via
 * @builtin(instance_index) — no CPU readback required.
 */

import {
  addToScene,
  createCapsule,
  createShaderMaterial,
  setShaderStorageBuffer,
  setThinInstanceColors,
  setThinInstanceCount,
  setThinInstances,
} from "@babylonjs/lite";
import type { EngineContext, Mesh, SceneContext } from "@babylonjs/lite";

// ── Constants ──────────────────────────────────────────────────────────────

export const GpuCapacity = 40_000;

const Bound = 20;
const GridDim = 8;
const TotalCells = GridDim * GridDim * GridDim; // 512

// Boid state: 2 × vec4 per boid (32 bytes)
//   posT: vec4(x, y, z, colorT)
//   vel : vec4(vx, vy, vz, 0)
const BytesPerBoid = 32;

// UBO: 16 × f32 = 64 bytes
//  [0]  count       u32
//  [1]  dt          f32 (seconds)
//  [2]  speed       f32
//  [3]  separation  f32
//  [4]  alignment   f32
//  [5]  cohesion    f32
//  [6]  radius      f32
//  [7]  sepRadius   f32
//  [8]  bound       f32
//  [9]  gridDim     u32
//  [10] cellSize    f32
//  [11] invCellSize f32
//  [12-15] padding
const UBOWords = 16;

// ── WGSL common header ──────────────────────────────────────────────────────

const WGSL_COMMON = /* wgsl */ `
struct Params {
  count        : u32,
  dt           : f32,
  speed        : f32,
  separation   : f32,
  alignment    : f32,
  cohesion     : f32,
  radius       : f32,
  sepRadius    : f32,
  bound        : f32,
  gridDim      : u32,
  cellSize     : f32,
  invCellSize  : f32,
  _p0 : u32, _p1 : u32, _p2 : u32, _p3 : u32,
}

struct Boid {
  posT : vec4<f32>,
  vel  : vec4<f32>,
}

fn cellOf(p : vec3<f32>, dim : u32, invCS : f32, bound : f32) -> u32 {
  let half = bound * 1.1;
  let cx = u32(clamp((p.x + half) * invCS, 0.0, f32(dim - 1u)));
  let cy = u32(clamp((p.y + half) * invCS, 0.0, f32(dim - 1u)));
  let cz = u32(clamp((p.z + half) * invCS, 0.0, f32(dim - 1u)));
  return cx + cy * dim + cz * dim * dim;
}
`;

// ── Pass 0: clear cell counts before each frame ────────────────────────────

const WGSL_CLEAR = /* wgsl */ `
${WGSL_COMMON}
@group(0) @binding(0) var<uniform>             params    : Params;
@group(0) @binding(3) var<storage, read_write> cellCount : array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.gridDim * params.gridDim * params.gridDim) { return; }
  atomicStore(&cellCount[i], 0u);
}
`;

// ── Pass 1: count boids per cell (atomic) ──────────────────────────────────

const WGSL_COUNT = /* wgsl */ `
${WGSL_COMMON}
@group(0) @binding(0) var<uniform>             params    : Params;
@group(0) @binding(1) var<storage, read>       front     : array<Boid>;
@group(0) @binding(3) var<storage, read_write> cellCount : array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  let c = cellOf(front[i].posT.xyz, params.gridDim, params.invCellSize, params.bound);
  atomicAdd(&cellCount[c], 1u);
}
`;

// ── Pass 2: sequential prefix scan → cellStart, reset cellCount ─────────────

const WGSL_SCAN = /* wgsl */ `
${WGSL_COMMON}
@group(0) @binding(0) var<uniform>             params     : Params;
@group(0) @binding(3) var<storage, read_write> cellCount  : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cellStart  : array<u32>;

@compute @workgroup_size(1)
fn main() {
  let total = params.gridDim * params.gridDim * params.gridDim;
  var sum = 0u;
  for (var i = 0u; i <= total; i++) {
    cellStart[i] = sum;
    if (i < total) {
      let cnt = atomicLoad(&cellCount[i]);
      sum += cnt;
      atomicStore(&cellCount[i], 0u);  // reset for scatter pass
    }
  }
}
`;

// ── Pass 3: scatter boid indices into sorted slots ──────────────────────────

const WGSL_SCATTER = /* wgsl */ `
${WGSL_COMMON}
@group(0) @binding(0) var<uniform>             params        : Params;
@group(0) @binding(1) var<storage, read>       front         : array<Boid>;
@group(0) @binding(3) var<storage, read_write> cellCount     : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cellStart     : array<u32>;
@group(0) @binding(5) var<storage, read_write> sortedIndices : array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  let c   = cellOf(front[i].posT.xyz, params.gridDim, params.invCellSize, params.bound);
  let off = cellStart[c] + atomicAdd(&cellCount[c], 1u);
  sortedIndices[off] = i;
}
`;

// ── Pass 4: simulate — neighbor scan + integrate + ping-pong write ─────────

const WGSL_SIMULATE = /* wgsl */ `
${WGSL_COMMON}
@group(0) @binding(0) var<uniform>             params        : Params;
@group(0) @binding(1) var<storage, read>       front         : array<Boid>;
@group(0) @binding(2) var<storage, read_write> back          : array<Boid>;
@group(0) @binding(4) var<storage, read_write> cellStart     : array<u32>;
@group(0) @binding(5) var<storage, read_write> sortedIndices : array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  let dim    = params.gridDim;
  let pos    = front[i].posT.xyz;
  let vel    = front[i].vel.xyz;
  let half   = params.bound * 1.1;
  let gx     = i32(clamp((pos.x + half) * params.invCellSize, 0.0, f32(dim - 1u)));
  let gy     = i32(clamp((pos.y + half) * params.invCellSize, 0.0, f32(dim - 1u)));
  let gz     = i32(clamp((pos.z + half) * params.invCellSize, 0.0, f32(dim - 1u)));

  let r2  = params.radius    * params.radius;
  let sr2 = params.sepRadius * params.sepRadius;

  var sepF   = vec3<f32>(0.0);
  var aliSum = vec3<f32>(0.0);
  var cohSum = vec3<f32>(0.0);
  var nc     = 0u;
  var ns     = 0u;

  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let nx = gx + dx;
        let ny = gy + dy;
        let nz = gz + dz;
        if (nx < 0 || ny < 0 || nz < 0) { continue; }
        let unx = u32(nx); let uny = u32(ny); let unz = u32(nz);
        if (unx >= dim || uny >= dim || unz >= dim) { continue; }
        let cell  = unx + uny * dim + unz * dim * dim;
        let start = cellStart[cell];
        let endC  = cellStart[cell + 1u];
        for (var k = start; k < endC; k++) {
          let j = sortedIndices[k];
          if (j == i) { continue; }
          let dp = front[j].posT.xyz - pos;
          let d2 = dot(dp, dp);
          if (d2 >= r2) { continue; }
          aliSum += front[j].vel.xyz;
          cohSum += front[j].posT.xyz;
          nc++;
          if (d2 < sr2) { sepF -= dp; ns++; }
        }
      }
    }
  }

  var force = vec3<f32>(0.0);
  if (ns > 0u) { force += sepF * params.separation; }
  if (nc > 0u) {
    let invN = 1.0 / f32(nc);
    force += (aliSum * invN - vel) * params.alignment;
    force += (cohSum * invN - pos) * params.cohesion * 0.05;
  }

  // Boundary avoidance
  let margin = params.bound * 0.15;
  let bx = abs(pos.x) - (params.bound - margin);
  let by = abs(pos.y) - (params.bound * 0.5 - margin);
  let bz = abs(pos.z) - (params.bound - margin);
  if (bx > 0.0) { force.x -= sign(pos.x) * bx * 0.5; }
  if (by > 0.0) { force.y -= sign(pos.y) * by * 0.5; }
  if (bz > 0.0) { force.z -= sign(pos.z) * bz * 0.5; }

  var newVel = vel + force * params.dt;
  let vLen   = length(newVel);
  if (vLen > 0.001) { newVel = newVel * (params.speed / vLen); }

  let newPos  = pos + newVel * params.dt;
  let colorT  = clamp(vLen / (params.speed * 1.5), 0.0, 1.0);

  back[i].posT = vec4<f32>(newPos, colorT);
  back[i].vel  = vec4<f32>(newVel, 0.0);
}
`;

// ── Render WGSL ─────────────────────────────────────────────────────────────
// Lib prepends the following to BOTH vertex and fragment modules
// (from shader-pipeline.js buildShaderPrelude):
//
//   ${SCENE_UBO_WGSL}
//   struct ShaderSystemUniforms { worldViewProjection: mat4x4<f32>, ... }
//   @group(1) @binding(0) var<uniform> shaderSystem: ShaderSystemUniforms;
//   // (no custom uniforms → no shaderUniforms block)
//   @group(1) @binding(1) var<storage, read> boids: array<Boid>;
//   struct VertexInput {
//     @location(0) position: vec3<f32>,
//     @location(1) normal: vec3<f32>,
//     // thin-instance module appends world0..world3 at locations 2..5
//   };
//
// Entry points must be named mainVertex / mainFragment.
// System uniforms are read via shaderSystem.<name>, NOT shaderUniforms.
// Boid struct must be defined in BOTH vertex and fragment sources because
// the lib injects the var<storage> boids decl into both modules.

const WGSL_VERTEX = /* wgsl */ `
struct Boid {
  posT : vec4<f32>,
  vel  : vec4<f32>,
}

struct VertexOutput {
  @builtin(position) pos   : vec4<f32>,
  @location(0)       norm  : vec3<f32>,
  @location(1)       color : vec3<f32>,
}

// Shortest-arc rotation matrix: align local +Y to direction d.
fn rotMat(d : vec3<f32>) -> mat3x3<f32> {
  let len = length(d);
  let f   = select(vec3<f32>(0.0, 1.0, 0.0), d / len, len > 1e-6);
  if (f.y < -0.9999) {
    return mat3x3<f32>(
      vec3<f32>( 1.0,  0.0,  0.0),
      vec3<f32>( 0.0, -1.0,  0.0),
      vec3<f32>( 0.0,  0.0, -1.0)
    );
  }
  // Shortest-arc quaternion from +Y to f: qy=0, so only qx,qz,qw are non-zero.
  // Column-major 3×3 rotation matrix for q=(qx,0,qz,qw):
  //   col0 = (1-zz,   wz,  xz)
  //   col1 = ( -wz, 1-xx-zz,  wx)
  //   col2 = (  xz,  -wx, 1-xx)
  let w   = 1.0 + f.y;
  let inv = 1.0 / sqrt(2.0 * w);
  let qx  =  f.z * inv;
  let qz  = -f.x * inv;
  let qw  =  w   * inv;
  let x2  = qx + qx; let z2 = qz + qz;
  let xx  = qx * x2; let xz = qx * z2;
  let zz  = qz * z2; let wx = qw * x2; let wz = qw * z2;
  return mat3x3<f32>(
    vec3<f32>(1.0 - zz,        wz,            xz      ),   // col 0
    vec3<f32>(-wz,             1.0 - xx - zz, wx      ),   // col 1
    vec3<f32>(xz,             -wx,            1.0 - xx)    // col 2
  );
}

@vertex
fn mainVertex(
  input : VertexInput,
  @builtin(instance_index) instanceIndex : u32,
) -> VertexOutput {
  let boid = boids[instanceIndex];
  let R    = rotMat(boid.vel.xyz);
  let wpos = R * input.position + boid.posT.xyz;
  let wnorm = R * input.normal;

  let t   = boid.posT.w;
  let col = vec3<f32>(t, 1.0 - t * 0.8, 1.0 - t * 0.4);

  var out  : VertexOutput;
  out.pos   = shaderSystem.worldViewProjection * vec4<f32>(wpos, 1.0);
  out.norm  = wnorm;
  out.color = col;
  return out;
}
`;

const WGSL_FRAGMENT = /* wgsl */ `
// Must be defined here too — the prelude injects
// var<storage, read> boids: array<Boid> into both vertex and fragment modules.
struct Boid {
  posT : vec4<f32>,
  vel  : vec4<f32>,
}

struct FragInput {
  @location(0) norm  : vec3<f32>,
  @location(1) color : vec3<f32>,
}

@fragment
fn mainFragment(in : FragInput) -> @location(0) vec4<f32> {
  let n   = normalize(in.norm);
  let lit = clamp(dot(n, normalize(vec3<f32>(0.3, 1.0, 0.5))), 0.0, 1.0);
  return vec4<f32>(in.color * (0.35 + 0.65 * lit), 1.0);
}
`;

// ── GpuBoids class ─────────────────────────────────────────────────────────

export interface GpuBoidsParams {
  count: number;
  speed: number;
  separation: number;
  alignment: number;
  cohesion: number;
  radius: number;
  separationRadius: number;
}

export class GpuBoids {
  readonly mesh: Mesh;
  readonly material: ReturnType<typeof createShaderMaterial>;

  private readonly device: GPUDevice;

  private stateA: GPUBuffer;
  private stateB: GPUBuffer;
  /** Buffer being read this frame (compute input). */
  private front: GPUBuffer;
  /** Buffer being written this frame (compute output). */
  private back: GPUBuffer;

  // Grid buffers
  private readonly cellCount: GPUBuffer;   // atomic<u32>[TotalCells]
  private readonly cellStart: GPUBuffer;   // u32[TotalCells + 1] (includes sentinel)
  private readonly sortedIndices: GPUBuffer; // u32[GpuCapacity]

  private readonly ubo: GPUBuffer;
  private readonly uboData: Float32Array;

  private readonly clearPipeline: GPUComputePipeline;
  private readonly countPipeline: GPUComputePipeline;
  private readonly scanPipeline: GPUComputePipeline;
  private readonly scatterPipeline: GPUComputePipeline;
  private readonly simulatePipeline: GPUComputePipeline;

  private readonly bgl: GPUBindGroupLayout;
  /** Bind group with stateA as front, stateB as back. */
  private readonly bgA: GPUBindGroup;
  /** Bind group with stateB as front, stateA as back. */
  private readonly bgB: GPUBindGroup;

  /** False if pipeline creation failed; the scene falls back to CPU mode. */
  private _ok = false;
  get ok(): boolean { return this._ok; }

  constructor(engine: EngineContext, scene: SceneContext) {

    const dev = (engine as unknown as { _device: GPUDevice })._device;
    this.device = dev;

    // ── Render mesh ────────────────────────────────────────────────────────
    const capsule = createCapsule(engine, { height: 1.2, radius: 0.35, tessellation: 6 });
    addToScene(scene, capsule);
    this.mesh = capsule;

    // Establish the mesh as thin-instanced (matrices are ignored — vertex shader
    // reads boid state directly from the storage buffer).
    const identities = new Float32Array(GpuCapacity * 16);
    for (let i = 0; i < GpuCapacity; i++) {
      const b = i * 16;
      identities[b] = 1; identities[b + 5] = 1; identities[b + 10] = 1; identities[b + 15] = 1;
    }
    const colors = new Float32Array(GpuCapacity * 4).fill(1);
    setThinInstances(capsule, identities, GpuCapacity);
    setThinInstanceColors(capsule, colors);
    setThinInstanceCount(capsule, 0);

    // ── GPU buffers ────────────────────────────────────────────────────────
    const stateUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.stateA = dev.createBuffer({ label: "boidsA", size: GpuCapacity * BytesPerBoid, usage: stateUsage });
    this.stateB = dev.createBuffer({ label: "boidsB", size: GpuCapacity * BytesPerBoid, usage: stateUsage });
    this.front = this.stateA;
    this.back = this.stateB;

    const gridUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.cellCount = dev.createBuffer({ label: "cellCount", size: TotalCells * 4, usage: gridUsage });
    this.cellStart = dev.createBuffer({
      label: "cellStart",
      size: (TotalCells + 1) * 4, // +1 for the sentinel
      usage: gridUsage,
    });
    this.sortedIndices = dev.createBuffer({
      label: "sortedIdx",
      size: GpuCapacity * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    this.uboData = new Float32Array(UBOWords);
    this.ubo = dev.createBuffer({
      label: "boidsUBO",
      size: UBOWords * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ── Compute pipelines ──────────────────────────────────────────────────
    let ok = true;
    try {
      this.bgl = this.buildBGL();
      this.clearPipeline    = this.buildComputePipeline("clear",    WGSL_CLEAR);
      this.countPipeline    = this.buildComputePipeline("count",    WGSL_COUNT);
      this.scanPipeline     = this.buildComputePipeline("scan",     WGSL_SCAN);
      this.scatterPipeline  = this.buildComputePipeline("scatter",  WGSL_SCATTER);
      this.simulatePipeline = this.buildComputePipeline("simulate", WGSL_SIMULATE);
      this.bgA = this.buildBindGroup("A", this.stateA, this.stateB);
      this.bgB = this.buildBindGroup("B", this.stateB, this.stateA);
    } catch (e) {
      console.error("[GpuBoids] pipeline build failed:", e);
      ok = false;
      this.bgl = null!;
      this.clearPipeline = null!; this.countPipeline = null!;
      this.scanPipeline = null!; this.scatterPipeline = null!;
      this.simulatePipeline = null!;
      this.bgA = null!; this.bgB = null!;
      this.material = null!;
      return;
    }

    // ── Render material ────────────────────────────────────────────────────
    // The lib injects:
    //   @group(1) @binding(1) var<uniform> shaderUniforms: ShaderUniforms;
    //     (auto-updated with viewProjection each frame)
    //   @group(1) @binding(2) var<storage, read> boids: array<Boid>;
    //   struct VertexInput { @location(0) position, @location(1) normal, ... };
    try {
      this.material = createShaderMaterial({
        name: "gpuBoidsMat",
        vertexSource: WGSL_VERTEX,
        fragmentSource: WGSL_FRAGMENT,
        attributes: ["position", "normal"],
        uniforms: ["worldViewProjection"],
        storageBuffers: [{ name: "boids", type: "array<Boid>" }],
      });
      capsule.material = this.material;
      setShaderStorageBuffer(this.material, "boids", this.front);
    } catch (e) {
      console.error("[GpuBoids] ShaderMaterial build failed:", e);
      this.material = null!;
      return;
    }

    this._ok = ok;
  }

  private buildBGL(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "boidsBGL",
      entries: [
        // 0: params UBO
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        // 1: front (read-only state)
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        // 2: back (write for simulate; unused by other passes but must be bound)
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        // 3: cellCount (atomic read_write)
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        // 4: cellStart (read_write in scan; read in scatter/simulate)
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        // 5: sortedIndices (write in scatter; read in simulate)
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
  }

  private buildComputePipeline(label: string, wgsl: string): GPUComputePipeline {
    const module = this.device.createShaderModule({ label, code: wgsl });
    return this.device.createComputePipeline({
      label,
      layout: this.device.createPipelineLayout({
        label: `${label}-layout`,
        bindGroupLayouts: [this.bgl],
      }),
      compute: { module, entryPoint: "main" },
    });
  }

  private buildBindGroup(label: string, front: GPUBuffer, back: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({
      label: `boidsBG-${label}`,
      layout: this.bgl,
      entries: [
        { binding: 0, resource: { buffer: this.ubo } },
        { binding: 1, resource: { buffer: front } },
        { binding: 2, resource: { buffer: back } },
        { binding: 3, resource: { buffer: this.cellCount } },
        { binding: 4, resource: { buffer: this.cellStart } },
        { binding: 5, resource: { buffer: this.sortedIndices } },
      ],
    });
  }

  /** Write CPU boid state into the GPU front buffer (instant, no sync). */
  seedFrom(
    count: number,
    px: Float32Array, py: Float32Array, pz: Float32Array,
    vx: Float32Array, vy: Float32Array, vz: Float32Array,
  ): void {
    const data = new Float32Array(GpuCapacity * 8);
    for (let i = 0; i < count; i++) {
      const b = i * 8;
      data[b] = px[i]; data[b + 1] = py[i]; data[b + 2] = pz[i]; data[b + 3] = 0;
      data[b + 4] = vx[i]; data[b + 5] = vy[i]; data[b + 6] = vz[i]; data[b + 7] = 0;
    }
    // Seed remaining slots with random positions so they aren't all at origin
    for (let i = count; i < GpuCapacity; i++) {
      const b = i * 8;
      const angle = Math.random() * Math.PI * 2;
      data[b] = (Math.random() - 0.5) * Bound * 1.5;
      data[b + 1] = (Math.random() - 0.5) * Bound;
      data[b + 2] = (Math.random() - 0.5) * Bound * 1.5;
      data[b + 4] = Math.cos(angle); data[b + 5] = 0.1; data[b + 6] = Math.sin(angle);
    }
    this.device.queue.writeBuffer(this.front, 0, data);
  }

  /**
   * Async GPU→CPU readback for the current front buffer.
   * Returns once the data has been copied into the caller's arrays.
   * Used for the GPU→CPU mode switch handoff.
   */
  async readbackInto(
    count: number,
    px: Float32Array, py: Float32Array, pz: Float32Array,
    vx: Float32Array, vy: Float32Array, vz: Float32Array,
  ): Promise<void> {
    const readSize = count * BytesPerBoid;
    const staging = this.device.createBuffer({
      size: readSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.front, 0, staging, 0, readSize);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ, 0, readSize);
    const src = new Float32Array(staging.getMappedRange(0, readSize));
    for (let i = 0; i < count; i++) {
      const b = i * 8;
      px[i] = src[b]; py[i] = src[b + 1]; pz[i] = src[b + 2];
      vx[i] = src[b + 4]; vy[i] = src[b + 5]; vz[i] = src[b + 6];
    }
    staging.unmap();
    staging.destroy();
  }

  /** Dispatch one frame of compute. dt is in milliseconds. */
  dispatch(dt: number, params: GpuBoidsParams): void {
    if (!this.ok) return;
    const n = params.count;
    const cellSize = (Bound * 2 * 1.1) / GridDim;

    const u = this.uboData;
    // Write u32 count via a DataView to avoid float reinterpretation
    new Uint32Array(u.buffer)[0] = n;
    u[1] = dt * 0.001;
    u[2] = params.speed;
    u[3] = params.separation;
    u[4] = params.alignment;
    u[5] = params.cohesion;
    u[6] = params.radius;
    u[7] = params.separationRadius;
    u[8] = Bound;
    new Uint32Array(u.buffer)[9] = GridDim;
    u[10] = cellSize;
    u[11] = 1 / cellSize;
    this.device.queue.writeBuffer(this.ubo, 0, u);

    const bg = this.front === this.stateA ? this.bgA : this.bgB;
    const wgN64 = Math.ceil(n / 64);

    const enc = this.device.createCommandEncoder({ label: "boids-compute" });
    const pass = enc.beginComputePass({ label: "boids" });
    const wgCells = Math.ceil(TotalCells / 64); // ceil(512/64) = 8

    // 0. Clear cell counts (required every frame: scatter re-increments cellCount,
    //    so without clearing it doubles each frame from frame 2 onward)
    pass.setPipeline(this.clearPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgCells);

    // 1. Count
    pass.setPipeline(this.countPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgN64);

    // 2. Scan (prefix sum, single thread)
    pass.setPipeline(this.scanPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1);

    // 3. Scatter
    pass.setPipeline(this.scatterPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgN64);

    // 4. Simulate
    pass.setPipeline(this.simulatePipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgN64);

    pass.end();
    this.device.queue.submit([enc.finish()]);

    // Ping-pong: swap front ↔ back
    const tmp = this.front;
    this.front = this.back;
    this.back = tmp;

    setShaderStorageBuffer(this.material, "boids", this.front);
    setThinInstanceCount(this.mesh, n);
  }

  destroy(): void {
    this.stateA.destroy();
    this.stateB.destroy();
    this.cellCount.destroy();
    this.cellStart.destroy();
    this.sortedIndices.destroy();
    this.ubo.destroy();
  }
}
