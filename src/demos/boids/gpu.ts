/**
 * GPU Boids — fully GPU-driven rendering (no readback), bucketed uniform grid.
 *
 * Boids are O(n × neighbors), unlike independent particles (O(n)). To keep it
 * fast at 200k we bound the neighbor work two ways:
 *
 *   1. Bucketed uniform grid with FIXED per-cell capacity (MaxPerCell). Each
 *      boid is scattered into its cell's bucket via one atomicAdd; overflow past
 *      MaxPerCell is dropped. The simulate pass scans the 3×3×3 neighbor cells,
 *      visiting at most 27×MaxPerCell candidates per boid — a hard cap. This also
 *      removes the serial prefix-sum scan (which itself doesn't scale).
 *   2. Density-scaled domain: the world bound and grid resolution grow with the
 *      boid count so boids-per-cell (and thus candidates per boid) stays roughly
 *      constant — genuinely O(n).
 *
 * Rendering is the ordinary PBR thin-instance path (same as the CPU boids). The
 * simulate pass writes each boid's column-major TRS mat4 into matBuf and RGBA
 * into colBuf; those are OWNED here and handed to Lite as the mesh's instance
 * matrix/color vertex buffers, with versions synced so Lite never overwrites them.
 */

import {
  addToScene,
  createCapsule,
  createPbrMaterial,
  invalidateRenderBundles,
  setThinInstanceColors,
  setThinInstanceCount,
  setThinInstances,
} from "@babylonjs/lite";
import type { EngineContext, Mesh, SceneContext } from "@babylonjs/lite";

// ── Constants ──────────────────────────────────────────────────────────────

export const GpuCapacity = 200_000;

const GridDimMax = 32;
const MaxCells   = GridDimMax * GridDimMax * GridDimMax; // 32768
const MaxPerCell = 64;                                    // hard neighbor-candidate cap per cell

// Boid state: 2 × vec4<f32> per boid = 32 bytes
const BytesPerBoid = 32;

// UBO: 16 × 4 bytes
const UBOWords = 16;

/**
 * Density-scaled world half-extent for a given boid count.
 * Grows ∝ cbrt(count) above a baseline so density (and neighbors/boid) stays
 * roughly constant. Floored at 20 so small counts keep the original framing.
 */
export function gpuBoundForCount(count: number): number {
  return Math.max(20, 20 * Math.cbrt(count / 2000));
}

// ── WGSL common ─────────────────────────────────────────────────────────────

const WGSL_COMMON = /* wgsl */ `
const MAX_PER_CELL : u32 = ${MaxPerCell}u;

struct Params {
  count       : u32,
  dt          : f32,
  speed       : f32,
  separation  : f32,
  alignment   : f32,
  cohesion    : f32,
  radius      : f32,
  sepRadius   : f32,
  bound       : f32,
  gridDim     : u32,
  cellSize    : f32,
  invCellSize : f32,
  cellCountU   : u32,
  collRadius   : f32,
  collStrength : f32,
  collEnabled  : u32,
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

// ── Pass 0: clear cell counts ────────────────────────────────────────────────

const WGSL_CLEAR = /* wgsl */ `
${WGSL_COMMON}
@group(0) @binding(0) var<uniform>             params    : Params;
@group(0) @binding(3) var<storage, read_write> cellCount : array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.cellCountU) { return; }
  atomicStore(&cellCount[i], 0u);
}
`;

// ── Pass 1: scatter boids into fixed-size cell buckets ───────────────────────

const WGSL_SCATTER = /* wgsl */ `
${WGSL_COMMON}
@group(0) @binding(0) var<uniform>             params    : Params;
@group(0) @binding(1) var<storage, read>       front     : array<Boid>;
@group(0) @binding(3) var<storage, read_write> cellCount : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cellBoids : array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  let c    = cellOf(front[i].posT.xyz, params.gridDim, params.invCellSize, params.bound);
  let slot = atomicAdd(&cellCount[c], 1u);
  if (slot < MAX_PER_CELL) {
    cellBoids[c * MAX_PER_CELL + slot] = i;
  }
}
`;

// ── Pass 2: simulate — scan 3×3×3 buckets, integrate, write matrix + color ───

const WGSL_SIMULATE = /* wgsl */ `
${WGSL_COMMON}
@group(0) @binding(0) var<uniform>             params    : Params;
@group(0) @binding(1) var<storage, read>       front     : array<Boid>;
@group(0) @binding(2) var<storage, read_write> back      : array<Boid>;
@group(0) @binding(3) var<storage, read_write> cellCount : array<u32>;
@group(0) @binding(4) var<storage, read_write> cellBoids : array<u32>;
@group(0) @binding(5) var<storage, read_write> matBuf    : array<f32>;
@group(0) @binding(6) var<storage, read_write> colBuf    : array<f32>;

fn upToDirQuat(d : vec3<f32>) -> vec4<f32> {
  let len = length(d);
  if (len < 1e-6) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
  let f = d / len;
  if (f.y < -0.999999) { return vec4<f32>(1.0, 0.0, 0.0, 0.0); }
  let w   = 1.0 + f.y;
  let inv = 1.0 / sqrt(2.0 * w);
  return vec4<f32>(f.z * inv, 0.0, -f.x * inv, w * inv);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  let dim  = params.gridDim;
  let pos  = front[i].posT.xyz;
  let vel  = front[i].vel.xyz;
  let half = params.bound * 1.1;
  let gx   = i32(clamp((pos.x + half) * params.invCellSize, 0.0, f32(dim - 1u)));
  let gy   = i32(clamp((pos.y + half) * params.invCellSize, 0.0, f32(dim - 1u)));
  let gz   = i32(clamp((pos.z + half) * params.invCellSize, 0.0, f32(dim - 1u)));

  let r2  = params.radius    * params.radius;
  let sr2 = params.sepRadius * params.sepRadius;

  var sepF     = vec3<f32>(0.0);
  var aliSum   = vec3<f32>(0.0);
  var cohSum   = vec3<f32>(0.0);
  var collPush = vec3<f32>(0.0);
  var nc       = 0u;
  var ns     = 0u;

  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let nx = gx + dx;  let ny = gy + dy;  let nz = gz + dz;
        if (nx < 0 || ny < 0 || nz < 0) { continue; }
        let unx = u32(nx);  let uny = u32(ny);  let unz = u32(nz);
        if (unx >= dim || uny >= dim || unz >= dim) { continue; }
        let cell = unx + uny * dim + unz * dim * dim;
        let cnt  = min(cellCount[cell], MAX_PER_CELL);
        let base = cell * MAX_PER_CELL;
        for (var s = 0u; s < cnt; s++) {
          let j = cellBoids[base + s];
          if (j == i) { continue; }
          let dp = front[j].posT.xyz - pos;
          let d2 = dot(dp, dp);
          if (d2 >= r2) { continue; }
          aliSum += front[j].vel.xyz;
          cohSum += front[j].posT.xyz;
          nc++;
          if (d2 < sr2) { sepF -= dp; ns++; }
          // Soft collision pushout (independent of flocking, same scan pass).
          if (params.collEnabled == 1u) {
            let collDist = 2.0 * params.collRadius;
            if (d2 < collDist * collDist && d2 > 1e-8) {
              let d    = sqrt(d2);
              let push = (collDist - d) * 0.5 * params.collStrength / d;
              collPush -= dp * push;
            }
          }
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

  let newPos   = pos + newVel * params.dt;
  // Apply positional collision pushout after normal integration.
  let finalPos = newPos + collPush;
  let colorT   = clamp(vLen / (params.speed * 1.5), 0.0, 1.0);

  back[i].posT = vec4<f32>(finalPos, colorT);
  back[i].vel  = vec4<f32>(newVel, 0.0);

  // ── Column-major TRS mat4 (matches AgentBuffer.writeTransform) ───────────
  let q  = upToDirQuat(newVel);
  let qx = q.x; let qy = q.y; let qz = q.z; let qw = q.w;
  let x2 = qx + qx; let y2 = qy + qy; let z2 = qz + qz;
  let xx = qx * x2; let xy = qx * y2; let xz = qx * z2;
  let yy = qy * y2; let yz = qy * z2; let zz = qz * z2;
  let wx = qw * x2; let wy = qw * y2; let wz = qw * z2;

  let mb = i * 16u;
  matBuf[mb +  0u] = 1.0 - (yy + zz); matBuf[mb +  1u] = xy + wz;         matBuf[mb +  2u] = xz - wy;         matBuf[mb +  3u] = 0.0;
  matBuf[mb +  4u] = xy - wz;         matBuf[mb +  5u] = 1.0 - (xx + zz); matBuf[mb +  6u] = yz + wx;         matBuf[mb +  7u] = 0.0;
  matBuf[mb +  8u] = xz + wy;         matBuf[mb +  9u] = yz - wx;         matBuf[mb + 10u] = 1.0 - (xx + yy); matBuf[mb + 11u] = 0.0;
  matBuf[mb + 12u] = finalPos.x;      matBuf[mb + 13u] = finalPos.y;      matBuf[mb + 14u] = finalPos.z;      matBuf[mb + 15u] = 1.0;

  let cb = i * 4u;
  colBuf[cb +  0u] = colorT;
  colBuf[cb +  1u] = 1.0 - colorT * 0.8;
  colBuf[cb +  2u] = 1.0 - colorT * 0.4;
  colBuf[cb +  3u] = 1.0;
}
`;

// ── Internal thin-instance shape we mutate for buffer ownership ──────────────

type ThinInstData = {
  _gpuBuffer: GPUBuffer | null;
  _gpuBufferStorage: boolean;
  _gpuVersion: number;
  _version: number;
  _colorGpuBuffer: GPUBuffer | null;
  _colorGpuBufferStorage: boolean;
  _colorGpuVersion: number;
  _colorVersion: number;
};

// ── GpuBoids ─────────────────────────────────────────────────────────────────

export interface GpuBoidsParams {
  count: number;
  speed: number;
  separation: number;
  alignment: number;
  cohesion: number;
  radius: number;
  separationRadius: number;
  collision: boolean;
  collisionRadius: number;
  collisionStrength: number;
}

export class GpuBoids {
  readonly mesh: Mesh;

  private readonly engine: EngineContext;
  private readonly device: GPUDevice;
  private readonly ti: ThinInstData;

  private stateA: GPUBuffer;
  private stateB: GPUBuffer;
  private front: GPUBuffer;
  private back: GPUBuffer;

  private readonly matBuf: GPUBuffer;
  private readonly colBuf: GPUBuffer;

  private readonly cellCount: GPUBuffer;
  private readonly cellBoids: GPUBuffer;

  private readonly ubo: GPUBuffer;
  private readonly uboData: Float32Array;

  private readonly bgl: GPUBindGroupLayout;
  private readonly clearPipeline: GPUComputePipeline;
  private readonly scatterPipeline: GPUComputePipeline;
  private readonly simulatePipeline: GPUComputePipeline;
  private readonly bgA: GPUBindGroup;
  private readonly bgB: GPUBindGroup;

  private _ok = false;
  get ok(): boolean { return this._ok; }

  constructor(engine: EngineContext, scene: SceneContext) {
    this.engine = engine;
    const dev = (engine as unknown as { _device: GPUDevice })._device;
    this.device = dev;

    // Owned instance vertex buffers.
    const instUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST;
    this.matBuf = dev.createBuffer({ label: "boids-mat", size: GpuCapacity * 64, usage: instUsage });
    this.colBuf = dev.createBuffer({ label: "boids-col", size: GpuCapacity * 16, usage: instUsage });

    // PBR thin-instanced mesh (same render path as CPU boids).
    const capsule = createCapsule(engine, { height: 1.2, radius: 0.35, tessellation: 6 });
    capsule.material = createPbrMaterial({ baseColorFactor: [0.05, 0.8, 1, 1], metallicFactor: 0.2, roughnessFactor: 0.5 });
    addToScene(scene, capsule);
    this.mesh = capsule;

    const identities = new Float32Array(GpuCapacity * 16);
    for (let i = 0; i < GpuCapacity; i++) {
      const b = i * 16;
      identities[b] = 1; identities[b + 5] = 1; identities[b + 10] = 1; identities[b + 15] = 1;
    }
    const colorsPlaceholder = new Float32Array(GpuCapacity * 4).fill(1);

    setThinInstances(capsule, identities, GpuCapacity);
    setThinInstanceColors(capsule, colorsPlaceholder);
    setThinInstanceCount(capsule, 0);

    const ti = (capsule as unknown as { thinInstances: ThinInstData }).thinInstances;
    ti._gpuBuffer             = this.matBuf;
    ti._gpuBufferStorage      = false;
    ti._gpuVersion            = ti._version;
    ti._colorGpuBuffer        = this.colBuf;
    ti._colorGpuBufferStorage = false;
    ti._colorGpuVersion       = ti._colorVersion;
    this.ti = ti;

    // State ping-pong.
    const stateUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.stateA = dev.createBuffer({ label: "boids-stateA", size: GpuCapacity * BytesPerBoid, usage: stateUsage });
    this.stateB = dev.createBuffer({ label: "boids-stateB", size: GpuCapacity * BytesPerBoid, usage: stateUsage });
    this.front = this.stateA;
    this.back  = this.stateB;

    // Grid: fixed-capacity buckets.
    const gridUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.cellCount = dev.createBuffer({ label: "cellCount", size: MaxCells * 4,              usage: gridUsage });
    this.cellBoids = dev.createBuffer({ label: "cellBoids", size: MaxCells * MaxPerCell * 4, usage: gridUsage });

    this.uboData = new Float32Array(UBOWords);
    this.ubo     = dev.createBuffer({ label: "boidsUBO", size: UBOWords * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    try {
      this.bgl              = this.buildBGL();
      this.clearPipeline    = this.buildComputePipeline("clear",    WGSL_CLEAR);
      this.scatterPipeline  = this.buildComputePipeline("scatter",  WGSL_SCATTER);
      this.simulatePipeline = this.buildComputePipeline("simulate", WGSL_SIMULATE);
      this.bgA = this.buildBG("A", this.stateA, this.stateB);
      this.bgB = this.buildBG("B", this.stateB, this.stateA);
    } catch (e) {
      console.error("[GpuBoids] pipeline build failed:", e);
      this.bgl = null!; this.clearPipeline = null!; this.scatterPipeline = null!; this.simulatePipeline = null!;
      this.bgA = null!; this.bgB = null!;
      return;
    }

    this._ok = true;
  }

  private buildBGL(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "boids-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // cellCount
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // cellBoids
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // matBuf
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // colBuf
      ],
    });
  }

  private buildComputePipeline(label: string, wgsl: string): GPUComputePipeline {
    const mod = this.device.createShaderModule({ label, code: wgsl });
    return this.device.createComputePipeline({
      label,
      layout: this.device.createPipelineLayout({ label: `${label}-layout`, bindGroupLayouts: [this.bgl] }),
      compute: { module: mod, entryPoint: "main" },
    });
  }

  private buildBG(label: string, front: GPUBuffer, back: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({
      label: `boids-bg-${label}`,
      layout: this.bgl,
      entries: [
        { binding: 0, resource: { buffer: this.ubo } },
        { binding: 1, resource: { buffer: front } },
        { binding: 2, resource: { buffer: back } },
        { binding: 3, resource: { buffer: this.cellCount } },
        { binding: 4, resource: { buffer: this.cellBoids } },
        { binding: 5, resource: { buffer: this.matBuf } },
        { binding: 6, resource: { buffer: this.colBuf } },
      ],
    });
  }

  setCount(n: number): void {
    setThinInstanceCount(this.mesh, n);
    this.ti._gpuVersion      = this.ti._version;
    this.ti._colorGpuVersion = this.ti._colorVersion;
    invalidateRenderBundles(this.engine);
  }

  /** Seamless CPU→GPU seed: copy [0,count) from CPU arrays, spread the rest. */
  seedFrom(
    count: number,
    px: Float32Array, py: Float32Array, pz: Float32Array,
    vx: Float32Array, vy: Float32Array, vz: Float32Array,
  ): void {
    const bound = gpuBoundForCount(count);
    const data = new Float32Array(GpuCapacity * 8);
    for (let i = 0; i < count; i++) {
      const b = i * 8;
      data[b] = px[i]; data[b + 1] = py[i]; data[b + 2] = pz[i]; data[b + 3] = 0;
      data[b + 4] = vx[i]; data[b + 5] = vy[i]; data[b + 6] = vz[i]; data[b + 7] = 0;
    }
    for (let i = count; i < GpuCapacity; i++) writeSpread(data, i, bound);
    this.device.queue.writeBuffer(this.front, 0, data);
  }

  /** Reseed [0,n) spread across the density-scaled world (used on count change). */
  spawnSpread(n: number): void {
    const bound = gpuBoundForCount(n);
    const data = new Float32Array(n * 8);
    for (let i = 0; i < n; i++) writeSpread(data, i, bound);
    this.device.queue.writeBuffer(this.front, 0, data, 0, n * 8);
  }

  /** One-shot async GPU→CPU readback (for GPU→CPU handoff). */
  async readbackInto(
    count: number,
    px: Float32Array, py: Float32Array, pz: Float32Array,
    vx: Float32Array, vy: Float32Array, vz: Float32Array,
  ): Promise<void> {
    const readSize = count * BytesPerBoid;
    const staging  = this.device.createBuffer({ size: readSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc      = this.device.createCommandEncoder();
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

  /** Dispatch one frame of compute. dt is in milliseconds. No readback. */
  dispatch(dt: number, params: GpuBoidsParams): void {
    if (!this._ok) return;
    const n = params.count;

    // Density-scaled grid: cells sized ~= neighbor radius, world grows with count.
    const bound       = gpuBoundForCount(n);
    const radius      = Math.max(params.radius, 0.5);
    const worldExtent = 2 * bound * 1.1;
    const gridDim     = Math.max(2, Math.min(GridDimMax, Math.floor(worldExtent / radius)));
    const cellSize    = worldExtent / gridDim;
    const cells       = gridDim * gridDim * gridDim;

    const u    = this.uboData;
    const uU32 = new Uint32Array(u.buffer);
    uU32[0] = n;
    u[1]  = dt * 0.001;
    u[2]  = params.speed;
    u[3]  = params.separation;
    u[4]  = params.alignment;
    u[5]  = params.cohesion;
    u[6]  = params.radius;
    u[7]  = params.separationRadius;
    u[8]  = bound;
    uU32[9] = gridDim;
    u[10] = cellSize;
    u[11] = 1 / cellSize;
    uU32[12] = cells;
    u[13]    = params.collisionRadius;
    u[14]    = params.collisionStrength;
    uU32[15] = params.collision ? 1 : 0;
    this.device.queue.writeBuffer(this.ubo, 0, u);

    const bg      = this.front === this.stateA ? this.bgA : this.bgB;
    const wgN64   = Math.ceil(n / 64);
    const wgCells = Math.ceil(cells / 64);

    const enc  = this.device.createCommandEncoder({ label: "boids-compute" });
    const pass = enc.beginComputePass({ label: "boids" });
    pass.setPipeline(this.clearPipeline);    pass.setBindGroup(0, bg); pass.dispatchWorkgroups(wgCells);
    pass.setPipeline(this.scatterPipeline);  pass.setBindGroup(0, bg); pass.dispatchWorkgroups(wgN64);
    pass.setPipeline(this.simulatePipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(wgN64);
    pass.end();
    this.device.queue.submit([enc.finish()]);

    const tmp  = this.front;
    this.front = this.back;
    this.back  = tmp;
  }

  destroy(): void {
    this.stateA.destroy();
    this.stateB.destroy();
    this.matBuf.destroy();
    this.colBuf.destroy();
    this.cellCount.destroy();
    this.cellBoids.destroy();
    this.ubo.destroy();
  }
}

/** Write a single spread boid (random pos within the scaled world + random heading). */
function writeSpread(data: Float32Array, i: number, bound: number): void {
  const b = i * 8;
  const a = Math.random() * Math.PI * 2;
  data[b]     = (Math.random() - 0.5) * bound * 1.8;
  data[b + 1] = (Math.random() - 0.5) * bound * 0.9;
  data[b + 2] = (Math.random() - 0.5) * bound * 1.8;
  data[b + 4] = Math.cos(a); data[b + 5] = 0.1; data[b + 6] = Math.sin(a);
}
