/**
 * GPU Boids — compute-only approach with async CPU readback.
 *
 * The spatial-grid compute pipeline (clear→count→scan→scatter→simulate) runs
 * fully on the GPU and scales to tens of thousands of boids. After each
 * dispatch the front buffer is copied to a staging buffer and `mapAsync` is
 * started non-blocking. When the mapping resolves (typically next frame),
 * `consumeReadback()` returns the count and the px/py/pz/vx/vy/vz arrays are
 * populated, ready for the caller to write into an AgentBuffer for rendering
 * with the standard PBR + thin-instance path.
 *
 * This mirrors how bjs-webgpu-particles/Emitter.ts handles sub-emitters:
 * GPU computes positions, async readback drives the rendering side.
 *
 * Compute passes per frame (single command encoder, single submit):
 *   0. clear   — zero cellCount atomics (required every frame — scatter leaves
 *                cellCount at the per-cell count, which doubles on the next
 *                COUNT pass if not cleared first)
 *   1. count   — each boid increments its cell counter
 *   2. scan    — single-thread sequential prefix sum → cellStart[], sentinel
 *   3. scatter — place boid indices into sorted slots
 *   4. simulate— integrate forces, write back pos/vel to back buffer
 */

import type { EngineContext } from "@babylonjs/lite";

// ── Constants ──────────────────────────────────────────────────────────────

export const GpuCapacity = 200_000;

const Bound = 20;
const GridDim = 8;
const TotalCells = GridDim * GridDim * GridDim; // 512

// Boid state: 2 × vec4<f32> per boid = 32 bytes
//   posT: vec4(x, y, z, colorT)
//   vel:  vec4(vx, vy, vz, 0)
const BytesPerBoid = 32;

// UBO: 16 × f32 = 64 bytes
const UBOWords = 16;

// ── WGSL common ─────────────────────────────────────────────────────────────

const WGSL_COMMON = /* wgsl */ `
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

// ── Pass 0: clear cell counts (required every frame) ─────────────────────────

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

// ── Pass 1: count boids per cell ─────────────────────────────────────────────

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

// ── Pass 2: sequential prefix scan → cellStart + sentinel ───────────────────

const WGSL_SCAN = /* wgsl */ `
${WGSL_COMMON}
@group(0) @binding(0) var<uniform>             params    : Params;
@group(0) @binding(3) var<storage, read_write> cellCount : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cellStart : array<u32>;

@compute @workgroup_size(1)
fn main() {
  let total = params.gridDim * params.gridDim * params.gridDim;
  var sum = 0u;
  for (var i = 0u; i <= total; i++) {
    cellStart[i] = sum;
    if (i < total) {
      let cnt = atomicLoad(&cellCount[i]);
      sum += cnt;
      atomicStore(&cellCount[i], 0u);  // reset for scatter
    }
  }
}
`;

// ── Pass 3: scatter boid indices into sorted slots ───────────────────────────

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

// ── Pass 4: simulate — neighbor scan + integrate ──────────────────────────────

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

  let dim  = params.gridDim;
  let pos  = front[i].posT.xyz;
  let vel  = front[i].vel.xyz;
  let half = params.bound * 1.1;
  let gx   = i32(clamp((pos.x + half) * params.invCellSize, 0.0, f32(dim - 1u)));
  let gy   = i32(clamp((pos.y + half) * params.invCellSize, 0.0, f32(dim - 1u)));
  let gz   = i32(clamp((pos.z + half) * params.invCellSize, 0.0, f32(dim - 1u)));

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
        let nx = gx + dx;  let ny = gy + dy;  let nz = gz + dz;
        if (nx < 0 || ny < 0 || nz < 0) { continue; }
        let unx = u32(nx);  let uny = u32(ny);  let unz = u32(nz);
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

  let newPos = pos + newVel * params.dt;
  let colorT = clamp(vLen / (params.speed * 1.5), 0.0, 1.0);

  back[i].posT = vec4<f32>(newPos, colorT);
  back[i].vel  = vec4<f32>(newVel, 0.0);
}
`;

// ── GpuBoids ─────────────────────────────────────────────────────────────────

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
  private readonly device: GPUDevice;

  // Ping-pong boid state
  private stateA: GPUBuffer;
  private stateB: GPUBuffer;
  private front: GPUBuffer;
  private back: GPUBuffer;

  // Grid buffers
  private readonly cellCount: GPUBuffer;
  private readonly cellStart: GPUBuffer;   // TotalCells + 1 (sentinel)
  private readonly sortedIndices: GPUBuffer;

  // UBO
  private readonly ubo: GPUBuffer;
  private readonly uboData: Float32Array;

  // Compute
  private readonly bgl: GPUBindGroupLayout;
  private readonly clearPipeline: GPUComputePipeline;
  private readonly countPipeline: GPUComputePipeline;
  private readonly scanPipeline: GPUComputePipeline;
  private readonly scatterPipeline: GPUComputePipeline;
  private readonly simulatePipeline: GPUComputePipeline;
  private readonly bgA: GPUBindGroup;
  private readonly bgB: GPUBindGroup;

  // Async readback
  private readonly stagingBuf: GPUBuffer;
  private readbackPending: boolean = false;
  private readbackResolvedCount: number = 0;

  // Output arrays — populated by readback, consumed by scene.ts each frame
  readonly px: Float32Array = new Float32Array(GpuCapacity);
  readonly py: Float32Array = new Float32Array(GpuCapacity);
  readonly pz: Float32Array = new Float32Array(GpuCapacity);
  readonly vx: Float32Array = new Float32Array(GpuCapacity);
  readonly vy: Float32Array = new Float32Array(GpuCapacity);
  readonly vz: Float32Array = new Float32Array(GpuCapacity);
  readonly colorT: Float32Array = new Float32Array(GpuCapacity);

  private _ok = false;
  get ok(): boolean { return this._ok; }

  constructor(engine: EngineContext) {
    const dev = (engine as unknown as { _device: GPUDevice })._device;
    this.device = dev;

    // ── Boid state buffers ────────────────────────────────────────────────
    const stateUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.stateA = dev.createBuffer({ label: "boids-stateA", size: GpuCapacity * BytesPerBoid, usage: stateUsage });
    this.stateB = dev.createBuffer({ label: "boids-stateB", size: GpuCapacity * BytesPerBoid, usage: stateUsage });
    this.front = this.stateA;
    this.back  = this.stateB;

    // Staging buffer for async GPU→CPU readback (MAP_READ + COPY_DST)
    this.stagingBuf = dev.createBuffer({
      label: "boids-staging",
      size: GpuCapacity * BytesPerBoid,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // ── Spatial grid buffers ──────────────────────────────────────────────
    const gridUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.cellCount     = dev.createBuffer({ label: "cellCount",  size: TotalCells * 4,       usage: gridUsage });
    this.cellStart     = dev.createBuffer({ label: "cellStart",  size: (TotalCells + 1) * 4, usage: gridUsage });
    this.sortedIndices = dev.createBuffer({ label: "sortedIdx",  size: GpuCapacity * 4,      usage: GPUBufferUsage.STORAGE });

    this.uboData = new Float32Array(UBOWords);
    this.ubo     = dev.createBuffer({ label: "boidsUBO", size: UBOWords * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // ── Compute pipelines ─────────────────────────────────────────────────
    try {
      this.bgl              = this.buildBGL();
      this.clearPipeline    = this.buildComputePipeline("clear",    WGSL_CLEAR);
      this.countPipeline    = this.buildComputePipeline("count",    WGSL_COUNT);
      this.scanPipeline     = this.buildComputePipeline("scan",     WGSL_SCAN);
      this.scatterPipeline  = this.buildComputePipeline("scatter",  WGSL_SCATTER);
      this.simulatePipeline = this.buildComputePipeline("simulate", WGSL_SIMULATE);
      this.bgA = this.buildBG("A", this.stateA, this.stateB);
      this.bgB = this.buildBG("B", this.stateB, this.stateA);
    } catch (e) {
      console.error("[GpuBoids] pipeline build failed:", e);
      this.bgl = null!; this.clearPipeline = null!; this.countPipeline = null!;
      this.scanPipeline = null!; this.scatterPipeline = null!; this.simulatePipeline = null!;
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
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
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
        { binding: 4, resource: { buffer: this.cellStart } },
        { binding: 5, resource: { buffer: this.sortedIndices } },
      ],
    });
  }

  /** Seed boid state from CPU arrays (instant writeBuffer). */
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
    for (let i = count; i < GpuCapacity; i++) {
      const b = i * 8;
      const a = Math.random() * Math.PI * 2;
      data[b]     = (Math.random() - 0.5) * 40;
      data[b + 1] = (Math.random() - 0.5) * 20;
      data[b + 2] = (Math.random() - 0.5) * 40;
      data[b + 4] = Math.cos(a); data[b + 5] = 0.1; data[b + 6] = Math.sin(a);
    }
    this.device.queue.writeBuffer(this.front, 0, data);
    // Also pre-populate output arrays so the first render frame shows something
    for (let i = 0; i < count; i++) {
      this.px[i] = px[i]; this.py[i] = py[i]; this.pz[i] = pz[i];
      this.vx[i] = vx[i]; this.vy[i] = vy[i]; this.vz[i] = vz[i];
      this.colorT[i] = 0;
    }
    this.readbackResolvedCount = count;
  }

  /**
   * Dispatch one frame of compute and start a non-blocking readback.
   * dt is in milliseconds.
   */
  dispatch(dt: number, params: GpuBoidsParams): void {
    if (!this._ok) return;
    const n = params.count;

    // Update UBO
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
    u[8]  = Bound;
    uU32[9] = GridDim;
    const cs = (Bound * 2 * 1.1) / GridDim;
    u[10] = cs;
    u[11] = 1 / cs;
    this.device.queue.writeBuffer(this.ubo, 0, u);

    const bg      = this.front === this.stateA ? this.bgA : this.bgB;
    const wgN64   = Math.ceil(n / 64);
    const wgCells = Math.ceil(TotalCells / 64); // ceil(512/64) = 8

    // Compute passes
    const enc  = this.device.createCommandEncoder({ label: "boids-compute" });
    const pass = enc.beginComputePass({ label: "boids" });
    pass.setPipeline(this.clearPipeline);    pass.setBindGroup(0, bg); pass.dispatchWorkgroups(wgCells);
    pass.setPipeline(this.countPipeline);    pass.setBindGroup(0, bg); pass.dispatchWorkgroups(wgN64);
    pass.setPipeline(this.scanPipeline);     pass.setBindGroup(0, bg); pass.dispatchWorkgroups(1);
    pass.setPipeline(this.scatterPipeline);  pass.setBindGroup(0, bg); pass.dispatchWorkgroups(wgN64);
    pass.setPipeline(this.simulatePipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(wgN64);
    pass.end();
    this.device.queue.submit([enc.finish()]);

    // Ping-pong: front now points at the freshly-computed buffer.
    const tmp  = this.front;
    this.front = this.back;
    this.back  = tmp;

    // Read back ONLY when no map is in flight. Copying into a buffer that is
    // pending-map (or mapped) is a WebGPU validation error that would drop the
    // submit and freeze the sim — so the copy and the mapAsync are done
    // together, guarded by readbackPending.
    if (!this.readbackPending) {
      this.readbackPending = true;
      const count    = n;
      const copySize = n * BytesPerBoid;

      const enc2 = this.device.createCommandEncoder({ label: "boids-readback" });
      enc2.copyBufferToBuffer(this.front, 0, this.stagingBuf, 0, copySize);
      this.device.queue.submit([enc2.finish()]);

      this.stagingBuf.mapAsync(GPUMapMode.READ, 0, copySize).then(() => {
        const src = new Float32Array(this.stagingBuf.getMappedRange(0, copySize));
        for (let i = 0; i < count; i++) {
          const b = i * 8;
          this.px[i]     = src[b];
          this.py[i]     = src[b + 1];
          this.pz[i]     = src[b + 2];
          this.colorT[i] = src[b + 3];
          this.vx[i]     = src[b + 4];
          this.vy[i]     = src[b + 5];
          this.vz[i]     = src[b + 6];
        }
        this.stagingBuf.unmap();
        this.readbackResolvedCount = count;
        this.readbackPending = false;
      }).catch((e) => {
        console.error("[GpuBoids] readback failed:", e);
        this.readbackPending = false;
      });
    }
  }

  /**
   * Returns the most recently resolved readback count (0 if none yet).
   * The caller should then read px/py/pz/vx/vy/vz/colorT arrays.
   */
  get readbackCount(): number { return this.readbackResolvedCount; }

  /**
   * For seamless GPU→CPU handoff: async readback the latest front buffer
   * into the caller-supplied arrays.
   */
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

  destroy(): void {
    this.stateA.destroy();
    this.stateB.destroy();
    this.stagingBuf.destroy();
    this.cellCount.destroy();
    this.cellStart.destroy();
    this.sortedIndices.destroy();
    this.ubo.destroy();
  }
}
