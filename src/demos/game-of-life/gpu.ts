/**
 * GPU Game of Life — pure-compute step + GPU-driven rendering (no readback).
 *
 * Two compute passes per dispatch:
 *   step   — one thread per cell; reads front (u32 per cell), applies toroidal
 *             8-neighbor B/S rules, writes back. Skipped when the sim is paused.
 *   render — one thread per cell; reads front, writes a column-major TRS mat4
 *             into matBuf and RGBA into colBuf — the same transform/color mapping
 *             as the CPU writeInstances (alive: y=0 scale=0.88, dead: y=-0.38
 *             flat-squished scale=(0.82,0.08,0.82)).
 *
 * Buffer-ownership pattern mirrors boids/gpu.ts:
 *   matBuf / colBuf are STORAGE|VERTEX|COPY_DST buffers assigned directly to
 *   ti._gpuBuffer / ti._colorGpuBuffer with versions synced so Lite never
 *   recreates or overwrites them.
 */

import {
  addToScene,
  createBox,
  createPbrMaterial,
  invalidateRenderBundles,
  setThinInstanceColors,
  setThinInstanceCount,
  setThinInstances,
} from "@babylonjs/lite";
import type { EngineContext, Mesh, SceneContext } from "@babylonjs/lite";

// ── Constants ──────────────────────────────────────────────────────────────

export const GpuMaxDim   = 512;
export const GpuMaxCells = GpuMaxDim * GpuMaxDim;   // 262 144

const CellSize = 0.9;   // must match scene.ts CPU constant

// UBO: 8 u32/f32 values, padded to 16 words (64 bytes).
const UBOWords = 16;

// ── Thin-instance internal shape ───────────────────────────────────────────

type ThinInstData = {
  _gpuBuffer: GPUBuffer | null;       _gpuBufferStorage: boolean;
  _gpuVersion: number;                _version: number;
  _colorGpuBuffer: GPUBuffer | null;  _colorGpuBufferStorage: boolean;
  _colorGpuVersion: number;           _colorVersion: number;
};

// ── WGSL UBO struct (shared by both shaders) ───────────────────────────────

const WGSL_PARAMS = /* wgsl */ `
struct Params {
  dim      : u32,   // [0] active grid dimension
  survive2 : u32,   // [1]
  survive3 : u32,   // [2]
  birth3   : u32,   // [3]
  cellSize : f32,   // [4]
  _p0:u32, _p1:u32, _p2:u32, _p3:u32,
  _p4:u32, _p5:u32, _p6:u32, _p7:u32,
  _p8:u32, _p9:u32, _p10:u32,
}
@group(0) @binding(0) var<uniform> params : Params;
`;

// ── Pass 1: step (apply B/S rules into back buffer) ────────────────────────

const WGSL_STEP = /* wgsl */ `
${WGSL_PARAMS}
@group(0) @binding(1) var<storage, read>       front : array<u32>;
@group(0) @binding(2) var<storage, read_write> back  : array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let dim = params.dim;
  if (idx >= dim * dim) { return; }

  let x = idx % dim;
  let y = idx / dim;

  // Toroidal 8-neighbour count.
  var n : u32 = 0u;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let nx = (x + u32(dim) + u32(dx)) % dim;
      let ny = (y + u32(dim) + u32(dy)) % dim;
      n += front[ny * dim + nx];
    }
  }

  let alive = front[idx];
  if (alive == 1u) {
    back[idx] = select(0u, 1u,
      (n == 2u && params.survive2 == 1u) ||
      (n == 3u && params.survive3 == 1u));
  } else {
    back[idx] = select(0u, 1u, n == 3u && params.birth3 == 1u);
  }
}
`;

// ── Pass 2: render (write TRS + RGBA into owned instance buffers) ──────────

const WGSL_RENDER = /* wgsl */ `
${WGSL_PARAMS}
@group(0) @binding(1) var<storage, read>       front  : array<u32>;
@group(0) @binding(2) var<storage, read_write> matBuf : array<f32>;
@group(0) @binding(3) var<storage, read_write> colBuf : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let dim = params.dim;
  if (idx >= dim * dim) { return; }

  let cs   = params.cellSize;
  let half = f32(dim) * 0.5;
  let x    = f32(idx % dim);
  let z    = f32(idx / dim);
  let wx   = (x - half) * cs;
  let wz   = (z - half) * cs;

  let alive = front[idx] == 1u;
  let mb = idx * 16u;

  if (alive) {
    let s = cs * 0.88;
    // Column-major mat4: uniform scale, translation y=0.
    matBuf[mb+ 0u]=s;   matBuf[mb+ 1u]=0.0; matBuf[mb+ 2u]=0.0; matBuf[mb+ 3u]=0.0;
    matBuf[mb+ 4u]=0.0; matBuf[mb+ 5u]=s;   matBuf[mb+ 6u]=0.0; matBuf[mb+ 7u]=0.0;
    matBuf[mb+ 8u]=0.0; matBuf[mb+ 9u]=0.0; matBuf[mb+10u]=s;   matBuf[mb+11u]=0.0;
    matBuf[mb+12u]=wx;  matBuf[mb+13u]=0.0; matBuf[mb+14u]=wz;  matBuf[mb+15u]=1.0;
    let cb = idx * 4u;
    colBuf[cb+0u]=0.05; colBuf[cb+1u]=0.85; colBuf[cb+2u]=1.0;  colBuf[cb+3u]=1.0;
  } else {
    let sx = cs * 0.82;
    let sy = cs * 0.08;
    // Non-uniform scale (flat slab), translation y=-0.38.
    matBuf[mb+ 0u]=sx;  matBuf[mb+ 1u]=0.0; matBuf[mb+ 2u]=0.0; matBuf[mb+ 3u]=0.0;
    matBuf[mb+ 4u]=0.0; matBuf[mb+ 5u]=sy;  matBuf[mb+ 6u]=0.0; matBuf[mb+ 7u]=0.0;
    matBuf[mb+ 8u]=0.0; matBuf[mb+ 9u]=0.0; matBuf[mb+10u]=sx;  matBuf[mb+11u]=0.0;
    matBuf[mb+12u]=wx;  matBuf[mb+13u]=-0.38; matBuf[mb+14u]=wz; matBuf[mb+15u]=1.0;
    let cb = idx * 4u;
    colBuf[cb+0u]=0.08; colBuf[cb+1u]=0.12; colBuf[cb+2u]=0.18; colBuf[cb+3u]=1.0;
  }
}
`;

// ── GolGpuParams ───────────────────────────────────────────────────────────

export interface GolGpuParams {
  dim:      number;
  survive2: boolean;
  survive3: boolean;
  birth3:   boolean;
}

// ── GolGpu ─────────────────────────────────────────────────────────────────

export class GolGpu {
  readonly mesh: Mesh;

  private readonly engine: EngineContext;
  private readonly device: GPUDevice;
  private readonly ti: ThinInstData;

  private readonly matBuf: GPUBuffer;
  private readonly colBuf: GPUBuffer;

  private readonly stateA: GPUBuffer;
  private readonly stateB: GPUBuffer;
  private front: GPUBuffer;
  private back:  GPUBuffer;

  private readonly ubo:     GPUBuffer;
  private readonly uboData: Uint32Array;

  private readonly stepBGL:   GPUBindGroupLayout;
  private readonly renderBGL: GPUBindGroupLayout;
  private readonly stepPipeline:   GPUComputePipeline;
  private readonly renderPipeline: GPUComputePipeline;

  // Bind groups are rebuilt whenever dim changes (front/back ptrs are stable,
  // but the render pass needs matBuf/colBuf in separate slots).
  private bgStepA:   GPUBindGroup;
  private bgStepB:   GPUBindGroup;
  private bgRenderA: GPUBindGroup;
  private bgRenderB: GPUBindGroup;

  private _ok = false;
  get ok(): boolean { return this._ok; }

  private _dim = 0;

  constructor(engine: EngineContext, scene: SceneContext) {
    this.engine = engine;
    const dev = (engine as unknown as { _device: GPUDevice })._device;
    this.device = dev;

    // ── Owned instance vertex buffers ────────────────────────────────────
    const instUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST;
    this.matBuf = dev.createBuffer({ label: "gol-mat", size: GpuMaxCells * 64, usage: instUsage });
    this.colBuf = dev.createBuffer({ label: "gol-col", size: GpuMaxCells * 16, usage: instUsage });

    // ── PBR thin-instanced box mesh ───────────────────────────────────────
    const box = createBox(engine, 1);
    box.material = createPbrMaterial({ baseColorFactor: [0.05, 0.8, 1, 1], metallicFactor: 0.1, roughnessFactor: 0.4 });
    addToScene(scene, box);
    this.mesh = box;

    const identMats = new Float32Array(GpuMaxCells * 16);
    for (let i = 0; i < GpuMaxCells; i++) {
      const b = i * 16;
      identMats[b] = 1; identMats[b+5] = 1; identMats[b+10] = 1; identMats[b+15] = 1;
    }
    setThinInstances(box, identMats, GpuMaxCells);
    setThinInstanceColors(box, new Float32Array(GpuMaxCells * 4).fill(1));
    setThinInstanceCount(box, 0);

    const ti = (box as unknown as { thinInstances: ThinInstData }).thinInstances;
    ti._gpuBuffer             = this.matBuf;
    ti._gpuBufferStorage      = false;
    ti._gpuVersion            = ti._version;
    ti._colorGpuBuffer        = this.colBuf;
    ti._colorGpuBufferStorage = false;
    ti._colorGpuVersion       = ti._colorVersion;
    this.ti = ti;

    // ── Ping-pong cell state buffers (one u32 per cell) ───────────────────
    const stUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.stateA = dev.createBuffer({ label: "gol-stateA", size: GpuMaxCells * 4, usage: stUsage });
    this.stateB = dev.createBuffer({ label: "gol-stateB", size: GpuMaxCells * 4, usage: stUsage });
    this.front = this.stateA;
    this.back  = this.stateB;

    // ── UBO (interpreted as Uint32Array for flag fields) ──────────────────
    this.uboData = new Uint32Array(UBOWords);
    this.ubo = dev.createBuffer({ label: "gol-ubo", size: UBOWords * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // ── Build pipelines ───────────────────────────────────────────────────
    try {
      this.stepBGL   = this.buildStepBGL();
      this.renderBGL = this.buildRenderBGL();
      this.stepPipeline   = this.buildPipeline("gol-step",   WGSL_STEP,   this.stepBGL);
      this.renderPipeline = this.buildPipeline("gol-render", WGSL_RENDER, this.renderBGL);
      this.bgStepA   = this.buildStepBG("A",   this.stateA, this.stateB);
      this.bgStepB   = this.buildStepBG("B",   this.stateB, this.stateA);
      this.bgRenderA = this.buildRenderBG("A", this.stateA);
      this.bgRenderB = this.buildRenderBG("B", this.stateB);
    } catch (e) {
      console.error("[GolGpu] pipeline build failed:", e);
      this.stepBGL = null!; this.renderBGL = null!;
      this.stepPipeline = null!; this.renderPipeline = null!;
      this.bgStepA = null!; this.bgStepB = null!;
      this.bgRenderA = null!; this.bgRenderB = null!;
      return;
    }

    this._ok = true;
  }

  private buildStepBGL(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "gol-step-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
  }

  private buildRenderBGL(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "gol-render-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
  }

  private buildPipeline(label: string, wgsl: string, bgl: GPUBindGroupLayout): GPUComputePipeline {
    const mod = this.device.createShaderModule({ label, code: wgsl });
    return this.device.createComputePipeline({
      label,
      layout: this.device.createPipelineLayout({ label: `${label}-layout`, bindGroupLayouts: [bgl] }),
      compute: { module: mod, entryPoint: "main" },
    });
  }

  private buildStepBG(label: string, front: GPUBuffer, back: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({
      label: `gol-step-bg-${label}`, layout: this.stepBGL,
      entries: [
        { binding: 0, resource: { buffer: this.ubo } },
        { binding: 1, resource: { buffer: front } },
        { binding: 2, resource: { buffer: back } },
      ],
    });
  }

  private buildRenderBG(label: string, front: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({
      label: `gol-render-bg-${label}`, layout: this.renderBGL,
      entries: [
        { binding: 0, resource: { buffer: this.ubo } },
        { binding: 1, resource: { buffer: front } },
        { binding: 2, resource: { buffer: this.matBuf } },
        { binding: 3, resource: { buffer: this.colBuf } },
      ],
    });
  }

  /** Update the active cell count and resync version guard. */
  setDim(dim: number): void {
    this._dim = dim;
    setThinInstanceCount(this.mesh, dim * dim);
    this.ti._gpuVersion      = this.ti._version;
    this.ti._colorGpuVersion = this.ti._colorVersion;
    invalidateRenderBundles(this.engine);
  }

  /** Fill `front` with a random initial state at the given density. */
  seedRandom(dim: number, density: number): void {
    const cells = dim * dim;
    const data  = new Uint32Array(cells);
    for (let i = 0; i < cells; i++) data[i] = Math.random() < density ? 1 : 0;
    this.device.queue.writeBuffer(this.stateA, 0, data.buffer, 0, cells * 4);
    this.front = this.stateA;
    this.back  = this.stateB;
  }

  /**
   * Stamp a rectangular brush of alive (1) cells directly into `front`.
   * `cells` is an array of (x,y) pairs within [0, dim).
   */
  paintCells(dim: number, cells: Array<[number, number]>): void {
    const one = new Uint32Array([1]);
    for (const [cx, cy] of cells) {
      if (cx < 0 || cx >= dim || cy < 0 || cy >= dim) continue;
      const offset = (cy * dim + cx) * 4;
      this.device.queue.writeBuffer(this.front, offset, one.buffer);
    }
  }

  /**
   * Dispatch one frame.
   * @param doStep  - Run the rule step (false when the sim is paused).
   * @param p       - Current rule flags and active dim.
   */
  dispatch(doStep: boolean, p: GolGpuParams): void {
    if (!this._ok) return;
    const dim   = p.dim;
    const cells = dim * dim;

    // Write UBO.
    const u = this.uboData;
    u[0] = dim;
    u[1] = p.survive2 ? 1 : 0;
    u[2] = p.survive3 ? 1 : 0;
    u[3] = p.birth3   ? 1 : 0;
    // u[4] is cellSize as f32 — write via DataView.
    new DataView(u.buffer).setFloat32(4 * 4, CellSize, true);
    this.device.queue.writeBuffer(this.ubo, 0, u.buffer);

    const wg     = Math.ceil(cells / 64);
    const frontIsA = this.front === this.stateA;
    const stepBG   = frontIsA ? this.bgStepA   : this.bgStepB;
    // When stepping: step writes into back, so render must read back (post-step).
    // When not stepping: render reads front (current live state).
    const renderBG = doStep
      ? (frontIsA ? this.bgRenderB : this.bgRenderA)
      : (frontIsA ? this.bgRenderA : this.bgRenderB);

    const enc  = this.device.createCommandEncoder({ label: "gol-compute" });
    const pass = enc.beginComputePass({ label: "gol" });

    if (doStep) {
      pass.setPipeline(this.stepPipeline);
      pass.setBindGroup(0, stepBG);
      pass.dispatchWorkgroups(wg);
    }

    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, renderBG);
    pass.dispatchWorkgroups(wg);

    pass.end();
    this.device.queue.submit([enc.finish()]);

    // Ping-pong only when we actually stepped.
    if (doStep) {
      const tmp  = this.front;
      this.front = this.back;
      this.back  = tmp;
    }
  }

  destroy(): void {
    this.matBuf.destroy();
    this.colBuf.destroy();
    this.stateA.destroy();
    this.stateB.destroy();
    this.ubo.destroy();
  }
}
