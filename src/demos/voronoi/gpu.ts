/**
 * GPU Voronoi Landscape — compute-driven heightfield (no readback).
 *
 * One 2D compute pass per frame (@workgroup_size(16,16)):
 *   For each grid cell (gx, gy): loop over all seeds tracking nearest d1/id
 *   and 2nd-nearest d2, then compute height by mode and write a column-major
 *   TRS mat4 + RGBA color into the owned matBuf / colBuf that feed directly
 *   into Lite's thin-instance vertex slots.
 *
 * Height modes (matches CPU scene.ts):
 *   0 = ridge  (d2-d1)/cellNorm  — shattered crystal, valleys at edges
 *   1 = cone   1-d1/cellNorm      — peak at each seed
 *   2 = mesa   seed[id].w         — region area (normalized, CPU-side)
 *   3 = noise  animated value noise per region
 *
 * Seed positions, hues, and mesa heights are uploaded from the CPU each frame
 * via a small seedBuf (GpuMaxSeeds x vec4<f32>), so seed animation / Lloyd /
 * mesa region-areas stay in JS where they are already cheap.
 *
 * Buffer sizes at max:
 *   matBuf 262144 * 64 = 16.7 MB
 *   colBuf 262144 * 16 =  4.2 MB
 *   seedBuf 256 * 16   =  4 KB
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

export const GpuMaxGrid    = 512;
export const GpuMaxColumns = GpuMaxGrid * GpuMaxGrid;
export const GpuMaxSeeds   = 256;

// UBO: 12 words (48 bytes padded to 16-word boundary = 64 bytes).
const UBOWords = 16;

// ── Thin-instance internal shape ───────────────────────────────────────────

type ThinInstData = {
  _gpuBuffer: GPUBuffer | null;       _gpuBufferStorage: boolean;
  _gpuVersion: number;                _version: number;
  _colorGpuBuffer: GPUBuffer | null;  _colorGpuBufferStorage: boolean;
  _colorGpuVersion: number;           _colorVersion: number;
};

// ── WGSL ─────────────────────────────────────────────────────────────────

const WGSL_BUILD = /* wgsl */ `
struct Params {
  gridRes     : u32,
  seedCount   : u32,
  heightMode  : u32,   // 0=ridge 1=cone 2=mesa 3=noise
  _pad0       : u32,
  heightScale : f32,
  cellNorm    : f32,
  worldSize   : f32,
  time        : f32,
  _p0:u32,_p1:u32,_p2:u32,_p3:u32,_p4:u32,_p5:u32,_p6:u32,_p7:u32,
}

struct Seed {
  x    : f32,
  y    : f32,
  hue  : f32,
  mesaH: f32,   // normalized region area (CPU-computed)
}

@group(0) @binding(0) var<uniform>             params   : Params;
@group(0) @binding(1) var<storage, read>       seeds    : array<Seed>;
@group(0) @binding(2) var<storage, read_write> matBuf   : array<f32>;
@group(0) @binding(3) var<storage, read_write> colBuf   : array<f32>;

// ── Helpers ────────────────────────────────────────────────────────────────

fn hue2rgb(p: f32, q: f32, t_in: f32) -> f32 {
  var t = t_in;
  if (t < 0.0) { t += 1.0; }
  if (t > 1.0) { t -= 1.0; }
  if (t < 0.1667) { return p + (q - p) * 6.0 * t; }
  if (t < 0.5)    { return q; }
  if (t < 0.6667) { return p + (q - p) * (0.6667 - t) * 6.0; }
  return p;
}

fn hslToRgb(h: f32, s: f32, l: f32) -> vec3<f32> {
  let q = select(l + s - l * s, l * (1.0 + s), l < 0.5);
  let p = 2.0 * l - q;
  return vec3<f32>(
    hue2rgb(p, q, h + 0.3333),
    hue2rgb(p, q, h),
    hue2rgb(p, q, h - 0.3333),
  );
}

// Simple hash value noise: bilinear over a hash lattice.
fn pcg(v: u32) -> f32 {
  var s = v * 747796405u + 2891336453u;
  s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  s = (s >> 22u) ^ s;
  return f32(s) / 4294967296.0;
}
fn hashF(ix: i32, iy: i32) -> f32 {
  let h = (u32(abs(ix)) * 1619u + u32(abs(iy)) * 31337u + 6791u);
  return pcg(h);
}
fn valueNoise(x: f32, y: f32) -> f32 {
  let ix = i32(floor(x)); let iy = i32(floor(y));
  let fx = x - f32(ix);   let fy = y - f32(iy);
  let ux = fx*fx*(3.0-2.0*fx); let uy = fy*fy*(3.0-2.0*fy);
  let a = hashF(ix,   iy  ); let b = hashF(ix+1, iy  );
  let c = hashF(ix,   iy+1); let d = hashF(ix+1, iy+1);
  return (a*(1.0-ux)+b*ux)*(1.0-uy) + (c*(1.0-ux)+d*ux)*uy;
}

// ── Main ───────────────────────────────────────────────────────────────────

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let gx = gid.x; let gy = gid.y;
  let G  = params.gridRes;
  if (gx >= G || gy >= G) { return; }

  let fx = (f32(gx) + 0.5) / f32(G);
  let fy = (f32(gy) + 0.5) / f32(G);
  let n  = params.seedCount;

  // Nearest + 2nd-nearest seed search.
  var d1  = 1e30; var d2 = 1e30;
  var sid = 0u;
  for (var i = 0u; i < n; i++) {
    let dx = fx - seeds[i].x;
    let dy = fy - seeds[i].y;
    let d2_ = dx*dx + dy*dy;
    if (d2_ < d1) { d2 = d1; d1 = d2_; sid = i; }
    else if (d2_ < d2) { d2 = d2_; }
  }
  let dist1 = sqrt(d1);
  let dist2 = sqrt(d2);

  let cn  = params.cellNorm;
  let hue = seeds[sid].hue;

  // Height [0,1] by mode.
  var hNorm: f32;
  if (params.heightMode == 0u) {
    hNorm = min((dist2 - dist1) / cn, 1.0);                // ridge
  } else if (params.heightMode == 1u) {
    hNorm = max(1.0 - dist1 / cn, 0.0);                    // cone
  } else if (params.heightMode == 2u) {
    hNorm = seeds[sid].mesaH;                              // mesa (CPU-normalised)
  } else {
    let rOff = hue * 3.7;
    hNorm = clamp(valueNoise(fx*4.0 + rOff, fy*4.0 + params.time*0.2), 0.0, 1.0); // noise
  }

  let hs  = params.heightScale;
  let h   = max(hNorm * hs, 0.05);
  let ws  = params.worldSize;
  let cw  = (ws / f32(G)) * 0.96;

  let wx = (fx - 0.5) * ws;
  let wz = (fy - 0.5) * ws;

  // Column mat4: non-uniform scale (cw, h, cw), translation (wx, h/2, wz).
  let mb = (gy * G + gx) * 16u;
  matBuf[mb+ 0u]=cw;  matBuf[mb+ 1u]=0.0; matBuf[mb+ 2u]=0.0; matBuf[mb+ 3u]=0.0;
  matBuf[mb+ 4u]=0.0; matBuf[mb+ 5u]=h;   matBuf[mb+ 6u]=0.0; matBuf[mb+ 7u]=0.0;
  matBuf[mb+ 8u]=0.0; matBuf[mb+ 9u]=0.0; matBuf[mb+10u]=cw;  matBuf[mb+11u]=0.0;
  matBuf[mb+12u]=wx;  matBuf[mb+13u]=h*0.5; matBuf[mb+14u]=wz; matBuf[mb+15u]=1.0;

  // Color: hslToRgb(hue, 0.72, 0.32 + 0.32*hNorm).
  let rgb = hslToRgb(hue, 0.72, 0.32 + 0.32 * hNorm);
  let cb  = (gy * G + gx) * 4u;
  colBuf[cb+0u]=rgb.r; colBuf[cb+1u]=rgb.g; colBuf[cb+2u]=rgb.b; colBuf[cb+3u]=1.0;
}
`;

// ── VoronoiGpuParams ───────────────────────────────────────────────────────

export interface VoronoiGpuParams {
  gridRes:     number;
  seedCount:   number;
  heightMode:  number;   // 0=ridge 1=cone 2=mesa 3=noise
  heightScale: number;
  cellNorm:    number;
  worldSize:   number;
  time:        number;
}

// ── VoronoiGpu ────────────────────────────────────────────────────────────

export class VoronoiGpu {
  readonly columnMesh: Mesh;

  private readonly engine: EngineContext;
  private readonly device: GPUDevice;
  private readonly ti:     ThinInstData;

  private readonly matBuf:  GPUBuffer;
  private readonly colBuf:  GPUBuffer;
  private readonly seedBuf: GPUBuffer;
  private readonly ubo:     GPUBuffer;
  private readonly uboData: Uint32Array;

  private readonly bgl:      GPUBindGroupLayout;
  private readonly bg:       GPUBindGroup;
  private readonly pipeline: GPUComputePipeline;

  private _lastGrid = 0;
  private _ok = false;
  get ok(): boolean { return this._ok; }

  constructor(engine: EngineContext, scene: SceneContext) {
    this.engine = engine;
    const dev = (engine as unknown as { _device: GPUDevice })._device;
    this.device = dev;

    // ── Owned instance buffers ────────────────────────────────────────────
    const instUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST;
    this.matBuf = dev.createBuffer({ label: "voro-mat", size: GpuMaxColumns * 64, usage: instUsage });
    this.colBuf = dev.createBuffer({ label: "voro-col", size: GpuMaxColumns * 16, usage: instUsage });

    // ── PBR column box mesh ───────────────────────────────────────────────
    const box = createBox(engine, 1);
    box.material = createPbrMaterial({ baseColorFactor: [0.8, 0.8, 0.8, 1], metallicFactor: 0.2, roughnessFactor: 0.7 });
    addToScene(scene, box);
    this.columnMesh = box;

    const identMats = new Float32Array(GpuMaxColumns * 16);
    for (let i = 0; i < GpuMaxColumns; i++) {
      const b = i * 16;
      identMats[b] = 1; identMats[b+5] = 1; identMats[b+10] = 1; identMats[b+15] = 1;
    }
    setThinInstances(box, identMats, GpuMaxColumns);
    setThinInstanceColors(box, new Float32Array(GpuMaxColumns * 4).fill(1));
    setThinInstanceCount(box, 0);

    const ti = (box as unknown as { thinInstances: ThinInstData }).thinInstances;
    ti._gpuBuffer             = this.matBuf;
    ti._gpuBufferStorage      = false;
    ti._gpuVersion            = ti._version;
    ti._colorGpuBuffer        = this.colBuf;
    ti._colorGpuBufferStorage = false;
    ti._colorGpuVersion       = ti._colorVersion;
    this.ti = ti;

    // ── Seed buffer (vec4 per seed: x, y, hue, mesaH) ────────────────────
    this.seedBuf = dev.createBuffer({
      label: "voro-seeds",
      size:  GpuMaxSeeds * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // ── UBO ───────────────────────────────────────────────────────────────
    this.uboData = new Uint32Array(UBOWords);
    this.ubo = dev.createBuffer({
      label: "voro-ubo",
      size:  UBOWords * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ── Pipeline ──────────────────────────────────────────────────────────
    let bgl: GPUBindGroupLayout;
    let bg:  GPUBindGroup;
    let pipeline: GPUComputePipeline;
    try {
      bgl = dev.createBindGroupLayout({
        label: "voro-bgl",
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ],
      });
      const mod = dev.createShaderModule({ label: "voro-build", code: WGSL_BUILD });
      pipeline = dev.createComputePipeline({
        label:   "voro-build",
        layout:  dev.createPipelineLayout({ label: "voro-build-layout", bindGroupLayouts: [bgl] }),
        compute: { module: mod, entryPoint: "main" },
      });
      bg = dev.createBindGroup({
        label: "voro-bg", layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: this.ubo } },
          { binding: 1, resource: { buffer: this.seedBuf } },
          { binding: 2, resource: { buffer: this.matBuf } },
          { binding: 3, resource: { buffer: this.colBuf } },
        ],
      });
    } catch (e) {
      console.error("[VoronoiGpu] pipeline build failed:", e);
      this.bgl = null!; this.bg = null!; this.pipeline = null!;
      return;
    }
    this.bgl = bgl;
    this.bg  = bg;
    this.pipeline = pipeline;
    this._ok = true;
  }

  /** Update active column count (must be called when gridRes changes). */
  setGrid(G: number): void {
    const n = G * G;
    setThinInstanceCount(this.columnMesh, n);
    this.ti._gpuVersion      = this.ti._version;
    this.ti._colorGpuVersion = this.ti._colorVersion;
    if (this._lastGrid !== G) {
      this._lastGrid = G;
      invalidateRenderBundles(this.engine);
    }
  }

  /**
   * Upload seed positions, hues, and CPU-computed mesa heights.
   * `mesaH[i]` should be pre-normalized to [0,1] (regionArea / maxArea).
   * Pass 0 for all mesaH when not in mesa mode.
   */
  uploadSeeds(
    sx: Float32Array, sy: Float32Array, hues: Float32Array,
    mesaH: Float32Array, n: number
  ): void {
    const data = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      data[i*4  ] = sx[i];
      data[i*4+1] = sy[i];
      data[i*4+2] = hues[i];
      data[i*4+3] = mesaH[i];
    }
    this.device.queue.writeBuffer(this.seedBuf, 0, data.buffer, 0, n * 16);
  }

  /** Dispatch one frame of compute. */
  dispatch(p: VoronoiGpuParams): void {
    if (!this._ok) return;
    const G  = p.gridRes;
    const u  = this.uboData;
    const fv = new DataView(u.buffer);
    u[0] = G;
    u[1] = p.seedCount;
    u[2] = p.heightMode;
    fv.setFloat32(4 * 4, p.heightScale, true);
    fv.setFloat32(5 * 4, p.cellNorm,    true);
    fv.setFloat32(6 * 4, p.worldSize,   true);
    fv.setFloat32(7 * 4, p.time,        true);
    this.device.queue.writeBuffer(this.ubo, 0, u.buffer);

    const wg  = Math.ceil(G / 16);
    const enc = this.device.createCommandEncoder({ label: "voro-compute" });
    const pass = enc.beginComputePass({ label: "voro" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bg);
    pass.dispatchWorkgroups(wg, wg);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  destroy(): void {
    this.matBuf.destroy();
    this.colBuf.destroy();
    this.seedBuf.destroy();
    this.ubo.destroy();
  }
}
