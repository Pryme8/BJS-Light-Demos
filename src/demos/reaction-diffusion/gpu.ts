/**
 * GPU Reaction-Diffusion — fully GPU-driven Gray-Scott simulation (no readback).
 *
 * Two compute passes per dispatch (run N times + once for colorize):
 *   step      — @workgroup_size(16,16) 2D dispatch; reads uFront/vFront, applies
 *               toroidal 4-neighbor Laplacian + Gray-Scott update clamped to [0,1],
 *               writes uBack/vBack. Mirrors grayScottStep() exactly (du *= 0.2).
 *   colorize  — reads final vFront, writes packed rgba8 u32 into colorBuf with the
 *               same v->RGB mapping as writeTexture().
 *
 * After the passes a copyBufferToTexture writes colorBuf into the owned Texture2D
 * (created with COPY_DST by createTexture2DFromPixels). No CPU readback ever.
 *
 * Grid buffers are allocated per-size (NOT pre-allocated at 4096 max) to keep
 * small grids cheap. At 4096: each f32 buffer = 64MB (< 128MB device limit);
 * u+v are kept separate to stay under that limit.
 *
 * bytesPerRow alignment: n*4 bytes. WebGPU requires ≥256 and a multiple of 256.
 * The slider step is 64, so n is always a multiple of 64; n*4 is always a
 * multiple of 256. ✓
 */

import {
  createTexture2DFromPixels,
} from "@babylonjs/lite";
import type { EngineContext, Texture2D } from "@babylonjs/lite";

// ── Constants ──────────────────────────────────────────────────────────────

export const GpuMaxGrid = 4096;

// UBO: 8 words (32 bytes).
const UBOWords = 8;

// ── WGSL ─────────────────────────────────────────────────────────────────

const WGSL_STEP = /* wgsl */ `
struct Params {
  gridN : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
  feed  : f32,
  kill  : f32,
  du    : f32,
  dv    : f32,
}

@group(0) @binding(0) var<uniform>             params : Params;
@group(0) @binding(1) var<storage, read>       uFront : array<f32>;
@group(0) @binding(2) var<storage, read>       vFront : array<f32>;
@group(0) @binding(3) var<storage, read_write> uBack  : array<f32>;
@group(0) @binding(4) var<storage, read_write> vBack  : array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let n = params.gridN;
  if (x >= n || y >= n) { return; }

  let i     = y * n + x;
  let left  = y * n + select(x - 1u, n - 1u, x == 0u);
  let right = y * n + select(x + 1u, 0u,     x == n - 1u);
  let up    = select(y - 1u, n - 1u, y == 0u) * n + x;
  let down  = select(y + 1u, 0u,     y == n - 1u) * n + x;

  let ui  = uFront[i];
  let vi  = vFront[i];
  let lapU = uFront[left] + uFront[right] + uFront[up] + uFront[down] - 4.0 * ui;
  let lapV = vFront[left] + vFront[right] + vFront[up] + vFront[down] - 4.0 * vi;
  let uvv  = ui * vi * vi;

  uBack[i] = clamp(ui + params.du * lapU - uvv + params.feed * (1.0 - ui), 0.0, 1.0);
  vBack[i] = clamp(vi + params.dv * lapV + uvv - (params.feed + params.kill) * vi, 0.0, 1.0);
}
`;

const WGSL_COLORIZE = /* wgsl */ `
struct Params {
  gridN : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
  feed  : f32,
  kill  : f32,
  du    : f32,
  dv    : f32,
}

@group(0) @binding(0) var<uniform>             params   : Params;
@group(0) @binding(1) var<storage, read>       vFront   : array<f32>;
@group(0) @binding(2) var<storage, read_write> colorBuf : array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let n = params.gridN;
  if (x >= n || y >= n) { return; }

  let i  = y * n + x;
  let vi = vFront[i];

  // Mirrors writeTexture() color mapping:
  //   t = min(vi * 4, 1)
  //   r = t*t*20 + vi*30
  //   g = vi*180 + t*60
  //   b = vi*255 + (1-vi)*40
  let t  = min(vi * 4.0, 1.0);
  let r  = u32(clamp(t * t * 20.0 + vi * 30.0,   0.0, 255.0));
  let g  = u32(clamp(vi * 180.0   + t  * 60.0,   0.0, 255.0));
  let b  = u32(clamp(vi * 255.0   + (1.0 - vi) * 40.0, 0.0, 255.0));

  colorBuf[i] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}
`;

// ── Params type ───────────────────────────────────────────────────────────

export interface RDGpuParams {
  feed: number;
  kill: number;
  Du:   number;
  Dv:   number;
}

// ── Per-size state (reallocated on setGrid) ───────────────────────────────

interface GridState {
  n:        number;
  uA:       GPUBuffer;
  uB:       GPUBuffer;
  vA:       GPUBuffer;
  vB:       GPUBuffer;
  colorBuf: GPUBuffer;
  tex:      Texture2D;
  bgAtoB:   GPUBindGroup;  // step:  front=A, back=B
  bgBtoA:   GPUBindGroup;  // step:  front=B, back=A
  bgColorA: GPUBindGroup;  // colorize: vFront=A
  bgColorB: GPUBindGroup;  // colorize: vFront=B
}

// ── ReactionDiffusionGpu ─────────────────────────────────────────────────

export class ReactionDiffusionGpu {
  private readonly engine: EngineContext;
  private readonly device: GPUDevice;

  private readonly ubo:     GPUBuffer;
  private readonly uboData: Uint32Array;

  private readonly stepBGL:     GPUBindGroupLayout;
  private readonly colorizeBGL: GPUBindGroupLayout;
  private readonly stepPipeline:     GPUComputePipeline;
  private readonly colorizePipeline: GPUComputePipeline;

  private state: GridState | null = null;
  private front: "A" | "B" = "A";

  private _ok = false;
  get ok(): boolean { return this._ok; }

  get texture(): Texture2D | null { return this.state?.tex ?? null; }

  constructor(engine: EngineContext) {
    this.engine = engine;
    const dev = (engine as unknown as { _device: GPUDevice })._device;
    this.device = dev;

    this.uboData = new Uint32Array(UBOWords);
    this.ubo = dev.createBuffer({
      label: "rd-ubo",
      size:  UBOWords * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    try {
      this.stepBGL     = this.buildStepBGL();
      this.colorizeBGL = this.buildColorizeBGL();
      this.stepPipeline     = this.buildPipeline("rd-step",     WGSL_STEP,     this.stepBGL);
      this.colorizePipeline = this.buildPipeline("rd-colorize", WGSL_COLORIZE, this.colorizeBGL);
    } catch (e) {
      console.error("[ReactionDiffusionGpu] pipeline build failed:", e);
      this.stepBGL = null!; this.colorizeBGL = null!;
      this.stepPipeline = null!; this.colorizePipeline = null!;
      return;
    }

    this._ok = true;
  }

  private buildStepBGL(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "rd-step-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
  }

  private buildColorizeBGL(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "rd-colorize-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
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

  /**
   * Allocate (or reallocate) all per-grid buffers and texture for size n×n.
   * Destroys the previous allocation. Calls seedCenter() immediately.
   * Returns the new Texture2D so the scene can swap `mat.emissiveTexture`.
   *
   * NOTE: at n=4096 each f32 buffer is 64 MB (×4 = 256 MB total state +
   * 64 MB color buffer = 320 MB). Use with care on low-memory devices.
   */
  setGrid(n: number): Texture2D {
    if (!this._ok) return null!;

    // Destroy previous allocation.
    if (this.state) {
      this.state.uA.destroy();
      this.state.uB.destroy();
      this.state.vA.destroy();
      this.state.vB.destroy();
      this.state.colorBuf.destroy();
      // Texture2D.texture is a GPUTexture; call .destroy() on it.
      this.state.tex.texture.destroy();
    }

    const dev  = this.device;
    const size = n * n * 4;

    const stUsage  = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const colUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    const uA = dev.createBuffer({ label: "rd-uA", size, usage: stUsage });
    const uB = dev.createBuffer({ label: "rd-uB", size, usage: stUsage });
    const vA = dev.createBuffer({ label: "rd-vA", size, usage: stUsage });
    const vB = dev.createBuffer({ label: "rd-vB", size, usage: stUsage });
    const colorBuf = dev.createBuffer({ label: "rd-color", size, usage: colUsage });

    const tex = createTexture2DFromPixels(
      this.engine,
      new Uint8Array(n * n * 4),
      n, n,
      { format: "rgba8unorm", generateMipmaps: false },
    );

    const bgAtoB = dev.createBindGroup({ label: "rd-step-AtoB", layout: this.stepBGL,
      entries: [
        { binding: 0, resource: { buffer: this.ubo } },
        { binding: 1, resource: { buffer: uA } },
        { binding: 2, resource: { buffer: vA } },
        { binding: 3, resource: { buffer: uB } },
        { binding: 4, resource: { buffer: vB } },
      ],
    });
    const bgBtoA = dev.createBindGroup({ label: "rd-step-BtoA", layout: this.stepBGL,
      entries: [
        { binding: 0, resource: { buffer: this.ubo } },
        { binding: 1, resource: { buffer: uB } },
        { binding: 2, resource: { buffer: vB } },
        { binding: 3, resource: { buffer: uA } },
        { binding: 4, resource: { buffer: vA } },
      ],
    });
    const bgColorA = dev.createBindGroup({ label: "rd-colorize-A", layout: this.colorizeBGL,
      entries: [
        { binding: 0, resource: { buffer: this.ubo } },
        { binding: 1, resource: { buffer: vA } },
        { binding: 2, resource: { buffer: colorBuf } },
      ],
    });
    const bgColorB = dev.createBindGroup({ label: "rd-colorize-B", layout: this.colorizeBGL,
      entries: [
        { binding: 0, resource: { buffer: this.ubo } },
        { binding: 1, resource: { buffer: vB } },
        { binding: 2, resource: { buffer: colorBuf } },
      ],
    });

    this.state = { n, uA, uB, vA, vB, colorBuf, tex, bgAtoB, bgBtoA, bgColorA, bgColorB };
    this.front = "A";
    this.seedCenter();
    return tex;
  }

  /**
   * Seed the current front buffer: u=1 everywhere, v=0, then a center square
   * of u=0.5/v=0.25 (radius scales with grid size to keep it proportional).
   */
  seedCenter(): void {
    if (!this.state) return;
    const { n, uA, vA } = this.state;
    const cells = n * n;
    const uData = new Float32Array(cells);
    const vData = new Float32Array(cells);
    uData.fill(1);
    const cx = n >> 1;
    const r  = Math.max(4, Math.round(n * 0.047));  // ~12 cells at 256, scales up
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const i = (cx + dx) + (cx + dy) * n;
        if (i >= 0 && i < cells) { uData[i] = 0.5; vData[i] = 0.25; }
      }
    }
    this.device.queue.writeBuffer(uA, 0, uData.buffer);
    this.device.queue.writeBuffer(vA, 0, vData.buffer);
    this.front = "A";
  }

  /**
   * Dispatch one frame: `steps` Gray-Scott step passes, one colorize pass, then
   * copy colorBuf -> display texture. dt is unused (each step is one discrete
   * Gray-Scott iteration, same as the CPU version).
   */
  dispatch(steps: number, p: RDGpuParams): void {
    if (!this._ok || !this.state) return;
    const { n, bgAtoB, bgBtoA, bgColorA, bgColorB, colorBuf, tex } = this.state;

    // Write UBO.
    const u  = this.uboData;
    const fv = new DataView(u.buffer);
    u[0] = n;
    fv.setFloat32(4 * 4, p.feed,      true);
    fv.setFloat32(5 * 4, p.kill,      true);
    fv.setFloat32(6 * 4, p.Du * 0.2,  true);
    fv.setFloat32(7 * 4, p.Dv * 0.2,  true);
    this.device.queue.writeBuffer(this.ubo, 0, u.buffer);

    const wg   = Math.ceil(n / 16);
    const enc  = this.device.createCommandEncoder({ label: "rd-compute" });
    const pass = enc.beginComputePass({ label: "rd" });

    // Step passes.
    pass.setPipeline(this.stepPipeline);
    for (let s = 0; s < steps; s++) {
      const bg = this.front === "A" ? bgAtoB : bgBtoA;
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wg, wg);
      this.front = this.front === "A" ? "B" : "A";
    }

    // Colorize pass: reads the current front (which has the final state).
    pass.setPipeline(this.colorizePipeline);
    const bgColor = this.front === "A" ? bgColorA : bgColorB;
    pass.setBindGroup(0, bgColor);
    pass.dispatchWorkgroups(wg, wg);

    pass.end();

    // Copy rgba8 buffer -> display texture.
    // bytesPerRow = n*4; n is always a multiple of 64 (slider step=64),
    // so n*4 is always a multiple of 256 — satisfying the WebGPU alignment req.
    enc.copyBufferToTexture(
      { buffer: colorBuf, bytesPerRow: n * 4, rowsPerImage: n },
      { texture: tex.texture },
      { width: n, height: n, depthOrArrayLayers: 1 },
    );

    this.device.queue.submit([enc.finish()]);
  }

  destroy(): void {
    if (this.state) {
      this.state.uA.destroy();
      this.state.uB.destroy();
      this.state.vA.destroy();
      this.state.vB.destroy();
      this.state.colorBuf.destroy();
      this.state.tex.texture.destroy();
      this.state = null;
    }
    this.ubo.destroy();
  }
}
