/**
 * GPU Double Pendulum — thousands of independent double pendulums integrated
 * on the GPU, all overlaid at one shared pivot (chaos fan visualization).
 *
 * One compute pass per frame, `@workgroup_size(64)`:
 *   - Runs `substeps` RK4 substeps per pendulum, replicating `doublePendulumAccel`
 *     and `rk4Step` exactly in f32.
 *   - Computes arm1/arm2 cylinder and bob sphere TRS mat4s with the SAME transform
 *     formula as the CPU demo (Z-axis rotation, arm centered at midpoint, bob at b2).
 *   - Colors each arm+bob by the rainbow `pendHue` mapping (cyan→magenta by index).
 *   - Writes directly into owned STORAGE|VERTEX buffers that bypass Lite's CPU sync.
 *
 * No trailing history in GPU mode (n×200 trail instances would reach millions).
 * State is updated in place — pendulums are independent, so no ping-pong needed.
 *
 * NOTE: f32 RK4 on GPU vs f64 RK4 on CPU — individual trajectories diverge after
 * several seconds due to floating-point precision differences. This is expected and
 * doesn't affect the visual (both are chaotic anyway).
 */

import {
  addToScene,
  createCylinder,
  createPbrMaterial,
  createSphere,
  invalidateRenderBundles,
  setThinInstanceColors,
  setThinInstanceCount,
  setThinInstances,
} from "@babylonjs/lite";
import type { EngineContext, Mesh, SceneContext } from "@babylonjs/lite";

// ── Constants ──────────────────────────────────────────────────────────────

export const GpuMaxPendulums = 4096;

// 4 f32 per pendulum: theta1, theta2, omega1, omega2.
const BytesPerPendulum = 16;

// UBO: 16 words (64 bytes).
const UBOWords = 16;

// ── Thin-instance internal shape ───────────────────────────────────────────

type ThinInstData = {
  _gpuBuffer: GPUBuffer | null;       _gpuBufferStorage: boolean;
  _gpuVersion: number;                _version: number;
  _colorGpuBuffer: GPUBuffer | null;  _colorGpuBufferStorage: boolean;
  _colorGpuVersion: number;           _colorVersion: number;
};

// ── WGSL ─────────────────────────────────────────────────────────────────

const WGSL_STEP = /* wgsl */ `
struct Params {
  count    : u32,
  substeps : u32,
  dt_s     : f32,
  L1       : f32,
  L2       : f32,
  m1       : f32,
  m2       : f32,
  g        : f32,
  pivotY   : f32,
  _p0:u32, _p1:u32, _p2:u32, _p3:u32, _p4:u32, _p5:u32, _p6:u32,
}

struct State {
  theta1 : f32,
  theta2 : f32,
  omega1 : f32,
  omega2 : f32,
}

@group(0) @binding(0) var<uniform>             params   : Params;
@group(0) @binding(1) var<storage, read_write> state    : array<State>;
@group(0) @binding(2) var<storage, read_write> arm1Mat  : array<f32>;
@group(0) @binding(3) var<storage, read_write> arm1Col  : array<f32>;
@group(0) @binding(4) var<storage, read_write> arm2Mat  : array<f32>;
@group(0) @binding(5) var<storage, read_write> arm2Col  : array<f32>;
@group(0) @binding(6) var<storage, read_write> bobMat   : array<f32>;
@group(0) @binding(7) var<storage, read_write> bobCol   : array<f32>;

// Double-pendulum angular accelerations — mirrors doublePendulumAccel() exactly.
fn dpAccel(t1: f32, t2: f32, o1: f32, o2: f32,
           L1: f32, L2: f32, m1: f32, m2: f32, g: f32) -> vec2<f32> {
  let dt     = t2 - t1;
  let cosdt  = cos(dt);
  let denom1 = (m1 + m2) * L1 - m2 * L1 * cosdt * cosdt;
  let denom2 = (L2 / L1) * denom1;
  let a1 = (m2 * L1 * o1 * o1 * sin(dt) * cosdt
           + m2 * g  * sin(t2) * cosdt
           + m2 * L2 * o2 * o2 * sin(dt)
           - (m1 + m2) * g * sin(t1)) / denom1;
  let a2 = (-(m1 + m2) * L1 * o1 * o1 * sin(dt) * cosdt
            + (m1 + m2) * g * sin(t1) * cosdt
            - (m1 + m2) * L2 * o2 * o2 * sin(dt)
            - (m1 + m2) * g * sin(t2)) / denom2;
  return vec2<f32>(a1, a2);
}

// One RK4 substep — mirrors rk4Step() exactly.
fn rk4(t1: f32, t2: f32, o1: f32, o2: f32,
        ss: f32, L1: f32, L2: f32, m1: f32, m2: f32, g: f32) -> vec4<f32> {
  let h = ss;
  let k1 = dpAccel(t1,              t2,              o1,              o2,              L1, L2, m1, m2, g);
  let k2 = dpAccel(t1 + o1*h*0.5,  t2 + o2*h*0.5,  o1 + k1.x*h*0.5, o2 + k1.y*h*0.5, L1, L2, m1, m2, g);
  let k3 = dpAccel(t1 + o1*h*0.5,  t2 + o2*h*0.5,  o1 + k2.x*h*0.5, o2 + k2.y*h*0.5, L1, L2, m1, m2, g);
  let k4 = dpAccel(t1 + o1*h,      t2 + o2*h,      o1 + k3.x*h,     o2 + k3.y*h,     L1, L2, m1, m2, g);

  let new_o1 = o1 + (k1.x + 2.0*k2.x + 2.0*k3.x + k4.x) * h * 0.1667;
  let new_o2 = o2 + (k1.y + 2.0*k2.y + 2.0*k3.y + k4.y) * h * 0.1667;
  let new_t1 = t1 + new_o1 * h;
  let new_t2 = t2 + new_o2 * h;
  return vec4<f32>(new_t1, new_t2, new_o1, new_o2);
}

// Write a Z-rotation cylinder mat4 (column-major).
// Local +Y cylinder, centered at (cx,cy,0), rotated angle a about Z,
// Y-scaled to full arm length len (cylinder prototype height=1 → scale len).
fn writeArmMat(base: u32, cx: f32, cy: f32, ang: f32, len: f32) {
  let ca = cos(ang); let sa = sin(ang);
  // Column-major: rotation about Z, non-uniform scale sx=1, sy=len, sz=1.
  arm1Mat[base+ 0u]=ca;  arm1Mat[base+ 1u]=sa;  arm1Mat[base+ 2u]=0.0; arm1Mat[base+ 3u]=0.0;
  arm1Mat[base+ 4u]=-sa*len; arm1Mat[base+ 5u]=ca*len; arm1Mat[base+ 6u]=0.0; arm1Mat[base+ 7u]=0.0;
  arm1Mat[base+ 8u]=0.0; arm1Mat[base+ 9u]=0.0; arm1Mat[base+10u]=1.0; arm1Mat[base+11u]=0.0;
  arm1Mat[base+12u]=cx;  arm1Mat[base+13u]=cy;  arm1Mat[base+14u]=0.0; arm1Mat[base+15u]=1.0;
}
fn writeArm2Mat(base: u32, cx: f32, cy: f32, ang: f32, len: f32) {
  let ca = cos(ang); let sa = sin(ang);
  arm2Mat[base+ 0u]=ca;  arm2Mat[base+ 1u]=sa;  arm2Mat[base+ 2u]=0.0; arm2Mat[base+ 3u]=0.0;
  arm2Mat[base+ 4u]=-sa*len; arm2Mat[base+ 5u]=ca*len; arm2Mat[base+ 6u]=0.0; arm2Mat[base+ 7u]=0.0;
  arm2Mat[base+ 8u]=0.0; arm2Mat[base+ 9u]=0.0; arm2Mat[base+10u]=1.0; arm2Mat[base+11u]=0.0;
  arm2Mat[base+12u]=cx;  arm2Mat[base+13u]=cy;  arm2Mat[base+14u]=0.0; arm2Mat[base+15u]=1.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  var t1 = state[i].theta1;
  var t2 = state[i].theta2;
  var o1 = state[i].omega1;
  var o2 = state[i].omega2;

  let L1 = params.L1; let L2 = params.L2;
  let m1 = params.m1; let m2 = params.m2;
  let g  = params.g;
  let ss = params.dt_s / f32(params.substeps);

  // RK4 substeps.
  for (var k = 0u; k < params.substeps; k++) {
    let r = rk4(t1, t2, o1, o2, ss, L1, L2, m1, m2, g);
    t1 = r.x; t2 = r.y; o1 = r.z; o2 = r.w;
  }

  state[i].theta1 = t1; state[i].theta2 = t2;
  state[i].omega1 = o1; state[i].omega2 = o2;

  // Shared pivot (chaos fan — all pendulums start at the same point).
  let ox = 0.0;
  let oy = params.pivotY;

  let b1x = ox + L1 * sin(t1);
  let b1y = oy - L1 * cos(t1);
  let b2x = b1x + L2 * sin(t2);
  let b2y = b1y - L2 * cos(t2);

  // Arm 1: midpoint between pivot and bob1; angle = atan2(b1x-ox, oy-b1y).
  let a1cx = (ox + b1x) * 0.5;
  let a1cy = (oy + b1y) * 0.5;
  let a1ang = atan2(b1x - ox, oy - b1y);
  let a1len = length(vec2<f32>(b1x - ox, b1y - oy));

  // Arm 2: midpoint between bob1 and bob2.
  let a2cx = (b1x + b2x) * 0.5;
  let a2cy = (b1y + b2y) * 0.5;
  let a2ang = atan2(b2x - b1x, b1y - b2y);
  let a2len = length(vec2<f32>(b2x - b1x, b2y - b1y));

  // Write arm mats.
  let mb = i * 16u;
  writeArmMat(mb, a1cx, a1cy, a1ang, a1len);
  writeArm2Mat(mb, a2cx, a2cy, a2ang, a2len);

  // Bob mat4: uniform scale 1, translation at b2.
  bobMat[mb+ 0u]=1.0; bobMat[mb+ 1u]=0.0; bobMat[mb+ 2u]=0.0; bobMat[mb+ 3u]=0.0;
  bobMat[mb+ 4u]=0.0; bobMat[mb+ 5u]=1.0; bobMat[mb+ 6u]=0.0; bobMat[mb+ 7u]=0.0;
  bobMat[mb+ 8u]=0.0; bobMat[mb+ 9u]=0.0; bobMat[mb+10u]=1.0; bobMat[mb+11u]=0.0;
  bobMat[mb+12u]=b2x; bobMat[mb+13u]=b2y; bobMat[mb+14u]=0.0; bobMat[mb+15u]=1.0;

  // Rainbow hue: pendHue(i, count) = (t, 1-t*0.8, 1-t*0.4) cyan→magenta.
  let t  = select(f32(i) / f32(params.count - 1u), 0.0, params.count <= 1u);
  let r  = t;
  let gr = 1.0 - t * 0.8;
  let b  = 1.0 - t * 0.4;

  let cb = i * 4u;
  // Arm colors (slightly dimmed for contrast with bobs).
  arm1Col[cb+0u]=r*0.7; arm1Col[cb+1u]=gr*0.7; arm1Col[cb+2u]=b*0.7; arm1Col[cb+3u]=1.0;
  arm2Col[cb+0u]=r*0.55; arm2Col[cb+1u]=gr*0.55; arm2Col[cb+2u]=b*0.55; arm2Col[cb+3u]=1.0;
  // Bob at full brightness.
  bobCol[cb+0u]=r; bobCol[cb+1u]=gr; bobCol[cb+2u]=b; bobCol[cb+3u]=1.0;
}
`;

// ── MeshGroup — per-mesh buffers wired to Lite's thin-instance internals ──

interface MeshGroup {
  mesh:   Mesh;
  ti:     ThinInstData;
  matBuf: GPUBuffer;
  colBuf: GPUBuffer;
}

// ── DoublePendulumGpu ─────────────────────────────────────────────────────

export interface DPGpuParams {
  count:    number;
  L1:       number;
  L2:       number;
  m1:       number;
  m2:       number;
  g:        number;
  speed:    number;
  substeps: number;
  pivotY:   number;
}

export class DoublePendulumGpu {
  private readonly engine: EngineContext;
  private readonly device: GPUDevice;

  private readonly arm1: MeshGroup;
  private readonly arm2: MeshGroup;
  private readonly bob:  MeshGroup;

  private readonly stateBuf: GPUBuffer;
  private readonly ubo:      GPUBuffer;
  private readonly uboData:  Uint32Array;

  private readonly bgl:      GPUBindGroupLayout;
  private readonly bg:       GPUBindGroup;
  private readonly pipeline: GPUComputePipeline;

  private _lastCount = 0;
  private _ok = false;
  get ok(): boolean { return this._ok; }

  get arm1Mesh(): Mesh { return this.arm1.mesh; }
  get arm2Mesh(): Mesh { return this.arm2.mesh; }
  get bobMesh():  Mesh { return this.bob.mesh; }

  constructor(engine: EngineContext, scene: SceneContext) {
    this.engine = engine;
    const dev = (engine as unknown as { _device: GPUDevice })._device;
    this.device = dev;

    // ── Per-mesh instance buffers ─────────────────────────────────────────
    this.arm1 = this.buildGroup(engine, scene,
      createCylinder(engine, { height: 1, diameter: 0.15, tessellation: 8 }),
      createPbrMaterial({ baseColorFactor: [0.9, 0.9, 0.9, 1], metallicFactor: 0.8, roughnessFactor: 0.3 }),
    );
    this.arm2 = this.buildGroup(engine, scene,
      createCylinder(engine, { height: 1, diameter: 0.15, tessellation: 8 }),
      createPbrMaterial({ baseColorFactor: [0.6, 0.6, 0.6, 1], metallicFactor: 0.8, roughnessFactor: 0.3 }),
    );
    this.bob = this.buildGroup(engine, scene,
      createSphere(engine, { segments: 8, diameter: 0.5 }),
      createPbrMaterial({ baseColorFactor: [0.05, 0.85, 1, 1], metallicFactor: 0.4, roughnessFactor: 0.3 }),
    );

    // ── State buffer ──────────────────────────────────────────────────────
    this.stateBuf = dev.createBuffer({
      label: "dp-state",
      size:  GpuMaxPendulums * BytesPerPendulum,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // ── UBO ───────────────────────────────────────────────────────────────
    this.uboData = new Uint32Array(UBOWords);
    this.ubo = dev.createBuffer({
      label: "dp-ubo",
      size:  UBOWords * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ── Pipeline ──────────────────────────────────────────────────────────
    let bgl: GPUBindGroupLayout;
    let bg:  GPUBindGroup;
    let pipeline: GPUComputePipeline;
    try {
      bgl = dev.createBindGroupLayout({
        label: "dp-bgl",
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ],
      });
      const mod = dev.createShaderModule({ label: "dp-step", code: WGSL_STEP });
      pipeline = dev.createComputePipeline({
        label: "dp-step",
        layout: dev.createPipelineLayout({ label: "dp-step-layout", bindGroupLayouts: [bgl] }),
        compute: { module: mod, entryPoint: "main" },
      });
      bg = dev.createBindGroup({
        label: "dp-bg", layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: this.ubo } },
          { binding: 1, resource: { buffer: this.stateBuf } },
          { binding: 2, resource: { buffer: this.arm1.matBuf } },
          { binding: 3, resource: { buffer: this.arm1.colBuf } },
          { binding: 4, resource: { buffer: this.arm2.matBuf } },
          { binding: 5, resource: { buffer: this.arm2.colBuf } },
          { binding: 6, resource: { buffer: this.bob.matBuf } },
          { binding: 7, resource: { buffer: this.bob.colBuf } },
        ],
      });
    } catch (e) {
      console.error("[DoublePendulumGpu] pipeline build failed:", e);
      this.bgl = null!; this.bg = null!; this.pipeline = null!;
      return;
    }
    this.bgl = bgl;
    this.bg  = bg;
    this.pipeline = pipeline;
    this._ok = true;
  }

  private buildGroup(engine: EngineContext, scene: SceneContext, mesh: Mesh, mat: ReturnType<typeof createPbrMaterial>): MeshGroup {
    const dev = this.device;

    const instUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST;
    const matBuf = dev.createBuffer({ label: "dp-mat", size: GpuMaxPendulums * 64, usage: instUsage });
    const colBuf = dev.createBuffer({ label: "dp-col", size: GpuMaxPendulums * 16, usage: instUsage });

    mesh.material = mat;
    addToScene(scene, mesh);

    const identMats = new Float32Array(GpuMaxPendulums * 16);
    for (let i = 0; i < GpuMaxPendulums; i++) {
      const b = i * 16;
      identMats[b] = 1; identMats[b+5] = 1; identMats[b+10] = 1; identMats[b+15] = 1;
    }
    setThinInstances(mesh, identMats, GpuMaxPendulums);
    setThinInstanceColors(mesh, new Float32Array(GpuMaxPendulums * 4).fill(1));
    setThinInstanceCount(mesh, 0);

    const ti = (mesh as unknown as { thinInstances: ThinInstData }).thinInstances;
    ti._gpuBuffer             = matBuf;
    ti._gpuBufferStorage      = false;
    ti._gpuVersion            = ti._version;
    ti._colorGpuBuffer        = colBuf;
    ti._colorGpuBufferStorage = false;
    ti._colorGpuVersion       = ti._colorVersion;

    return { mesh, ti, matBuf, colBuf };
  }

  private syncGroup(g: MeshGroup): void {
    g.ti._gpuVersion      = g.ti._version;
    g.ti._colorGpuVersion = g.ti._colorVersion;
  }

  /** Set the active thin-instance count for all three meshes. */
  setCount(n: number): void {
    setThinInstanceCount(this.arm1.mesh, n);
    setThinInstanceCount(this.arm2.mesh, n);
    setThinInstanceCount(this.bob.mesh,  n);
    this.syncGroup(this.arm1);
    this.syncGroup(this.arm2);
    this.syncGroup(this.bob);
    if (this._lastCount !== n) {
      this._lastCount = n;
      invalidateRenderBundles(this.engine);
    }
  }

  /**
   * Seed all pendulums. Initial angles are spread linearly over `spread` radians
   * normalized to count so the fan width stays constant regardless of N.
   */
  seedSpread(count: number, spread: number, base1 = Math.PI * 0.7, base2 = Math.PI * 0.3): void {
    const data = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
      const t = count > 1 ? i / (count - 1) : 0;
      data[i*4  ] = base1 + t * spread;
      data[i*4+1] = base2 + t * spread * 0.8;
      data[i*4+2] = 0;
      data[i*4+3] = 0;
    }
    this.device.queue.writeBuffer(this.stateBuf, 0, data.buffer, 0, count * BytesPerPendulum);
  }

  /** Dispatch one frame of RK4 integration + TRS/color writes. */
  dispatch(dt: number, p: DPGpuParams): void {
    if (!this._ok) return;
    const n     = p.count;
    const dt_s  = dt * 0.001 * p.speed;

    const u  = this.uboData;
    const fv = new DataView(u.buffer);
    u[0] = n;
    u[1] = p.substeps;
    fv.setFloat32(2 * 4, dt_s,    true);
    fv.setFloat32(3 * 4, p.L1,    true);
    fv.setFloat32(4 * 4, p.L2,    true);
    fv.setFloat32(5 * 4, p.m1,    true);
    fv.setFloat32(6 * 4, p.m2,    true);
    fv.setFloat32(7 * 4, p.g,     true);
    fv.setFloat32(8 * 4, p.pivotY, true);
    this.device.queue.writeBuffer(this.ubo, 0, u.buffer);

    const enc  = this.device.createCommandEncoder({ label: "dp-compute" });
    const pass = enc.beginComputePass({ label: "dp" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bg);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  destroy(): void {
    this.arm1.matBuf.destroy();
    this.arm1.colBuf.destroy();
    this.arm2.matBuf.destroy();
    this.arm2.colBuf.destroy();
    this.bob.matBuf.destroy();
    this.bob.colBuf.destroy();
    this.stateBuf.destroy();
    this.ubo.destroy();
  }
}
