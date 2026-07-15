/**
 * GPU Lotka-Volterra — compute-driven agent random walk (no readback).
 *
 * Architecture
 * ──────────────
 * The scalar RK4 ODE (preyPop/predPop) runs on the CPU as before; only the
 * per-agent random-walk and instance-matrix writes are offloaded here.
 *
 * Each group (prey / predator) has:
 *   state  — STORAGE|COPY_DST, 2 × vec4<f32> per agent (pos + vel), updated
 *             in place each frame (no ping-pong — agents are independent).
 *   matBuf — STORAGE|VERTEX|COPY_DST, 16 × f32 per agent (column-major mat4).
 *             Owned and wired directly into ti._gpuBuffer so Lite never
 *             recreates or overwrites it.
 *
 * One shared WGSL compute pipeline replicates `moveAgents` from scene.ts:
 *   - PCG hash PRNG seeded per (agent-index × frame-seed) gives two randoms.
 *   - vx += (r1-0.5)*0.4; vz += (r2-0.5)*0.4 (xz heading jitter).
 *   - Normalize xz heading.
 *   - Boundary avoidance: if |px| > bound: vx -= sign(px)*0.3 (same for pz).
 *   - px += vx*speed*dt; pz += vz*speed*dt; y unchanged.
 *   - Write identity-rotation, scale-1 translation mat4 into matBuf.
 */

import {
  addToScene,
  createPbrMaterial,
  createSphere,
  invalidateRenderBundles,
  setThinInstanceCount,
  setThinInstances,
} from "@babylonjs/lite";
import type { EngineContext, Mesh, SceneContext } from "@babylonjs/lite";

// ── Exported constants ────────────────────────────────────────────────────

export const GpuMaxPrey = 80_000;
export const GpuMaxPred = 20_000;
export const GpuBound   = 60;

// ── Internal ──────────────────────────────────────────────────────────────

// State: pos (vec4) + vel (vec4) = 32 bytes per agent.
const BytesPerAgent = 32;

// UBO: count, dt_s, speed, bound, seed, + padding to 16 words (64 bytes).
const UBOWords = 16;

type ThinInstData = {
  _gpuBuffer: GPUBuffer | null;  _gpuBufferStorage: boolean;
  _gpuVersion: number;           _version: number;
};

// ── WGSL ─────────────────────────────────────────────────────────────────

const WGSL_MOVE = /* wgsl */ `
struct Params {
  count : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
  dt_s  : f32,
  speed : f32,
  bound : f32,
  seed  : u32,
  _p0:u32,_p1:u32,_p2:u32,_p3:u32,_p4:u32,_p5:u32,_p6:u32,_p7:u32,
}

struct Agent {
  pos : vec4<f32>,
  vel : vec4<f32>,
}

@group(0) @binding(0) var<uniform>             params : Params;
@group(0) @binding(1) var<storage, read_write> agents : array<Agent>;
@group(0) @binding(2) var<storage, read_write> matBuf : array<f32>;

// PCG hash — cheap, high-quality random u32 from a u32 seed.
fn pcg(v: u32) -> u32 {
  var s = v * 747796405u + 2891336453u;
  s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (s >> 22u) ^ s;
}

// Two independent floats in [0,1) from one index + frame seed.
fn rand2(idx: u32) -> vec2<f32> {
  let h1 = pcg(idx ^ (params.seed * 1664525u + 1013904223u));
  let h2 = pcg(h1 ^ 0xDEADBEEFu);
  return vec2<f32>(f32(h1) / 4294967296.0, f32(h2) / 4294967296.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  var pos = agents[i].pos.xyz;
  var vel = agents[i].vel;
  var vx  = vel.x;
  var vz  = vel.z;

  // Heading jitter — mirrors: vx += (r-0.5)*0.4; vz += (r-0.5)*0.4
  let r   = rand2(i);
  vx += (r.x - 0.5) * 0.4;
  vz += (r.y - 0.5) * 0.4;

  // Normalize xz — mirrors: len = sqrt(vx*vx+vz*vz); if len>0.001 normalize
  let len = sqrt(vx*vx + vz*vz);
  if (len > 0.001) { vx /= len; vz /= len; }

  // Boundary avoidance — mirrors: if abs(px)>Bound vx -= sign(px)*0.3
  let bound = params.bound;
  if (abs(pos.x) > bound) { vx -= sign(pos.x) * 0.3; }
  if (abs(pos.z) > bound) { vz -= sign(pos.z) * 0.3; }

  // Integrate — mirrors: px += vx*speed*s; pz += vz*speed*s; y unchanged
  pos.x += vx * params.speed * params.dt_s;
  pos.z += vz * params.speed * params.dt_s;

  // Write back state.
  agents[i].pos = vec4<f32>(pos, agents[i].pos.w);
  agents[i].vel = vec4<f32>(vx, agents[i].vel.y, vz, 0.0);

  // Write scale-1 translation mat4 (column-major).
  let mb = i * 16u;
  matBuf[mb+ 0u]=1.0; matBuf[mb+ 1u]=0.0; matBuf[mb+ 2u]=0.0; matBuf[mb+ 3u]=0.0;
  matBuf[mb+ 4u]=0.0; matBuf[mb+ 5u]=1.0; matBuf[mb+ 6u]=0.0; matBuf[mb+ 7u]=0.0;
  matBuf[mb+ 8u]=0.0; matBuf[mb+ 9u]=0.0; matBuf[mb+10u]=1.0; matBuf[mb+11u]=0.0;
  matBuf[mb+12u]=pos.x; matBuf[mb+13u]=pos.y; matBuf[mb+14u]=pos.z; matBuf[mb+15u]=1.0;
}
`;

// ── AgentGroup — per-species buffers / mesh / bind-group ─────────────────

interface AgentGroup {
  mesh:     Mesh;
  ti:       ThinInstData;
  stateBuf: GPUBuffer;
  matBuf:   GPUBuffer;
  bg:       GPUBindGroup;
  lastCount: number;
}

// ── LotkaVolterraGpu ─────────────────────────────────────────────────────

export class LotkaVolterraGpu {
  private readonly engine: EngineContext;
  private readonly device: GPUDevice;

  private readonly prey: AgentGroup;
  private readonly pred: AgentGroup;

  private readonly ubo:     GPUBuffer;
  private readonly uboData: Uint32Array;

  private readonly bgl:      GPUBindGroupLayout;
  private readonly pipeline: GPUComputePipeline;

  private _ok = false;
  get ok(): boolean { return this._ok; }

  private _frameCount = 0;

  get preyMesh(): Mesh { return this.prey.mesh; }
  get predMesh(): Mesh { return this.pred.mesh; }

  constructor(engine: EngineContext, scene: SceneContext) {
    this.engine = engine;
    const dev = (engine as unknown as { _device: GPUDevice })._device;
    this.device = dev;

    this.uboData = new Uint32Array(UBOWords);
    this.ubo = dev.createBuffer({
      label: "lv-ubo",
      size:  UBOWords * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ── Build shared pipeline ─────────────────────────────────────────────
    let bgl: GPUBindGroupLayout;
    let pipeline: GPUComputePipeline;
    try {
      bgl = dev.createBindGroupLayout({
        label: "lv-bgl",
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ],
      });
      const mod = dev.createShaderModule({ label: "lv-move", code: WGSL_MOVE });
      pipeline = dev.createComputePipeline({
        label: "lv-move",
        layout: dev.createPipelineLayout({ label: "lv-move-layout", bindGroupLayouts: [bgl] }),
        compute: { module: mod, entryPoint: "main" },
      });
    } catch (e) {
      console.error("[LotkaVolterraGpu] pipeline build failed:", e);
      this.bgl = null!; this.pipeline = null!;
      this.prey = null!; this.pred = null!;
      return;
    }
    this.bgl      = bgl;
    this.pipeline = pipeline;

    // ── Build agent groups ─────────────────────────────────────────────────
    this.prey = this.buildGroup(
      engine, scene, dev, bgl, GpuMaxPrey,
      { segments: 6, diameter: 0.6 },
      { baseColorFactor: [0.05, 0.85, 0.9, 1], metallicFactor: 0.1, roughnessFactor: 0.6 },
    );
    this.pred = this.buildGroup(
      engine, scene, dev, bgl, GpuMaxPred,
      { segments: 6, diameter: 1.0 },
      { baseColorFactor: [1, 0.05, 0.7, 1], metallicFactor: 0.3, roughnessFactor: 0.4 },
    );

    this._ok = true;
  }

  private buildGroup(
    engine: EngineContext,
    scene: SceneContext,
    dev: GPUDevice,
    bgl: GPUBindGroupLayout,
    maxAgents: number,
    sphereOpts: { segments: number; diameter: number },
    matOpts: { baseColorFactor: [number,number,number,number]; metallicFactor: number; roughnessFactor: number },
  ): AgentGroup {
    // Owned instance matrix buffer.
    const matBuf = dev.createBuffer({
      label: "lv-mat",
      size:  maxAgents * 64,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    // In-place agent state buffer (no ping-pong).
    const stateBuf = dev.createBuffer({
      label: "lv-state",
      size:  maxAgents * BytesPerAgent,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // PBR sphere mesh — addToScene first (mirrors boids/n-body pattern).
    const sphere = createSphere(engine, sphereOpts);
    sphere.material = createPbrMaterial(matOpts);
    addToScene(scene, sphere);

    // Establish thin-instance pool at max capacity, start with count 0.
    const identMats = new Float32Array(maxAgents * 16);
    for (let i = 0; i < maxAgents; i++) {
      const b = i * 16;
      identMats[b] = 1; identMats[b+5] = 1; identMats[b+10] = 1; identMats[b+15] = 1;
    }
    setThinInstances(sphere, identMats, maxAgents);
    setThinInstanceCount(sphere, 0);

    // Hand our owned buffer to Lite; keep versions in sync so it never recreates.
    const ti = (sphere as unknown as { thinInstances: ThinInstData }).thinInstances;
    ti._gpuBuffer        = matBuf;
    ti._gpuBufferStorage = false;
    ti._gpuVersion       = ti._version;

    // Bind group for this species.
    const bg = dev.createBindGroup({
      label: "lv-bg",
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: this.ubo } },
        { binding: 1, resource: { buffer: stateBuf } },
        { binding: 2, resource: { buffer: matBuf } },
      ],
    });

    return { mesh: sphere, ti, stateBuf, matBuf, bg, lastCount: 0 };
  }

  /** Set visible prey count and resync the version guard. */
  setPreyCount(n: number): void { this.setGroupCount(this.prey, n); }

  /** Set visible predator count and resync the version guard. */
  setPredCount(n: number): void { this.setGroupCount(this.pred, n); }

  private setGroupCount(g: AgentGroup, n: number): void {
    setThinInstanceCount(g.mesh, n);
    g.ti._gpuVersion = g.ti._version;
    if (g.lastCount !== n) {
      g.lastCount = n;
      invalidateRenderBundles(this.engine);
    }
  }

  /**
   * Seed both groups to their full capacity with random positions/headings.
   * Only the live slice [0, count) will be drawn; pre-filling the rest avoids
   * uninitialised reads in the shader when count grows mid-sim.
   */
  seedRandom(bound: number): void {
    const fill = (max: number) => {
      const data = new Float32Array(max * 8);
      for (let i = 0; i < max; i++) {
        const b = i * 8;
        data[b  ] = (Math.random() - 0.5) * bound * 2;
        data[b+1] = (Math.random() - 0.5) * 2;
        data[b+2] = (Math.random() - 0.5) * bound * 2;
        data[b+3] = 0;
        const a = Math.random() * Math.PI * 2;
        data[b+4] = Math.cos(a);
        data[b+5] = 0;
        data[b+6] = Math.sin(a);
        data[b+7] = 0;
      }
      return data;
    };
    this.device.queue.writeBuffer(this.prey.stateBuf, 0, fill(GpuMaxPrey).buffer);
    this.device.queue.writeBuffer(this.pred.stateBuf, 0, fill(GpuMaxPred).buffer);
    this._frameCount = 0;
  }

  /**
   * Dispatch one frame of agent movement for both species.
   * Prey and pred get distinct seeds so their headings evolve independently.
   * Each group uses a separate command buffer so UBO writes don't race with
   * an in-flight compute pass.
   */
  dispatch(dt: number, p: { preyCount: number; predCount: number; preySpeed: number; predSpeed: number; bound: number }): void {
    if (!this._ok) return;
    this._frameCount++;
    const dt_s = dt * 0.001;
    const seed = (this._frameCount * 1664525 + 1013904223) >>> 0;

    const cmds: GPUCommandBuffer[] = [];

    // Prey command.
    if (p.preyCount > 0) {
      this.writeUBO(p.preyCount, dt_s, p.preySpeed, p.bound, seed);
      const enc  = this.device.createCommandEncoder({ label: "lv-prey" });
      const pass = enc.beginComputePass({ label: "lv-prey" });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.prey.bg);
      pass.dispatchWorkgroups(Math.ceil(p.preyCount / 64));
      pass.end();
      cmds.push(enc.finish());
    }

    // Pred command (distinct seed by XOR so headings diverge).
    if (p.predCount > 0) {
      this.writeUBO(p.predCount, dt_s, p.predSpeed, p.bound, seed ^ 0xC0FFEE);
      const enc  = this.device.createCommandEncoder({ label: "lv-pred" });
      const pass = enc.beginComputePass({ label: "lv-pred" });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.pred.bg);
      pass.dispatchWorkgroups(Math.ceil(p.predCount / 64));
      pass.end();
      cmds.push(enc.finish());
    }

    if (cmds.length) this.device.queue.submit(cmds);
  }

  private writeUBO(count: number, dt_s: number, speed: number, bound: number, seed: number): void {
    const u   = this.uboData;
    const fv  = new DataView(u.buffer);
    u[0] = count;
    fv.setFloat32(4  * 4, dt_s,  true);
    fv.setFloat32(5  * 4, speed, true);
    fv.setFloat32(6  * 4, bound, true);
    u[7] = seed >>> 0;
    this.device.queue.writeBuffer(this.ubo, 0, u.buffer);
  }

  destroy(): void {
    this.prey.matBuf.destroy();
    this.prey.stateBuf.destroy();
    this.pred.matBuf.destroy();
    this.pred.stateBuf.destroy();
    this.ubo.destroy();
  }
}
