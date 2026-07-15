/**
 * N-Body GPU path — tiled O(n²) gravity + GPU orbital trails.
 *
 * Compute passes per dispatch:
 *   simulate  — tiled shared-memory all-pairs gravity (Tile=256); writes body
 *               instance matrices/colors into owned STORAGE|VERTEX buffers
 *               that the PBR sphere mesh reads directly (no readback); optionally
 *               records body positions into a ring-buffer trail history.
 *   buildTrails — one thread per (body × segment); reads two adjacent ring slots,
 *               builds a tapered cylinder TRS into the trail mesh's instance buffers.
 *
 * NOTE: GPU runs in f32; the CPU path uses f64 (n-body is chaotic so trajectories
 * visually diverge over time — that is expected, not a bug).
 *
 * Buffer-ownership pattern mirrors src/demos/boids/gpu.ts:
 * Owned STORAGE|VERTEX|COPY_DST buffers are assigned to `ti._gpuBuffer` /
 * `ti._colorGpuBuffer` and versions are kept in sync, so Lite's CPU upload path
 * never recreates or overwrites them.
 */

import {
  addToScene,
  createCapsule,   // unused – kept for import consistency
  createCylinder,
  createPbrMaterial,
  createSphere,
  invalidateRenderBundles,
  setThinInstanceColors,
  setThinInstanceCount,
  setThinInstances,
} from "@babylonjs/lite";
import type { EngineContext, Mesh, SceneContext } from "@babylonjs/lite";

// Suppress unused import warning for createCapsule
void createCapsule;

// ── Constants ──────────────────────────────────────────────────────────────

export const GpuMaxBodies = 8_192;
const Tile       = 256;   // workgroup size for gravity (powers-of-2, ≤ GPU limit)
export const GpuTrailLen = 48;    // ring-buffer slots; GpuTrailLen-1 segments/body
const HeadThick  = 0.22;  // trail head thickness (tapers to 0 at tail)

// State buffer: 2 × vec4<f32> per body = 32 bytes
//   pos = vec4(x, y, z, mass)
//   vel = vec4(vx, vy, vz, 0)
const BytesPerBody = 32;

// UBO: 16 × f32 = 64 bytes
const UBOWords = 16;

// ── Thin-instance internal shape ───────────────────────────────────────────

type ThinInstData = {
  _gpuBuffer: GPUBuffer | null; _gpuBufferStorage: boolean;
  _gpuVersion: number;          _version: number;
  _colorGpuBuffer: GPUBuffer | null; _colorGpuBufferStorage: boolean;
  _colorGpuVersion: number;     _colorVersion: number;
};

// ── WGSL helpers shared by both shaders ────────────────────────────────────

const WGSL_HELPERS = /* wgsl */ `
struct Params {
  count       : u32,   // [0]
  s           : f32,   // [1] dt*speed*0.001
  G           : f32,   // [2]
  eps2        : f32,   // [3]
  trailHead   : u32,   // [4] write slot for this frame
  trailLen    : u32,   // [5]
  recordFrame : u32,   // [6] 1 = record trail this frame
  maxBodies   : u32,   // [7]
  headThick   : f32,   // [8]
  _p0:u32,_p1:u32,_p2:u32,_p3:u32,_p4:u32,_p5:u32,_p6:u32,
}

struct Body {
  pos : vec4<f32>,
  vel : vec4<f32>,
}

fn cbrt_(x: f32) -> f32 { return sign(x) * pow(abs(x), 0.333333333); }

// HSL → RGB using the CSS formula  k(n) = (n + 12*h) mod 12, then clamp.
// Must multiply by 12 AFTER fract so k stays in [0,12).
fn hslToRgb(h: f32, s: f32, l: f32) -> vec3<f32> {
  let a  = s * min(l, 1.0 - l);
  let k0 = fract(h          ) * 12.0;   // n=0  (R)
  let k8 = fract(h + 0.66666) * 12.0;   // n=8  (G)
  let k4 = fract(h + 0.33333) * 12.0;   // n=4  (B)
  return vec3<f32>(
    l - a * clamp(min(k0 - 3.0, 9.0 - k0), -1.0, 1.0),
    l - a * clamp(min(k8 - 3.0, 9.0 - k8), -1.0, 1.0),
    l - a * clamp(min(k4 - 3.0, 9.0 - k4), -1.0, 1.0),
  );
}

fn upToDirQuat(d: vec3<f32>) -> vec4<f32> {
  let len = length(d);
  if (len < 1e-6) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
  let f = d / len;
  if (f.y < -0.999999) { return vec4<f32>(1.0, 0.0, 0.0, 0.0); }
  let w   = 1.0 + f.y;
  let inv = 1.0 / sqrt(2.0 * w);
  return vec4<f32>(f.z * inv, 0.0, -f.x * inv, w * inv);
}
`;

// ── Pass 1: simulate (tiled all-pairs gravity + write body instances + trail history) ──

const WGSL_SIMULATE = /* wgsl */ `
${WGSL_HELPERS}

var<workgroup> shTile : array<vec4<f32>, ${Tile}>;  // shared posMass cache

@group(0) @binding(0) var<uniform>             params    : Params;
@group(0) @binding(1) var<storage, read>       front     : array<Body>;
@group(0) @binding(2) var<storage, read_write> back      : array<Body>;
@group(0) @binding(3) var<storage, read_write> bodyMat   : array<f32>;
@group(0) @binding(4) var<storage, read_write> bodyCol   : array<f32>;
@group(0) @binding(5) var<storage, read_write> trailHist : array<vec4<f32>>;

@compute @workgroup_size(${Tile})
fn main(
  @builtin(global_invocation_id) gid   : vec3<u32>,
  @builtin(local_invocation_id)  lid_v : vec3<u32>,
) {
  let i   = gid.x;
  let lid = lid_v.x;

  // Load this body's state (padding threads carry zero mass — contribute nothing).
  var posM = vec4<f32>(0.0);
  var vel  = vec4<f32>(0.0);
  if (i < params.count) {
    posM = front[i].pos;
    vel  = front[i].vel;
  }
  let pos = posM.xyz;
  var acc = vec3<f32>(0.0);

  // Tiled all-pairs gravity: stream tiles of ${Tile} bodies through shared memory.
  // Self-interaction (gj == i): dp = 0, acc contribution = 0 (safe with eps2 > 0).
  let numTiles = (params.count + ${Tile}u - 1u) / ${Tile}u;
  for (var tile = 0u; tile < numTiles; tile++) {
    let loadIdx = tile * ${Tile}u + lid;
    shTile[lid] = select(vec4<f32>(0.0), front[loadIdx].pos, loadIdx < params.count);
    workgroupBarrier();

    for (var k = 0u; k < ${Tile}u; k++) {
      let gj = tile * ${Tile}u + k;
      if (gj < params.count) {
        let dp = shTile[k].xyz - pos;
        let d2 = dot(dp, dp) + params.eps2;
        let d3 = d2 * sqrt(d2);
        acc += (params.G * shTile[k].w / d3) * dp;
      }
    }
    workgroupBarrier();
  }

  if (i >= params.count) { return; }

  let newVel = vel.xyz + acc * params.s;
  let newPos = pos + newVel * params.s;

  back[i].pos = vec4<f32>(newPos, posM.w);
  back[i].vel = vec4<f32>(newVel, 0.0);

  // Body sphere instance matrix: identity rotation, uniform scale = cbrt(mass)*0.7.
  let sc = cbrt_(posM.w) * 0.7;
  let mb = i * 16u;
  bodyMat[mb+ 0u]=sc;  bodyMat[mb+ 1u]=0.0; bodyMat[mb+ 2u]=0.0; bodyMat[mb+ 3u]=0.0;
  bodyMat[mb+ 4u]=0.0; bodyMat[mb+ 5u]=sc;  bodyMat[mb+ 6u]=0.0; bodyMat[mb+ 7u]=0.0;
  bodyMat[mb+ 8u]=0.0; bodyMat[mb+ 9u]=0.0; bodyMat[mb+10u]=sc;  bodyMat[mb+11u]=0.0;
  bodyMat[mb+12u]=newPos.x; bodyMat[mb+13u]=newPos.y; bodyMat[mb+14u]=newPos.z; bodyMat[mb+15u]=1.0;

  // Body color: procedural hue via golden ratio, distinct per body index.
  let hue = fract(f32(i) * 0.61803398875);
  let rgb = hslToRgb(hue, 0.9, 0.65);
  let cb  = i * 4u;
  bodyCol[cb+0u]=rgb.r; bodyCol[cb+1u]=rgb.g; bodyCol[cb+2u]=rgb.b; bodyCol[cb+3u]=1.0;

  // Record trail history (every other frame).
  if (params.recordFrame == 1u) {
    trailHist[params.trailHead * params.maxBodies + i] = vec4<f32>(newPos, 1.0);
  }
}
`;

// ── Pass 2: buildTrails (one thread per body × segment, tapered cylinder TRS) ──

const WGSL_BUILD_TRAILS = /* wgsl */ `
${WGSL_HELPERS}

@group(0) @binding(0) var<uniform>             params    : Params;
@group(0) @binding(1) var<storage, read>       trailHist : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> trailMat  : array<f32>;
@group(0) @binding(3) var<storage, read_write> trailCol  : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let segIdx      = gid.x;
  // Skip t=0: that segment starts exactly at the sphere position and looks like
  // a bar extending from it.  Start from t=1 so there is a clean head gap.
  let segsPerBody = params.trailLen - 2u;
  if (segIdx >= params.count * segsPerBody) { return; }

  // Decompose global segment index into (t = age, i = body).
  // t starts at 1 (not 0) — see segsPerBody above.
  let t = segIdx / params.count + 1u;
  let i = segIdx % params.count;

  // Map t to ring-buffer slots. buildHead = head AFTER recording (head+1):
  //   t=0 -> slotNew = trailHead (just written this frame)
  //   t=1 -> slotNew = trailHead - 1  ... etc.
  let buildHead = (params.trailHead + 1u) % params.trailLen;
  let slotNew   = (buildHead + params.trailLen - t - 1u) % params.trailLen;
  let slotOld   = (buildHead + params.trailLen - t - 2u) % params.trailLen;

  let entryNew = trailHist[slotNew * params.maxBodies + i];
  let entryOld = trailHist[slotOld * params.maxBodies + i];

  let mb = segIdx * 16u;
  let cb = segIdx * 4u;

  let dp  = entryOld.xyz - entryNew.xyz;
  let len = length(dp);

  // Write zero-scale matrix (invisible) for un-filled history slots or short segments.
  // w=0 means the slot was never recorded (zero-initialized ring buffer).
  if (len < 0.01 || entryNew.w < 0.5 || entryOld.w < 0.5) {
    for (var k = 0u; k < 16u; k++) { trailMat[mb + k] = 0.0; }
    trailCol[cb+0u]=0.0; trailCol[cb+1u]=0.0; trailCol[cb+2u]=0.0; trailCol[cb+3u]=0.0;
    return;
  }

  let mid   = (entryNew.xyz + entryOld.xyz) * 0.5;
  // f in [0,1]: 0 = head (t=1), 1 = tail (t=trailLen-2).
  let f = f32(t - 1u) / f32(segsPerBody - 1u);
  let thick = params.headThick * (1.0 - f);

  // Column-major TRS matrix matching AgentBuffer.writeTransform.
  let q  = upToDirQuat(dp);
  let qx = q.x; let qy = q.y; let qz = q.z; let qw = q.w;
  let x2 = qx+qx; let y2 = qy+qy; let z2 = qz+qz;
  let xx = qx*x2; let xy = qx*y2; let xz = qx*z2;
  let yy = qy*y2; let yz = qy*z2; let zz = qz*z2;
  let wx = qw*x2; let wy = qw*y2; let wz = qw*z2;
  let sx = thick; let sy = len; let sz = thick;
  trailMat[mb+ 0u]=(1.0-(yy+zz))*sx; trailMat[mb+ 1u]=(xy+wz)*sx;      trailMat[mb+ 2u]=(xz-wy)*sx;      trailMat[mb+ 3u]=0.0;
  trailMat[mb+ 4u]=(xy-wz)*sy;       trailMat[mb+ 5u]=(1.0-(xx+zz))*sy; trailMat[mb+ 6u]=(yz+wx)*sy;      trailMat[mb+ 7u]=0.0;
  trailMat[mb+ 8u]=(xz+wy)*sz;       trailMat[mb+ 9u]=(yz-wx)*sz;       trailMat[mb+10u]=(1.0-(xx+yy))*sz; trailMat[mb+11u]=0.0;
  trailMat[mb+12u]=mid.x;            trailMat[mb+13u]=mid.y;            trailMat[mb+14u]=mid.z;            trailMat[mb+15u]=1.0;

  // Trail color: same hue as body, fades to dark toward the tail.
  let hue = fract(f32(i) * 0.61803398875);
  // Fade the oldest ~25% to alpha=0 so periodic orbits don't create a visible
  // loop when the tail tip comes back near the sphere.
  let alpha = max(0.0, 1.0 - f * 1.3);
  let rgb   = hslToRgb(hue, 0.9, 0.65 * (1.0 - f * 0.85));
  trailCol[cb+0u]=rgb.r; trailCol[cb+1u]=rgb.g; trailCol[cb+2u]=rgb.b; trailCol[cb+3u]=alpha;
}
`;

// ── NBodyGpu params ─────────────────────────────────────────────────────────

export interface NBodyGpuParams {
  bodies:     number;
  G:          number;
  softening:  number;
  speed:      number;
  showTrails: boolean;
}

// ── NBodyGpu class ──────────────────────────────────────────────────────────

export class NBodyGpu {
  readonly bodyMesh:  Mesh;
  readonly trailMesh: Mesh;

  private readonly engine: EngineContext;
  private readonly device: GPUDevice;
  private readonly tiBody:  ThinInstData;
  private readonly tiTrail: ThinInstData;

  // Ping-pong body state
  private stateA: GPUBuffer;
  private stateB: GPUBuffer;
  private front:  GPUBuffer;
  private back:   GPUBuffer;

  // Owned instance vertex buffers
  private readonly bodyMat:  GPUBuffer;  // GpuMaxBodies × 64
  private readonly bodyCol:  GPUBuffer;  // GpuMaxBodies × 16
  private readonly trailMat: GPUBuffer;  // GpuMaxBodies × (GpuTrailLen-1) × 64
  private readonly trailCol: GPUBuffer;  // × 16

  // Trail history ring buffer
  private readonly trailHist: GPUBuffer; // GpuMaxBodies × GpuTrailLen × 16

  // UBO
  private readonly ubo:     GPUBuffer;
  private readonly uboData: Float32Array;

  // Compute
  private readonly simBGL:    GPUBindGroupLayout;
  private readonly trailBGL:  GPUBindGroupLayout;
  private readonly simPipeline:    GPUComputePipeline;
  private readonly trailsPipeline: GPUComputePipeline;
  private readonly bgSimA:   GPUBindGroup;
  private readonly bgSimB:   GPUBindGroup;
  private readonly bgTrails: GPUBindGroup;

  // CPU-side trail state
  private trailHead   = 0;
  private frameCount  = 0;
  private _trailCount = 0;

  private _ok = false;
  get ok(): boolean { return this._ok; }

  constructor(engine: EngineContext, scene: SceneContext) {
    this.engine = engine;
    const dev   = (engine as unknown as { _device: GPUDevice })._device;
    this.device = dev;

    const maxSegs = GpuMaxBodies * (GpuTrailLen - 1);
    const instUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST;

    // Owned instance buffers
    this.bodyMat  = dev.createBuffer({ label: "nbody-bodyMat",  size: GpuMaxBodies * 64,  usage: instUsage });
    this.bodyCol  = dev.createBuffer({ label: "nbody-bodyCol",  size: GpuMaxBodies * 16,  usage: instUsage });
    this.trailMat = dev.createBuffer({ label: "nbody-trailMat", size: maxSegs * 64,        usage: instUsage });
    this.trailCol = dev.createBuffer({ label: "nbody-trailCol", size: maxSegs * 16,        usage: instUsage });

    // Trail history (positions only, not instanced)
    this.trailHist = dev.createBuffer({
      label: "nbody-trailHist",
      size: GpuMaxBodies * GpuTrailLen * 16,  // vec4<f32> per slot per body
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // ── Body sphere mesh ──────────────────────────────────────────────────
    // Order matches the working boids GPU mesh: addToScene FIRST, then
    // setThinInstances / setThinInstanceCount(0) (all still before registerScene).
    const sphere = createSphere(engine, { segments: 8, diameter: 1 });
    sphere.material = createPbrMaterial({ baseColorFactor: [1, 0.85, 0.3, 1], metallicFactor: 0.6, roughnessFactor: 0.3 });
    addToScene(scene, sphere);
    this.bodyMesh = sphere;

    const identBodies = new Float32Array(GpuMaxBodies * 16);
    for (let i = 0; i < GpuMaxBodies; i++) {
      const b = i * 16; identBodies[b] = 1; identBodies[b+5] = 1; identBodies[b+10] = 1; identBodies[b+15] = 1;
    }
    setThinInstances(sphere, identBodies, GpuMaxBodies);
    setThinInstanceColors(sphere, new Float32Array(GpuMaxBodies * 4).fill(1));
    setThinInstanceCount(sphere, 0);

    const tiB = (sphere as unknown as { thinInstances: ThinInstData }).thinInstances;
    tiB._gpuBuffer = this.bodyMat; tiB._gpuBufferStorage = false; tiB._gpuVersion = tiB._version;
    tiB._colorGpuBuffer = this.bodyCol; tiB._colorGpuBufferStorage = false; tiB._colorGpuVersion = tiB._colorVersion;
    this.tiBody = tiB;

    // ── Trail cylinder mesh ───────────────────────────────────────────────
    const cyl = createCylinder(engine, { height: 1, diameter: 1, tessellation: 6 });
    cyl.material = createPbrMaterial({ unlit: true, baseColorFactor: [1, 1, 1, 1], needAlphaBlending: true });
    addToScene(scene, cyl);
    this.trailMesh = cyl;

    const identTrails = new Float32Array(maxSegs * 16);
    for (let i = 0; i < maxSegs; i++) {
      const b = i * 16; identTrails[b] = 1; identTrails[b+5] = 1; identTrails[b+10] = 1; identTrails[b+15] = 1;
    }
    setThinInstances(cyl, identTrails, maxSegs);
    setThinInstanceColors(cyl, new Float32Array(maxSegs * 4).fill(1));
    setThinInstanceCount(cyl, 0);

    const tiT = (cyl as unknown as { thinInstances: ThinInstData }).thinInstances;
    tiT._gpuBuffer = this.trailMat; tiT._gpuBufferStorage = false; tiT._gpuVersion = tiT._version;
    tiT._colorGpuBuffer = this.trailCol; tiT._colorGpuBufferStorage = false; tiT._colorGpuVersion = tiT._colorVersion;
    this.tiTrail = tiT;

    // ── State ping-pong ───────────────────────────────────────────────────
    const stUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.stateA = dev.createBuffer({ label: "nbody-stateA", size: GpuMaxBodies * BytesPerBody, usage: stUsage });
    this.stateB = dev.createBuffer({ label: "nbody-stateB", size: GpuMaxBodies * BytesPerBody, usage: stUsage });
    this.front = this.stateA;
    this.back  = this.stateB;

    // ── UBO ───────────────────────────────────────────────────────────────
    this.uboData = new Float32Array(UBOWords);
    this.ubo = dev.createBuffer({ label: "nbody-ubo", size: UBOWords * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // ── Compute pipelines ─────────────────────────────────────────────────
    try {
      this.simBGL   = this.buildSimBGL();
      this.trailBGL = this.buildTrailBGL();
      this.simPipeline    = this.buildPipeline("nbody-sim",    WGSL_SIMULATE,     this.simBGL);
      this.trailsPipeline = this.buildPipeline("nbody-trails", WGSL_BUILD_TRAILS, this.trailBGL);
      this.bgSimA   = this.buildSimBG("A", this.stateA, this.stateB);
      this.bgSimB   = this.buildSimBG("B", this.stateB, this.stateA);
      this.bgTrails = this.buildTrailBG();
    } catch (e) {
      console.error("[NBodyGpu] pipeline build failed:", e);
      this.simBGL = null!; this.trailBGL = null!;
      this.simPipeline = null!; this.trailsPipeline = null!;
      this.bgSimA = null!; this.bgSimB = null!; this.bgTrails = null!;
      return;
    }

    this._ok = true;
  }

  private buildSimBGL(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "nbody-sim-bgl",
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

  private buildTrailBGL(): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      label: "nbody-trail-bgl",
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

  private buildSimBG(label: string, front: GPUBuffer, back: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({
      label: `nbody-sim-bg-${label}`, layout: this.simBGL,
      entries: [
        { binding: 0, resource: { buffer: this.ubo } },
        { binding: 1, resource: { buffer: front } },
        { binding: 2, resource: { buffer: back } },
        { binding: 3, resource: { buffer: this.bodyMat } },
        { binding: 4, resource: { buffer: this.bodyCol } },
        { binding: 5, resource: { buffer: this.trailHist } },
      ],
    });
  }

  private buildTrailBG(): GPUBindGroup {
    return this.device.createBindGroup({
      label: "nbody-trail-bg", layout: this.trailBGL,
      entries: [
        { binding: 0, resource: { buffer: this.ubo } },
        { binding: 1, resource: { buffer: this.trailHist } },
        { binding: 2, resource: { buffer: this.trailMat } },
        { binding: 3, resource: { buffer: this.trailCol } },
      ],
    });
  }

  /** Update drawn body count and resync version guard. */
  setBodyCount(n: number): void {
    setThinInstanceCount(this.bodyMesh, n);
    this.tiBody._gpuVersion      = this.tiBody._version;
    this.tiBody._colorGpuVersion = this.tiBody._colorVersion;
    invalidateRenderBundles(this.engine);
  }

  /** Update drawn trail segment count (= n × (GpuTrailLen-2)) and resync. */
  setTrailCount(n: number): void {
    this._trailCount = n;
    setThinInstanceCount(this.trailMesh, n * (GpuTrailLen - 2));
    this.tiTrail._gpuVersion      = this.tiTrail._version;
    this.tiTrail._colorGpuVersion = this.tiTrail._colorVersion;
    invalidateRenderBundles(this.engine);
  }

  /** Seed from a random configuration, center-of-mass corrected (f32). */
  seedRandom(n: number): void {
    const data = new Float32Array(n * 8);
    let cx = 0, cy = 0, cz = 0, cvx = 0, cvy = 0, cvz = 0, totM = 0;
    for (let i = 0; i < n; i++) {
      const b    = i * 8;
      const mass = 0.5 + Math.random() * 4;
      const px   = (Math.random() - 0.5) * 30;
      const py   = (Math.random() - 0.5) * 30;
      const pz   = (Math.random() - 0.5) * 30;
      const vx   = (Math.random() - 0.5) * 4;
      const vy   = (Math.random() - 0.5) * 4;
      const vz   = (Math.random() - 0.5) * 4;
      data[b]=px; data[b+1]=py; data[b+2]=pz; data[b+3]=mass;
      data[b+4]=vx; data[b+5]=vy; data[b+6]=vz; data[b+7]=0;
      cx+=px*mass; cy+=py*mass; cz+=pz*mass;
      cvx+=vx*mass; cvy+=vy*mass; cvz+=vz*mass; totM+=mass;
    }
    const invM = 1 / totM;
    for (let i = 0; i < n; i++) {
      const b = i * 8;
      data[b]  -=cx*invM; data[b+1]-=cy*invM; data[b+2]-=cz*invM;
      data[b+4]-=cvx*invM; data[b+5]-=cvy*invM; data[b+6]-=cvz*invM;
    }
    this.device.queue.writeBuffer(this.front, 0, data.buffer, 0, n * 8 * 4);
    this._clearTrailHist();
    this.trailHead  = 0;
    this.frameCount = 0;
  }

  /** Seed the binary star preset (always 2 bodies). */
  seedBinary(): void {
    const data = new Float32Array(2 * 8);
    data[ 0]=-8; data[ 1]=0; data[ 2]=0; data[ 3]=10; // pos + mass body 0
    data[ 4]=0;  data[ 5]=4; data[ 6]=0; data[ 7]=0;  // vel body 0
    data[ 8]= 8; data[ 9]=0; data[10]=0; data[11]=10; // pos + mass body 1
    data[12]=0;  data[13]=-4;data[14]=0; data[15]=0;  // vel body 1
    this.device.queue.writeBuffer(this.front, 0, data.buffer, 0, 2 * 8 * 4);
    this._clearTrailHist();
    this.trailHead  = 0;
    this.frameCount = 0;
  }

  private _clearTrailHist(): void {
    this.device.queue.writeBuffer(
      this.trailHist, 0,
      new Float32Array(GpuMaxBodies * GpuTrailLen * 4),
    );
  }

  /** Dispatch one frame of physics + optional trail building. dt in milliseconds. */
  dispatch(dt: number, params: NBodyGpuParams): void {
    if (!this._ok) return;
    const n  = params.bodies;
    const s  = (dt * 0.001) * params.speed;

    this.frameCount++;
    const recordFrame = params.showTrails && this.frameCount % 2 === 0 ? 1 : 0;

    // Sync trail mesh instance count with the showTrails toggle.
    if (!params.showTrails && this._trailCount > 0) {
      this.setTrailCount(0);   // toggled off — hide immediately
    } else if (params.showTrails && this._trailCount !== n) {
      this.setTrailCount(n);   // toggled on (or body count changed) — re-show
    }

    // Write UBO
    const u    = this.uboData;
    const uU32 = new Uint32Array(u.buffer);
    uU32[0] = n;
    u[1]    = s;
    u[2]    = params.G;
    u[3]    = params.softening * params.softening;  // eps2
    uU32[4] = this.trailHead;
    uU32[5] = GpuTrailLen;
    uU32[6] = recordFrame;
    uU32[7] = GpuMaxBodies;
    u[8]    = HeadThick;
    this.device.queue.writeBuffer(this.ubo, 0, u);

    const bg      = this.front === this.stateA ? this.bgSimA : this.bgSimB;
    const wgSim   = Math.ceil(n / Tile);
    const wgTrail = params.showTrails ? Math.ceil(n * (GpuTrailLen - 2) / 64) : 0;

    const enc  = this.device.createCommandEncoder({ label: "nbody-compute" });
    const pass = enc.beginComputePass({ label: "nbody" });

    pass.setPipeline(this.simPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgSim);

    if (wgTrail > 0) {
      pass.setPipeline(this.trailsPipeline);
      pass.setBindGroup(0, this.bgTrails);
      pass.dispatchWorkgroups(wgTrail);
    }

    pass.end();
    this.device.queue.submit([enc.finish()]);

    // Ping-pong state buffers
    const tmp  = this.front;
    this.front = this.back;
    this.back  = tmp;

    // Advance trail head after recording
    if (recordFrame) {
      this.trailHead = (this.trailHead + 1) % GpuTrailLen;
    }
  }

  destroy(): void {
    this.stateA.destroy(); this.stateB.destroy();
    this.bodyMat.destroy(); this.bodyCol.destroy();
    this.trailMat.destroy(); this.trailCol.destroy();
    this.trailHist.destroy(); this.ubo.destroy();
  }
}
