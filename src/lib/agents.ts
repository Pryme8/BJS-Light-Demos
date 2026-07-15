import {
  setThinInstances,
  setThinInstanceColors,
  setThinInstanceCount,
  invalidateRenderBundles,
} from "@babylonjs/lite";
import type { EngineContext, Mesh } from "@babylonjs/lite";

/**
 * Zero-allocation per-frame thin-instance buffer.
 *
 * BJS Lite requires the mesh to be established as thin-instanced BEFORE
 * `registerScene()` (that is what wires the instanced render pipeline). After
 * that, per-frame updates must reuse the SAME matrix/color arrays and only bump
 * the active count / mark dirty — never reallocate — so the cached render bundle
 * stays valid.
 *
 * Usage:
 *   const buf = new AgentBuffer(capacity);
 *   buf.attach(mesh);            // in buildScene, BEFORE registerScene
 *   ...
 *   buf.writeScale(i, x, y, z);  // each frame
 *   buf.commit(mesh, count);     // each frame
 */
export class AgentBuffer {
  readonly matrices: Float32Array;
  readonly colors: Float32Array;
  private readonly capacity: number;
  private readonly attached = new WeakSet<Mesh>();
  private readonly lastCount = new WeakMap<Mesh, number>();
  private engine: EngineContext | null = null;

  constructor(capacity: number) {
    this.capacity = capacity;
    this.matrices = new Float32Array(capacity * 16);
    this.colors = new Float32Array(capacity * 4);
    // Default: identity matrices
    for (let i = 0; i < capacity; i++) {
      this.writeScale(i, 1, 1, 1);
    }
    // Default: white opaque
    for (let i = 0; i < capacity * 4; i += 4) {
      this.colors[i] = 1; this.colors[i + 1] = 1; this.colors[i + 2] = 1; this.colors[i + 3] = 1;
    }
  }

  /** Write a TRS matrix (translation + uniform scale, no rotation). */
  writeScale(i: number, tx: number, ty: number, tz: number, s = 1): void {
    const b = i * 16;
    const m = this.matrices;
    m[b]    = s; m[b+1]  = 0; m[b+2]  = 0; m[b+3]  = 0;
    m[b+4]  = 0; m[b+5]  = s; m[b+6]  = 0; m[b+7]  = 0;
    m[b+8]  = 0; m[b+9]  = 0; m[b+10] = s; m[b+11] = 0;
    m[b+12] = tx; m[b+13] = ty; m[b+14] = tz; m[b+15] = 1;
  }

  /** Write a TRS matrix with non-uniform scale. */
  writeTRS(i: number, tx: number, ty: number, tz: number, sx: number, sy: number, sz: number): void {
    const b = i * 16;
    const m = this.matrices;
    m[b]    = sx; m[b+1]  = 0;  m[b+2]  = 0;  m[b+3]  = 0;
    m[b+4]  = 0;  m[b+5]  = sy; m[b+6]  = 0;  m[b+7]  = 0;
    m[b+8]  = 0;  m[b+9]  = 0;  m[b+10] = sz; m[b+11] = 0;
    m[b+12] = tx; m[b+13] = ty; m[b+14] = tz; m[b+15] = 1;
  }

  /**
   * Write a full TRS matrix (translation + quaternion rotation + scale).
   * Zero-alloc — computes the quaternion→matrix inline.
   */
  writeTransform(
    i: number,
    tx: number, ty: number, tz: number,
    qx: number, qy: number, qz: number, qw: number,
    sx: number, sy: number, sz: number
  ): void {
    const b = i * 16;
    const m = this.matrices;
    const x2 = qx + qx, y2 = qy + qy, z2 = qz + qz;
    const xx = qx * x2, xy = qx * y2, xz = qx * z2;
    const yy = qy * y2, yz = qy * z2, zz = qz * z2;
    const wx = qw * x2, wy = qw * y2, wz = qw * z2;
    m[b]    = (1 - (yy + zz)) * sx;
    m[b+1]  = (xy + wz) * sx;
    m[b+2]  = (xz - wy) * sx;
    m[b+3]  = 0;
    m[b+4]  = (xy - wz) * sy;
    m[b+5]  = (1 - (xx + zz)) * sy;
    m[b+6]  = (yz + wx) * sy;
    m[b+7]  = 0;
    m[b+8]  = (xz + wy) * sz;
    m[b+9]  = (yz - wx) * sz;
    m[b+10] = (1 - (xx + yy)) * sz;
    m[b+11] = 0;
    m[b+12] = tx; m[b+13] = ty; m[b+14] = tz; m[b+15] = 1;
  }

  /** Write RGBA color for instance i (0–1 range). */
  writeColor(i: number, r: number, g: number, b: number, a = 1): void {
    const o = i * 4;
    this.colors[o] = r; this.colors[o+1] = g; this.colors[o+2] = b; this.colors[o+3] = a;
  }

  /**
   * Establish the mesh as thin-instanced at full capacity. MUST be called in
   * buildScene BEFORE registerScene(). Starts with an active count of 0 so
   * nothing draws until the first commit().
   *
   * The engine is stored so {@link commit} can invalidate the cached render
   * bundle when the active instance count changes (the bundle bakes the draw's
   * instance count, so a count change alone would otherwise not be drawn).
   */
  attach(engine: EngineContext, mesh: Mesh, withColors = true): void {
    if (this.attached.has(mesh)) return;
    this.engine = engine;
    setThinInstances(mesh, this.matrices, this.capacity);
    if (withColors) setThinInstanceColors(mesh, this.colors);
    setThinInstanceCount(mesh, 0);
    this.lastCount.set(mesh, 0);
    this.attached.add(mesh);
  }

  /**
   * Per-frame update: set the active instance count (which re-uploads the
   * [0,count) matrix range) and, when colors change, re-push the color array via
   * setThinInstanceColors (which bumps the color version so the color buffer is
   * actually re-uploaded — flushThinInstances only marks matrices dirty). Never
   * reallocates the GPU buffer. When the active count changes, invalidate the
   * cached render bundle so the draw re-records with the new instance count.
   */
  commit(mesh: Mesh, count: number, withColors = true): void {
    const n = Math.min(count, this.capacity);

    // Fallback: if a demo commits before attaching (e.g. a one-shot build that
    // runs inside buildScene, pre-register), establish the pool here.
    if (!this.attached.has(mesh)) {
      setThinInstances(mesh, this.matrices, this.capacity);
      this.attached.add(mesh);
      setThinInstanceCount(mesh, n);
      if (withColors) setThinInstanceColors(mesh, this.colors);
      this.lastCount.set(mesh, n);
      return;
    }

    setThinInstanceCount(mesh, n);
    if (withColors) setThinInstanceColors(mesh, this.colors);

    if (this.lastCount.get(mesh) !== n) {
      this.lastCount.set(mesh, n);
      if (this.engine) invalidateRenderBundles(this.engine);
    }
  }
}

/**
 * Unit quaternion that rotates the mesh's local +Y (up) axis onto the travel
 * direction (dx, dy, dz). Use for capsule / cylinder agents whose long axis is
 * +Y so they point the way they are moving.
 *
 * Returns a UNIT quaternion — required because {@link AgentBuffer.writeTransform}
 * assumes |q| = 1; a non-unit quaternion would bake scale/shear into the matrix.
 */
export function upToDirQuat(
  dx: number, dy: number, dz: number
): [number, number, number, number] {
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (len < 1e-6) return [0, 0, 0, 1];
  const fx = dx / len, fy = dy / len, fz = dz / len;
  // Shortest-arc rotation from +Y (0,1,0) to (fx,fy,fz).
  if (fy < -0.999999) return [1, 0, 0, 0]; // pointing straight down: 180° flip
  // cross(+Y, dir) = (fz, 0, -fx); w = 1 + dot(+Y, dir) = 1 + fy
  const w = 1 + fy;
  const inv = 1 / Math.sqrt(2 * w); // = 1 / |(fz, 0, -fx, w)|
  return [fz * inv, 0, -fx * inv, w * inv];
}
