import { upToDirQuat } from "@/lib/agents";
import type { AgentBuffer } from "@/lib/agents";

const Bound = 20;

export interface CpuBoidsParams {
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

export class CpuBoids {
  readonly capacity: number;
  readonly px: Float32Array;
  readonly py: Float32Array;
  readonly pz: Float32Array;
  readonly vx: Float32Array;
  readonly vy: Float32Array;
  readonly vz: Float32Array;

  private readonly sx2: Float32Array;
  private readonly sy2: Float32Array;
  private readonly sz2: Float32Array;
  private readonly ax: Float32Array;
  private readonly ay: Float32Array;
  private readonly az: Float32Array;
  private readonly cx2: Float32Array;
  private readonly cy2: Float32Array;
  private readonly cz2: Float32Array;
  private readonly nc: Int32Array;
  private readonly ns: Int32Array;
  // Positional collision-pushout accumulators
  private readonly cpx: Float32Array;
  private readonly cpy: Float32Array;
  private readonly cpz: Float32Array;

  constructor(capacity: number) {
    this.capacity = capacity;
    this.px = new Float32Array(capacity);
    this.py = new Float32Array(capacity);
    this.pz = new Float32Array(capacity);
    this.vx = new Float32Array(capacity);
    this.vy = new Float32Array(capacity);
    this.vz = new Float32Array(capacity);
    this.sx2 = new Float32Array(capacity);
    this.sy2 = new Float32Array(capacity);
    this.sz2 = new Float32Array(capacity);
    this.ax = new Float32Array(capacity);
    this.ay = new Float32Array(capacity);
    this.az = new Float32Array(capacity);
    this.cx2 = new Float32Array(capacity);
    this.cy2 = new Float32Array(capacity);
    this.cz2 = new Float32Array(capacity);
    this.nc = new Int32Array(capacity);
    this.ns = new Int32Array(capacity);
    this.cpx = new Float32Array(capacity);
    this.cpy = new Float32Array(capacity);
    this.cpz = new Float32Array(capacity);
  }

  spawnOne(i: number): void {
    this.px[i] = (Math.random() - 0.5) * Bound * 1.5;
    this.py[i] = (Math.random() - 0.5) * Bound;
    this.pz[i] = (Math.random() - 0.5) * Bound * 1.5;
    const angle = Math.random() * Math.PI * 2;
    const pitch = (Math.random() - 0.5) * Math.PI;
    this.vx[i] = Math.cos(angle) * Math.cos(pitch);
    this.vy[i] = Math.sin(pitch);
    this.vz[i] = Math.sin(angle) * Math.cos(pitch);
  }

  spawnAll(): void {
    for (let i = 0; i < this.capacity; i++) this.spawnOne(i);
  }

  update(dt: number, params: CpuBoidsParams, buf: AgentBuffer, mesh: import("@babylonjs/lite").Mesh): void {
    const n = params.count;
    const s = dt * 0.001;
    const spd = params.speed;
    const sepW = params.separation;
    const aliW = params.alignment;
    const cohW = params.cohesion;
    const r = params.radius;
    const r2 = r * r;
    const sr = params.separationRadius;
    const sr2 = sr * sr;
    const doCollision = params.collision;
    const collDist  = 2 * params.collisionRadius;
    const collDist2 = collDist * collDist;
    const collStr   = params.collisionStrength;

    for (let i = 0; i < n; i++) {
      this.sx2[i] = this.sy2[i] = this.sz2[i] = 0;
      this.ax[i] = this.ay[i] = this.az[i] = 0;
      this.cx2[i] = this.cy2[i] = this.cz2[i] = 0;
      this.nc[i] = this.ns[i] = 0;
      this.cpx[i] = this.cpy[i] = this.cpz[i] = 0;
    }

    for (let i = 0; i < n - 1; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = this.px[j] - this.px[i];
        const dy = this.py[j] - this.py[i];
        const dz = this.pz[j] - this.pz[i];
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < r2) {
          this.ax[i] += this.vx[j]; this.ay[i] += this.vy[j]; this.az[i] += this.vz[j];
          this.ax[j] += this.vx[i]; this.ay[j] += this.vy[i]; this.az[j] += this.vz[i];
          this.cx2[i] += this.px[j]; this.cy2[i] += this.py[j]; this.cz2[i] += this.pz[j];
          this.cx2[j] += this.px[i]; this.cy2[j] += this.py[i]; this.cz2[j] += this.pz[i];
          this.nc[i]++; this.nc[j]++;
          if (d2 < sr2) {
            this.sx2[i] -= dx; this.sy2[i] -= dy; this.sz2[i] -= dz;
            this.sx2[j] += dx; this.sy2[j] += dy; this.sz2[j] += dz;
            this.ns[i]++; this.ns[j]++;
          }
        }
        // Soft collision: direct positional pushout (independent of steering).
        // collDist is small (~0.8) and always < r (~5), so no extra loop needed.
        if (doCollision && d2 < collDist2 && d2 > 1e-8) {
          const d    = Math.sqrt(d2);
          const push = (collDist - d) * 0.5 * collStr / d;
          this.cpx[i] -= dx * push; this.cpy[i] -= dy * push; this.cpz[i] -= dz * push;
          this.cpx[j] += dx * push; this.cpy[j] += dy * push; this.cpz[j] += dz * push;
        }
      }
    }

    for (let i = 0; i < n; i++) {
      let fx = 0, fy = 0, fz = 0;

      if (this.ns[i] > 0) { fx += this.sx2[i] * sepW; fy += this.sy2[i] * sepW; fz += this.sz2[i] * sepW; }

      if (this.nc[i] > 0) {
        const inv = 1 / this.nc[i];
        fx += (this.ax[i] * inv - this.vx[i]) * aliW;
        fy += (this.ay[i] * inv - this.vy[i]) * aliW;
        fz += (this.az[i] * inv - this.vz[i]) * aliW;
      }

      if (this.nc[i] > 0) {
        const inv = 1 / this.nc[i];
        fx += (this.cx2[i] * inv - this.px[i]) * cohW * 0.05;
        fy += (this.cy2[i] * inv - this.py[i]) * cohW * 0.05;
        fz += (this.cz2[i] * inv - this.pz[i]) * cohW * 0.05;
      }

      const margin = Bound * 0.15;
      const bx = Math.abs(this.px[i]) - (Bound - margin);
      const by = Math.abs(this.py[i]) - (Bound * 0.5 - margin);
      const bz = Math.abs(this.pz[i]) - (Bound - margin);
      if (bx > 0) fx -= Math.sign(this.px[i]) * bx * 0.5;
      if (by > 0) fy -= Math.sign(this.py[i]) * by * 0.5;
      if (bz > 0) fz -= Math.sign(this.pz[i]) * bz * 0.5;

      this.vx[i] += fx * s; this.vy[i] += fy * s; this.vz[i] += fz * s;

      const vLen = Math.sqrt(this.vx[i] * this.vx[i] + this.vy[i] * this.vy[i] + this.vz[i] * this.vz[i]);
      if (vLen > 0.001) {
        const scale = spd / vLen;
        this.vx[i] *= scale; this.vy[i] *= scale; this.vz[i] *= scale;
      }

      this.px[i] += this.vx[i] * s;
      this.py[i] += this.vy[i] * s;
      this.pz[i] += this.vz[i] * s;

      // Apply positional collision pushout after normal integration.
      if (doCollision) {
        this.px[i] += this.cpx[i];
        this.py[i] += this.cpy[i];
        this.pz[i] += this.cpz[i];
      }

      const [qx, qy, qz, qw] = upToDirQuat(this.vx[i], this.vy[i], this.vz[i]);
      buf.writeTransform(i, this.px[i], this.py[i], this.pz[i], qx, qy, qz, qw, 1, 1, 1);
      const t = Math.min(vLen / (spd * 1.5), 1);
      buf.writeColor(i, t, 1 - t * 0.8, 1 - t * 0.4);
    }

    buf.commit(mesh, n);
  }
}
