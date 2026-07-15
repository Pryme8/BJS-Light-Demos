import { reactive, ref } from "vue";
import {
  addToScene,
  attachControl,
  createArcRotateCamera,
  createHemisphericLight,
  createDirectionalLight,
  createCylinder,
  createPbrMaterial,
} from "@babylonjs/lite";
import type { EngineContext, SceneContext } from "@babylonjs/lite";
import { AgentBuffer, upToDirQuat } from "@/lib/agents";
import type { SimHandle } from "@/types/sim";

const MaxSegments = 8000;

const Rules: Record<string, { axiom: string; rules: Record<string, string> }> = {
  "Plant A":  { axiom: "X", rules: { X: "F+[[X]-X]-F[-FX]+X", F: "FF" } },
  "Fractal Tree": { axiom: "F", rules: { F: "FF+[+F-F-F]-[-F+F+F]" } },
  "Sierpinski": { axiom: "F-G-G", rules: { F: "F-G+F+G-F", G: "GG" } },
  "Dragon":   { axiom: "FX", rules: { X: "X+YF+", Y: "-FX-Y" } },
};

// Each L-system only renders correctly at its characteristic turn angle (and a
// sensible iteration depth). Applied automatically when the system is selected.
const SystemDefaults: Record<string, { angle: number; iterations: number }> = {
  "Plant A":      { angle: 25,  iterations: 4 },
  "Fractal Tree": { angle: 25,  iterations: 4 },
  "Sierpinski":   { angle: 120, iterations: 5 },
  "Dragon":       { angle: 90,  iterations: 11 },
};

interface TurtleState {
  x: number; y: number; z: number;
  dx: number; dy: number; dz: number;  // heading
  ux: number; uy: number; uz: number;  // up
  len: number; width: number;
}

function rotateAround(vx: number, vy: number, vz: number, ax: number, ay: number, az: number, angle: number): [number, number, number] {
  const c = Math.cos(angle), s = Math.sin(angle);
  const t = 1 - c;
  return [
    (t * ax * ax + c) * vx + (t * ax * ay - s * az) * vy + (t * ax * az + s * ay) * vz,
    (t * ax * ay + s * az) * vx + (t * ay * ay + c) * vy + (t * ay * az - s * ax) * vz,
    (t * ax * az - s * ay) * vx + (t * ay * az + s * ax) * vy + (t * az * az + c) * vz,
  ];
}

export function buildSimScene(
  engine: EngineContext,
  scene: SceneContext,
  canvas: HTMLCanvasElement
): SimHandle {
  // The plant grows in the XY plane (up vector +Z), so view it front-on from -Z.
  const camera = createArcRotateCamera(-Math.PI / 2, Math.PI * 0.5, 30, { x: 0, y: 2, z: 0 });
  scene.camera = camera;
  const detach = attachControl(camera, canvas, scene);

  addToScene(scene, createHemisphericLight([0, 1, 0], 0.6));
  addToScene(scene, createDirectionalLight([-0.5, -1, 0.3]));

  const segMesh = createCylinder(engine, { height: 1, diameter: 1, tessellation: 6 });
  segMesh.material = createPbrMaterial({ baseColorFactor: [0.25, 0.65, 0.15, 1], metallicFactor: 0, roughnessFactor: 0.8 });
  addToScene(scene, segMesh);

  const params = reactive({
    system: "Plant A",
    iterations: 4,
    angle: 25,
    length: 1.2,
    shrink: 0.7,
    width: 0.18,
  });

  const segCountRef = ref(0);
  const buf = new AgentBuffer(MaxSegments);
  buf.attach(engine, segMesh);

  let currentString = "";
  let segCount = 0;

  function expand(axiom: string, rules: Record<string, string>, iters: number): string {
    let s = axiom;
    for (let i = 0; i < iters; i++) {
      let ns = "";
      for (const c of s) ns += rules[c] ?? c;
      s = ns;
      if (s.length > 60000) break;
    }
    return s;
  }

  function buildTree() {
    const ruleSet = Rules[params.system as string] ?? Rules["Plant A"];
    currentString = expand(ruleSet.axiom, ruleSet.rules, params.iterations);

    const stack: TurtleState[] = [];
    const state: TurtleState = {
      x: 0, y: -8, z: 0,
      dx: 0, dy: 1, dz: 0,
      ux: 0, uy: 0, uz: 1,   // rotate branches around +Z → plant lies in the XY plane
      len: params.length,
      width: params.width,
    };

    const angleRad = params.angle * (Math.PI / 180);
    segCount = 0;

    // Track bounds (XY) to auto-frame the camera afterwards.
    let minX = state.x, maxX = state.x, minY = state.y, maxY = state.y;

    for (const ch of currentString) {
      if (segCount >= MaxSegments) break;
      if (ch === "F" || ch === "G") {
        const nx = state.x + state.dx * state.len;
        const ny = state.y + state.dy * state.len;
        const nz = state.z + state.dz * state.len;

        // Cylinder center and orientation: rotate the cylinder's +Y axis onto the
        // turtle heading (unit quaternion — see upToDirQuat).
        const cx = (state.x + nx) * 0.5, cy = (state.y + ny) * 0.5, cz = (state.z + nz) * 0.5;
        const [qx, qy, qz, qw] = upToDirQuat(state.dx, state.dy, state.dz);

        const depth = stack.length;
        const t = Math.min(depth * 0.15, 1);
        buf.writeTransform(segCount, cx, cy, cz, qx, qy, qz, qw, state.width, state.len, state.width);
        buf.writeColor(segCount, 0.15 + t * 0.15, 0.55 + t * 0.2, 0.05 + t * 0.4);
        segCount++;

        state.x = nx; state.y = ny; state.z = nz;
        if (nx < minX) minX = nx; else if (nx > maxX) maxX = nx;
        if (ny < minY) minY = ny; else if (ny > maxY) maxY = ny;
      } else if (ch === "+") {
        [state.dx, state.dy, state.dz] = rotateAround(state.dx, state.dy, state.dz, state.ux, state.uy, state.uz, angleRad);
      } else if (ch === "-") {
        [state.dx, state.dy, state.dz] = rotateAround(state.dx, state.dy, state.dz, state.ux, state.uy, state.uz, -angleRad);
      } else if (ch === "[") {
        stack.push({ ...state });
        state.len *= params.shrink;
        state.width *= params.shrink;
      } else if (ch === "]") {
        const s2 = stack.pop();
        if (s2) {
          state.x = s2.x; state.y = s2.y; state.z = s2.z;
          state.dx = s2.dx; state.dy = s2.dy; state.dz = s2.dz;
          state.ux = s2.ux; state.uy = s2.uy; state.uz = s2.uz;
          state.len = s2.len; state.width = s2.width;
        }
      }
    }
    segCountRef.value = segCount;
    buf.commit(segMesh, segCount);

    // Auto-frame: center the camera target on the figure and set the radius so
    // it fits, regardless of system / iterations / length.
    const centerX = (minX + maxX) * 0.5;
    const centerY = (minY + maxY) * 0.5;
    const spanX = maxX - minX;
    const spanY = maxY - minY;
    const halfExtent = Math.max(spanX, spanY, 1) * 0.5;
    camera.target = { x: centerX, y: centerY, z: 0 };
    camera.radius = (halfExtent / Math.tan(camera.fov * 0.5)) * 1.35 + 4;
  }

  buildTree();

  let dirty = false;
  function markDirty() { dirty = true; }

  function update(_dt: number) {
    if (dirty) { buildTree(); dirty = false; }
  }

  function reset() { buildTree(); }

  // Watch param changes to rebuild. Selecting a system also snaps the angle and
  // iterations to that system's correct defaults.
  const proxy = new Proxy(params, {
    set(target, key, value) {
      (target as Record<string, unknown>)[key as string] = value;
      if (key === "system") {
        const d = SystemDefaults[value as string];
        if (d) { target.angle = d.angle; target.iterations = d.iterations; }
      }
      markDirty();
      return true;
    }
  });

  return {
    params: proxy,
    schema: [
      { type: "select", key: "system", label: "L-System", options: Object.keys(Rules) },
      { type: "slider", key: "iterations", label: "Iterations", min: 1, max: 12, step: 1 },
      { type: "slider", key: "angle", label: "Branch Angle °", min: 5, max: 120, step: 1 },
      { type: "slider", key: "length", label: "Segment Length", min: 0.3, max: 3, step: 0.1 },
      { type: "slider", key: "shrink", label: "Shrink Factor", min: 0.4, max: 0.95, step: 0.05 },
      { type: "slider", key: "width", label: "Branch Width", min: 0.05, max: 0.5, step: 0.025 },
      { type: "button", key: "_grow", label: "Rebuild", action: buildTree },
    ],
    readouts: { segments: segCountRef },
    update,
    reset,
    detach,
  };
}
