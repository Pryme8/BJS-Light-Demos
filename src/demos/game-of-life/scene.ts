import { reactive, ref } from "vue";
import {
  addToScene,
  attachControl,
  createArcRotateCamera,
  createBox,
  createHemisphericLight,
  createDirectionalLight,
  createPbrMaterial,
  getViewMatrix,
  getProjectionMatrix,
  mat4Multiply,
  mat4Invert,
} from "@babylonjs/lite";
import type { EngineContext, SceneContext, Mat4 } from "@babylonjs/lite";
import { AgentBuffer } from "@/lib/agents";
import type { SimHandle } from "@/types/sim";

const GridW = 80;
const GridH = 80;
const CellSize = 0.9;

export function buildSimScene(
  engine: EngineContext,
  scene: SceneContext,
  canvas: HTMLCanvasElement
): SimHandle {
  const camera = createArcRotateCamera(-Math.PI * 0.5, Math.PI * 0.32, GridW * 0.85, { x: 0, y: 0, z: 0 });
  scene.camera = camera;
  const detachControls = attachControl(camera, canvas, scene);

  addToScene(scene, createHemisphericLight([0, 1, 0], 0.5));
  addToScene(scene, createDirectionalLight([-0.4, -1, 0.6]));

  const cellMesh = createBox(engine, 1);
  cellMesh.material = createPbrMaterial({ baseColorFactor: [0.05, 0.8, 1, 1], metallicFactor: 0.1, roughnessFactor: 0.4 });
  addToScene(scene, cellMesh);

  const params = reactive({
    speed: 6,      // steps per second
    density: 0.3,
    brush: 1,      // click brush radius (in cells)
    survival2: true,
    survival3: true,
    birth3: true,
  });

  const liveCount = ref(0);

  const grid = new Uint8Array(GridW * GridH);
  const next = new Uint8Array(GridW * GridH);
  const buf = new AgentBuffer(GridW * GridH);
  buf.attach(engine, cellMesh);

  function idx(x: number, y: number) {
    return ((x + GridW) % GridW) + ((y + GridH) % GridH) * GridW;
  }

  function spawnRandom() {
    const d = params.density;
    for (let i = 0; i < grid.length; i++) grid[i] = Math.random() < d ? 1 : 0;
  }
  spawnAll();

  function spawnAll() { spawnRandom(); }

  // Write all instances (alive = full height, dead = flat)
  function writeInstances() {
    let alive = 0;
    for (let y = 0; y < GridH; y++) {
      for (let x = 0; x < GridW; x++) {
        const i = x + y * GridW;
        const live = grid[i] === 1;
        const wx = (x - GridW * 0.5) * CellSize;
        const wz = (y - GridH * 0.5) * CellSize;
        if (live) {
          buf.writeTRS(i, wx, 0, wz, CellSize * 0.88, CellSize * 0.88, CellSize * 0.88);
          buf.writeColor(i, 0.05, 0.85, 1, 1);
          alive++;
        } else {
          buf.writeTRS(i, wx, -0.38, wz, CellSize * 0.82, CellSize * 0.08, CellSize * 0.82);
          buf.writeColor(i, 0.08, 0.12, 0.18, 1);
        }
      }
    }
    liveCount.value = alive;
    buf.commit(cellMesh, GridW * GridH);
  }

  let accumDt = 0;

  function update(dt: number) {
    accumDt += dt;
    const stepMs = 1000 / params.speed;
    if (accumDt < stepMs) {
      writeInstances();
      return;
    }
    accumDt -= stepMs;

    // Compute next generation
    for (let y = 0; y < GridH; y++) {
      for (let x = 0; x < GridW; x++) {
        const neighbors =
          grid[idx(x - 1, y - 1)] + grid[idx(x, y - 1)] + grid[idx(x + 1, y - 1)] +
          grid[idx(x - 1, y)]                             + grid[idx(x + 1, y)] +
          grid[idx(x - 1, y + 1)] + grid[idx(x, y + 1)] + grid[idx(x + 1, y + 1)];
        const alive = grid[x + y * GridW];
        if (alive) {
          next[x + y * GridW] = (
            (neighbors === 2 && params.survival2) ||
            (neighbors === 3 && params.survival3)
          ) ? 1 : 0;
        } else {
          next[x + y * GridW] = (neighbors === 3 && params.birth3) ? 1 : 0;
        }
      }
    }
    grid.set(next);
    writeInstances();
  }

  function reset() { spawnRandom(); }

  // ── Click-to-seed ──────────────────────────────────────────────────────────
  // Unproject the click into a world ray, intersect the grid plane (y = 0), map
  // to a cell, and toggle a small brush. Left-drag still orbits (we only paint
  // when the pointer barely moved between down and up).

  function mulVec4(m: Mat4, x: number, y: number, z: number, w: number): [number, number, number, number] {
    // Column-major: out[r] = sum_c m[c*4 + r] * v[c]
    return [
      m[0] * x + m[4] * y + m[8]  * z + m[12] * w,
      m[1] * x + m[5] * y + m[9]  * z + m[13] * w,
      m[2] * x + m[6] * y + m[10] * z + m[14] * w,
      m[3] * x + m[7] * y + m[11] * z + m[15] * w,
    ];
  }

  function paintAtPixel(clientX: number, clientY: number) {
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return;
    const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = 1 - ((clientY - rect.top) / rect.height) * 2;

    const view = getViewMatrix(camera);
    const proj = getProjectionMatrix(camera, rect.width / rect.height);
    const inv = mat4Invert(mat4Multiply(proj, view));
    if (!inv) return;

    const near = mulVec4(inv, ndcX, ndcY, 0, 1);
    const far = mulVec4(inv, ndcX, ndcY, 1, 1);
    const ox = near[0] / near[3], oy = near[1] / near[3], oz = near[2] / near[3];
    const fx = far[0] / far[3], fy = far[1] / far[3], fz = far[2] / far[3];
    const dx = fx - ox, dy = fy - oy, dz = fz - oz;
    if (Math.abs(dy) < 1e-6) return;

    // Intersect the grid plane y = 0.
    const t = -oy / dy;
    if (t < 0) return;
    const hx = ox + dx * t;
    const hz = oz + dz * t;

    const cx = Math.round(hx / CellSize + GridW * 0.5);
    const cy = Math.round(hz / CellSize + GridH * 0.5);
    if (cx < 0 || cx >= GridW || cy < 0 || cy >= GridH) return;

    // Toggle a brush: alive if the clicked cell was dead, else clear.
    const brushRadius = params.brush;
    const target = grid[cx + cy * GridW] === 1 ? 0 : 1;
    for (let by = -brushRadius; by <= brushRadius; by++) {
      for (let bx = -brushRadius; bx <= brushRadius; bx++) {
        const gx = cx + bx, gy = cy + by;
        if (gx >= 0 && gx < GridW && gy >= 0 && gy < GridH) {
          grid[gx + gy * GridW] = target;
        }
      }
    }
    writeInstances();
  }

  let downX = 0, downY = 0;
  function onPointerDown(e: PointerEvent) { downX = e.clientX; downY = e.clientY; }
  function onPointerUp(e: PointerEvent) {
    if (Math.hypot(e.clientX - downX, e.clientY - downY) <= 6) {
      paintAtPixel(e.clientX, e.clientY);
    }
  }
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointerup", onPointerUp);

  function detach() {
    detachControls();
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointerup", onPointerUp);
  }

  return {
    params,
    schema: [
      { type: "slider", key: "speed", label: "Steps/sec", min: 1, max: 30, step: 1 },
      { type: "slider", key: "density", label: "Init. Density", min: 0.05, max: 0.8, step: 0.05 },
      { type: "slider", key: "brush", label: "Brush Size", min: 0, max: 10, step: 1 },
      { type: "toggle", key: "survival2", label: "Survive with 2" },
      { type: "toggle", key: "survival3", label: "Survive with 3" },
      { type: "toggle", key: "birth3", label: "Born with 3" },
      { type: "button", key: "_respawn", label: "Respawn", action: spawnRandom },
    ],
    readouts: { alive: liveCount },
    update,
    reset,
    detach,
  };
}
