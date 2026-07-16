import { reactive, ref } from "vue";
import {
  addToScene,
  attachControl,
  createArcRotateCamera,
  createCapsule,
  createSphere,
  createHemisphericLight,
  createPbrMaterial,
  invalidateRenderBundles,
  setMeshVisible,
  setThinInstanceCount,
} from "@babylonjs/lite";
import type { EngineContext, SceneContext } from "@babylonjs/lite";
import { AgentBuffer, upToDirQuat } from "@/lib/agents";
import type { SimHandle } from "@/types/sim";
import { EcosystemGpu, GpuMaxPrey, GpuMaxPred, GpuMaxFood } from "./gpu";

/**
 * Agent-based ecosystem (Boids + energy/hunger + reproduction):
 *  - Food spawns randomly on a flat, toroidal (edge-wrapping) plane.
 *  - Prey flock (separation/alignment/cohesion), seek food, and flee predators.
 *    Eating food restores energy; enough energy makes a prey split into a copy.
 *  - Predators hunt the nearest prey; eating a prey restores energy and lets
 *    them reproduce. Both burn energy over time and STARVE at zero energy.
 *  - A lifespan cap exists but is long, so starvation/predation dominate.
 * Populations are emergent from these rules — there is no population equation.
 */

const MaxPrey = 800;
const MaxPred = 150;
const MaxFood = 500;
const Bound = 26;
const World = Bound * 2;

// Interaction radii (world units).
const FoodEatR2 = 1.8 * 1.8;   // prey close enough to eat food
const PreyEatR2 = 1.8 * 1.8;   // predator close enough to eat prey
const FleeR = 12, FleeR2 = FleeR * FleeR;
// Reference energies used only for the hunger-based color shading.
const PreyFull = 50, PredFull = 90;

export function buildSimScene(
  engine: EngineContext,
  scene: SceneContext,
  canvas: HTMLCanvasElement
): SimHandle {
  // Flat ecosystem viewed from a high angle.
  const camera = createArcRotateCamera(-Math.PI * 0.5, Math.PI * 0.28, 74, { x: 0, y: 0, z: 0 });
  scene.camera = camera;
  const detach = attachControl(camera, canvas, scene);

  addToScene(scene, createHemisphericLight([0, 1, 0], 0.9));

  const preyMesh = createCapsule(engine, { height: 0.9, radius: 0.25, tessellation: 6 });
  preyMesh.material = createPbrMaterial({ baseColorFactor: [1, 1, 1, 1], metallicFactor: 0.1, roughnessFactor: 0.5 });
  addToScene(scene, preyMesh);

  const predMesh = createSphere(engine, { segments: 7, diameter: 1.5 });
  predMesh.material = createPbrMaterial({ baseColorFactor: [1, 1, 1, 1], metallicFactor: 0.3, roughnessFactor: 0.4 });
  addToScene(scene, predMesh);

  const foodMesh = createSphere(engine, { segments: 5, diameter: 0.5 });
  foodMesh.material = createPbrMaterial({ unlit: true, baseColorFactor: [1, 1, 1, 1] });
  addToScene(scene, foodMesh);

  const params = reactive({
    timeScale: 1.0,
    // Movement
    preySpeed: 6,
    predSpeed: 6.5,
    separation: 1.5,
    alignment: 0.8,
    cohesion: 0.5,
    // Food
    foodRate: 12,            // food items spawned per second
    foodEnergy: 26,          // energy a prey gains from one food
    // Prey energy
    preyMetabolism: 4,       // energy burned per second
    preyFoodToRepro: 1,      // food a prey must eat to spawn one offspring
    // Predator energy
    predGain: 45,            // energy gained from eating one prey
    predMetabolism: 6,       // energy burned per second
    predPreyToRepro: 2,      // prey a predator must eat to spawn one offspring
    // Aging
    lifespan: 90,            // seconds before old age (rare cause of death)
    // Population caps (limited by the fixed buffer sizes)
    maxPrey: MaxPrey,
    maxPred: MaxPred,
    maxFood: MaxFood,
    // Simulation area (flat plane that wraps at the edges)
    width: World,
    depth: World,
    // GPU toggle
    gpuMode: false,
    // Collision (GPU only, same as boids GPU collision)
    collision:         true,
    collisionRadius:   0.3,
    collisionStrength: 0.5,
  });

  const preyCountRef = ref(0);
  const predCountRef = ref(0);
  const foodCountRef = ref(0);

  // ── Dense agent pools (swap-remove on death, append on birth) ──
  const ppx = new Float32Array(MaxPrey), ppz = new Float32Array(MaxPrey);
  const pvx = new Float32Array(MaxPrey), pvz = new Float32Array(MaxPrey);
  const pen = new Float32Array(MaxPrey), page = new Float32Array(MaxPrey);
  const pfood = new Int32Array(MaxPrey);   // food eaten since last reproduction
  let nPrey = 0;

  const edpx = new Float32Array(MaxPred), edpz = new Float32Array(MaxPred);
  const edvx = new Float32Array(MaxPred), edvz = new Float32Array(MaxPred);
  const eden = new Float32Array(MaxPred), edage = new Float32Array(MaxPred);
  const eeaten = new Int32Array(MaxPred);  // prey eaten since last reproduction
  let nPred = 0;

  const fpx = new Float32Array(MaxFood), fpz = new Float32Array(MaxFood);
  let nFood = 0;

  // Boids accumulators (flat XZ).
  const sepX = new Float32Array(MaxPrey), sepZ = new Float32Array(MaxPrey);
  const aliX = new Float32Array(MaxPrey), aliZ = new Float32Array(MaxPrey);
  const cohX = new Float32Array(MaxPrey), cohZ = new Float32Array(MaxPrey);
  const ncArr = new Int32Array(MaxPrey), nsArr = new Int32Array(MaxPrey);
  // Same-type positional-collision pushout accumulators (prey + predators).
  const pColX = new Float32Array(MaxPrey), pColZ = new Float32Array(MaxPrey);
  const eColX = new Float32Array(MaxPred), eColZ = new Float32Array(MaxPred);

  const preyBuf = new AgentBuffer(MaxPrey);
  const predBuf = new AgentBuffer(MaxPred);
  const foodBuf = new AgentBuffer(MaxFood);
  preyBuf.attach(engine, preyMesh);
  predBuf.attach(engine, predMesh);
  foodBuf.attach(engine, foodMesh);

  // Rectangular toroidal world; dimensions are live (params.width × params.depth),
  // refreshed each frame. wrapD* = nearest-image delta, wrapP* = wrap a position.
  let worldX = params.width, boundX = worldX * 0.5;
  let worldZ = params.depth, boundZ = worldZ * 0.5;
  const wrapDX = (d: number) => (d > boundX ? d - worldX : d < -boundX ? d + worldX : d);
  const wrapDZ = (d: number) => (d > boundZ ? d - worldZ : d < -boundZ ? d + worldZ : d);
  const wrapPX = (p: number) => (p > boundX ? p - worldX : p < -boundX ? p + worldX : p);
  const wrapPZ = (p: number) => (p > boundZ ? p - worldZ : p < -boundZ ? p + worldZ : p);
  const randX = () => (Math.random() - 0.5) * worldX;
  const randZ = () => (Math.random() - 0.5) * worldZ;

  function addPrey(x: number, z: number, energy: number) {
    if (nPrey >= MaxPrey) return;
    const i = nPrey++;
    ppx[i] = x; ppz[i] = z;
    const a = Math.random() * Math.PI * 2;
    pvx[i] = Math.cos(a); pvz[i] = Math.sin(a);
    pen[i] = energy; page[i] = 0; pfood[i] = 0;
  }
  function removePrey(i: number) {
    const last = --nPrey;
    ppx[i] = ppx[last]; ppz[i] = ppz[last];
    pvx[i] = pvx[last]; pvz[i] = pvz[last];
    pen[i] = pen[last]; page[i] = page[last]; pfood[i] = pfood[last];
  }
  function addPred(x: number, z: number, energy: number) {
    if (nPred >= MaxPred) return;
    const i = nPred++;
    edpx[i] = x; edpz[i] = z;
    const a = Math.random() * Math.PI * 2;
    edvx[i] = Math.cos(a); edvz[i] = Math.sin(a);
    eden[i] = energy; edage[i] = 0; eeaten[i] = 0;
  }
  function removePred(i: number) {
    const last = --nPred;
    edpx[i] = edpx[last]; edpz[i] = edpz[last];
    edvx[i] = edvx[last]; edvz[i] = edvz[last];
    eden[i] = eden[last]; edage[i] = edage[last]; eeaten[i] = eeaten[last];
  }
  function addFood(x: number, z: number) {
    if (nFood >= MaxFood) return;
    const i = nFood++;
    fpx[i] = x; fpz[i] = z;
  }
  function removeFood(i: number) {
    const last = --nFood;
    fpx[i] = fpx[last]; fpz[i] = fpz[last];
  }

  function spawn() {
    // Match the current dimensions when scattering the initial agents.
    worldX = params.width; boundX = worldX * 0.5;
    worldZ = params.depth; boundZ = worldZ * 0.5;
    nPrey = nPred = nFood = 0;
    for (let i = 0; i < 140; i++) addPrey(randX(), randZ(), 30);
    for (let i = 0; i < 16; i++) addPred(randX(), randZ(), 55);
    for (let i = 0; i < 160; i++) addFood(randX(), randZ());
  }
  spawn();

  // ── GPU path ──────────────────────────────────────────────────────────────
  const gpu          = new EcosystemGpu(engine, scene);
  const gpuAvailable = gpu.ok;

  // Per-mode world-size ceilings for the Sim Width/Depth sliders.
  const CpuWorldMax = 120;
  const GpuWorldMax = 300;
  const GpuWorldDefault = 200;
  let gpuFoodAccum = 0;

  // Reactive Sim Width/Depth sliders — max swaps with mode so the GPU can use a
  // much larger toroidal world (needed for tens of thousands of agents).
  const widthSlider = reactive<import("@/types/sim").SliderParam>({
    type: "slider", key: "width", label: "Sim Width", min: 20, max: CpuWorldMax, step: 2,
  });
  const depthSlider = reactive<import("@/types/sim").SliderParam>({
    type: "slider", key: "depth", label: "Sim Depth", min: 20, max: CpuWorldMax, step: 2,
  });

  // Visibility: keep CPU and GPU meshes mutually exclusive to avoid the stray
  // base-instance issue (thin-instance count 0 still draws one base geometry).
  function applyVisibility(): void {
    const g = params.gpuMode && gpuAvailable;
    setMeshVisible(preyMesh, !g);
    setMeshVisible(predMesh, !g);
    setMeshVisible(foodMesh, !g);
    if (gpuAvailable) {
      setMeshVisible(gpu.preyMesh,  g);
      setMeshVisible(gpu.predMesh,  g);
      setMeshVisible(gpu.foodMesh,  g);
    }
  }

  function seedGpuPopulation(): void {
    // Scale the starting population to the world area (density ~0.4 prey/unit²),
    // capped by the GPU buffer sizes.
    const wX = params.width, wZ = params.depth;
    const area = wX * wZ;
    const n  = Math.min(Math.round(area * 0.4), Math.floor(GpuMaxPrey * 0.7));
    const np = Math.min(Math.max(Math.round(n / 50), 8), GpuMaxPred >> 1);
    const nf = Math.min(Math.round(n / 7), GpuMaxFood >> 1);
    const rx = () => (Math.random() - 0.5) * wX;
    const rz = () => (Math.random() - 0.5) * wZ;

    const pX = new Float32Array(n), pZ = new Float32Array(n);
    const pVX = new Float32Array(n), pVZ = new Float32Array(n), pE = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      pX[i] = rx(); pZ[i] = rz();
      const a = Math.random() * Math.PI * 2; pVX[i] = Math.cos(a); pVZ[i] = Math.sin(a);
      pE[i] = 30;
    }
    const dX = new Float32Array(np), dZ = new Float32Array(np), dE = new Float32Array(np);
    for (let i = 0; i < np; i++) { dX[i] = rx(); dZ[i] = rz(); dE[i] = 55; }
    const fX = new Float32Array(nf), fZ = new Float32Array(nf);
    for (let i = 0; i < nf; i++) { fX[i] = rx(); fZ[i] = rz(); }

    gpu.seed({ x: pX, z: pZ, vx: pVX, vz: pVZ, energy: pE, n },
             { x: dX, z: dZ, energy: dE, n: np },
             { x: fX, z: fZ, n: nf });
    gpuFoodAccum = 0;
    preyCountRef.value = n; predCountRef.value = np; foodCountRef.value = nf;
  }

  function switchToGpu(): void {
    if (!gpuAvailable) { params.gpuMode = false; return; }
    // Grow the world for the GPU (bigger toroidal plane for big populations).
    widthSlider.max = GpuWorldMax; depthSlider.max = GpuWorldMax;
    params.width = GpuWorldDefault; params.depth = GpuWorldDefault;
    // Zero CPU meshes so they don't draw during GPU mode.
    setThinInstanceCount(preyMesh, 0);
    setThinInstanceCount(predMesh, 0);
    setThinInstanceCount(foodMesh, 0);
    invalidateRenderBundles(engine);
    seedGpuPopulation();
    camera.radius = Math.max(params.width, params.depth) * 0.75;
    applyVisibility();
  }

  function switchToCpu(): void {
    widthSlider.max = CpuWorldMax; depthSlider.max = CpuWorldMax;
    if (params.width > CpuWorldMax) params.width = World;
    if (params.depth > CpuWorldMax) params.depth = World;
    gpu.hideMeshes();
    camera.radius = 74;
    spawn();
    applyVisibility();
  }

  let prevGpu = false;
  // Initial visibility: GPU meshes hidden at startup.
  applyVisibility();

  let foodAccum = 0;

  function update(dt: number) {
    const isGpu = params.gpuMode;

    // React to mode toggle.
    if (isGpu !== prevGpu) {
      prevGpu = isGpu;
      if (isGpu) switchToGpu(); else switchToCpu();
    }

    if (isGpu && gpuAvailable) {
      // ── GPU path ────────────────────────────────────────────────────────
      const ds = dt * 0.001 * params.timeScale;
      // Food spawn is an absolute per-second rate tuned for the small CPU world.
      // Scale it by the world-area ratio so food DENSITY (and therefore the
      // prey/predator balance) matches the CPU sim — otherwise the much larger
      // GPU world is food-starved and predators annihilate whole flocks instantly.
      const areaScale = (params.width * params.depth) / (World * World);
      gpuFoodAccum += params.foodRate * areaScale * ds;
      const spawnCount = Math.floor(gpuFoodAccum);
      gpuFoodAccum -= spawnCount;
      const c = gpu.counts;
      // Clamp food spawns so the buffer can't overflow.
      const spawnRoom = Math.max(0, GpuMaxFood - c.nFood);
      gpu.dispatch({
        nPrey: c.nPrey, nPred: c.nPred, nFood: c.nFood,
        foodSpawnCount: Math.min(spawnCount, spawnRoom),
        // GPU honours the Sim Width/Depth sliders (toroidal world).
        worldX: params.width, worldZ: params.depth,
        dt: ds,
        preySpeed: params.preySpeed, predSpeed: params.predSpeed,
        separation: params.separation, alignment: params.alignment, cohesion: params.cohesion,
        foodEnergy: params.foodEnergy,
        preyMetabolism: params.preyMetabolism, preyFoodToRepro: params.preyFoodToRepro,
        predGain: params.predGain, predMetabolism: params.predMetabolism, predPreyToRepro: params.predPreyToRepro,
        lifespan: params.lifespan,
        // GPU caps = buffer capacity (population is food-limited, not slider-capped).
        maxPrey: GpuMaxPrey, maxPred: GpuMaxPred, maxFood: GpuMaxFood,
        collRadius: params.collisionRadius, collStrength: params.collisionStrength,
      });
      // Update readouts from latest readback (1-2 frames stale).
      preyCountRef.value = gpu.counts.nPrey;
      predCountRef.value = gpu.counts.nPred;
      foodCountRef.value = gpu.counts.nFood;
      return;
    }

    const ds = dt * 0.001 * params.timeScale;

    // Refresh the (live-adjustable) world dimensions.
    worldX = params.width; boundX = worldX * 0.5;
    worldZ = params.depth; boundZ = worldZ * 0.5;

    // ── Spawn food at the configured rate ──
    foodAccum += params.foodRate * ds;
    while (foodAccum >= 1) { if (nFood < params.maxFood) addFood(randX(), randZ()); foodAccum -= 1; }

    // ── Prey flocking accumulators (toroidal, O(n²)) ──
    const r2 = 36, sr2 = 6;
    const doColl   = params.collision;
    const collDist = params.collisionRadius * 2;
    const collD2   = collDist * collDist;
    const collStr  = params.collisionStrength;
    for (let i = 0; i < nPrey; i++) {
      sepX[i] = sepZ[i] = aliX[i] = aliZ[i] = cohX[i] = cohZ[i] = 0;
      ncArr[i] = nsArr[i] = 0;
      pColX[i] = pColZ[i] = 0;
    }
    for (let i = 0; i < nPrey - 1; i++) {
      for (let j = i + 1; j < nPrey; j++) {
        const dx = wrapDX(ppx[j] - ppx[i]);
        const dz = wrapDZ(ppz[j] - ppz[i]);
        const d2 = dx * dx + dz * dz;
        if (d2 < r2) {
          aliX[i] += pvx[j]; aliZ[i] += pvz[j];
          aliX[j] += pvx[i]; aliZ[j] += pvz[i];
          cohX[i] += dx; cohZ[i] += dz;
          cohX[j] -= dx; cohZ[j] -= dz;
          ncArr[i]++; ncArr[j]++;
          if (d2 < sr2) {
            sepX[i] -= dx; sepZ[i] -= dz;
            sepX[j] += dx; sepZ[j] += dz;
            nsArr[i]++; nsArr[j]++;
          }
        }
        // Same-type soft collision pushout (positional; matches the GPU path).
        if (doColl && d2 < collD2 && d2 > 1e-8) {
          const d    = Math.sqrt(d2);
          const push = (collDist - d) * 0.5 * collStr / d;
          pColX[i] -= dx * push; pColZ[i] -= dz * push;
          pColX[j] += dx * push; pColZ[j] += dz * push;
        }
      }
    }

    // ── Prey update (movement, eating, energy, reproduction, death) ──
    const preySpd = params.preySpeed;
    let i = 0;
    while (i < nPrey) {
      let fx = 0, fz = 0;
      if (nsArr[i] > 0) { fx += sepX[i] * params.separation; fz += sepZ[i] * params.separation; }
      if (ncArr[i] > 0) {
        const inv = 1 / ncArr[i];
        fx += (aliX[i] * inv - pvx[i]) * params.alignment;
        fz += (aliZ[i] * inv - pvz[i]) * params.alignment;
        fx += (cohX[i] * inv) * params.cohesion * 0.05;
        fz += (cohZ[i] * inv) * params.cohesion * 0.05;
      }

      // Seek nearest food (and eat it if close enough).
      let fnd = Infinity, fdx = 0, fdz = 0, fdi = -1;
      for (let f = 0; f < nFood; f++) {
        const dx = wrapDX(fpx[f] - ppx[i]);
        const dz = wrapDZ(fpz[f] - ppz[i]);
        const d2 = dx * dx + dz * dz;
        if (d2 < fnd) { fnd = d2; fdx = dx; fdz = dz; fdi = f; }
      }
      if (fdi >= 0) {
        // Hungrier prey seek food more aggressively.
        const hunger = 1 - Math.min(pen[i] / PreyFull, 1);
        const seek = (0.6 + hunger * 1.4);
        const len = Math.sqrt(fnd) + 0.001;
        fx += (fdx / len) * seek;
        fz += (fdz / len) * seek;
        if (fnd < FoodEatR2) {
          pen[i] += params.foodEnergy;
          removeFood(fdi);
          // Reproduce after eating enough food (split energy with the copy).
          if (++pfood[i] >= params.preyFoodToRepro && nPrey < params.maxPrey) {
            pfood[i] = 0;
            pen[i] *= 0.5;
            addPrey(wrapPX(ppx[i] + (Math.random() - 0.5) * 2), wrapPZ(ppz[i] + (Math.random() - 0.5) * 2), pen[i]);
          }
        }
      }

      // Flee nearest predator.
      let pnd = Infinity, pdx = 0, pdz = 0;
      for (let p = 0; p < nPred; p++) {
        const dx = wrapDX(edpx[p] - ppx[i]);
        const dz = wrapDZ(edpz[p] - ppz[i]);
        const d2 = dx * dx + dz * dz;
        if (d2 < pnd) { pnd = d2; pdx = dx; pdz = dz; }
      }
      if (pnd < FleeR2) {
        const dist = Math.sqrt(pnd) + 0.001;
        const flee = (FleeR - dist) / dist * 3;
        fx -= pdx * flee; fz -= pdz * flee;
      }

      // Integrate.
      pvx[i] += fx * (dt * 0.001); pvz[i] += fz * (dt * 0.001);
      const vLen = Math.sqrt(pvx[i] * pvx[i] + pvz[i] * pvz[i]);
      if (vLen > 0.001) { const sc = preySpd / vLen; pvx[i] *= sc; pvz[i] *= sc; }
      ppx[i] = wrapPX(ppx[i] + pvx[i] * ds + pColX[i]);
      ppz[i] = wrapPZ(ppz[i] + pvz[i] * ds + pColZ[i]);

      // Energy + aging.
      pen[i] -= params.preyMetabolism * ds;
      page[i] += ds;
      if (pen[i] <= 0 || page[i] > params.lifespan) { removePrey(i); continue; }
      i++;
    }

    // ── Predator same-type collision pre-pass (O(nPred²), toroidal) ──
    for (let k = 0; k < nPred; k++) { eColX[k] = eColZ[k] = 0; }
    if (doColl) {
      for (let a = 0; a < nPred - 1; a++) {
        for (let b = a + 1; b < nPred; b++) {
          const dx = wrapDX(edpx[b] - edpx[a]);
          const dz = wrapDZ(edpz[b] - edpz[a]);
          const d2 = dx * dx + dz * dz;
          if (d2 < collD2 && d2 > 1e-8) {
            const d    = Math.sqrt(d2);
            const push = (collDist - d) * 0.5 * collStr / d;
            eColX[a] -= dx * push; eColZ[a] -= dz * push;
            eColX[b] += dx * push; eColZ[b] += dz * push;
          }
        }
      }
    }

    // ── Predator update (hunt, eat prey, energy, reproduction, death) ──
    const predSpd = params.predSpeed;
    let p = 0;
    while (p < nPred) {
      let nd = Infinity, ndx = 0, ndz = 0, ndi = -1;
      for (let k = 0; k < nPrey; k++) {
        const dx = wrapDX(ppx[k] - edpx[p]);
        const dz = wrapDZ(ppz[k] - edpz[p]);
        const d2 = dx * dx + dz * dz;
        if (d2 < nd) { nd = d2; ndx = dx; ndz = dz; ndi = k; }
      }
      if (ndi >= 0) {
        const len = Math.sqrt(ndx * ndx + ndz * ndz) + 0.001;
        edvx[p] += (ndx / len - edvx[p]) * 0.08;
        edvz[p] += (ndz / len - edvz[p]) * 0.08;
        if (nd < PreyEatR2) {
          eden[p] += params.predGain;
          removePrey(ndi);
          // Reproduce after eating enough prey (split energy with the copy).
          if (++eeaten[p] >= params.predPreyToRepro && nPred < params.maxPred) {
            eeaten[p] = 0;
            eden[p] *= 0.5;
            addPred(wrapPX(edpx[p] + (Math.random() - 0.5) * 2), wrapPZ(edpz[p] + (Math.random() - 0.5) * 2), eden[p]);
          }
        }
      }
      const vl = Math.sqrt(edvx[p] * edvx[p] + edvz[p] * edvz[p]);
      if (vl > 0.001) { edvx[p] /= vl; edvz[p] /= vl; }
      edpx[p] = wrapPX(edpx[p] + edvx[p] * predSpd * ds + eColX[p]);
      edpz[p] = wrapPZ(edpz[p] + edvz[p] * predSpd * ds + eColZ[p]);

      eden[p] -= params.predMetabolism * ds;
      edage[p] += ds;
      if (eden[p] <= 0 || edage[p] > params.lifespan) { removePred(p); continue; }
      p++;
    }

    // ── Write instances ──
    for (let k = 0; k < nPrey; k++) {
      const frac = Math.min(pen[k] / PreyFull, 1);
      const [qx, qy, qz, qw] = upToDirQuat(pvx[k], 0, pvz[k]);
      preyBuf.writeTransform(k, ppx[k], 0, ppz[k], qx, qy, qz, qw, 1, 1, 1);
      preyBuf.writeColor(k, 0.05, 0.3 + 0.6 * frac, 0.45 + 0.55 * frac);
    }
    preyBuf.commit(preyMesh, nPrey);

    for (let k = 0; k < nPred; k++) {
      const frac = Math.min(eden[k] / PredFull, 1);
      predBuf.writeScale(k, edpx[k], 0, edpz[k]);
      predBuf.writeColor(k, 0.4 + 0.6 * frac, 0.05, 0.3 + 0.35 * frac);
    }
    predBuf.commit(predMesh, nPred);

    for (let k = 0; k < nFood; k++) {
      foodBuf.writeScale(k, fpx[k], 0, fpz[k]);
      foodBuf.writeColor(k, 0.55, 0.95, 0.2);
    }
    foodBuf.commit(foodMesh, nFood);

    preyCountRef.value = nPrey;
    predCountRef.value = nPred;
    foodCountRef.value = nFood;
  }

  function reset() {
    if (params.gpuMode && gpuAvailable) seedGpuPopulation();
    else spawn();
  }

  return {
    params,
    schema: [
      {
        type:  "toggle",
        key:   "gpuMode",
        label: gpuAvailable ? "GPU Mode" : "GPU Mode (unavailable)",
      },
      { type: "slider", key: "timeScale",         label: "Time Scale",           min: 0.1, max: 3,    step: 0.1 },
      { type: "slider", key: "preySpeed",          label: "Prey Speed",           min: 1,   max: 14,   step: 0.5 },
      { type: "slider", key: "predSpeed",          label: "Predator Speed",       min: 1,   max: 14,   step: 0.5 },
      { type: "slider", key: "separation",         label: "Separation",           min: 0,   max: 4,    step: 0.1 },
      { type: "slider", key: "alignment",          label: "Alignment",            min: 0,   max: 4,    step: 0.1 },
      { type: "slider", key: "cohesion",           label: "Cohesion",             min: 0,   max: 4,    step: 0.1 },
      { type: "toggle", key: "collision",          label: "Same-type Collision" },
      { type: "slider", key: "collisionRadius",    label: "Collision Radius",     min: 0.1, max: 2,    step: 0.05 },
      { type: "slider", key: "collisionStrength",  label: "Collision Strength",   min: 0,   max: 1,    step: 0.05 },
      { type: "slider", key: "foodRate",           label: "Food Spawn / s",       min: 0,   max: 60,   step: 1 },
      { type: "slider", key: "foodEnergy",         label: "Food Energy",          min: 5,   max: 60,   step: 1 },
      { type: "slider", key: "preyMetabolism",     label: "Prey Metabolism",      min: 0.5, max: 15,   step: 0.5 },
      { type: "slider", key: "preyFoodToRepro",    label: "Food to Reproduce",    min: 1,   max: 10,   step: 1 },
      { type: "slider", key: "predGain",           label: "Predator Gain / Prey", min: 10,  max: 100,  step: 5 },
      { type: "slider", key: "predMetabolism",     label: "Predator Metabolism",  min: 0.5, max: 15,   step: 0.5 },
      { type: "slider", key: "predPreyToRepro",    label: "Prey to Reproduce",    min: 1,   max: 10,   step: 1 },
      { type: "slider", key: "lifespan",           label: "Lifespan (s)",         min: 20,  max: 240,  step: 5 },
      { type: "slider", key: "maxPrey",            label: "Max Prey",             min: 50,  max: GpuMaxPrey, step: 100 },
      { type: "slider", key: "maxPred",            label: "Max Predators",        min: 10,  max: GpuMaxPred, step: 10 },
      { type: "slider", key: "maxFood",            label: "Max Food",             min: 50,  max: GpuMaxFood, step: 100 },
      widthSlider,
      depthSlider,
    ],
    readouts: { prey: preyCountRef, predators: predCountRef, food: foodCountRef },
    seriesLabels: ["Prey", "Predators", "Food"],
    seriesColors: ["#00e5ff", "#ff00cc", "#8fff2a"],
    // getSeries reads the count refs — updated by both CPU and GPU paths.
    getSeries: () => [preyCountRef.value as number, predCountRef.value as number, foodCountRef.value as number],
    update,
    reset,
    detach,
  };
}
