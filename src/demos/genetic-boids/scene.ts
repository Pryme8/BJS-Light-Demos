import { reactive, ref } from "vue";
import {
  addToScene,
  attachControl,
  createArcRotateCamera,
  createBox,
  createCapsule,
  createCylinder,
  createSphere,
  createHemisphericLight,
  createDirectionalLight,
  createPbrMaterial,
} from "@babylonjs/lite";
import type { EngineContext, SceneContext } from "@babylonjs/lite";
import { AgentBuffer, upToDirQuat } from "@/lib/agents";
import type { SimHandle } from "@/types/sim";

/**
 * Evolving Navigators — Genetic Algorithm obstacle-course demo.
 *
 * Each agent carries a 7-gene steering genome evolved by the GA:
 *   seek        — strength of pull toward the goal
 *   visionArc   — half-angle of sensor cone in degrees (e.g. 60 → ±60° view)
 *   visionRange — max ray length (how far it can "see")
 *   rayCount    — number of rays spread across the arc (sensor resolution)
 *   turnRate    — max angular velocity in rad/s (manoeuvrability)
 *   avoidWeight — how strongly to react when a sensor ray hits a pillar
 *   wander      — random heading jitter (exploration noise)
 *
 * Each frame, alive agents:
 *  1. Cast rayCount rays evenly across [-visionArc, +visionArc] of their heading.
 *  2. Each ray that hits a pillar within visionRange produces an avoidance force
 *     opposing that ray direction (stronger the closer the hit).
 *  3. Seek force pulls toward goal; wander adds noise.
 *  4. Desired heading = atan2 of summed forces; actual heading change is clamped
 *     by turnRate (so low-turnRate agents can't dodge sharp obstacles).
 *
 * The best agent's sensor rays are drawn live each frame — watch the evolved
 * vision configuration in real time.
 *
 * GA: elitism + tournament selection → uniform crossover → per-gene mutation.
 * Ghost trail of the previous generation's best route persists into the next.
 */

// ── Capacity constants ────────────────────────────────────────────────────────
const MaxPop      = 400;
const MaxPillars  = 40;
const MaxWalls    = 30;
const MaxRays     = 13;      // max rays any genome can have
const TrailPts    = 100;     // stored path positions per agent
const GhostPts    = 200;     // max ghost-trail positions
const MaxTrailSegs = MaxPop * (TrailPts - 1);
const MaxGhostSegs = GhostPts - 1;

// ── Arena geometry ────────────────────────────────────────────────────────────
const ArenaW    = 30;
const ArenaD    = 42;
const StartZ    = -ArenaD + 5;
const GoalZ     =  ArenaD - 5;
const GoalR     =  3.5;
const AgentR    =  0.4;
const WallThick =  0.3;   // half-thickness of wall obstacle (collision + ray)
const WallH     =  6;     // visual height of walls
const InitialDist = GoalZ - StartZ;   // ≈ 74

// ── Genome definition ─────────────────────────────────────────────────────────
interface Genome {
  seek:        number;   // goal pull            [0.1, 5]
  visionArc:   number;   // half-arc degrees     [15, 160]
  visionRange: number;   // ray length           [2, 25]
  rayCount:    number;   // rays, int [3–13]     [3, 13]
  turnRate:    number;   // max turn rad/s       [0.5, 12]
  avoidWeight: number;   // ray repulsion        [0, 6]
  wander:      number;   // noise                [0, 1]
  maxSpeed:    number;   // top speed units/s       [3, 22]
  accel:       number;   // acceleration units/s²   [4, 40]
  stamina:     number;   // total energy pool              [40, 300]
  endurance:   number;   // min perf mult at zero stamina  [0.05, 1.0]
  // At full stamina: perf = 1.0.
  // At zero stamina: perf = endurance.
  // Interpolated linearly between — high endurance = gradual degradation,
  // low endurance = rapid performance collapse as the tank empties.
}

const GeneLimits = {
  seek:        [0.1,  5],
  visionArc:   [15,  160],
  visionRange: [2,    25],
  rayCount:    [3,    13],
  turnRate:    [0.5,  12],
  avoidWeight: [0,     6],
  wander:      [0,     1],
  maxSpeed:    [3,    22],
  accel:       [4,    40],
  stamina:     [40,  300],
  endurance:   [0.05, 1.0],
} as const;

// Stamina drain rates (per second):
//   speed drain: curSpeed × SpeedDrain/s
//   vision drain: visionArc° × nRays × VisionDrain/s
//   base drain: constant/s
// Calibrated so a moderate agent (speed 8, 6 rays, 60°) with stamina 100
// lasts ~10 s — just enough for the default 9 s episode.
const SpeedDrain  = 0.7;    // per unit-speed per second
const VisionDrain = 0.010;  // per (degree × ray) per second
const BaseDrain   = 1.5;    // flat cost per second

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

export function buildSimScene(
  engine: EngineContext,
  scene: SceneContext,
  canvas: HTMLCanvasElement
): SimHandle {

  const camera = createArcRotateCamera(-Math.PI * 0.5, Math.PI * 0.3, 105, { x: 0, y: 0, z: 2 });
  scene.camera = camera;
  const detach = attachControl(camera, canvas, scene);

  addToScene(scene, createHemisphericLight([0, 1, 0], 0.55));
  addToScene(scene, createDirectionalLight([-0.3, -1, 0.4]));

  // ── Meshes ────────────────────────────────────────────────────────────────
  const agentMesh = createCapsule(engine, { height: 1, radius: AgentR, tessellation: 6 });
  agentMesh.material = createPbrMaterial({ baseColorFactor: [1, 1, 1, 1], metallicFactor: 0.1, roughnessFactor: 0.5 });
  addToScene(scene, agentMesh);

  const pillarMesh = createCylinder(engine, { height: 7, diameter: 1, tessellation: 8 });
  pillarMesh.material = createPbrMaterial({ baseColorFactor: [0.28, 0.32, 0.42, 1], metallicFactor: 0.5, roughnessFactor: 0.5 });
  addToScene(scene, pillarMesh);

  const goalMesh = createSphere(engine, { segments: 10, diameter: GoalR * 2 });
  goalMesh.material = createPbrMaterial({ unlit: true, baseColorFactor: [1, 0.85, 0.1, 1] });
  addToScene(scene, goalMesh);

  // Live per-agent fading trails (unlit cylinder segments, alpha-blended).
  const trailMesh = createCylinder(engine, { height: 1, diameter: 1, tessellation: 5 });
  trailMesh.material = createPbrMaterial({ unlit: true, baseColorFactor: [1, 1, 1, 1], needAlphaBlending: true });
  addToScene(scene, trailMesh);

  // Ghost trail of the previous generation's best route.
  const ghostMesh = createCylinder(engine, { height: 1, diameter: 1, tessellation: 5 });
  ghostMesh.material = createPbrMaterial({ unlit: true, baseColorFactor: [1, 0.8, 0.05, 1], needAlphaBlending: true });
  addToScene(scene, ghostMesh);

  // Sensor ray visualisation — drawn for the best agent each frame.
  const rayMesh = createCylinder(engine, { height: 1, diameter: 1, tessellation: 4 });
  rayMesh.material = createPbrMaterial({ unlit: true, baseColorFactor: [1, 1, 1, 1], needAlphaBlending: true });
  addToScene(scene, rayMesh);

  // Wall obstacles — unit boxes scaled and rotated per instance.
  const wallMesh = createBox(engine, 1);
  wallMesh.material = createPbrMaterial({ baseColorFactor: [0.55, 0.32, 0.22, 1], metallicFactor: 0.3, roughnessFactor: 0.6 });
  addToScene(scene, wallMesh);

  // ── Agent buffers ─────────────────────────────────────────────────────────
  const agentBuf = new AgentBuffer(MaxPop);
  const pillarBuf = new AgentBuffer(MaxPillars);
  const goalBuf = new AgentBuffer(1);
  const trailBuf = new AgentBuffer(MaxTrailSegs);
  const ghostBuf = new AgentBuffer(MaxGhostSegs);
  const rayBuf = new AgentBuffer(MaxRays);
  const wallBuf = new AgentBuffer(MaxWalls);

  agentBuf.attach(engine, agentMesh);
  pillarBuf.attach(engine, pillarMesh);
  goalBuf.attach(engine, goalMesh, false);
  trailBuf.attach(engine, trailMesh);
  ghostBuf.attach(engine, ghostMesh);
  rayBuf.attach(engine, rayMesh);
  wallBuf.attach(engine, wallMesh);

  // ── Params ────────────────────────────────────────────────────────────────
  const params = reactive({
    population:   120,
    lifetime:     9,
    mutRate:      0.22,
    mutStrength:  0.35,
    elitism:      0.20,
    obstacles:    26,
    showRays:     true,

    // ── System evolutionary limits — mutation clamps to [sXxxLo, sXxxHi] ────
    // These are the absolute min/max any gene can reach via evolution.
    // Defaults match GeneLimits (the hard physical maximum).
    sSeekLo:   0.1,  sSeekHi:   5.0,
    sArcLo:   15.0,  sArcHi:  160.0,
    sRangeLo:  2.0,  sRangeHi: 25.0,
    sRaysLo:   3.0,  sRaysHi:  13.0,
    sTurnLo:   0.5,  sTurnHi:  12.0,
    sAvoidLo:  0.0,  sAvoidHi:  6.0,
    sWanderLo: 0.0,  sWanderHi: 1.0,
    sSpeedLo:  3.0,  sSpeedHi: 22.0,
    sAccelLo:  4.0,  sAccelHi: 40.0,
    sStamLo:  40.0,  sStamHi: 300.0,
    sEndurLo:  0.05, sEndurHi:  1.0,

    // ── Initial gene ranges — randGenome() samples uniformly from [iXxxLo, iXxxHi] ──
    // Adjust before hitting Reset to seed generation 0 with a particular bias.
    // Low starting ranges — agents earn better values through evolution.
    iSeekLo:       0.2,  iSeekHi:       1.0,
    iArcLo:       20.0,  iArcHi:       55.0,
    iRangeLo:      2.0,  iRangeHi:      7.0,
    iRaysLo:       3.0,  iRaysHi:       5.0,
    iTurnLo:       0.5,  iTurnHi:       2.5,
    iAvoidLo:      0.2,  iAvoidHi:      1.5,
    iWanderLo:     0.0,  iWanderHi:     0.2,
    iSpeedLo:      3.0,  iSpeedHi:      7.0,
    iAccelLo:      4.0,  iAccelHi:     10.0,
    iStamLo:      40.0,  iStamHi:      80.0,
    iEndurLo:      0.05, iEndurHi:      0.3,
  });

  // ── Reactive readouts (best agent genome) ─────────────────────────────────
  const generationRef  = ref(0);
  const bestFitRef     = ref(0);
  const reachedRef     = ref(0);
  const bestSightRef   = ref("");
  const bestTurnRef    = ref(0);
  const bestSpeedRef   = ref(0);
  const bestStamRef    = ref("");
  const bestDrainRef   = ref(0);
  const bestFit = ref(0);
  const avgFit  = ref(0);

  // ── Obstacle field (pillars + walls) ─────────────────────────────────────
  const pillarX = new Float32Array(MaxPillars);
  const pillarZ = new Float32Array(MaxPillars);
  const pillarR = new Float32Array(MaxPillars);
  let nPillars = 0;

  // Wall: center, Y-axis rotation angle, half-length. Thickness is constant.
  const wallCx    = new Float32Array(MaxWalls);
  const wallCz    = new Float32Array(MaxWalls);
  const wallAngle = new Float32Array(MaxWalls);  // rotation around Y in radians
  const wallHLen  = new Float32Array(MaxWalls);  // half-length
  let nWalls = 0;

  const MidZ = StartZ + 8;
  const MidLen = GoalZ - StartZ - 16;

  function buildField() {
    nPillars = 0;
    nWalls = 0;
    const total = params.obstacles;
    // ~60% pillars, ~40% walls, both capped by their maximums.
    const wantPillars = Math.min(Math.round(total * 0.60), MaxPillars);
    const wantWalls   = Math.min(total - wantPillars, MaxWalls);

    for (let i = 0; i < wantPillars; i++) {
      pillarX[i] = (Math.random() - 0.5) * (ArenaW - 2) * 2;
      pillarZ[i] = MidZ + Math.random() * MidLen;
      pillarR[i] = 0.55 + Math.random() * 1.15;
      nPillars++;
    }
    for (let i = 0; i < wantWalls; i++) {
      wallCx[i]    = (Math.random() - 0.5) * (ArenaW - 4) * 2;
      wallCz[i]    = MidZ + Math.random() * MidLen;
      wallAngle[i] = Math.random() * Math.PI;   // random rotation 0–180°
      wallHLen[i]  = 2 + Math.random() * 5;     // half-length 2–7 units
      nWalls++;
    }
    writeObstacles();
  }

  function writeObstacles() {
    // Pillars.
    for (let i = 0; i < nPillars; i++) {
      const r = pillarR[i];
      pillarBuf.writeTRS(i, pillarX[i], 0, pillarZ[i], r * 2, 7, r * 2);
      pillarBuf.writeColor(i, 0.28, 0.32, 0.42);
    }
    pillarBuf.commit(pillarMesh, nPillars);

    // Walls — unit box rotated around Y by wallAngle[i], scaled to [len, h, thick].
    for (let i = 0; i < nWalls; i++) {
      const a = wallAngle[i];
      const ha = a * 0.5;
      // Y-axis quaternion: [0, sin(a/2), 0, cos(a/2)]
      wallBuf.writeTransform(i, wallCx[i], 0, wallCz[i],
        0, Math.sin(ha), 0, Math.cos(ha),
        wallHLen[i] * 2, WallH, WallThick * 2);
      wallBuf.writeColor(i, 0.55, 0.32, 0.22);
    }
    wallBuf.commit(wallMesh, nWalls);
  }

  // ── Agent state ───────────────────────────────────────────────────────────
  const px = new Float32Array(MaxPop), pz = new Float32Array(MaxPop);
  const hdg = new Float32Array(MaxPop);          // heading angle in radians
  const fitness = new Float32Array(MaxPop);
  const closestDist = new Float32Array(MaxPop);
  const alive = new Uint8Array(MaxPop);
  const reached = new Uint8Array(MaxPop);
  const curSpd = new Float32Array(MaxPop);   // current speed per agent (evolves via accel gene)
  const stam   = new Float32Array(MaxPop);   // remaining stamina this episode
  const genomes: Genome[] = [];

  // Circular trail buffer.
  const trailPX = new Float32Array(MaxPop * TrailPts);
  const trailPZ = new Float32Array(MaxPop * TrailPts);
  const trailHead = new Int32Array(MaxPop);
  const trailCount = new Int32Array(MaxPop);

  const ghostX = new Float32Array(GhostPts);
  const ghostZ = new Float32Array(GhostPts);
  let nGhost = 0;

  // ── Genome helpers ────────────────────────────────────────────────────────
  // Sample uniformly from [lo, hi], then clamp to the current system limits.
  // GeneLimits is only used as the outer bound for the system-limit sliders.
  function randGene(lo: number, hi: number, slo: number, shi: number): number {
    const lo2 = Math.min(lo, hi);
    const hi2 = Math.max(lo, hi);
    return clamp(lo2 + Math.random() * (hi2 - lo2), slo, shi);
  }

  function randGenome(): Genome {
    const p = params;
    return {
      seek:        randGene(p.iSeekLo,   p.iSeekHi,   p.sSeekLo,   p.sSeekHi),
      visionArc:   randGene(p.iArcLo,    p.iArcHi,    p.sArcLo,    p.sArcHi),
      visionRange: randGene(p.iRangeLo,  p.iRangeHi,  p.sRangeLo,  p.sRangeHi),
      rayCount:    randGene(p.iRaysLo,   p.iRaysHi,   p.sRaysLo,   p.sRaysHi),
      turnRate:    randGene(p.iTurnLo,   p.iTurnHi,   p.sTurnLo,   p.sTurnHi),
      avoidWeight: randGene(p.iAvoidLo,  p.iAvoidHi,  p.sAvoidLo,  p.sAvoidHi),
      wander:      randGene(p.iWanderLo, p.iWanderHi, p.sWanderLo, p.sWanderHi),
      maxSpeed:    randGene(p.iSpeedLo,  p.iSpeedHi,  p.sSpeedLo,  p.sSpeedHi),
      accel:       randGene(p.iAccelLo,  p.iAccelHi,  p.sAccelLo,  p.sAccelHi),
      stamina:     randGene(p.iStamLo,   p.iStamHi,   p.sStamLo,   p.sStamHi),
      endurance:   randGene(p.iEndurLo,  p.iEndurHi,  p.sEndurLo,  p.sEndurHi),
    };
  }

  function uniformCross(a: Genome, b: Genome): Genome {
    const pick = <K extends keyof Genome>(k: K) => (Math.random() < 0.5 ? a[k] : b[k]);
    return {
      seek:        pick("seek"),
      visionArc:   pick("visionArc"),
      visionRange: pick("visionRange"),
      rayCount:    pick("rayCount"),
      turnRate:    pick("turnRate"),
      avoidWeight: pick("avoidWeight"),
      wander:      pick("wander"),
      maxSpeed:    pick("maxSpeed"),
      accel:       pick("accel"),
      stamina:     pick("stamina"),
      endurance:   pick("endurance"),
    };
  }

  function mutate(g: Genome): Genome {
    const rate = params.mutRate, str = params.mutStrength;
    const j = (scale: number) => Math.random() < rate ? (Math.random() - 0.5) * str * scale : 0;
    const p = params;
    return {
      seek:        clamp(g.seek        + j(2),    p.sSeekLo,   p.sSeekHi),
      visionArc:   clamp(g.visionArc   + j(50),   p.sArcLo,    p.sArcHi),
      visionRange: clamp(g.visionRange + j(10),   p.sRangeLo,  p.sRangeHi),
      rayCount:    clamp(g.rayCount    + j(5),    p.sRaysLo,   p.sRaysHi),
      turnRate:    clamp(g.turnRate    + j(4),    p.sTurnLo,   p.sTurnHi),
      avoidWeight: clamp(g.avoidWeight + j(2),    p.sAvoidLo,  p.sAvoidHi),
      wander:      clamp(g.wander      + j(0.3),  p.sWanderLo, p.sWanderHi),
      maxSpeed:    clamp(g.maxSpeed    + j(5),    p.sSpeedLo,  p.sSpeedHi),
      accel:       clamp(g.accel       + j(10),   p.sAccelLo,  p.sAccelHi),
      stamina:     clamp(g.stamina     + j(60),   p.sStamLo,   p.sStamHi),
      endurance:   clamp(g.endurance   + j(0.2),  p.sEndurLo,  p.sEndurHi),
    };
  }

  function tournamentSelect(pool: number[], k = 3): number {
    let best = pool[Math.floor(Math.random() * pool.length)];
    for (let t = 1; t < k; t++) {
      const c = pool[Math.floor(Math.random() * pool.length)];
      if (fitness[c] > fitness[best]) best = c;
    }
    return best;
  }

  // ── Ray casting ───────────────────────────────────────────────────────────
  // Returns the hit distance along the ray, or `maxDist` if nothing is hit.
  // Tests against: pillar cylinders + rotated wall OBBs + 4 arena planes.
  function castRay(ox: number, oz: number, rdx: number, rdz: number, maxDist: number): number {
    let minT = maxDist;

    // Pillar intersections (ray vs. circle, with AgentR offset).
    for (let p = 0; p < nPillars; p++) {
      const ex = pillarX[p] - ox, ez = pillarZ[p] - oz;
      const t = ex * rdx + ez * rdz;
      if (t < 0) continue;
      const perpX = ex - t * rdx, perpZ = ez - t * rdz;
      const perpD2 = perpX * perpX + perpZ * perpZ;
      const r = pillarR[p] + AgentR;
      if (perpD2 > r * r) continue;
      const hitT = t - Math.sqrt(r * r - perpD2);
      if (hitT > 0 && hitT < minT) minT = hitT;
    }

    // Wall OBB intersections (slab method in wall-local frame).
    for (let w = 0; w < nWalls; w++) {
      const ca = Math.cos(-wallAngle[w]), sa = Math.sin(-wallAngle[w]);
      const lox = (ox - wallCx[w]) * ca - (oz - wallCz[w]) * sa;
      const loz = (ox - wallCx[w]) * sa + (oz - wallCz[w]) * ca;
      const ldx = rdx * ca - rdz * sa;
      const ldz = rdx * sa + rdz * ca;
      const hl = wallHLen[w] + AgentR;
      const ht = WallThick + AgentR;
      let tMin = 0, tMax = minT;
      if (Math.abs(ldx) < 1e-7) { if (lox < -hl || lox > hl) continue; }
      else {
        const inv = 1 / ldx;
        const t1 = (-hl - lox) * inv, t2 = (hl - lox) * inv;
        tMin = Math.max(tMin, Math.min(t1, t2));
        tMax = Math.min(tMax, Math.max(t1, t2));
      }
      if (Math.abs(ldz) < 1e-7) { if (loz < -ht || loz > ht) continue; }
      else {
        const inv = 1 / ldz;
        const t1 = (-ht - loz) * inv, t2 = (ht - loz) * inv;
        tMin = Math.max(tMin, Math.min(t1, t2));
        tMax = Math.min(tMax, Math.max(t1, t2));
      }
      if (tMax < tMin || tMax < 0) continue;
      const hit = tMin >= 0 ? tMin : tMax;
      if (hit < minT) minT = hit;
    }

    // Arena boundary planes — left, right, and start (-Z) walls only.
    // The goal-side (+Z) wall is intentionally excluded: agents should approach
    // the goal freely; sensing that wall causes avoidance that turns them away
    // just before they can enter the goal radius.
    if (rdx > 0.001)  { const t = (ArenaW - ox) / rdx;  if (t > 0 && t < minT) minT = t; }
    if (rdx < -0.001) { const t = (-ArenaW - ox) / rdx; if (t > 0 && t < minT) minT = t; }
    if (rdz < -0.001) { const t = (-ArenaD - oz) / rdz; if (t > 0 && t < minT) minT = t; }

    return minT;
  }

  // Closest distance from point (ax, az) to wall-segment i (ignoring Y).
  function agentToWall(ax: number, az: number, w: number): number {
    const ca = Math.cos(wallAngle[w]), sa = Math.sin(wallAngle[w]);
    const ex = ax - wallCx[w], ez = az - wallCz[w];
    const t = clamp(ex * ca + ez * sa, -wallHLen[w], wallHLen[w]);
    const cx2 = wallCx[w] + t * ca, cz2 = wallCz[w] + t * sa;
    const dx = ax - cx2, dz = az - cz2;
    return Math.sqrt(dx * dx + dz * dz);
  }

  // ── Agent reset ───────────────────────────────────────────────────────────
  function resetAgents() {
    const n = params.population;
    // Goal direction is roughly +Z; start heading with a small spread.
    const goalDir = Math.PI * 0.5;
    for (let i = 0; i < n; i++) {
      px[i]  = (Math.random() - 0.5) * ArenaW * 1.6;
      pz[i]  = StartZ + (Math.random() - 0.5) * 5;
      hdg[i] = goalDir + (Math.random() - 0.5) * Math.PI * 0.5;
      fitness[i] = 0;
      closestDist[i] = InitialDist;
      alive[i] = 1;
      reached[i] = 0;
      curSpd[i] = genomes[i].maxSpeed * 0.2;   // start at 20% of top speed
      stam[i]   = genomes[i].stamina;          // full tank
      trailHead[i] = 0;
      trailCount[i] = 0;
    }
  }

  function spawnAll() {
    genomes.length = 0;
    for (let i = 0; i < MaxPop; i++) genomes.push(randGenome());
    nGhost = 0;
    resetAgents();
  }

  // ── GA step ───────────────────────────────────────────────────────────────
  function evolveGeneration() {
    const n = params.population;

    // Identify best agent → record its trail as ghost.
    let bestI = 0;
    for (let i = 1; i < n; i++) if (fitness[i] > fitness[bestI]) bestI = i;

    bestFit.value = Math.round(fitness[bestI] * 10) * 0.1;
    bestFitRef.value = bestFit.value;
    let total = 0;
    for (let i = 0; i < n; i++) total += fitness[i];
    avgFit.value = Math.round(total / n * 10) * 0.1;

    const cnt = trailCount[bestI], head = trailHead[bestI];
    nGhost = cnt;
    const base = bestI * TrailPts;
    for (let t = 0; t < cnt; t++) {
      const slot = (head - cnt + t + TrailPts) % TrailPts;
      ghostX[t] = trailPX[base + slot];
      ghostZ[t] = trailPZ[base + slot];
    }

    // Evolution.
    const elite = Math.max(1, Math.round(n * params.elitism));
    const sorted = Array.from({ length: n }, (_, i) => i).sort((a, b) => fitness[b] - fitness[a]);
    const eliteGenomes = sorted.slice(0, elite).map(i => ({ ...genomes[i] }));
    const parentPool = sorted.slice(0, Math.max(elite, Math.round(n * 0.5)));

    for (let i = 0; i < elite; i++) genomes[sorted[i]] = eliteGenomes[i];
    for (let i = elite; i < n; i++) {
      const pa = genomes[tournamentSelect(parentPool)];
      const pb = genomes[tournamentSelect(parentPool)];
      genomes[sorted[i]] = mutate(uniformCross(pa, pb));
    }

    generationRef.value++;
    let reachedCount = 0;
    for (let i = 0; i < n; i++) if (reached[i]) reachedCount++;
    reachedRef.value = reachedCount;
    resetAgents();
  }

  buildField();
  spawnAll();

  let episodeTimer = 0;
  let trailSampleTimer = 0;
  // After the episode ends, hold the final frame for this many ms before
  // evolving so the user can see which agents reached and which didn't.
  const HoldMs = 900;
  let holdTimer = -1;   // -1 = not in hold; ≥0 = counting toward next evolve

  function update(dt: number) {
    const n = params.population;
    const ds = dt * 0.001;
    episodeTimer += dt;
    trailSampleTimer += dt;
    const sampleTrail = trailSampleTimer > 150; // ~6–7 points/sec
    if (sampleTrail) trailSampleTimer = 0;

    // ── Steer alive agents (skip during end-of-episode hold) ───────────────
    let anyAlive = false;
    if (holdTimer >= 0) { /* in hold — skip steering, render frozen state */ }
    for (let i = 0; i < n && holdTimer < 0; i++) {
      if (!alive[i]) continue;
      anyAlive = true;

      const g = genomes[i];

      // Performance multiplier: lerp from endurance (at zero stamina) to 1.0 (full).
      // High endurance = near-full performance even when tired.
      // Low endurance = rapid capability loss as the tank drains.
      const stamFrac = clamp(stam[i] / g.stamina, 0, 1);
      const perfMult = g.endurance + (1 - g.endurance) * stamFrac;

      // Forward direction from current heading.
      const fdx = Math.cos(hdg[i]), fdz = Math.sin(hdg[i]);

      // Sensor rays → accumulate avoidance force.
      const nRays = Math.round(clamp(g.rayCount, 3, MaxRays));
      const arcRad = g.visionArc * (Math.PI / 180);
      const arcStep = nRays > 1 ? arcRad * 2 / (nRays - 1) : 0;
      const arcStart = -arcRad;

      let avFx = 0, avFz = 0;
      for (let r = 0; r < nRays; r++) {
        const angle = hdg[i] + arcStart + r * arcStep;
        const rdx = Math.cos(angle), rdz = Math.sin(angle);
        const hitT = castRay(px[i], pz[i], rdx, rdz, g.visionRange);
        if (hitT < g.visionRange) {
          const signal = 1 - hitT / g.visionRange;
          // Repel backward along the ray direction (away from hit).
          avFx -= rdx * signal * g.avoidWeight;
          avFz -= rdz * signal * g.avoidWeight;
        }
      }

      // Seek force (toward goal).
      const gDx = 0 - px[i], gDz = GoalZ - pz[i];
      const gDist = Math.sqrt(gDx * gDx + gDz * gDz);
      const skFx = (gDx / (gDist + 0.001)) * g.seek;
      const skFz = (gDz / (gDist + 0.001)) * g.seek;

      // Wander: rotate current heading by random jitter.
      const wanderAngle = hdg[i] + (Math.random() - 0.5) * g.wander * Math.PI;
      const wnFx = Math.cos(wanderAngle) * 0.25;
      const wnFz = Math.sin(wanderAngle) * 0.25;

      // Combined desired force → desired heading.
      const totalFx = skFx + avFx + wnFx;
      const totalFz = skFz + avFz + wnFz;
      const fMag = Math.sqrt(totalFx * totalFx + totalFz * totalFz);

      // Turn rate is scaled by perfMult — tired agents steer more sluggishly.
      let dHdg = 0;
      const effectiveTurnRate = g.turnRate * perfMult;
      if (fMag > 0.001) {
        const desiredHdg = Math.atan2(totalFz, totalFx);
        dHdg = desiredHdg - hdg[i];
        while (dHdg >  Math.PI) dHdg -= 2 * Math.PI;
        while (dHdg < -Math.PI) dHdg += 2 * Math.PI;
        const maxDHdg = effectiveTurnRate * ds;
        dHdg = clamp(dHdg, -maxDHdg, maxDHdg);
        hdg[i] += dHdg;
      }

      // Accelerate toward maxSpeed (also scaled by perfMult). Sharp turns bleed
      // off current speed — forces agents to plan smooth arcs under fatigue.
      const maxDHdg2 = effectiveTurnRate * ds;
      const turnFrac = maxDHdg2 > 0 ? Math.abs(dHdg) / maxDHdg2 : 0; // [0, 1]
      curSpd[i] += g.accel * perfMult * ds;                // accel also scales with perf
      curSpd[i] -= curSpd[i] * turnFrac * 0.35 * ds;      // drag from turning
      curSpd[i] = clamp(curSpd[i], 0, g.maxSpeed * perfMult);

      const newFdx = Math.cos(hdg[i]), newFdz = Math.sin(hdg[i]);
      px[i] = clamp(px[i] + newFdx * curSpd[i] * ds, -ArenaW, ArenaW);
      pz[i] = clamp(pz[i] + newFdz * curSpd[i] * ds, -ArenaD, ArenaD);

      // Recompute distance after movement so fast agents can't overshoot the goal.
      const postDx = 0 - px[i], postDz = GoalZ - pz[i];
      const postDist = Math.sqrt(postDx * postDx + postDz * postDz);
      if (postDist < closestDist[i]) closestDist[i] = postDist;

      // Goal check comes FIRST — entering the goal sphere wins even if the back
      // wall is also reached on the same frame.
      if (postDist < GoalR) {
        alive[i] = 0;
        reached[i] = 1;
        px[i] = clamp(px[i], -GoalR * 0.5, GoalR * 0.5);
        pz[i] = GoalZ;
        const fraction = Math.max(0, 1 - episodeTimer / (params.lifetime * 1000));
        fitness[i] = InitialDist * 3 * (1 + 2 * fraction);
        continue;
      }

      // Collision: arena boundary, pillars, or walls → freeze (dead).
      let collide = Math.abs(px[i]) >= ArenaW - AgentR || Math.abs(pz[i]) >= ArenaD - AgentR;
      if (!collide) {
        for (let p = 0; p < nPillars; p++) {
          const dx = px[i] - pillarX[p], dz = pz[i] - pillarZ[p];
          if (dx * dx + dz * dz < (pillarR[p] + AgentR) ** 2) { collide = true; break; }
        }
      }
      if (!collide) {
        for (let w = 0; w < nWalls; w++) {
          if (agentToWall(px[i], pz[i], w) < WallThick + AgentR) { collide = true; break; }
        }
      }
      if (collide) {
        alive[i] = 0;
        // Crashed: half-credit for progress, so crashing always loses to surviving.
        fitness[i] = Math.max(0, InitialDist - closestDist[i]) * 0.5;
        continue;
      }

      // Stamina drain (per second):
      //  • speed component  — faster movement burns more energy
      //  • vision component — wider arc × more rays = more sensory processing cost
      //  • flat base cost   — just for existing
      stam[i] -= (curSpd[i] * SpeedDrain
                + g.visionArc * nRays * VisionDrain
                + BaseDrain) * ds;
      if (stam[i] <= 0) {
        // Exhausted: same partial credit as a crash.
        alive[i] = 0;
        fitness[i] = Math.max(0, InitialDist - closestDist[i]) * 0.5;
        continue;
      }

      // Accumulate progress fitness.
      fitness[i] = Math.max(fitness[i], InitialDist - postDist);

      // Trail sample.
      if (sampleTrail) {
        const base = i * TrailPts;
        const slot = trailHead[i];
        trailPX[base + slot] = px[i];
        trailPZ[base + slot] = pz[i];
        trailHead[i] = (slot + 1) % TrailPts;
        if (trailCount[i] < TrailPts) trailCount[i]++;
      }
    }

    // Episode ends when all agents are done or the timer expires.
    // Start the hold phase — keep rendering the final frozen state for HoldMs
    // so the user can see who made it before the next episode begins.
    if (holdTimer < 0 && (!anyAlive || episodeTimer >= params.lifetime * 1000)) {
      holdTimer = 0;
    }
    if (holdTimer >= 0) {
      holdTimer += dt;
      if (holdTimer >= HoldMs) {
        evolveGeneration();
        episodeTimer = 0;
        holdTimer = -1;
      }
      // Don't steer during the hold — rendering still runs below.
    }

    // ── Identify best current agent ─────────────────────────────────────────
    let bestI = 0;
    for (let i = 1; i < n; i++) if (fitness[i] > fitness[bestI]) bestI = i;
    const bestFitNow = fitness[bestI];

    // ── Write agents ────────────────────────────────────────────────────────
    for (let i = 0; i < n; i++) {
      const frac = bestFitNow > 0 ? Math.min(fitness[i] / (bestFitNow + 0.001), 1) : 0;
      const fwdX = Math.cos(hdg[i]), fwdZ = Math.sin(hdg[i]);
      const [qx, qy, qz, qw] = upToDirQuat(fwdX, 0, fwdZ);
      agentBuf.writeTransform(i, px[i], 0, pz[i], qx, qy, qz, qw, 1, 1, 1);
      if (i === bestI) {
        agentBuf.writeColor(i, 1, 0.9, 0.05);           // gold = current best
      } else if (!alive[i] && !reached[i]) {
        agentBuf.writeColor(i, 0.2, 0.2, 0.2);           // grey = dead/crashed
      } else if (reached[i]) {
        agentBuf.writeColor(i, 0.15, 1, 0.3);            // green = reached goal
      } else {
        // Colour: blue-cyan when healthy, shifting to orange-red as stamina depletes.
        const stamFrac = clamp(stam[i] / genomes[i].stamina, 0, 1);
        agentBuf.writeColor(i,
          0.05 + frac * 0.1 + (1 - stamFrac) * 0.85,   // red rises as stamina falls
          (0.3 + frac * 0.5) * stamFrac,                 // green fades
          (0.6 + frac * 0.35) * stamFrac                 // blue fades
        );
      }
    }
    agentBuf.commit(agentMesh, n);

    // ── Goal marker ─────────────────────────────────────────────────────────
    goalBuf.writeScale(0, 0, 0, GoalZ);
    goalBuf.commit(goalMesh, 1, false);

    // ── Sensor ray visualisation (best agent) ────────────────────────────────
    let nRayDraw = 0;
    if (params.showRays && bestI >= 0 && alive[bestI]) {
      const g = genomes[bestI];
      const nRays = Math.round(clamp(g.rayCount, 3, MaxRays));
      const arcRad = g.visionArc * (Math.PI / 180);
      const arcStep = nRays > 1 ? arcRad * 2 / (nRays - 1) : 0;
      for (let r = 0; r < nRays; r++) {
        const angle = hdg[bestI] - arcRad + r * arcStep;
        const rdx = Math.cos(angle), rdz = Math.sin(angle);
        const hitT = castRay(px[bestI], pz[bestI], rdx, rdz, g.visionRange);
        const drawLen = hitT;
        const cx = px[bestI] + rdx * drawLen * 0.5;
        const cz = pz[bestI] + rdz * drawLen * 0.5;
        const [qx, qy, qz, qw] = upToDirQuat(rdx, 0, rdz);
        rayBuf.writeTransform(r, cx, 0.3, cz, qx, qy, qz, qw, 0.07, drawLen, 0.07);
        const hit = hitT < g.visionRange;
        const sig = hit ? 1 - hitT / g.visionRange : 0;
        // Clear rays: dim cyan. Hit rays: bright red-orange, intensity ∝ signal.
        rayBuf.writeColor(r, hit ? 0.5 + sig * 0.5 : 0.1, hit ? 0.2 * (1 - sig) : 0.7, hit ? 0.05 : 0.55, 0.65 + sig * 0.3);
        nRayDraw++;
      }
      // Update best-agent genome readouts.
      bestSightRef.value = `${nRays} rays / ${Math.round(g.visionArc * 2)}°`;
      bestTurnRef.value  = Math.round(g.turnRate * 10) * 0.1;
      bestSpeedRef.value = Math.round(g.maxSpeed * 10) * 0.1;
      const stamPct = Math.round(stam[bestI] / g.stamina * 100);
      bestStamRef.value  = `${stamPct}% (pool ${Math.round(g.stamina)})`;
      // Live drain rate: how many stamina points per second this genome burns right now.
      const liveDrain = curSpd[bestI] * SpeedDrain
                      + g.visionArc * nRays * VisionDrain
                      + BaseDrain;
      bestDrainRef.value = Math.round(liveDrain * 10) * 0.1;
    }
    rayBuf.commit(rayMesh, nRayDraw);

    // ── Live trails ──────────────────────────────────────────────────────────
    const TrailThick = 0.15;
    let tc = 0;
    for (let i = 0; i < n && tc < MaxTrailSegs - TrailPts; i++) {
      const cnt = trailCount[i];
      if (cnt < 2) continue;
      const head = trailHead[i];
      const base = i * TrailPts;
      for (let t = 0; t < cnt - 1; t++) {
        const f = t / cnt;
        const slotA = (head - cnt + t     + TrailPts) % TrailPts;
        const slotB = (head - cnt + t + 1 + TrailPts) % TrailPts;
        const ax = trailPX[base + slotA], az = trailPZ[base + slotA];
        const bx = trailPX[base + slotB], bz = trailPZ[base + slotB];
        const ldx = bx - ax, ldz = bz - az;
        const ll = Math.sqrt(ldx * ldx + ldz * ldz);
        if (ll < 0.05) continue;
        const mx = (ax + bx) * 0.5, mz = (az + bz) * 0.5;
        const thickness = TrailThick * (f * 0.85 + 0.15);
        const [qx, qy, qz, qw] = upToDirQuat(ldx, 0, ldz);
        trailBuf.writeTransform(tc, mx, 0, mz, qx, qy, qz, qw, thickness, ll, thickness);
        trailBuf.writeColor(tc, 0.1, 0.4 + f * 0.4, 0.8, 0.12 + f * 0.35);
        tc++;
      }
    }
    trailBuf.commit(trailMesh, tc);

    // ── Ghost trail ──────────────────────────────────────────────────────────
    let gc = 0;
    for (let t = 0; t < nGhost - 1 && gc < MaxGhostSegs; t++) {
      const f = t / nGhost;
      const ldx = ghostX[t + 1] - ghostX[t], ldz = ghostZ[t + 1] - ghostZ[t];
      const ll = Math.sqrt(ldx * ldx + ldz * ldz);
      if (ll < 0.05) continue;
      const mx = (ghostX[t] + ghostX[t + 1]) * 0.5, mz = (ghostZ[t] + ghostZ[t + 1]) * 0.5;
      const thickness = 0.10 * (0.3 + f * 0.7);
      const [qx, qy, qz, qw] = upToDirQuat(ldx, 0, ldz);
      ghostBuf.writeTransform(gc, mx, 0.12, mz, qx, qy, qz, qw, thickness, ll, thickness);
      ghostBuf.writeColor(gc, 1, 0.75, 0.05, 0.2 + f * 0.55);
      gc++;
    }
    ghostBuf.commit(ghostMesh, gc);
  }

  function reset() {
    genomes.length = 0;
    for (let i = 0; i < MaxPop; i++) genomes.push(randGenome());
    nGhost = 0;
    generationRef.value = 0;
    bestFitRef.value = 0;
    reachedRef.value = 0;
    episodeTimer = 0;
    trailSampleTimer = 0;
    holdTimer = -1;
    resetAgents();
  }

  function newField() { buildField(); reset(); }

  return {
    params,
    schema: [
      { type: "slider",  key: "population",  label: "Population",        min: 20,   max: MaxPop,      step: 10   },
      { type: "slider",  key: "lifetime",    label: "Episode Time (s)",  min: 3,    max: 30,          step: 0.5  },
      { type: "slider",  key: "elitism",     label: "Elitism %",         min: 0.05, max: 0.5,         step: 0.05 },
      { type: "slider",  key: "mutRate",     label: "Mutation Rate",     min: 0,    max: 0.6,         step: 0.01 },
      { type: "slider",  key: "mutStrength", label: "Mutation Strength", min: 0.05, max: 1,           step: 0.05 },
      { type: "slider",  key: "obstacles",   label: "Obstacles",         min: 4,    max: MaxPillars,  step: 1    },
      { type: "toggle",  key: "showRays",    label: "Show Sensor Rays"  },
      { type: "button",  key: "_new",        label: "New Field + Reset", action: newField },

      // ── System Limits sub-panel ───────────────────────────────────────────
      { type: "section",  key: "secSys",   label: "Evolutionary Limits", collapsed: true },
      { type: "range", keyLo: "sSeekLo",   keyHi: "sSeekHi",   label: "Seek",     min: 0.1,  max: 5,    step: 0.05 },
      { type: "range", keyLo: "sArcLo",    keyHi: "sArcHi",    label: "Vision Arc°", min: 15, max: 160, step: 5    },
      { type: "range", keyLo: "sRangeLo",  keyHi: "sRangeHi",  label: "Vision Range", min: 2, max: 25, step: 0.5  },
      { type: "range", keyLo: "sRaysLo",   keyHi: "sRaysHi",   label: "Ray Count", min: 3,   max: 13,   step: 1    },
      { type: "range", keyLo: "sTurnLo",   keyHi: "sTurnHi",   label: "Turn Rate", min: 0.5, max: 12,   step: 0.25 },
      { type: "range", keyLo: "sAvoidLo",  keyHi: "sAvoidHi",  label: "Avoid Wt",  min: 0,   max: 6,    step: 0.1  },
      { type: "range", keyLo: "sWanderLo", keyHi: "sWanderHi", label: "Wander",    min: 0,   max: 1,    step: 0.05 },
      { type: "range", keyLo: "sSpeedLo",  keyHi: "sSpeedHi",  label: "Max Speed", min: 3,   max: 22,   step: 0.5  },
      { type: "range", keyLo: "sAccelLo",  keyHi: "sAccelHi",  label: "Accel",     min: 4,   max: 40,   step: 1    },
      { type: "range", keyLo: "sStamLo",   keyHi: "sStamHi",   label: "Stamina",   min: 40,  max: 300,  step: 5    },
      { type: "range", keyLo: "sEndurLo",  keyHi: "sEndurHi",  label: "Endurance", min: 0.05,max: 1,    step: 0.05 },

      // ── Initial Ranges sub-panel ──────────────────────────────────────────
      { type: "section",  key: "secInit",  label: "Initial Ranges (Gen 0)", collapsed: true },
      { type: "range", keyLo: "iSeekLo",   keyHi: "iSeekHi",   label: "Seek",     min: 0.1,  max: 5,    step: 0.05 },
      { type: "range", keyLo: "iArcLo",    keyHi: "iArcHi",    label: "Vision Arc°", min: 15, max: 160, step: 5    },
      { type: "range", keyLo: "iRangeLo",  keyHi: "iRangeHi",  label: "Vision Range", min: 2, max: 25, step: 0.5  },
      { type: "range", keyLo: "iRaysLo",   keyHi: "iRaysHi",   label: "Ray Count", min: 3,   max: 13,   step: 1    },
      { type: "range", keyLo: "iTurnLo",   keyHi: "iTurnHi",   label: "Turn Rate", min: 0.5, max: 12,   step: 0.25 },
      { type: "range", keyLo: "iAvoidLo",  keyHi: "iAvoidHi",  label: "Avoid Wt",  min: 0,   max: 6,    step: 0.1  },
      { type: "range", keyLo: "iWanderLo", keyHi: "iWanderHi", label: "Wander",    min: 0,   max: 1,    step: 0.05 },
      { type: "range", keyLo: "iSpeedLo",  keyHi: "iSpeedHi",  label: "Max Speed", min: 3,   max: 22,   step: 0.5  },
      { type: "range", keyLo: "iAccelLo",  keyHi: "iAccelHi",  label: "Accel",     min: 4,   max: 40,   step: 1    },
      { type: "range", keyLo: "iStamLo",   keyHi: "iStamHi",   label: "Stamina",   min: 40,  max: 300,  step: 5    },
      { type: "range", keyLo: "iEndurLo",  keyHi: "iEndurHi",  label: "Endurance", min: 0.05,max: 1,    step: 0.05 },
    ],
    readouts: {
      gen:      generationRef,
      best:     bestFitRef,
      reached:  reachedRef,
      sight:    bestSightRef,
      turnRate: bestTurnRef,
      speed:    bestSpeedRef,
      stamina:  bestStamRef,
      "drain/s": bestDrainRef,
    },
    seriesLabels: ["Best Fitness", "Avg Fitness"],
    seriesColors: ["#ffb700", "#00e5ff"],
    getSeries: () => [bestFit.value as number, avgFit.value as number],
    update,
    reset,
    detach,
  };
}
