# BJS Light Demos

A collection of interactive simulations that run live in your browser — no install, no sign-up, just open and play.

**[→ Live site](https://pryme8.github.io/BJS-Light-Demos/)** *(or run locally below)*

---

## What's inside

Each demo is a self-contained simulation you can poke, pause, and tune with the sliders in the corner. Here's the lineup:

| Demo | What's happening |
|------|-----------------|
| **Boids Flocking** | Hundreds of little agents that flock like birds using just three simple rules: stay close, stay separated, match your neighbors' heading. Tweak the sliders and watch the swarm reshape itself in real time. |
| **Boids vs Predator** | Add predators to the flock. Prey reproduce, predators starve, populations boom and crash — a tiny working ecosystem. |
| **Evolving Navigators** | A colony of agents tries to navigate an obstacle field. Every generation the survivors pass their steering instincts to the next round. Watch dumb random walkers slowly become competent pilots. |
| **N-Body Gravity** | Stars pulling on each other, drawing orbital trails as they dance. Crank the body count up and watch the chaos. |
| **Conway's Game of Life** | The 1970 cellular automaton that convinced a generation of scientists that complexity can emerge from absurdly simple rules. Each cell lives or dies based only on its eight neighbors. |
| **Reaction-Diffusion** | Two chemicals spread and react across a grid, spontaneously growing coral branches, zebra stripes, leopard spots, and labyrinth mazes — depending only on two sliders. |
| **Double Pendulum** | Fan out a hundred pendulums that all start almost identically and watch them diverge completely within seconds. The textbook demo of chaos and sensitive dependence. |
| **Lotka-Volterra** | The classic predator-prey equations charted live: rabbit population explodes → fox population follows → rabbits collapse → foxes crash → repeat forever. |
| **L-System Plants** | A grammar that rewrites itself to grow 3D trees and branching structures. Change the rules and grow entirely different species. |
| **Voronoi Diagram** | Moving seeds carve up space into color-coded territories. Lloyd relaxation gradually makes the cells more uniform — the same algorithm phone towers and cell networks use to divide coverage. |

---

## Running locally

Requires **Chrome 113+** or **Edge 113+** (WebGPU).

```bash
npm install
npm run dev
```

Then open `http://localhost:5173`.

---

## Tech

Built with [Babylon.js Lite](https://github.com/babylonjs/lite) and [Vue 3](https://vuejs.org/). Renders via WebGPU — no WebGL fallback, which is why it's fast.

---

## License

MIT
