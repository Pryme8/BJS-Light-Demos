import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "game-of-life",
  title: "Conway's Game of Life",
  description: "Cellular automata on a 3D instanced cube grid",
  component: defineAsyncComponent(() => import("./GameOfLife.vue")),
  source,
});
