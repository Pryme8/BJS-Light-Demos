import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "game-of-life",
  title: "Conway's Game of Life",
  description: "CPU or GPU cellular automata — scalable grid up to 512×512",
  component: defineAsyncComponent(() => import("./GameOfLife.vue")),
  source,
  badges: ["CPU", "GPU"],
});
