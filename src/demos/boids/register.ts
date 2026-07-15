import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "boids",
  title: "Boids Flocking",
  description: "Emergent flocking via separation, alignment, cohesion",
  component: defineAsyncComponent(() => import("./Boids.vue")),
  source,
});
