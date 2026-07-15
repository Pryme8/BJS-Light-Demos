import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "boids",
  title: "Boids Flocking",
  description: "CPU or GPU compute — scales to tens of thousands",
  component: defineAsyncComponent(() => import("./Boids.vue")),
  source,
  badges: ["CPU", "GPU"],
});
