import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "boids-vs-predator",
  title: "Boids vs Predator",
  description: "Agent-based ecosystem — food, hunger, starvation & reproduction",
  component: defineAsyncComponent(() => import("./BoidsVsPredator.vue")),
  source,
});
