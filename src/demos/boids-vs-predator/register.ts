import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "boids-vs-predator",
  title: "Boids vs Predator",
  description: "CPU or GPU ecosystem — prey flock & flee, predators hunt, true birth/death",
  badges: ["CPU", "GPU"],
  component: defineAsyncComponent(() => import("./BoidsVsPredator.vue")),
  source,
});
