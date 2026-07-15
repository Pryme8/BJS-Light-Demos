import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "genetic-boids",
  title: "Evolving Navigators",
  description: "Genetic algorithm evolves steering genes to navigate a pillar field",
  component: defineAsyncComponent(() => import("./GeneticBoids.vue")),
  source,
});
