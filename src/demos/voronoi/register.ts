import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "voronoi",
  title: "Voronoi Diagram",
  description: "Territory coloring with animated seeds + Lloyd relaxation",
  component: defineAsyncComponent(() => import("./Voronoi.vue")),
  source,
});
