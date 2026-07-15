import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "voronoi-forest",
  title: "Voronoi Forest",
  description: "Voronoi biomes × L-system species forests via thin instances",
  component: defineAsyncComponent(() => import("./VoronoiForest.vue")),
  source,
});
