import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "reaction-terrain",
  title: "Reaction Terrain",
  description: "Gray-Scott chemical patterns sculpted by thermal erosion",
  component: defineAsyncComponent(() => import("./ReactionTerrain.vue")),
  source,
});
