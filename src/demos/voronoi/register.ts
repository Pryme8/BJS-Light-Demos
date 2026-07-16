import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "voronoi",
  title: "Voronoi Landscape",
  description: "CPU or GPU 3D Voronoi terrain — ridge, cone, mesa, noise · up to 512×512",
  badges: ["CPU", "GPU"],
  component: defineAsyncComponent(() => import("./Voronoi.vue")),
  source,
});
