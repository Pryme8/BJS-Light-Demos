import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "reaction-diffusion",
  title: "Reaction–Diffusion",
  description: "Gray-Scott model: coral, stripes, spots, maze",
  component: defineAsyncComponent(() => import("./ReactionDiffusion.vue")),
  source,
});
