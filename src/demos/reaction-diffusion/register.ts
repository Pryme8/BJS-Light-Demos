import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "reaction-diffusion",
  badges: ["CPU", "GPU"],
  title: "Reaction–Diffusion",
  description: "CPU or GPU Gray-Scott reaction-diffusion — scalable grid up to 4096×4096",
  component: defineAsyncComponent(() => import("./ReactionDiffusion.vue")),
  source,
});
