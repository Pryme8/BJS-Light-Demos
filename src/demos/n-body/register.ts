import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "n-body",
  title: "N-Body Gravity",
  description: "CPU or GPU gravity — tiled O(n²), up to 8192 bodies with orbital trails",
  component: defineAsyncComponent(() => import("./NBody.vue")),
  source,
  badges: ["CPU", "GPU"],
});
