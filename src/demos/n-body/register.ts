import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "n-body",
  title: "N-Body Gravity",
  description: "Gravitational simulation with orbital trails",
  component: defineAsyncComponent(() => import("./NBody.vue")),
  source,
});
