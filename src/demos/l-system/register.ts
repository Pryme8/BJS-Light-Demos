import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "l-system",
  title: "L-System Plants",
  description: "Lindenmayer grammar → 3D branching structures",
  component: defineAsyncComponent(() => import("./LSystem.vue")),
  source,
});
