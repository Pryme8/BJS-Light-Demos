import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "lotka-volterra",
  title: "Lotka–Volterra",
  description: "Predator-prey population oscillations + RK4",
  component: defineAsyncComponent(() => import("./LotkaVolterra.vue")),
  source,
});
