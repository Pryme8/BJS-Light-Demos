import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "lotka-volterra",
  title: "Lotka–Volterra",
  description: "CPU or GPU predator-prey dynamics — RK4 ODE + scaled agent swarms",
  component: defineAsyncComponent(() => import("./LotkaVolterra.vue")),
  source,
  badges: ["CPU", "GPU"],
});
