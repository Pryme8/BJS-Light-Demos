import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "double-pendulum",
  title: "Double Pendulum",
  description: "Chaos theory visualized — butterfly effect fan-out",
  component: defineAsyncComponent(() => import("./DoublePendulum.vue")),
  source,
});
