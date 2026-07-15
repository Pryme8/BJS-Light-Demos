import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "double-pendulum",
  title: "Double Pendulum",
  description: "CPU or GPU chaos fan — thousands of pendulums, butterfly effect",
  component: defineAsyncComponent(() => import("./DoublePendulum.vue")),
  source,
  badges: ["CPU", "GPU"],
});
