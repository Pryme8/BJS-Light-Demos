import { defineAsyncComponent } from "vue";
import { registerDemo } from "@/demos/registry";
import source from "./scene.ts?raw";

registerDemo({
  id: "basic-cube",
  title: "Basic Cube",
  description: "Box mesh · hemispheric light · arc-rotate camera",
  component: defineAsyncComponent(() => import("./BasicCube.vue")),
  source,
});
