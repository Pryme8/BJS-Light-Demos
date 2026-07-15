import { createRouter, createWebHashHistory } from "vue-router";
import { getDemoRegistry } from "@/demos/registry";
import "@/demos/index";

export const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    {
      path: "/",
      redirect: () => {
        const first = getDemoRegistry()[0];
        return first ? `/demo/${first.id}` : "/";
      },
    },
    ...getDemoRegistry().map((demo) => ({
      path: `/demo/${demo.id}`,
      component: demo.component,
    })),
  ],
});
