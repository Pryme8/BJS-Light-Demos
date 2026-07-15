import type { Component } from "vue";

export interface DemoMeta {
  id: string;
  title: string;
  description: string;
  component: Component;
  source: string;
  badges?: string[];
}

const _registry: DemoMeta[] = [];

export function registerDemo(meta: DemoMeta) {
  _registry.push(meta);
}

export function getDemoRegistry(): readonly DemoMeta[] {
  return _registry;
}
