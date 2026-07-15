import type { Ref } from "vue";

export interface SliderParam {
  type: "slider";
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
}

export interface ToggleParam {
  type: "toggle";
  key: string;
  label: string;
}

export interface SelectParam {
  type: "select";
  key: string;
  label: string;
  options: string[];
}

export interface ButtonParam {
  type: "button";
  key: string;
  label: string;
  action(): void;
}

export interface RangeParam {
  type: "range";
  keyLo: string;
  keyHi: string;
  label: string;
  min: number;
  max: number;
  step: number;
}

export interface SectionParam {
  type: "section";
  key: string;
  label: string;
  collapsed?: boolean;
}

export type ParamSchema = SliderParam | ToggleParam | SelectParam | ButtonParam | RangeParam | SectionParam;

export interface SimHandle {
  params: Record<string, unknown>;
  schema: ParamSchema[];
  readouts?: Record<string, Ref<number | string>>;
  seriesLabels?: string[];
  seriesColors?: string[];
  getSeries?(): number[];
  update(dt: number): void;
  reset(): void;
  detach?(): void;
}

export type BuildSimSceneFn = (
  engine: import("@babylonjs/lite").EngineContext,
  scene: import("@babylonjs/lite").SceneContext,
  canvas: HTMLCanvasElement
) => SimHandle | Promise<SimHandle>;

export interface SimControls {
  running: Ref<boolean>;
  fps: Ref<number>;
  handle: Ref<SimHandle | null>;
  play(): void;
  pause(): void;
  step(): void;
  reset(): void;
}
