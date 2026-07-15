const DefaultColors = [
  "#00e5ff", "#ff00cc", "#39ff14", "#ffb700",
  "#ff4466", "#8888ff", "#ff8800", "#00ff88",
];

export interface SeriesConfig {
  label: string;
  color?: string;
}

/**
 * Lightweight time-series graph drawn onto a 2D canvas context.
 * Maintains a circular buffer of history; call push() each sample
 * and draw() to render.
 */
export class TimeSeriesGraph {
  private readonly data: Float32Array[];
  private readonly maxSamples: number;
  private head = 0;
  private count = 0;
  private readonly seriesCount: number;
  private readonly colors: string[];
  private readonly labels: string[];

  constructor(series: SeriesConfig[], maxSamples = 300) {
    this.seriesCount = series.length;
    this.maxSamples = maxSamples;
    this.data = series.map(() => new Float32Array(maxSamples));
    this.colors = series.map((s, i) => s.color ?? DefaultColors[i % DefaultColors.length]);
    this.labels = series.map((s) => s.label);
  }

  push(values: number[]): void {
    for (let s = 0; s < this.seriesCount; s++) {
      this.data[s][this.head] = values[s] ?? 0;
    }
    this.head = (this.head + 1) % this.maxSamples;
    if (this.count < this.maxSamples) this.count++;
  }

  draw(ctx: CanvasRenderingContext2D, maxValue?: number): void {
    const { width, height } = ctx.canvas;
    if (this.count < 2) return;

    // Background
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "rgba(8,11,15,0.82)";
    ctx.fillRect(0, 0, width, height);

    // Border
    ctx.strokeStyle = "rgba(0,229,255,0.25)";
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, width - 1, height - 1);

    const pad = 4;
    const labelH = 14 * this.seriesCount + 4;
    const plotW = width - pad * 2;
    const plotH = height - pad * 2 - labelH;
    const plotY = pad;

    // Determine scale
    let peak = maxValue ?? 0;
    if (!maxValue) {
      for (let s = 0; s < this.seriesCount; s++) {
        for (let t = 0; t < this.count; t++) {
          const idx = (this.head - this.count + t + this.maxSamples) % this.maxSamples;
          if (this.data[s][idx] > peak) peak = this.data[s][idx];
        }
      }
    }
    if (peak === 0) peak = 1;

    // Grid line at 50%
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    const midY = plotY + plotH * 0.5;
    ctx.moveTo(pad, midY);
    ctx.lineTo(pad + plotW, midY);
    ctx.stroke();

    // Series lines
    for (let s = 0; s < this.seriesCount; s++) {
      ctx.beginPath();
      ctx.strokeStyle = this.colors[s];
      ctx.lineWidth = 1.5;
      let first = true;
      for (let t = 0; t < this.count; t++) {
        const idx = (this.head - this.count + t + this.maxSamples) % this.maxSamples;
        const v = this.data[s][idx];
        const x = pad + (t / (this.count - 1)) * plotW;
        const y = plotY + plotH - (v / peak) * plotH;
        if (first) { ctx.moveTo(x, y); first = false; }
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Labels + current values
    ctx.font = "11px 'JetBrains Mono', monospace";
    for (let s = 0; s < this.seriesCount; s++) {
      const last = this.data[s][(this.head - 1 + this.maxSamples) % this.maxSamples];
      const ly = height - labelH + s * 14 + 11;
      ctx.fillStyle = this.colors[s];
      ctx.fillRect(pad, ly - 7, 8, 8);
      ctx.fillStyle = "rgba(232,237,243,0.9)";
      ctx.fillText(`${this.labels[s]}: ${last.toFixed(0)}`, pad + 12, ly);
    }
  }
}
