<script setup lang="ts">
import { computed } from "vue";
import { useRoute } from "vue-router";
import { getDemoRegistry } from "@/demos/registry";
import { activeMode } from "@/lib/active-mode";

const route = useRoute();
const demos = computed(() => getDemoRegistry());

function isActive(id: string) {
  return route.path === `/demo/${id}`;
}

/**
 * Only the demo matching the current route actually has a running mode; for
 * every other row the badges are just static capability labels, so neither
 * is "lit". For the active row, light whichever badge matches the demo's
 * live `params.gpuMode` (published to `activeMode` by useSimulation).
 */
function badgeClass(demoId: string, badge: string): string {
  if (!isActive(demoId)) return "shell__nav-badge--dim";
  const isGpuBadge = badge === "GPU";
  return isGpuBadge === activeMode.gpuMode ? "shell__nav-badge--lit" : "shell__nav-badge--dim";
}
</script>

<template>
  <div class="shell">
    <aside class="shell__sidebar">
      <div class="shell__logo">
        <span class="shell__logo-bjs font-mono">BJS</span>
        <span class="shell__logo-light font-mono">Light</span>
        <span class="shell__logo-demos font-mono">Demos</span>
      </div>

      <nav class="shell__nav">
        <p class="shell__nav-label font-mono">demos</p>
        <RouterLink
          v-for="demo in demos"
          :key="demo.id"
          :to="`/demo/${demo.id}`"
          class="shell__nav-item"
          :class="{ 'shell__nav-item--active': isActive(demo.id) }"
        >
          <span class="shell__nav-indicator" />
          <span class="shell__nav-text">
            <span class="shell__nav-title-row">
              <span class="shell__nav-title">{{ demo.title }}</span>
              <span v-if="demo.badges?.length" class="shell__nav-badges">
                <span
                  v-for="badge in demo.badges"
                  :key="badge"
                  class="shell__nav-badge"
                  :class="badgeClass(demo.id, badge)"
                >{{ badge }}</span>
              </span>
            </span>
            <span class="shell__nav-desc" :title="demo.description">{{ demo.description }}</span>
          </span>
        </RouterLink>
      </nav>

      <footer class="shell__footer font-mono">
        <span>@babylonjs/lite</span>
        <span class="shell__footer-dot">·</span>
        <span>WebGPU</span>
      </footer>
    </aside>

    <main class="shell__main">
      <RouterView />
    </main>
  </div>
</template>

<style scoped>
.shell {
  display: flex;
  height: 100%;
  width: 100%;
  overflow: hidden;
}

/* ── Sidebar ── */
.shell__sidebar {
  width: var(--sidebar-w);
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  background: var(--bg-surface);
  border-right: 1px solid var(--border-subtle);
  overflow: hidden;
}

.shell__logo {
  display: flex;
  align-items: baseline;
  gap: 5px;
  padding: 18px 16px 14px;
  border-bottom: 1px solid var(--border-subtle);
  flex-shrink: 0;
}

.shell__logo-bjs {
  font-size: 16px;
  font-weight: 700;
  color: var(--cyan);
  text-shadow: var(--glow-cyan);
  letter-spacing: -0.02em;
}

.shell__logo-light {
  font-size: 14px;
  font-weight: 400;
  color: var(--text-primary);
  letter-spacing: 0.02em;
}

.shell__logo-demos {
  font-size: 11px;
  font-weight: 300;
  color: var(--text-muted);
  letter-spacing: 0.06em;
}

/* ── Nav ── */
.shell__nav {
  flex: 1;
  overflow-y: auto;
  padding: 12px 0;
}

.shell__nav-label {
  font-size: 9px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--text-muted);
  padding: 0 16px 8px;
}

.shell__nav-item {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 8px 16px;
  text-decoration: none;
  cursor: pointer;
  transition: background 0.12s;
  border-left: 2px solid transparent;
}

.shell__nav-item:hover {
  background: var(--bg-hover);
}

.shell__nav-item--active {
  background: var(--bg-active);
  border-left-color: var(--cyan);
}

.shell__nav-indicator {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--border-dim);
  flex-shrink: 0;
  margin-top: 6px;
  transition: background 0.12s, box-shadow 0.12s;
}

.shell__nav-item--active .shell__nav-indicator {
  background: var(--cyan);
  box-shadow: 0 0 6px rgba(0, 229, 255, 0.6);
}

.shell__nav-text {
  display: flex;
  flex-direction: column;
  gap: 2px;
  overflow: hidden;
}

.shell__nav-title-row {
  display: flex;
  align-items: center;
  gap: 5px;
  overflow: hidden;
}

.shell__nav-title {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  transition: color 0.12s;
}

.shell__nav-badges {
  display: flex;
  gap: 3px;
  flex-shrink: 0;
}

.shell__nav-badge {
  font-family: var(--font-mono);
  font-size: 8px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  border-radius: 2px;
  padding: 1px 4px;
  line-height: 1.4;
  flex-shrink: 0;
}

.shell__nav-badge--lit {
  color: var(--cyan);
  border: 1px solid var(--cyan-dim);
}

.shell__nav-badge--dim {
  color: var(--text-muted);
  border: 1px solid var(--border-dim);
}

.shell__nav-item--active .shell__nav-title {
  color: var(--cyan);
}

.shell__nav-desc {
  font-size: 11px;
  color: var(--text-muted);
  white-space: normal;
  line-height: 1.4;
  overflow-wrap: break-word;
}

/* ── Footer ── */
.shell__footer {
  font-size: 10px;
  color: var(--text-muted);
  padding: 10px 16px;
  border-top: 1px solid var(--border-subtle);
  display: flex;
  gap: 6px;
  flex-shrink: 0;
}

.shell__footer-dot {
  color: var(--border-bright);
}

/* ── Main ── */
.shell__main {
  flex: 1;
  overflow: hidden;
}
</style>
