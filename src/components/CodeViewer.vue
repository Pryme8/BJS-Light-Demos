<script setup lang="ts">
import { ref, watch, onMounted } from "vue";
import { codeToHtml } from "shiki";

const props = defineProps<{ source: string }>();

const highlighted = ref<string>("");

async function highlight(code: string) {
  highlighted.value = await codeToHtml(code, {
    lang: "typescript",
    theme: "tokyo-night",
  });
}

onMounted(() => highlight(props.source));
watch(() => props.source, highlight);
</script>

<template>
  <div class="code-viewer">
    <div class="code-viewer__label font-mono">source</div>
    <div class="code-viewer__scroll">
      <div v-if="highlighted" class="code-viewer__html" v-html="highlighted" />
      <pre v-else class="code-viewer__fallback">{{ source }}</pre>
    </div>
  </div>
</template>

<style scoped>
.code-viewer {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--bg-base);
  border-left: 1px solid var(--border-subtle);
  overflow: hidden;
}

.code-viewer__label {
  padding: 8px 16px;
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border-subtle);
  background: var(--bg-surface);
  flex-shrink: 0;
}

.code-viewer__scroll {
  flex: 1;
  overflow: auto;
  padding: 16px;
}

.code-viewer__html :deep(pre) {
  background: transparent !important;
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1.65;
  white-space: pre;
  tab-size: 2;
}

.code-viewer__html :deep(code) {
  font-family: inherit;
}

.code-viewer__fallback {
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1.65;
  color: var(--text-secondary);
  white-space: pre;
}
</style>
