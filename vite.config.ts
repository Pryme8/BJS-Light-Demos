import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { resolve } from "path";

// DigitalOcean App Platform injects the port to bind on via PORT.
const port = Number(process.env.PORT) || 8080;

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
  build: {
    target: "esnext",
  },
  // Used by `npm run start` (vite preview) when deployed as a Web Service.
  preview: {
    host: true,
    port,
    // Allow the platform-assigned hostname (e.g. *.ondigitalocean.app / custom domain).
    allowedHosts: true,
  },
  server: {
    host: true,
  },
});
