import { defineConfig } from "vite";

// The build lands inside the Python package so `sp serve` ships it.
export default defineConfig({
  base: "./",
  build: {
    outDir: "../src/sonicproof/web",
    emptyOutDir: true,
    assetsDir: "assets",
    chunkSizeWarningLimit: 600,
  },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:7788" },
  },
});
