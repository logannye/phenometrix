import { defineConfig } from "vite";
import { productionPublicAssets } from "./vite-public-assets.js";

export default defineConfig({
  base: "./",
  plugins: [productionPublicAssets()],
  server: {
    host: "127.0.0.1",
    port: 4173,
    strictPort: true
  },
  preview: {
    host: "127.0.0.1",
    port: 4173,
    strictPort: true
  }
});
