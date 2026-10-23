import { defineConfig } from "vite";
import { productionPublicAssets } from "./vite-public-assets.js";

export default defineConfig({
  base: "./",
  plugins: [productionPublicAssets()],
  worker: { format: "es" },
  build: {
    target: "es2022",
    outDir: "dist-embedded",
    lib: { entry: "src/embedded.ts", formats: ["es"], fileName: "phenometrix-embedded" }
  }
});
