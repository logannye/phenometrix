import { defineConfig } from "vite";

export default defineConfig({
  build: {
    target: "es2022",
    outDir: "dist-library",
    lib: { entry: "src/index.ts", formats: ["es"], fileName: "phenometrix-evidence" }
  }
});
