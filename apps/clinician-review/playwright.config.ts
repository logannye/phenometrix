import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e", workers: 1,
  use: { baseURL: "http://127.0.0.1:4175", channel: "chrome", headless: true },
  webServer: { command: "pnpm dev", url: "http://127.0.0.1:4175", reuseExistingServer: false }
});
