import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e", testMatch: ["voice-research.spec.ts", "voice-hybrid.spec.ts"], timeout: 30_000,
  fullyParallel: false, workers: 1,
  use: { baseURL: "http://127.0.0.1:4177", channel: "chrome", headless: true,
    launchOptions: { args: ["--autoplay-policy=no-user-gesture-required"] }, trace: "retain-on-failure" },
  webServer: { command: "pnpm exec vite --config vite.research.config.ts",
    url: "http://127.0.0.1:4177/research-voice.html", reuseExistingServer: false, timeout: 30_000 }
});
