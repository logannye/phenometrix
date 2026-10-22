import { cpSync, existsSync } from "node:fs";
import { relative, resolve } from "node:path";
import type { Plugin, ResolvedConfig } from "vite";

/** Keep local research artifacts available to dev servers but out of shipped builds. */
export function productionPublicAssets(): Plugin {
  let config: ResolvedConfig;
  return {
    name: "phenometrix-production-public-assets",
    apply: "build",
    config: () => ({ build: { copyPublicDir: false } }),
    configResolved(resolved) { config = resolved; },
    writeBundle(output) {
      const source = config.publicDir;
      if (!source || !existsSync(source)) return;
      cpSync(source, resolve(config.root, output.dir ?? config.build.outDir), {
        recursive: true,
        dereference: true,
        // Reject the directory itself so its entire subtree is skipped.
        filter: path => relative(source, path) !== ".local"
      });
    }
  };
}
