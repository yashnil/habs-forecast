// MapLibre GL v6 ships its web worker as an ES module (plus a shared chunk) that the
// Next.js bundler does not emit. Copy both into public/ and point setWorkerUrl at them.
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "node_modules", "maplibre-gl", "dist");
const dest = join(root, "public", "vendor", "maplibre");
mkdirSync(dest, { recursive: true });
for (const f of ["maplibre-gl-worker.mjs", "maplibre-gl-shared.mjs"]) copyFileSync(join(src, f), join(dest, f));
console.log(`maplibre worker -> ${dest}`);
