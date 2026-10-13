// Same approach as coastwatch-web/scripts/copy-maplibre-worker.mjs: MapLibre v6's ES-module worker
// (and its shared chunk) are served from public/ and pointed to with setWorkerUrl.
import { copyFileSync, mkdirSync } from "node:fs";
const src = "node_modules/maplibre-gl/dist", dest = "public/vendor/maplibre";
mkdirSync(dest, { recursive: true });
for (const f of ["maplibre-gl-worker.mjs", "maplibre-gl-shared.mjs"]) copyFileSync(`${src}/${f}`, `${dest}/${f}`);
