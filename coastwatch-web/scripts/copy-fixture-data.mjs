// Copy the committed, deterministic fixture datasets to public/data/fixture{,-failed}/v1 so the app
// can be run and tested without network access: CW_DATA_BASE_URL=/data/fixture/v1
import { cpSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
for (const [src, name] of [
  [join(root, "tests", "fixture-data", "v1"), "fixture"],
  [join(root, "tests", "fixture-data-failed", "v1"), "fixture-failed"],
  // JSON artifacts actually published by the M2 pipeline (no M3 artifacts, no images)
  [join(root, "..", "pipeline", "tests", "fixtures", "compat", "m2"), "compat-m2"],
]) {
  const dest = join(root, "public", "data", name, "v1");
  rmSync(dest, { recursive: true, force: true });
  cpSync(src, dest, { recursive: true });
  console.log(`fixture data -> ${dest}`);
}
