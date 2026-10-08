// Copy the committed, deterministic fixture datasets to public/data/fixture{,-failed}/v1 so the app
// can be run and tested without network access: CW_DATA_BASE_URL=/data/fixture/v1
import { cpSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
for (const [src, name] of [
  ["fixture-data", "fixture"],
  ["fixture-data-failed", "fixture-failed"],
]) {
  const dest = join(root, "public", "data", name, "v1");
  rmSync(dest, { recursive: true, force: true });
  cpSync(join(root, "tests", src, "v1"), dest, { recursive: true });
  console.log(`fixture data -> ${dest}`);
}
