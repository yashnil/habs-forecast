import { copyFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const repo = join(root, "..");
const dest = join(root, "public", "data");
mkdirSync(dest, { recursive: true });

const pairs = [
  [join(repo, "dashboard", "data", "snapshot.json"), join(dest, "snapshot.json")],
  [join(repo, "dashboard", "data", "overlay.png"), join(dest, "overlay.png")],
  [join(repo, "dashboard", "fisheries_context.json"), join(dest, "fisheries_context.json")],
];

for (const [src, out] of pairs) {
  if (!existsSync(src)) {
    console.warn("skip (missing):", src);
    continue;
  }
  copyFileSync(src, out);
  console.log("copied", src, "->", out);
}
