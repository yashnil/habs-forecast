// Copy the committed, deterministic fixture datasets to public/data/fixture{,-failed}/v1 so the app
// can be run and tested without network access: CW_DATA_BASE_URL=/data/fixture/v1
import { chmodSync, cpSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
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
// the derived products (lib/derived) over the main fixture: kept apart because CI regenerates
// tests/fixture-data from the pipeline, which does not write derived/ on this branch
cpSync(join(root, "tests", "fixture-derived", "v1"), join(root, "public", "data", "fixture", "v1"), { recursive: true });
console.log("derived fixture -> public/data/fixture/v1/derived");

// v10: the main fixture with its newest radar hour blank over Monterey Bay (the staging data of Oct 21:
// the domain-wide coverage still looked usable, the region's own figure was 0). Derived here, never
// committed, so it follows tests/fixture-data when CI regenerates that folder.
{
  const src = join(root, "public", "data", "fixture", "v1");
  const dest = join(root, "public", "data", "fixture-hfrgap", "v1");
  rmSync(dest, { recursive: true, force: true });
  cpSync(src, dest, { recursive: true });
  // readable by the build whatever modes the copy left (on a shared Docker volume a copy can come out unreadable)
  const readable = (d) => {
    for (const n of readdirSync(d)) {
      const p = join(d, n);
      if (statSync(p).isDirectory()) {
        chmodSync(p, 0o755);
        readable(p);
      } else chmodSync(p, 0o644);
    }
  };
  readable(dest);
  const mf = join(dest, "manifest.json");
  const m = JSON.parse(readFileSync(mf, "utf8"));
  const hours = m.layers.filter((l) => l.group_id === "currents" && l.layer_id.startsWith("hfr2km_currents_2") && l.vectors).sort((a, b) => a.time.valid_time.localeCompare(b.time.valid_time));
  const newest = hours[hours.length - 1];
  for (const r of newest.coverage?.regions ?? []) if (r.region_id === "monterey_bay") Object.assign(r, { observed_fraction: 0, observed_km2: 0 });
  for (const gname of ["u_grid", "v_grid"]) {
    const g = newest.vectors[gname];
    const codes = new Uint16Array(g.width * g.height).fill(g.nodata ?? 65535);
    writeFileSync(join(dest, g.url), gzipSync(Buffer.from(codes.buffer)));
    chmodSync(join(dest, g.url), 0o644);
  }
  writeFileSync(mf, JSON.stringify(m));
  chmodSync(mf, 0o644);
  console.log(`radar-gap fixture -> ${dest} (${newest.layer_id} blank over Monterey Bay)`);
}
