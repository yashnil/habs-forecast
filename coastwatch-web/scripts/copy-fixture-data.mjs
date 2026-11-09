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

// Pilot sidecars on the fixture clock. `cwp fixture` builds v1/fisheries-status from the repository's
// LIVE curated file (data/curated/fisheries-status, read by a person in November) and v1/biotoxin from
// a recorded CDPH PDF, while every e2e server runs with CW_FIXED_NOW = 2026-10-13T06:00Z
// (playwright.config.ts): the records would say "checked Nov 8" on a page whose clock is Oct 12. A
// record cannot have been read after the moment it is shown, so the dates that say WHEN NERITIC LOOKED
// (checked_at, reviewed_at, fetched_at, the CDPH file's as_of) are held to the fixture clock's Pacific
// day. What the agencies decided (effective and assessment dates, sample dates, every sentence) is not
// touched. Derived here and never committed, so it follows tests/fixture-data when CI regenerates it.
{
  const FIXTURE_NOW = process.env.CW_FIXED_NOW ?? "2026-10-13T06:00:00Z";
  const clockDay = new Intl.DateTimeFormat("en-CA", { timeZone: "America/Los_Angeles", year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date(FIXTURE_NOW));
  const LOOKED = new Set(["checked_at", "reviewed_at", "fetched_at"]);
  const hold = (v) => (typeof v !== "string" ? v : v.length <= 10 ? (v > clockDay ? clockDay : v) : Date.parse(v) > Date.parse(FIXTURE_NOW) ? FIXTURE_NOW : v);
  const walk = (o) => {
    if (Array.isArray(o)) return o.forEach(walk);
    if (!o || typeof o !== "object") return;
    for (const [k, v] of Object.entries(o)) {
      if (LOOKED.has(k)) o[k] = hold(v);
      else walk(v);
    }
  };
  let held = 0;
  for (const name of ["fixture", "fixture-failed"]) {
    for (const rel of ["fisheries-status/index.json", "biotoxin/index.json"]) {
      const f = join(root, "public", "data", name, "v1", rel);
      let j;
      try {
        j = JSON.parse(readFileSync(f, "utf8"));
      } catch {
        continue; // a dataset without the sidecar stays without it
      }
      const before = JSON.stringify(j);
      walk(j);
      if (j.source && typeof j.source.as_of === "string") j.source.as_of = hold(j.source.as_of);
      if (JSON.stringify(j) !== before) held++;
      writeFileSync(f, JSON.stringify(j));
    }
  }
  console.log(`pilot sidecars: read dates held to the fixture clock (${clockDay}) in ${held} file(s)`);
}

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
