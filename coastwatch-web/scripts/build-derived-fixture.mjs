// Deterministic test fixture for the derived-products index (lib/derived; the C→B contract in the
// v5 brief): tests/fixture-derived/v1/derived/. Synthetic values over Monterey Bay and San Francisco
// Bay, for end-to-end tests only; the published index carries the pipeline's own products and numbers.
//   node scripts/build-derived-fixture.mjs
import { gunzipSync, gzipSync } from "node:zlib";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const fx = join(root, "tests", "fixture-data", "v1");
// kept outside tests/fixture-data: CI regenerates that folder from the pipeline (cwp fixture), which
// does not write derived products yet; scripts/copy-fixture-data.mjs overlays this folder onto it
const out = join(root, "tests", "fixture-derived", "v1", "derived");
rmSync(out, { recursive: true, force: true });
const NODATA = 65535;
const SCALE = 9.155552842799158e-5; // log10 chlorophyll, as the satellite grids
const OFFSET = -3.0;

function writeGrid(dir, w, h, value, chunk = 512) {
  const present = [];
  for (let cr = 0; cr * chunk < h; cr++)
    for (let cc = 0; cc * chunk < w; cc++) {
      const cw = Math.min(chunk, w - cc * chunk);
      const ch = Math.min(chunk, h - cr * chunk);
      const buf = Buffer.alloc(cw * ch * 2);
      let any = false;
      for (let r = 0; r < ch; r++)
        for (let c = 0; c < cw; c++) {
          const v = value(cr * chunk + r, cc * chunk + c);
          const code = v == null ? NODATA : Math.max(0, Math.min(65534, Math.round(v)));
          if (code !== NODATA) any = true;
          buf.writeUInt16LE(code, (r * cw + c) * 2);
        }
      if (!any) continue;
      mkdirSync(join(out, dir), { recursive: true });
      writeFileSync(join(out, dir, `${cr}_${cc}.u16.gz`), gzipSync(buf, { level: 9 }));
      present.push(`${cr}_${cc}`);
    }
  return present;
}

// the observed OLCI latest view of the fixture: the estimate fills only where it has no value
const m = JSON.parse(readFileSync(join(fx, "manifest.json"), "utf8"));
const olci = m.layers.find((l) => l.layer_id === "olci300_chl_latest").grid;
const olciCodes = new Uint16Array(new Uint8Array(gunzipSync(readFileSync(join(fx, olci.chunks.url_template.replace("{row}", "0").replace("{col}", "0"))))).buffer);
const observed = (lat, lon) => {
  const r = Math.round((lat - olci.lat_first) / olci.lat_step);
  const c = Math.round((lon - olci.lon_first) / olci.lon_step);
  if (r < 0 || c < 0 || r >= olci.height || c >= olci.width) return false;
  return olciCodes[r * olci.width + c] !== NODATA;
};

const inSfBay = (lat, lon) => lat > 37.45 && lat < 38.12 && lon > -122.52 && lon < -122.0;
const field = (lat, lon) => 0.15 + 0.35 * Math.sin(lat * 23) * Math.cos(lon * 17) + (inSfBay(lat, lon) ? 0.45 : 0) + 0.25 * Math.exp(-((lon + 121.95) ** 2) / 0.02);
const code = (log10) => (log10 - OFFSET) / SCALE;

// estimate: 300 m, aligned with the OLCI grid, Monterey Bay to San Pablo Bay
const E = { lat_first: 38.20125, lat_step: -0.0025, lon_first: -123.10125, lon_step: 0.0025, width: 540, height: 720 };
const at = (g, r, c) => [g.lat_first + r * g.lat_step, g.lon_first + c * g.lon_step];
const estPresent = writeGrid("chl_estimate_latest/grid", E.width, E.height, (r, c) => {
  const [lat, lon] = at(E, r, c);
  return observed(lat, lon) ? null : code(field(lat, lon));
});
const UNC_SCALE = 1e-4;
const uncPresent = writeGrid("chl_estimate_latest/uncertainty", E.width, E.height, (r, c) => {
  const [lat, lon] = at(E, r, c);
  return observed(lat, lon) ? null : (inSfBay(lat, lon) ? 0.29 : 0.24) / UNC_SCALE;
});
// outlook: 1 km, the whole box
const O = { lat_first: 38.205, lat_step: -0.01, lon_first: -123.105, lon_step: 0.01, width: 136, height: 181 };
const outPresent = writeGrid("chl_outlook_8d/grid", O.width, O.height, (r, c) => code(field(...at(O, r, c)) - 0.05));
// sea-surface temperature: 1 km, °C, a cool upwelling band near shore and a front offshore
const SST_SCALE = 0.001;
const sstPresent = writeGrid("sst_latest/grid", O.width, O.height, (r, c) => {
  const [lat, lon] = at(O, r, c);
  const t = 11.2 + 3.6 / (1 + Math.exp(-(-(lon + 122.55) * 18))) + 0.6 * Math.sin(lat * 9) + (inSfBay(lat, lon) ? 2.4 : 0);
  return t / SST_SCALE;
});
const oUncPresent = writeGrid("chl_outlook_8d/uncertainty", O.width, O.height, () => 0.33 / UNC_SCALE);

const grid = (g, dir, present, scale = SCALE, offset = OFFSET, transform = "log10") => ({
  url: `derived/${dir}/{row}_{col}.u16.gz`,
  encoding: "uint16le+gzip",
  ...g,
  scale_factor: scale,
  add_offset: offset,
  nodata: NODATA,
  max_quantization_error: scale / 2,
  transform,
  chunks: { rows: 512, cols: 512, url_template: `derived/${dir}/{row}_{col}.u16.gz`, present },
});

const index = {
  version: 1,
  generated_at: "2026-10-12T15:00:00Z",
  layers: [
    {
      id: "chl_outlook_8d",
      kind: "chl_outlook",
      label: "Chlorophyll outlook, next 8 days",
      issued: "2026-10-12",
      valid_from: "2026-10-13",
      valid_to: "2026-10-20",
      method: "Fixture: climatology plus a damped anomaly from the last clear views (test data only).",
      basis: "Fixture: synthetic values over Monterey Bay and San Francisco Bay.",
      skill: { metric: "RMSE ln(mg m-3)", model: 0.754, climatology_ar1: 0.764, climatology: 0.85, persistence: 0.892, test_period: "2019 to 2021-06", n: 185000, note: "Fixture numbers for tests." },
      caveats: ["Fixture data for tests, not a product.", "One 8-day mean, not a day-by-day forecast.", "Chlorophyll is algae of every kind, not toxin."],
      units: "mg m-3",
      grid: grid(O, "chl_outlook_8d/grid", outPresent),
      uncertainty_grid: grid(O, "chl_outlook_8d/uncertainty", oUncPresent, UNC_SCALE, 0, "none"),
    },
    {
      id: "chl_estimate_latest",
      kind: "chl_estimate",
      label: "Chlorophyll, estimated where clouds or the coast hide it",
      issued: "2026-10-08",
      based_on_days: ["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06"],
      method: "Fixture: each pixel's recent median plus a same-day anomaly from clear pixels within 3 km (test data only).",
      validation: { metric: "RMSE log10(mg m-3)", by_band: { "0-0.6 km": 0.61, "0.6-3 km": 0.41, bays: 0.29 }, baseline: "last clear observation within 7 days", n: 5200 },
      caveats: ["Fixture data for tests, not a product.", "Least reliable in the first 600 m from shore and in turbid bays."],
      units: "mg m-3",
      grid: grid(E, "chl_estimate_latest/grid", estPresent),
      uncertainty_grid: grid(E, "chl_estimate_latest/uncertainty", uncPresent, UNC_SCALE, 0, "none"),
    },
    {
      id: "sst_latest",
      kind: "sst",
      label: "Sea surface temperature",
      units: "degC",
      observed_dates: ["2026-10-11", "2026-10-12"],
      source: "NOAA CoastWatch VIIRS SST (fixture)",
      resolution_km: 0.75,
      caveats: ["Fixture data for tests, not a product.", "The skin of the sea, not the temperature at depth."],
      grid: grid(O, "sst_latest/grid", sstPresent, SST_SCALE, 0, "none"),
    },
    {
      id: "charm_nearshore_extension",
      kind: "validation",
      skill: { metric: "AUC, pDA above 500 ng/L, nearest C-HARM cell vs pier samples", n: 1058, period: "2022-11 to 2026-10", auc: 0.3, brier_skill_score: -11.5 },
      has_demonstrated_skill: false,
      note: "Fixture: a station-by-month climatology does better.",
      readout: "Context only: checked against 1,058 CalHABMAP pier samples (2022-11 to 2026-10), the nearest NOAA cell had no demonstrated skill at the shore (pDA AUC 0.30).",
      notes: "fixture: extra keys are tolerated",
    },
  ],
};
mkdirSync(out, { recursive: true });
writeFileSync(join(out, "index.json"), JSON.stringify(index, null, 2) + "\n");
console.log(`derived fixture -> ${out}`);
