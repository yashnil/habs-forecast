// Sequential display ramp for the C-HARM probability classes on a dark bathymetric basemap.
//
// Designed in OKLCH (perceptually uniform lightness/chroma/hue), then validated in CIELAB:
// L* must rise monotonically from about 30 to about 85 across the 10 display classes, so the
// order of the classes is carried by lightness alone (hue and chroma only add discrimination).
// Hue sweeps from deep indigo (low probabilities, low chroma: they recede into the sea) through
// magenta to a warm amber (high probabilities, the brightest and warmest tone on the map).
//
//   node ramp.mjs            prints the stops and writes data/ramp.json
//   node ramp.mjs --round 1  the round-1 parameters (see CRITIQUE.md)
//
// Only colour-space maths lives here; no browser or map code.
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

// ---------------------------------------------------------------- colour maths
const clamp01 = (v) => Math.min(1, Math.max(0, v));
const lin2srgb = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);
const srgb2lin = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));

/** OKLab -> linear sRGB (Björn Ottosson's reference matrices). */
function oklabToLinear(L, a, b) {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;
  const l = l_ ** 3, m = m_ ** 3, s = s_ ** 3;
  return [
    +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
}
const inGamut = (rgb) => rgb.every((c) => c >= -1e-6 && c <= 1 + 1e-6);

/** OKLCH -> sRGB hex, reducing chroma (only) until the colour is inside the sRGB gamut. */
function oklchToHex(L, C, hDeg) {
  const h = (hDeg * Math.PI) / 180;
  let c = C;
  let rgb;
  for (let i = 0; i < 64; i++) {
    rgb = oklabToLinear(L, c * Math.cos(h), c * Math.sin(h));
    if (inGamut(rgb)) break;
    c *= 0.96;
  }
  const to255 = (v) => Math.round(clamp01(lin2srgb(clamp01(v))) * 255);
  const [r, g, b] = rgb.map(to255);
  return { hex: "#" + [r, g, b].map((v) => v.toString(16).padStart(2, "0")).join(""), chromaUsed: c };
}

/** sRGB hex -> CIELAB (D65, 2°), for the monotonic-L* check and for reporting. */
function hexToLab(hex) {
  const [r, g, b] = [1, 3, 5].map((i) => srgb2lin(parseInt(hex.slice(i, i + 2), 16) / 255));
  const X = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047;
  const Y = 0.2126729 * r + 0.7151522 * g + 0.072175 * b;
  const Z = (0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883;
  const f = (t) => (t > 216 / 24389 ? Math.cbrt(t) : (841 / 108) * t + 4 / 29);
  const [fx, fy, fz] = [X, Y, Z].map(f);
  return { L: 116 * fy - 16, a: 500 * (fx - fy), b: 200 * (fy - fz), Y };
}
const relLum = (hex) => hexToLab(hex).Y;
/** WCAG 2 contrast ratio between two sRGB hex colours. */
export function contrast(h1, h2) {
  const [a, b] = [relLum(h1), relLum(h2)].sort((x, y) => y - x);
  return (a + 0.05) / (b + 0.05);
}

// ---------------------------------------------------------------- the ramp
/**
 * Ten classes (lower bounds 0.0, 0.1, ... 0.9). Parameters per round of the critique:
 * L (OKLab lightness), C (chroma) and hue, each a piecewise-linear function of the class
 * position t in [0, 1].
 */
const ROUNDS = {
  // Round 0: first draft. Indigo 285° -> magenta 345° -> amber 75°. Low chroma at the bottom.
  0: { L: [0.4, 0.9], C: [[0, 0.045], [0.35, 0.11], [0.7, 0.14], [1, 0.14]], H: [[0, 285], [0.5, 345], [1, 435]] },
  // Round 1 (CRITIQUE.md): the 0.5-0.7 classes were a vivid coral red that read as an alarm
  // over the whole coast. Chroma is held to ~0.10 through the red hues (dusty rose, terracotta)
  // and rises again only in the amber top; the bottom is lifted a little (L 0.42) so the lowest
  // classes do not sink into the sea at 0.72 opacity.
  1: { L: [0.42, 0.89], C: [[0, 0.045], [0.3, 0.1], [0.55, 0.095], [0.8, 0.11], [1, 0.12]], H: [[0, 285], [0.45, 345], [0.7, 385], [1, 435]] },
  // Round 2: the 0.8 and 0.9 classes were two near-identical tans; the top gets a little more
  // chroma and a hue 5° further towards yellow so the highest class is a distinct amber. The
  // rest of the ramp is unchanged (round 1 fixed the alarm red).
  2: { L: [0.42, 0.89], C: [[0, 0.045], [0.3, 0.1], [0.55, 0.095], [0.8, 0.115], [1, 0.135]], H: [[0, 285], [0.45, 345], [0.7, 385], [1, 440]] },
};
// For a light basemap (T4) the same hue path with a lower lightness ceiling: the amber top of
// the dark ramp (L* 88) is as light as positron's sea (L* ~80) and disappears on it.
const LIGHT_VARIANT = { L: [0.4, 0.78] };
const POSITRON_SEA = "#c2c8ca";
const LATEST = Math.max(...Object.keys(ROUNDS).map(Number));

const pw = (pts, t) => {
  if (typeof pts[0] === "number") return pts[0] + (pts[1] - pts[0]) * t;
  for (let i = 0; i + 1 < pts.length; i++) {
    const [t0, v0] = pts[i], [t1, v1] = pts[i + 1];
    if (t <= t1) return v0 + ((v1 - v0) * (t - t0)) / (t1 - t0);
  }
  return pts[pts.length - 1][1];
};

export function buildRamp(round = LATEST, n = 10, variant = null) {
  const p = { ...ROUNDS[round], ...(variant === "light" ? LIGHT_VARIANT : {}) };
  const stops = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const L = pw(p.L, t), C = pw(p.C, t), H = pw(p.H, t) % 360;
    const { hex, chromaUsed } = oklchToHex(L, C, H);
    const lab = hexToLab(hex);
    stops.push({
      value: +(i / n).toFixed(1),
      color: hex,
      oklch: { L: +L.toFixed(3), C: +chromaUsed.toFixed(3), h: +H.toFixed(0) },
      lab_L: +lab.L.toFixed(1),
      contrast_white: +contrast(hex, "#ffffff").toFixed(2),
      contrast_ink: +contrast(hex, "#0b1d33").toFixed(2),
      contrast_dark_sea: +contrast(hex, "#0b1d33").toFixed(2),
      contrast_light_sea: +contrast(hex, POSITRON_SEA).toFixed(2),
    });
  }
  const monotonic = stops.every((s, i) => i === 0 || s.lab_L > stops[i - 1].lab_L);
  const minStep = Math.min(...stops.slice(1).map((s, i) => s.lab_L - stops[i].lab_L));
  return { id: `cw-probability-display-r${round}${variant ? "-" + variant : ""}`, round, variant, domain: [0, 1], interpolation: "step", stops, monotonic_Lstar: monotonic, min_delta_Lstar: +minStep.toFixed(1) };
}

const here = dirname(fileURLToPath(import.meta.url));
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const ri = process.argv.indexOf("--round");
  const round = ri > 0 ? +process.argv[ri + 1] : LATEST;
  const print = (ramp) => {
    console.log(`ramp ${ramp.id}: monotonic L* = ${ramp.monotonic_Lstar}, min ΔL* between classes = ${ramp.min_delta_Lstar}`);
    console.log("class  colour   OKLCH(L,C,h)        L*    contrast vs white / ink   vs dark sea / light sea");
    for (const s of ramp.stops)
      console.log(
        `${s.value.toFixed(1)}    ${s.color}  (${s.oklch.L.toFixed(2)}, ${s.oklch.C.toFixed(3)}, ${String(s.oklch.h).padStart(3)})  ${String(s.lab_L).padStart(5)}   ${String(s.contrast_white).padStart(5)} / ${String(s.contrast_ink).padStart(5)}   ${String(s.contrast_dark_sea).padStart(5)} / ${s.contrast_light_sea}`,
      );
    if (!ramp.monotonic_Lstar) {
      console.error("L* is not monotonic");
      process.exit(1);
    }
  };
  const ramp = buildRamp(round), light = buildRamp(round, 10, "light");
  // experiment (CRITIQUE.md round 2): 20 display classes of 5 % on the same curves
  const fine = buildRamp(round, 20);
  fine.id += "-fine20";
  print(ramp);
  print(light);
  console.log(`fine (20 classes): monotonic L* = ${fine.monotonic_Lstar}, min ΔL* ${fine.min_delta_Lstar}; stops ${fine.stops.map((s) => s.color).join(" ")}`);
  const outFile = join(here, "data", "ramp.json");
  writeFileSync(outFile, JSON.stringify({ ...ramp, light, fine, rounds: Object.fromEntries(Object.keys(ROUNDS).map((r) => [r, buildRamp(+r)])) }, null, 1));
  console.log("wrote", outFile);
}
