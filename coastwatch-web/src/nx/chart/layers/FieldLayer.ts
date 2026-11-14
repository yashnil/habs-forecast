"use client";

import type { CustomLayerInterface, Map as MapLibreMap } from "maplibre-gl";
import type { ModeColours, RGBA } from "./colors";
import { CHART_FRAME } from "../style";
import { gridBounds, mercX, mercY, type FieldGrid } from "./grid";

/**
 * A published value grid drawn on the chart in the map's own WebGL context (no second context, no canvas over
 * the map). The raw uint16 codes go to the GPU as they were published and are decoded per pixel, so what is
 * drawn is exactly what a readout reads. Truth rules (E §2) carried here:
 * - nearest cell, no smoothing, for observations (T19: nothing is filled); a gap in an observed layer is the
 *   "no view" hatch over a paper veil, never a low colour (T19, T20);
 * - a model or outlook never shares a mark with an observation (T2): an outlook carries a hatch over its ramp,
 *   and cells where it equals the climatology carry dots and no colour ("no signal", E finding 2); C-HARM is
 *   drawn in cw-probability-classes-v1's ten display intervals on the quiet --prob ramp with a model hatch over
 *   it (T12: no alarm colour), lighter when the run is past (T3);
 * - cells the pipeline left out beside land (QC `land_adjacency`) are a flat pale band, never the cloud hatch
 *   (adjacent.ts marks them; the published codes are untouched);
 * - water a model or an outlook has no cell for is ruled "outside the model" to the chart's edge, never left as
 *   plain water (outside the footprint is unavailable, not zero);
 * - a temperature analysis is drawn as isotherms (lines), so it never borrows the chlorophyll ramp (T22);
 * - colours come from the mode's tokens (day, dusk, night); a new frame swaps the texture, a cut (T26).
 * Screen-space patterns (hatch, dots) are drawn in device pixels, so they read the same at every zoom.
 */
export type FieldStyle =
  /** observed, on the ramp; gaps hatched "no view" */
  | { kind: "ramp"; domain: [number, number]; scale: "log" | "linear"; opacity?: number }
  /** an outlook on the same ramp, hatched as a model; `aux` (the climatology) marks "no signal" cells with dots */
  | { kind: "outlook"; domain: [number, number]; scale: "log" | "linear"; opacity?: number }
  /** a probability 0..1 in cw-probability-classes-v1 on the --prob ramp, hatched as a model; `past` draws it lighter */
  | { kind: "classes"; past: boolean }
  /** lines of equal value every `step` (in `value * scale + offset`: °C drawn as whole °F), heavier every `index` steps */
  | { kind: "isolines"; step: number; index: number; scale?: number; offset?: number };

/** `adjacentCode`: the marker `markAdjacent` (adjacent.ts) wrote for cells left out beside land; null or absent: none */
export type FieldData = { grid: FieldGrid; codes: Uint16Array; aux?: { grid: FieldGrid; codes: Uint16Array } | null; adjacentCode?: number | null; outCode?: number | null; loadingCode?: number | null };

const VS = `#version 300 es
uniform mat4 u_matrix;
in vec2 a_pos;
out vec2 v_merc;
void main() { v_merc = a_pos; gl_Position = u_matrix * vec4(a_pos, 0.0, 1.0); }`;

export const FIELD_FS = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
uniform usampler2D u_data;
uniform usampler2D u_aux;
uniform int u_hasAux;
uniform vec4 u_geo;
uniform ivec2 u_size;
uniform float u_stride;
uniform vec3 u_dec;
uniform vec3 u_decAux;
uniform int u_takeLog;
uniform int u_variant;
uniform vec4 u_ramp[8];
uniform vec4 u_prob[10];
uniform vec2 u_domain;
uniform vec4 u_ink;
uniform vec4 u_ink2;
uniform vec4 u_hatch;
uniform vec4 u_paper;
uniform float u_dpr;
uniform float u_opacity;
uniform float u_step;
uniform float u_index;
uniform float u_past;
uniform float u_adj;
uniform float u_out;
uniform float u_loading;
uniform float u_fade;
uniform vec2 u_iso;
in vec2 v_merc;
out vec4 o;
const float PI = 3.141592653589793;

vec2 cellf() {
  float lon = v_merc.x * 360.0 - 180.0;
  float lat = degrees(atan(sinh(PI * (1.0 - 2.0 * v_merc.y))));
  return vec2((lon - u_geo.x) / u_geo.y, (lat - u_geo.z) / u_geo.w) / u_stride;
}
bool inside(ivec2 c) { return c.x >= 0 && c.y >= 0 && c.x < u_size.x && c.y < u_size.y; }
uint codeAt(ivec2 c) { return texelFetch(u_data, c, 0).r; }
// the published value in the grid's own units (log10 for chlorophyll), or a huge negative for no data
float q(uint code, vec3 d) { return code == uint(d.z) ? -1e30 : float(code) * d.x + d.y; }
vec4 pm(vec4 c, float a) { return vec4(c.rgb * c.a * a, c.a * a); }
vec4 over(vec4 top, vec4 under) { return top + under * (1.0 - top.a); }
// a 45 degree line pattern in device pixels: 1 on a line
float hatch(float spacing, float width) {
  float s = spacing * u_dpr;
  float d = mod(gl_FragCoord.x + gl_FragCoord.y, s);
  float dist = min(d, s - d) * 0.7071;
  float hw = width * u_dpr * 0.5;
  return 1.0 - smoothstep(hw - 0.5, hw + 0.5, dist);
}
float dots(float spacing, float radius) {
  float s = spacing * u_dpr;
  vec2 p = mod(gl_FragCoord.xy, s) - 0.5 * s;
  return 1.0 - smoothstep(radius * u_dpr - 0.6, radius * u_dpr + 0.6, length(p));
}
// level lines in device pixels: 1 on a line
float rules(float spacing, float width) {
  float s = spacing * u_dpr;
  float d = mod(gl_FragCoord.y, s);
  float hw = width * u_dpr * 0.5;
  return 1.0 - smoothstep(hw - 0.5, hw + 0.5, min(d, s - d));
}
// "outside the model": water a model or an outlook has no cell for (beyond its grid, its nearshore band, a hole).
// Level rules over a paper veil: not the model's hatch, not the observed layers' "no view" hatch, never blank
// water that could be read as a low value or as covered
vec4 outsideModel() { return over(pm(u_ink2, 0.42 * rules(7.0, 1.0)), pm(u_paper, 0.3)); }
vec4 ramp(float t) {
  float p = clamp(t, 0.0, 1.0) * 7.0;
  int i = int(min(floor(p), 6.0));
  return mix(u_ramp[i], u_ramp[i + 1], p - float(i));
}
float rampT(float v) {
  float x = u_takeLog == 1 ? log(max(v, 1e-9)) / log(10.0) : v;
  return (x - u_domain.x) / (u_domain.y - u_domain.x);
}

void field();
// the whole layer fades in once, whole (v15.1: never chunk by chunk)
void main() { field(); o *= u_fade; }
// v15.1 "outside this layer's area": the chart's own water, untouched (neither the cloud hatch nor a value). No
// tint at all, so no edge of a tint can ever draw a seam across the sea (review: a seam beside the panel)
vec4 outsideArea() { return vec4(0.0); }

void field() {
  vec2 f = cellf();
  ivec2 c = ivec2(floor(f + 0.5));
  bool within = inside(c);
  // an observed layer is drawn past its grid (the quad covers the chart): there is no view there either; a model
  // or an outlook is too, with its own "outside the model" mark; isotherms stop at their grid
  if (!within && u_variant == 3) discard;
  // past an observed grid: outside the layer's area
  if (!within && u_variant == 0) { o = outsideArea(); return; }
  // a chunk on its way: nothing (the page says it is loading); west of the week's westernmost pixel: outside
  if (within && u_variant == 0 && u_loading >= 0.0 && codeAt(c) == uint(u_loading)) discard;
  if (within && u_variant == 0 && u_out >= 0.0 && codeAt(c) == uint(u_out)) { o = outsideArea(); return; }
  // left out beside land (an observed layer only): a flat pale band along the shore, its own mark: no hatch, so
  // it is never read as cloud, and no ramp colour, so it is never read as a value
  if (within && u_variant == 0 && u_adj >= 0.0 && codeAt(c) == uint(u_adj)) {
    o = over(pm(u_ink2, 0.30), pm(u_paper, 0.62));
    return;
  }
  float v = within ? q(codeAt(c), u_dec) : -1e30;
  bool nodata = v < -1e29;

  if (u_variant == 0) {
    if (nodata) {
      // "no view" (T19): a quiet hatch over a light paper veil, so measured water dominates; only inside the area
      // the sensor saw (v15.1: past it, the plain "outside this layer's area" tint above)
      vec4 veil = pm(u_paper, 0.22);
      o = over(pm(u_hatch, 0.32 * hatch(11.0, 1.0)), veil);
      return;
    }
    o = pm(ramp(rampT(v)), u_opacity);
    return;
  }
  if (u_variant == 1) {
    if (nodata) { o = outsideModel(); return; }
    if (u_hasAux == 1) {
      float a = q(texelFetch(u_aux, c, 0).r, u_decAux);
      if (a > -1e29 && abs(v - a) < 0.002) {
        o = pm(u_ink2, 0.7 * dots(6.0, 1.1));
        return;
      }
    }
    vec4 col = pm(ramp(rampT(v)), u_opacity);
    o = over(pm(u_ink, 0.28 * hatch(8.0, 1.0)), col);
    return;
  }
  if (u_variant == 2) {
    if (nodata) { o = outsideModel(); return; }
    // cw-probability-classes-v1: ten 10-point display intervals in the mode's quiet --prob ramp, at 0.72 as the
    // legend draws it (T23), with the model's hatch over it (T2); a past run steps back (T3)
    int k = int(clamp(floor(v * 10.0), 0.0, 9.0));
    vec4 col = pm(u_prob[k], u_past > 0.5 ? 0.42 : 0.72);
    o = over(pm(u_past > 0.5 ? u_ink2 : u_ink, 0.2 * hatch(8.0, 1.0)), col);
    return;
  }
  // isolines: bilinear between four valid cells only (an analysis is smooth by construction); none across land
  vec2 b = f - 0.5;
  ivec2 c0 = ivec2(floor(b));
  vec2 t = b - vec2(c0);
  ivec2 c1 = c0 + ivec2(1, 0), c2 = c0 + ivec2(0, 1), c3 = c0 + ivec2(1, 1);
  if (!inside(c0) || !inside(c3)) discard;
  float v0 = q(codeAt(c0), u_dec), v1 = q(codeAt(c1), u_dec), v2 = q(codeAt(c2), u_dec), v3 = q(codeAt(c3), u_dec);
  if (min(min(v0, v1), min(v2, v3)) < -1e29) discard;
  float vv = mix(mix(v0, v1, t.x), mix(v2, v3, t.x), t.y);
  float g = (vv * u_iso.x + u_iso.y) / u_step;
  float w = max(fwidth(g), 1e-6);
  float d = abs(fract(g + 0.5) - 0.5) / w;
  bool major = mod(floor(g + 0.5), u_index) < 0.5;
  float hw = (major ? 0.95 : 0.55) * u_dpr;
  float line = 1.0 - smoothstep(hw - 0.5, hw + 0.5, d);
  o = pm(u_ink, (major ? 0.9 : 0.55) * line);
  return;
}`;

function compile(gl: WebGL2RenderingContext, vs: string, fs: string) {
  const sh = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
    return s;
  };
  const p = gl.createProgram()!;
  gl.attachShader(p, sh(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link");
  return p;
}

/** Every `stride`-th cell (nearest, never averaged) when a grid is larger than the GPU's texture limit. */
export function decimate(codes: Uint16Array, w: number, h: number, stride: number): { codes: Uint16Array; w: number; h: number } {
  if (stride <= 1) return { codes, w, h };
  const dw = Math.ceil(w / stride), dh = Math.ceil(h / stride);
  const out = new Uint16Array(dw * dh);
  for (let r = 0; r < dh; r++) for (let c = 0; c < dw; c++) out[r * dw + c] = codes[r * stride * w + c * stride];
  return { codes: out, w: dw, h: dh };
}

/** The chart's whole drawn extent (public/chart/chart.json frame.bounds): how far "no view" is hatched. */
const OUTER = CHART_FRAME.bounds;

/**
 * How far a layer is drawn. An observed layer's "no view" hatch and a model's or outlook's "outside the model"
 * rules reach past the grid to the chart's edge (the land masks them), so the grid's edge is never a line across
 * the sea and no blank water reads as a value. Isotherms are lines and stop at their grid.
 */
export function fieldQuadBounds(kind: FieldStyle["kind"], grid: Parameters<typeof gridBounds>[0]): [number, number, number, number] {
  return kind === "isolines" ? gridBounds(grid) : (OUTER as [number, number, number, number]);
}

const VARIANT: Record<FieldStyle["kind"], number> = { ramp: 0, outlook: 1, classes: 2, isolines: 3 };

export type FieldController = CustomLayerInterface & {
  /** A new frame (a cut) or the same frame with more of it loaded. */
  setData: (d: FieldData | null, style: FieldStyle) => void;
  setColours: (c: ModeColours) => void;
  /** the whole layer's opacity, 0..1 (a fade in once the frame is whole) */
  setFade: (a: number) => void;
  /** frames drawn since added (the e2e hook) */
  draws: () => number;
};

export function createFieldLayer(id: string, colours: ModeColours): FieldController {
  let map: MapLibreMap | null = null;
  let gl: WebGL2RenderingContext | null = null;
  let prog: WebGLProgram | null = null;
  let vao: WebGLVertexArrayObject | null = null;
  let quad: WebGLBuffer | null = null;
  let tex: WebGLTexture | null = null;
  let texAux: WebGLTexture | null = null;
  const u: Record<string, WebGLUniformLocation | null> = {};
  let data: FieldData | null = null;
  let style: FieldStyle = { kind: "ramp", domain: [0.05, 50], scale: "log" };
  let cols = colours;
  let stride = 1;
  let size: [number, number] = [0, 0];
  let dirty = true;
  let drawn = 0;
  let fade = 1;
  const mat = new Float32Array(16);

  const upload = (g: WebGL2RenderingContext, t: WebGLTexture, codes: Uint16Array, w: number, h: number) => {
    g.bindTexture(g.TEXTURE_2D, t);
    g.texParameteri(g.TEXTURE_2D, g.TEXTURE_MIN_FILTER, g.NEAREST);
    g.texParameteri(g.TEXTURE_2D, g.TEXTURE_MAG_FILTER, g.NEAREST);
    g.texParameteri(g.TEXTURE_2D, g.TEXTURE_WRAP_S, g.CLAMP_TO_EDGE);
    g.texParameteri(g.TEXTURE_2D, g.TEXTURE_WRAP_T, g.CLAMP_TO_EDGE);
    g.pixelStorei(g.UNPACK_ALIGNMENT, 1);
    g.texImage2D(g.TEXTURE_2D, 0, g.R16UI, w, h, 0, g.RED_INTEGER, g.UNSIGNED_SHORT, codes);
    g.pixelStorei(g.UNPACK_ALIGNMENT, 4);
  };

  const pushData = () => {
    if (!gl || !data || !quad) return;
    const g = gl;
    const { grid, codes } = data;
    const max = g.getParameter(g.MAX_TEXTURE_SIZE) as number;
    stride = Math.max(1, Math.ceil(Math.max(grid.width, grid.height) / max));
    const d = decimate(codes, grid.width, grid.height, stride);
    size = [d.w, d.h];
    tex ??= g.createTexture();
    upload(g, tex!, d.codes, d.w, d.h);
    if (data.aux && data.aux.grid.width === grid.width && data.aux.grid.height === grid.height) {
      const a = decimate(data.aux.codes, grid.width, grid.height, stride);
      texAux ??= g.createTexture();
      upload(g, texAux!, a.codes, a.w, a.h);
    }
    const [w, s, e, n] = fieldQuadBounds(style.kind, grid);
    g.bindBuffer(g.ARRAY_BUFFER, quad);
    g.bufferData(g.ARRAY_BUFFER, new Float32Array([mercX(w), mercY(n), mercX(e), mercY(n), mercX(w), mercY(s), mercX(e), mercY(s)]), g.STATIC_DRAW);
    g.bindTexture(g.TEXTURE_2D, null);
    dirty = false;
  };

  const layer: FieldController = {
    id,
    type: "custom",
    renderingMode: "2d",
    draws: () => drawn,
    setData(d, st) {
      data = d;
      style = st;
      dirty = true;
      map?.triggerRepaint();
    },
    setColours(c) {
      cols = c;
      map?.triggerRepaint();
    },
    setFade(a) {
      fade = Math.max(0, Math.min(1, a));
      map?.triggerRepaint();
    },
    onAdd(m, ctx) {
      map = m;
      gl = ctx as WebGL2RenderingContext;
      prog = compile(gl, VS, FIELD_FS);
      for (const k of ["u_matrix", "u_data", "u_aux", "u_hasAux", "u_geo", "u_size", "u_stride", "u_dec", "u_decAux", "u_adj", "u_takeLog", "u_variant", "u_ramp", "u_prob", "u_domain", "u_ink", "u_ink2", "u_hatch", "u_paper", "u_dpr", "u_opacity", "u_step", "u_index", "u_past", "u_adj", "u_iso", "u_out", "u_loading", "u_fade"]) u[k] = gl.getUniformLocation(prog, k);
      vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      quad = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, quad);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(8), gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(prog, "a_pos");
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);
      tex = texAux = null;
      dirty = true;
    },
    onRemove() {
      // a lost context has already freed these; a live one frees them here
      if (gl && !gl.isContextLost()) {
        if (tex) gl.deleteTexture(tex);
        if (texAux) gl.deleteTexture(texAux);
        if (prog) gl.deleteProgram(prog);
      }
      map = null;
      gl = null;
      prog = vao = quad = tex = texAux = null;
    },
    render(g, args) {
      if (!map || !prog || !data) return;
      if (dirty) pushData();
      if (!tex) return;
      const src = args.modelViewProjectionMatrix as unknown as ArrayLike<number>;
      const ws = 512 * Math.pow(2, map.getZoom());
      for (let i = 0; i < 16; i++) mat[i] = src[i] * (i < 8 ? ws : 1);
      const { grid } = data;
      g.disable(g.STENCIL_TEST);
      // the depth test MapLibre set for this layer's slot stays on: its opaque land, drawn earlier front to back, masks the data
      g.disable(g.CULL_FACE);
      g.enable(g.BLEND);
      g.blendFunc(g.ONE, g.ONE_MINUS_SRC_ALPHA);
      g.useProgram(prog);
      g.uniformMatrix4fv(u.u_matrix, false, mat);
      g.activeTexture(g.TEXTURE0);
      g.bindTexture(g.TEXTURE_2D, tex);
      g.uniform1i(u.u_data, 0);
      const hasAux = !!(texAux && data.aux);
      g.activeTexture(g.TEXTURE1);
      g.bindTexture(g.TEXTURE_2D, hasAux ? texAux : tex);
      g.uniform1i(u.u_aux, 1);
      g.uniform1i(u.u_hasAux, hasAux ? 1 : 0);
      g.uniform4f(u.u_geo, grid.lon_first, grid.lon_step, grid.lat_first, grid.lat_step);
      g.uniform2i(u.u_size, size[0], size[1]);
      g.uniform1f(u.u_stride, stride);
      g.uniform3f(u.u_dec, grid.scale_factor, grid.add_offset, grid.nodata);
      const ag = data.aux?.grid ?? grid;
      g.uniform3f(u.u_decAux, ag.scale_factor, ag.add_offset, ag.nodata);
      const logGrid = grid.transform === "log10";
      g.uniform1i(u.u_variant, VARIANT[style.kind]);
      // a linear grid on a log ramp: the shader takes the log of each value
      g.uniform1i(u.u_takeLog, !logGrid && "scale" in style && style.scale === "log" ? 1 : 0);
      g.uniform4fv(u.u_ramp, new Float32Array(cols.chl.flatMap((c: RGBA) => c)));
      g.uniform4fv(u.u_prob, new Float32Array(cols.prob.flatMap((c: RGBA) => c)));
      if (style.kind === "ramp" || style.kind === "outlook") {
        const [lo, hi] = style.domain;
        // the ramp's domain in the grid's own units: log10 values for a log10 grid, values otherwise
        const dom = style.scale === "log" ? [Math.log10(lo), Math.log10(hi)] : [lo, hi];
        g.uniform2f(u.u_domain, dom[0], dom[1]);
        g.uniform1f(u.u_opacity, style.opacity ?? 1);
      } else g.uniform1f(u.u_opacity, 1);
      g.uniform4fv(u.u_ink, cols.ink);
      g.uniform4fv(u.u_ink2, cols.ink2);
      g.uniform4fv(u.u_hatch, cols.hatch);
      g.uniform4fv(u.u_paper, cols.paper);
      g.uniform1f(u.u_dpr, map.getPixelRatio());
      g.uniform1f(u.u_step, style.kind === "isolines" ? style.step : 1);
      g.uniform1f(u.u_index, style.kind === "isolines" ? style.index : 1);
      g.uniform1f(u.u_past, style.kind === "classes" && style.past ? 1 : 0);
      g.uniform1f(u.u_adj, style.kind === "ramp" && data.adjacentCode != null ? data.adjacentCode : -1);
      g.uniform1f(u.u_out, style.kind === "ramp" && data.outCode != null ? data.outCode : -1);
      g.uniform1f(u.u_loading, style.kind === "ramp" && data.loadingCode != null ? data.loadingCode : -1);
      g.uniform1f(u.u_fade, fade);
      g.uniform2f(u.u_iso, style.kind === "isolines" ? (style.scale ?? 1) : 1, style.kind === "isolines" ? (style.offset ?? 0) : 0);
      g.bindVertexArray(vao);
      g.drawArrays(g.TRIANGLE_STRIP, 0, 4);
      g.bindVertexArray(null);
      g.activeTexture(g.TEXTURE1);
      g.bindTexture(g.TEXTURE_2D, null);
      g.activeTexture(g.TEXTURE0);
      g.bindTexture(g.TEXTURE_2D, null);
      drawn++;
    },
  };
  return layer;
}
