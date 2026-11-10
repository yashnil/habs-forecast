"use client";

import type { CustomLayerInterface, Map as MapLibreMap } from "maplibre-gl";
import type { ModeColours, RGBA } from "./colors";
import { gridBounds, mercX, mercY, type FieldGrid } from "./grid";

/**
 * A published value grid drawn on the chart in the map's own WebGL context (no second context, no canvas over
 * the map). The raw uint16 codes go to the GPU as they were published and are decoded per pixel, so what is
 * drawn is exactly what a readout reads. Truth rules (E §2) carried here:
 * - nearest cell, no smoothing, for observations (T19: nothing is filled); a gap in an observed layer is the
 *   "no view" hatch over a paper veil, never a low colour (T19, T20);
 * - a model or outlook never shares a mark with an observation (T2): an outlook carries a hatch over its ramp,
 *   and cells where it equals the climatology carry dots and no colour ("no signal", E finding 2); C-HARM is
 *   drawn as hatch density in ink by display class (T12: no alarm colour), lighter when the run is past (T3);
 * - a temperature analysis is drawn as isotherms (lines), so it never borrows the chlorophyll ramp (T22);
 * - colours come from the mode's tokens (day, dusk, night); a new frame swaps the texture, a cut (T26).
 * Screen-space patterns (hatch, dots) are drawn in device pixels, so they read the same at every zoom.
 */
export type FieldStyle =
  /** observed, on the ramp; gaps hatched "no view" */
  | { kind: "ramp"; domain: [number, number]; scale: "log" | "linear"; opacity?: number }
  /** an outlook on the same ramp, hatched as a model; `aux` (the climatology) marks "no signal" cells with dots */
  | { kind: "outlook"; domain: [number, number]; scale: "log" | "linear"; opacity?: number }
  /** a probability 0..1 as ink hatch density by display class; `past` draws it lighter */
  | { kind: "classes"; past: boolean }
  /** lines of equal value every `step` (in `value * scale + offset`: °C drawn as whole °F), heavier every `index` steps */
  | { kind: "isolines"; step: number; index: number; scale?: number; offset?: number };

export type FieldData = { grid: FieldGrid; codes: Uint16Array; aux?: { grid: FieldGrid; codes: Uint16Array } | null };

const VS = `#version 300 es
uniform mat4 u_matrix;
in vec2 a_pos;
out vec2 v_merc;
void main() { v_merc = a_pos; gl_Position = u_matrix * vec4(a_pos, 0.0, 1.0); }`;

const FS = `#version 300 es
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
vec4 ramp(float t) {
  float p = clamp(t, 0.0, 1.0) * 7.0;
  int i = int(min(floor(p), 6.0));
  return mix(u_ramp[i], u_ramp[i + 1], p - float(i));
}
float rampT(float v) {
  float x = u_takeLog == 1 ? log(max(v, 1e-9)) / log(10.0) : v;
  return (x - u_domain.x) / (u_domain.y - u_domain.x);
}

void main() {
  vec2 f = cellf();
  ivec2 c = ivec2(floor(f + 0.5));
  if (!inside(c)) discard;
  float v = q(codeAt(c), u_dec);
  bool nodata = v < -1e29;

  if (u_variant == 0) {
    if (nodata) {
      vec4 veil = pm(u_paper, 0.55);
      o = over(pm(u_hatch, hatch(7.0, 1.0)), veil);
      return;
    }
    o = pm(ramp(rampT(v)), u_opacity);
    return;
  }
  if (u_variant == 1) {
    if (nodata) discard;
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
    if (nodata) discard;
    int k = int(clamp(floor(v * 5.0), 0.0, 4.0));
    float spacing = k == 0 ? 18.0 : k == 1 ? 12.0 : k == 2 ? 8.5 : k == 3 ? 6.0 : 4.5;
    vec4 ink = u_past > 0.5 ? u_ink2 : u_ink;
    o = pm(ink, (u_past > 0.5 ? 0.5 : 0.72) * hatch(spacing, 1.0));
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

const VARIANT: Record<FieldStyle["kind"], number> = { ramp: 0, outlook: 1, classes: 2, isolines: 3 };

export type FieldController = CustomLayerInterface & {
  /** A new frame (a cut) or the same frame with more of it loaded. */
  setData: (d: FieldData | null, style: FieldStyle) => void;
  setColours: (c: ModeColours) => void;
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
    const [w, s, e, n] = gridBounds(grid);
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
    onAdd(m, ctx) {
      map = m;
      gl = ctx as WebGL2RenderingContext;
      prog = compile(gl, VS, FS);
      for (const k of ["u_matrix", "u_data", "u_aux", "u_hasAux", "u_geo", "u_size", "u_stride", "u_dec", "u_decAux", "u_takeLog", "u_variant", "u_ramp", "u_domain", "u_ink", "u_ink2", "u_hatch", "u_paper", "u_dpr", "u_opacity", "u_step", "u_index", "u_past", "u_iso"]) u[k] = gl.getUniformLocation(prog, k);
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
      g.disable(g.DEPTH_TEST);
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
