"use client";

import type { CustomLayerInterface, Map as MapLibreMap } from "maplibre-gl";
import type { CurrentField } from "@/lib/currents";
import { FlowSim, HIST, HIST_MS, SEG, mercX, mercY } from "@/lib/flowSim";


import { noteSlowFrames } from "@/lib/capability";
import type { ModeColours } from "./colors";
import { coverageEdges } from "./grid";

/**
 * Measured HF-radar currents as tapered streamlines on the chart (D-motion-feasibility §6; truth rules T24–T28).
 * The particle physics is lib/flowSim's, unchanged and tested: one measured hour, frozen; nearest cell; a
 * particle dies entering a cell without a measurement, so a streak never crosses or fills a gap; screen speed is
 * relative. This file only draws it, in the map's own context:
 * - each streak tapers: width and ink rise toward its head, and with the cell's speed;
 * - a streak is fainter where the radar measured its cell in fewer of the published hours (`quality`, 0..1);
 * - the real edge of coverage is a hairline, so the end of the streaks reads as the end of the measurements;
 * - streak and casing colours are the mode's tokens (`--streak` on a `--halo` casing);
 * - a new hour is a cut (every particle reborn in it); still (reduced motion, paused) draws one settled frame:
 *   the same streaks, not moving (T28); the reduced tier draws fewer at 30 fps.
 */
export type FlowTier = "full" | "reduced";
export type FlowState = { still: boolean; paused: boolean };
export type FlowDebug = { running: () => boolean; count: () => number; segments: () => number; edges: () => number; heads: () => [number, number][]; fieldId: () => string; frames: () => number };

/** Streaks move this many times faster than lib/flowSim's base rate (still proportional to the water's speed). */
export const FLOW_TIME_SCALE = 4;
/** Seconds between trail samples: a trail of HIST samples spans about 3 s, long enough to read as a streamline. */
export const TRAIL_SAMPLE_S = 0.2;

const VS = `#version 300 es
uniform mat4 u_matrix; uniform vec2 u_viewport; uniform float u_width; uniform float u_taper;
in vec2 a_corner; in vec2 a_a; in vec2 a_b; in float a_alpha;
out float v_alpha; out float v_side;
void main() {
  vec4 pa = u_matrix * vec4(a_a, 0.0, 1.0);
  vec4 pb = u_matrix * vec4(a_b, 0.0, 1.0);
  vec2 sa = pa.xy / pa.w * u_viewport * 0.5;
  vec2 sb = pb.xy / pb.w * u_viewport * 0.5;
  vec2 d = sb - sa; float len = length(d);
  vec2 dir = len > 1e-4 ? d / len : vec2(1.0, 0.0);
  vec2 n = vec2(-dir.y, dir.x);
  float w = u_width * mix(1.0, mix(0.3, 1.0, sqrt(clamp(a_alpha * 1.6, 0.0, 1.0))), u_taper);
  vec2 s = mix(sa, sb, a_corner.x) + n * a_corner.y * w * 0.5 + dir * (a_corner.x * 2.0 - 1.0) * w * 0.35;
  gl_Position = vec4(s / (u_viewport * 0.5), pa.z / pa.w, 1.0);
  v_alpha = a_alpha; v_side = a_corner.y;
}`;
const FS = `#version 300 es
precision mediump float;
uniform vec4 u_color;
in float v_alpha; in float v_side; out vec4 o;
void main() { float e = 1.0 - smoothstep(0.5, 1.0, abs(v_side)); float a = clamp(v_alpha * 1.7, 0.0, 1.0) * e * u_color.a; o = vec4(u_color.rgb * a, a); }`;

function compile(gl: WebGL2RenderingContext) {
  const sh = (t: number, s: string) => {
    const x = gl.createShader(t)!;
    gl.shaderSource(x, s);
    gl.compileShader(x);
    if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x) ?? "shader");
    return x;
  };
  const p = gl.createProgram()!;
  gl.attachShader(p, sh(gl.VERTEX_SHADER, VS));
  gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link");
  return p;
}

const MAX = { full: 460, reduced: 220 } as const;
/**
 * About one streak for every two to three measured cells in view, and never more than one per ~26 x 26 screen
 * pixels of measured water, so a zoomed-out view shows the structure of the flow, not a scrawl.
 */
const countFor = (visible: number, tier: FlowTier, cellPx: number) => Math.min(MAX[tier], Math.max(12, Math.round(Math.min(visible * 0.42, (visible * cellPx * cellPx) / 680))));
/** A map frame measured over this (ms) means a device that should animate at 30 fps (FlowLayer v4's budget). */
const FRAME_BUDGET_MS = 8;

export type FlowController = CustomLayerInterface & {
  setField: (id: string, f: CurrentField, quality: Float32Array | null) => void;
  setState: (s: FlowState) => void;
  setColours: (c: ModeColours) => void;
  debug: FlowDebug;
};

export function createFlowLayer(layerId: string, tier: FlowTier, colours: ModeColours, state0: FlowState): FlowController {
  const sim = new FlowSim(MAX.full);
  const inst = new Float32Array(MAX.full * HIST * SEG);
  const mat = new Float32Array(16);
  let fieldId = "";
  let quality: Float32Array | null = null;
  let state = state0;
  let cols = colours;
  let fps = tier === "reduced" ? 30 : 60;
  let map: MapLibreMap | null = null;
  let gl: WebGL2RenderingContext | null = null;
  let prog: WebGLProgram | null = null;
  let vao: WebGLVertexArrayObject | null = null;
  let instBuf: WebGLBuffer | null = null;
  let edgeBuf: WebGLBuffer | null = null;
  let edgeVao: WebGLVertexArrayObject | null = null;
  let edgeCount = 0;
  let segs = 0;
  let frames = 0;
  let u: Record<string, WebGLUniformLocation | null> = {};
  let raf = 0;
  let last = 0;
  let lastHist = 0;
  let costN = 0;
  let costSum = 0;
  let requested = false;

  const worldPx = () => 512 * Math.pow(2, map!.getZoom());
  const stopped = () => state.still || state.paused;
  const shouldRun = () => !!map && !stopped() && !document.hidden && sim.visible.length > 0;

  const recomputeView = () => {
    if (!map || !sim.field) return;
    const b = map.getBounds();
    sim.setView({ west: b.getWest(), south: b.getSouth(), east: b.getEast(), north: b.getNorth() });
    const cellPx = (Math.abs(sim.field.g.lon_step) / 360) * worldPx();
    sim.setCount(countFor(sim.visible.length, fps < 60 ? "reduced" : tier, cellPx));
  };

  const settle = () => {
    if (!map || !sim.field) return;
    // the still frame's trails span what a moving one's do (flowSim settles in HIST_MS steps)
    sim.settle((worldPx() * HIST_MS) / 1000 / (TRAIL_SAMPLE_S * FLOW_TIME_SCALE));
    map.triggerRepaint();
  };

  const uploadEdges = () => {
    if (!gl || !edgeBuf || !sim.field) return;
    const f = sim.field;
    const edges = coverageEdges(f.g, (k) => Number.isFinite(f.u[k]) && Number.isFinite(f.v[k]));
    const buf = new Float32Array(edges.length * SEG);
    edges.forEach(([lo0, la0, lo1, la1], i) => {
      buf.set([mercX(lo0), mercY(la0), mercX(lo1), mercY(la1), 1, 0, 0, 0], i * SEG);
    });
    gl.bindBuffer(gl.ARRAY_BUFFER, edgeBuf);
    gl.bufferData(gl.ARRAY_BUFFER, buf, gl.STATIC_DRAW);
    edgeCount = edges.length;
  };

  const tick = (t: number) => {
    raf = 0;
    if (!map || !shouldRun()) return;
    // the map's own frame cost, measured while the camera is still: over budget drops to 30 fps, once
    if (requested && fps === 60 && !map.isMoving()) {
      costSum += performance.now() - t;
      if (++costN >= 45) {
        if (costSum / costN > FRAME_BUDGET_MS) {
          fps = 30;
          noteSlowFrames();
          recomputeView();
        }
        costN = 0;
        costSum = 0;
      }
    }
    requested = false;
    if (fps < 60 && t - last < 1000 / fps - 2) {
      raf = requestAnimationFrame(tick);
      return;
    }
    const dt = Math.min(0.05, (t - last) / 1000);
    last = t;
    const sample = t - lastHist >= TRAIL_SAMPLE_S * 1000;
    if (sample) lastHist = t;
    sim.step(dt * FLOW_TIME_SCALE, worldPx(), sample);
    map.triggerRepaint();
    requested = true;
    raf = requestAnimationFrame(tick);
  };

  const sync = () => {
    if (shouldRun()) {
      if (!raf) {
        last = lastHist = performance.now();
        raf = requestAnimationFrame(tick);
      }
    } else if (raf) {
      cancelAnimationFrame(raf);
      raf = 0;
    }
  };

  const onMoveEnd = () => {
    recomputeView();
    if (stopped()) settle();
    sync();
  };
  const onVisibility = () => sync();

  const layer: FlowController = {
    id: layerId,
    type: "custom",
    renderingMode: "2d",
    debug: {
      running: () => raf !== 0,
      count: () => sim.count,
      segments: () => segs,
      edges: () => edgeCount,
      heads: () => sim.heads(),
      fieldId: () => fieldId,
      frames: () => frames,
    },
    setField(id, f, q) {
      fieldId = id;
      quality = q;
      sim.setField(f);
      uploadEdges();
      if (!map) return;
      recomputeView();
      // a cut: every particle is reborn in the new hour
      if (stopped()) settle();
      else {
        sim.respawnAll();
        map.triggerRepaint();
      }
      sync();
    },
    setState(s) {
      const was = state;
      state = s;
      if (!map) return;
      if (stopped() && !(was.still || was.paused)) settle();
      if (!stopped() && (was.still || was.paused)) last = lastHist = performance.now();
      sync();
    },
    setColours(c) {
      cols = c;
      map?.triggerRepaint();
    },
    onAdd(m, ctx) {
      map = m;
      gl = ctx as WebGL2RenderingContext;
      prog = compile(gl);
      u = Object.fromEntries(["u_matrix", "u_viewport", "u_width", "u_taper", "u_color"].map((k) => [k, gl!.getUniformLocation(prog!, k)]));
      const corner = new Float32Array([0, -1, 1, -1, 0, 1, 1, 1]);
      const mk = (buf: WebGLBuffer) => {
        const v = gl!.createVertexArray()!;
        gl!.bindVertexArray(v);
        gl!.bindBuffer(gl!.ARRAY_BUFFER, gl!.createBuffer());
        gl!.bufferData(gl!.ARRAY_BUFFER, corner, gl!.STATIC_DRAW);
        const lc = gl!.getAttribLocation(prog!, "a_corner");
        gl!.enableVertexAttribArray(lc);
        gl!.vertexAttribPointer(lc, 2, gl!.FLOAT, false, 0, 0);
        gl!.bindBuffer(gl!.ARRAY_BUFFER, buf);
        for (const [name, size, off] of [["a_a", 2, 0], ["a_b", 2, 8], ["a_alpha", 1, 16]] as const) {
          const loc = gl!.getAttribLocation(prog!, name);
          gl!.enableVertexAttribArray(loc);
          gl!.vertexAttribPointer(loc, size, gl!.FLOAT, false, SEG * 4, off);
          gl!.vertexAttribDivisor(loc, 1);
        }
        gl!.bindVertexArray(null);
        return v;
      };
      instBuf = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
      gl.bufferData(gl.ARRAY_BUFFER, inst.byteLength, gl.DYNAMIC_DRAW);
      vao = mk(instBuf);
      edgeBuf = gl.createBuffer()!;
      edgeVao = mk(edgeBuf);
      uploadEdges();
      m.on("moveend", onMoveEnd);
      document.addEventListener("visibilitychange", onVisibility);
      recomputeView();
      if (stopped()) settle();
      else sim.respawnAll();
      sync();
    },
    onRemove(m) {
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
      m?.off?.("moveend", onMoveEnd);
      map?.off("moveend", onMoveEnd);
      document.removeEventListener("visibilitychange", onVisibility);
      if (gl && !gl.isContextLost() && prog) gl.deleteProgram(prog);
      map = null;
      gl = null;
      prog = vao = edgeVao = instBuf = edgeBuf = null;
    },
    render(g, args) {
      if (!map || !prog || !sim.field) return;
      const src = args.modelViewProjectionMatrix as unknown as ArrayLike<number>;
      const ws = worldPx();
      for (let i = 0; i < 16; i++) mat[i] = src[i] * (i < 8 ? ws : 1);
      g.disable(g.STENCIL_TEST);
      // the depth test MapLibre set for this layer's slot stays on: its opaque land, drawn earlier front to back, masks the data
      g.disable(g.CULL_FACE);
      g.enable(g.BLEND);
      g.blendFunc(g.ONE, g.ONE_MINUS_SRC_ALPHA);
      g.useProgram(prog);
      g.uniformMatrix4fv(u.u_matrix, false, mat);
      g.uniform2f(u.u_viewport, g.drawingBufferWidth, g.drawingBufferHeight);
      const dpr = map.getPixelRatio();
      // the edge of coverage: a hairline
      if (edgeCount && edgeVao) {
        g.uniform1f(u.u_taper, 0);
        g.uniform1f(u.u_width, 1.2 * dpr);
        const [r, gg, b] = cols.ink2;
        g.uniform4f(u.u_color, r, gg, b, 0.32);
        g.bindVertexArray(edgeVao);
        g.drawArraysInstanced(g.TRIANGLE_STRIP, 0, 4, edgeCount);
      }
      let n = sim.build(inst, { colour: false, alpha: 1 });
      if (quality && sim.field) {
        // fainter where the cell was measured in fewer hours (its own quality, never a neighbour's)
        for (let i = 0; i < n; i++) {
          const o = i * SEG;
          const k = sim.cellAt(inst[o + 2], inst[o + 3]);
          inst[o + 4] *= k >= 0 ? 0.3 + 0.7 * quality[k] : 0;
        }
      }
      segs = n;
      if (n && vao && instBuf) {
        g.bindVertexArray(vao);
        g.bindBuffer(g.ARRAY_BUFFER, instBuf);
        g.bufferSubData(g.ARRAY_BUFFER, 0, inst, 0, n * SEG);
        g.uniform1f(u.u_taper, 1);
        // a casing in the halo colour holds each streak's edge over any ground (it carries no value)
        // thinner when zoomed out, where a 2 km cell is only a few pixels
        const zw = Math.min(1, Math.max(0.6, (map.getZoom() - 6.5) / 3));
        g.uniform1f(u.u_width, 4.2 * dpr * zw);
        const [hr, hg, hb, ha] = cols.halo;
        g.uniform4f(u.u_color, hr, hg, hb, ha * 0.38);
        g.drawArraysInstanced(g.TRIANGLE_STRIP, 0, 4, n);
        g.uniform1f(u.u_width, 2.4 * dpr * zw);
        const [sr, sg, sb, sa] = cols.streak;
        g.uniform4f(u.u_color, sr, sg, sb, sa * 0.92);
        g.drawArraysInstanced(g.TRIANGLE_STRIP, 0, 4, n);
      }
      g.bindVertexArray(null);
      frames++;
      n = 0;
    },
  };
  return layer;
}
