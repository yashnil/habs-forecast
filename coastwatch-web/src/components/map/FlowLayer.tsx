"use client";

import { useEffect, useRef } from "react";
import type { CustomLayerInterface, Map as MLMap } from "maplibre-gl";
import type { CurrentField } from "@/lib/currents";
import { FlowSim, HIST, HIST_MS, SEG, mercX, mercY } from "@/lib/flowSim";

/**
 * Observed HF-radar currents as moving particles, drawn by a MapLibre custom layer in the map's own
 * WebGL context (no second context, no canvas over the map), under the labels, the official notices and
 * the selection. The honesty rules live in lib/flowSim; this file draws them:
 * - a faint fill of every observed 2 km cell at its true extent, so "no particles" visibly means "no
 *   observation", never calm water;
 * - particles are georeferenced and keep moving while the map pans and zooms;
 * - a new hour swaps the buffers and respawns every particle (a cut): the layer is never remounted;
 * - the loop runs only while it shows something: it stops when paused (WCAG 2.2.2), when the tab is
 *   hidden, when no observed cell is in view, and never starts under reduced motion (one still frame);
 * - 30 fps on low-power devices, decided by the device (cores, memory) or by the measured cost of the
 *   map's own frames, never by the screen's width (every animated frame repaints the whole map).
 */
export const FLOW_LAYER_ID = "nx-flow";
const FOOT_ALPHA = 0.07;
/**
 * A map frame measured over this (ms: half of a 60 Hz frame) means a device that should animate at 30 fps.
 * Measured from the frame's start to this layer's tick, which runs after the map's own render: on an
 * Apple-silicon desktop about 5 ms, on the same CPU throttled 4x (a mid-range phone) about 16 ms.
 */
const FRAME_BUDGET_MS = 8;

export type FlowOptions = { paused: boolean; still: boolean; colour: boolean; coarser: boolean };
export type FlowStats = { frames: number; simMs: number; drawMs: number; segments: number; footprintCells: number; frameCostMs: number };

const VS_LINE = `#version 300 es
uniform mat4 u_matrix; uniform vec2 u_viewport; uniform float u_width;
in vec2 a_corner; in vec2 a_a; in vec2 a_b; in float a_alpha; in vec3 a_color;
out float v_alpha; out float v_side; out vec3 v_color;
void main() {
  vec4 pa = u_matrix * vec4(a_a, 0.0, 1.0);
  vec4 pb = u_matrix * vec4(a_b, 0.0, 1.0);
  vec2 sa = pa.xy / pa.w * u_viewport * 0.5;
  vec2 sb = pb.xy / pb.w * u_viewport * 0.5;
  vec2 d = sb - sa; float len = length(d);
  vec2 dir = len > 1e-4 ? d / len : vec2(1.0, 0.0);
  vec2 n = vec2(-dir.y, dir.x);
  vec2 s = mix(sa, sb, a_corner.x) + n * a_corner.y * u_width * 0.5 + dir * (a_corner.x * 2.0 - 1.0) * u_width * 0.5;
  gl_Position = vec4(s / (u_viewport * 0.5), 0.0, 1.0);
  v_alpha = a_alpha; v_side = a_corner.y; v_color = a_color;
}`;
const FS_LINE = `#version 300 es
precision mediump float;
in float v_alpha; in float v_side; in vec3 v_color; out vec4 o;
void main() { float e = 1.0 - smoothstep(0.55, 1.0, abs(v_side)); o = vec4(v_color, 1.0) * v_alpha * e; }`;
const VS_FOOT = `#version 300 es
uniform mat4 u_matrix;
in vec2 a_corner; in vec4 a_box;
void main() { gl_Position = u_matrix * vec4(mix(a_box.xy, a_box.zw, a_corner), 0.0, 1.0); }`;
const FS_FOOT = `#version 300 es
precision mediump float;
uniform vec4 u_color; out vec4 o;
void main() { o = u_color; }`;

function program(gl: WebGL2RenderingContext, vs: string, fs: string) {
  const sh = (t: number, s: string) => {
    const x = gl.createShader(t)!;
    gl.shaderSource(x, s);
    gl.compileShader(x);
    if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x) ?? "shader");
    return x;
  };
  const p = gl.createProgram()!;
  gl.attachShader(p, sh(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link");
  return p;
}

// a device found to be low-power keeps 30 fps for the session (measured once, never flapping)
let measuredLowPower = false;

/** 30 fps on a low-power device: few cores or little memory, or map frames measured over budget. */
const forcedFps = () => (window as unknown as { __cwFlowFps?: number }).__cwFlowFps;
export function flowFps(): number {
  const n = navigator as Navigator & { deviceMemory?: number };
  // a measuring script may fix the rate (then nothing is measured)
  const forced = forcedFps();
  if (forced) return forced;
  if (measuredLowPower) return 30;
  if ((n.hardwareConcurrency && n.hardwareConcurrency <= 4) || (n.deviceMemory && n.deviceMemory <= 4)) return 30;
  return 60;
}

/** Particle budget: up to 1,400 on a capable device, 600 on a low-power one; scaled to the observed cells in view. */
const maxParticles = (fps: number) => (fps < 60 ? 600 : 1400);
const countFor = (visible: number, max: number, coarser: boolean) => Math.min(max, Math.max(60, Math.round(visible * 1.5))) * (coarser ? 0.6 : 1);

export type FlowController = CustomLayerInterface & {
  setField: (f: CurrentField) => void;
  setOptions: (o: FlowOptions) => void;
  running: () => boolean;
  fps: () => number;
  sim: FlowSim;
  stats: FlowStats;
};

export function createFlowLayer(initial: CurrentField, opts0: FlowOptions): FlowController {
  let fps = flowFps();
  const sim = new FlowSim(maxParticles(60));
  let opts = opts0;
  const stats: FlowStats = { frames: 0, simMs: 0, drawMs: 0, segments: 0, footprintCells: 0, frameCostMs: 0 };
  const inst = new Float32Array(sim.max * HIST * SEG);
  const mat = new Float32Array(16);
  let map: MLMap | null = null;
  let gl: WebGL2RenderingContext | null = null;
  let last = 0;
  let lastHist = 0;
  let raf = 0;
  let pLine: WebGLProgram, pFoot: WebGLProgram;
  let vaoLine: WebGLVertexArrayObject, vaoFoot: WebGLVertexArrayObject;
  let instBuf: WebGLBuffer, boxBuf: WebGLBuffer;
  let footCount = 0;
  let uLine: Record<string, WebGLUniformLocation | null> = {};
  let uFoot: Record<string, WebGLUniformLocation | null> = {};
  let costN = 0;
  let costSum = 0;
  let requested = false;

  const worldPx = () => 512 * Math.pow(2, map!.getZoom());
  const visibleNow = () => sim.visible.length > 0;
  const shouldRun = () => !!map && !opts.still && !opts.paused && !document.hidden && visibleNow();

  const uploadFootprint = () => {
    if (!gl || !sim.field) return;
    const g = sim.field.g;
    const boxes = new Float32Array(sim.valid.length * 4);
    sim.valid.forEach((k, i) => {
      const r = Math.floor(k / g.width), c = k % g.width;
      const lat = g.lat_first + r * g.lat_step, lon = g.lon_first + c * g.lon_step;
      boxes[i * 4] = mercX(lon - g.lon_step / 2);
      boxes[i * 4 + 1] = mercY(lat + g.lat_step / 2);
      boxes[i * 4 + 2] = mercX(lon + g.lon_step / 2);
      boxes[i * 4 + 3] = mercY(lat - g.lat_step / 2);
    });
    gl.bindBuffer(gl.ARRAY_BUFFER, boxBuf);
    gl.bufferData(gl.ARRAY_BUFFER, boxes, gl.STATIC_DRAW);
    footCount = sim.valid.length;
    stats.footprintCells = footCount;
  };

  const recomputeView = () => {
    if (!map) return;
    const b = map.getBounds();
    sim.setView({ west: b.getWest(), south: b.getSouth(), east: b.getEast(), north: b.getNorth() });
    sim.setCount(countFor(sim.visible.length, maxParticles(fps), opts.coarser));
  };

  /** Not animating (reduced motion, paused): one settled frame for what is in view. */
  const still = () => {
    if (!map) return;
    sim.settle(worldPx());
    map.triggerRepaint();
  };

  const tick = (t: number) => {
    raf = 0;
    if (!map || !shouldRun()) return;
    // what the frame's earlier callbacks cost (the map's own render, requested last frame): measured, so
    // a device too slow for 60 fps drops to 30 once (never while the camera moves: a pan costs more anyway)
    if (requested && fps === 60 && !forcedFps() && !map.isMoving()) {
      costSum += performance.now() - t;
      costN++;
      if (costN >= 45) {
        stats.frameCostMs = costSum / costN;
        if (stats.frameCostMs > FRAME_BUDGET_MS) {
          measuredLowPower = true;
          fps = 30;
          // and the low-power particle budget
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
    const t0 = performance.now();
    const sample = t - lastHist >= HIST_MS;
    if (sample) lastHist = t;
    sim.step(dt, worldPx(), sample);
    stats.simMs += performance.now() - t0;
    // the map's frame is requested first, so it runs before this tick next frame (and is measured above)
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
    } else {
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
      if (map && !opts.still && !opts.paused) map.triggerRepaint();
    }
  };

  const onMoveEnd = () => {
    recomputeView();
    if (opts.still || opts.paused) still();
    sync();
  };
  const onVisibility = () => sync();

  const layer: FlowController = {
    id: FLOW_LAYER_ID,
    type: "custom",
    renderingMode: "2d",
    sim,
    stats,
    running: () => raf !== 0,
    fps: () => fps,
    setField(f) {
      sim.setField(f);
      uploadFootprint();
      if (!map) return;
      recomputeView();
      // a cut: every particle is reborn in the new hour; a paused or still layer settles once in it
      if (opts.still || opts.paused) still();
      else {
        sim.respawnAll();
        map.triggerRepaint();
      }
      sync();
    },
    setOptions(o) {
      const was = opts;
      opts = o;
      if (!map) return;
      if (was.coarser !== o.coarser) recomputeView();
      if ((o.still || o.paused) && !(was.still || was.paused)) map.triggerRepaint();
      if (!o.paused && was.paused && !o.still) last = lastHist = performance.now();
      if (was.colour !== o.colour) map.triggerRepaint();
      sync();
    },
    onAdd(m, ctx) {
      map = m;
      gl = ctx as WebGL2RenderingContext;
      pLine = program(gl, VS_LINE, FS_LINE);
      pFoot = program(gl, VS_FOOT, FS_FOOT);
      uLine = { m: gl.getUniformLocation(pLine, "u_matrix"), vp: gl.getUniformLocation(pLine, "u_viewport"), w: gl.getUniformLocation(pLine, "u_width") };
      uFoot = { m: gl.getUniformLocation(pFoot, "u_matrix"), c: gl.getUniformLocation(pFoot, "u_color") };
      // particles: one instanced quad per trail segment
      vaoLine = gl.createVertexArray()!;
      gl.bindVertexArray(vaoLine);
      gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, -1, 1, -1, 0, 1, 1, 1]), gl.STATIC_DRAW);
      const lc = gl.getAttribLocation(pLine, "a_corner");
      gl.enableVertexAttribArray(lc);
      gl.vertexAttribPointer(lc, 2, gl.FLOAT, false, 0, 0);
      instBuf = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
      gl.bufferData(gl.ARRAY_BUFFER, inst.byteLength, gl.DYNAMIC_DRAW);
      for (const [name, size, off] of [["a_a", 2, 0], ["a_b", 2, 8], ["a_alpha", 1, 16], ["a_color", 3, 20]] as const) {
        const loc = gl.getAttribLocation(pLine, name);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, gl.FLOAT, false, SEG * 4, off);
        gl.vertexAttribDivisor(loc, 1);
      }
      // observed footprint: one quad per observed cell, its true extent
      vaoFoot = gl.createVertexArray()!;
      gl.bindVertexArray(vaoFoot);
      gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
      const fc = gl.getAttribLocation(pFoot, "a_corner");
      gl.enableVertexAttribArray(fc);
      gl.vertexAttribPointer(fc, 2, gl.FLOAT, false, 0, 0);
      boxBuf = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, boxBuf);
      const fb = gl.getAttribLocation(pFoot, "a_box");
      gl.enableVertexAttribArray(fb);
      gl.vertexAttribPointer(fb, 4, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(fb, 1);
      gl.bindVertexArray(null);

      m.on("moveend", onMoveEnd);
      document.addEventListener("visibilitychange", onVisibility);
      layer.setField(sim.field ?? initial);
    },
    onRemove(m) {
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
      m.off("moveend", onMoveEnd);
      document.removeEventListener("visibilitychange", onVisibility);
      map = null;
      gl = null;
    },
    render(g, args) {
      if (!map) return;
      const t0 = performance.now();
      // MapLibre 6: modelViewProjectionMatrix takes world-pixel coordinates (Mercator x 512 * 2^zoom), not
      // Mercator 0..1; the scale is folded in so the vertex data stay in Mercator 0..1
      const src = args.modelViewProjectionMatrix as unknown as ArrayLike<number>;
      const ws = worldPx();
      for (let i = 0; i < 16; i++) mat[i] = src[i] * (i < 8 ? ws : 1);
      // MapLibre leaves stencil (raster clipping), depth and culling state from earlier layers
      g.disable(g.STENCIL_TEST);
      g.disable(g.DEPTH_TEST);
      g.disable(g.CULL_FACE);
      g.enable(g.BLEND);
      g.blendFunc(g.ONE, g.ONE_MINUS_SRC_ALPHA);
      if (footCount) {
        g.useProgram(pFoot);
        g.uniformMatrix4fv(uFoot.m, false, mat);
        g.uniform4f(uFoot.c, FOOT_ALPHA, FOOT_ALPHA, FOOT_ALPHA * 1.1, FOOT_ALPHA);
        g.bindVertexArray(vaoFoot);
        g.drawArraysInstanced(g.TRIANGLE_STRIP, 0, 4, footCount);
      }
      const n = sim.build(inst, { colour: opts.colour, alpha: opts.coarser ? 0.7 : 1 });
      stats.segments = n;
      if (n) {
        g.useProgram(pLine);
        g.uniformMatrix4fv(uLine.m, false, mat);
        g.uniform2f(uLine.vp, g.drawingBufferWidth, g.drawingBufferHeight);
        g.uniform1f(uLine.w, 1.6 * Math.min(2, window.devicePixelRatio || 1));
        g.bindVertexArray(vaoLine);
        g.bindBuffer(g.ARRAY_BUFFER, instBuf);
        g.bufferSubData(g.ARRAY_BUFFER, 0, inst, 0, n * SEG);
        g.drawArraysInstanced(g.TRIANGLE_STRIP, 0, 4, n);
      }
      g.bindVertexArray(null);
      stats.frames++;
      stats.drawMs += performance.now() - t0;
    },
  };
  sim.setField(initial);
  return layer;
}

/** The e2e and measuring hook: the layer's own state, read from the page. */
export type FlowDebug = {
  running: () => boolean;
  fps: () => number;
  stats: FlowStats;
  count: () => number;
  /** live particle heads and trail points, [lon, lat] */
  points: () => [number, number][];
  /** the field's lattice and its observed-cell mask (1 = observed), to check positions against */
  mask: () => { lat_first: number; lat_step: number; lon_first: number; lon_step: number; width: number; height: number; observed: number[] } | null;
  fieldId: () => string;
};

/**
 * Mounts the custom layer once, under `beforeId` (and keeps it there when other layers are added); a new
 * field or new options update it in place.
 */
export default function FlowLayer({ map, id, field, options, beforeId }: { map: MLMap; id: string; field: CurrentField; options: FlowOptions; beforeId: string }) {
  const ref = useRef<FlowController | null>(null);
  const fieldRef = useRef({ id, field });
  fieldRef.current = { id, field };
  const optsRef = useRef(options);
  optsRef.current = options;
  const shownId = useRef(id);
  useEffect(() => {
    const layer = createFlowLayer(fieldRef.current.field, optsRef.current);
    ref.current = layer;
    shownId.current = fieldRef.current.id;
    const placed = () => {
      if (!map.getLayer(FLOW_LAYER_ID) || !map.getLayer(beforeId)) return;
      const order = map.getLayersOrder();
      const i = order.indexOf(FLOW_LAYER_ID);
      if (i >= 0 && order[i + 1] !== beforeId) map.moveLayer(FLOW_LAYER_ID, beforeId);
    };
    map.addLayer(layer, map.getLayer(beforeId) ? beforeId : undefined);
    map.on("styledata", placed);
    const dbg: FlowDebug = {
      running: layer.running,
      fps: layer.fps,
      stats: layer.stats,
      count: () => layer.sim.count,
      points: () => layer.sim.points(),
      mask: () => {
        const f = layer.sim.field;
        if (!f) return null;
        const { lat_first, lat_step, lon_first, lon_step, width, height } = f.g;
        return { lat_first, lat_step, lon_first, lon_step, width, height, observed: Array.from(f.u, (u, k) => (Number.isFinite(u) && Number.isFinite(f.v[k]) ? 1 : 0)) };
      },
      fieldId: () => shownId.current,
    };
    (window as unknown as { __cwFlow?: FlowDebug }).__cwFlow = dbg;
    return () => {
      map.off("styledata", placed);
      if (map.getLayer(FLOW_LAYER_ID)) map.removeLayer(FLOW_LAYER_ID);
      ref.current = null;
      const w = window as unknown as { __cwFlow?: FlowDebug };
      if (w.__cwFlow === dbg) delete w.__cwFlow;
    };
  }, [map, beforeId]);
  // a new hour: the buffers are swapped, the layer stays
  useEffect(() => {
    const l = ref.current;
    if (!l || l.sim.field === field) return;
    l.setField(field);
    shownId.current = id;
  }, [field, id]);
  useEffect(() => {
    ref.current?.setOptions(options);
  }, [options]);
  return null;
}
