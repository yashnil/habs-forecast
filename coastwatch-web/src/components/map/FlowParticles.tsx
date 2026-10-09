"use client";

import { useEffect, useRef } from "react";
import { useMap } from "react-map-gl/maplibre";
import type { CurrentField } from "@/lib/currents";

/**
 * Animated particles moving through ONE observed hourly field (or the 24-hour mean).
 * Honest by construction:
 * - the field never changes during the animation (no motion between hours is implied);
 * - each particle takes the value of the cell it is in (nearest cell, no interpolation) and
 *   disappears in a cell without an observation: particles never cross gaps or land;
 * - screen speed is proportional to current speed but the same at every zoom (1 m/s =
 *   PX_PER_MS px/s), so it shows direction and relative speed, not a trajectory or forecast;
 * - short lives (1-2 s), so no particle traces a long, trajectory-like path.
 * Not rendered when the user prefers reduced motion (the dock shows arrows instead).
 */
const PX_PER_MS = 30;

export default function FlowParticles({ field, count }: { field: CurrentField; count: number }) {
  const { current } = useMap();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const map = current?.getMap();
    if (!map) return;
    const container = map.getContainer();
    const canvas = document.createElement("canvas");
    canvas.setAttribute("data-testid", "flow-particles");
    canvas.setAttribute("aria-hidden", "true");
    Object.assign(canvas.style, { position: "absolute", inset: "0", pointerEvents: "none", zIndex: "1" });
    container.appendChild(canvas);
    canvasRef.current = canvas;
    const ctx = canvas.getContext("2d")!;
    const g = field.g;
    const valid: number[] = [];
    for (let k = 0; k < field.u.length; k++) if (Number.isFinite(field.u[k])) valid.push(k);

    type P = { lon: number; lat: number; age: number; life: number };
    let ps: P[] = [];
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const resize = () => {
      const r = container.getBoundingClientRect();
      canvas.width = r.width * dpr;
      canvas.height = r.height * dpr;
      canvas.style.width = `${r.width}px`;
      canvas.style.height = `${r.height}px`;
    };
    resize();
    const cellAt = (lon: number, lat: number) => {
      const r = Math.round((lat - g.lat_first) / g.lat_step);
      const c = Math.round((lon - g.lon_first) / g.lon_step);
      if (r < 0 || c < 0 || r >= g.height || c >= g.width) return -1;
      return r * g.width + c;
    };
    const spawn = (): P | null => {
      const b = map.getBounds();
      for (let tries = 0; tries < 20; tries++) {
        const k = valid[Math.floor(Math.random() * valid.length)];
        if (k == null) return null;
        const r = Math.floor(k / g.width);
        const c = k % g.width;
        // anywhere inside the cell, so particles do not line up on cell centres
        const lat = g.lat_first + (r + Math.random() - 0.5) * g.lat_step;
        const lon = g.lon_first + (c + Math.random() - 0.5) * g.lon_step;
        if (b.contains([lon, lat])) return { lon, lat, age: 0, life: 1 + Math.random() };
      }
      return null;
    };
    const reset = () => {
      ps = [];
      for (let i = 0; i < count; i++) {
        const p = spawn();
        if (p) {
          p.age = Math.random() * p.life;
          ps.push(p);
        }
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    };
    reset();
    let last = performance.now();
    let raf = 0;
    let moving = false;
    const frame = (t: number) => {
      const dt = Math.min(0.05, (t - last) / 1000);
      last = t;
      // fade the previous frame: short trails
      ctx.globalCompositeOperation = "destination-in";
      ctx.fillStyle = "rgba(0,0,0,0.88)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.globalCompositeOperation = "source-over";
      if (!moving) {
        ctx.lineWidth = 1.4 * dpr;
        ctx.lineCap = "round";
        for (let i = 0; i < ps.length; i++) {
          const p = ps[i];
          const k = cellAt(p.lon, p.lat);
          const u = k >= 0 ? field.u[k] : NaN;
          const v = k >= 0 ? field.v[k] : NaN;
          p.age += dt;
          if (!Number.isFinite(u) || !Number.isFinite(v) || p.age > p.life) {
            const n = spawn();
            if (n) ps[i] = n;
            continue;
          }
          const a = map.project([p.lon, p.lat]);
          const bx = a.x + u * PX_PER_MS * dt;
          const by = a.y - v * PX_PER_MS * dt;
          const ll = map.unproject([bx, by]);
          const sp = Math.hypot(u, v);
          ctx.strokeStyle = `rgba(238,244,250,${Math.min(0.95, 0.35 + sp * 1.6)})`;
          ctx.beginPath();
          ctx.moveTo(a.x * dpr, a.y * dpr);
          ctx.lineTo(bx * dpr, by * dpr);
          ctx.stroke();
          p.lon = ll.lng;
          p.lat = ll.lat;
        }
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    const onMoveStart = () => {
      moving = true;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    };
    const onMoveEnd = () => {
      moving = false;
      reset();
    };
    const onVis = () => {
      if (document.hidden) cancelAnimationFrame(raf);
      else {
        last = performance.now();
        raf = requestAnimationFrame(frame);
      }
    };
    map.on("movestart", onMoveStart);
    map.on("moveend", onMoveEnd);
    map.on("resize", resize);
    document.addEventListener("visibilitychange", onVis);
    return () => {
      cancelAnimationFrame(raf);
      map.off("movestart", onMoveStart);
      map.off("moveend", onMoveEnd);
      map.off("resize", resize);
      document.removeEventListener("visibilitychange", onVis);
      canvas.remove();
    };
  }, [current, field, count]);

  return null;
}
