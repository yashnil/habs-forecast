"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import { DEEP_SPEED } from "@/nx/ds/motion";
import { flying } from "./flight";
import { getFrame, onFrame, type ChartFrame } from "./runtime";

/**
 * The one ambient effect (v14 README §5, SYSTEM.md "Ambient"): Paper Shaders' `water` caustics inside the water deeper
 * than the shelf edge (200 m, 109 fm) only. It marks where the shelf ends and the open ocean begins.
 *
 * Progressive enhancement: loaded after the chart has drawn (never in the first paint's way), never on the static tier
 * or in Night mode. Its mask (everything outside the "shallower than 200 m" areas, which include the land) is redrawn
 * when the camera settles; the texture hides while the camera moves, pauses while the tab is hidden, holds still under
 * reduced motion, and renders at 1x on a phone. Without WebGL or the module, the chart is complete without it.
 */
const Water = dynamic(() => import("@paper-design/shaders-react").then((m) => m.Water), { ssr: false, loading: () => null });

type Ring = Float64Array; // lon, lat pairs
let shelf: Promise<Ring[]> | null = null;
function shelfRings(): Promise<Ring[]> {
  return (shelf ??= fetch("/chart/depth.geojson")
    .then((r) => r.json())
    .then((j: { features: { properties: { d: number }; geometry: { type: string; coordinates: number[][][] | number[][][][] } }[] }) => {
      const out: Ring[] = [];
      for (const f of j.features) {
        if (f.properties.d !== 200) continue;
        const polys = (f.geometry.type === "MultiPolygon" ? f.geometry.coordinates : [f.geometry.coordinates]) as number[][][][];
        for (const poly of polys) for (const ring of poly) out.push(Float64Array.from(ring.flat()));
      }
      return out;
    })
    .catch(() => []));
}

/** The deep water as an alpha mask in screen pixels, at `scale` of the frame. */
function deepMask(f: ChartFrame, rings: Ring[], scale: number): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(f.viewport.width * scale));
  c.height = Math.max(1, Math.round(f.viewport.height * scale));
  const g = c.getContext("2d")!;
  g.fillStyle = "black";
  g.fillRect(0, 0, c.width, c.height);
  g.globalCompositeOperation = "destination-out";
  const path = new Path2D();
  for (const r of rings) {
    for (let i = 0; i < r.length; i += 2) {
      const p = f.project(r[i], r[i + 1]);
      if (i === 0) path.moveTo(p.x * scale, p.y * scale);
      else path.lineTo(p.x * scale, p.y * scale);
    }
    path.closePath();
  }
  g.fill(path, "evenodd");
  // keep the moving water well off the shelf edge
  g.lineWidth = 13 * scale;
  g.lineJoin = "round";
  g.stroke(path);
  return c;
}

export function DeepWater({ ready }: { ready: boolean }) {
  const host = useRef<HTMLDivElement>(null);
  const [on, setOn] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [speed, setSpeed] = useState(DEEP_SPEED);
  const [phone, setPhone] = useState(false);
  const [colors, setColors] = useState<{ back: string; hi: string } | null>(null);

  // load after the chart has drawn and the browser is idle
  useEffect(() => {
    if (!ready) return;
    const mode = () => document.documentElement.getAttribute("data-chart-mode") || "day";
    const cs = () => getComputedStyle(document.documentElement);
    const read = () => setColors({ back: cs().getPropertyValue("--deep-back").trim() || "transparent", hi: cs().getPropertyValue("--deep-highlight").trim() || "white" });
    read();
    const mo = new MutationObserver(() => {
      read();
      setMounted((m) => m && mode() !== "night");
    });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-chart-mode"] });
    const mq = matchMedia("(prefers-reduced-motion: reduce)");
    const pace = () => setSpeed(mq.matches || document.hidden ? 0 : DEEP_SPEED);
    pace();
    mq.addEventListener("change", pace);
    document.addEventListener("visibilitychange", pace);
    setPhone(matchMedia("(max-width: 640px)").matches);
    const go = () => mode() !== "night" && setMounted(true);
    const idle = (window as unknown as { requestIdleCallback?: (f: () => void, o?: { timeout: number }) => number }).requestIdleCallback;
    const id = idle ? idle(go, { timeout: 2500 }) : window.setTimeout(go, 1200);
    return () => {
      mo.disconnect();
      mq.removeEventListener("change", pace);
      document.removeEventListener("visibilitychange", pace);
      if (!idle) window.clearTimeout(id);
    };
  }, [ready]);

  // the mask: redrawn when the camera settles; hidden while it moves
  useEffect(() => {
    if (!mounted) return;
    let rings: Ring[] = [];
    let url: string | null = null;
    let settle = 0;
    let alive = true;
    const update = () => {
      const f = getFrame();
      const el = host.current;
      if (!f || !el || !rings.length || flying()) return;
      deepMask(f, rings, 0.5).toBlob((b) => {
        if (!b || !alive) return;
        const next = URL.createObjectURL(b);
        el.style.maskImage = el.style.webkitMaskImage = `url(${next})`;
        if (url) URL.revokeObjectURL(url);
        url = next;
        setOn(true);
      });
    };
    shelfRings().then((r) => {
      rings = r;
      update();
    });
    const off = onFrame(() => {
      setOn(false);
      window.clearTimeout(settle);
      settle = window.setTimeout(update, 220);
    });
    return () => {
      alive = false;
      off();
      window.clearTimeout(settle);
      if (url) URL.revokeObjectURL(url);
    };
  }, [mounted]);

  if (!mounted || !colors) return null;
  return (
    <div ref={host} className="n12-deep" data-on={on ? "1" : "0"} aria-hidden="true" data-testid="nx-deep">
      <Water
        style={{ position: "absolute", inset: 0 }}
        colorBack={colors.back}
        colorHighlight={colors.hi}
        highlights={0.32}
        layering={0.55}
        edges={0}
        waves={0.18}
        caustic={0}
        size={0.3}
        speed={speed}
        fit="cover"
        minPixelRatio={phone ? 1 : 1.5}
        maxPixelCount={phone ? 900 * 1600 : 1920 * 1200}
      />
    </div>
  );
}
