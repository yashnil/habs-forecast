"use client";

import { motion } from "motion/react";
import { useEffect, useRef } from "react";
import { ink } from "@/nx/ds/motion";
import { unproject } from "./mercator";
import { RAMP_BOUNDARIES, RAMP_ZONES, rampZoneAt, shoreLon, useRegulation, type NoticeExtent } from "./regulation";
import { getFrame, onFrame, type ChartFrame } from "./runtime";

/**
 * Regulation on the chart, in the one foreign hue (README §2.4, §4 `--reg`). Pages declare what to draw with
 * <ChartRegulation> (regulation.ts); this layer draws it, following the camera by attribute writes each frame.
 *
 * - RAMP zone boundaries (E T43: latitude bands drawn as lines, never polygons): a line from the shore out to the
 *   sheet's west edge, named on the line itself ("Pigeon Point 37°11′ N", or the latitude alone when the sheet is
 *   narrow). A line is drawn only with its name (T38/T39, critic N1): when no spot on the water takes the name,
 *   the line is not drawn either. Each zone is named beside one of its drawn lines; when the whole view lies in
 *   one zone and no line is in view, a corner note says so ("Whole view: Whale zone 4"), never a name floating
 *   in the water (critic N7).
 * - Official notices' latitude bands: the two limits as dashed lines from the sheet's west edge to the shore, and
 *   a bar on the latitude edge of the sheet between them; no offshore side, since the notices give none (critic
 *   N6). The band is drawn only with its label, which always says "not verified" (T38).
 *
 * Lettering is knockout text on the halo. Line and band names hold their place (the host treats them as
 * obstacles: readings and names go elsewhere); zone names and the corner note yield to readings, stations and
 * ports and are dropped first (critic N2). Lines draw on with the `ink` preset; reduced motion shows the final
 * state (MotionPrefs skips every Motion animation).
 */
const SIDE = 10; // px in from the free area's west edge, where the lettering starts
const BAR = 6; // px between notice bars
const NONE: NoticeExtent[] = [];

/** "37°11′ N", "38°46.125′ N": a limit's latitude as the regulation writes it (decimal minutes kept). */
export function limitLat(lat: number): string {
  const d = Math.floor(lat + 1e-9);
  const m = Math.round((lat - d) * 60 * 1000) / 1000;
  return `${d}°${m < 10 ? "0" : ""}${m}′ N`;
}

/** A notice band's label: the full one on a wide sheet, a short one on a phone; "not verified" in both (T38). */
export function extentLabel(e: NoticeExtent, wide: boolean): string {
  const short = e.count > 1 ? `${e.count} notices` : `${e.label.split(" ")[0]} notice`;
  return `${wide ? e.label : short} · not verified`;
}

type Box = { x: number; y: number; width: number; height: number };
const overlap = (a: Box, b: Box, gap = 3) => a.x < b.x + b.width + gap && a.x + a.width + gap > b.x && a.y < b.y + b.height + gap && a.y + a.height + gap > b.y;

export function RegulationLayer() {
  const spec = useRegulation();
  const svgRef = useRef<SVGSVGElement>(null);
  const zones = spec?.zones ?? null;
  const extents = spec?.extents ?? NONE;
  const hl = zones?.highlight ?? null;
  const key = JSON.stringify(spec);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const q = (sel: string) => svg.querySelector(sel);
    const set = (el: Element | null, attrs: Record<string, string | number>) => {
      if (el) for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
    };
    // a label the host made yield to a reading (ChartHost placeNow, `data-reg-yield`) stays hidden until it re-places
    const show = (el: Element | null, on: boolean) => {
      if (!el) return;
      if (!on) el.removeAttribute("data-reg-yield");
      el.setAttribute("visibility", on && !el.hasAttribute("data-reg-yield") ? "visible" : "hidden");
    };
    const showLine = (el: Element | null, on: boolean) => el?.setAttribute("visibility", on ? "visible" : "hidden");

    const draw = (f: ChartFrame | null) => {
      if (!f) return;
      const { width: W, height: H } = f.viewport;
      const ins = f.insets;
      const left = ins.left; // the sheet's (or the pane's) west edge
      const x0 = left + SIDE;
      const freeR = W - ins.right - 8;
      const yTop = ins.top + 4, yBot = H - ins.bottom - 4;
      const wide = freeR - x0 >= 400;
      const yOf = (lat: number) => f.project(-120, lat).y;
      const xOf = (lon: number, lat: number) => f.project(lon, lat).x;
      const shoreX = (lat: number) => {
        const s = shoreLon(lat);
        return Math.min(W, Math.max(left, s == null ? W : xOf(s, lat)));
      };

      // what the lettering must keep clear of: the scale bar and the sources line, and each label placed this frame
      const frameEl = svg.parentElement?.closest(".n12-frame");
      const fr = frameEl?.getBoundingClientRect();
      const blockers: Box[] = [];
      if (frameEl && fr)
        for (const sel of [".n12-scale", ".n12-attrib"]) {
          const r = frameEl.querySelector(sel)?.getBoundingClientRect();
          if (r && r.width) blockers.push({ x: r.left - fr.left, y: r.top - fr.top, width: r.width, height: r.height });
        }
      const placed: Box[] = [];
      /** Set `text` at (x, y); keep it when it lies on the free water clear of everything placed, else null. */
      const tryText = (el: Element | null, text: string, x: number, y: number, anchor: "start" | "end" = "start"): Box | null => {
        if (!el) return null;
        el.textContent = text;
        set(el, { x, y, "text-anchor": anchor });
        const b = (el as SVGTextElement).getBBox();
        if (!b.width || b.x < x0 - 1 || b.x + b.width > freeR || b.y < yTop || b.y + b.height > yBot) return null;
        if (blockers.some((k) => overlap(b, k)) || placed.some((k) => overlap(b, k))) return null;
        placed.push(b);
        return b;
      };

      if (zones) {
        // the lines, each with its name on it, or not at all
        const drawn = new Map<number, number>();
        RAMP_BOUNDARIES.forEach((b, i) => {
          const line = q(`[data-reg-line="${i}"]`);
          const label = q(`[data-reg-limit="${i}"]`);
          const y = yOf(b.lat);
          const xs = shoreX(b.lat);
          let box: Box | null = null;
          if (y > yTop + 10 && y < yBot - 10 && xs > x0 + 40) {
            const words = wide ? [`${b.name} ${limitLat(b.lat)}`, limitLat(b.lat)] : [limitLat(b.lat)];
            for (const w of words) {
              box = tryText(label, w, x0, y + 5) ?? tryText(label, w, xs - 8, y + 5, "end");
              if (box) break;
            }
          }
          show(label, !!box);
          showLine(line, !!box);
          if (box) {
            set(line, { d: `M${xs.toFixed(1)},${y.toFixed(1)}H${left.toFixed(1)}` });
            drawn.set(i, y);
          }
        });
        // each zone named beside one of its drawn lines: under its north limit, else over its south limit
        RAMP_ZONES.forEach((z) => {
          const t = q(`[data-reg-zone="${z.zone}"]`);
          const n = drawn.get(RAMP_BOUNDARIES.findIndex((b) => b.lat === z.north));
          const s = drawn.get(RAMP_BOUNDARIES.findIndex((b) => b.lat === z.south));
          const words = `Whale zone ${z.zone}`;
          const box = (n != null ? tryText(t, words, x0, n + 30) : null) ?? (s != null ? tryText(t, words, x0, s - 16) : null);
          show(t, !!box);
        });
        // no line in view: when the whole view lies in one zone, a note in the corner says so
        const corner = q("[data-reg-corner]");
        let cornerBox: Box | null = null;
        if (!drawn.size) {
          const top = rampZoneAt(unproject(f.camera, f.viewport, (x0 + freeR) / 2, yTop)[1]);
          const bottom = rampZoneAt(unproject(f.camera, f.viewport, (x0 + freeR) / 2, yBot)[1]);
          if (top && top === bottom) cornerBox = tryText(corner, `Whole view: Whale zone ${top}`, x0, yTop + 18);
        }
        show(corner, !!cornerBox);
      }

      extents.forEach((e, j) => {
        const yn = yOf(e.north), ys = yOf(e.south);
        const label = q(`[data-reg-ext-label="${j}"]`);
        const parts = [q(`[data-reg-bracket="${j}"]`), q(`[data-reg-ext-n="${j}"]`), q(`[data-reg-ext-s="${j}"]`)];
        const lats = [q(`[data-reg-ext-nl="${j}"]`), q(`[data-reg-ext-sl="${j}"]`)];
        const visTop = Math.max(yn, yTop), visBot = Math.min(ys, yBot);
        const lx = x0 + extents.length * BAR + 4;
        let box: Box | null = null;
        if (visBot - visTop > 4) {
          for (const w of wide ? [extentLabel(e, true), extentLabel(e, false)] : [extentLabel(e, false)]) {
            box =
              (visBot - visTop >= 26 ? tryText(label, w, lx, (visTop + visBot) / 2 + 5) : null) ??
              (yn > yTop + 18 && yn < yBot ? tryText(label, w, lx, yn - 7) : null) ??
              (ys > yTop && ys < yBot - 18 ? tryText(label, w, lx, ys + 19) : null);
            if (box) break;
          }
        }
        // a band is drawn only with its label (T38)
        show(label, !!box);
        parts.forEach((p) => showLine(p, !!box));
        if (!box) {
          lats.forEach((l) => show(l, false));
          return;
        }
        const bx = left + 2 + j * BAR;
        const nIn = yn >= 0 && yn <= H, sIn = ys >= 0 && ys <= H;
        // the bar on the sheet's latitude edge, with a tick at each limit in view
        set(parts[0], { d: `M${bx},${Math.max(yn, 0).toFixed(1)}V${Math.min(ys, H).toFixed(1)}${nIn ? `M${left},${yn.toFixed(1)}h${12 + j * BAR}` : ""}${sIn ? `M${left},${ys.toFixed(1)}h${12 + j * BAR}` : ""}` });
        set(parts[1], { d: nIn ? `M${left + 12 + j * BAR},${yn.toFixed(1)}H${shoreX(e.north).toFixed(1)}` : "" });
        set(parts[2], { d: sIn ? `M${left + 12 + j * BAR},${ys.toFixed(1)}H${shoreX(e.south).toFixed(1)}` : "" });
        // each limit's latitude on its line, where there is room (the band's label already names it)
        show(lats[0], nIn && !!tryText(lats[0], limitLat(e.north), lx, yn - 7));
        show(lats[1], sIn && !!tryText(lats[1], limitLat(e.south), lx, ys + 19));
      });
    };
    draw(getFrame());
    return onFrame(draw);
  }, [zones, extents, hl]);

  if (!spec || (!zones && !extents.length)) return null;
  return (
    <svg ref={svgRef} className="n12-reg" aria-hidden="true" data-testid="nx-regulation" key={key}>
      {zones ? (
        <g data-testid="nx-reg-zones">
          {RAMP_BOUNDARIES.map((b, i) => {
            const heavy = !!hl && RAMP_ZONES.some((z) => z.zone === hl && (z.north === b.lat || z.south === b.lat));
            return <motion.path key={`l${i}`} data-reg-line={i} visibility="hidden" className={heavy ? "n12-reg-line n12-reg-line-hl" : "n12-reg-line"} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={ink(1.4)} />;
          })}
          {RAMP_BOUNDARIES.map((b, i) => (
            <text key={`t${i}`} data-reg-limit={i} data-reg-hold="" className="n12-reg-text n12-reg-limit" visibility="hidden" data-testid="nx-reg-limit" />
          ))}
          {RAMP_ZONES.map((z) => (
            <text key={`z${z.zone}`} data-reg-zone={z.zone} className={z.zone === hl ? "n12-reg-text n12-reg-zone n12-reg-zone-hl" : "n12-reg-text n12-reg-zone"} visibility="hidden" data-testid={`nx-reg-zone-${z.zone}`} />
          ))}
          <text data-reg-corner="" className={hl ? "n12-reg-text n12-reg-zone n12-reg-zone-hl" : "n12-reg-text n12-reg-zone"} visibility="hidden" data-testid="nx-reg-corner" />
        </g>
      ) : null}
      {extents.length ? (
        <g data-testid="nx-reg-extents">
          {extents.map((e, j) => (
            <Extent key={e.id} j={j} />
          ))}
        </g>
      ) : null}
    </svg>
  );
}

function Extent({ j }: { j: number }) {
  return (
    <>
      <motion.path data-reg-ext-n={j} visibility="hidden" className="n12-reg-limitline" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={ink(1.4)} />
      <motion.path data-reg-ext-s={j} visibility="hidden" className="n12-reg-limitline" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={ink(1.4)} />
      <motion.path data-reg-bracket={j} visibility="hidden" className="n12-reg-bracket" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={ink(1.1)} />
      <text data-reg-ext-label={j} data-reg-hold="" className="n12-reg-text n12-reg-ext" visibility="hidden" data-testid="nx-reg-extent" />
      <text data-reg-ext-nl={j} className="n12-reg-text" visibility="hidden" />
      <text data-reg-ext-sl={j} className="n12-reg-text" visibility="hidden" />
    </>
  );
}
