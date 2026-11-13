"use client";

import { AnimatePresence, motion, useIsPresent } from "motion/react";
import { useEffect, useId, useRef } from "react";
import { layerIn, layerOut, zoneDraw } from "@/nx/ds/motion";
import { unproject } from "./mercator";
import { LAYERS_DRAWN_EVENT, useMapLayer } from "./mapLayers";
import { RAMP_BOUNDARIES, RAMP_ZONES, rampZoneAt, shoreLon, useRegulation, type ChartRegulationSpec, type NoticeExtent } from "./regulation";
import { getFrame, onFrame, type ChartFrame } from "./runtime";

/**
 * Regulation on the chart, in coral and nothing else (v14 README §4, the synthesis's Regulation layer). Pages declare
 * what to draw with <ChartRegulation> (regulation.ts); this layer draws it while the Regulation chip is on, following
 * the camera by attribute writes each frame.
 *
 * - RAMP whale-zone limits at their real latitudes: a coral line from the shore out to sea (it draws itself in from
 *   the shore), its latitude ("40°10′ N") at the shore end and the cape or line that names it ("CAPE MENDOCINO") on the
 *   land. A line is drawn only with its latitude (T38/T39): when no spot takes the label, the line is not drawn.
 * - The port's zone tinted on the water between its limits, with a ribbon inside each limit, and named on the water
 *   in a tag ("WHALE ZONE 2" over the page's caption, with the official-record ring). Neighbouring zones stay quiet:
 *   their number ("Zone 3") on the water by the shore.
 * - When no limit is in view and the whole view lies in one zone, a corner note says so ("Whole view: Whale zone 4").
 * - Official notices' latitude bands: two dashed limits from the west edge to the shore and a bar on the west edge;
 *   no offshore side (the notices give none). Always labelled "not verified".
 *
 * Lettering is haloed text on the water. It keeps clear of the bar, the chips and every object on the stage.
 * Reduced motion shows the final state (MotionPrefs).
 */
const SIDE = 10;
const BAR = 6;
const NONE: NoticeExtent[] = [];

/** "37°11′ N", "38°46.125′ N": a limit's latitude as the regulation writes it (decimal minutes kept). */
export function limitLat(lat: number): string {
  const d = Math.floor(lat + 1e-9);
  const m = Math.round((lat - d) * 60 * 1000) / 1000;
  return `${d}°${m < 10 ? "0" : ""}${m}′ N`;
}

/** The chart's short form for a limit's latitude: whole minutes ("38°46′ N"), as the prototype sets it on the water. */
export function limitLatShort(lat: number): string {
  const d = Math.floor(lat + 1e-9);
  const m = Math.round((lat - d) * 60);
  return `${d}°${m < 10 ? "0" : ""}${m}′ N`;
}

/** A notice band's label: the full one on a wide sheet, a short one on a phone; "not verified" in both (T38). */
export function extentLabel(e: NoticeExtent, wide: boolean): string {
  const short = e.count > 1 ? `${e.count} notices` : `${e.label.split(" ")[0]} notice`;
  return `${wide ? e.label : short} · not verified`;
}

type Box = { x: number; y: number; width: number; height: number };

/**
 * A regulation label's box without a layout per frame: its size is measured once per class and words (getBBox forces
 * a layout, and the layer places a dozen labels on every frame of a flight), its place read from its own attributes.
 */
const sizes = new Map<string, { w: number; h: number; asc: number }>();
export function regTextBox(el: SVGTextElement): Box {
  const x = parseFloat(el.getAttribute("x") || "0"), y = parseFloat(el.getAttribute("y") || "0");
  const key = `${el.getAttribute("class")}|${el.textContent}`;
  let m = sizes.get(key);
  if (!m) {
    const b = el.getBBox();
    m = { w: b.width, h: b.height, asc: y - b.y };
    if (b.width) sizes.set(key, m);
  }
  const anchor = el.getAttribute("text-anchor");
  const left = anchor === "end" ? x - m.w : anchor === "middle" ? x - m.w / 2 : x;
  return { x: left, y: y - m.asc, width: m.w, height: m.h };
}
/** A drawn limit line's box from its own path ("M x,y H x2"). */
export function regLineBox(el: SVGPathElement): Box | null {
  const m = /M(-?[\d.]+),(-?[\d.]+)H(-?[\d.]+)/.exec(el.getAttribute("d") || "");
  if (!m) return null;
  const a = parseFloat(m[1]), y = parseFloat(m[2]), b = parseFloat(m[3]);
  return { x: Math.min(a, b), y, width: Math.abs(b - a), height: 0 };
}
const overlap = (a: Box, b: Box, gap = 3) => a.x < b.x + b.width + gap && a.x + a.width + gap > b.x && a.y < b.y + b.height + gap && a.y + a.height + gap > b.y;

/** What regulation's lettering keeps clear of, in frame px. */
function stageBlockers(frameEl: Element | null | undefined): Box[] {
  if (!frameEl) return [];
  const fr = frameEl.getBoundingClientRect();
  return [...document.querySelectorAll<HTMLElement>(".n12-bar > *, .nx-layers, [data-nx-ticket], [data-nx-dock], [data-nx-pass], .n12-scale, .n12-controls, .n12-dock-slot, .n12-app:not(:has([data-nx-layout=stage])) .n12-margin")]
    .map((e) => e.getBoundingClientRect())
    .filter((r) => r.width && r.height)
    .map((r) => ({ x: r.left - fr.left, y: r.top - fr.top, width: r.width, height: r.height }));
}

/** With the chip on and nothing declared by the page, the layer still answers its question: every whale-zone limit. */
const ALL_ZONES: ChartRegulationSpec = { zones: { highlight: null } };

export function RegulationLayer() {
  const asked = useRegulation();
  const on = useMapLayer("regulation");
  const spec = asked && (asked.zones || asked.extents?.length || asked.advisory) ? asked : ALL_ZONES;
  const told = () => window.dispatchEvent(new Event(LAYERS_DRAWN_EVENT));
  return (
    <AnimatePresence onExitComplete={told}>
      {on ? <Drawing key={JSON.stringify(spec)} spec={spec} /> : null}
    </AnimatePresence>
  );
}

function Drawing({ spec }: { spec: ChartRegulationSpec }) {
  const present = useIsPresent();
  useEffect(() => {
    window.dispatchEvent(new Event(LAYERS_DRAWN_EVENT));
  }, [present]);
  const svgRef = useRef<SVGSVGElement>(null);
  const pinRef = useRef<HTMLDivElement>(null);
  const zones = spec.zones ?? null;
  const extents = spec.extents ?? NONE;
  const advisory = spec.advisory ?? null;
  const hatchId = useId().replace(/[^a-zA-Z0-9]/g, "");
  const hl = zones?.highlight ?? null;
  const hlZone = RAMP_ZONES.find((z) => z.zone === hl) ?? null;

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const q = (sel: string) => svg.querySelector(sel);
    const set = (el: Element | null, attrs: Record<string, string | number>) => {
      if (el) for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
    };
    // a label the host made yield to a reading (ChartHost placeNow, `data-reg-yield`) stays hidden until it re-places
    const show = (el: Element | null, isOn: boolean) => {
      if (!el) return;
      if (!isOn) el.removeAttribute("data-reg-yield");
      el.setAttribute("visibility", isOn && !el.hasAttribute("data-reg-yield") ? "visible" : "hidden");
    };
    const showLine = (el: Element | null, isOn: boolean) => el?.setAttribute("visibility", isOn ? "visible" : "hidden");

    const draw = (f: ChartFrame | null) => {
      if (!f) return;
      const { width: W, height: H } = f.viewport;
      const phone = W <= 640;
      const x0 = SIDE;
      const freeR = W - 8;
      const yTop = 4, yBot = H - f.insets.bottom - 4;
      const yOf = (lat: number) => f.project(-120, lat).y;
      const xOf = (lon: number, lat: number) => f.project(lon, lat).x;
      const shoreX = (lat: number) => {
        const s = shoreLon(lat);
        return Math.min(W + 40, Math.max(-40, s == null ? W : xOf(s, lat)));
      };
      const frameEl = svg.parentElement?.closest(".n12-frame");
      const blockers = stageBlockers(frameEl);
      const placed: Box[] = [];
      const tryText = (el: Element | null, text: string, x: number, y: number, anchor: "start" | "end" = "start", land = false): Box | null => {
        if (!el) return null;
        el.textContent = text;
        set(el, { x, y, "text-anchor": anchor });
        const b = regTextBox(el as SVGTextElement);
        if (!b.width || b.x < x0 - 1 || b.x + b.width > freeR || b.y < yTop || b.y + b.height > yBot) return null;
        if (!land && blockers.some((k) => overlap(b, k))) return null;
        if (land && blockers.some((k) => overlap(b, k, 6))) return null;
        if (placed.some((k) => overlap(b, k))) return null;
        placed.push(b);
        return b;
      };
      // the water between two latitudes, from the screen's west edge to the shore
      const band = (n: number, s: number) => {
        const steps = 32;
        let d = "";
        for (let i = 0; i <= steps; i++) {
          const la = n - ((n - s) * i) / steps;
          d += `${i ? "L" : ""}${shoreX(la).toFixed(1)},${yOf(la).toFixed(1)}`;
        }
        return `M-8,${yOf(n).toFixed(1)}L${d}L-8,${yOf(s).toFixed(1)}Z`;
      };

      if (zones) {
        const drawn = new Map<number, number>();
        RAMP_BOUNDARIES.forEach((b, i) => {
          const line = q(`[data-reg-line="${i}"]`);
          const label = q(`[data-reg-limit="${i}"]`);
          const name = q(`[data-reg-cape="${i}"]`);
          const tick = q(`[data-reg-tick="${i}"]`);
          const y = yOf(b.lat);
          const xs = shoreX(b.lat);
          const heavy = !!hlZone && (hlZone.north === b.lat || hlZone.south === b.lat);
          // the zone's limits are labelled on the water side the zone lies (below a north limit, above a south one)
          const inside = hlZone ? (hlZone.north === b.lat ? "below" : hlZone.south === b.lat ? "above" : null) : null;
          let box: Box | null = null;
          if (y > yTop + 10 && y < yBot - 10 && xs > x0 + 40) {
            const words = phone ? limitLatShort(b.lat) : limitLatShort(b.lat);
            const ys = inside === "above" ? [y + 17, y - 8] : [y - 8, y + 17];
            const xsCand = [xs - 10, Math.min(xs - 10, W * 0.46), phone ? 12 : 292];
            outer: for (const ty of ys)
              for (const cx of xsCand) {
                box = tryText(label, words, cx, ty, cx === 12 || cx === 292 ? "start" : "end");
                if (box) break outer;
              }
          }
          show(label, !!box);
          showLine(line, !!box);
          showLine(tick, !!box);
          if (box) {
            set(line, { d: `M${xs.toFixed(1)},${y.toFixed(1)}H-4`, "data-heavy": heavy ? "1" : "0" });
            set(tick, { d: `M${xs.toFixed(1)},${(y - 6).toFixed(1)}v12` });
            drawn.set(i, y);
            // the cape or line that names the limit, on the land beside it
            show(name, !phone && !!tryText(name, b.name, xs + 12, y + 4, "start", true));
          } else show(name, false);
        });
        // the port's zone: tinted water and a ribbon inside each limit
        const wash = q("[data-reg-wash]");
        const ribs = [q("[data-reg-rib='n']"), q("[data-reg-rib='s']")];
        if (hlZone && wash) {
          set(wash, { d: band(hlZone.north, hlZone.south) });
          const yn = yOf(hlZone.north), ys = yOf(hlZone.south);
          set(ribs[0], { x: -4, y: yn, width: Math.max(0, shoreX(hlZone.north) + 4), height: 9 });
          set(ribs[1], { x: -4, y: ys - 9, width: Math.max(0, shoreX(hlZone.south) + 4), height: 9 });
          showLine(ribs[0], drawn.has(RAMP_BOUNDARIES.findIndex((b) => b.lat === hlZone.north)));
          showLine(ribs[1], drawn.has(RAMP_BOUNDARIES.findIndex((b) => b.lat === hlZone.south)));
        }
        // the port's zone named on the water in its tag; its neighbours' numbers quiet by the shore
        const pin = pinRef.current;
        let pinShown = false;
        if (pin && hlZone) {
          const [, nTop] = unproject(f.camera, f.viewport, W / 2, yTop + 90);
          const [, sBot] = unproject(f.camera, f.viewport, W / 2, yBot - 60);
          const n = Math.min(hlZone.north, nTop), s = Math.max(hlZone.south, sBot);
          let ok = n > s;
          if (ok) {
            const mid = (n + s) / 2;
            const y = yOf(mid);
            const xs = shoreX(mid);
            const w = pin.offsetWidth, h = pin.offsetHeight;
            const cx = Math.max(x0 + w / 2 + 8, Math.min(xs - w / 2 - 60, xs - W * 0.36));
            const b = { x: cx - w / 2, y: y - h / 2, width: w, height: h };
            // never over a port's name or symbol (on a phone the strip is small: the tag gives way, as the prototype's does)
            const fr = frameEl?.getBoundingClientRect();
            const names = fr
              ? [...document.querySelectorAll<HTMLElement>(".n12-labels .n12-mark, .n12-labels .n12-point")]
                  .filter((e) => e.style.visibility === "visible")
                  .map((e) => e.getBoundingClientRect())
                  .map((r) => ({ x: r.left - fr.left - 6, y: r.top - fr.top - 6, width: Math.max(r.width, 12) + 12, height: Math.max(r.height, 12) + 12 }))
              : [];
            ok = b.x > 0 && b.x + b.width < xs - 8 && !blockers.some((k) => overlap(b, k, 8)) && !names.some((k) => overlap(b, k, 4)) && !placed.some((k) => overlap(b, k));
            if (ok) {
              placed.push(b);
              pin.style.transform = `translate(${Math.round(b.x)}px, ${Math.round(b.y)}px)`;
            }
          }
          pin.style.visibility = ok ? "visible" : "hidden";
          pinShown = ok;
        }
        RAMP_ZONES.forEach((z) => {
          const t = q(`[data-reg-zone="${z.zone}"]`);
          if (z.zone === hl) return show(t, false);
          const [, nTop] = unproject(f.camera, f.viewport, W / 2, yTop);
          const [, sBot] = unproject(f.camera, f.viewport, W / 2, yBot);
          const n = Math.min(z.north, nTop), s = Math.max(z.south, sBot);
          if (n - s < 0.05) return show(t, false);
          const mid = (n + s) / 2;
          show(t, !phone && !!tryText(t, `Zone ${z.zone}`, Math.max(16, shoreX(mid) - 190), yOf(mid) + 4));
        });
        const corner = q("[data-reg-corner]");
        let cornerBox: Box | null = null;
        // the port's zone named in its tag already says which zone the view lies in
        if (!drawn.size && !pinShown) {
          const top = rampZoneAt(unproject(f.camera, f.viewport, W / 2, yTop + 120)[1]);
          const bottom = rampZoneAt(unproject(f.camera, f.viewport, W / 2, yBot)[1]);
          if (top && top === bottom) cornerBox = tryText(corner, `Whole view: Whale zone ${top}`, phone ? 12 : 292, phone ? 80 : 160);
        }
        show(corner, !!cornerBox);
      }

      if (advisory) {
        const bandEl = q("[data-reg-adv-band]");
        const yn = yOf(advisory.north), ys = yOf(advisory.south);
        set(bandEl, { d: band(advisory.north, advisory.south) });
        set(q("[data-reg-adv-n]"), { x: -8, y: yn, width: Math.max(0, shoreX(advisory.north) + 8), height: 14 });
        set(q("[data-reg-adv-s]"), { x: -8, y: ys - 14, width: Math.max(0, shoreX(advisory.south) + 8), height: 14 });
        // named on the water near its south limit (or its middle when that limit is out of view)
        const xs = shoreX(Math.min(advisory.north, Math.max(advisory.south, unproject(f.camera, f.viewport, W / 2, yBot)[1])));
        const ty = Math.min(ys, yBot - 24) - 30;
        const x0a = phone ? 12 : Math.max(24, xs - 470);
        const t1 = tryText(q("[data-reg-adv-t]"), advisory.title, x0a, ty);
        const t2 = t1 ? tryText(q("[data-reg-adv-s2]"), advisory.note, x0a, ty + 17) : null;
        show(q("[data-reg-adv-t]"), !!t1);
        show(q("[data-reg-adv-s2]"), !!t2);
      }

      extents.forEach((e, j) => {
        const yn = yOf(e.north), ys = yOf(e.south);
        const label = q(`[data-reg-ext-label="${j}"]`);
        const parts = [q(`[data-reg-bracket="${j}"]`), q(`[data-reg-ext-n="${j}"]`), q(`[data-reg-ext-s="${j}"]`)];
        const lats = [q(`[data-reg-ext-nl="${j}"]`), q(`[data-reg-ext-sl="${j}"]`)];
        const visTop = Math.max(yn, yTop), visBot = Math.min(ys, yBot);
        const lx = (phone ? 12 : 292) + extents.length * BAR + 4;
        let box: Box | null = null;
        if (visBot - visTop > 4) {
          for (const w of phone ? [extentLabel(e, false)] : [extentLabel(e, true), extentLabel(e, false)]) {
            box =
              (visBot - visTop >= 26 ? tryText(label, w, lx, (visTop + visBot) / 2 + 5) : null) ??
              (yn > yTop + 18 && yn < yBot ? tryText(label, w, lx, yn - 7) : null) ??
              (ys > yTop && ys < yBot - 18 ? tryText(label, w, lx, ys + 19) : null);
            if (box) break;
          }
        }
        show(label, !!box);
        parts.forEach((p) => showLine(p, !!box));
        if (!box) {
          lats.forEach((l) => show(l, false));
          return;
        }
        const bx = 2 + j * BAR;
        const nIn = yn >= 0 && yn <= H, sIn = ys >= 0 && ys <= H;
        set(parts[0], { d: `M${bx},${Math.max(yn, 0).toFixed(1)}V${Math.min(ys, H).toFixed(1)}${nIn ? `M0,${yn.toFixed(1)}h${12 + j * BAR}` : ""}${sIn ? `M0,${ys.toFixed(1)}h${12 + j * BAR}` : ""}` });
        set(parts[1], { d: nIn ? `M${12 + j * BAR},${yn.toFixed(1)}H${shoreX(e.north).toFixed(1)}` : "" });
        set(parts[2], { d: sIn ? `M${12 + j * BAR},${ys.toFixed(1)}H${shoreX(e.south).toFixed(1)}` : "" });
        show(lats[0], nIn && !!tryText(lats[0], limitLatShort(e.north), lx, yn - 7));
        show(lats[1], sIn && !!tryText(lats[1], limitLatShort(e.south), lx, ys + 19));
      });
    };
    draw(getFrame());
    return onFrame(draw);
  }, [zones, extents, hl, hlZone, advisory]);

  return (
    <motion.div className="n12-reg-wrap" data-leaving={present ? undefined : ""} initial={{ opacity: 0 }} animate={{ opacity: 1, transition: layerIn }} exit={{ opacity: 0, transition: layerOut }}>
      <svg ref={svgRef} className="n12-reg" aria-hidden="true" data-testid="nx-regulation">
        {advisory ? (
          <g data-testid="nx-reg-advisory">
            <defs>
              <pattern id={`a${hatchId}`} width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <rect width="7" height="7" className="n12-reg-hatch-ground" />
                <rect width="2.4" height="7" className="n12-reg-hatch-stripe" />
              </pattern>
            </defs>
            <path data-reg-adv-band="" className="n12-reg-adv-band" style={{ fill: `url(#a${hatchId})` }} />
            <rect data-reg-adv-n="" className="n12-reg-adv-edge" style={{ fill: `url(#a${hatchId})` }} />
            <rect data-reg-adv-s="" className="n12-reg-adv-edge" style={{ fill: `url(#a${hatchId})` }} />
            <text data-reg-adv-t="" data-reg-hold="" className="n12-reg-adv-title" visibility="hidden" />
            <text data-reg-adv-s2="" className="n12-reg-adv-note" visibility="hidden" />
          </g>
        ) : null}
        {zones ? (
          <g data-testid="nx-reg-zones">
            {hlZone ? (
              <>
                <path data-reg-wash="" className="n12-reg-wash" />
                <rect data-reg-rib="n" className="n12-reg-ribbon" visibility="hidden" />
                <rect data-reg-rib="s" className="n12-reg-ribbon" visibility="hidden" />
              </>
            ) : null}
            {RAMP_BOUNDARIES.map((b, i) => {
              const heavy = !!hlZone && (hlZone.north === b.lat || hlZone.south === b.lat);
              return (
                <g key={`l${i}`}>
                  <motion.path data-reg-line={i} visibility="hidden" className={heavy ? "n12-reg-line n12-reg-line-hl" : hlZone ? "n12-reg-line n12-reg-line-thin" : "n12-reg-line"} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={zoneDraw} />
                  <path data-reg-tick={i} visibility="hidden" className="n12-reg-tick" />
                </g>
              );
            })}
            {RAMP_BOUNDARIES.map((b, i) => (
              <text key={`t${i}`} data-reg-limit={i} data-reg-hold="" className="n12-reg-text n12-reg-limit" visibility="hidden" data-testid="nx-reg-limit" />
            ))}
            {RAMP_BOUNDARIES.map((b, i) => (
              <text key={`c${i}`} data-reg-cape={i} className="n12-reg-cape" visibility="hidden" data-testid="nx-reg-cape" />
            ))}
            {RAMP_ZONES.map((z) => (
              <text key={`z${z.zone}`} data-reg-zone={z.zone} className="n12-reg-text n12-reg-zone" visibility="hidden" data-testid={z.zone === hl ? undefined : `nx-reg-zone-${z.zone}`} />
            ))}
            <text data-reg-corner="" className="n12-reg-text n12-reg-zone n12-reg-zone-hl" visibility="hidden" data-testid="nx-reg-corner" />
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
      {hlZone ? (
        <div ref={pinRef} className="n12-zonepin" data-testid={`nx-reg-zone-${hlZone.zone}`} style={{ visibility: "hidden" }}>
          <b>Whale zone {hlZone.zone}</b>
          {zones?.caption ? (
            <span>
              <i className="n12-zonepin-g" aria-hidden="true" />
              {zones.caption}
            </span>
          ) : null}
        </div>
      ) : null}
    </motion.div>
  );
}

function Extent({ j }: { j: number }) {
  return (
    <>
      <motion.path data-reg-ext-n={j} visibility="hidden" className="n12-reg-limitline" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={zoneDraw} />
      <motion.path data-reg-ext-s={j} visibility="hidden" className="n12-reg-limitline" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={zoneDraw} />
      <motion.path data-reg-bracket={j} visibility="hidden" className="n12-reg-bracket" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={zoneDraw} />
      <text data-reg-ext-label={j} data-reg-hold="" className="n12-reg-text n12-reg-ext" visibility="hidden" data-testid="nx-reg-extent" />
      <text data-reg-ext-nl={j} className="n12-reg-text n12-reg-limit" visibility="hidden" />
      <text data-reg-ext-sl={j} className="n12-reg-text n12-reg-limit" visibility="hidden" />
    </>
  );
}
