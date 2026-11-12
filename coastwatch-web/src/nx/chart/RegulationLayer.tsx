"use client";

import { motion } from "motion/react";
import { useEffect, useRef } from "react";
import { ink } from "@/nx/ds/motion";
import { RAMP_BOUNDARIES, RAMP_ZONES, shoreLon, useRegulation, type NoticeExtent } from "./regulation";
import { getFrame, onFrame, type ChartFrame } from "./runtime";

/**
 * Regulation on the chart, in the one foreign hue (README §2.4, §4 `--reg`): RAMP zone boundaries as lines from
 * the shore out to sea (E T43: latitude bands drawn as lines, never polygons), each named where it meets the
 * neatline side of the free area, and official notices' latitude bands as brackets with their two limits. Pages
 * declare what to draw with <ChartRegulation> (regulation.ts). Lines draw on with the `ink` preset, from the shore
 * outward; reduced motion shows the final state (MotionPrefs skips every Motion animation).
 *
 * Geometry follows the camera every frame by attribute writes (no React render per frame), on both tiers: the
 * live map's projector or the static tier's. Labels are knockout text on the halo, and the host treats them as
 * obstacles, so a port name or a sounding never lands on them.
 */
const SIDE = 12; // px in from the free area's west edge, where the labels stand
const BRACKET = 16; // px between notice brackets
const NONE: NoticeExtent[] = [];

/** "37°11′ N", "38°46.125′ N": a limit's latitude as the regulation writes it (decimal minutes kept). */
export function limitLat(lat: number): string {
  const d = Math.floor(lat + 1e-9);
  const m = Math.round((lat - d) * 60 * 1000) / 1000;
  return `${d}°${m < 10 ? "0" : ""}${m}′ N`;
}

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
    const draw = (f: ChartFrame | null) => {
      if (!f) return;
      const { width: W, height: H } = f.viewport;
      const ins = f.insets;
      const x0 = ins.left + SIDE;
      const yTop = ins.top + 8, yBot = H - ins.bottom - 8;
      const yOf = (lat: number) => f.project(-120, lat).y;
      const xOf = (lon: number, lat: number) => f.project(lon, lat).x;
      const set = (el: Element | null, attrs: Record<string, string | number>) => {
        if (el) for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
      };
      // a label the host made yield to a reading (ChartHost placeNow, `data-reg-yield`) stays hidden until it re-places
      const show = (el: Element | null, on: boolean) => {
        if (!el) return;
        if (!on) el.removeAttribute("data-reg-yield");
        el.setAttribute("visibility", on && !el.hasAttribute("data-reg-yield") ? "visible" : "hidden");
      };
      // the free area's room on screen, for labels
      const inView = (y: number) => y > yTop + 14 && y < yBot - 4;

      if (zones) {
        RAMP_BOUNDARIES.forEach((b, i) => {
          const y = yOf(b.lat);
          const shore = shoreLon(b.lat);
          const xs = shore == null ? W : xOf(shore, b.lat);
          set(svg.querySelector(`[data-reg-line="${i}"]`), { d: `M${xs.toFixed(1)},${y.toFixed(1)}H${(-8).toFixed(1)}` });
          const t = svg.querySelector(`[data-reg-limit="${i}"]`);
          // the limit's name and latitude sit on the line's sea end, when there is room beside the shore
          const fits = inView(y) && xs - x0 > 220;
          set(t, { x: x0, y: y - 6 });
          show(t, fits);
        });
        RAMP_ZONES.forEach((z) => {
          const t = svg.querySelector(`[data-reg-zone="${z.zone}"]`);
          const a = Math.max(yOf(z.north), yTop), b = Math.min(yOf(z.south), yBot);
          const room = b - a;
          // the zone's name in the middle of what is in view of its band, kept clear of the limit labels
          const y = (a + b) / 2 + 6;
          set(t, { x: x0, y: z.zone === hl ? y - 9 : y });
          t?.querySelectorAll("tspan").forEach((s) => s.setAttribute("x", String(x0)));
          show(t, room > (z.zone === hl ? 84 : 44));
        });
      }

      extents.forEach((e, j) => {
        const x = x0 + j * BRACKET;
        const yn = yOf(e.north), ys = yOf(e.south);
        const sn = shoreLon(e.north), ss = shoreLon(e.south);
        set(svg.querySelector(`[data-reg-bracket="${j}"]`), { d: `M${x + 8},${yn.toFixed(1)}H${x}V${ys.toFixed(1)}H${x + 8}` });
        set(svg.querySelector(`[data-reg-ext-n="${j}"]`), { d: `M${x + 8},${yn.toFixed(1)}H${(sn == null ? W : xOf(sn, e.north)).toFixed(1)}` });
        set(svg.querySelector(`[data-reg-ext-s="${j}"]`), { d: `M${x + 8},${ys.toFixed(1)}H${(ss == null ? W : xOf(ss, e.south)).toFixed(1)}` });
        const t = svg.querySelector(`[data-reg-ext-label="${j}"]`);
        const a = Math.max(yn, yTop), b = Math.min(ys, yBot);
        const lx = x0 + extents.length * BRACKET + 4;
        set(t, { x: lx, y: (a + b) / 2 - 4 });
        t?.querySelectorAll("tspan").forEach((s) => s.setAttribute("x", String(lx)));
        show(t, b - a > 48);
        // each limit's latitude, on its line just off the bracket
        const ln = svg.querySelector(`[data-reg-ext-nl="${j}"]`), ls = svg.querySelector(`[data-reg-ext-sl="${j}"]`);
        const lx2 = x0 + extents.length * BRACKET + 4;
        set(ln, { x: lx2, y: yn - 6 });
        set(ls, { x: lx2, y: ys + 18 });
        show(ln, inView(yn) && (j === 0 || !extents.slice(0, j).some((o) => Math.abs(o.north - e.north) < 0.01)));
        show(ls, inView(ys + 18) && (j === 0 || !extents.slice(0, j).some((o) => Math.abs(o.south - e.south) < 0.01)));
      });
      // the lettering never overprints itself: by priority (the reader's zone, zones and notices, then limits)
      const kept: DOMRect[] = [];
      [...svg.querySelectorAll<SVGTextElement>("text[data-reg-prio]")]
        .sort((p, q) => Number(p.dataset.regPrio) - Number(q.dataset.regPrio))
        .forEach((t) => {
          if (t.getAttribute("visibility") === "hidden") return;
          const b = t.getBBox();
          if (kept.some((k) => b.x < k.x + k.width + 4 && b.x + b.width + 4 > k.x && b.y < k.y + k.height + 2 && b.y + b.height + 2 > k.y)) show(t, false);
          else kept.push(b);
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
            return <motion.path key={`l${i}`} data-reg-line={i} className={heavy ? "n12-reg-line n12-reg-line-hl" : "n12-reg-line"} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={ink(1.4)} />;
          })}
          {RAMP_BOUNDARIES.map((b, i) => (
            <text key={`t${i}`} data-reg-limit={i} data-reg-prio="2" className="n12-reg-text n12-reg-limit" visibility="hidden">
              {b.name} {limitLat(b.lat)}
            </text>
          ))}
          {RAMP_ZONES.map((z) => (
            <text key={`z${z.zone}`} data-reg-zone={z.zone} data-reg-prio={z.zone === hl ? "0" : "1"} className={z.zone === hl ? "n12-reg-text n12-reg-zone n12-reg-zone-hl" : "n12-reg-text n12-reg-zone"} visibility="hidden" data-testid={`nx-reg-zone-${z.zone}`}>
              <tspan>Whale zone {z.zone}</tspan>
              {/* the reader's zone also names its limits, which a close view leaves off the sheet */}
              {z.zone === hl ? (
                <tspan dy="20" className="n12-reg-zone-sub">
                  {z.northName} to {z.southName}
                </tspan>
              ) : null}
            </text>
          ))}
        </g>
      ) : null}
      {extents.length ? (
        <g data-testid="nx-reg-extents">
          {extents.map((e, j) => (
            <Extent key={e.id} e={e} j={j} />
          ))}
        </g>
      ) : null}
    </svg>
  );
}

function Extent({ e, j }: { e: NoticeExtent; j: number }) {
  return (
    <>
      <motion.path data-reg-ext-n={j} className="n12-reg-limitline" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={ink(1.4)} />
      <motion.path data-reg-ext-s={j} className="n12-reg-limitline" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={ink(1.4)} />
      <motion.path data-reg-bracket={j} className="n12-reg-bracket" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={ink(1.1)} />
      <text data-reg-ext-label={j} data-reg-prio="1" className="n12-reg-text n12-reg-ext" visibility="hidden" data-testid="nx-reg-extent">
        <tspan dy="0">{e.label}</tspan>
        <tspan dy="18">not verified</tspan>
      </text>
      <text data-reg-ext-nl={j} data-reg-prio="3" className="n12-reg-text" visibility="hidden">
        {limitLat(e.north)}
      </text>
      <text data-reg-ext-sl={j} data-reg-prio="3" className="n12-reg-text" visibility="hidden">
        {limitLat(e.south)}
      </text>
    </>
  );
}
