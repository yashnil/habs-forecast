import { BRAND } from "@/content/brand";
import { LOCKUP_WIDTH, lockupSvg } from "./mark";
import { MARK_FIELD } from "./mark.geometry";

/**
 * The sharing card (Open Graph and Twitter): the contour lockup on the instrument ground, Monterey
 * Canyon's isobaths running off the right edge, the tagline, and the research-preview status said
 * plainly. Rendered at build time by next/og (its bundled font; the wordmark is outlined).
 */
export const shareCardSize = { width: 1200, height: 630 };
export const shareCardAlt = `${BRAND.name}: ${BRAND.tagline}`;

const svgUri = (svg: string) => `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
// the field, in tile units around the canyon head (the tile is 0-48), placed right of centre
const field = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="-60 -40 180 189" width="720" height="756"><g fill="none" stroke-linecap="round">${MARK_FIELD.lines
  .map((l) => `<path d="${l.d}" stroke="${l.depth === 200 ? "#5fd0e6" : "#7fb2c4"}" stroke-opacity="${l.depth === 200 ? 0.55 : l.depth === 0 ? 0.5 : 0.22}" stroke-width="${l.depth === 200 ? 0.7 : 0.4}"/>`)
  .join("")}</g></svg>`;

export function ShareCard() {
  const h = 104;
  return (
    <div style={{ width: "100%", height: "100%", display: "flex", position: "relative", background: "radial-gradient(circle at 78% 42%, #0f2c40 0%, #08161f 46%, #061016 100%)", color: "#ffffff" }}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={svgUri(field)} width={720} height={756} alt="" style={{ position: "absolute", right: -120, top: -80 }} />
      {/* the field fades in from the left rather than starting at an edge */}
      <div style={{ position: "absolute", left: 560, top: 0, width: 260, height: 630, background: "linear-gradient(90deg, #07131b 0%, rgba(7, 19, 27, 0) 100%)" }} />
      <div style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", padding: "72px 80px", width: "100%" }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={svgUri(lockupSvg({ tone: "dark", height: h }))} width={(LOCKUP_WIDTH / 48) * h} height={h} alt="" />
        <div style={{ display: "flex", flexDirection: "column", gap: 18, maxWidth: 900 }}>
          <div style={{ fontSize: 46, color: "#ebf2f6", lineHeight: 1.15, maxWidth: 760 }}>{BRAND.tagline}</div>
          <div style={{ fontSize: 24, color: "#7fe3f5" }}>California coast · harmful algal blooms, fisheries and the water · research preview</div>
        </div>
        <div style={{ display: "flex", fontSize: 21, color: "#8ea2af" }}>Independent research project · not affiliated with NOAA, CDFW or CDPH</div>
      </div>
    </div>
  );
}
